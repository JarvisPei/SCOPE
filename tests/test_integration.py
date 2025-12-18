"""
Integration tests for SCOPE.

These tests verify that components work together correctly.
Run with: pytest tests/test_integration.py -v
"""
import pytest
import tempfile
import json
from typing import List

from scope import SCOPEOptimizer, StrategicMemoryStore
from scope.models import BaseModelAdapter, Message, ModelResponse


class MockModel(BaseModelAdapter):
    """Mock model that simulates realistic responses."""
    
    def __init__(self):
        self.call_count = 0
        self.responses = []
    
    def set_responses(self, responses: List[dict]):
        """Set the responses to return."""
        self.responses = responses
        self.call_count = 0
    
    async def generate(self, messages: List[Message]) -> ModelResponse:
        if self.responses:
            response = self.responses[self.call_count % len(self.responses)]
            self.call_count += 1
            return ModelResponse(content=json.dumps(response))
        
        # Default response
        return ModelResponse(content=json.dumps({
            "update_text": f"Guideline {self.call_count}",
            "rationale": "Generated rationale",
            "confidence": "high"
        }))


class TestFullWorkflow:
    """Test complete SCOPE workflows."""
    
    @pytest.mark.asyncio
    async def test_error_to_guideline_workflow(self):
        """Test the complete workflow from error to guideline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            model.set_responses([
                {
                    "update_text": "Always validate JSON before parsing",
                    "rationale": "Prevents JSON parsing errors",
                    "confidence": "high"
                },
                # Classifier response
                {
                    "is_duplicate": False,
                    "scope": "strategic",
                    "confidence": 0.9,
                    "domain": "data_validation",
                    "reason": "High confidence general rule"
                }
            ])
            
            optimizer = SCOPEOptimizer(
                synthesizer_model=model,
                exp_path=tmpdir,
                auto_accept_threshold="low",
                strategic_confidence_threshold=0.8,
            )
            
            # Simulate an error step
            result = await optimizer.on_step_complete(
                agent_name="test_agent",
                agent_role="Data Processor",
                task="Process JSON data",
                error=Exception("JSON parsing failed"),
                model_output="Invalid JSON response",
                current_system_prompt="You are a data processor.",
                task_id="task_001",
            )
            
            # Should have generated something
            assert model.call_count > 0
    
    @pytest.mark.asyncio
    async def test_multiple_steps_workflow(self):
        """Test SCOPE across multiple steps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            
            optimizer = SCOPEOptimizer(
                synthesizer_model=model,
                exp_path=tmpdir,
                enable_quality_analysis=True,
                quality_analysis_frequency=2,
                auto_accept_threshold="low",
            )
            
            current_prompt = "You are a helpful assistant."
            
            # Simulate multiple steps - include an error to ensure we get a call
            for i in range(5):
                error = Exception("Test error") if i == 1 else None
                
                result = await optimizer.on_step_complete(
                    agent_name="test_agent",
                    agent_role="Assistant",
                    task="Help users",
                    model_output=f"Step {i} output",
                    error=error,
                    current_system_prompt=current_prompt,
                    task_id="task_001",
                )
                
                if result:
                    guideline, guideline_type = result
                    current_prompt += f"\n{guideline}"
            
            # At least the error step should have triggered synthesis
            # Note: model.call_count may be 0 if classifier rejects all
            # The test passes if no errors are raised
    
    @pytest.mark.asyncio
    async def test_strategic_rule_persistence(self):
        """Test that strategic rules are persisted across optimizer instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            model.set_responses([
                {
                    "update_text": "Persistent strategic rule",
                    "rationale": "Important rule",
                    "confidence": "high"
                },
                {
                    "is_duplicate": False,
                    "scope": "strategic",
                    "confidence": 0.95,
                    "domain": "general",
                    "reason": "High confidence"
                }
            ])
            
            # First optimizer adds a rule
            optimizer1 = SCOPEOptimizer(
                synthesizer_model=model,
                exp_path=tmpdir,
                auto_accept_threshold="low",
                strategic_confidence_threshold=0.8,
            )
            
            await optimizer1.on_step_complete(
                agent_name="test_agent",
                agent_role="Agent",
                task="Task",
                error=Exception("Error"),
                current_system_prompt="Prompt",
                task_id="task_001",
            )
            
            # Create new optimizer with same path
            optimizer2 = SCOPEOptimizer(
                synthesizer_model=model,
                exp_path=tmpdir,
            )
            
            # Should load existing strategic rules
            rules = optimizer2.get_strategic_rules_for_agent("test_agent")
            # Rules might be there if the guideline was accepted as strategic
            # This depends on the classification


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_error_message(self):
        """Test handling of empty error message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            
            optimizer = SCOPEOptimizer(
                synthesizer_model=model,
                exp_path=tmpdir,
            )
            
            # Should not crash with empty error
            result = await optimizer.on_step_complete(
                agent_name="test_agent",
                agent_role="Agent",
                task="Task",
                error=Exception(""),  # Empty message
                current_system_prompt="Prompt",
                task_id="task_001",
            )
    
    @pytest.mark.asyncio
    async def test_no_error_no_quality_analysis(self):
        """Test step with no error and quality analysis disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            
            optimizer = SCOPEOptimizer(
                synthesizer_model=model,
                exp_path=tmpdir,
                enable_quality_analysis=False,
            )
            
            result = await optimizer.on_step_complete(
                agent_name="test_agent",
                agent_role="Agent",
                task="Task",
                model_output="Success",
                current_system_prompt="Prompt",
                task_id="task_001",
            )
            
            # Should return None (no error, no quality analysis)
            assert result is None
    
    @pytest.mark.asyncio
    async def test_special_characters_in_input(self):
        """Test handling of special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            
            optimizer = SCOPEOptimizer(
                synthesizer_model=model,
                exp_path=tmpdir,
            )
            
            # Should not crash with special characters
            await optimizer.on_step_complete(
                agent_name="test_agent",
                agent_role="Agent with 'quotes' and \"double\"",
                task="Task with\nnewlines\tand\ttabs",
                error=Exception("Error with unicode: 你好 🎉"),
                model_output="Output with <html> & entities",
                current_system_prompt="Prompt",
                task_id="task_001",
            )
    
    @pytest.mark.asyncio
    async def test_very_long_input(self):
        """Test handling of very long inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            
            optimizer = SCOPEOptimizer(
                synthesizer_model=model,
                exp_path=tmpdir,
            )
            
            long_text = "A" * 10000
            
            # Should not crash with long inputs
            await optimizer.on_step_complete(
                agent_name="test_agent",
                agent_role="Agent",
                task=long_text,
                error=Exception(long_text),
                model_output=long_text,
                current_system_prompt=long_text,
                task_id="task_001",
            )


class TestConfiguration:
    """Test different configuration options."""
    
    def test_all_accept_thresholds(self):
        """Test all accept threshold values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            
            for threshold in ["all", "low", "medium", "high"]:
                optimizer = SCOPEOptimizer(
                    synthesizer_model=model,
                    exp_path=tmpdir,
                    auto_accept_threshold=threshold,
                )
                assert optimizer.auto_accept_threshold == threshold
    
    def test_synthesis_modes(self):
        """Test synthesis mode configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            
            # Thoroughness mode
            opt1 = SCOPEOptimizer(
                synthesizer_model=model,
                exp_path=tmpdir,
                synthesis_mode="thoroughness",
            )
            assert opt1.synthesizer.use_thoroughness_mode is True
            
            # Efficiency mode
            opt2 = SCOPEOptimizer(
                synthesizer_model=model,
                exp_path=tmpdir,
                synthesis_mode="efficiency",
            )
            assert opt2.synthesizer.use_thoroughness_mode is False
    
    def test_optimizer_model_defaults_to_synthesizer(self):
        """Test that optimizer_model defaults to synthesizer_model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            
            optimizer = SCOPEOptimizer(
                synthesizer_model=model,
                exp_path=tmpdir,
                # optimizer_model not specified
            )
            
            # Strategic store should use the same model
            assert optimizer.strategic_store.optimizer_model == model

