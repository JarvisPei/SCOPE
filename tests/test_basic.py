"""
Basic tests for SCOPE.

Run with: pytest tests/
"""
import pytest
import tempfile
import json
from typing import List

from scope import SCOPEOptimizer, Guideline, GuidelineSynthesizer
from scope.models import BaseModelAdapter, Message, ModelResponse


class MockModel(BaseModelAdapter):
    """Mock model for testing."""
    
    def __init__(self, responses: List[dict] = None):
        self.responses = responses or [
            {
                "update_text": "Test guideline",
                "rationale": "Test rationale",
                "confidence": "high"
            }
        ]
        self.call_count = 0
    
    async def generate(self, messages: List[Message]) -> ModelResponse:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return ModelResponse(content=json.dumps(response))


class TestGuideline:
    """Tests for Guideline dataclass."""
    
    def test_guideline_creation(self):
        """Test creating a Guideline."""
        guideline = Guideline(
            update_text="Always validate inputs",
            rationale="Prevents errors",
            scope="strategic",
            confidence="high"
        )
        assert guideline.update_text == "Always validate inputs"
        assert guideline.rationale == "Prevents errors"
        assert guideline.scope == "strategic"
        assert guideline.confidence == "high"
    
    def test_guideline_to_dict(self):
        """Test Guideline.to_dict()."""
        guideline = Guideline(
            update_text="Test",
            rationale="Test rationale"
        )
        d = guideline.to_dict()
        assert d["update_text"] == "Test"
        assert d["rationale"] == "Test rationale"


class TestSCOPEOptimizer:
    """Tests for SCOPEOptimizer."""
    
    @pytest.fixture
    def mock_model(self):
        return MockModel()
    
    @pytest.fixture
    def optimizer(self, mock_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield SCOPEOptimizer(
                synthesizer_model=mock_model,
                exp_path=tmpdir,
            )
    
    def test_optimizer_creation(self, mock_model):
        """Test creating an optimizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = SCOPEOptimizer(
                synthesizer_model=mock_model,
                exp_path=tmpdir,
            )
            assert optimizer.synthesizer is not None
            assert optimizer.strategic_store is not None
    
    def test_optimizer_invalid_threshold(self, mock_model):
        """Test that invalid auto_accept_threshold raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Invalid auto_accept_threshold"):
                SCOPEOptimizer(
                    synthesizer_model=mock_model,
                    exp_path=tmpdir,
                    auto_accept_threshold="invalid",
                )
    
    def test_optimizer_invalid_synthesis_mode(self, mock_model):
        """Test that invalid synthesis_mode raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Invalid synthesis_mode"):
                SCOPEOptimizer(
                    synthesizer_model=mock_model,
                    exp_path=tmpdir,
                    synthesis_mode="invalid",
                )
    
    @pytest.mark.asyncio
    async def test_on_step_complete_with_error(self, mock_model):
        """Test on_step_complete with an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = SCOPEOptimizer(
                synthesizer_model=mock_model,
                exp_path=tmpdir,
                auto_accept_threshold="low",  # Accept all
            )
            
            result = await optimizer.on_step_complete(
                agent_name="test_agent",
                agent_role="Test role",
                task="Test task",
                error=Exception("Test error"),
                current_system_prompt="You are helpful.",
                task_id="test_001",
            )
            
            # Should generate a guideline
            assert result is not None or mock_model.call_count > 0
    
    @pytest.mark.asyncio
    async def test_on_step_complete_returns_none_without_context(self, mock_model, tmpdir):
        """Test that on_step_complete returns None when no meaningful context is provided."""
        optimizer = SCOPEOptimizer(
            synthesizer_model=mock_model,
            exp_path=tmpdir,
        )
        
        # Call with no error, no model_output, no observations
        result = await optimizer.on_step_complete(
            agent_name="test_agent",
            agent_role="Test role",
            task="Test task",
            model_output=None,
            observations=None,
            error=None,
            current_system_prompt="You are helpful.",
            task_id="test_001",
        )
        
        # Should return None - not enough context to analyze
        assert result is None
        # Model should not have been called
        assert mock_model.call_count == 0
    
    @pytest.mark.asyncio
    async def test_on_step_complete_returns_none_with_empty_agent_name(self, mock_model, tmpdir):
        """Test that on_step_complete returns None when agent_name is empty."""
        optimizer = SCOPEOptimizer(
            synthesizer_model=mock_model,
            exp_path=tmpdir,
        )
        
        result = await optimizer.on_step_complete(
            agent_name="",  # Empty agent name
            agent_role="Test role",
            task="Test task",
            error=Exception("Test error"),
        )
        
        assert result is None
        assert mock_model.call_count == 0
    
    @pytest.mark.asyncio
    async def test_on_step_complete_returns_none_with_empty_task(self, mock_model, tmpdir):
        """Test that on_step_complete returns None when task is empty."""
        optimizer = SCOPEOptimizer(
            synthesizer_model=mock_model,
            exp_path=tmpdir,
        )
        
        result = await optimizer.on_step_complete(
            agent_name="test_agent",
            agent_role="Test role",
            task="",  # Empty task
            error=Exception("Test error"),
        )
        
        assert result is None
        assert mock_model.call_count == 0


class TestModelAdapters:
    """Tests for model adapters."""
    
    def test_message_creation(self):
        """Test Message helper methods."""
        user_msg = Message.user("Hello")
        assert user_msg.role == "user"
        assert user_msg.content == "Hello"
        
        assistant_msg = Message.assistant("Hi there")
        assert assistant_msg.role == "assistant"
        
        system_msg = Message.system("You are helpful")
        assert system_msg.role == "system"
    
    @pytest.mark.asyncio
    async def test_mock_model(self):
        """Test mock model generates responses."""
        model = MockModel([
            {"update_text": "Test 1", "rationale": "R1", "confidence": "high"},
            {"update_text": "Test 2", "rationale": "R2", "confidence": "low"},
        ])
        
        msg = Message.user("Test")
        
        r1 = await model.generate([msg])
        assert "Test 1" in r1.content
        
        r2 = await model.generate([msg])
        assert "Test 2" in r2.content
        
        # Should cycle back
        r3 = await model.generate([msg])
        assert "Test 1" in r3.content

