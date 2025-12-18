"""
Tests for GuidelineSynthesizer.

Run with: pytest tests/test_synthesizer.py -v
"""
import pytest
import json
from typing import List

from scope import GuidelineSynthesizer, Guideline
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
        self.last_messages = None
    
    async def generate(self, messages: List[Message]) -> ModelResponse:
        self.last_messages = messages
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return ModelResponse(content=json.dumps(response))


class TestGuidelineSynthesizer:
    """Tests for GuidelineSynthesizer."""
    
    @pytest.fixture
    def mock_model(self):
        return MockModel()
    
    @pytest.fixture
    def synthesizer(self, mock_model):
        return GuidelineSynthesizer(mock_model)
    
    def test_synthesizer_creation(self, mock_model):
        """Test creating a synthesizer."""
        synthesizer = GuidelineSynthesizer(mock_model)
        assert synthesizer.model == mock_model
        assert synthesizer.use_best_of_n is False
        assert synthesizer.candidate_models == []
    
    def test_synthesizer_with_thoroughness_mode(self, mock_model):
        """Test synthesizer with thoroughness mode."""
        synthesizer = GuidelineSynthesizer(
            mock_model,
            use_thoroughness_mode=True,
        )
        assert synthesizer.use_thoroughness_mode is True
    
    def test_synthesizer_with_efficiency_mode(self, mock_model):
        """Test synthesizer with efficiency mode."""
        synthesizer = GuidelineSynthesizer(
            mock_model,
            use_thoroughness_mode=False,
        )
        assert synthesizer.use_thoroughness_mode is False
    
    @pytest.mark.asyncio
    async def test_generate_update_from_error(self, synthesizer):
        """Test generating update from error."""
        result = await synthesizer.generate_update_from_error(
            agent_name="test_agent",
            agent_role="Test Agent",
            task="Test task",
            error_type="ValueError",
            error_message="Invalid value",
            last_step_summary="Previous step context",
            current_system_prompt="Current prompt",
        )
        
        assert result is not None
        assert isinstance(result, Guideline)
        assert result.update_text == "Test guideline"
        assert result.rationale == "Test rationale"
    
    @pytest.mark.asyncio
    async def test_generate_update_from_quality(self, synthesizer):
        """Test generating update from quality analysis."""
        result = await synthesizer.generate_update_from_quality(
            agent_name="test_agent",
            agent_role="Test Agent",
            task="Test task",
            last_step_summary="Last step summary",
            current_system_prompt="Current prompt",
        )
        
        # May return None if model says no improvement needed
        # But should not raise an error
        assert result is None or isinstance(result, Guideline)
    
    @pytest.mark.asyncio
    async def test_generate_handles_invalid_json(self):
        """Test that synthesizer handles invalid JSON gracefully."""
        # Model that returns invalid JSON
        bad_model = MockModel()
        bad_model.responses = [{"invalid": "response"}]  # Missing required fields
        
        synthesizer = GuidelineSynthesizer(bad_model)
        
        result = await synthesizer.generate_update_from_error(
            agent_name="test_agent",
            agent_role="Test Agent",
            task="Test task",
            error_type="Error",
            error_message="Test",
            last_step_summary="Context",
            current_system_prompt="Prompt",
        )
        
        # Should handle gracefully - either None or empty guideline
        # The key is that it doesn't crash
        assert result is None or (isinstance(result, Guideline) and result.update_text == "")


class TestGuidelineSynthesizerBestOfN:
    """Tests for Best-of-N functionality."""
    
    @pytest.fixture
    def models(self):
        """Create multiple mock models with different responses."""
        model1 = MockModel([{
            "update_text": "Guideline from model 1",
            "rationale": "Rationale 1",
            "confidence": "medium"
        }])
        model2 = MockModel([{
            "update_text": "Guideline from model 2",
            "rationale": "Rationale 2",
            "confidence": "high"
        }])
        return model1, model2
    
    def test_best_of_n_setup(self, models):
        """Test Best-of-N setup."""
        primary, candidate = models
        
        synthesizer = GuidelineSynthesizer(
            primary,
            candidate_models=[candidate],
            use_best_of_n=True,
        )
        
        assert synthesizer.use_best_of_n is True
        assert len(synthesizer.candidate_models) == 1


class TestGuideline:
    """Tests for Guideline dataclass."""
    
    def test_guideline_defaults(self):
        """Test Guideline default values."""
        guideline = Guideline(
            update_text="Test",
            rationale="Test rationale",
        )
        assert guideline.scope == "session"
        assert guideline.confidence == "medium"
    
    def test_guideline_to_dict(self):
        """Test Guideline serialization."""
        guideline = Guideline(
            update_text="Test guideline",
            rationale="Test rationale",
            scope="strategic",
            confidence="high",
        )
        
        d = guideline.to_dict()
        
        assert d["update_text"] == "Test guideline"
        assert d["rationale"] == "Test rationale"
        assert d["scope"] == "strategic"
        assert d["confidence"] == "high"
    
    def test_guideline_equality(self):
        """Test Guideline equality comparison."""
        g1 = Guideline(update_text="Test", rationale="R1")
        g2 = Guideline(update_text="Test", rationale="R1")
        g3 = Guideline(update_text="Different", rationale="R1")
        
        assert g1 == g2
        assert g1 != g3
