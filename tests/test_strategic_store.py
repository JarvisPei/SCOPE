"""
Tests for StrategicMemoryStore.

Run with: pytest tests/test_strategic_store.py -v
"""
import pytest
import tempfile
import os

from scope import StrategicMemoryStore


class TestStrategicMemoryStore:
    """Tests for StrategicMemoryStore."""
    
    @pytest.fixture
    def store(self):
        """Create a temporary store for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield StrategicMemoryStore(
                exp_path=tmpdir,
                max_rules_per_domain=5,
            )
    
    def test_store_creation(self):
        """Test creating a store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StrategicMemoryStore(exp_path=tmpdir)
            assert store.rules == {}
            assert os.path.exists(store.strategic_dir)
    
    @pytest.mark.asyncio
    async def test_add_strategic_rule(self):
        """Test adding a strategic rule."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StrategicMemoryStore(exp_path=tmpdir)
            
            result = await store.add_strategic_rule(
                agent_name="test_agent",
                rule_text="Always validate inputs",
                rationale="Prevents errors",
                confidence=0.9,
                domain="data_validation",
                source_task_id="task_001",
            )
            
            assert result is True
            assert "test_agent" in store.rules
            assert "data_validation" in store.rules["test_agent"]
            assert len(store.rules["test_agent"]["data_validation"]) == 1
    
    @pytest.mark.asyncio
    async def test_duplicate_detection(self):
        """Test that duplicate rules are not added."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StrategicMemoryStore(exp_path=tmpdir)
            
            # Add first rule
            await store.add_strategic_rule(
                agent_name="test_agent",
                rule_text="Always validate inputs",
                rationale="Prevents errors",
                confidence=0.9,
                domain="general",
            )
            
            # Try to add same rule
            result = await store.add_strategic_rule(
                agent_name="test_agent",
                rule_text="Always validate inputs",  # Same text
                rationale="Different rationale",
                confidence=0.95,
                domain="general",
            )
            
            # Should be rejected as duplicate
            assert result is False
            assert len(store.rules["test_agent"]["general"]) == 1
    
    @pytest.mark.asyncio
    async def test_get_strategic_rules_text(self):
        """Test getting formatted rules text for an agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StrategicMemoryStore(exp_path=tmpdir)
            
            await store.add_strategic_rule(
                agent_name="test_agent",
                rule_text="Rule 1",
                rationale="Rationale 1",
                confidence=0.9,
                domain="general",
            )
            
            rules = store.get_strategic_rules_text("test_agent")
            assert "Rule 1" in rules
    
    @pytest.mark.asyncio
    async def test_get_strategic_rules_text_empty(self):
        """Test getting rules text when no rules exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StrategicMemoryStore(exp_path=tmpdir)
            
            rules = store.get_strategic_rules_text("test_agent")
            assert rules == ""
    
    @pytest.mark.asyncio
    async def test_persistence(self):
        """Test that rules are persisted to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create store and add rule
            store1 = StrategicMemoryStore(exp_path=tmpdir)
            await store1.add_strategic_rule(
                agent_name="test_agent",
                rule_text="Persistent rule",
                rationale="Test",
                confidence=0.9,
                domain="general",
            )
            
            # Create new store from same path
            store2 = StrategicMemoryStore(exp_path=tmpdir)
            
            # Should load the rule
            assert "test_agent" in store2.rules
            assert len(store2.rules["test_agent"]["general"]) == 1
            # The key might be 'text' not 'rule_text' depending on storage format
            rule = store2.rules["test_agent"]["general"][0]
            assert "Persistent rule" in str(rule)
    
    @pytest.mark.asyncio
    async def test_max_rules_per_domain(self):
        """Test that rules are limited per domain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StrategicMemoryStore(
                exp_path=tmpdir,
                max_rules_per_domain=3,
                enable_rule_optimization=False,  # Disable to test simple truncation
            )
            
            # Add more rules than the limit
            for i in range(5):
                await store.add_strategic_rule(
                    agent_name="test_agent",
                    rule_text=f"Rule {i}",
                    rationale=f"Rationale {i}",
                    confidence=0.9 - (i * 0.01),  # Decreasing confidence
                    domain="general",
                )
            
            # Should be limited to max_rules_per_domain
            assert len(store.rules["test_agent"]["general"]) <= 3
    
    @pytest.mark.asyncio
    async def test_get_rule_count(self):
        """Test getting rule count for an agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StrategicMemoryStore(exp_path=tmpdir)
            
            # Initially 0
            assert store.get_rule_count("test_agent") == 0
            
            await store.add_strategic_rule(
                agent_name="test_agent",
                rule_text="Rule 1",
                rationale="Test",
                confidence=0.9,
                domain="general",
            )
            
            assert store.get_rule_count("test_agent") == 1
            
            await store.add_strategic_rule(
                agent_name="test_agent",
                rule_text="Rule 2",
                rationale="Test",
                confidence=0.85,
                domain="tool_usage",
            )
            
            assert store.get_rule_count("test_agent") == 2


class TestStrategicMemoryStoreMultipleAgents:
    """Tests for multiple agents."""
    
    @pytest.mark.asyncio
    async def test_separate_agents(self):
        """Test that agents have separate rule sets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StrategicMemoryStore(exp_path=tmpdir)
            
            await store.add_strategic_rule(
                agent_name="agent1",
                rule_text="Rule for agent 1",
                rationale="Test",
                confidence=0.9,
                domain="general",
            )
            
            await store.add_strategic_rule(
                agent_name="agent2",
                rule_text="Rule for agent 2",
                rationale="Test",
                confidence=0.9,
                domain="general",
            )
            
            assert "agent1" in store.rules
            assert "agent2" in store.rules
            assert store.get_rule_count("agent1") == 1
            assert store.get_rule_count("agent2") == 1
