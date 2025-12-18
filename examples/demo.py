#!/usr/bin/env python
"""
Demo script for SCOPE (Self-evolving Context Optimization via Prompt Evolution).

This demonstrates the core functionality without requiring the full agent system.
"""
import tempfile
import os
from dataclasses import dataclass
from typing import Any

# Simulate a simple model for testing
@dataclass
class MockMessage:
    role: str
    content: Any
    
@dataclass  
class MockResponse:
    content: str

class MockModel:
    """A mock model that returns predefined responses with varying confidence and scope."""
    
    def __init__(self):
        self.call_count = 0
        # Predefined responses with different confidence levels, scopes, and domains
        self.responses = [
            # Call 1: High confidence STRATEGIC error update (should be promoted)
            {
                "update_text": "Always validate JSON format before returning tool outputs. Use json.dumps() to ensure proper formatting.",
                "rationale": "The error occurred because the tool returned unformatted text instead of valid JSON.",
                "scope": "strategic",
                "domain": "tool_usage",
                "confidence": 0.92
            },
            # Call 2: Low confidence TACTICAL quality update (should be rejected)
            {
                "update_text": "Consider slightly shorter outputs for this particular dataset.",
                "rationale": "Output was a bit verbose for this specific task.",
                "scope": "tactical",
                "domain": "other",
                "confidence": 0.35
            },
            # Call 3: Medium confidence STRATEGIC quality update (should be accepted but NOT promoted - below 0.85)
            {
                "update_text": "When retrieving related data, batch requests in a single tool call instead of multiple sequential calls.",
                "rationale": "Batching reduces API calls and improves efficiency.",
                "scope": "strategic",
                "domain": "tool_usage",
                "confidence": 0.75
            },
            # Call 4: High confidence TACTICAL update (should be accepted but NOT promoted - tactical scope)
            {
                "update_text": "For this API endpoint, rate limit is 10 requests per minute - add delays between calls.",
                "rationale": "Task-specific rate limit constraint.",
                "scope": "tactical",
                "domain": "tool_usage",
                "confidence": 0.88
            },
            # Call 5: High confidence STRATEGIC planning update (should be promoted)
            {
                "update_text": "Always validate input parameters before processing to prevent downstream errors.",
                "rationale": "Missing input validation can cause cascading failures.",
                "scope": "strategic",
                "domain": "data_validation",
                "confidence": 0.90
            },
        ]
    
    async def generate(self, messages):
        # Return responses in sequence, cycling if needed
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        
        import json
        response_json = json.dumps(response)
        return MockResponse(content=response_json)


def demo_synthesizer():
    """Demonstrate the GuidelineSynthesizer."""
    print("=" * 60)
    print("DEMO 1: GuidelineSynthesizer")
    print("=" * 60)
    
    from scope import GuidelineSynthesizer
    
    # Create synthesizer with mock model
    model = MockModel()
    synthesizer = GuidelineSynthesizer(model)
    
    # Simulate an error scenario
    update = synthesizer.generate_update_from_error(
        agent_name="deep_analyzer_agent",
        agent_role="A deep analyzer that performs systematic analysis",
        task="Analyze the sentiment of user reviews",
        error_type="AgentParsingError",
        error_message="Expected JSON format but got plain text: 'The sentiment is positive'",
        last_step_summary="Step 5:\nModel output: Let me analyze...\nTool calls: deep_analyzer_tool(...)\nObservations: The sentiment is positive",
        current_system_prompt="You are a helpful AI assistant. Always use tools to complete tasks.",
    )
    
    if update:
        print(f"\n✓ Generated Update:")
        print(f"  Text: {update.update_text}")
        print(f"  Rationale: {update.rationale}")
        print(f"  Confidence: {update.confidence}")
    else:
        print("\n✗ Failed to generate update")
    
    print()


def demo_guideline_history():
    """Demonstrate the GuidelineHistory."""
    print("=" * 60)
    print("DEMO 2: GuidelineHistory (Guideline Logging)")
    print("=" * 60)
    
    from scope import GuidelineHistory
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nUsing temporary directory: {tmpdir}")
        
        # Create history store
        history = GuidelineHistory(tmpdir)
        
        # Add some updates
        print("\n→ Adding updates...")
        history.add_update(
            agent_name="deep_analyzer_agent",
            update_text="Always validate JSON format before returning outputs.",
            rationale="Prevents parsing errors",
            error_type="AgentParsingError",
            task_id="task_001",
            confidence="high",
        )
        
        history.add_update(
            agent_name="deep_analyzer_agent",
            update_text="When using tools, always provide complete parameter values.",
            rationale="Incomplete parameters cause execution failures",
            error_type="AgentError",
            task_id="task_002",
            confidence="medium",
        )
        
        # Add duplicate (should be rejected)
        history.add_update(
            agent_name="deep_analyzer_agent",
            update_text="Always validate JSON format before returning outputs.",
            rationale="Duplicate test",
            error_type="AgentParsingError",
            task_id="task_003",
            confidence="high",
        )
        
        # Get active rules
        print("\n→ Active rules for deep_analyzer_agent:")
        rules = history.get_active_rules("deep_analyzer_agent")
        for i, rule in enumerate(rules, 1):
            print(f"  {i}. {rule}")
        
        # Get statistics
        print("\n→ Statistics:")
        stats = history.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Show file structure
        print("\n→ Created files:")
        for root, dirs, files in os.walk(tmpdir):
            level = root.replace(tmpdir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    
    print()


async def demo_scope_optimizer():
    """Demonstrate the SCOPEOptimizer with auto-accept mechanism."""
    print("=" * 60)
    print("DEMO 3: SCOPEOptimizer with Auto-Accept")
    print("=" * 60)
    
    from scope import SCOPEOptimizer
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nUsing temporary directory: {tmpdir}")
        
        # Create prompt updater with auto-accept threshold
        model = MockModel()
        updater = SCOPEOptimizer(
            synthesizer_model=model,
            exp_path=tmpdir,
            store_history=True,
            enable_quality_analysis=True,
            quality_analysis_frequency=3,  # Every 3 successful steps
            auto_accept_threshold="medium",  # Only accept medium+ confidence
            max_rules_per_task=5,  # Max 5 rules per task
        )
        
        print(f"\nConfiguration:")
        print(f"  - Auto-accept threshold: medium")
        print(f"  - Max rules per task: 5")
        print(f"  - Quality analysis: every 3 successful steps")
        
        # Simulate step completion with error
        print("\n→ Simulating step completion with error...")
        
        # Create a mock error
        class MockParsingError(Exception):
            pass
        
        error = MockParsingError("Expected JSON, got text")
        
        update_text = await updater.on_step_complete(
            agent_name="deep_analyzer_agent",
            agent_role="Deep analysis agent",
            task="Analyze user sentiment",
            model_output="The sentiment is positive",
            tool_calls='[{"name": "deep_analyzer_tool", "arguments": {...}}]',
            observations="Completed analysis",
            error=error,
            current_system_prompt="You are a helpful assistant.",
            task_id="task_001",
        )
        
        if update_text:
            text, guideline_type = update_text
            print(f"✓ Update generated ({guideline_type}): {text[:80]}...")
        
        # Try another step with error
        print("\n→ Another step with error...")
        
        error2 = Exception("Another error")
        update_text = await updater.on_step_complete(
            agent_name="deep_analyzer_agent",
            agent_role="Deep analysis agent",
            task="Another task",
            error=error2,
            current_system_prompt="You are a helpful assistant.",
            task_id="task_002",
        )
        
        if update_text:
            text, guideline_type = update_text
            print(f"✓ Update generated ({guideline_type}): {text[:80]}...")
        else:
            print("✗ No update generated")
        
        # Simulate successful steps for quality analysis
        print("\n→ Simulating successful steps (quality analysis)...")
        print("   (Quality analysis triggers every 3 successful steps)")
        
        for i in range(9):  # 9 successful steps
            update_text = await updater.on_step_complete(
                agent_name="deep_analyzer_agent",
                agent_role="Deep analysis agent",
                task="Analyze user sentiment",
                model_output=f"Step: Analyzed data successfully",
                tool_calls='[{"name": "deep_analyzer_tool", "arguments": {}}]',
                observations="Tool executed successfully",
                error=None,  # No error - quality analysis
                current_system_prompt="You are a helpful assistant.",
                task_id="task_003",
            )
            
            # Only print on quality analysis steps
            if (i + 1) % 3 == 0:  # Every 3rd successful step
                if update_text:
                    text, guideline_type = update_text
                    print(f"  Quality analysis {i+1}: Generated ({guideline_type})")
                else:
                    print(f"  Quality analysis {i+1}: No update generated")
        
        print("\n→ Final Statistics:")
        stats = updater.get_statistics()
        for key, value in stats.items():
            if key != "agents":
                print(f"  {key}: {value}")
        if "agents" in stats:
            for agent, agent_stats in stats["agents"].items():
                print(f"  {agent}:")
                for k, v in agent_stats.items():
                    print(f"    {k}: {v}")
    
    print()


async def demo_two_tier_learning():
    """Demonstrate two-tier learning: tactical (short-term) vs strategic (long-term)."""
    print("=" * 60)
    print("DEMO 4: Two-Tier Learning (Tactical vs Strategic)")
    print("=" * 60)
    
    try:
        from scope import SCOPEOptimizer
    except ImportError:
        from scope import SCOPEOptimizer
    import asyncio
    
    # Create temporary directory for this demo
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n→ Using temporary directory: {temp_dir}\n")
        
        # Create updater with strategic memory enabled
        model = MockModel()
        updater = SCOPEOptimizer(
            synthesizer_model=model,
            exp_path=temp_dir,
            enable_quality_analysis=True,
            quality_analysis_frequency=1,
            auto_accept_threshold="low",  # Accept most updates for demo
            strategic_confidence_threshold=0.85,  # Threshold for strategic promotion
        )
        
        agent_name = "demo_agent"
        
        print("→ Simulating 5 updates with different scopes and confidence levels:\n")
        
        for i in range(5):
            print(f"--- Update {i+1} ---")
            update_text = await updater.on_step_complete(
                agent_name=agent_name,
                agent_role="Demo agent",
                task="Demo task",
                model_output=f"Completed action {i+1}",
                tool_calls='[{"name": "demo_tool", "arguments": {}}]',
                observations="Action completed",
                error=None if i != 0 else Exception("Demo error"),
                current_system_prompt="You are a demo agent.",
                task_id="demo_task_001",
            )
            print()
        
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        # Show tactical rules (active_rules.json)
        print("\n→ Tactical Rules (task-specific, discarded after task):")
        active_rules_path = os.path.join(temp_dir, "prompt_updates", "active_rules.json")
        if os.path.exists(active_rules_path):
            with open(active_rules_path, 'r') as f:
                import json
                active_rules = json.load(f)
                if "demo_task_001" in active_rules and agent_name in active_rules["demo_task_001"]:
                    for idx, rule in enumerate(active_rules["demo_task_001"][agent_name], 1):
                        print(f"  {idx}. {rule['update_text'][:80]}...")
                else:
                    print("  (none)")
        else:
            print("  (none)")
        
        # Show strategic rules (global_rules.json)
        print("\n→ Strategic Rules (cross-task, persisted for future):")
        strategic_rules_path = os.path.join(temp_dir, "strategic_memory", "global_rules.json")
        if os.path.exists(strategic_rules_path):
            with open(strategic_rules_path, 'r') as f:
                import json
                strategic_rules = json.load(f)
                if agent_name in strategic_rules:
                    for domain, rules in strategic_rules[agent_name].items():
                        print(f"\n  Domain: {domain}")
                        for idx, rule in enumerate(rules, 1):
                            print(f"    {idx}. {rule['rule'][:80]}...")
                            print(f"       (confidence: {rule['confidence']:.2f})")
                else:
                    print("  (none)")
        else:
            print("  (none)")
        
        # Show what would be loaded at next agent initialization
        print("\n→ What next agent instance would load:")
        strategic_text = updater.get_strategic_rules_for_agent(agent_name)
        if strategic_text:
            print(strategic_text)
        else:
            print("  (no strategic rules yet)")
        
        print("\n" + "=" * 60)
        print("KEY TAKEAWAYS")
        print("=" * 60)
        print("✓ TACTICAL rules: Applied immediately, task-specific, discarded")
        print("✓ STRATEGIC rules: High-confidence (≥0.85), cross-task, persisted")
        print("✓ Strategic rules are auto-loaded at next agent initialization")
        print("✓ Only general best practices get promoted to strategic memory")
    
    print()


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("PROMPT OPTIMIZER DEMO - Two-Tier Learning")
    print("=" * 60)
    print("\nThis demonstrates the two-tier learning system:")
    print("- TACTICAL: Task-specific rules (short-term)")
    print("- STRATEGIC: General best practices (long-term)\n")
    
    await demo_two_tier_learning()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nThe two-tier system allows agents to:")
    print("✓ Learn from specific task context (tactical)")
    print("✓ Build cross-task knowledge base (strategic)")
    print("✓ Auto-load best practices at initialization\n")


if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()

