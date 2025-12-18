#!/usr/bin/env python
"""
Basic usage example for SCOPE.

This example demonstrates:
1. Creating a model adapter
2. Initializing the SCOPEOptimizer
3. Using it in an agent loop
"""
import asyncio
import tempfile

from dotenv import load_dotenv

# Import SCOPE components
from scope import SCOPEOptimizer
from scope.models import create_openai_model

# Load environment variables from .env file
load_dotenv()


async def main():
    """Basic SCOPE usage example."""
    
    # Create a temporary directory for SCOPE data
    with tempfile.TemporaryDirectory() as exp_path:
        print(f"Using temporary directory: {exp_path}")
        
        # Step 1: Create a model adapter
        # Option A: OpenAI (requires OPENAI_API_KEY env var)
        model = create_openai_model("gpt-4o-mini")
        
        # Option B: Anthropic (requires ANTHROPIC_API_KEY env var)
        # from scope.models import create_anthropic_model
        # model = create_anthropic_model("claude-3-5-sonnet-20241022")
        
        # Option C: LiteLLM for any provider
        # from scope.models import create_litellm_model
        # model = create_litellm_model("gpt-4o-mini")
        
        # Step 2: Initialize the optimizer
        optimizer = SCOPEOptimizer(
            synthesizer_model=model,
            exp_path=exp_path,
            enable_quality_analysis=True,
            quality_analysis_frequency=3,  # Analyze every 3 steps
            auto_accept_threshold="medium",  # Accept medium+ confidence
            max_rules_per_task=10,
        )
        
        print("SCOPEOptimizer initialized!")
        
        # Step 3: Load previously learned strategic rules
        # This is critical for cross-task learning - strategic rules persist across runs
        base_prompt = "You are a helpful AI assistant."
        strategic_rules = optimizer.get_strategic_rules_for_agent("demo_agent")
        current_prompt = base_prompt + strategic_rules
        
        if strategic_rules:
            print(f"Loaded strategic rules ({len(strategic_rules)} chars)")
        else:
            print("No strategic rules loaded yet (first run)")
        
        # Step 4: Simulate an agent loop
        
        for step in range(1, 6):
            print(f"\n--- Step {step} ---")
            
            # Simulate an error on step 2
            error = Exception("JSON parsing failed") if step == 2 else None
            
            # Call SCOPE after each step
            result = await optimizer.on_step_complete(
                agent_name="demo_agent",
                agent_role="A helpful AI assistant",
                task="Answer user questions accurately",
                model_output=f"Step {step}: Processed the request",
                tool_calls='[{"name": "search", "arguments": {}}]',
                observations="Search completed successfully",
                error=error,
                current_system_prompt=current_prompt,
                task_id="demo_task_001",
            )
            
            # Apply the update if one was generated
            if result:
                update_text, guideline_type = result
                print(f"✓ Guideline generated ({guideline_type}): {update_text[:50]}...")
                current_prompt += f"\n\n## Learned Guideline:\n{update_text}"
            else:
                print("  No guideline generated")
        
        # Show final statistics
        print("\n" + "=" * 50)
        print("Final Statistics:")
        stats = optimizer.get_statistics()
        if 'strategic_rules_count' in stats:
            print(f"  Strategic rules: {stats['strategic_rules_count']}")


if __name__ == "__main__":
    asyncio.run(main())

