#!/usr/bin/env python
"""
Simple API Integration Test for SCOPE.

This script tests that the model adapters work correctly with real API calls.

Usage:
    # Use defaults (gpt-4o-mini via LiteLLM, reads from .env)
    python examples/test_simple.py
    
    # Specify model
    python examples/test_simple.py --model gpt-4o
    
    # Use different provider
    python examples/test_simple.py --provider openai --model gpt-4o-mini
    python examples/test_simple.py --provider anthropic --model claude-3-5-sonnet-20241022
    python examples/test_simple.py --provider litellm --model gemini/gemini-1.5-flash
    
    # Override API credentials
    python examples/test_simple.py --api-key sk-xxx --base-url https://api.example.com/v1
"""
import argparse
import asyncio
import sys
import os
import tempfile

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test SCOPE API integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_simple.py                                    # Default: gpt-4o-mini via litellm
  python test_simple.py --model gpt-4o                     # Use GPT-4o
  python test_simple.py --provider openai                  # Use OpenAI adapter directly
  python test_simple.py --provider anthropic --model claude-3-5-sonnet-20241022
  python test_simple.py --provider litellm --model gemini/gemini-1.5-flash
  python test_simple.py --api-key sk-xxx                   # Override API key
        """
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="Model name to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "anthropic", "litellm"],
        default="litellm",
        help="Provider adapter to use (default: litellm)"
    )
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="API key (overrides environment variable)"
    )
    parser.add_argument(
        "--base-url", "-u",
        default=None,
        help="Base URL for API (overrides environment variable)"
    )
    parser.add_argument(
        "--skip-optimizer",
        action="store_true",
        help="Skip the SCOPEOptimizer test (only test model adapter)"
    )
    return parser.parse_args()


def create_model(args):
    """Create model based on arguments."""
    if args.provider == "openai":
        from scope.models import create_openai_model
        return create_openai_model(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    elif args.provider == "anthropic":
        from scope.models import create_anthropic_model
        return create_anthropic_model(
            model=args.model,
            api_key=args.api_key,
        )
    else:  # litellm
        from scope.models import create_litellm_model
        return create_litellm_model(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )


async def test_model_adapter(args):
    """Test model adapter with a simple completion."""
    print("\n" + "=" * 50)
    print(f"Testing {args.provider.upper()} Adapter")
    print(f"Model: {args.model}")
    print("=" * 50)
    
    from scope.models import Message
    
    try:
        model = create_model(args)
        
        messages = [
            Message(role="user", content="Say 'Hello SCOPE!' and nothing else.")
        ]
        
        print("Sending request...")
        response = await model.generate(messages)
        
        print(f"✓ Response received: {response.content}")
        print("✓ Model adapter test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Model adapter test FAILED: {e}")
        return False


async def test_scope_optimizer(args):
    """Test full SCOPEOptimizer with a real model."""
    print("\n" + "=" * 50)
    print("Testing SCOPEOptimizer Integration")
    print("=" * 50)
    
    from scope import SCOPEOptimizer
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = create_model(args)
            optimizer = SCOPEOptimizer(
                synthesizer_model=model,
                exp_path=tmpdir,
            )
            
            print("Simulating a step with error...")
            
            result = await optimizer.on_step_complete(
                agent_name="test_agent",
                agent_role="A helpful AI assistant",
                task="Answer user questions",
                model_output="The answer is: yes",
                error=Exception("JSON parsing failed: expected object, got string"),
                current_system_prompt="You are a helpful assistant.",
                task_id="test_001",
            )
            
            if result:
                guideline, guideline_type = result
                print(f"✓ Generated {guideline_type} guideline:")
                print(f"  {guideline[:100]}..." if len(guideline) > 100 else f"  {guideline}")
                print("✓ SCOPEOptimizer integration test PASSED")
                return True
            else:
                print("  No guideline generated (confidence below threshold)")
                print("✓ SCOPEOptimizer integration test PASSED (no errors)")
                return True
                
    except Exception as e:
        print(f"✗ SCOPEOptimizer integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run API tests."""
    args = parse_args()
    
    print("\n" + "=" * 50)
    print("SCOPE Simple API Test")
    print("=" * 50)
    print(f"\nProvider: {args.provider}")
    print(f"Model: {args.model}")
    if args.api_key:
        print(f"API Key: {args.api_key[:8]}...")
    if args.base_url:
        print(f"Base URL: {args.base_url}")
    
    results = []
    
    # Test 1: Model Adapter
    results.append(("Model Adapter", await test_model_adapter(args)))
    
    # Test 2: SCOPEOptimizer (optional)
    if not args.skip_optimizer:
        results.append(("SCOPEOptimizer", await test_scope_optimizer(args)))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All tests passed! 🎉")
    else:
        print("Some tests failed. Check the output above for details.")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
