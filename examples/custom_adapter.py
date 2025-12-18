#!/usr/bin/env python
"""
Custom Model Adapter Example for SCOPE.

This example shows how to create your own model adapter to integrate
SCOPE with any LLM provider or local model.
"""
import asyncio
from typing import List

from scope import SCOPEOptimizer
from scope.models import BaseModelAdapter, Message, ModelResponse


class MyCustomAdapter(BaseModelAdapter):
    """
    Example custom adapter for your own LLM integration.
    
    This could be used for:
    - Local models (llama.cpp, vLLM, TGI)
    - Custom API endpoints
    - Research models
    - Mock models for testing
    """
    
    def __init__(self, endpoint_url: str, api_key: str = None):
        """Initialize your custom adapter."""
        self.endpoint_url = endpoint_url
        self.api_key = api_key
    
    async def generate(self, messages: List[Message]) -> ModelResponse:
        """
        Generate a response from your model.
        
        This is where you implement your custom logic.
        """
        # Convert messages to your format
        formatted_messages = self._convert_messages(messages)
        
        # Example: Make HTTP request to your endpoint
        # In a real implementation, you'd use aiohttp or httpx
        # 
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(
        #         self.endpoint_url,
        #         json={"messages": formatted_messages},
        #         headers={"Authorization": f"Bearer {self.api_key}"}
        #     ) as response:
        #         data = await response.json()
        #         return ModelResponse(content=data["content"])
        
        # For this example, return a mock response
        prompt = formatted_messages[-1]["content"] if formatted_messages else ""
        
        # Simulate analysis and generate a mock guideline
        mock_response = {
            "update_text": "Always validate JSON before parsing to prevent errors.",
            "rationale": "JSON parsing errors can be prevented with validation.",
            "confidence": "high"
        }
        
        import json
        return ModelResponse(content=json.dumps(mock_response))


class MockModelForTesting(BaseModelAdapter):
    """
    A mock model useful for testing SCOPE integration.
    
    Returns predefined responses for deterministic testing.
    """
    
    def __init__(self, responses: List[dict] = None):
        """
        Initialize with a list of predefined responses.
        
        Args:
            responses: List of response dicts with 'update_text', 'rationale', 'confidence'
        """
        self.responses = responses or [
            {
                "update_text": "Always check input types before processing.",
                "rationale": "Type errors are common and easily preventable.",
                "confidence": "high"
            }
        ]
        self.call_count = 0
    
    async def generate(self, messages: List[Message]) -> ModelResponse:
        """Return the next predefined response."""
        import json
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return ModelResponse(content=json.dumps(response))


async def main():
    """Demonstrate custom adapter usage."""
    
    print("=== Custom Adapter Example ===\n")
    
    # Example 1: Using a custom adapter
    print("1. Using MyCustomAdapter:")
    custom_model = MyCustomAdapter(
        endpoint_url="http://localhost:8000/v1/chat",
        api_key="your-api-key"
    )
    
    optimizer = SCOPEOptimizer(
        synthesizer_model=custom_model,
        exp_path="./scope_data",
    )
    
    result = await optimizer.on_step_complete(
        agent_name="test_agent",
        agent_role="Test Agent",
        task="Test task",
        error=Exception("Test error"),
        current_system_prompt="You are helpful.",
        task_id="test_001",
    )
    
    if result:
        guideline, guideline_type = result
        print(f"   Generated guideline: {guideline[:50]}...")
    
    # Example 2: Using mock model for testing
    print("\n2. Using MockModelForTesting:")
    mock_model = MockModelForTesting([
        {"update_text": "Test guideline 1", "rationale": "Test", "confidence": "high"},
        {"update_text": "Test guideline 2", "rationale": "Test", "confidence": "medium"},
    ])
    
    test_optimizer = SCOPEOptimizer(
        synthesizer_model=mock_model,
        exp_path="./test_scope_data",
    )
    
    for i in range(3):
        result = await test_optimizer.on_step_complete(
            agent_name="test_agent",
            agent_role="Test Agent",
            task="Test task",
            error=Exception(f"Error {i + 1}"),
            current_system_prompt="You are helpful.",
            task_id="test_002",
        )
        if result:
            print(f"   Step {i + 1}: {result[0][:30]}...")
    
    print("\n=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())

