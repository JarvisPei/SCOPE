"""
Model adapters for SCOPE.

This module provides adapters for various LLM providers:
- OpenAI (and compatible APIs like Azure, vLLM)
- Anthropic Claude
- LiteLLM (100+ providers)
- Custom adapters (sync or async)

Usage:
    ```python
    from scope.models import create_openai_model, create_anthropic_model
    
    # OpenAI
    model = create_openai_model("gpt-4o-mini")
    
    # Anthropic
    model = create_anthropic_model("claude-3-5-sonnet-20241022")
    ```

Creating Custom Adapters:
    ```python
    # Async adapter
    from scope.models import BaseModelAdapter, Message, ModelResponse
    
    class MyAsyncAdapter(BaseModelAdapter):
        async def generate(self, messages):
            return ModelResponse(content="...")
    
    # Sync adapter (for non-async code)
    from scope.models import SyncModelAdapter, Message, ModelResponse
    
    class MySyncAdapter(SyncModelAdapter):
        def generate_sync(self, messages):
            return ModelResponse(content="...")
    
    # Wrap any callable (sync or async)
    from scope.models import CallableModelAdapter
    
    def my_func(messages):
        return "response"
    
    model = CallableModelAdapter(my_func)
    ```
"""

from .anthropic_adapter import AnthropicAdapter, create_anthropic_model
from .base import (
    BaseModelAdapter,
    CallableModelAdapter,
    Message,
    ModelProtocol,
    ModelResponse,
    SyncModelAdapter,
)
from .litellm_adapter import LiteLLMAdapter, create_litellm_model
from .openai_adapter import OpenAIAdapter, create_openai_model

__all__ = [
    # Base classes
    "Message",
    "ModelResponse",
    "ModelProtocol",
    "BaseModelAdapter",
    "CallableModelAdapter",
    "SyncModelAdapter",
    # OpenAI
    "OpenAIAdapter",
    "create_openai_model",
    # Anthropic
    "AnthropicAdapter",
    "create_anthropic_model",
    # LiteLLM
    "LiteLLMAdapter",
    "create_litellm_model",
]

