"""
OpenAI API adapter for SCOPE.

This adapter supports both the official OpenAI client and any OpenAI-compatible API
(e.g., Azure OpenAI, vLLM, local servers with OpenAI-compatible endpoints).
"""
from typing import Any, List, Optional

from .base import BaseModelAdapter, Message, ModelResponse


class OpenAIAdapter(BaseModelAdapter):
    """
    Adapter for OpenAI API and compatible endpoints.
    
    Supports:
    - Official OpenAI API
    - Azure OpenAI
    - Any OpenAI-compatible API (vLLM, LocalAI, etc.)
    
    Example:
        ```python
        from openai import AsyncOpenAI
        from scope.models import OpenAIAdapter
        
        # Using official OpenAI
        client = AsyncOpenAI()
        model = OpenAIAdapter(client, model="gpt-4o-mini")
        
        # Using Azure OpenAI
        from openai import AsyncAzureOpenAI
        client = AsyncAzureOpenAI(...)
        model = OpenAIAdapter(client, model="gpt-4")
        
        # Using local vLLM server
        client = AsyncOpenAI(base_url="http://localhost:8000/v1")
        model = OpenAIAdapter(client, model="meta-llama/Llama-2-7b-chat-hf")
        ```
    """

    def __init__(
        self,
        client: Any,
        model: str = "gpt-4o-mini",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the OpenAI adapter.
        
        Args:
            client: OpenAI client instance (AsyncOpenAI or AsyncAzureOpenAI)
            model: Model name/ID to use
            temperature: Sampling temperature (0.0 to 2.0), None for API default
            max_tokens: Maximum tokens in response (None for model default)
            **kwargs: Additional parameters passed to chat.completions.create()
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_kwargs = kwargs

    async def generate(self, messages: List[Message]) -> ModelResponse:
        """
        Generate a response using OpenAI API.
        
        Args:
            messages: List of Message objects
            
        Returns:
            ModelResponse with generated text
        """
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages)

        # Build request parameters
        params = {
            "model": self.model,
            "messages": openai_messages,
            **self.extra_kwargs
        }

        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        # Make API call
        response = await self.client.chat.completions.create(**params)

        # Extract content
        content = response.choices[0].message.content or ""

        return ModelResponse(content=content, raw_response=response)

    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        """Convert Message objects to OpenAI format."""
        result = []
        for msg in messages:
            if isinstance(msg.content, str):
                result.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg.content, list):
                # Handle multimodal content (text + images)
                content_parts = []
                for part in msg.content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            content_parts.append({
                                "type": "text",
                                "text": part.get("text", "")
                            })
                        elif part.get("type") == "image_url":
                            content_parts.append(part)
                    elif isinstance(part, str):
                        content_parts.append({"type": "text", "text": part})
                result.append({"role": msg.role, "content": content_parts})
            else:
                result.append({"role": msg.role, "content": str(msg.content)})
        return result


def create_openai_model(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs
) -> OpenAIAdapter:
    """
    Convenience function to create an OpenAI adapter.
    
    Args:
        model: Model name (default: gpt-4o-mini)
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
        base_url: Custom base URL (uses OPENAI_API_BASE or OPENAI_BASE_URL env var if not provided)
        temperature: Sampling temperature (None for API default)
        **kwargs: Additional parameters
        
    Returns:
        Configured OpenAIAdapter
        
    Example:
        ```python
        from scope.models import create_openai_model
        
        # Simple usage (uses OPENAI_API_KEY env var)
        model = create_openai_model("gpt-4o-mini")
        
        # With explicit API key
        model = create_openai_model("gpt-4o", api_key="sk-...")
        
        # With custom endpoint (vLLM, LocalAI, etc.)
        model = create_openai_model(
            model="meta-llama/Llama-2-7b-chat-hf",
            base_url="http://localhost:8000/v1"
        )
        ```
    """
    import os

    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            "OpenAI package not installed. Install with: pip install openai"
        )

    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key

    # Check for base_url from parameter or environment
    # Support both OPENAI_API_BASE (LiteLLM style) and OPENAI_BASE_URL (OpenAI style)
    effective_base_url = base_url or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
    if effective_base_url:
        client_kwargs["base_url"] = effective_base_url

    client = AsyncOpenAI(**client_kwargs)
    return OpenAIAdapter(client, model=model, temperature=temperature, **kwargs)

