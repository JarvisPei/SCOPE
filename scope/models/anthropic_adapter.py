"""
Anthropic Claude API adapter for SCOPE.

This adapter supports Claude models via the Anthropic API.
"""
from typing import Any, List, Optional

from .base import BaseModelAdapter, Message, ModelResponse


class AnthropicAdapter(BaseModelAdapter):
    """
    Adapter for Anthropic Claude API.
    
    Example:
        ```python
        from anthropic import AsyncAnthropic
        from scope.models import AnthropicAdapter
        
        client = AsyncAnthropic()
        model = AnthropicAdapter(client, model="claude-3-5-sonnet-20241022")
        ```
    """

    def __init__(
        self,
        client: Any,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize the Anthropic adapter.
        
        Args:
            client: Anthropic client instance (AsyncAnthropic)
            model: Model name/ID to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 to 1.0), None for API default
            **kwargs: Additional parameters passed to messages.create()
        """
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_kwargs = kwargs

    async def generate(self, messages: List[Message]) -> ModelResponse:
        """
        Generate a response using Anthropic API.
        
        Args:
            messages: List of Message objects
            
        Returns:
            ModelResponse with generated text
        """
        # Separate system message from other messages
        system_content = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_content = self._extract_text(msg.content)
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": self._convert_content(msg.content)
                })

        # Build request parameters
        params = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens,
            **self.extra_kwargs
        }

        if self.temperature is not None:
            params["temperature"] = self.temperature
        if system_content:
            params["system"] = system_content

        # Make API call
        response = await self.client.messages.create(**params)

        # Extract content
        content = ""
        for block in response.content:
            if hasattr(block, 'text'):
                content += block.text

        return ModelResponse(content=content, raw_response=response)

    def _extract_text(self, content: Any) -> str:
        """Extract text from content (string or list)."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif isinstance(part, str):
                    texts.append(part)
            return "".join(texts)
        return str(content)

    def _convert_content(self, content: Any) -> Any:
        """Convert content to Anthropic format."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Handle multimodal content
            result = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        result.append({
                            "type": "text",
                            "text": part.get("text", "")
                        })
                    elif part.get("type") == "image_url":
                        # Convert OpenAI image format to Anthropic format
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            # Base64 image
                            media_type, data = url.split(";base64,", 1)
                            media_type = media_type.replace("data:", "")
                            result.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data
                                }
                            })
                elif isinstance(part, str):
                    result.append({"type": "text", "text": part})
            return result
        return str(content)


def create_anthropic_model(
    model: str = "claude-3-5-sonnet-20241022",
    api_key: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: Optional[float] = None,
    **kwargs
) -> AnthropicAdapter:
    """
    Convenience function to create an Anthropic adapter.
    
    Args:
        model: Model name (default: claude-3-5-sonnet-20241022)
        api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (None for API default)
        **kwargs: Additional parameters
        
    Returns:
        Configured AnthropicAdapter
        
    Example:
        ```python
        from scope.models import create_anthropic_model
        
        # Simple usage (uses ANTHROPIC_API_KEY env var)
        model = create_anthropic_model()
        
        # With explicit API key
        model = create_anthropic_model(api_key="sk-ant-...")
        
        # Using Claude 3 Opus
        model = create_anthropic_model("claude-3-opus-20240229")
        ```
    """
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        raise ImportError(
            "Anthropic package not installed. Install with: pip install anthropic"
        )

    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key

    client = AsyncAnthropic(**client_kwargs)
    return AnthropicAdapter(
        client,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )

