"""
LiteLLM adapter for SCOPE.

LiteLLM provides a unified interface to 100+ LLM providers including:
- OpenAI, Azure, Anthropic, Google, AWS Bedrock, Cohere, Replicate
- Hugging Face, Together AI, Ollama, and many more

This adapter allows SCOPE to work with any LiteLLM-supported model.
"""
from typing import List, Optional

from .base import BaseModelAdapter, Message, ModelResponse


class LiteLLMAdapter(BaseModelAdapter):
    """
    Adapter using LiteLLM for universal model support.
    
    LiteLLM supports 100+ providers with a unified API:
    - OpenAI: "gpt-4o", "gpt-4o-mini"
    - Anthropic: "claude-3-5-sonnet-20241022"
    - Google: "gemini/gemini-1.5-pro"
    - AWS Bedrock: "bedrock/anthropic.claude-3"
    - Azure: "azure/<deployment_name>"
    - Ollama: "ollama/llama2"
    - And many more...
    
    Example:
        ```python
        from scope.models import LiteLLMAdapter
        
        # OpenAI
        model = LiteLLMAdapter("gpt-4o-mini")
        
        # Anthropic
        model = LiteLLMAdapter("claude-3-5-sonnet-20241022")
        
        # Google Gemini
        model = LiteLLMAdapter("gemini/gemini-1.5-pro")
        
        # Local Ollama
        model = LiteLLMAdapter("ollama/llama2")
        ```
    """

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the LiteLLM adapter.
        
        Args:
            model: Model identifier (see LiteLLM docs for provider prefixes)
            temperature: Sampling temperature (None for API default)
            max_tokens: Maximum tokens in response
            api_key: API key (optional, uses env vars if not provided)
            base_url: Custom API base URL (optional)
            **kwargs: Additional parameters passed to litellm.acompletion()
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url
        self.extra_kwargs = kwargs

    async def generate(self, messages: List[Message]) -> ModelResponse:
        """
        Generate a response using LiteLLM.
        
        Args:
            messages: List of Message objects
            
        Returns:
            ModelResponse with generated text
        """
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "LiteLLM package not installed. Install with: pip install litellm"
            )

        # Convert messages to LiteLLM format (OpenAI-compatible)
        litellm_messages = self._convert_messages(messages)

        # Build request parameters
        params = {
            "model": self.model,
            "messages": litellm_messages,
            **self.extra_kwargs
        }

        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.api_key is not None:
            params["api_key"] = self.api_key
        if self.base_url is not None:
            params["api_base"] = self.base_url

        # Make API call
        response = await litellm.acompletion(**params)

        # Extract content
        content = response.choices[0].message.content or ""

        return ModelResponse(content=content, raw_response=response)


def create_litellm_model(
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> LiteLLMAdapter:
    """
    Convenience function to create a LiteLLM adapter.
    
    Args:
        model: Model identifier with provider prefix
        api_key: API key (optional, uses env vars if not provided)
        base_url: Custom API base URL (optional)
        temperature: Sampling temperature (None for API default)
        max_tokens: Maximum tokens in response
        **kwargs: Additional parameters
        
    Returns:
        Configured LiteLLMAdapter
        
    Example:
        ```python
        from scope.models import create_litellm_model
        
        # OpenAI (uses OPENAI_API_KEY env var)
        model = create_litellm_model("gpt-4o-mini")
        
        # Anthropic (uses ANTHROPIC_API_KEY env var)
        model = create_litellm_model("claude-3-5-sonnet-20241022")
        
        # Google Gemini (uses GOOGLE_API_KEY env var)
        model = create_litellm_model("gemini/gemini-1.5-pro")
        
        # Azure OpenAI with explicit credentials
        model = create_litellm_model(
            "azure/my-deployment",
            api_key="...",
            base_url="https://my-resource.openai.azure.com"
        )
        
        # Local Ollama
        model = create_litellm_model("ollama/llama2")
        
        # Switch providers easily
        model = create_litellm_model(
            "deepseek/deepseek-chat",
            api_key="your-deepseek-key"
        )
        ```
    
    Provider Prefixes (common ones):
        - OpenAI: No prefix needed ("gpt-4o", "gpt-4o-mini")
        - Anthropic: No prefix needed ("claude-3-5-sonnet-20241022")
        - Google: "gemini/" prefix
        - Azure: "azure/" prefix
        - AWS Bedrock: "bedrock/" prefix
        - Ollama: "ollama/" prefix
        - Together AI: "together_ai/" prefix
        - Hugging Face: "huggingface/" prefix
        - DeepSeek: "deepseek/" prefix
        
    See https://docs.litellm.ai/docs/providers for full list.
    """
    return LiteLLMAdapter(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )

