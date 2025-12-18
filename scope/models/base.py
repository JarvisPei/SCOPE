"""
Base model interface for SCOPE.

This module defines the protocol that all model adapters must implement.
Users can create custom adapters by implementing this protocol.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, runtime_checkable


@dataclass
class Message:
    """
    Standard message format for SCOPE.
    
    Attributes:
        role: The role of the message sender ("user", "assistant", "system")
        content: The message content (string or list of content blocks)
        tool_calls: Optional tool calls (for assistant messages)
    """
    role: str
    content: Any
    tool_calls: Optional[Any] = None

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role="assistant", content=content)

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role="system", content=content)


@dataclass
class ModelResponse:
    """
    Standard response format from models.
    
    Attributes:
        content: The text content of the response (raw string from the model)
        raw_response: Optional raw response from the underlying API
    
    Note:
        SCOPE's prompts ask the model to return JSON. Your adapter should just
        return the raw model output as `content` - SCOPE handles JSON parsing.
        
        Example model output that SCOPE expects:
        ```json
        {
            "update_text": "Always validate JSON before parsing",
            "rationale": "Prevents parsing errors",
            "confidence": "high"
        }
        ```
        
        Your adapter does NOT need to parse or format this - just pass through
        the raw string from the model.
    """
    content: str
    raw_response: Optional[Any] = None


@runtime_checkable
class ModelProtocol(Protocol):
    """
    Protocol defining the interface for model adapters.
    
    Any model adapter must implement this interface to work with SCOPE.
    The `generate` method must be async and accept a list of messages.
    
    Example:
        ```python
        class MyModel:
            async def generate(self, messages: List[Message]) -> ModelResponse:
                # Your implementation here
                return ModelResponse(content="...")
        ```
    """

    async def generate(self, messages: List[Message]) -> ModelResponse:
        """
        Generate a response from the model.
        
        Args:
            messages: List of Message objects representing the conversation
            
        Returns:
            ModelResponse containing the generated text
        """
        ...


class BaseModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    Provides common functionality and enforces the interface.
    Subclass this for more structured implementations.
    
    Important:
        Your adapter should return the raw model output as a string in
        ModelResponse.content. SCOPE sends prompts that ask the model
        to return JSON, and SCOPE handles the parsing internally.
        
        You do NOT need to:
        - Parse the model output
        - Format the response as JSON
        - Handle any SCOPE-specific logic
        
        Just call your model API and return the text response.
    """

    @abstractmethod
    async def generate(self, messages: List[Message]) -> ModelResponse:
        """
        Generate a response from the model.
        
        Args:
            messages: List of Message objects (system, user, assistant roles)
            
        Returns:
            ModelResponse containing the raw text from the model
        """
        pass

    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        """
        Convert Message objects to dict format.
        
        Override this in subclasses for custom conversion logic.
        """
        result = []
        for msg in messages:
            if isinstance(msg.content, str):
                result.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg.content, list):
                # Handle structured content (e.g., with images)
                text_parts = []
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                result.append({"role": msg.role, "content": "".join(text_parts)})
            else:
                result.append({"role": msg.role, "content": str(msg.content)})
        return result


class CallableModelAdapter(BaseModelAdapter):
    """
    Adapter that wraps any callable (sync or async) as a model.
    
    This is useful for custom model implementations or testing.
    Automatically handles both synchronous and asynchronous functions.
    
    Example:
        ```python
        # Async function
        async def my_async_model(messages):
            return "Response text"
        
        # Sync function  
        def my_sync_model(messages):
            return "Response text"
        
        # Both work!
        model1 = CallableModelAdapter(my_async_model)
        model2 = CallableModelAdapter(my_sync_model)
        ```
    """

    def __init__(self, fn):
        """
        Initialize with a callable (sync or async).
        
        Args:
            fn: Function that takes messages and returns a string or ModelResponse.
                Can be either sync or async.
        """
        import asyncio
        self.fn = fn
        self._is_async = asyncio.iscoroutinefunction(fn)

    async def generate(self, messages: List[Message]) -> ModelResponse:
        """Generate response using the wrapped callable."""
        import asyncio

        if self._is_async:
            result = await self.fn(messages)
        else:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.fn, messages)

        if isinstance(result, ModelResponse):
            return result
        elif isinstance(result, str):
            return ModelResponse(content=result)
        elif hasattr(result, 'content'):
            return ModelResponse(content=result.content)
        else:
            return ModelResponse(content=str(result))


class SyncModelAdapter(BaseModelAdapter):
    """
    Base class for synchronous model adapters.
    
    Use this when your model implementation is synchronous.
    Override `generate_sync` instead of `generate`.
    
    Example:
        ```python
        from scope.models import SyncModelAdapter, Message, ModelResponse
        
        class MySyncAdapter(SyncModelAdapter):
            def generate_sync(self, messages: List[Message]) -> ModelResponse:
                # Your synchronous implementation
                response = requests.post(api_url, json={"messages": messages})
                return ModelResponse(content=response.json()["text"])
        
        # Use it like any other adapter
        optimizer = SCOPEOptimizer(synthesizer_model=MySyncAdapter())
        ```
    """

    def generate_sync(self, messages: List[Message]) -> ModelResponse:
        """
        Synchronous generate method. Override this in subclasses.
        
        Args:
            messages: List of Message objects
            
        Returns:
            ModelResponse containing the generated text
        """
        raise NotImplementedError("Subclasses must implement generate_sync")

    async def generate(self, messages: List[Message]) -> ModelResponse:
        """Async wrapper that runs generate_sync in a thread pool."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_sync, messages)

