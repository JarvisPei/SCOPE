"""
Tests for model adapters.

Run with: pytest tests/test_models.py -v
"""
import pytest
from typing import List

from scope.models import (
    Message,
    ModelResponse,
    ModelProtocol,
    BaseModelAdapter,
    CallableModelAdapter,
    SyncModelAdapter,
)


class TestMessage:
    """Tests for Message class."""
    
    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_message_user_helper(self):
        """Test Message.user() helper."""
        msg = Message.user("Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_message_assistant_helper(self):
        """Test Message.assistant() helper."""
        msg = Message.assistant("Hi there")
        assert msg.role == "assistant"
        assert msg.content == "Hi there"
    
    def test_message_system_helper(self):
        """Test Message.system() helper."""
        msg = Message.system("You are helpful")
        assert msg.role == "system"
        assert msg.content == "You are helpful"


class TestModelResponse:
    """Tests for ModelResponse class."""
    
    def test_response_creation(self):
        """Test creating a response."""
        response = ModelResponse(content="Hello")
        assert response.content == "Hello"
    
    def test_response_with_raw(self):
        """Test response with raw_response."""
        response = ModelResponse(
            content="Hello",
            raw_response={"id": "123"}
        )
        assert response.content == "Hello"
        assert response.raw_response == {"id": "123"}


class TestBaseModelAdapter:
    """Tests for BaseModelAdapter."""
    
    def test_abstract_class(self):
        """Test that BaseModelAdapter is abstract."""
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            BaseModelAdapter()
    
    @pytest.mark.asyncio
    async def test_custom_adapter(self):
        """Test creating a custom adapter."""
        class CustomAdapter(BaseModelAdapter):
            async def generate(self, messages: List[Message]) -> ModelResponse:
                return ModelResponse(content="Custom response")
        
        adapter = CustomAdapter()
        response = await adapter.generate([Message.user("Test")])
        assert response.content == "Custom response"


class TestCallableModelAdapter:
    """Tests for CallableModelAdapter."""
    
    @pytest.mark.asyncio
    async def test_callable_adapter_async(self):
        """Test using an async callable as adapter."""
        async def my_model(messages):
            return ModelResponse(content="From async callable")
        
        adapter = CallableModelAdapter(my_model)
        response = await adapter.generate([Message.user("Test")])
        assert response.content == "From async callable"
    
    @pytest.mark.asyncio
    async def test_callable_adapter_sync(self):
        """Test using a sync callable as adapter."""
        def my_sync_model(messages):
            return ModelResponse(content="From sync callable")
        
        adapter = CallableModelAdapter(my_sync_model)
        response = await adapter.generate([Message.user("Test")])
        assert response.content == "From sync callable"
    
    @pytest.mark.asyncio
    async def test_callable_with_string_return(self):
        """Test callable that returns a string."""
        def my_model(messages):
            return "Simple string response"
        
        adapter = CallableModelAdapter(my_model)
        response = await adapter.generate([Message.user("Test")])
        assert response.content == "Simple string response"
    
    @pytest.mark.asyncio
    async def test_callable_receives_messages(self):
        """Test that callable receives messages correctly."""
        received_messages = []
        
        async def capture_model(messages):
            received_messages.extend(messages)
            return ModelResponse(content="OK")
        
        adapter = CallableModelAdapter(capture_model)
        test_messages = [
            Message.system("System"),
            Message.user("User"),
        ]
        await adapter.generate(test_messages)
        
        assert len(received_messages) == 2
        assert received_messages[0].role == "system"
        assert received_messages[1].role == "user"


class TestSyncModelAdapter:
    """Tests for SyncModelAdapter."""
    
    @pytest.mark.asyncio
    async def test_sync_adapter(self):
        """Test creating a sync adapter."""
        class MySyncAdapter(SyncModelAdapter):
            def generate_sync(self, messages: List[Message]) -> ModelResponse:
                return ModelResponse(content="From sync adapter")
        
        adapter = MySyncAdapter()
        response = await adapter.generate([Message.user("Test")])
        assert response.content == "From sync adapter"
    
    @pytest.mark.asyncio
    async def test_sync_adapter_receives_messages(self):
        """Test that sync adapter receives messages correctly."""
        received_messages = []
        
        class CaptureSyncAdapter(SyncModelAdapter):
            def generate_sync(self, messages: List[Message]) -> ModelResponse:
                received_messages.extend(messages)
                return ModelResponse(content="OK")
        
        adapter = CaptureSyncAdapter()
        test_messages = [
            Message.system("System"),
            Message.user("User"),
        ]
        await adapter.generate(test_messages)
        
        assert len(received_messages) == 2
        assert received_messages[0].role == "system"
        assert received_messages[1].role == "user"


class TestModelProtocol:
    """Tests for ModelProtocol compliance."""
    
    def test_protocol_compliance(self):
        """Test that adapters comply with ModelProtocol."""
        class MyAdapter(BaseModelAdapter):
            async def generate(self, messages: List[Message]) -> ModelResponse:
                return ModelResponse(content="Test")
        
        adapter = MyAdapter()
        
        # Should have generate method
        assert hasattr(adapter, 'generate')
        assert callable(adapter.generate)
