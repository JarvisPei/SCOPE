"""
SCOPE: Self-evolving Context Optimization via Prompt Evolution

A framework for automatic prompt optimization that learns from agent execution traces.
SCOPE synthesizes guidelines from execution traces, routes them via dual-stream 
(tactical/strategic), and explores diverse strategies through multiple perspectives.

Quick Start:
    ```python
    from scope import SCOPEOptimizer
    from scope.models import create_openai_model
    
    # Create model adapter
    model = create_openai_model("gpt-4o-mini")
    
    # Initialize optimizer
    optimizer = SCOPEOptimizer(
        synthesizer_model=model,
        exp_path="./scope_data",
    )
    
    # Use in your agent loop
    result = await optimizer.on_step_complete(
        agent_name="my_agent",
        agent_role="AI Assistant",
        task="...",
        model_output="...",
        current_system_prompt="...",
        task_id="task_001",
    )
    ```

Key Components:
- SCOPEOptimizer: Main orchestrator for prompt optimization
- GuidelineSynthesizer: Generates guidelines from execution traces
- StrategicMemoryStore: Manages persistent cross-task strategic rules
- GuidelineHistory: Optional history logging for analysis
- MemoryOptimizer: Optimizes accumulated rules

Customization:
    All LLM prompts are centralized in `scope.prompts` for easy customization:
    
    ```python
    from scope import prompts
    # View/modify prompts as needed
    print(prompts.ERROR_REFLECTION_PROMPT)
    ```

Logging:
    SCOPE uses Python's standard logging module. By default, logging is silent
    (NullHandler). To enable logging:

    ```python
    import logging
    logging.getLogger("scope").setLevel(logging.INFO)
    logging.getLogger("scope").addHandler(logging.StreamHandler())
    ```
"""

import logging

__version__ = "0.1.1"

# Setup default logger with NullHandler (silent by default, following library best practices)
logger = logging.getLogger("scope")
logger.addHandler(logging.NullHandler())

# Core components
from . import prompts
from .history_store import GuidelineHistory
from .memory_optimizer import MemoryOptimizer

# Model adapters (re-export for convenience)
from .models import (
    AnthropicAdapter,
    BaseModelAdapter,
    CallableModelAdapter,
    LiteLLMAdapter,
    Message,
    ModelProtocol,
    ModelResponse,
    OpenAIAdapter,
    SyncModelAdapter,
    create_anthropic_model,
    create_litellm_model,
    create_openai_model,
)
from .optimizer import SCOPEOptimizer
from .strategic_store import StrategicMemoryStore
from .synthesizer import Guideline, GuidelineSynthesizer

__all__ = [
    # Main interface
    "SCOPEOptimizer",
    # Guideline synthesis
    "GuidelineSynthesizer",
    "Guideline",
    # Memory stores
    "GuidelineHistory",
    "StrategicMemoryStore",
    # Optimization
    "MemoryOptimizer",
    # Model interface
    "Message",
    "ModelResponse",
    "ModelProtocol",
    "BaseModelAdapter",
    "SyncModelAdapter",
    "CallableModelAdapter",
    # Model adapters
    "OpenAIAdapter",
    "AnthropicAdapter",
    "LiteLLMAdapter",
    "create_openai_model",
    "create_anthropic_model",
    "create_litellm_model",
    # Prompt templates (for customization)
    "prompts",
]
