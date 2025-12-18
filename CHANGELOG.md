# Changelog

All notable changes to SCOPE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2025-12-18

### Fixed
- Package metadata updates for PyPI release

## [0.1.0] - 2025-12-18

Initial public release accompanying the paper: [SCOPE: Prompt Evolution for Enhancing Agent Effectiveness](https://arxiv.org/abs/2512.15374)

### Added
- `SCOPEOptimizer` - Main orchestrator for prompt optimization
- `GuidelineSynthesizer` - Generates guidelines from execution traces (π_φ, π_σ)
- `StrategicMemoryStore` - Persistent cross-task strategic rules
- `GuidelineHistory` - Optional history logging for analysis
- `MemoryOptimizer` - Rule consolidation, conflict resolution, subsumption pruning (π_ω)
- Dual-stream routing: tactical (task-specific) vs strategic (cross-task) (π_γ)
- Best-of-N guideline selection with multiple candidate models
- Two synthesis modes: "efficiency" (fast) and "thoroughness" (comprehensive 7-dimension analysis)
- Model adapters for OpenAI, Anthropic, LiteLLM (100+ providers)
- `SyncModelAdapter` for synchronous model implementations
- `CallableModelAdapter` for wrapping any function
- Centralized prompts in `scope.prompts` module for easy customization
- Parameter validation for `synthesis_mode` and `auto_accept_threshold`
- `truncate_context` parameter for full context preservation
- Validation warnings for missing `agent_name` or `task`
- Comprehensive test suite (55+ tests)
- GitHub Actions CI/CD pipeline (tests + automated PyPI publishing)
- Pre-commit hooks configuration
- `.env.template` for API key management
- Comprehensive examples and documentation

