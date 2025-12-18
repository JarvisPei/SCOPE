"""
SCOPEOptimizer: Main orchestrator for SCOPE (Self-evolving Context Optimization via Prompt Evolution).

This module implements:
- Guideline classification (π_γ): Routes guidelines to tactical or strategic memory
- Dual-stream routing: Tactical (task-specific) vs Strategic (cross-task)
- Automatic persistence of strategic rules to disk
- Confidence-based filtering and acceptance
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .history_store import GuidelineHistory
from .prompts import CLASSIFICATION_PROMPT
from .strategic_store import StrategicMemoryStore
from .synthesizer import Guideline, GuidelineSynthesizer

logger = logging.getLogger("scope.optimizer")


@dataclass
class SimpleMessage:
    """Simple message format for model communication."""
    role: str
    content: Any
    tool_calls: Any = None


class SCOPEOptimizer:
    """
    Main orchestrator for prompt optimization with strategic/tactical classification.
    
    DESIGN: Two-tier learning system
    1. TACTICAL (task-specific, in-memory):
       - Applied immediately during current task execution
       - Cleared after task completes
       - Examples: "This dataset uses X format", "For this API, rate limit is Y"
    
    2. STRATEGIC (cross-task, persistent):
       - High-confidence general best practices
       - Saved to disk and loaded at agent initialization
       - Examples: "Always validate tool arguments", "Prefer batch operations"
    
    Workflow:
    1. Synthesizer generates guideline based on error/quality analysis
    2. Classifier checks for duplicates and classifies as STRATEGIC/TACTICAL
    3. Classifier can refine confidence score
    4. If accepted, update is applied to current task immediately
    5. If STRATEGIC and high confidence, also saved to disk
    
    Analysis Types:
    1. Error Analysis (Priority 1): Triggered when step has an error
    2. Quality Analysis (Priority 2): Triggered every N successful steps
       - Detects inefficient tool usage
       - Identifies verbose outputs
       - Suggests best practices
    """
    
    ALLOWED_DOMAINS = [
        "tool_usage",           # How to use tools correctly (arguments, sequencing, batching)
        "data_validation",      # Validating inputs, outputs, and intermediate data
        "error_handling",       # Recovering from errors, retries, and fallbacks
        "efficiency",           # Optimizing for speed, cost, and resource usage
        "analysis_methodology", # Core logic for problem-solving (verification, scientific methods)
        "safety",               # Preventing harmful outcomes (e.g., clinical safety)
        "general",              # A catch-all for high-quality, uncategorized rules
    ]

    
    def __init__(
        self,
        synthesizer_model,
        exp_path: str,
        enable_quality_analysis: bool = True,
        quality_analysis_frequency: int = 1,  # Analyze quality every N steps
        auto_accept_threshold: str = "medium",  # "low", "medium", "high", or "all"
        max_rules_per_task: int = 20,  # Maximum rules to apply in a single task
        strategic_confidence_threshold: float = 0.85,  # Minimum confidence for strategic promotion
        max_strategic_rules_per_domain: int = 10,  # Maximum strategic rules per domain per agent
        candidate_models: list = None,  # Additional models for best-of-N
        use_best_of_n: bool = False,  # Enable best-of-N reflection
        synthesis_mode: str = "thoroughness",  # "efficiency" or "thoroughness"
        optimizer_model = None,  # Separate model for rule optimization (defaults to synthesizer_model)
        enable_rule_optimization: bool = True,  # Enable/disable automatic rule optimization
        store_history: bool = False,  # Whether to store guideline history to disk
    ):
        """
        Initialize the SCOPE optimizer.
        
        Args:
            synthesizer_model: Model instance for guideline synthesis (e.g., gpt-4o-mini)
            exp_path: Experiment path for storing updates and strategic rules
            enable_quality_analysis: Whether to analyze quality of successful steps
            quality_analysis_frequency: Analyze quality every N successful steps
            auto_accept_threshold: Confidence threshold for auto-accepting updates
                - "all": Accept all updates regardless of confidence
                - "low": Accept low, medium, and high confidence
                - "medium": Accept only medium and high confidence (default)
                - "high": Accept only high confidence
            max_rules_per_task: Maximum number of rules to apply in a single task
            strategic_confidence_threshold: Minimum confidence for promoting to strategic memory (0.0-1.0)
            max_strategic_rules_per_domain: Maximum strategic rules per domain per agent
            candidate_models: List of additional model instances for best-of-N approach
            use_best_of_n: Enable multi-model candidate generation with selection
            synthesis_mode: Mode for guideline synthesis rubrics
                - "efficiency": Lightweight, faster analysis (original V1)
                - "thoroughness": Comprehensive analysis with domain learning (V2, default)
            optimizer_model: Separate model instance for rule optimization (defaults to synthesizer_model)
            enable_rule_optimization: Enable/disable automatic rule optimization when threshold reached
            store_history: Whether to store guideline generation history to disk (default: False)
        """
        # Validate parameters
        valid_thresholds = {"all", "low", "medium", "high"}
        if auto_accept_threshold.lower() not in valid_thresholds:
            raise ValueError(
                f"Invalid auto_accept_threshold: '{auto_accept_threshold}'. "
                f"Must be one of: {', '.join(sorted(valid_thresholds))}"
            )

        valid_modes = {"efficiency", "thoroughness"}
        if synthesis_mode.lower() not in valid_modes:
            raise ValueError(
                f"Invalid synthesis_mode: '{synthesis_mode}'. "
                f"Must be one of: {', '.join(sorted(valid_modes))}"
            )

        self.enable_quality_analysis = enable_quality_analysis
        self.quality_analysis_frequency = quality_analysis_frequency
        self.auto_accept_threshold = auto_accept_threshold
        self.max_rules_per_task = max_rules_per_task
        self.strategic_confidence_threshold = strategic_confidence_threshold
        self.store_history = store_history
        self.allowed_domains = self.ALLOWED_DOMAINS
        self.synthesizer_model = synthesizer_model  # Store for classifier

        # Map synthesis_mode to internal flag
        use_thoroughness = synthesis_mode.lower() == "thoroughness"

        self.synthesizer = GuidelineSynthesizer(
            synthesizer_model,
            candidate_models=candidate_models,
            use_best_of_n=use_best_of_n,
            use_thoroughness_mode=use_thoroughness,
        )

        # Initialize history store (only if store_history is enabled)
        self.history = GuidelineHistory(exp_path) if store_history else None

        # Default optimizer_model to synthesizer_model if not provided
        if optimizer_model is None:
            optimizer_model = synthesizer_model

        # Initialize strategic memory store
        self.strategic_store = StrategicMemoryStore(
            exp_path=exp_path,
            max_rules_per_domain=max_strategic_rules_per_domain,
            optimizer_model=optimizer_model,
            enable_rule_optimization=enable_rule_optimization,
        )

        self._successful_steps_count = {}  # Track successful steps for quality analysis
        self._applied_rules_count = {}  # Track number of rules applied per agent in current task
        self._applied_rules_by_task = {}  # Track applied rules per task: {task_id: {agent_name: [rules]}}

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return (
            f"SCOPEOptimizer("
            f"quality_analysis={self.enable_quality_analysis}, "
            f"threshold='{self.auto_accept_threshold}', "
            f"max_rules={self.max_rules_per_task}, "
            f"strategic_rules={self.strategic_store.get_rule_count()})"
        )
    
    def _should_accept_update(self, update: Guideline, agent_name: str, analysis_type: str) -> Tuple[bool, str]:
        """
        Decide whether to accept and apply an update based on confidence and other criteria.
        
        Args:
            update: The generated update
            agent_name: Name of the agent
            analysis_type: Type of analysis ("error" or "quality")
            
        Returns:
            Tuple of (should_accept: bool, reason: str)
        """
        # Check max rules limit
        applied_count = self._applied_rules_count.get(agent_name, 0)
        if applied_count >= self.max_rules_per_task:
            return False, f"max_rules_per_task ({self.max_rules_per_task}) reached"
        
        # Check confidence threshold (now using float from classifier)
        confidence = update.confidence  # float between 0.0 and 1.0
        threshold = self.auto_accept_threshold.lower()  # "all", "low", "medium", "high"
        
        # Map threshold strings to float values
        threshold_values = {
            "all": 0.0,      # Accept all updates
            "low": 0.3,      # Accept low, medium, high (>= 0.3)
            "medium": 0.5,   # Accept medium, high (>= 0.5)
            "high": 0.8,     # Accept only high (>= 0.8)
        }
        
        required_confidence = threshold_values.get(threshold, 0.5)  # Default to medium
        
        if confidence < required_confidence:
            return False, f"confidence {confidence:.2f} below threshold '{threshold}' ({required_confidence:.2f})"
        
        # Both error and quality updates use the same confidence threshold
        return True, f"{analysis_type} analysis passed (confidence: {confidence:.2f})"
    
    async def _classify_and_check_duplicate(
        self,
        agent_name: str,
        update_text: str,
        rationale: str,
        initial_confidence: float,
        applied_rules: list,
    ) -> Dict[str, Any]:
        """
        Use a mini-agent (synthesizer_model) to classify update and check for duplicates.
        
        This checks against BOTH:
        - STRATEGIC rules (loaded from disk)
        - TACTICAL rules (current task, in applied_rules)
        
        Args:
            agent_name: Name of the agent
            update_text: The proposed update text
            rationale: Rationale for the update
            initial_confidence: Initial confidence from synthesizer (0.0-1.0)
            applied_rules: List of applied rules for current task
            
        Returns:
            Dict with keys:
                - is_duplicate (bool): Whether this is redundant/duplicate
                - scope (str): "strategic" or "tactical"
                - confidence (float): Refined confidence score (0.0-1.0)
                - domain (str): Domain category for strategic rules
                - reason (str): Explanation of classification
        """
        # Load STRATEGIC rules from disk
        strategic_rules_text = self.strategic_store.get_strategic_rules_text(agent_name)
        
        # Combine STRATEGIC and TACTICAL rules for comprehensive duplicate checking
        all_rules_context = f"""
=== STRATEGIC RULES (Cross-task, persistent): ===
{strategic_rules_text if strategic_rules_text else "No strategic rules yet."}

=== TACTICAL RULES (Current task only): ===
"""
        if applied_rules:
            for i, rule_info in enumerate(applied_rules, 1):
                rule_text = rule_info.get('rule', rule_info) if isinstance(rule_info, dict) else rule_info
                all_rules_context += f"{i}. {rule_text}\n"
        else:
            all_rules_context += "No tactical rules yet.\n"
        
        # Construct classification prompt using template from prompts.py
        classification_prompt = CLASSIFICATION_PROMPT.format(
            allowed_domains=", ".join(self.allowed_domains),
            update_text=update_text,
            rationale=rationale,
            initial_confidence=initial_confidence,
            all_rules_context=all_rules_context,
        )

        try:
            # Use synthesizer_model to perform classification
            messages = [
                SimpleMessage(
                    role="user",
                    content=[{"type": "text", "text": classification_prompt}]
                )
            ]
            
            response = await self.synthesizer_model.generate(messages)
            response_text = response.content.strip()
            
            # Extract JSON from response
            import json
            # Try to find JSON in response (may be wrapped in markdown code blocks)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                # Assume entire response is JSON
                json_text = response_text
            
            classification = json.loads(json_text)
            
            # Validate and set defaults
            classification.setdefault("is_duplicate", False)
            classification.setdefault("scope", "tactical")
            classification.setdefault("confidence", initial_confidence)
            classification.setdefault("domain", "general")
            classification.setdefault("reason", "Classification completed")
            
            # Post-processing to enforce allowed domains
            domain = classification.get("domain", "general")
            if classification["scope"] == "strategic" and domain not in self.allowed_domains:
                logger.warning(f"[Classifier] Domain '{domain}' not in allowed list. Mapping to 'general'.")
                classification["domain"] = "general"
            
            # Ensure confidence is a float
            classification["confidence"] = float(classification["confidence"])
            
            logger.info(f"[Classifier] Result for {agent_name}: duplicate={classification['is_duplicate']}, scope={classification['scope']}, confidence={classification['confidence']:.2f}, reason={classification['reason'][:100]}")
            
            return classification
            
        except Exception as e:
            # Fallback: if classification fails, default to tactical with original confidence
            logger.warning(f"[Classifier] Failed to classify update: {e}. Using defaults.")
            return {
                "is_duplicate": False,
                "scope": "tactical",
                "confidence": initial_confidence,
                "domain": "general",
                "reason": f"Classification failed: {str(e)}"
            }
    
    async def on_step_complete(
        self,
        agent_name: str,
        agent_role: str,
        task: str,
        model_output: Optional[str] = None,
        tool_calls: Optional[str] = None,
        observations: Optional[str] = None,
        error: Optional[Exception] = None,
        current_system_prompt: str = "",
        task_id: Optional[str] = None,
        truncate_context: bool = True,
    ) -> Optional[str]:
        """
        Analyze a completed step and potentially generate a prompt update.
        
        This is called after every step (not just errors) to allow the system
        to learn from both successful and failed executions.
        
        Args:
            agent_name: Name of the agent
            agent_role: Role/description of the agent
            task: Current task
            model_output: Model's output
            tool_calls: Tool calls attempted
            observations: Observations received
            error: Optional error that occurred
            current_system_prompt: Current system prompt
            task_id: Task ID
            truncate_context: Whether to truncate model_output/tool_calls/observations
                in step summary. Default True for efficiency; set False to preserve
                full context for more detailed analysis.
            
        Returns:
            Tuple of (guideline_text, guideline_type) if successful, None otherwise.
                guideline_type is "tactical" or "strategic"
        """
        # Validate required parameters
        if not agent_name or not agent_name.strip():
            logger.warning(
                "[Guideliner] Skipping step analysis: 'agent_name' is empty or missing. "
                "Please provide a valid agent name."
            )
            return None

        if not task or not task.strip():
            logger.warning(
                f"[Guideliner] Skipping step analysis for {agent_name}: 'task' is empty or missing. "
                "Please provide a task description."
            )
            return None

        # Early validation: check if we have enough context to analyze
        has_error = error is not None
        has_context = bool(model_output) or bool(observations)

        if not has_error and not has_context:
            # No error and no model output/observations - nothing meaningful to analyze
            logger.warning(
                f"[Guideliner] Skipping step analysis for {agent_name}: "
                "insufficient context provided. Please ensure at least one of "
                "'error', 'model_output', or 'observations' is passed to on_step_complete()."
            )
            return None
        
        # Log that we're analyzing a step
        logger.info(f"[Guideliner] Analyzing step for {agent_name} (error={has_error})")
        
        # Build step summary
        last_step_summary = self._build_step_summary(
            model_output, tool_calls, observations, truncate=truncate_context
        )
        
        update = None
        analysis_type = None
        
        # Get currently applied rules for this task and agent
        applied_rules = self._get_applied_rules(task_id, agent_name)
        
        # Priority 1: Handle errors
        if error is not None:
            error_type = type(error).__name__
            error_message = str(error)
            
            update = await self.synthesizer.generate_update_from_error(
                agent_name=agent_name,
                agent_role=agent_role,
                task=task,
                error_type=error_type,
                error_message=error_message,
                last_step_summary=last_step_summary,
                current_system_prompt=current_system_prompt,
                applied_rules=applied_rules,
            )
            analysis_type = "error"
        
        # Priority 2: Analyze quality of successful steps (if enabled)
        elif self.enable_quality_analysis:
            # Track successful steps
            if agent_name not in self._successful_steps_count:
                self._successful_steps_count[agent_name] = 0
            self._successful_steps_count[agent_name] += 1
            
            # Only analyze every N successful steps to reduce overhead
            if self._successful_steps_count[agent_name] % self.quality_analysis_frequency == 0:
                logger.info(f"[Guideliner] Triggering quality analysis for {agent_name} (successful steps: {self._successful_steps_count[agent_name]})")
                update = await self.synthesizer.generate_update_from_quality(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    task=task,
                    last_step_summary=last_step_summary,
                    current_system_prompt=current_system_prompt,
                    applied_rules=applied_rules,
                )
                analysis_type = "quality"
            else:
                logger.info(f"[Guideliner] Skipping quality analysis for {agent_name} (successful steps: {self._successful_steps_count[agent_name]}, frequency: {self.quality_analysis_frequency})")
        
        # Apply update if generated
        if update and update.update_text:
            logger.info(f"[Guideliner] Generated {analysis_type} update for {agent_name}: {update.update_text[:100]}... (confidence: {update.confidence})")
            
            # STEP 1: Classify the update and check for duplicates
            # Convert string confidence to float if needed
            if isinstance(update.confidence, str):
                # Map string confidence to float
                confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
                initial_confidence = confidence_map.get(update.confidence.lower(), 0.5)
            else:
                # Already a float
                initial_confidence = float(update.confidence)
                
            applied_rules = self._get_applied_rules(task_id, agent_name)
            
            classification = await self._classify_and_check_duplicate(
                agent_name=agent_name,
                update_text=update.update_text,
                rationale=update.rationale,
                initial_confidence=initial_confidence,
                applied_rules=applied_rules,
            )
            
            # STEP 2: Reject if duplicate
            if classification["is_duplicate"]:
                message = f"[Guideliner] ✗ REJECTED {analysis_type} update for {agent_name} (duplicate): {update.update_text[:100]}..."
                logger.info(message)
                
                # Store in history as rejected duplicate (if history tracking enabled)
                if self.store_history and self.history:
                    self.history.add_update(
                        agent_name=agent_name,
                        update_text=update.update_text,
                        rationale=update.rationale,
                        error_type=f"{analysis_type}_rejected_duplicate",
                        task_id=task_id,
                        scope=classification["scope"],
                        confidence=classification["confidence"],
                    )
                return None
            
            # STEP 3: Update confidence and scope from classifier
            update.confidence = classification["confidence"]
            update.scope = classification["scope"]
            if not hasattr(update, 'domain'):
                update.domain = classification.get("domain", "general")
            
            # STEP 4: Decide whether to accept based on refined confidence
            should_accept, reason = self._should_accept_update(update, agent_name, analysis_type)
            
            # Store in history (if history tracking enabled)
            if self.store_history and self.history:
                self.history.add_update(
                    agent_name=agent_name,
                    update_text=update.update_text,
                    rationale=update.rationale,
                    error_type=f"{analysis_type}{'_accepted' if should_accept else '_rejected'}",
                    task_id=task_id,
                    scope=update.scope,
                    confidence=update.confidence,
                )
            
            if should_accept:
                # Track rules applied
                self._applied_rules_count[agent_name] = self._applied_rules_count.get(agent_name, 0) + 1
                
                # Track applied rule for this task and agent
                self._add_applied_rule(task_id, agent_name, update.update_text, update.rationale)
                
                # STEP 5: If STRATEGIC and high confidence, save to disk
                if update.scope == "strategic" and update.confidence >= self.strategic_confidence_threshold:
                    promoted = await self.strategic_store.add_strategic_rule(
                        agent_name=agent_name,
                        rule_text=update.update_text,
                        rationale=update.rationale,
                        confidence=update.confidence,
                        domain=update.domain,
                        source_task_id=task_id,
                    )
                    if promoted:
                        promo_msg = f"[Guideliner] ⭐ STRATEGIC RULE PROMOTED for {agent_name} (confidence: {update.confidence:.2f}, domain: {update.domain})"
                        logger.info(promo_msg)
                
                # Log the acceptance
                scope_label = f"[{update.scope.upper()}]"
                message = f"[Guideliner] ✓ ACCEPTED {analysis_type} update {scope_label} for {agent_name} ({reason}): {update.update_text[:100]}..."
                logger.info(message)
                
                # Return the update text to be immediately applied to the system prompt
                return (update.update_text, update.scope)
            else:
                # Update was rejected
                message = f"[Guideliner] ✗ REJECTED {analysis_type} update for {agent_name} ({reason}): {update.update_text[:100]}..."
                logger.info(message)
                return None
        else:
            # No update was generated
            logger.info(f"[Guideliner] No update generated for {agent_name}")
        
        return None
    
    def _build_step_summary(
        self,
        model_output: Optional[str],
        tool_calls: Optional[str],
        observations: Optional[str],
        truncate: bool = True,
    ) -> str:
        """Build a summary of the last step.
        
        Args:
            model_output: Model's output text
            tool_calls: Tool calls attempted
            observations: Observations received
            truncate: If True (default), truncate long fields for efficiency.
                If False, preserve full context for detailed analysis.
        
        Returns:
            Formatted step summary string
        """
        summary_parts = []
        
        if model_output:
            if truncate and len(model_output) > 200:
                summary_parts.append(f"Model output: {model_output[:200]}...")
            else:
                summary_parts.append(f"Model output: {model_output}")
        
        if tool_calls:
            if truncate and len(tool_calls) > 150:
                summary_parts.append(f"Tool calls: {tool_calls[:150]}...")
            else:
                summary_parts.append(f"Tool calls: {tool_calls}")
        
        if observations:
            if truncate and len(observations) > 150:
                summary_parts.append(f"Observations: {observations[:150]}...")
            else:
                summary_parts.append(f"Observations: {observations}")
        
        return "\n".join(summary_parts) if summary_parts else "(no step details)"
    
    def _get_applied_rules(self, task_id: Optional[str], agent_name: str) -> list:
        """
        Get list of applied rules for a specific task and agent.
        
        Args:
            task_id: Task ID (can be None)
            agent_name: Agent name
            
        Returns:
            List of applied rule texts
        """
        if not task_id or task_id not in self._applied_rules_by_task:
            return []
        
        task_rules = self._applied_rules_by_task[task_id]
        return task_rules.get(agent_name, [])
    
    def _add_applied_rule(self, task_id: Optional[str], agent_name: str, rule_text: str, rationale: str):
        """
        Track an applied rule for a specific task and agent.
        
        Args:
            task_id: Task ID (can be None)
            agent_name: Agent name
            rule_text: The rule text that was applied
            rationale: Why the rule was applied
        """
        if not task_id:
            return
        
        if task_id not in self._applied_rules_by_task:
            self._applied_rules_by_task[task_id] = {}
        
        if agent_name not in self._applied_rules_by_task[task_id]:
            self._applied_rules_by_task[task_id][agent_name] = []
        
        self._applied_rules_by_task[task_id][agent_name].append({
            "rule": rule_text,
            "rationale": rationale,
        })
    
    def get_strategic_rules_for_agent(self, agent_name: str) -> str:
        """
        Get strategic rules formatted as text to append to agent's initial system prompt.
        
        This should be called during agent initialization to load cross-task learned rules.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Formatted string ready to append to system prompt (empty if no rules)
        """
        return self.strategic_store.get_strategic_rules_text(agent_name)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about guideline generation and strategic rules."""
        stats = {}
        
        # Add history stats (only if history tracking enabled)
        if self.store_history and self.history:
            stats.update(self.history.get_statistics())
        
        # Add strategic memory stats
        if self.strategic_store:
            stats['strategic_rules_count'] = {}
            for agent_name in self.strategic_store.rules.keys():
                stats['strategic_rules_count'][agent_name] = self.strategic_store.get_rule_count(agent_name)
        
        return stats
