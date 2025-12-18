"""
GuidelineSynthesizer: Synthesizes guidelines from agent execution traces.

This module implements the Guideline Synthesis component of SCOPE:
- Generator (π_φ): Generates candidate guidelines from traces
- Selector (π_σ): Selects the best guideline via Best-of-N

Synthesis Modes:
- Efficiency: Lightweight, faster analysis (original design)
- Thoroughness: Comprehensive analysis with prioritized dimensions and domain learning
"""
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .prompts import (
    ERROR_REFLECTION_PROMPT,
    QUALITY_REFLECTION_PROMPT_EFFICIENCY,
    QUALITY_REFLECTION_PROMPT_THOROUGHNESS,
    SELECTOR_PROMPT,
)

logger = logging.getLogger("scope.synthesizer")


@dataclass
class Guideline:
    """Represents a synthesized guideline for prompt evolution."""
    update_text: str
    rationale: str
    scope: str = "session"  # tactical | strategic
    confidence: str = "medium"  # low | medium | high
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "update_text": self.update_text,
            "rationale": self.rationale,
            "scope": self.scope,
            "confidence": self.confidence,
        }


class GuidelineSynthesizer:
    """
    Small agent that reflects on errors/behaviors and proposes prompt updates.
    
    This is designed to be lightweight and independent - it only needs a model
    that has a generate() method compatible with the standard ChatMessage format.
    
    Prompts are imported from scope.prompts module for easy customization.
    """

    def __init__(self, model, candidate_models=None, use_best_of_n=False, use_thoroughness_mode=True):
        """
        Initialize the guideline synthesizer.
        
        Args:
            model: A model instance with generate() method that accepts messages
                   and returns a ChatMessage with .content
            candidate_models: List of additional model instances for "best of N" approach
            use_best_of_n: Enable multi-model candidate generation with selection
            use_thoroughness_mode: Use thoroughness mode (comprehensive) vs efficiency mode (lightweight)
        """
        self.model = model
        self.candidate_models = candidate_models or []
        self.use_best_of_n = use_best_of_n
        self.use_thoroughness_mode = use_thoroughness_mode
    
    async def _generate_single_candidate(
        self,
        model,
        prompt: str,
        model_name: str = "unknown"
    ) -> Optional[Guideline]:
        """
        Generate a single candidate update from a given model.
        
        Args:
            model: Model instance to generate with
            prompt: The reflection prompt
            model_name: Name of the model for logging
            
        Returns:
            Guideline if successful, None if generation fails
        """
        try:
            # Create message format
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            
            # Call model - wrap in ChatMessage-like object for compatibility
            from dataclasses import dataclass as dc
            
            @dc
            class SimpleMessage:
                role: str
                content: Any
                tool_calls: Any = None
                
            simple_messages = [SimpleMessage(role=m["role"], content=m["content"]) for m in messages]
            
            # Generate response (await the coroutine)
            response = await model.generate(simple_messages)
            
            # Parse response
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            update_data = self._extract_json(content)
            
            if update_data:
                return Guideline(
                    update_text=update_data.get("update_text", ""),
                    rationale=update_data.get("rationale", ""),
                    scope="session",
                    confidence=update_data.get("confidence", "medium"),
                )
            
            return None
            
        except Exception as e:
            msg = f"[Synthesizer] Error generating candidate from {model_name}: {e}"
            logger.warning(msg)
            return None
    
    async def _generate_candidates(
        self,
        prompt: str,
    ) -> List[Guideline]:
        """
        Generate multiple candidate updates from all available models.
        
        Args:
            prompt: The reflection prompt
            
        Returns:
            List of Guideline objects (filters out None/failed generations)
        """
        # Combine primary model with candidate models
        all_models = [self.model] + self.candidate_models
        model_names = ["primary"] + [f"candidate_{i}" for i in range(len(self.candidate_models))]
        
        logger.info(f"[Synthesizer] Generating {len(all_models)} candidates in parallel")
        
        # Generate candidates in parallel
        tasks = [
            self._generate_single_candidate(model, prompt, name)
            for model, name in zip(all_models, model_names)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None and exceptions
        candidates = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"[Synthesizer] Model {model_names[i]} raised exception: {result}")
            elif result is not None:
                candidates.append(result)
        
        logger.info(f"[Synthesizer] Successfully generated {len(candidates)}/{len(all_models)} candidates")
        
        return candidates
    
    async def _select_best_update(
        self,
        candidates: List[Guideline],
        agent_name: str,
        agent_role: str,
        task: str,
        issue_type: str,
        issue_details: str,
        current_system_prompt: str,
    ) -> Optional[Guideline]:
        """
        Select the best update from multiple candidates.
        
        Args:
            candidates: List of Guideline candidates
            agent_name: Name of the agent
            agent_role: Role of the agent
            task: The task being performed
            issue_type: "error" or "quality"
            issue_details: Details about the error or quality issue
            current_system_prompt: Current system prompt for reference
            
        Returns:
            Best Guideline, or None if selection fails
        """
        if not candidates:
            logger.warning("[Synthesizer] No candidates to select from")
            return None
        
        if len(candidates) == 1:
            logger.info("[Synthesizer] Only one candidate, returning it directly")
            return candidates[0]
        
        try:
            # Format candidates for comparison
            candidates_text = ""
            for i, candidate in enumerate(candidates):
                candidates_text += f"\n[Candidate {i}]\n"
                candidates_text += f"Update: {candidate.update_text}\n"
                candidates_text += f"Rationale: {candidate.rationale}\n"
                candidates_text += f"Confidence: {candidate.confidence}\n"
            
            # Format the selector prompt
            prompt = SELECTOR_PROMPT.format(
                agent_name=agent_name,
                agent_role=agent_role,
                task=task,
                issue_type=issue_type,
                issue_details=issue_details,
                current_system_prompt=current_system_prompt,
                candidates=candidates_text,
            )
            
            # Create message format
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            
            # Call model
            from dataclasses import dataclass as dc
            
            @dc
            class SimpleMessage:
                role: str
                content: Any
                tool_calls: Any = None
                
            simple_messages = [SimpleMessage(role=m["role"], content=m["content"]) for m in messages]
            
            # Generate response (await the coroutine)
            response = await self.model.generate(simple_messages)
            
            # Parse response
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            selection_data = self._extract_json(content)
            
            if selection_data and "selected_index" in selection_data:
                selected_idx = selection_data["selected_index"]
                rationale = selection_data.get("rationale", "")
                
                if 0 <= selected_idx < len(candidates):
                    logger.info(f"[Synthesizer] Selected candidate {selected_idx}: {rationale}")
                    return candidates[selected_idx]
                else:
                    logger.warning(f"[Synthesizer] Invalid index {selected_idx}, using first candidate")
                    return candidates[0]
            else:
                logger.warning("[Synthesizer] Failed to parse selection, using first candidate")
                return candidates[0]
            
        except Exception as e:
            msg = f"[Synthesizer] Error selecting best update: {e}, using first candidate"
            logger.warning(msg)
            return candidates[0] if candidates else None
    
    async def generate_update_from_error(
        self,
        agent_name: str,
        agent_role: str,
        task: str,
        error_type: str,
        error_message: str,
        last_step_summary: str,
        current_system_prompt: str,
        applied_rules: list = None,
    ) -> Optional[Guideline]:
        """
        Generate a prompt update based on error analysis.
        
        Args:
            applied_rules: List of already applied rules to avoid duplication
        
        Returns:
            Guideline if successful, None if generation fails
        """
        try:
            # Format applied rules for prompt
            if applied_rules:
                rules_text = "\n".join([f"- {r['rule']}" for r in applied_rules])
            else:
                rules_text = "(none)"
            
            # Format the reflection prompt (no truncation limits to preserve context)
            prompt = ERROR_REFLECTION_PROMPT.format(
                agent_name=agent_name,
                agent_role=agent_role,
                task=task,  # No truncation
                error_type=error_type,
                error_message=error_message,  # No truncation
                last_step_summary=last_step_summary,  # No truncation
                current_system_prompt=current_system_prompt,  # No truncation
                applied_rules=rules_text,
            )
            
            # Check if best-of-N is enabled
            if self.use_best_of_n and self.candidate_models:
                logger.info("[Synthesizer] Using best-of-N approach for error reflection")
                
                # Generate multiple candidates
                candidates = await self._generate_candidates(prompt)
                
                if not candidates:
                    logger.warning("[Synthesizer] No candidates generated, returning None")
                    return None
                
                # Select best candidate
                issue_details = f"Error Type: {error_type}\nError Message: {error_message}\n\nLast Step:\n{last_step_summary}"
                best_update = await self._select_best_update(
                    candidates=candidates,
                    agent_name=agent_name,
                    agent_role=agent_role,
                    task=task,
                    issue_type="error",
                    issue_details=issue_details,
                    current_system_prompt=current_system_prompt,
                )
                
                return best_update
            
            else:
                # Original single-model approach
                # Create message format (simple dict-based, no imports from src)
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]
                
                # Call model - wrap in ChatMessage-like object for compatibility
                from dataclasses import dataclass as dc
                
                @dc
                class SimpleMessage:
                    role: str
                    content: Any
                    tool_calls: Any = None
                    
                simple_messages = [SimpleMessage(role=m["role"], content=m["content"]) for m in messages]
                
                # Generate response (await the coroutine)
                response = await self.model.generate(simple_messages)
                
                # Parse response
                content = response.content if hasattr(response, 'content') else str(response)
                
                # Extract JSON from response
                update_data = self._extract_json(content)
                
                if update_data:
                    return Guideline(
                        update_text=update_data.get("update_text", ""),
                        rationale=update_data.get("rationale", ""),
                        scope="session",
                        confidence=update_data.get("confidence", "medium"),
                    )
                
                return None
            
        except Exception as e:
            msg = f"[Synthesizer] Error generating update from error: {e}"
            logger.warning(msg)
            return None
    
    async def generate_update_from_quality(
        self,
        agent_name: str,
        agent_role: str,
        task: str,
        last_step_summary: str,
        current_system_prompt: str,
        applied_rules: list = None,
    ) -> Optional[Guideline]:
        """
        Generate a prompt update based on quality/performance analysis of a successful step.
        
        Args:
            applied_rules: List of already applied rules to avoid duplication
        
        Returns:
            Guideline if successful and improvement found, None otherwise
        """
        try:
            # Format applied rules for prompt
            if applied_rules:
                rules_text = "\n".join([f"- {r['rule']}" for r in applied_rules])
            else:
                rules_text = "(none)"
            
            if self.use_thoroughness_mode:
                prompt = QUALITY_REFLECTION_PROMPT_THOROUGHNESS.format(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    task=task,
                    last_step_summary=last_step_summary,
                    current_system_prompt=current_system_prompt,
                    applied_rules=rules_text,
                )
            else:
                prompt = QUALITY_REFLECTION_PROMPT_EFFICIENCY.format(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    task=task,
                    last_step_summary=last_step_summary,
                    current_system_prompt=current_system_prompt,
                    applied_rules=rules_text,
                )
            
            # Check if best-of-N is enabled
            if self.use_best_of_n and self.candidate_models:
                logger.info("[Synthesizer] Using best-of-N approach for quality reflection")
                
                # Generate multiple candidates
                candidates = await self._generate_candidates(prompt)
                
                # Filter out empty/no-improvement candidates
                valid_candidates = []
                for candidate in candidates:
                    update_text = candidate.update_text.strip()
                    if update_text and update_text.lower() not in ["", "no improvement needed", "none"]:
                        valid_candidates.append(candidate)
                
                if not valid_candidates:
                    logger.info("[Synthesizer] No valid quality improvements suggested by any candidate")
                    return None
                
                # Select best candidate
                issue_details = f"Step Details:\n{last_step_summary}"
                best_update = await self._select_best_update(
                    candidates=valid_candidates,
                    agent_name=agent_name,
                    agent_role=agent_role,
                    task=task,
                    issue_type="quality",
                    issue_details=issue_details,
                    current_system_prompt=current_system_prompt,
                )
                
                return best_update
            
            else:
                # Original single-model approach
                # Create message format
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]
                
                # Call model
                from dataclasses import dataclass as dc
                
                @dc
                class SimpleMessage:
                    role: str
                    content: Any
                    tool_calls: Any = None
                    
                simple_messages = [SimpleMessage(role=m["role"], content=m["content"]) for m in messages]
                
                # Generate response (await the coroutine)
                response = await self.model.generate(simple_messages)
                
                # Parse response
                content = response.content if hasattr(response, 'content') else str(response)
                
                logger.info(f"[Synthesizer] Quality analysis response: {content[:200]}...")
                
                # Extract JSON from response
                update_data = self._extract_json(content)
                
                if update_data:
                    update_text = update_data.get("update_text", "").strip()
                    
                    # Only return if there's actual improvement suggested
                    if update_text and update_text.lower() not in ["", "no improvement needed", "none"]:
                        return Guideline(
                            update_text=update_text,
                            rationale=update_data.get("rationale", ""),
                            scope="session",
                            confidence=update_data.get("confidence", "medium"),
                        )
                    else:
                        logger.info(f"[Synthesizer] Quality analysis found no improvement needed: '{update_text}'")
                else:
                    logger.warning("[Synthesizer] Failed to extract JSON from quality analysis response")
                
                return None
            
        except Exception as e:
            msg = f"[Synthesizer] Error generating update from quality: {e}"
            logger.warning(msg)
            return None
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text that may contain markdown code blocks."""
        try:
            # Try direct parse first
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in code blocks
        import re
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # Try to find any JSON object
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match)
                if "update_text" in data:
                    return data
            except json.JSONDecodeError:
                continue
        
        return None
