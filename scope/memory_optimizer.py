"""
MemoryOptimizer: Automatic optimization of strategic memory.

This module implements the Memory Optimization (π_ω) component of SCOPE:
1. GuidelineAnalyzer: Identifies consolidation, subsumption, and conflict opportunities
2. ConflictResolver: Merges contradictory guidelines
3. SubsumptionPruner: Removes specific guidelines covered by general ones
4. Consolidator: Merges similar guidelines into comprehensive ones
5. MemoryOptimizer: Coordinates the optimization pipeline
"""
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .prompts import (
    CONFLICT_RESOLVE_PROMPT,
    RULE_ANALYSIS_PROMPT,
    RULE_MERGE_PROMPT,
    SUBSUMPTION_VERIFY_PROMPT,
)

logger = logging.getLogger("scope.memory_optimizer")


@dataclass
class SimpleMessage:
    """Simple message format for model communication."""
    role: str
    content: Any
    tool_calls: Any = None


class RuleAnalyzer:
    """
    Analyzes rules in a domain and identifies optimization opportunities.
    """
    
    def __init__(self, model):
        """
        Initialize the analyzer.
        
        Args:
            model: LLM model for analysis
        """
        self.model = model
    
    async def analyze(self, rules_list: List[Dict[str, Any]]) -> Dict[str, List]:
        """
        Analyze rules and identify optimization opportunities.
        
        Args:
            rules_list: List of rule dictionaries (each will get a stable '_opt_id' field)
            
        Returns:
            Dict with keys:
                - consolidation: [[id1, id2], ...] groups that can merge (uses _opt_id, not index)
                - subsumption: [[general_id, specific_id], ...] pairs (uses _opt_id)
                - conflicts: [[id1, id2], ...] conflicting pairs (uses _opt_id)
        """
        if len(rules_list) <= 1:
            return {"consolidation": [], "subsumption": [], "conflicts": []}
        
        # Assign stable IDs to each rule (if not already present)
        for i, rule in enumerate(rules_list):
            if '_opt_id' not in rule:
                rule['_opt_id'] = i
        
        # Build the prompt with indexed rules (using _opt_id for stability)
        rules_text = ""
        for rule in rules_list:
            rule_id = rule['_opt_id']
            rule_text = rule.get("rule", str(rule))
            rules_text += f"Rule {rule_id}: {rule_text}\n"
        
        prompt = RULE_ANALYSIS_PROMPT.format(
            num_rules=len(rules_list),
            rules_text=rules_text,
        )

        try:
            messages = [
                SimpleMessage(
                    role="user",
                    content=[{"type": "text", "text": prompt}]
                )
            ]
            
            response = await self.model.generate(messages)
            response_text = response.content.strip()
            
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text
            
            # Try to find JSON object boundaries if parsing fails
            try:
                analysis = json.loads(json_text)
            except json.JSONDecodeError as e1:
                # Try to extract just the JSON object
                start_idx = json_text.find("{")
                end_idx = json_text.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_text = json_text[start_idx:end_idx]
                    try:
                        analysis = json.loads(json_text)
                    except json.JSONDecodeError:
                        # Last resort: try to fix common JSON issues
                        # Replace single quotes with double quotes
                        json_text_fixed = json_text.replace("'", '"')
                        try:
                            analysis = json.loads(json_text_fixed)
                        except json.JSONDecodeError:
                            # Give up and raise original error
                            raise e1
                else:
                    raise e1
            
            # Validate structure
            analysis.setdefault("consolidation", [])
            analysis.setdefault("subsumption", [])
            analysis.setdefault("conflicts", [])
            
            logger.info(f"[RuleAnalyzer] Found {len(analysis['consolidation'])} consolidation groups, "
                               f"{len(analysis['subsumption'])} subsumptions, {len(analysis['conflicts'])} conflicts")
            
            return analysis
            
        except Exception as e:
            logger.warning(f"[RuleAnalyzer] Analysis failed: {e}")
            try:
                logger.warning(f"[RuleAnalyzer] Response was: {response_text[:500]}")
            except:
                pass
            # Return empty results on failure
            return {"consolidation": [], "subsumption": [], "conflicts": []}


class ConsolidationOptimizer:
    """
    Merges similar rules into more comprehensive ones.
    """
    
    def __init__(self, model):
        self.model = model
    
    async def consolidate(
        self,
        rules_list: List[Dict[str, Any]],
        consolidation_groups: List[List[int]]
    ) -> List[Dict[str, Any]]:
        """
        Merge groups of similar rules.
        
        Args:
            rules_list: Original list of rules (with _opt_id fields)
            consolidation_groups: List of ID groups to merge (uses _opt_id, not indices)
            
        Returns:
            New list with merged rules
        """
        if not consolidation_groups:
            return rules_list
        
        # Build a map from _opt_id to rule
        id_to_rule = {rule['_opt_id']: rule for rule in rules_list}
        
        # Track which IDs are being merged
        merged_ids = set()
        for group in consolidation_groups:
            merged_ids.update(group)
        
        # Keep unmerged rules
        result = []
        for rule in rules_list:
            if rule['_opt_id'] not in merged_ids:
                result.append(rule)
        
        # Process each consolidation group
        for group in consolidation_groups:
            if len(group) < 2:
                # Can't merge a single rule, keep it
                for rule_id in group:
                    if rule_id in id_to_rule:
                        result.append(id_to_rule[rule_id])
                continue
            
            # Get rules to merge
            rules_to_merge = []
            for rule_id in group:
                if rule_id in id_to_rule:
                    rules_to_merge.append(id_to_rule[rule_id])
            
            if not rules_to_merge:
                continue
            
            # Merge these rules
            merged_rule = await self._merge_rules(rules_to_merge, group)
            if merged_rule:
                result.append(merged_rule)
                logger.info(f"[ConsolidationOptimizer] Merged rules {group} into 1 rule")
            else:
                # If merge fails, keep original rules
                result.extend(rules_to_merge)
        
        return result
    
    async def _merge_rules(self, rules: List[Dict[str, Any]], indices: List[int]) -> Optional[Dict[str, Any]]:
        """Merge multiple rules into one comprehensive rule."""
        rules_text = ""
        for i, rule in enumerate(rules):
            rule_text = rule.get("rule", str(rule))
            rationale = rule.get("rationale", "")
            rules_text += f"\nRule {indices[i]}:\n  Text: {rule_text}\n  Rationale: {rationale}\n"
        
        prompt = RULE_MERGE_PROMPT.format(rules_text=rules_text)

        try:
            messages = [
                SimpleMessage(
                    role="user",
                    content=[{"type": "text", "text": prompt}]
                )
            ]
            
            response = await self.model.generate(messages)
            response_text = response.content.strip()
            
            # Extract JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text
            
            # Try to find JSON object boundaries if parsing fails
            try:
                merged = json.loads(json_text)
            except json.JSONDecodeError as e1:
                start_idx = json_text.find("{")
                end_idx = json_text.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_text = json_text[start_idx:end_idx]
                    try:
                        merged = json.loads(json_text)
                    except json.JSONDecodeError:
                        json_text_fixed = json_text.replace("'", '"')
                        try:
                            merged = json.loads(json_text_fixed)
                        except json.JSONDecodeError:
                            raise e1
                else:
                    raise e1
            
            # Create new rule entry with max confidence from originals
            max_confidence = max(r.get("confidence", 0.85) for r in rules)
            
            # Use the ID of the first rule in the group as the stable ID for the new rule
            new_rule = {
                "rule": merged["rule"],
                "rationale": merged["rationale"],
                "confidence": max_confidence,
                "added_timestamp": rules[0].get("added_timestamp", ""),
                "source_task_id": rules[0].get("source_task_id", ""),
            }
            if '_opt_id' in rules[0]:
                new_rule['_opt_id'] = rules[0]['_opt_id']
            return new_rule
            
        except Exception as e:
            logger.info(f"[ConsolidationOptimizer] Merge failed: {e}")
            return None


class SubsumptionOptimizer:
    """
    Removes redundant specific rules that are covered by general ones.
    """
    
    def __init__(self, model):
        self.model = model
    
    async def prune_subsumed(
        self,
        rules_list: List[Dict[str, Any]],
        subsumption_pairs: List[List[int]]
    ) -> List[Dict[str, Any]]:
        """
        Remove rules that are subsumed by more general ones.
        
        Args:
            rules_list: Original list of rules (with _opt_id fields)
            subsumption_pairs: [[general_id, specific_id], ...] pairs (uses _opt_id, not indices)
            
        Returns:
            Pruned list with subsumed rules removed
        """
        if not subsumption_pairs:
            return rules_list
        
        # Build a map from _opt_id to rule
        id_to_rule = {rule['_opt_id']: rule for rule in rules_list}
        
        # Collect IDs to remove (specific rules that are subsumed)
        ids_to_remove = set()
        for pair in subsumption_pairs:
            if len(pair) >= 2:
                general_id, specific_id = pair[0], pair[1]
                # Verify subsumption before removing
                if general_id in id_to_rule and specific_id in id_to_rule:
                    if await self._verify_subsumption(id_to_rule, general_id, specific_id):
                        ids_to_remove.add(specific_id)
                        logger.info(f"[SubsumptionOptimizer] Removed rule {specific_id} (subsumed by rule {general_id})")
        
        # Keep rules that aren't subsumed
        result = []
        for rule in rules_list:
            if rule['_opt_id'] not in ids_to_remove:
                result.append(rule)
        
        return result
    
    async def _verify_subsumption(
        self,
        id_to_rule: Dict[int, Dict[str, Any]],
        general_id: int,
        specific_id: int
    ) -> bool:
        """Verify that the general rule truly subsumes the specific one."""
        if general_id not in id_to_rule or specific_id not in id_to_rule:
            return False
        
        general_rule = id_to_rule[general_id].get("rule", "")
        specific_rule = id_to_rule[specific_id].get("rule", "")
        
        prompt = SUBSUMPTION_VERIFY_PROMPT.format(
            general_rule=general_rule,
            specific_rule=specific_rule,
        )

        try:
            messages = [
                SimpleMessage(
                    role="user",
                    content=[{"type": "text", "text": prompt}]
                )
            ]
            
            response = await self.model.generate(messages)
            response_text = response.content.strip()
            
            # Extract JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text
            
            # Try to find JSON object boundaries if parsing fails
            try:
                result = json.loads(json_text)
            except json.JSONDecodeError as e1:
                start_idx = json_text.find("{")
                end_idx = json_text.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_text = json_text[start_idx:end_idx]
                    try:
                        result = json.loads(json_text)
                    except json.JSONDecodeError:
                        json_text_fixed = json_text.replace("'", '"')
                        try:
                            result = json.loads(json_text_fixed)
                        except json.JSONDecodeError:
                            raise e1
                else:
                    raise e1
            return result.get("subsumed", False)
            
        except Exception:
            # If verification fails, don't remove the rule
            return False


class ConflictOptimizer:
    """
    Detects and resolves conflicting rules.
    """
    
    def __init__(self, model):
        self.model = model
    
    async def resolve_conflicts(
        self,
        rules_list: List[Dict[str, Any]],
        conflict_pairs: List[List[int]]
    ) -> List[Dict[str, Any]]:
        """
        Resolve conflicts between rules.
        
        Args:
            rules_list: Original list of rules (with _opt_id fields)
            conflict_pairs: [[id1, id2], ...] pairs of conflicting rules (uses _opt_id, not indices)
            
        Returns:
            List with conflicts resolved
        """
        if not conflict_pairs:
            return rules_list
        
        # Build a map from _opt_id to rule
        id_to_rule = {rule['_opt_id']: rule for rule in rules_list}
        
        # Track which IDs have been resolved
        resolved_ids = set()
        resolved_rules = {}
        
        for pair in conflict_pairs:
            if len(pair) < 2:
                continue
            
            id1, id2 = pair[0], pair[1]
            if id1 not in id_to_rule or id2 not in id_to_rule:
                continue
            
            # Skip if already resolved
            if id1 in resolved_ids or id2 in resolved_ids:
                continue
            
            # Resolve this conflict
            rule1 = id_to_rule[id1]
            rule2 = id_to_rule[id2]
            resolved_rule = await self._resolve_conflict(rule1, rule2, id1, id2)
            
            if resolved_rule:
                resolved_ids.add(id1)
                resolved_ids.add(id2)
                resolved_rules[id1] = resolved_rule  # Store at first ID
                logger.info(f"[ConflictOptimizer] Resolved conflict between rules [{id1}, {id2}]")
        
        # Build result list
        result = []
        for rule in rules_list:
            rule_id = rule['_opt_id']
            if rule_id in resolved_ids:
                # If this is the first ID of a resolved pair, add the resolved rule
                if rule_id in resolved_rules:
                    result.append(resolved_rules[rule_id])
                # Otherwise skip (it's the second ID)
            else:
                # Keep unresolved rules
                result.append(rule)
        
        return result
    
    async def _resolve_conflict(
        self,
        rule1: Dict[str, Any],
        rule2: Dict[str, Any],
        idx1: int,
        idx2: int
    ) -> Optional[Dict[str, Any]]:
        """Resolve conflict between two rules."""
        rule1_text = rule1.get("rule", "")
        rule1_rationale = rule1.get("rationale", "")
        rule2_text = rule2.get("rule", "")
        rule2_rationale = rule2.get("rationale", "")
        
        prompt = CONFLICT_RESOLVE_PROMPT.format(
            idx1=idx1,
            rule1_text=rule1_text,
            rule1_rationale=rule1_rationale,
            idx2=idx2,
            rule2_text=rule2_text,
            rule2_rationale=rule2_rationale,
        )

        try:
            messages = [
                SimpleMessage(
                    role="user",
                    content=[{"type": "text", "text": prompt}]
                )
            ]
            
            response = await self.model.generate(messages)
            response_text = response.content.strip()
            
            # Extract JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text
            
            # Try to find JSON object boundaries if parsing fails
            try:
                resolved = json.loads(json_text)
            except json.JSONDecodeError as e1:
                start_idx = json_text.find("{")
                end_idx = json_text.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_text = json_text[start_idx:end_idx]
                    try:
                        resolved = json.loads(json_text)
                    except json.JSONDecodeError:
                        json_text_fixed = json_text.replace("'", '"')
                        try:
                            resolved = json.loads(json_text_fixed)
                        except json.JSONDecodeError:
                            raise e1
                else:
                    raise e1
            
            # Use max confidence from both rules
            max_confidence = max(
                rule1.get("confidence", 0.85),
                rule2.get("confidence", 0.85)
            )
            
            new_rule = {
                "rule": resolved["rule"],
                "rationale": resolved["rationale"],
                "confidence": max_confidence,
                "added_timestamp": rule1.get("added_timestamp", ""),
                "source_task_id": rule1.get("source_task_id", ""),
            }
            if '_opt_id' in rule1:
                new_rule['_opt_id'] = rule1['_opt_id']
            return new_rule
            
        except Exception as e:
            logger.info(f"[ConflictOptimizer] Resolution failed: {e}")
            return None


class MemoryOptimizer:
    """
    Coordinates the rule optimization pipeline.
    """
    
    def __init__(self, model):
        """
        Initialize the orchestrator.
        
        Args:
            model: LLM model for optimization
        """
        self.model = model
        self.analyzer = RuleAnalyzer(model)
        self.consolidation_optimizer = ConsolidationOptimizer(model)
        self.subsumption_optimizer = SubsumptionOptimizer(model)
        self.conflict_optimizer = ConflictOptimizer(model)
    
    async def optimize_rules(
        self,
        rules_list: List[Dict[str, Any]],
        target_count: int,
        max_iterations: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Optimize a list of rules.
        
        Args:
            rules_list: Original list of rules
            target_count: Target number of rules after optimization
            max_iterations: Maximum optimization passes
            
        Returns:
            Optimized list of rules
        """
        if len(rules_list) <= target_count:
            logger.info(f"[Orchestrator] No optimization needed: {len(rules_list)} <= {target_count}")
            return rules_list
        
        logger.info(f"[Orchestrator] Starting optimization: {len(rules_list)} rules -> target {target_count}")
        
        current_rules = rules_list
        
        for iteration in range(max_iterations):
            if len(current_rules) <= target_count:
                break
            
            logger.info(f"[Orchestrator] Iteration {iteration + 1}: {len(current_rules)} rules")
            
            # Step 1: Analyze current rules
            analysis = await self.analyzer.analyze(current_rules)
            
            # Check if any optimizations found
            has_optimizations = (
                len(analysis.get("consolidation", [])) > 0 or
                len(analysis.get("subsumption", [])) > 0 or
                len(analysis.get("conflicts", [])) > 0
            )
            
            if not has_optimizations:
                logger.info(f"[Orchestrator] No optimizations found in iteration {iteration + 1}")
                break
            
            # Apply optimizations in priority order: Conflicts > Subsumption > Consolidation
            
            # Step 2: HIGHEST PRIORITY - Resolve conflicts first
            if analysis.get("conflicts"):
                current_rules = await self.conflict_optimizer.resolve_conflicts(
                    current_rules,
                    analysis["conflicts"]
                )
            
            # Step 3: HIGH PRIORITY - Apply subsumption pruning
            if analysis.get("subsumption"):
                current_rules = await self.subsumption_optimizer.prune_subsumed(
                    current_rules,
                    analysis["subsumption"]
                )
            
            # Step 4: MEDIUM PRIORITY - Apply consolidation
            if analysis.get("consolidation"):
                current_rules = await self.consolidation_optimizer.consolidate(
                    current_rules,
                    analysis["consolidation"]
                )
            
            logger.info(f"[Orchestrator] After iteration {iteration + 1}: {len(current_rules)} rules")
        
        # Final result
        final_count = len(current_rules)
        logger.info(f"[Orchestrator] Optimization complete: {len(rules_list)} -> {final_count} rules")
        
        # Clean up temporary _opt_id fields
        for rule in current_rules:
            if '_opt_id' in rule:
                del rule['_opt_id']
        
        return current_rules

