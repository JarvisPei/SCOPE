"""
StrategicMemoryStore: Manages long-term, cross-task strategic rules for agents.

This store handles the persistence and retrieval of high-confidence, generalizable
rules that should be applied across all tasks for an agent.
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .utils import lock_file, unlock_file

logger = logging.getLogger("scope.strategic_store")


class StrategicMemoryStore:
    """
    Manages strategic (cross-task) prompt rules with simple confidence-based promotion.
    
    Features:
    - Stores rules per agent and domain
    - Auto-limits rules per domain (keeps top N by confidence)
    - Simple duplicate detection
    - No complex metrics tracking (KISS principle)
    """

    def __init__(
        self,
        exp_path: str,
        max_rules_per_domain: int = 10,
        optimizer_model = None,
        enable_rule_optimization: bool = True,
    ):
        """
        Initialize the strategic memory store.
        
        Args:
            exp_path: Experiment path (e.g., workdir/hle)
            max_rules_per_domain: Maximum strategic rules per domain per agent
            optimizer_model: Model for rule optimization (optional)
            enable_rule_optimization: Whether to enable automatic optimization
        """
        self.exp_path = exp_path
        self.max_rules_per_domain = max_rules_per_domain
        self.optimizer_model = optimizer_model
        self.enable_rule_optimization = enable_rule_optimization

        # Create strategic_memory directory
        self.strategic_dir = os.path.join(exp_path, "strategic_memory")
        os.makedirs(self.strategic_dir, exist_ok=True)

        self.global_rules_path = os.path.join(self.strategic_dir, "global_rules.json")

        # Load existing rules
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Load strategic rules from disk."""
        if os.path.exists(self.global_rules_path):
            try:
                with open(self.global_rules_path, encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[StrategicMemoryStore] Failed to load rules: {e}")
        return {}

    def _save_rules(self, force_overwrite: bool = False):
        """Save strategic rules to disk with file locking for concurrent safety.
        
        Args:
            force_overwrite: If True, skip merge and directly overwrite. Used after optimization.
        """
        try:
            # Use file locking to prevent concurrent write conflicts
            mode = 'r+' if os.path.exists(self.global_rules_path) else 'w'

            with open(self.global_rules_path, mode, encoding='utf-8') as f:
                try:
                    # Acquire exclusive lock (cross-platform)
                    lock_file(f)

                    # Read current content from disk (for merge if needed)
                    if mode == 'r+' and not force_overwrite:
                        f.seek(0)
                        try:
                            disk_rules = json.load(f)
                        except (json.JSONDecodeError, ValueError):
                            disk_rules = {}

                        # Merge our rules with disk rules (deep merge)
                        # This handles the case where another process added rules
                        for agent_name, domains in self.rules.items():
                            if agent_name not in disk_rules:
                                disk_rules[agent_name] = {}
                            for domain, rules in domains.items():
                                if domain not in disk_rules[agent_name]:
                                    disk_rules[agent_name][domain] = []

                                # Merge rules, avoiding duplicates
                                existing_texts = {r.get("rule", "").strip().lower()
                                                for r in disk_rules[agent_name][domain]}
                                for rule in rules:
                                    rule_text = rule.get("rule", "").strip().lower()
                                    if rule_text not in existing_texts:
                                        disk_rules[agent_name][domain].append(rule)
                                        existing_texts.add(rule_text)

                                # Re-sort by confidence and enforce limit
                                disk_rules[agent_name][domain].sort(
                                    key=lambda x: x.get("confidence", 0),
                                    reverse=True
                                )
                                disk_rules[agent_name][domain] = disk_rules[agent_name][domain][:self.max_rules_per_domain]

                        # Write merged data
                        f.seek(0)
                        f.truncate()
                        json.dump(disk_rules, f, indent=2, ensure_ascii=False)
                    else:
                        # New file or force overwrite: directly write self.rules
                        if mode == 'r+':
                            f.seek(0)
                            f.truncate()
                        json.dump(self.rules, f, indent=2, ensure_ascii=False)

                finally:
                    # Release lock (cross-platform)
                    unlock_file(f)

        except Exception as e:
            logger.warning(f"[StrategicMemoryStore] Failed to save rules: {e}")

    def _is_duplicate(self, agent_name: str, domain: str, rule_text: str) -> bool:
        """
        Check if a similar rule already exists (simple substring matching).
        
        For more advanced deduplication, could use:
        - Semantic similarity (sentence embeddings)
        - Edit distance (Levenshtein)
        
        But keeping it simple for now.
        """
        if agent_name not in self.rules or domain not in self.rules[agent_name]:
            return False

        normalized_new = rule_text.strip().lower()
        for existing in self.rules[agent_name][domain]:
            normalized_existing = existing["rule"].strip().lower()

            # Check if either is substring of the other (covers similar rules)
            if normalized_new in normalized_existing or normalized_existing in normalized_new:
                return True

            # Check if they share >70% of words (simple word overlap)
            new_words = set(normalized_new.split())
            existing_words = set(normalized_existing.split())
            if len(new_words) > 0 and len(existing_words) > 0:
                overlap = len(new_words & existing_words)
                similarity = overlap / max(len(new_words), len(existing_words))
                if similarity > 0.7:
                    return True

        return False

    async def add_strategic_rule(
        self,
        agent_name: str,
        rule_text: str,
        rationale: str,
        confidence: float,
        domain: str,
        source_task_id: Optional[str] = None,
    ) -> bool:
        """
        Add a strategic rule if it meets criteria.
        
        Criteria:
        1. Confidence >= 0.85 (high confidence only)
        2. Not a duplicate
        
        Args:
            agent_name: Name of the agent
            rule_text: The rule text
            rationale: Why this rule helps
            confidence: Confidence score (0.0 to 1.0)
            domain: Domain category
            source_task_id: Optional task ID where rule was discovered
            
        Returns:
            True if rule was added, False if rejected
        """
        # Reload from disk to get latest state (for concurrent execution)
        self.rules = self._load_rules()

        # Check confidence threshold
        if confidence < 0.85:
            return False

        # Check for duplicates
        if self._is_duplicate(agent_name, domain, rule_text):
            return False

        # Initialize nested structure if needed
        if agent_name not in self.rules:
            self.rules[agent_name] = {}
        if domain not in self.rules[agent_name]:
            self.rules[agent_name][domain] = []

        # Add rule
        rule_entry = {
            "rule": rule_text,
            "rationale": rationale,
            "confidence": confidence,
            "added_timestamp": datetime.now().isoformat(),
            "source_task_id": source_task_id,
        }
        self.rules[agent_name][domain].append(rule_entry)

        # Sort by confidence (highest first)
        self.rules[agent_name][domain].sort(key=lambda x: x["confidence"], reverse=True)

        # Check if optimization is needed
        optimization_occurred = False
        if len(self.rules[agent_name][domain]) > self.max_rules_per_domain:
            if self.enable_rule_optimization and self.optimizer_model:
                # Trigger optimization
                optimized_rules = await self._optimize_domain_rules(agent_name, domain)
                self.rules[agent_name][domain] = optimized_rules
                optimization_occurred = True
                # Fallback truncation if optimizer fails to reduce enough
                if len(self.rules[agent_name][domain]) > self.max_rules_per_domain:
                    logger.info(f"[RuleOptimization] Truncating after insufficient optimization for {agent_name}.{domain}: {len(self.rules[agent_name][domain])} -> {self.max_rules_per_domain} rules")
                    self.rules[agent_name][domain] = self.rules[agent_name][domain][:self.max_rules_per_domain]

            else:
                # Fallback: simple truncation
                logger.info(f"[RuleOptimization] Truncating without optimization for {agent_name}.{domain}: {len(self.rules[agent_name][domain])} -> {self.max_rules_per_domain} rules")
                self.rules[agent_name][domain] = self.rules[agent_name][domain][:self.max_rules_per_domain]
                optimization_occurred = True

        # Save to disk (force overwrite if optimization occurred to avoid merge conflicts)
        self._save_rules(force_overwrite=optimization_occurred)

        return True

    async def _optimize_domain_rules(self, agent_name: str, domain: str) -> List[Dict[str, Any]]:
        """
        Optimize rules in a domain when count exceeds threshold.
        
        Args:
            agent_name: Name of the agent
            domain: Domain name
            
        Returns:
            Optimized list of rules (length <= max_rules_per_domain)
        """
        from .memory_optimizer import MemoryOptimizer

        current_rules = self.rules[agent_name][domain]
        rules_before = len(current_rules)

        # Target: optimize to ~70% of max to leave room for new rules
        target_count = int(self.max_rules_per_domain * 0.8)
        # target_count = int(self.max_rules_per_domain - 1)

        logger.info(f"[RuleOptimization] Triggered for {agent_name}.{domain}: {rules_before} rules -> target {target_count}")

        try:
            # Initialize orchestrator
            orchestrator = MemoryOptimizer(
                model=self.optimizer_model,
            )

            # Run optimization
            optimized_rules = await orchestrator.optimize_rules(
                rules_list=current_rules,
                target_count=target_count
            )

            rules_after = len(optimized_rules)
            logger.info(f"[RuleOptimization] Completed for {agent_name}.{domain}: {rules_before} -> {rules_after} rules")

            return optimized_rules

        except Exception as e:
            logger.warning(f"[RuleOptimization] Failed for {agent_name}.{domain}: {e}")
            logger.warning("[RuleOptimization] Falling back to truncation")
            # Fallback to simple truncation if optimization fails
            return current_rules[:self.max_rules_per_domain]

    def get_strategic_rules(
        self,
        agent_name: str,
        max_rules_per_domain: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all strategic rules for an agent, organized by domain.
        
        IMPORTANT: Reloads from disk to get the latest rules from concurrent tasks.
        
        Args:
            agent_name: Name of the agent
            max_rules_per_domain: Optional limit (overrides default)
            
        Returns:
            Dict mapping domain -> list of rules
        """
        # Reload from disk to get latest state (critical for concurrent execution!)
        # Without this, we'd return stale rules and miss rules added by other processes
        self.rules = self._load_rules()

        if agent_name not in self.rules:
            return {}

        limit = max_rules_per_domain or self.max_rules_per_domain

        # Return limited rules per domain
        result = {}
        for domain, rules in self.rules[agent_name].items():
            result[domain] = rules[:limit]

        return result

    def get_strategic_rules_text(
        self,
        agent_name: str,
        max_rules_per_domain: Optional[int] = None,
    ) -> str:
        """
        Get strategic rules formatted as text to append to system prompt.
        
        IMPORTANT: Automatically reloads from disk (via get_strategic_rules) to get
        the latest rules from concurrent tasks.
        
        Args:
            agent_name: Name of the agent
            max_rules_per_domain: Optional limit per domain
            
        Returns:
            Formatted string ready to append to system prompt
        """
        # This calls get_strategic_rules(), which reloads from disk
        rules_by_domain = self.get_strategic_rules(agent_name, max_rules_per_domain)

        if not rules_by_domain:
            return ""

        lines = ["\n## Strategic Guidelines (Learned Best Practices):"]
        lines.append("These are high-confidence rules learned from previous tasks:")

        for domain, rules in rules_by_domain.items():
            if rules:
                lines.append(f"\n### {domain.replace('_', ' ').title()}:")
                for rule in rules:
                    lines.append(f"- {rule['rule']}")

        return "\n".join(lines)

    def get_rule_count(self, agent_name: str) -> int:
        """Get total number of strategic rules for an agent."""
        if agent_name not in self.rules:
            return 0
        return sum(len(rules) for rules in self.rules[agent_name].values())

    def clear_rules(self, agent_name: Optional[str] = None, domain: Optional[str] = None):
        """
        Clear strategic rules (for manual maintenance).
        
        Args:
            agent_name: If None, clears all agents. If specified, clears that agent.
            domain: If specified (and agent_name specified), clears only that domain.
        """
        if agent_name is None:
            self.rules = {}
        elif domain is None:
            if agent_name in self.rules:
                del self.rules[agent_name]
        else:
            if agent_name in self.rules and domain in self.rules[agent_name]:
                del self.rules[agent_name][domain]

        # Use force_overwrite to ensure deletions are not undone by merge
        self._save_rules(force_overwrite=True)

