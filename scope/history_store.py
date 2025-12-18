"""
GuidelineHistory: Stores guideline generation history for analysis.

This module provides optional history logging for SCOPE:
- Logs all generated guidelines to disk (accepted/rejected)
- Tracks active rules per task for deduplication
- Useful for debugging and analyzing guideline generation patterns

Note: This is optional and disabled by default (store_history=False).
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .utils import lock_file, unlock_file


class GuidelineHistory:
    """
    Stores guideline generation history for analysis and debugging.
    
    This is an optional component, enabled via store_history=True in SCOPEOptimizer.
    
    Structure:
    {exp_path}/prompt_updates/
        ├── {agent_name}.jsonl  # One line per guideline
        └── active_rules.json   # Currently active rules per task
    
    active_rules.json format:
    {
        "task_123": {
            "planning_agent": [{"update_text": "...", ...}],
            ...
        },
        ...
    }
    """

    def __init__(self, exp_path: str):
        """
        Initialize the history store.
        
        Args:
            exp_path: Experiment path (e.g., workdir/hle)
        """
        self.exp_path = exp_path
        self.updates_dir = os.path.join(exp_path, "prompt_updates")
        os.makedirs(self.updates_dir, exist_ok=True)

        self.active_rules_path = os.path.join(self.updates_dir, "active_rules.json")
        self._active_rules = self._load_active_rules()

    def _load_active_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load currently active rules from disk."""
        if os.path.exists(self.active_rules_path):
            try:
                with open(self.active_rules_path, encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_active_rules(self):
        """Save active rules to disk, merging with existing data for concurrent safety."""
        try:
            # Use file locking to prevent concurrent write conflicts
            # Open in read-write mode to allow locking
            mode = 'r+' if os.path.exists(self.active_rules_path) else 'w'

            with open(self.active_rules_path, mode, encoding='utf-8') as f:
                try:
                    # Acquire exclusive lock (cross-platform)
                    lock_file(f)

                    # Read current content from disk
                    if mode == 'r+':
                        f.seek(0)
                        try:
                            disk_rules = json.load(f)
                        except (json.JSONDecodeError, ValueError):
                            disk_rules = {}
                    else:
                        disk_rules = {}

                    # Merge our rules with disk rules (deep merge)
                    for task_id, agents in self._active_rules.items():
                        if task_id not in disk_rules:
                            disk_rules[task_id] = {}
                        for agent_name, rules in agents.items():
                            if agent_name not in disk_rules[task_id]:
                                disk_rules[task_id][agent_name] = []

                            # Merge rules, avoiding duplicates
                            existing_texts = {r.get("update_text", "").strip().lower()
                                            for r in disk_rules[task_id][agent_name]}
                            for rule in rules:
                                rule_text = rule.get("update_text", "").strip().lower()
                                if rule_text not in existing_texts:
                                    disk_rules[task_id][agent_name].append(rule)
                                    existing_texts.add(rule_text)

                    # Write merged data back
                    f.seek(0)
                    f.truncate()
                    json.dump(disk_rules, f, indent=2, ensure_ascii=False)

                finally:
                    # Release lock (cross-platform)
                    unlock_file(f)

        except Exception as e:
            print(f"[GuidelineHistory] Failed to save active rules: {e}")

    def add_update(
        self,
        agent_name: str,
        update_text: str,
        rationale: str,
        error_type: str,
        task_id: Optional[str] = None,
        scope: str = "session",
        confidence: str = "medium",
    ) -> bool:
        """
        Add a new prompt update.
        
        Args:
            agent_name: Name of the agent this update applies to
            update_text: The actual prompt addition
            rationale: Why this update was created
            error_type: Type of error that triggered this
            task_id: Optional task ID
            scope: session | experiment | persistent
            confidence: low | medium | high
            
        Returns:
            True if added successfully, False otherwise
        """
        try:
            # Create entry
            entry = {
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent_name,
                "update_text": update_text,
                "rationale": rationale,
                "error_type": error_type,
                "task_id": task_id,
                "scope": scope,
                "confidence": confidence,
            }

            # Append to history file
            history_file = os.path.join(self.updates_dir, f"{agent_name}.jsonl")
            with open(history_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            # Add to active rules (task-aware, with deduplication)
            if task_id:  # Only track active rules for tasks with IDs
                # Reload from disk to get latest state (for concurrent execution)
                self._active_rules = self._load_active_rules()

                if task_id not in self._active_rules:
                    self._active_rules[task_id] = {}

                if agent_name not in self._active_rules[task_id]:
                    self._active_rules[task_id][agent_name] = []

                # Check if similar rule already exists (simple text match within this task/agent)
                normalized_text = update_text.strip().lower()
                task_agent_rules = self._active_rules[task_id][agent_name]
                if not any(r.get("update_text", "").strip().lower() == normalized_text
                          for r in task_agent_rules):
                    self._active_rules[task_id][agent_name].append({
                        "update_text": update_text,
                        "rationale": rationale,
                        "timestamp": entry["timestamp"],
                        "confidence": confidence,
                    })
                    self._save_active_rules()
                    return True
                else:
                    print(f"[GuidelineHistory] Duplicate rule detected for {agent_name} (task {task_id}), skipping")
                    return False

            return True  # If no task_id, still log to history but don't track active rules

        except Exception as e:
            print(f"[GuidelineHistory] Error adding update: {e}")
            return False

    def get_active_rules(self, agent_name: str, task_id: Optional[str] = None, max_rules: int = 5) -> List[str]:
        """
        Get active prompt rules for an agent in a specific task.
        
        Args:
            agent_name: Name of the agent
            task_id: Task ID (if None, returns empty list since rules are now task-specific)
            max_rules: Maximum number of rules to return
            
        Returns:
            List of rule texts (most recent first)
        """
        # Reload from disk to get latest state (for concurrent execution)
        self._active_rules = self._load_active_rules()

        if not task_id or task_id not in self._active_rules:
            return []

        rules = self._active_rules[task_id].get(agent_name, [])

        # Sort by timestamp (most recent first) and take top max_rules
        sorted_rules = sorted(rules, key=lambda x: x.get("timestamp", ""), reverse=True)
        return [r["update_text"] for r in sorted_rules[:max_rules]]

    def get_all_history(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get all historical updates for an agent."""
        history_file = os.path.join(self.updates_dir, f"{agent_name}.jsonl")

        if not os.path.exists(history_file):
            return []

        history = []
        try:
            with open(history_file, encoding='utf-8') as f:
                for line in f:
                    try:
                        history.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[GuidelineHistory] Error reading history: {e}")

        return history

    def clear_active_rules(self, agent_name: str):
        """Clear all active rules for an agent (useful for testing/reset)."""
        if agent_name in self._active_rules:
            self._active_rules[agent_name] = []
            self._save_active_rules()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about prompt updates."""
        stats = {
            "total_agents": len(self._active_rules),
            "agents": {}
        }

        for agent_name in self._active_rules:
            history = self.get_all_history(agent_name)
            stats["agents"][agent_name] = {
                "active_rules_count": len(self._active_rules[agent_name]),
                "total_updates": len(history),
            }

        return stats
