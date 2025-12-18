"""
Prompt templates for SCOPE components.

This module centralizes all LLM prompts used in SCOPE for:
- Easy maintenance and iteration
- Consistent formatting
- User customization (can be imported and modified)

Each prompt uses Python format string syntax with named placeholders.
"""

# =============================================================================
# GUIDELINE SYNTHESIS PROMPTS (synthesizer.py)
# =============================================================================

ERROR_REFLECTION_PROMPT = """You are a prompt engineering expert analyzing agent execution errors.

Your task: Generate a SHORT, TARGETED system prompt addition (1-3 lines) that will help prevent this error in the future.

Context:
- Agent Name: {agent_name}
- Agent Role: {agent_role}
- Task: {task}
- Error Type: {error_type}
- Error Message: {error_message}

Previous actions taken:
{last_step_summary}

Current system prompt (for reference, to avoid duplication):
{current_system_prompt}

Already applied rules (DO NOT duplicate these):
{applied_rules}

Guidelines:
1. Be SPECIFIC and ACTIONABLE - target the exact error cause
2. Be BRIEF - max 1-3 lines
3. Use imperative language ("Always...", "Never...", "When X, do Y...")
4. Don't repeat what's already in the current system prompt
5. Focus on formatting, structure, or procedure constraints

Output ONLY valid JSON with this exact format:
{{
  "update_text": "The actual prompt addition text here",
  "rationale": "Brief 1-sentence why this helps",
  "confidence": "low|medium|high"
}}"""


# Efficiency Mode: Lightweight, faster analysis
QUALITY_REFLECTION_PROMPT_EFFICIENCY = """You are a prompt engineering expert analyzing agent execution quality.

Your task: Analyze this successful step and determine if there are inefficiencies or areas for improvement. If found, generate a SHORT, TARGETED system prompt addition (1-3 lines).

Context:
- Agent Name: {agent_name}
- Agent Role: {agent_role}
- Task: {task}

Step details:
{last_step_summary}

Current system prompt (for reference):
{current_system_prompt}

Already applied rules (DO NOT duplicate these):
{applied_rules}

Analyze for:
1. **Inefficient tool usage**: Multiple calls when one would suffice, redundant operations
2. **Verbose outputs**: Unnecessarily long reasoning or outputs
3. **Missing best practices**: Not following domain-specific best practices
4. **Suboptimal approaches**: Using less efficient methods when better ones exist

Guidelines:
- Only suggest an update if there's a CLEAR, ACTIONABLE improvement
- Be SPECIFIC about what to improve
- Be BRIEF - max 1-3 lines
- Use imperative language ("Always...", "Prefer...", "When X, use Y instead of Z...")
- Don't repeat what's already in the current system prompt

Output ONLY valid JSON with this exact format:
{{
  "update_text": "The actual prompt addition text here (or empty string if no improvement needed)",
  "rationale": "Brief 1-sentence why this helps (or 'No improvement needed')",
  "confidence": "low|medium|high"
}}"""


# Thoroughness Mode: Comprehensive analysis with prioritized dimensions and domain learning
QUALITY_REFLECTION_PROMPT_THOROUGHNESS = """You are a prompt engineering expert analyzing agent execution quality.

Your task: Analyze this successful step and determine if there are inefficiencies or areas for improvement. If found, generate a SHORT, TARGETED system prompt addition (1-3 lines).

Context:
- Agent Name: {agent_name}
- Agent Role: {agent_role}
- Task: {task}

Step details:
{last_step_summary}

Current system prompt (for reference):
{current_system_prompt}

Already applied rules (DO NOT duplicate these):
{applied_rules}

Analyze for improvements in these dimensions:

1. **Correctness & Logic**:
   - Are there incorrect assumptions or flawed reasoning?
   - Are edge cases or special conditions being missed?
   - Is the approach fundamentally sound for this problem type?
   - Are validation/sanity checks missing?
   Examples: "Check if 1 ∈ S before computing", "Verify units match", "Confirm UTF-8 encoding"

2. **Domain-Specific Strategies**:
   - What terminology variants, synonyms, or phrasings should be searched?
     Examples: "ASEP/TASEP/exclusion process", "Émile/Emile", "Cafi/Caffi/Cafí"
   - Which authoritative sources, databases, or experts are most reliable?
     Examples: "arXiv for math", "uboat.net for naval", "cite Mumford/Milne"
   - What heuristics or shortcuts apply in this domain?
     Examples: "Check order-of-magnitude for physics", "Use closed-form when n<10"
   - What validation methods are domain-appropriate?
     Examples: "Verify stereochemistry", "Check orbital symmetry", "Confirm timestamp format"
   - Are there problem-type recognition patterns?
     Examples: "Sparse results → try variants"

3. **Strategic Planning & Approach**:
   - Is there a better problem decomposition or sequencing?
   - Should simpler methods be tried first? (trivial cases, closed forms, local computation)
   - Are batch operations being used where possible?
   - Is there missing lookahead or early termination logic?
   Examples: "Handle trivial case locally first", "Batch preprocessing before analysis"

4. **Tool Usage Efficiency**:
   - Multiple tool calls when one comprehensive call would suffice?
   - Missing consolidation opportunities?
   - Redundant fetches of the same information?
   - Could local computation replace a tool call?
   Examples: "Combine search + extraction", "Cache plan state", "Count locally"

5. **Information Preservation**:
   - Is critical context being lost between steps?
   - Are sources/citations/evidence being properly tracked?
   - Are intermediate results preserved for verification?
   Examples: "Attach artifact_refs", "Log clicked URLs", "Preserve evidence chain"

6. **Robustness & Error Recovery**:
   - Are there missing fallback strategies?
   - Is retry logic appropriate?
   - Are error conditions anticipated?
   Examples: "If search fails, try broader query", "Rate limit → exponential backoff"

7. **Output Quality**:
   - Is reasoning unnecessarily verbose?
   - Are outputs well-structured and parseable?
   - Is the answer format user-friendly?
   Examples: "Limit to ≤200 chars", "Return structured JSON", "Include confidence score"

Guidelines:
- PRIORITIZE correctness over efficiency
- Only suggest if there's a CLEAR, ACTIONABLE, GENERALIZABLE improvement
- For domain rules: Include SPECIFIC terms/sources/values, not placeholders
  Good: "Search 'Bean model' OR 'critical state' OR 'Brandt' on arXiv"
  Bad: "Use relevant search terms on appropriate databases"
- Be BRIEF - max 1-3 lines
- Use imperative language ("Always...", "Prefer...", "When X, do Y...")
- Don't repeat what's already in the current system prompt
- Look for PATTERNS that generalize beyond this single instance

Output ONLY valid JSON with this exact format:
{{
  "update_text": "The actual prompt addition text here (or empty string if no improvement needed)",
  "rationale": "Brief 1-sentence why this helps (or 'No improvement needed')",
  "confidence": "low|medium|high"
}}"""


SELECTOR_PROMPT = """You are a prompt engineering expert evaluating multiple candidate prompt updates.

Your task: Select the BEST candidate update from the options below.

Context:
- Agent Name: {agent_name}
- Agent Role: {agent_role}
- Task: {task}
- Issue Type: {issue_type}

Issue Details:
{issue_details}

Current system prompt (for reference):
{current_system_prompt}

Candidate Updates:
{candidates}

Evaluation Criteria (in priority order):
1. **Specificity & Relevance**: Most directly addresses the actual issue/error
2. **Actionability**: Clear, implementable instructions that the agent can follow
3. **Generalizability**: Useful beyond just this instance, but not too vague
4. **Brevity**: Concise and clear without unnecessary words
5. **Non-duplication**: Doesn't repeat what's already in the system prompt

Select the candidate that best balances these criteria for the current situation.

Output ONLY valid JSON with this exact format:
{{
  "selected_index": 0,
  "rationale": "Brief 1-2 sentence explanation of why this candidate is best"
}}"""


# =============================================================================
# GUIDELINE CLASSIFICATION PROMPT (optimizer.py)
# =============================================================================

CLASSIFICATION_PROMPT = """You are a rule classifier. Analyze the proposed update and determine:

1. **Is it a DUPLICATE/REDUNDANT?** Check if it's already covered by existing strategic or tactical rules.
2. **What is its SCOPE?** 
   - STRATEGIC: General best practice that applies across different tasks (e.g., "Always validate inputs", "Use batch operations when possible")
   - TACTICAL: Task-specific constraint for current task only (e.g., "This dataset has missing values in column X", "API rate limit is 100/min")
3. **Refined CONFIDENCE**: Assess confidence (0.0-1.0) based on how actionable and useful this rule is.
4. **DOMAIN**: If strategic, you MUST categorize it into ONE of the following allowed domains: {allowed_domains}

=== PROPOSED UPDATE ===
Update: {update_text}
Rationale: {rationale}
Initial Confidence: {initial_confidence:.2f}

{all_rules_context}

=== YOUR ANALYSIS ===
Respond in JSON format:
{{
    "is_duplicate": true/false,
    "scope": "strategic" or "tactical",
    "confidence": 0.0-1.0,
    "domain": "domain_name" (only if scope is strategic, otherwise ""),
    "reason": "Brief explanation of your classification"
}}

Think step by step:
1. Check if the proposed update is already covered by existing rules (exact match or semantic similarity)
2. Determine if it's a general best practice (strategic) or task-specific (tactical)
3. Assess the confidence based on clarity, actionability, and usefulness
4. If strategic, assign appropriate domain

JSON Response:"""


# =============================================================================
# MEMORY OPTIMIZATION PROMPTS (memory_optimizer.py)
# =============================================================================

RULE_ANALYSIS_PROMPT = """You are a rule optimization analyzer. Analyze these {num_rules} rules and identify optimization opportunities.

{rules_text}

Your task is to identify three types of optimization opportunities:

1. **CONFLICTS**: Pairs of rules that give contradictory guidance.
   - Example: "Minimize tool calls" vs "Call verification after each step"
   - PRIORITY: Highest - conflicts must be resolved first

2. **SUBSUMPTION**: Pairs where a specific rule is entirely covered by a more general rule.
   - Example: If "Check all file paths" exists, then "Check image file paths" is redundant
   - Format: [general_rule_index, specific_rule_index]
   - PRIORITY: High - clear redundancy should be removed

3. **CONSOLIDATION**: Groups of rules that express similar concepts and can be merged into a single, more comprehensive rule.
   - Example: "Validate file paths" + "Check file existence" → can merge
   - PRIORITY: Medium - merge rules that address the same concern from different angles

**IMPORTANT PRIORITY RULES** (to avoid overlaps):
1. If two rules CONFLICT, do NOT also mark them for consolidation or subsumption
2. If one rule SUBSUMES another, do NOT also mark them for consolidation
3. A rule pair should appear in AT MOST ONE category
4. When in doubt between categories, prefer: Conflicts > Subsumption > Consolidation

Guidelines:
- Only flag conflicts if there's a REAL contradiction (not just different aspects)
- For subsumption, the specific rule must be COMPLETELY covered by the general one
- Only group rules for consolidation if they truly address the same concern AND don't conflict
- Return empty arrays if no opportunities found
- Be conservative - only suggest optimizations you're confident about

You MUST respond with ONLY valid JSON in this exact format (no extra text before or after):
{{
  "consolidation": [[idx1, idx2], [idx3, idx4]],
  "subsumption": [[general_idx, specific_idx]],
  "conflicts": [[idx1, idx2]]
}}

Example valid response:
{{
  "consolidation": [[0, 1, 2], [5, 6]],
  "subsumption": [[3, 7]],
  "conflicts": [[4, 8]]
}}

If no opportunities found, return:
{{
  "consolidation": [],
  "subsumption": [],
  "conflicts": []
}}

JSON Response:"""


RULE_MERGE_PROMPT = """You are merging similar rules into one comprehensive rule.

Original rules to merge:
{rules_text}

Create a single, comprehensive rule that:
1. Captures all the key guidance from the original rules
2. Is clear and actionable
3. Eliminates redundancy
4. Maintains the original intent

Also provide a brief rationale explaining what the merged rule accomplishes.

Return JSON:
{{
  "rule": "The merged rule text",
  "rationale": "Brief explanation of what this rule accomplishes"
}}

JSON Response:"""


SUBSUMPTION_VERIFY_PROMPT = """Verify if the general rule subsumes the specific rule.

General Rule: {general_rule}
Specific Rule: {specific_rule}

A rule is subsumed if:
- Following the general rule AUTOMATICALLY means you follow the specific rule
- The specific rule adds NO additional constraints or guidance
- The specific rule is merely a special case or example of the general rule

Does the general rule completely subsume the specific rule? Answer with JSON:
{{
  "subsumed": true or false,
  "reasoning": "Brief explanation"
}}

JSON Response:"""


CONFLICT_RESOLVE_PROMPT = """You are resolving a conflict between two rules.

Rule {idx1}:
  Text: {rule1_text}
  Rationale: {rule1_rationale}

Rule {idx2}:
  Text: {rule2_text}
  Rationale: {rule2_rationale}

These rules appear to conflict. Your task is to:
1. Verify the conflict is real
2. Synthesize a single rule that reconciles both OR pick the better rule with justification

Return JSON:
{{
  "rule": "The resolved rule text",
  "rationale": "Explanation of how this resolves the conflict"
}}

JSON Response:"""

