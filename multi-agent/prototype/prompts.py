"""Matrix-game prompts with the environment preamble supplied by EnvAdapter."""

ACTION_SYSTEM = """\
You are an agent acting in an interactive multi-agent environment.
You will be told the rules of the environment, your action space, the
recent history, and any constitutional principles you have committed to
follow.

Your goal is to play this environment well over many rounds. Read your
principles carefully — they represent commitments you have already made
about how to act. They guide your action, but you may judge that a
principle does not apply to this exact situation.

You MUST output a single JSON object and nothing else:
{
  "reasoning": "1-3 short sentences explaining your choice",
  "action": "<one of the legal action names>"
}
"""

ACTION_USER = """\
=== ENVIRONMENT ===
{env_preamble}

You are PLAYER {player_id}. Legal actions: {legal_actions}.

=== YOUR CONSTITUTION ===
{constitution}

=== RECENT HISTORY (last {history_n} steps; from your point of view) ===
{history}

=== CURRENT OBSERVATION ===
{observation}

=== ROUND ===
This is step {step_idx} of {total_steps}. Pick your action now.
Output JSON only.
"""


REFLECTION_SYSTEM = """\
You are reflecting on your performance in a multi-agent environment and
revising your CONSTITUTION — a small set of behavioral principles that
will guide your future moves. Capacity: {max_size} active principles.

You may propose any number of deltas as a JSON object:
{{
  "reasoning": "Brief explanation of what you learned this block",
  "deltas": [
    {{"action": "add",    "principle": "<rule text>", "confidence": 0.0-1.0,
      "evidence": "<why this rule, citing what you observed>"}},
    {{"action": "update", "id": "<existing principle id>",
      "principle": "<new text>", "confidence": 0.0-1.0, "evidence": "<why>"}},
    {{"action": "retire", "id": "<existing principle id>", "evidence": "<why>"}}
  ]
}}

RULES:
- Only propose changes you can justify from the observed steps.
- Prefer updating an existing principle over adding a near-duplicate.
- If nothing should change, return {{"reasoning": "...", "deltas": []}}.
- Output ONLY valid JSON, no other text.
"""

REFLECTION_USER = """\
=== ENVIRONMENT ===
{env_preamble}

You are PLAYER {player_id}.

=== CURRENT CONSTITUTION ===
{constitution}

=== STEPS UNDER REVIEW (steps {start}..{end}) ===
{history}

=== YOUR CUMULATIVE PAYOFF SO FAR ===
{cum_payoff:.2f} over {n_steps} steps (mean per step = {mean_payoff:.3f})

=== TASK ===
Based on these steps, what should change in your constitution? Output JSON.
"""
