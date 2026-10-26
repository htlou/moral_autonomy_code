"""Build pre-reflection activation datasets from causal experiment traces.

The output is JSONL.  Each row contains the exact reflection messages when the
run recorded them, or a reconstruction from the constitution snapshot,
transition block, environment adapter, and canonical prompt template.

Labels are deliberately kept outside the prompt.  In particular,
``proposed_large_edit`` is an outcome label, not text shown to the model.
"""

from __future__ import annotations

import argparse
from collections import Counter
import difflib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from prototype.agent import format_history
from prototype.concordia_extended_causal_experiment import ENVIRONMENTS
from prototype.concordia_pub_coordination_experiment import (
    ConcordiaPubCoordinationEnv,
)
from prototype.prompts import REFLECTION_SYSTEM, REFLECTION_USER


RESULT_PATTERNS = (
    "concordia_pub_causal_seed*.json",
    "concordia_extended_causal_seed*.json",
    "clean_up_causal_seed*.json",
)


def iter_result_paths(roots: Sequence[Path]) -> Iterator[Path]:
    seen: set[Path] = set()
    for root in roots:
        if root.is_file():
            candidates: Iterable[Path] = (root,)
        else:
            candidates = (
                path
                for pattern in RESULT_PATTERNS
                for path in root.rglob(pattern)
            )
        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path


def _snapshot_summary(
    snapshots: Sequence[dict[str, Any]], before_round: int
) -> str:
    active = list(_snapshot_entries(snapshots, before_round).values())
    if not active:
        return "(no active principles)"
    active.sort(key=lambda entry: float(entry.get("confidence", 0.0)),
                reverse=True)
    return "\n".join(
        (
            f"[{entry.get('id', '?')} v{entry.get('version', 1)} "
            f"conf={float(entry.get('confidence', 0.0)):.2f}] "
            f"{entry.get('principle', '')}"
        )
        for entry in active
    )


def _snapshot_entries(
    snapshots: Sequence[dict[str, Any]], before_round: int
) -> dict[str, dict[str, Any]]:
    eligible = [
        snapshot
        for snapshot in snapshots
        if int(snapshot.get("round", -1)) <= before_round
    ]
    entries = max(
        eligible, key=lambda snapshot: int(snapshot["round"])
    ).get("entries", []) if eligible else []
    active = [entry for entry in entries if entry.get("active", True)]
    return {str(entry.get("id")): entry for entry in active}


def _environment_preamble(
    result: dict[str, Any],
    transition: dict[str, Any],
) -> str:
    settings = result["settings"]
    regime = transition["info"]["regime"]
    task = result.get("task")
    env_factory = (
        ENVIRONMENTS[task] if task in ENVIRONMENTS
        else ConcordiaPubCoordinationEnv
    )
    env = env_factory(
        regime=regime,
        rounds=int(settings["rounds_per_phase"]),
        n_background=int(settings["n_background"]),
        seed=int(result["seed"]) * 10_000,
    )
    return env.system_preamble(0)


def _reconstruct_messages(
    result: dict[str, Any],
    condition: dict[str, Any],
    reflection: dict[str, Any],
) -> list[dict[str, str]]:
    start, end = map(int, reflection["steps"])
    transitions = condition["transitions"]
    block_dicts = [
        transition
        for transition in transitions
        if start <= int(transition["step_idx"]) <= end
    ]
    if not block_dicts:
        raise ValueError(f"no transitions for reflection steps {start}..{end}")

    class Transition:
        def __init__(self, data: dict[str, Any]) -> None:
            self.step_idx = data["step_idx"]
            self.actions = data["actions"]
            self.action_names = data["action_names"]
            self.rewards = data["rewards"]
            self.info = data["info"]

    block = [Transition(data) for data in block_dicts]
    cumulative = sum(
        float(transition["rewards"][0])
        for transition in transitions
        if int(transition["step_idx"]) <= end
    )
    n_steps = sum(
        int(transition["step_idx"]) <= end for transition in transitions
    )
    constitution = _snapshot_summary(
        condition.get("constitution_snapshots", []),
        before_round=start,
    )
    sys_prompt = REFLECTION_SYSTEM.format(
        max_size=int(result["settings"]["max_constitution_size"])
    )
    user_prompt = REFLECTION_USER.format(
        env_preamble=_environment_preamble(result, block_dicts[0]),
        player_id=0,
        constitution=constitution,
        start=start,
        end=end,
        history=format_history(block, 0, len(block)),
        cum_payoff=cumulative,
        n_steps=n_steps,
        mean_payoff=(cumulative / n_steps) if n_steps else 0.0,
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _edit_statistics(
    reflection: dict[str, Any],
    large_edit_chars: int,
    prior_principles: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    parsed = reflection.get("parsed") or {}
    deltas = parsed.get("deltas", []) if isinstance(parsed, dict) else []
    deltas = [delta for delta in deltas if isinstance(delta, dict)]
    edit_chars = 0
    actions: Counter[str] = Counter()
    for delta in deltas:
        action = str(delta.get("action", "")).lower()
        actions[action] += 1
        principle = str(delta.get("principle", ""))
        existing = (prior_principles or {}).get(str(delta.get("id")))
        old_principle = str(existing.get("principle", "")) if existing else ""
        if action == "add":
            edit_chars += len(principle)
        elif action == "update":
            if old_principle:
                similarity = difflib.SequenceMatcher(
                    None, old_principle, principle
                ).ratio()
                edit_chars += round(
                    max(len(old_principle), len(principle))
                    * (1.0 - similarity)
                )
            else:
                edit_chars += len(principle)
        elif action == "retire":
            edit_chars += max(32, len(old_principle))
    large = edit_chars >= large_edit_chars or len(deltas) >= 2
    group = "none" if not deltas else ("large" if large else "small")
    return {
        "proposed_edit_count": len(deltas),
        "proposed_edit_chars": edit_chars,
        "proposed_edit_actions": dict(actions),
        "proposed_any_edit": bool(deltas),
        "proposed_large_edit": large,
        "edit_group": group,
        "applied_any_edit": bool(reflection.get("applied")),
    }


def _next_block_reward(
    transitions: Sequence[dict[str, Any]], end: int, width: int
) -> float | None:
    rewards = [
        float(transition["rewards"][0])
        for transition in transitions
        if end < int(transition["step_idx"]) <= end + width
    ]
    return sum(rewards) / len(rewards) if rewards else None


def build_events(
    paths: Iterable[Path],
    *,
    large_edit_chars: int = 120,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in paths:
        result = json.loads(path.read_text())
        settings = result.get("settings", {})
        for condition_name in ("revisable", "frozen", "forced_injection"):
            condition = result.get("conditions", {}).get(condition_name)
            if not condition:
                continue
            injections = {
                tuple(map(int, record["steps"])): record
                for record in condition.get("injection_log", [])
            }
            for reflection_index, reflection in enumerate(
                condition.get("reflection_log", [])
            ):
                start, end = map(int, reflection["steps"])
                messages = reflection.get("messages")
                input_source = "recorded"
                if not messages:
                    try:
                        messages = _reconstruct_messages(
                            result, condition, reflection
                        )
                        input_source = "reconstructed"
                    except Exception as error:
                        messages = [{
                            "role": "user",
                            "content": (
                                "Constitutional reflection context could not "
                                f"be reconstructed: {error}. Raw trace: "
                                + json.dumps(
                                    reflection.get("parsed", {}),
                                    ensure_ascii=False,
                                )
                            ),
                        }]
                        input_source = "fallback"

                block_rewards = [
                    float(transition["rewards"][0])
                    for transition in condition["transitions"]
                    if start <= int(transition["step_idx"]) <= end
                ]
                injection = injections.get((start, end))
                event = {
                    "event_id": (
                        f"{result.get('experiment')}:"
                        f"{result.get('task', 'pub_coordination')}:"
                        f"{result['seed']}:"
                        f"{condition_name}:{start}-{end}"
                    ),
                    "paired_event_id": (
                        f"{result.get('experiment')}:"
                        f"{result.get('task', 'pub_coordination')}:"
                        f"{result['seed']}:"
                        f"{start}-{end}"
                    ),
                    "source_path": str(path.resolve()),
                    "experiment": result.get("experiment"),
                    "task": result.get("task", "pub_coordination"),
                    "seed": int(result["seed"]),
                    "condition": condition_name,
                    "persistent_update_authority": (
                        condition.get("constitution_update_authority")
                        or {
                            "revisable": "agent",
                            "frozen": "none",
                            "forced_injection": "external_reward_rule",
                        }[condition_name]
                    ),
                    "agent_proposal_applied": (
                        condition_name == "revisable"
                    ),
                    "reflection_index": reflection_index,
                    "steps": [start, end],
                    "phase": next(
                        transition["info"]["regime"]
                        for transition in condition["transitions"]
                        if int(transition["step_idx"]) == start
                    ),
                    "block_mean_reward": (
                        sum(block_rewards) / len(block_rewards)
                        if block_rewards else 0.0
                    ),
                    "next_block_mean_reward": _next_block_reward(
                        condition["transitions"],
                        end,
                        int(settings.get("reflect_every", 5)),
                    ),
                    "forced_update_triggered": bool(
                        injection and injection.get("triggered")
                    ),
                    "forced_delta": (
                        injection.get("forced_delta") if injection else None
                    ),
                    "messages": messages,
                    "input_source": input_source,
                    "raw_output": reflection.get("raw", ""),
                    "proposed_deltas": (
                        (reflection.get("parsed") or {}).get("deltas", [])
                        if isinstance(reflection.get("parsed") or {}, dict)
                        else []
                    ),
                }
                prior_principles = _snapshot_entries(
                    condition.get("constitution_snapshots", []),
                    before_round=start,
                )
                event.update(_edit_statistics(
                    reflection,
                    large_edit_chars,
                    prior_principles=prior_principles,
                ))
                events.append(event)
    return events


def write_jsonl(events: Sequence[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(event, ensure_ascii=False) + "\n"
            for event in events
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--large-edit-chars", type=int, default=120)
    args = parser.parse_args()
    paths = list(iter_result_paths(args.roots))
    events = build_events(
        paths, large_edit_chars=args.large_edit_chars
    )
    if not events:
        raise SystemExit("no reflection events found")
    write_jsonl(events, args.output)
    counts = Counter(
        (
            event["condition"],
            event["edit_group"],
            event["input_source"],
        )
        for event in events
    )
    summary = {
        "result_files": len(paths),
        "events": len(events),
        "large_edit_chars": args.large_edit_chars,
        "counts": [
            {
                "condition": condition,
                "edit_group": group,
                "input_source": source,
                "n": count,
            }
            for (condition, group, source), count in sorted(counts.items())
        ],
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
