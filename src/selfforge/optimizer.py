"""Reflection + policy synthesis — the self-improvement step.

After an evaluation round, the optimizer looks at the failures, diagnoses the missing
capability, and rewrites the policy: it enables the capability, appends a human-readable
guidance line, and adds a worked example as a few-shot. This is the Reflexion / self-refine
pattern: the agent edits its own instructions from its own mistakes.
"""

from __future__ import annotations

from selfforge.dataset import ALL_RULES
from selfforge.engine import OfflineEngine
from selfforge.models import Example, Policy, Reflection

# Human-readable guidance attached when a capability is learned.
GUIDANCE: dict[str, str] = {
    "decimals": "Numbers can be decimals (e.g. '3.5' -> 3.5). Keep the fractional part.",
    "thousands": "Commas are thousands separators: '1,200' -> 1200.",
    "suffix": "'k' means thousand and 'm' means million: '3.5k' -> 3500, '1m' -> 1000000.",
    "words": "Spelled-out numbers map to digits: 'twelve' -> 12.",
    "dozen": "'dozen' = 12; multiply by the leading number: 'two dozen' -> 24, 'half a dozen' -> 6.",
    "couple_few": "'a couple' = 2; 'a few' and 'several' = 3.",
    "fractions": "'half' = 0.5, 'a quarter' = 0.25, 'three quarters' = 0.75, 'a third' ~ 0.333.",
}


def _probe_rule(example: Example, policy: Policy, rule: str) -> bool:
    """Would enabling `rule` make the offline engine solve this example?"""
    trial = policy.copy_deep()
    if rule not in trial.enabled_rules:
        trial.enabled_rules.append(rule)
    pred = OfflineEngine().predict(example.text, trial)
    return pred is not None and abs(pred - example.expected) < 1e-6


def reflect(failures: list[Example], policy: Policy, model=None) -> list[Reflection]:
    """Diagnose failures and propose capabilities to learn, most impactful first."""
    candidates = [r for r in ALL_RULES if r not in policy.enabled_rules]
    # Count how many current failures each candidate rule would fix.
    impact: dict[str, list[Example]] = {}
    for rule in candidates:
        fixed = [f for f in failures if _probe_rule(f, policy, rule)]
        if fixed:
            impact[rule] = fixed

    reflections: list[Reflection] = []
    for rule, fixed in sorted(impact.items(), key=lambda kv: len(kv[1]), reverse=True):
        sample = fixed[0]
        reflections.append(
            Reflection(
                text=sample.text,
                diagnosis=f"Missed {len(fixed)} input(s) like \"{sample.text}\" "
                f"(expected {_fmt(sample.expected)}) — no capability for '{rule}'.",
                proposed_rule=rule,
                guidance=GUIDANCE.get(rule, f"Handle '{rule}' inputs."),
            )
        )
    return reflections


def synthesize(policy: Policy, reflections: list[Reflection], failures: list[Example],
               max_new: int = 1) -> tuple[Policy, list[str]]:
    """Return an improved policy and the list of newly learned rule tags.

    Learns the single highest-impact capability per round so the curve is legible.
    """
    new_policy = policy.copy_deep()
    learned: list[str] = []
    for ref in reflections[:max_new]:
        if ref.proposed_rule in new_policy.enabled_rules:
            continue
        new_policy.enabled_rules.append(ref.proposed_rule)
        if ref.guidance not in new_policy.instructions:
            new_policy.instructions.append(ref.guidance)
        # add a worked example as a few-shot (grounds a real LLM)
        shot = next((f for f in failures if ref.proposed_rule in f.tags), None)
        if shot and all(s.text != shot.text for s in new_policy.few_shots):
            new_policy.few_shots.append(shot)
        learned.append(ref.proposed_rule)
    return new_policy, learned


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.3f}".rstrip("0").rstrip(".")
