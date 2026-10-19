"""The offline deterministic 'model'.

It parses a quantity phrase using ONLY the capabilities currently enabled in the policy.
This makes self-improvement real: when the optimizer enables a new rule (after reflecting
on failures), the engine genuinely starts solving a new class of inputs. No API key needed,
fully reproducible — ideal for tests, CI and the visual demo.
"""

from __future__ import annotations

import re

from selfforge.models import Policy

_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_FRACTIONS = {"half": 0.5, "quarter": 0.25, "third": 1 / 3}


class OfflineEngine:
    """A rule-based solver whose behaviour is gated by policy.enabled_rules."""

    name = "offline-deterministic"

    def predict(self, text: str, policy: Policy) -> float | None:
        rules = set(policy.enabled_rules)
        t = text.strip().lower()

        if "dozen" in rules and "dozen" in t:
            if "half" in t:
                return 6.0
            m = re.search(r"(\w+)\s+dozen", t)
            mult = 1
            if m:
                w = m.group(1)
                if w.isdigit():
                    mult = int(w)
                elif w in _WORDS:
                    mult = _WORDS[w]
                elif w in ("a", "one"):
                    mult = 1
            return float(12 * mult)

        if "fractions" in rules:
            if "three quarter" in t:
                return 0.75
            if "two third" in t:
                return 2 / 3
            for w, v in _FRACTIONS.items():
                if re.search(rf"\b(a |one )?{w}\b", t):
                    return float(v)

        if "couple_few" in rules:
            if "couple" in t:
                return 2.0
            if "few" in t or "several" in t:
                return 3.0

        if "suffix" in rules:
            m = re.fullmatch(r"([\d.]+)\s*([km])", t)
            if m:
                base = float(m.group(1))
                return base * (1000 if m.group(2) == "k" else 1_000_000)

        if "thousands" in rules:
            m = re.fullmatch(r"[\d,]+", t)
            if m and "," in t:
                return float(t.replace(",", ""))

        if "words" in rules:
            for w, v in _WORDS.items():
                if re.search(rf"\b{w}\b", t):
                    return float(v)

        if "decimals" in rules:
            m = re.fullmatch(r"\d+\.\d+", t)
            if m:
                return float(t)

        # base capability: plain integers
        m = re.fullmatch(r"\d+", t)
        if m:
            return float(t)

        return None
