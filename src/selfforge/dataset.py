"""The benchmark task: normalize natural-language quantity phrases to numbers.

Each example is tagged with the capability it exercises. The agent starts knowing only
how to read plain integers and must *learn* the rest from its own failures, which makes
the improvement curve genuine and reproducible rather than cosmetic.
"""

from __future__ import annotations

import random

from selfforge.models import Example

# text, expected, capability tag
_RAW: list[tuple[str, float, str]] = [
    # integers (known from the start)
    ("12", 12, "integers"), ("5", 5, "integers"), ("0", 0, "integers"),
    ("99", 99, "integers"), ("7", 7, "integers"), ("250", 250, "integers"),
    # decimals
    ("3.5", 3.5, "decimals"), ("0.25", 0.25, "decimals"), ("10.0", 10, "decimals"),
    ("2.75", 2.75, "decimals"), ("1.5", 1.5, "decimals"), ("8.2", 8.2, "decimals"),
    # thousands separators
    ("1,200", 1200, "thousands"), ("12,345", 12345, "thousands"),
    ("3,000", 3000, "thousands"), ("45,600", 45600, "thousands"),
    ("2,500", 2500, "thousands"), ("9,999", 9999, "thousands"),
    # k / m suffixes
    ("2k", 2000, "suffix"), ("3.5k", 3500, "suffix"), ("10k", 10000, "suffix"),
    ("1m", 1_000_000, "suffix"), ("1.2k", 1200, "suffix"), ("500k", 500_000, "suffix"),
    # number words
    ("twelve", 12, "words"), ("three", 3, "words"), ("twenty", 20, "words"),
    ("seven", 7, "words"), ("fifteen", 15, "words"), ("one", 1, "words"),
    # dozen
    ("a dozen", 12, "dozen"), ("two dozen", 24, "dozen"), ("dozen", 12, "dozen"),
    ("three dozen", 36, "dozen"), ("half a dozen", 6, "dozen"), ("a dozen eggs", 12, "dozen"),
    # couple / few / several
    ("a couple", 2, "couple_few"), ("a few", 3, "couple_few"), ("several", 3, "couple_few"),
    ("couple of apples", 2, "couple_few"), ("a few people", 3, "couple_few"),
    ("several books", 3, "couple_few"),
    # halves and fractions
    ("half", 0.5, "fractions"), ("a quarter", 0.25, "fractions"),
    ("three quarters", 0.75, "fractions"), ("half a kilo", 0.5, "fractions"),
    ("a third", 1 / 3, "fractions"), ("two thirds", 2 / 3, "fractions"),
]


def load_dataset(seed: int = 7, test_frac: float = 0.4) -> tuple[list[Example], list[Example]]:
    """Return (train, test) with a stratified split so every capability appears in both."""
    rng = random.Random(seed)
    by_tag: dict[str, list[Example]] = {}
    for text, expected, tag in _RAW:
        by_tag.setdefault(tag, []).append(Example(text=text, expected=float(expected), tags=[tag]))

    train: list[Example] = []
    test: list[Example] = []
    for tag, items in by_tag.items():
        rng.shuffle(items)
        n_test = max(1, round(len(items) * test_frac))
        test += items[:n_test]
        train += items[n_test:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


ALL_RULES = ["integers", "decimals", "thousands", "suffix", "words", "dozen", "couple_few", "fractions"]
