"""High-level entry point shared by the CLI, the web API and the AgentCore runtime."""

from __future__ import annotations

from selfforge.agent import run_self_improvement
from selfforge.dataset import load_dataset
from selfforge.llm import get_model
from selfforge.models import RunResult


def improve(
    provider: str = "offline",
    model: str | None = None,
    max_rounds: int = 8,
    target: float = 0.98,
    seed: int = 7,
) -> RunResult:
    """Run the full self-improvement loop and return the per-round history + final policy."""
    train, test = load_dataset(seed=seed)
    llm = get_model(provider=provider, model=model)
    return run_self_improvement(
        llm, train, test, max_rounds=max_rounds, target=target, provider=provider
    )
