"""Core data models for the self-improving agent."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Example(BaseModel):
    """A single task instance: an input phrase and its ground-truth value."""

    text: str
    expected: float
    tags: list[str] = Field(default_factory=list)  # which capability it exercises


class Policy(BaseModel):
    """The agent's current, improvable policy.

    This is what the agent rewrites each round. With a real LLM it renders into a
    system prompt (instructions + few-shot examples). The offline engine also reads
    `enabled_rules` to decide which parsing capabilities it currently has.
    """

    instructions: list[str] = Field(default_factory=list)
    few_shots: list[Example] = Field(default_factory=list)
    enabled_rules: list[str] = Field(default_factory=lambda: ["integers"])

    def render(self) -> str:
        """Render this policy into a system prompt for a real LLM."""
        lines = [
            "You convert a quantity phrase into a single numeric value.",
            "Reply with ONLY the number (digits), no words, no units.",
        ]
        if self.instructions:
            lines.append("\nRules you have learned:")
            lines += [f"- {i}" for i in self.instructions]
        if self.few_shots:
            lines.append("\nWorked examples:")
            lines += [f'  "{e.text}" -> {_fmt(e.expected)}' for e in self.few_shots]
        return "\n".join(lines)

    def copy_deep(self) -> Policy:
        return Policy(
            instructions=list(self.instructions),
            few_shots=list(self.few_shots),
            enabled_rules=list(self.enabled_rules),
        )


class Reflection(BaseModel):
    """A diagnosis of why an example failed and what capability would fix it."""

    text: str
    diagnosis: str
    proposed_rule: str
    guidance: str


class RoundMetrics(BaseModel):
    round: int
    train_accuracy: float
    test_accuracy: float
    n_rules: int
    newly_learned: list[str] = Field(default_factory=list)
    reflections: list[Reflection] = Field(default_factory=list)


class RunResult(BaseModel):
    provider: str
    model: str
    target_accuracy: float
    rounds: list[RoundMetrics]
    final_policy: Policy

    @property
    def start_test_accuracy(self) -> float:
        return self.rounds[0].test_accuracy if self.rounds else 0.0

    @property
    def final_test_accuracy(self) -> float:
        return self.rounds[-1].test_accuracy if self.rounds else 0.0

    @property
    def improvement(self) -> float:
        return self.final_test_accuracy - self.start_test_accuracy


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)
