"""LLM providers.

Two interchangeable 'models' plug into the same self-improvement loop:

* OfflineEngine (engine.py) — deterministic, no key, used for tests/CI/demo.
* GroqModel          — a real LLM served by Groq (llama-3.x / mixtral etc.).

Both expose .predict(text, policy) -> float | None.
"""

from __future__ import annotations

import os
import re

from selfforge.engine import OfflineEngine
from selfforge.models import Policy

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def _parse_number(raw: str) -> float | None:
    if raw is None:
        return None
    m = re.search(r"-?\d[\d,]*\.?\d*", raw.replace(" ", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


class GroqModel:
    """Calls a Groq-hosted model through langchain-groq.

    Requires GROQ_API_KEY in the environment. The policy is rendered into the system
    prompt, so the agent's self-written instructions and few-shots directly steer it.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        from langchain_groq import ChatGroq  # imported lazily so offline mode needs no key

        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set. Export it or use provider='offline'.")
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self._llm = ChatGroq(model=self.model, temperature=temperature)
        self.name = f"groq:{self.model}"

    def predict(self, text: str, policy: Policy) -> float | None:
        from langchain_core.messages import HumanMessage, SystemMessage

        resp = self._llm.invoke(
            [SystemMessage(content=policy.render()), HumanMessage(content=text)]
        )
        return _parse_number(getattr(resp, "content", str(resp)))

    def reflect(self, failures, policy: Policy) -> str:
        """Ask the model to explain its mistakes and propose one new rule (natural language)."""
        from langchain_core.messages import HumanMessage, SystemMessage

        shown = "\n".join(f'- "{e.text}" (correct answer: {e.expected})' for e in failures[:8])
        sys = (
            "You are improving your own instructions for a quantity-parsing task. "
            "Given examples you got wrong, state ONE concise rule (max 20 words) that would "
            "fix the most cases. Reply with just the rule."
        )
        resp = self._llm.invoke(
            [SystemMessage(content=sys), HumanMessage(content=f"I got these wrong:\n{shown}")]
        )
        return getattr(resp, "content", "").strip()


def get_model(provider: str = "offline", model: str | None = None):
    provider = (provider or "offline").lower()
    if provider == "offline":
        return OfflineEngine()
    if provider == "groq":
        return GroqModel(model=model)
    raise ValueError(f"Unknown provider '{provider}'. Use 'offline' or 'groq'.")
