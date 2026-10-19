"""The self-improving agent, built as a LangGraph state machine.

    evaluate ──▶ (improved? / target met? / budget left?) ──▶ reflect ──▶ optimize ──┐
        ▲                                                                            │
        └────────────────────────────────────────────────────────────────────────┘
                                    │
                                    └──▶ END

Each loop: score the current policy on train+test, reflect on failures, rewrite the policy.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from selfforge.models import Example, Policy, RoundMetrics, RunResult
from selfforge.optimizer import reflect, synthesize


def _accuracy(model, policy: Policy, data: list[Example]) -> float:
    if not data:
        return 0.0
    ok = 0
    for ex in data:
        pred = model.predict(ex.text, policy)
        if pred is not None and abs(pred - ex.expected) < 1e-6:
            ok += 1
    return ok / len(data)


def _failures(model, policy: Policy, data: list[Example]) -> list[Example]:
    out = []
    for ex in data:
        pred = model.predict(ex.text, policy)
        if pred is None or abs(pred - ex.expected) >= 1e-6:
            out.append(ex)
    return out


class ForgeState(TypedDict):
    policy: Policy
    round: int
    max_rounds: int
    target: float
    train: list[Example]
    test: list[Example]
    history: Annotated[list[RoundMetrics], lambda a, b: a + b]
    model: Any


def _evaluate(state: ForgeState) -> dict:
    model, policy = state["model"], state["policy"]
    train_acc = _accuracy(model, policy, state["train"])
    test_acc = _accuracy(model, policy, state["test"])
    metric = RoundMetrics(
        round=state["round"],
        train_accuracy=round(train_acc, 4),
        test_accuracy=round(test_acc, 4),
        n_rules=len(policy.enabled_rules),
    )
    return {"history": [metric]}


def _should_continue(state: ForgeState) -> str:
    last = state["history"][-1]
    if last.train_accuracy >= state["target"]:
        return "stop"
    if state["round"] >= state["max_rounds"]:
        return "stop"
    return "improve"


def _reflect_and_optimize(state: ForgeState) -> dict:
    model, policy = state["model"], state["policy"]
    fails = _failures(model, policy, state["train"])
    reflections = reflect(fails, policy, model)
    new_policy, learned = synthesize(policy, reflections, fails)

    last = state["history"][-1]
    last.reflections = reflections[:3]
    last.newly_learned = learned

    if not learned:  # nothing left to learn → stop next check
        return {"policy": new_policy, "round": state["max_rounds"]}
    return {"policy": new_policy, "round": state["round"] + 1}


def build_graph():
    g = StateGraph(ForgeState)
    g.add_node("evaluate", _evaluate)
    g.add_node("optimize", _reflect_and_optimize)
    g.add_edge(START, "evaluate")
    g.add_conditional_edges("evaluate", _should_continue, {"improve": "optimize", "stop": END})
    g.add_edge("optimize", "evaluate")
    return g.compile()


def run_self_improvement(
    model,
    train: list[Example],
    test: list[Example],
    *,
    max_rounds: int = 8,
    target: float = 0.98,
    provider: str = "offline",
) -> RunResult:
    graph = build_graph()
    init: ForgeState = {
        "policy": Policy(),
        "round": 0,
        "max_rounds": max_rounds,
        "target": target,
        "train": train,
        "test": test,
        "history": [],
        "model": model,
    }
    final = graph.invoke(init, config={"recursion_limit": 100})
    return RunResult(
        provider=provider,
        model=getattr(model, "name", "unknown"),
        target_accuracy=target,
        rounds=final["history"],
        final_policy=final["policy"],
    )
