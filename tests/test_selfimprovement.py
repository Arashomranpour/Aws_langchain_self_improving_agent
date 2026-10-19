"""Tests for the self-improvement loop (offline, deterministic — no API key)."""

from selfforge.agent import run_self_improvement
from selfforge.core import improve
from selfforge.dataset import ALL_RULES, load_dataset
from selfforge.engine import OfflineEngine
from selfforge.llm import get_model
from selfforge.models import Policy


def test_dataset_split_is_stratified():
    train, test = load_dataset()
    assert train and test
    train_tags = {t for e in train for t in e.tags}
    test_tags = {t for e in test for t in e.tags}
    assert train_tags == test_tags == set(ALL_RULES)


def test_base_policy_only_solves_integers():
    engine = OfflineEngine()
    base = Policy()  # only "integers"
    assert engine.predict("12", base) == 12
    assert engine.predict("two dozen", base) is None
    assert engine.predict("3.5", base) is None


def test_full_policy_solves_everything():
    engine = OfflineEngine()
    full = Policy(enabled_rules=list(ALL_RULES))
    train, test = load_dataset()
    for ex in train + test:
        pred = engine.predict(ex.text, full)
        assert pred is not None and abs(pred - ex.expected) < 1e-6, ex.text


def test_agent_improves_and_converges():
    result = improve(provider="offline", max_rounds=10, target=0.98)
    assert result.start_test_accuracy < 0.3          # starts weak
    assert result.final_test_accuracy >= 0.98        # learns the task
    assert result.improvement > 0.6                  # large, real gain


def test_accuracy_is_monotonic_non_decreasing():
    result = improve(provider="offline", max_rounds=10)
    import itertools

    accs = [r.test_accuracy for r in result.rounds]
    assert all(b >= a - 1e-9 for a, b in itertools.pairwise(accs)), accs


def test_policy_self_authored_each_rule_once():
    result = improve(provider="offline")
    rules = result.final_policy.enabled_rules
    assert len(rules) == len(set(rules))             # no duplicates
    assert "integers" in rules


def test_offline_provider_needs_no_key():
    model = get_model("offline")
    assert model.predict("a dozen", Policy(enabled_rules=["dozen"])) == 12


def test_langgraph_graph_builds_and_runs():
    train, test = load_dataset()
    res = run_self_improvement(OfflineEngine(), train, test, max_rounds=3, provider="offline")
    assert res.rounds and res.rounds[0].round == 0
