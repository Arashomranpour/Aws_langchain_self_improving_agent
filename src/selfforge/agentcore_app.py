"""Amazon Bedrock AgentCore Runtime entrypoint.

AgentCore Runtime invokes a container that serves POST /invocations and GET /ping on
0.0.0.0:8080. The BedrockAgentCoreApp SDK provides that server; we just define the
entrypoint. Deploy with:

    agentcore configure -e src/selfforge/agentcore_app.py
    agentcore launch

Payload (JSON):
    {"action": "improve", "provider": "groq", "max_rounds": 8, "target": 0.98}
    {"action": "predict", "text": "two dozen"}          # uses a fully-trained policy
"""

from __future__ import annotations

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from selfforge.core import improve
from selfforge.dataset import ALL_RULES
from selfforge.llm import get_model
from selfforge.models import Policy

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict, context=None) -> dict:
    action = (payload or {}).get("action", "improve")
    provider = (payload or {}).get("provider", "offline")
    model = (payload or {}).get("model")

    if action == "predict":
        # Answer a single phrase using a fully-trained policy.
        llm = get_model(provider=provider, model=model)
        policy = Policy(enabled_rules=list(ALL_RULES))
        value = llm.predict(payload.get("text", ""), policy)
        return {"text": payload.get("text", ""), "value": value}

    result = improve(
        provider=provider,
        model=model,
        max_rounds=int(payload.get("max_rounds", 8)),
        target=float(payload.get("target", 0.98)),
    )
    return {
        "provider": result.provider,
        "model": result.model,
        "start_test_accuracy": result.start_test_accuracy,
        "final_test_accuracy": result.final_test_accuracy,
        "improvement": result.improvement,
        "rounds": [r.model_dump() for r in result.rounds],
        "final_policy": result.final_policy.model_dump(),
    }


if __name__ == "__main__":
    app.run()
