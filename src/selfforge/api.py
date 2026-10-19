"""Local visual web app: a FastAPI server that runs the self-improvement loop and
serves a single-page UI so you can watch the agent teach itself.

    uvicorn selfforge.api:app --reload      # then open http://localhost:8080

This is the 'see the app' surface. The AgentCore Runtime contract lives in
agentcore_app.py; this one is for humans.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from selfforge.core import improve

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="SelfForge", description="A self-improving LangGraph agent")


class RunRequest(BaseModel):
    provider: str = "offline"
    model: str | None = None
    max_rounds: int = 8
    target: float = 0.98


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>SelfForge</h1><p>web/index.html not found.</p>"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/run")
def run(req: RunRequest) -> JSONResponse:
    try:
        result = improve(
            provider=req.provider,
            model=req.model,
            max_rounds=req.max_rounds,
            target=req.target,
        )
    except Exception as exc:  # noqa: BLE001 - surface key/provider errors to the UI
        return JSONResponse(status_code=400, content={"error": str(exc)})

    return JSONResponse(
        content={
            "provider": result.provider,
            "model": result.model,
            "start_test_accuracy": result.start_test_accuracy,
            "final_test_accuracy": result.final_test_accuracy,
            "improvement": result.improvement,
            "rounds": [r.model_dump() for r in result.rounds],
            "final_policy": result.final_policy.model_dump(),
        }
    )
