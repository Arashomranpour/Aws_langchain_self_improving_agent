# SelfForge — container for Amazon Bedrock AgentCore Runtime.
# AgentCore requires linux/arm64 and an HTTP server on port 8080 exposing
# POST /invocations and GET /ping. The bedrock-agentcore SDK provides that server.
FROM --platform=linux/arm64 python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY web/ ./web/
ENV PYTHONPATH=/app/src

EXPOSE 8080

# AgentCore invokes the entrypoint server (POST /invocations, GET /ping on :8080).
CMD ["python", "-m", "selfforge.agentcore_app"]
