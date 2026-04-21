import json
from typing import AsyncGenerator

import httpx
from orchestrator.config import PYTHON_QA_AGENT_URL, JAVA_ORDER_AGENT_URL, REQUEST_TIMEOUT

# trust_env=False disables system proxy env vars (HTTP_PROXY, HTTPS_PROXY, ALL_PROXY)
# for all direct service-to-service calls
_CLIENT_KWARGS = {"trust_env": False}


async def call_faq_agent(question: str) -> str:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, **_CLIENT_KWARGS) as client:
        resp = await client.post(
            f"{PYTHON_QA_AGENT_URL}/ask",
            json={"question": question},
        )
        resp.raise_for_status()
        return resp.json()["answer"]


async def call_order_agent(message: str, session_id: str) -> str:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, **_CLIENT_KWARGS) as client:
        resp = await client.post(
            f"{JAVA_ORDER_AGENT_URL}/api/chat",
            json={
                "message": message,
                "sessionId": session_id or None,
                "enableFunctionCalling": True,
                "enableRag": True,
            },
        )
        resp.raise_for_status()
        return resp.json()["content"]


async def stream_faq_agent(question: str) -> AsyncGenerator[str, None]:
    """Proxy the Python FAQ agent plain-text stream, yielding answer chunks."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, **_CLIENT_KWARGS) as client:
        async with client.stream("POST", f"{PYTHON_QA_AGENT_URL}/stream", json={"question": question}) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_text():
                yield chunk


async def stream_order_agent(message: str, session_id: str) -> AsyncGenerator[str, None]:
    """Proxy the Java order agent SSE stream, yielding token strings."""
    params: dict = {"message": message, "enableRag": "true"}
    if session_id:
        params["sessionId"] = session_id
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, **_CLIENT_KWARGS) as client:
        async with client.stream("GET", f"{JAVA_ORDER_AGENT_URL}/api/chat/stream", params=params) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                try:
                    obj = json.loads(data)
                    if "token" in obj:
                        yield obj["token"]
                except (json.JSONDecodeError, KeyError):
                    pass


async def health_check_faq() -> str:
    try:
        async with httpx.AsyncClient(timeout=5, **_CLIENT_KWARGS) as client:
            resp = await client.get(f"{PYTHON_QA_AGENT_URL}/health")
            return "ok" if resp.status_code == 200 else "degraded"
    except Exception:
        return "unreachable"


async def health_check_order() -> str:
    try:
        async with httpx.AsyncClient(timeout=5, **_CLIENT_KWARGS) as client:
            resp = await client.get(f"{JAVA_ORDER_AGENT_URL}/api/chat/health")
            return "ok" if resp.status_code == 200 else "degraded"
    except Exception:
        return "unreachable"
