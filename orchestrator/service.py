"""
Orchestrator service — unified entry point routing to FAQ or Order agent.

Endpoints:
  POST /chat    – route user message to the appropriate downstream agent
  GET  /health  – liveness probe + downstream status
"""

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from orchestrator.config import ORCHESTRATOR_PORT
from orchestrator.graph import build_graph, OrchestratorState, classify_question
from orchestrator import clients


# ─── App lifecycle ────────────────────────────────────────────────────────────

app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["graph"] = build_graph()
    print("Orchestrator ready.")
    yield
    app_state.clear()


app = FastAPI(
    title="智能客服编排层",
    description="将用户请求路由到飞享IM FAQ助手或订单/退款客服系统",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = ""


class ChatResponse(BaseModel):
    message: str
    answer: str
    intent: str
    source: str
    session_id: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    faq_status, order_status = await asyncio.gather(
        clients.health_check_faq(),
        clients.health_check_order(),
    )
    return {
        "status": "ok",
        "downstream": {
            "faq_agent": faq_status,
            "order_agent": order_status,
        },
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    graph = app_state["graph"]
    try:
        result = await graph.ainvoke(
            OrchestratorState(question=req.message, session_id=req.session_id)
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Downstream agent error: {e}")

    if isinstance(result, dict):
        return ChatResponse(
            message=req.message,
            answer=result.get("answer", ""),
            intent=result.get("intent", ""),
            source=result.get("source", ""),
            session_id=req.session_id,
        )
    return ChatResponse(
        message=req.message,
        answer=result.answer,
        intent=result.intent,
        source=result.source,
        session_id=req.session_id,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream the answer via SSE. First event carries routing metadata."""
    intent = await classify_question(req.message)
    source = "faq_agent" if intent == "faq" else "order_agent"

    async def event_generator():
        # First line: routing metadata as JSON
        yield json.dumps({"intent": intent, "source": source}, ensure_ascii=False) + "\n"
        try:
            if intent == "faq":
                stream = clients.stream_faq_agent(req.message)
            else:
                stream = clients.stream_order_agent(req.message, req.session_id)
            async for token in stream:
                yield token
        except Exception as e:
            yield f"\n[ERROR] {e}"

    return StreamingResponse(event_generator(), media_type="text/plain; charset=utf-8")


# ─── CLI convenience ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrator.service:app", host="0.0.0.0", port=ORCHESTRATOR_PORT, reload=False)
