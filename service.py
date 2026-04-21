"""
FastAPI service exposing the 飞享IM Q&A chatbot.

Endpoints:
  POST /ask      – single-turn Q&A
  POST /stream   – streaming Q&A (SSE)
  GET  /health   – liveness probe
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import DOCS_PERSIST_PATH
from graph import build_graph, QAState
from ingest import build_retriever, load_retriever


# ─── App lifecycle ────────────────────────────────────────────────────────────

app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(DOCS_PERSIST_PATH):
        print("Knowledge base not found – running ingestion...")
        retriever = build_retriever()
    else:
        print("Loading existing knowledge base...")
        retriever = load_retriever()

    app_state["graph"] = build_graph(retriever)
    print("Q&A service ready.")
    yield
    app_state.clear()


app = FastAPI(
    title="飞享IM 智能问答服务",
    description="基于 LangChain + LangGraph + Claude 的飞享IM知识助手",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
    route: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    graph = app_state["graph"]
    result = graph.invoke(QAState(question=req.question))
    if isinstance(result, dict):
        return AskResponse(
            question=result.get("question", req.question),
            answer=result.get("answer", ""),
            route=result.get("route", ""),
        )
    return AskResponse(
        question=result.question,
        answer=result.answer,
        route=result.route,
    )


@app.post("/stream")
def stream_ask(req: AskRequest):
    """Stream the answer token-by-token via Server-Sent Events."""
    graph = app_state["graph"]

    def event_generator():
        for chunk in graph.stream(QAState(question=req.question)):
            for node_name, updates in chunk.items():
                print(f"[node: {node_name}] {updates}")
                if "answer" in updates and updates["answer"]:
                    yield updates["answer"]

    return StreamingResponse(event_generator(), media_type="text/plain; charset=utf-8")


# ─── CLI convenience ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=False)
