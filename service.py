"""
FastAPI service exposing the 飞享IM Q&A chatbot.

Endpoints:
  POST /ask                        – single-turn Q&A
  POST /stream                     – streaming Q&A (SSE)
  GET  /health                     – liveness probe
  POST /nasdaq/trigger             – manually trigger Nasdaq premarket report
  POST /nasdaq/trigger/afterhours  – manually trigger Nasdaq afterhours report
"""

import asyncio
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
from collections import deque
from contextlib import asynccontextmanager
from datetime import date as date_type

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import DOCS_PERSIST_PATH
from graph import build_graph, QAState
from ingest import build_retriever, load_retriever
from nasdaq_agent.graph import build_nasdaq_graph, build_afterhours_graph


# ─── Nasdaq report runner ─────────────────────────────────────────────────────

async def _run_nasdaq_report(nasdaq_graph) -> None:
    today = date_type.today().isoformat()
    print(f"[nasdaq] Starting daily report for {today} ...")
    initial = {"date": today, "report_type": "premarket", "raw_articles": [], "index_summary": "", "stock_results": [], "report_content": "", "send_status": ""}
    try:
        await nasdaq_graph.ainvoke(initial)
        print("[nasdaq] Daily report complete.")
    except Exception as e:
        print(f"[nasdaq] Daily report failed: {e}")

async def _run_afterhours_report(afterhours_graph) -> None:
    today = date_type.today().isoformat()
    print(f"[afterhours] Starting afterhours report for {today} ...")
    initial = {"date": today, "report_type": "afterhours", "raw_articles": [], "index_summary": "", "stock_results": [], "report_content": "", "send_status": ""}
    try:
        await afterhours_graph.ainvoke(initial)
        print("[afterhours] Afterhours report complete.")
    except Exception as e:
        print(f"[afterhours] Afterhours report failed: {e}")


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

    # ── Nasdaq 日报定时任务（盘前 ET 08:00 + 盘后 ET 16:30） ────────────────
    nasdaq_graph = build_nasdaq_graph()
    app_state["nasdaq_graph"] = nasdaq_graph

    afterhours_graph = build_afterhours_graph()
    app_state["afterhours_graph"] = afterhours_graph

    scheduler = AsyncIOScheduler(timezone="America/New_York")
    scheduler.add_job(
        _run_nasdaq_report,
        "cron",
        day_of_week="mon-fri",
        hour=8,
        minute=0,
        args=[nasdaq_graph],
    )
    scheduler.add_job(
        _run_afterhours_report,
        "cron",
        day_of_week="mon-fri",
        hour=16,
        minute=30,
        args=[afterhours_graph],
    )
    scheduler.start()
    app_state["scheduler"] = scheduler
    print("Nasdaq scheduler started (ET 08:00 premarket / ET 16:30 afterhours, Mon–Fri).")

    yield

    scheduler.shutdown(wait=False)
    app_state.clear()


app = FastAPI(
    title="飞享IM 智能问答服务",
    description="基于 LangChain + LangGraph + Claude 的飞享IM知识助手",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Conversation memory ──────────────────────────────────────────────────────

MAX_HISTORY_TURNS = 3  # 每用户保留最近 3 轮（user + assistant 各一条 = 6 条消息）

# {userid: deque([{"role": "user"|"assistant", "content": "..."}])}
conversation_store: dict[str, deque] = {}


def get_history(userid: str) -> list[dict]:
    return list(conversation_store.get(userid, []))


def save_exchange(userid: str, question: str, answer: str) -> None:
    if userid not in conversation_store:
        conversation_store[userid] = deque(maxlen=MAX_HISTORY_TURNS * 2)
    store = conversation_store[userid]
    store.append({"role": "user", "content": question})
    store.append({"role": "assistant", "content": answer})


# ─── Schemas ──────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    userid: str = "default"


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
    history = get_history(req.userid)
    result = graph.invoke(QAState(question=req.question, history=history))
    if isinstance(result, dict):
        answer = result.get("answer", "")
        save_exchange(req.userid, req.question, answer)
        return AskResponse(
            question=result.get("question", req.question),
            answer=answer,
            route=result.get("route", ""),
        )
    save_exchange(req.userid, req.question, result.answer)
    return AskResponse(
        question=result.question,
        answer=result.answer,
        route=result.route,
    )


@app.post("/stream")
async def stream_ask(req: AskRequest):
    """Stream the answer token-by-token via Server-Sent Events."""
    graph = app_state["graph"]
    history = get_history(req.userid)
    collected: list[str] = []

    async def event_generator():
        async for event in graph.astream_events(
            QAState(question=req.question, history=history), version="v2"
        ):
            kind = event["event"]
            node = event.get("metadata", {}).get("langgraph_node", "")

            if kind == "on_chat_model_stream" and node in ("generate", "reject"):
                content = event["data"]["chunk"].content
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text" and part["text"]:
                            collected.append(part["text"])
                            yield part["text"]
                elif isinstance(content, str) and content:
                    collected.append(content)
                    yield content

            elif kind == "on_chain_end" and node == "fallback":
                output = event["data"].get("output", {})
                answer = output.get("answer", "") if isinstance(output, dict) else ""
                if answer:
                    collected.append(answer)
                    yield answer

        save_exchange(req.userid, req.question, "".join(collected))

    return StreamingResponse(
        event_generator(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Nasdaq manual trigger ────────────────────────────────────────────────────

@app.post("/nasdaq/trigger")
async def trigger_nasdaq_report():
    """立即触发一次纳斯达克盘前日报（用于测试或手动补发）。"""
    nasdaq_graph = app_state["nasdaq_graph"]
    asyncio.create_task(_run_nasdaq_report(nasdaq_graph))
    return {"status": "triggered", "date": date_type.today().isoformat()}

@app.post("/nasdaq/trigger/afterhours")
async def trigger_afterhours_report():
    """立即触发一次纳斯达克盘后日报（用于测试或手动补发）。"""
    afterhours_graph = app_state["afterhours_graph"]
    asyncio.create_task(_run_afterhours_report(afterhours_graph))
    return {"status": "triggered", "date": date_type.today().isoformat()}


# ─── CLI convenience ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=False)
