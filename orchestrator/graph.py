"""
Orchestrator LangGraph workflow.

Flow:
  user message
      │
      ▼
  [classify_intent]  ← Claude Haiku, returns "faq" or "order_ops"
      │
      ├─ faq ──────► [call_faq_agent]    → Python Q&A service (:8000)
      │
      └─ order_ops ► [call_order_agent]  → Java LangChain4j service (:8080)
      │
      ▼
  Unified response
"""

from __future__ import annotations

from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from orchestrator.config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from orchestrator import clients

__all__ = ["OrchestratorState", "build_graph", "classify_question"]


# ─── State ────────────────────────────────────────────────────────────────────

class OrchestratorState(BaseModel):
    question: str
    session_id: str = ""
    intent: Literal["faq", "order_ops"] = "faq"
    answer: str = ""
    source: Literal["faq_agent", "order_agent"] = "faq_agent"


# ─── LLM ──────────────────────────────────────────────────────────────────────

def _llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=CLAUDE_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=16,  # only need "faq" or "order_ops"
    )


# ─── Node: classify_intent ────────────────────────────────────────────────────

CLASSIFY_SYSTEM = """你是意图分类器。将用户输入分类为以下两类之一：
- "faq"：关于飞享IM/FshareChat产品的功能、使用方法、部署、价格、下载等常见问题
- "order_ops"：涉及订单查询、退款申请、物流查询、商品查询、账单/支付、投诉、售后服务

只返回 "faq" 或 "order_ops"，不含任何其他文字。"""


def classify_intent(state: OrchestratorState) -> dict:
    llm = _llm()
    resp = llm.invoke([
        SystemMessage(content=CLASSIFY_SYSTEM),
        HumanMessage(content=state.question),
    ])
    intent = "order_ops" if "order_ops" in resp.content else "faq"
    return {"intent": intent}


async def classify_question(question: str) -> str:
    """Standalone async intent classifier; returns 'faq' or 'order_ops'."""
    llm = _llm()
    resp = await llm.ainvoke([
        SystemMessage(content=CLASSIFY_SYSTEM),
        HumanMessage(content=question),
    ])
    return "order_ops" if "order_ops" in resp.content else "faq"


# ─── Node: call_faq_agent ─────────────────────────────────────────────────────

async def call_faq_agent(state: OrchestratorState) -> dict:
    answer = await clients.call_faq_agent(state.question)
    return {"answer": answer, "source": "faq_agent"}


# ─── Node: call_order_agent ───────────────────────────────────────────────────

async def call_order_agent(state: OrchestratorState) -> dict:
    answer = await clients.call_order_agent(state.question, state.session_id)
    return {"answer": answer, "source": "order_agent"}


# ─── Routing ──────────────────────────────────────────────────────────────────

def route_after_classify(state: OrchestratorState) -> str:
    return "call_faq_agent" if state.intent == "faq" else "call_order_agent"


# ─── Graph builder ────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(OrchestratorState)

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("call_faq_agent", call_faq_agent)
    builder.add_node("call_order_agent", call_order_agent)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges("classify_intent", route_after_classify)
    builder.add_edge("call_faq_agent", END)
    builder.add_edge("call_order_agent", END)

    return builder.compile()
