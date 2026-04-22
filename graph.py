"""
LangGraph workflow for 飞享IM Q&A service.

Flow:
  user question
      │
      ▼
  [classify] ─── off-topic ──► [reject]
      │
  on-topic
      │
      ▼
  [retrieve] → fetch top-k chunks from ChromaDB
      │
      ▼
  [grade_docs] → filter irrelevant chunks
      │
      ├─ has relevant docs ──► [generate] → stream answer with Claude
      │
      └─ no relevant docs ──► [fallback] → politely say knowledge is limited
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Literal
import operator

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLASSIFY_MODEL, GENERATE_MODEL


# ─── State ────────────────────────────────────────────────────────────────────

class QAState(BaseModel):
    question: str
    retrieved_docs: list[str] = Field(default_factory=list)
    relevant_docs: list[str] = Field(default_factory=list)
    answer: str = ""
    route: Literal["on_topic", "off_topic"] = "on_topic"


# ─── LLM ──────────────────────────────────────────────────────────────────────

def _llm(streaming: bool = False, model: str = CLAUDE_MODEL, thinking: bool = False, max_tokens: int = 2048) -> ChatAnthropic:
    kwargs: dict = dict(
        model=model,
        anthropic_api_key=ANTHROPIC_API_KEY,
        streaming=streaming,
        max_tokens=max_tokens,
        max_retries=3,
    )
    if thinking:
        kwargs["thinking"] = {"type": "adaptive"}
    return ChatAnthropic(**kwargs)


# ─── Node: classify ───────────────────────────────────────────────────────────

CLASSIFY_SYSTEM = """你是飞享IM智能助手的问题分类器。
判断用户问题是否与飞享IM相关（功能、部署、技术、价格、使用方法等）。
只回答 "on_topic" 或 "off_topic"，不要解释。"""

def classify(state: QAState, retriever) -> dict:
    t0 = time.perf_counter()
    llm = _llm(model=CLASSIFY_MODEL)
    resp = llm.invoke([
        SystemMessage(content=CLASSIFY_SYSTEM),
        HumanMessage(content=state.question),
    ])
    route = "on_topic" if "on_topic" in resp.content else "off_topic"
    print(f"[timing] classify: {time.perf_counter() - t0:.2f}s → {route}")
    return {"route": route}


# ─── Node: retrieve ───────────────────────────────────────────────────────────

def retrieve(state: QAState, retriever) -> dict:
    t0 = time.perf_counter()
    docs = retriever.invoke(state.question)
    chunks = [d.page_content for d in docs]
    print(f"[timing] retrieve: {time.perf_counter() - t0:.2f}s → {len(chunks)} docs")
    return {"retrieved_docs": chunks}


# ─── Node: grade_docs ─────────────────────────────────────────────────────────

GRADE_SYSTEM = """你是文档相关性评分器。
给定用户问题和一段文档，判断文档是否包含回答问题所需的相关信息。
只回答 "relevant" 或 "irrelevant"。"""

def grade_docs(state: QAState, retriever) -> dict:
    t0 = time.perf_counter()

    def _grade_one(i: int, doc: str) -> tuple[int, str, str]:
        t1 = time.perf_counter()
        llm = _llm(thinking=True)
        resp = llm.invoke([
            SystemMessage(content=GRADE_SYSTEM),
            HumanMessage(content=f"问题：{state.question}\n\n文档：{doc}"),
        ])
        verdict = "relevant" if "relevant" in resp.content.lower() else "irrelevant"
        print(f"[timing]   grade doc[{i}]: {time.perf_counter() - t1:.2f}s → {verdict}")
        return (i, doc, verdict)

    results: list[tuple[int, str, str]] = [None] * len(state.retrieved_docs)
    with ThreadPoolExecutor(max_workers=len(state.retrieved_docs)) as pool:
        futures = {pool.submit(_grade_one, i, doc): i for i, doc in enumerate(state.retrieved_docs)}
        for future in as_completed(futures):
            i, doc, verdict = future.result()
            results[i] = (i, doc, verdict)

    relevant = [doc for _, doc, verdict in results if verdict == "relevant"]
    print(f"[timing] grade_docs total: {time.perf_counter() - t0:.2f}s → {len(relevant)}/{len(state.retrieved_docs)} relevant")
    return {"relevant_docs": relevant}


# ─── Node: generate ───────────────────────────────────────────────────────────

GENERATE_SYSTEM = """你是飞享IM的专业客服助手，熟悉飞享IM的所有功能和使用方法。
请根据提供的参考资料，用中文准确、友好地回答用户问题。
- 回答控制在300字以内，简洁明了，条理清晰
- 如果参考资料不足以完整回答，请说明哪些信息你不确定
- 不要编造飞享IM没有的功能"""

def generate(state: QAState, retriever) -> dict:
    t0 = time.perf_counter()
    context = "\n\n---\n\n".join(state.relevant_docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", GENERATE_SYSTEM),
        ("human", "参考资料：\n{context}\n\n用户问题：{question}"),
    ])
    llm = _llm(streaming=True, model=GENERATE_MODEL, thinking=False, max_tokens=600)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": state.question})
    print(f"[timing] generate: {time.perf_counter() - t0:.2f}s")
    return {"answer": answer}


# ─── Node: fallback ───────────────────────────────────────────────────────────

def fallback(state: QAState, retriever) -> dict:
    answer = (
        "抱歉，我目前的知识库中没有找到与您问题直接相关的信息。\n"
        "建议您：\n"
        "1. 访问飞享IM官网 https://fsharechat.cn 获取最新资料\n"
        "2. 尝试换一种方式描述您的问题\n"
        "3. 联系飞享IM官方客服获取专业帮助"
    )
    return {"answer": answer}


# ─── Node: reject ─────────────────────────────────────────────────────────────

def reject(state: QAState, retriever) -> dict:
    answer = (
        "您好！我是飞享IM专属智能助手，只能回答与飞享IM相关的问题。\n"
        "请问您有关于飞享IM功能、部署、使用等方面的问题吗？"
    )
    return {"answer": answer}


# ─── Routing ──────────────────────────────────────────────────────────────────

def route_after_classify(state: QAState) -> str:
    return "retrieve" if state.route == "on_topic" else "reject"


def route_after_grade(state: QAState) -> str:
    return "generate" if state.relevant_docs else "fallback"


# ─── Graph builder ────────────────────────────────────────────────────────────

def build_graph(retriever):
    """Build and compile the LangGraph workflow."""

    # Bind retriever into each node via closure
    def _classify(state: QAState): return classify(state, retriever)
    def _retrieve(state: QAState): return retrieve(state, retriever)
    def _grade(state: QAState): return grade_docs(state, retriever)
    def _generate(state: QAState): return generate(state, retriever)
    def _fallback(state: QAState): return fallback(state, retriever)
    def _reject(state: QAState): return reject(state, retriever)

    builder = StateGraph(QAState)

    builder.add_node("classify", _classify)
    builder.add_node("retrieve", _retrieve)
    builder.add_node("grade_docs", _grade)
    builder.add_node("generate", _generate)
    builder.add_node("fallback", _fallback)
    builder.add_node("reject", _reject)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_after_classify)
    builder.add_edge("retrieve", "grade_docs")
    builder.add_conditional_edges("grade_docs", route_after_grade)
    builder.add_edge("generate", END)
    builder.add_edge("fallback", END)
    builder.add_edge("reject", END)

    return builder.compile()
