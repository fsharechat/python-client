"""
Interactive CLI for the 飞享IM Q&A chatbot (no FastAPI required).
"""

import asyncio
import os
import time

from config import DOCS_PERSIST_PATH
from graph import build_graph, QAState
from ingest import build_retriever, load_retriever

MAX_HISTORY_TURNS = 10  # 本地 session 保留最近 10 轮


async def stream_answer(graph, question: str, history: list[dict]) -> str:
    """Stream answer tokens to stdout and return the full collected answer."""
    print("助手：", end="", flush=True)
    collected: list[str] = []

    async for event in graph.astream_events(
        QAState(question=question, history=history), version="v2"
    ):
        kind = event["event"]
        node = event.get("metadata", {}).get("langgraph_node", "")

        if kind == "on_chat_model_stream" and node in ("generate", "reject"):
            content = event["data"]["chunk"].content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and part["text"]:
                        print(part["text"], end="", flush=True)
                        collected.append(part["text"])
            elif isinstance(content, str) and content:
                print(content, end="", flush=True)
                collected.append(content)

        elif kind == "on_chain_end" and node == "fallback":
            output = event["data"].get("output", {})
            answer = output.get("answer", "") if isinstance(output, dict) else ""
            if answer:
                print(answer, end="", flush=True)
                collected.append(answer)

    print()  # 输出换行
    return "".join(collected)


def main():
    if not os.path.exists(DOCS_PERSIST_PATH):
        print("首次运行，正在构建知识库（约需 1-2 分钟）...")
        retriever = build_retriever()
    else:
        print("加载知识库...")
        retriever = load_retriever()

    graph = build_graph(retriever)
    history: list[dict] = []  # session 级别会话历史

    print("\n" + "=" * 60)
    print("  飞享IM 智能问答助手（输入 q 退出）")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("您好，请问有什么问题？\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if question.lower() in ("q", "quit", "exit", "退出"):
            print("再见！")
            break

        if not question:
            continue

        print("\n思考中...\n")
        t0 = time.perf_counter()
        answer = asyncio.run(stream_answer(graph, question, history))
        print(f"\n[timing] total: {time.perf_counter() - t0:.2f}s")
        print("-" * 60 + "\n")

        # 保存本轮到历史，超出限制时丢弃最早的一轮
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        if len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-(MAX_HISTORY_TURNS * 2):]


if __name__ == "__main__":
    main()
