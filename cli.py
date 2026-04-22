"""
Interactive CLI for the 飞享IM Q&A chatbot (no FastAPI required).
"""

import asyncio
import os
import time

from config import DOCS_PERSIST_PATH
from graph import build_graph, QAState
from ingest import build_retriever, load_retriever


async def stream_answer(graph, question: str) -> None:
    print("助手：", end="", flush=True)
    async for event in graph.astream_events(QAState(question=question), version="v2"):
        kind = event["event"]
        node = event.get("metadata", {}).get("langgraph_node", "")

        if kind == "on_chat_model_stream" and node == "generate":
            content = event["data"]["chunk"].content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and part["text"]:
                        print(part["text"], end="", flush=True)
            elif isinstance(content, str) and content:
                print(content, end="", flush=True)

        elif kind == "on_chain_end" and node in ("reject", "fallback"):
            output = event["data"].get("output", {})
            answer = output.get("answer", "") if isinstance(output, dict) else ""
            if answer:
                print(answer, end="", flush=True)

    print()  # 输出换行


def main():
    if not os.path.exists(DOCS_PERSIST_PATH):
        print("首次运行，正在构建知识库（约需 1-2 分钟）...")
        retriever = build_retriever()
    else:
        print("加载知识库...")
        retriever = load_retriever()

    graph = build_graph(retriever)

    print("\n" + "=" * 60)
    print("  飞享IM 智能问答助手（输入 q 退出）")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("您好，请问有什么关于飞享IM的问题？\n> ").strip()
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
        asyncio.run(stream_answer(graph, question))
        print(f"\n[timing] total: {time.perf_counter() - t0:.2f}s")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
