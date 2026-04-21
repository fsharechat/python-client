"""
Interactive CLI for the 飞享IM Q&A chatbot (no FastAPI required).
"""

import os
import sys

from config import CHROMA_PERSIST_DIR
from graph import build_graph, QAState
from ingest import build_vectorstore, load_vectorstore


def main():
    # Prepare vector store
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print("首次运行，正在构建知识库（约需 1-2 分钟）...")
        vectorstore = build_vectorstore()
    else:
        print("加载知识库...")
        vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
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
        result = graph.invoke(QAState(question=question))
        answer = result.answer if isinstance(result, QAState) else result.get("answer", "")
        print(f"助手：{answer}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
