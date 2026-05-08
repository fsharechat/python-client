"""
Nasdaq 100 盘前日报 LangGraph 状态机。

并行搜索（4个节点同时运行）→ generate_report（等待所有搜索完成）→ send_notification
"""

from langgraph.graph import END, START, StateGraph

from nasdaq_agent.state import NasdaqReportState
from nasdaq_agent.nodes import (
    fetch_stock_movers,
    generate_report,
    search_earnings,
    search_futures,
    search_macro_news,
    search_tech_news,
    send_notification,
)


def build_nasdaq_graph():
    builder = StateGraph(NasdaqReportState)

    builder.add_node("search_tech", search_tech_news)
    builder.add_node("search_macro", search_macro_news)
    builder.add_node("search_earnings", search_earnings)
    builder.add_node("search_futures", search_futures)
    builder.add_node("fetch_stock_movers", fetch_stock_movers)
    builder.add_node("generate_report", generate_report)
    builder.add_node("send_notification", send_notification)

    # Fan-out: START → 5 节点并行（4路新闻搜索 + 1路股票行情）
    builder.add_edge(START, "search_tech")
    builder.add_edge(START, "search_macro")
    builder.add_edge(START, "search_earnings")
    builder.add_edge(START, "search_futures")
    builder.add_edge(START, "fetch_stock_movers")

    # Fan-in: 全部完成后才进入 generate_report
    builder.add_edge("search_tech", "generate_report")
    builder.add_edge("search_macro", "generate_report")
    builder.add_edge("search_earnings", "generate_report")
    builder.add_edge("search_futures", "generate_report")
    builder.add_edge("fetch_stock_movers", "generate_report")

    builder.add_edge("generate_report", "send_notification")
    builder.add_edge("send_notification", END)

    return builder.compile()
