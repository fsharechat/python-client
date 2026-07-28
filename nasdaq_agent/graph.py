"""
Nasdaq 100 日报 LangGraph 状态机。

build_nasdaq_graph():    盘前图（4路新闻搜索 + 行情 → 盘前报告 → 通知）
build_afterhours_graph(): 盘后图（4路盘后搜索 + 行情 → 盘后报告 → 通知）
build_intraday_graph():  盘中图（3路开盘搜索 + 行情 → 盘中报告 → 通知）
"""

from langgraph.graph import END, START, StateGraph

from nasdaq_agent.state import NasdaqReportState
from nasdaq_agent.nodes import (
    fetch_stock_movers,
    generate_report,
    generate_afterhours_report,
    generate_intraday_report,
    search_earnings,
    search_futures,
    search_macro_news,
    search_tech_news,
    search_earnings_results,
    search_afterhours_movers,
    search_closing_summary,
    search_tomorrow_preview,
    search_opening_movers,
    search_morning_economics,
    search_opening_news,
    analyze_top_movers,
    find_earnings_reporters,
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

    builder.add_edge(START, "search_tech")
    builder.add_edge(START, "search_macro")
    builder.add_edge(START, "search_earnings")
    builder.add_edge(START, "search_futures")
    builder.add_edge(START, "fetch_stock_movers")

    builder.add_edge("search_tech", "generate_report")
    builder.add_edge("search_macro", "generate_report")
    builder.add_edge("search_earnings", "generate_report")
    builder.add_edge("search_futures", "generate_report")
    builder.add_edge("fetch_stock_movers", "generate_report")

    builder.add_edge("generate_report", "send_notification")
    builder.add_edge("send_notification", END)

    return builder.compile()


def build_afterhours_graph():
    builder = StateGraph(NasdaqReportState)

    builder.add_node("search_earnings_results", search_earnings_results)
    builder.add_node("search_afterhours_movers", search_afterhours_movers)
    builder.add_node("search_closing_summary", search_closing_summary)
    builder.add_node("search_tomorrow_preview", search_tomorrow_preview)
    builder.add_node("fetch_stock_movers", fetch_stock_movers)
    builder.add_node("analyze_top_movers", analyze_top_movers)
    builder.add_node("find_earnings_reporters", find_earnings_reporters)
    builder.add_node("generate_afterhours_report", generate_afterhours_report)
    builder.add_node("send_notification", send_notification)

    builder.add_edge(START, "search_earnings_results")
    builder.add_edge(START, "search_afterhours_movers")
    builder.add_edge(START, "search_closing_summary")
    builder.add_edge(START, "search_tomorrow_preview")
    builder.add_edge(START, "fetch_stock_movers")

    # Track A / Track B 都串在 fetch_stock_movers 之后：
    # fetch_stock_movers 本身要顺序拉 ~105 次 Finnhub 行情（1次/秒），
    # 若 Track B 的 Finnhub 调用与之并发，会触发 429 并连带污染指数/板块表。
    builder.add_edge("fetch_stock_movers", "analyze_top_movers")
    builder.add_edge("fetch_stock_movers", "find_earnings_reporters")

    builder.add_edge("search_earnings_results", "generate_afterhours_report")
    builder.add_edge("search_afterhours_movers", "generate_afterhours_report")
    builder.add_edge("search_closing_summary", "generate_afterhours_report")
    builder.add_edge("search_tomorrow_preview", "generate_afterhours_report")
    builder.add_edge("analyze_top_movers", "generate_afterhours_report")
    builder.add_edge("find_earnings_reporters", "generate_afterhours_report")

    builder.add_edge("generate_afterhours_report", "send_notification")
    builder.add_edge("send_notification", END)

    return builder.compile()


def build_intraday_graph():
    builder = StateGraph(NasdaqReportState)

    builder.add_node("search_opening_movers", search_opening_movers)
    builder.add_node("search_morning_economics", search_morning_economics)
    builder.add_node("search_opening_news", search_opening_news)
    builder.add_node("fetch_stock_movers", fetch_stock_movers)
    builder.add_node("generate_intraday_report", generate_intraday_report)
    builder.add_node("send_notification", send_notification)

    builder.add_edge(START, "search_opening_movers")
    builder.add_edge(START, "search_morning_economics")
    builder.add_edge(START, "search_opening_news")
    builder.add_edge(START, "fetch_stock_movers")

    builder.add_edge("search_opening_movers", "generate_intraday_report")
    builder.add_edge("search_morning_economics", "generate_intraday_report")
    builder.add_edge("search_opening_news", "generate_intraday_report")
    builder.add_edge("fetch_stock_movers", "generate_intraday_report")

    builder.add_edge("generate_intraday_report", "send_notification")
    builder.add_edge("send_notification", END)

    return builder.compile()
