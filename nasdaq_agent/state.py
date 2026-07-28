import operator
from typing import Annotated
from typing_extensions import TypedDict


class NasdaqReportState(TypedDict):
    date: str
    report_type: str    # "premarket" | "afterhours" | "intraday"，service.py 初始化时写入
    raw_articles: Annotated[list[dict], operator.add]  # reducer: accumulated by all parallel search nodes
    index_summary: str  # 由 fetch_stock_movers 写入：指数行情 markdown 表格（盘前用QQQ/SPY，盘后用NDX/SPX）
    stock_results: list # 由 fetch_stock_movers 写入：原始 [(sym, price, chg), ...] 供报告节点按场景格式化
    movers_analysis: list  # 由 analyze_top_movers 写入（仅盘后）：[{"symbol","price","chg","news_snippets"}, ...]
    earnings_analysis: list  # 由 find_earnings_reporters 写入（仅盘后）：[{"symbol","eps_actual","eps_estimate","period","net_income","capex","operating_cf","fcf"}, ...]
    report_content: str
    send_status: str
