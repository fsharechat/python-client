import operator
from typing import Annotated
from typing_extensions import TypedDict


class NasdaqReportState(TypedDict):
    date: str
    raw_articles: Annotated[list[dict], operator.add]  # reducer: accumulated by all parallel search nodes
    index_summary: str  # 由 fetch_stock_movers 写入：纳指100 + 标普500 当日收盘指数表格
    stock_movers: str   # 由 fetch_stock_movers 节点写入：东方财富 API 行情，按板块分组的涨跌幅 markdown 表格
    report_content: str
    send_status: str
