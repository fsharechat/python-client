import operator
from typing import Annotated
from typing_extensions import TypedDict


class NasdaqReportState(TypedDict):
    date: str
    raw_articles: Annotated[list[dict], operator.add]  # reducer: accumulated by all parallel search nodes
    stock_movers: str   # yfinance 拉取的涨跌幅前十，由 fetch_stock_movers 节点写入
    report_content: str
    send_status: str
