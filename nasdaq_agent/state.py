import operator
from typing import Annotated
from typing_extensions import TypedDict


class NasdaqReportState(TypedDict):
    date: str
    raw_articles: Annotated[list[dict], operator.add]  # reducer: accumulated by all parallel search nodes
    report_content: str
    send_status: str
