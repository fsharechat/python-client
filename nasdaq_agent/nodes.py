"""
Nasdaq 100 盘前日报 Agent 各节点实现。

并行搜索节点（4个）→ generate_report → send_notification
"""

import time
import requests
from ddgs import DDGS

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from config import ANTHROPIC_API_KEY, GENERATE_MODEL
from nasdaq_agent.state import NasdaqReportState

NOTIFY_URL = "https://backend-http.fsharechat.cn/imopenapi/pushNotificationByMobile"
NOTIFY_MOBILES = [13900000001]
MAX_REPORT_CHARS = 2048


# ─── 搜索工具 ─────────────────────────────────────────────────────────────────

def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {"title": r.get("title", ""), "url": r.get("href", ""), "body": r.get("body", "")}
            for r in results
        ]
    except Exception as e:
        print(f"[nasdaq] DDG search failed: '{query}' → {e}")
        return []


# ─── 并行搜索节点 ──────────────────────────────────────────────────────────────

def search_tech_news(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    results = _ddg_search(f"Nasdaq 100 technology stocks premarket {state['date']}")
    print(f"[nasdaq] search_tech: {time.perf_counter() - t0:.2f}s → {len(results)} results")
    return {"raw_articles": results}


def search_macro_news(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    results = _ddg_search(f"Federal Reserve interest rate US economy market {state['date']}")
    print(f"[nasdaq] search_macro: {time.perf_counter() - t0:.2f}s → {len(results)} results")
    return {"raw_articles": results}


def search_earnings(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    results = _ddg_search(f"tech stocks earnings results premarket today {state['date']}")
    print(f"[nasdaq] search_earnings: {time.perf_counter() - t0:.2f}s → {len(results)} results")
    return {"raw_articles": results}


def search_futures(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    results = _ddg_search(f"QQQ Nasdaq futures premarket market open {state['date']}")
    print(f"[nasdaq] search_futures: {time.perf_counter() - t0:.2f}s → {len(results)} results")
    return {"raw_articles": results}


# ─── 报告生成节点 ──────────────────────────────────────────────────────────────

_REPORT_SYSTEM = """你是专业美股分析师，擅长整理纳斯达克100盘前动态。
根据以下新闻摘要，用中文生成一份简洁的盘前日报。

严格要求：
- 总字数不超过1700字（为通知渠道留余量）
- 信息客观准确，不要编造数据
- 使用以下固定格式，不要偏离

输出格式：
【纳斯达克100盘前日报】{date}

📊 市场概况
（2-3句：纳指期货方向、整体情绪）

🔥 盘前三大热点
1.
2.
3.

📈 重点个股动向
（3-5只：股票名(代码) 涨跌幅，简短原因）

⚠️ 风险提示
（1-2点关键风险或待关注事件）

来源：Reuters/CNBC/MarketWatch"""


def generate_report(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    articles = state["raw_articles"]

    # 取前20条，每条摘要截断为200字，避免token溢出
    context_parts = [
        f"标题：{a['title']}\n摘要：{a['body'][:200]}"
        for a in articles[:20]
        if a.get("title") or a.get("body")
    ]
    context = "\n\n".join(context_parts) if context_parts else "（暂无搜索结果）"

    llm = ChatAnthropic(
        model=GENERATE_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=1200,
        max_retries=3,
    )

    prompt = _REPORT_SYSTEM.replace("{date}", state["date"])
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"今日盘前新闻摘要如下：\n\n{context}"),
    ]

    report = (llm | StrOutputParser()).invoke(messages)

    if len(report) > MAX_REPORT_CHARS:
        report = report[: MAX_REPORT_CHARS - 3] + "..."

    print(f"[nasdaq] generate_report: {time.perf_counter() - t0:.2f}s → {len(report)} chars")
    return {"report_content": report}


# ─── 发送通知节点 ──────────────────────────────────────────────────────────────

def send_notification(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    report = state["report_content"]

    if not report:
        print("[nasdaq] send_notification: report is empty, skipped")
        return {"send_status": "skipped:empty_report"}

    payload = {
        "mobiles": NOTIFY_MOBILES,
        "content": {
            "content": {
                "type": 1,
                "searchableContent": report,
            }
        },
    }

    try:
        resp = requests.post(NOTIFY_URL, json=payload, timeout=15)
        status = f"ok:{resp.status_code}"
        print(f"[nasdaq] send_notification: {time.perf_counter() - t0:.2f}s → {status} | body={resp.text[:120]}")
    except Exception as e:
        status = f"error:{e}"
        print(f"[nasdaq] send_notification failed: {e}")

    return {"send_status": status}
