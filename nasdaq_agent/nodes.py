"""
Nasdaq 100 盘前日报 Agent 各节点实现（全 async）。

搜索使用 httpx 异步抓取 DuckDuckGo HTML，避免 ddgs/primp 在线程池中的 segfault。
并行节点（4路搜索 + 1路股票行情）→ generate_report → send_notification
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import yfinance as yf
from bs4 import BeautifulSoup
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from config import ANTHROPIC_API_KEY, GENERATE_MODEL
from nasdaq_agent.state import NasdaqReportState
from nasdaq_agent.tickers import NASDAQ100_TICKERS

NOTIFY_URL = "https://backend-http.fsharechat.cn/imopenapi/pushNotificationByMobile"
NOTIFY_MOBILES = [13900000001]
MAX_REPORT_CHARS = 2048

_DDG_URL = "https://html.duckduckgo.com/html/"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


# ─── 搜索工具（纯 async，无线程） ─────────────────────────────────────────────

async def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    try:
        async with httpx.AsyncClient(headers=_HEADERS, timeout=20, follow_redirects=True) as client:
            resp = await client.post(_DDG_URL, data={"q": query})
        soup = BeautifulSoup(resp.text, "lxml")
        results = []
        for el in soup.select(".result")[:max_results]:
            title = el.select_one(".result__title")
            snippet = el.select_one(".result__snippet")
            url = el.select_one(".result__url")
            if title and snippet:
                results.append({
                    "title": title.get_text(strip=True),
                    "url": url.get_text(strip=True) if url else "",
                    "body": snippet.get_text(strip=True),
                })
        return results
    except Exception as e:
        print(f"[nasdaq] DDG search failed: '{query}' → {e}")
        return []


# ─── 并行搜索节点（async，LangGraph ainvoke 并发执行） ────────────────────────

async def search_tech_news(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    query = f"Nasdaq 100 technology stocks premarket {state['date']}"
    print(f"[nasdaq] search_tech query: {query}")
    results = await _ddg_search(query)
    print(f"[nasdaq] search_tech: {time.perf_counter() - t0:.2f}s → {len(results)} results")
    return {"raw_articles": results}


async def search_macro_news(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    query = f"Federal Reserve interest rate US economy market {state['date']}"
    print(f"[nasdaq] search_macro query: {query}")
    results = await _ddg_search(query)
    print(f"[nasdaq] search_macro: {time.perf_counter() - t0:.2f}s → {len(results)} results")
    return {"raw_articles": results}


async def search_earnings(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    query = f"tech stocks earnings results premarket today {state['date']}"
    print(f"[nasdaq] search_earnings query: {query}")
    results = await _ddg_search(query)
    print(f"[nasdaq] search_earnings: {time.perf_counter() - t0:.2f}s → {len(results)} results")
    return {"raw_articles": results}


async def search_futures(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    query = f"QQQ Nasdaq futures premarket market open {state['date']}"
    print(f"[nasdaq] search_futures query: {query}")
    results = await _ddg_search(query)
    print(f"[nasdaq] search_futures: {time.perf_counter() - t0:.2f}s → {len(results)} results")
    return {"raw_articles": results}


# ─── 股票涨跌幅节点（yfinance，在独立线程中运行） ────────────────────────────

def _fetch_one(symbol: str) -> tuple | None:
    """单支股票盘前涨跌幅，yfinance fast_info 在盘前时段 last_price 即为盘前价。"""
    try:
        fi = yf.Ticker(symbol).fast_info
        last = fi.last_price
        prev = fi.regular_market_previous_close
        if last and prev and prev > 0:
            return (symbol, last, (last - prev) / prev * 100)
    except Exception:
        pass
    return None


def _fetch_movers_sync() -> str:
    """用线程池并发拉取全部 Nasdaq 100 成分股盘前价，返回格式化的涨跌榜文本。"""
    print(f"[nasdaq] fetch_stock_movers: fetching {len(NASDAQ100_TICKERS)} tickers ...")
    with ThreadPoolExecutor(max_workers=20) as ex:
        results = [r for r in ex.map(_fetch_one, NASDAQ100_TICKERS) if r]

    if not results:
        return "（盘前股票数据暂不可用）"

    results.sort(key=lambda x: x[2], reverse=True)
    gainers = results[:10]
    losers = results[-10:][::-1]

    lines = ["涨幅前十："]
    for sym, price, chg in gainers:
        lines.append(f"  {sym} ${price:.2f} ({chg:+.2f}%)")
    lines.append("跌幅前十：")
    for sym, price, chg in losers:
        lines.append(f"  {sym} ${price:.2f} ({chg:+.2f}%)")

    return "\n".join(lines)


async def fetch_stock_movers(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    movers = await asyncio.to_thread(_fetch_movers_sync)
    print(f"[nasdaq] fetch_stock_movers: {time.perf_counter() - t0:.2f}s")
    print(f"[nasdaq] stock_movers:\n{movers}")
    return {"stock_movers": movers}


# ─── 报告生成节点 ──────────────────────────────────────────────────────────────

_REPORT_SYSTEM = """你是专业美股分析师，擅长整理纳斯达克100盘前动态。
根据以下新闻摘要和实时股票行情，用中文生成一份简洁的盘前日报。

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

📈 盘前涨跌幅前十（数据来自实时行情）
{stock_movers}

⚠️ 风险提示
（1-2点关键风险或待关注事件）

来源：Reuters/CNBC/MarketWatch"""


async def generate_report(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    articles = state["raw_articles"]
    stock_movers = state.get("stock_movers", "（数据加载中）")

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

    prompt = _REPORT_SYSTEM.replace("{date}", state["date"]).replace("{stock_movers}", stock_movers)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"今日盘前新闻摘要如下：\n\n{context}"),
    ]

    report = await (llm | StrOutputParser()).ainvoke(messages)

    if len(report) > MAX_REPORT_CHARS:
        report = report[: MAX_REPORT_CHARS - 3] + "..."

    print(f"[nasdaq] generate_report: {time.perf_counter() - t0:.2f}s → {len(report)} chars")
    return {"report_content": report}


# ─── 发送通知节点 ──────────────────────────────────────────────────────────────

async def send_notification(state: NasdaqReportState) -> dict:
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
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(NOTIFY_URL, json=payload)
        status = f"ok:{resp.status_code}"
        print(f"[nasdaq] send_notification: {time.perf_counter() - t0:.2f}s → {status} | body={resp.text[:120]}")
    except Exception as e:
        status = f"error:{e}"
        print(f"[nasdaq] send_notification failed: {e}")

    return {"send_status": status}
