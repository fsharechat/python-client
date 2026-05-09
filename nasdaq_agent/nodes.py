"""
Nasdaq 100 日报 Agent 各节点实现（全 async）。

搜索使用 httpx 异步抓取 DuckDuckGo HTML。
股票行情使用东方财富 push2 API（国内服务器直连稳定，无需认证，单次批量拉取）。
盘前：4路搜索 + 行情 → generate_report → send_notification
盘后：4路盘后搜索 + 行情 → generate_afterhours_report → send_notification
"""

import time

import httpx
from bs4 import BeautifulSoup
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from config import ANTHROPIC_API_KEY, GENERATE_MODEL
from nasdaq_agent.state import NasdaqReportState
from nasdaq_agent.tickers import NASDAQ100_TICKERS, NASDAQ100_SECTOR_MAP, SECTOR_ORDER

NOTIFY_URL = "https://backend-http.fsharechat.cn/imopenapi/pushNotificationByMobile"
NOTIFY_MOBILES = [13900000001]

_DDG_URL = "https://html.duckduckgo.com/html/"
_DDG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# 东方财富 push2 批量行情接口（纳斯达克 secid 前缀 105）
_EM_URL = "https://push2.eastmoney.com/api/qt/ulist.np/get"
_EM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.eastmoney.com/",
}


# ─── 搜索工具（纯 async，无线程） ─────────────────────────────────────────────

async def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    try:
        async with httpx.AsyncClient(headers=_DDG_HEADERS, timeout=20, follow_redirects=True) as client:
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


# ─── 并行搜索节点 ──────────────────────────────────────────────────────────────

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


async def search_earnings_results(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    query = f"tech stocks earnings results after close beat miss {state['date']}"
    print(f"[nasdaq] search_earnings_results query: {query}")
    results = await _ddg_search(query)
    print(f"[nasdaq] search_earnings_results: {time.perf_counter() - t0:.2f}s → {len(results)} results")
    return {"raw_articles": results}


async def search_afterhours_movers(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    query = f"Nasdaq after hours movers stock gains losses {state['date']}"
    print(f"[nasdaq] search_afterhours_movers query: {query}")
    results = await _ddg_search(query)
    print(f"[nasdaq] search_afterhours_movers: {time.perf_counter() - t0:.2f}s → {len(results)} results")
    return {"raw_articles": results}


async def search_closing_summary(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    query = f"Nasdaq 100 stock market closing recap today {state['date']}"
    print(f"[nasdaq] search_closing_summary query: {query}")
    results = await _ddg_search(query)
    print(f"[nasdaq] search_closing_summary: {time.perf_counter() - t0:.2f}s → {len(results)} results")
    return {"raw_articles": results}


async def search_tomorrow_preview(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    query = f"stock market outlook tomorrow economic calendar earnings preview {state['date']}"
    print(f"[nasdaq] search_tomorrow_preview query: {query}")
    results = await _ddg_search(query)
    print(f"[nasdaq] search_tomorrow_preview: {time.perf_counter() - t0:.2f}s → {len(results)} results")
    return {"raw_articles": results}


# ─── 股票涨跌幅节点（东方财富 push2 API，国内稳定访问） ──────────────────────

def _sector_movers_table(all_results: list[tuple]) -> str:
    """
    all_results: [(sym, price, chg), ...] 全量行情数据
    按 SECTOR_ORDER 分组，每板块展示涨幅前N和跌幅前N（N=min(10, 板块内有数据的股票数）。
    """
    # 按板块分桶
    buckets: dict[str, list[tuple]] = {s: [] for s in SECTOR_ORDER}
    for sym, price, chg in all_results:
        sector = NASDAQ100_SECTOR_MAP.get(sym)
        if sector is None:
            print(f"[nasdaq] warning: {sym} not in NASDAQ100_SECTOR_MAP, skipping")
            continue
        buckets[sector].append((sym, price, chg))

    def _table(rows: list[tuple]) -> str:
        header = "| 代码 | 价格 | 涨跌幅 |\n|:---|---:|---:|"
        body = "\n".join(
            f"| {sym} | ${price:.2f} | {chg:+.2f}% |"
            for sym, price, chg in rows
        )
        return header + "\n" + body

    parts = []
    for sector in SECTOR_ORDER:
        stocks = buckets.get(sector, [])
        if not stocks:
            continue
        stocks_sorted = sorted(stocks, key=lambda x: x[2], reverse=True)
        total = len(stocks_sorted)
        if total <= 10:
            # small sector — show one full sorted table
            parts.append(f"▎{sector}（全板块排名）\n\n{_table(stocks_sorted)}")
        else:
            # ensure non-overlapping: each side gets at most half the stocks, capped at 10
            n = min(10, total // 2)
            gainers = stocks_sorted[:n]
            losers = stocks_sorted[-n:][::-1]
            parts.append(f"▎{sector}（涨幅前{n}）\n\n{_table(gainers)}")
            parts.append(f"▎{sector}（跌幅前{n}）\n\n{_table(losers)}")

    return "\n\n".join(parts)


async def _fetch_stock_data() -> tuple[list[tuple], str]:
    """
    调用东方财富 push2 批量行情接口，单次请求拉取全部 Nasdaq 100 成分股。
    纳斯达克 secid 前缀为 105，f2=最新价，f3=涨跌幅%。
    盘前时段 f2 即为盘前价，返回 (results[(symbol, price, change_pct)], label)。
    """
    secids = ",".join(f"105.{sym}" for sym in NASDAQ100_TICKERS)
    params = {
        "fltt": "2",   # 价格保留原始精度
        "invt": "2",
        "fields": "f12,f2,f3",  # f12=代码, f2=最新价, f3=涨跌幅%
        "secids": secids,
    }

    async with httpx.AsyncClient(headers=_EM_HEADERS, timeout=20, trust_env=False, follow_redirects=True) as client:
        resp = await client.get(_EM_URL, params=params)

    print(f"[nasdaq] East Money HTTP {resp.status_code}, body[:200]: {resp.text[:200]}")

    if resp.status_code != 200:
        raise RuntimeError(f"East Money API returned HTTP {resp.status_code}")

    raw = resp.json()
    print(f"[nasdaq] East Money data keys: {list(raw.get('data', {}).keys())}")
    diff = raw.get("data", {}).get("diff", [])
    items = diff if isinstance(diff, list) else list(diff.values())
    print(f"[nasdaq] East Money diff items count: {len(items)}")

    results = []
    for item in items:
        code = item.get("f12", "")
        price = item.get("f2")
        chg = item.get("f3")
        if code and price not in (None, "-", 0) and chg not in (None, "-"):
            try:
                results.append((code, float(price), float(chg)))
            except (ValueError, TypeError):
                pass

    return results, "实时行情"


async def fetch_stock_movers(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    print(f"[nasdaq] fetch_stock_movers: fetching {len(NASDAQ100_TICKERS)} tickers via East Money ...")

    try:
        results, label = await _fetch_stock_data()
    except Exception as e:
        import traceback
        print(f"[nasdaq] fetch_stock_movers failed: {e}")
        print(traceback.format_exc())
        return {"stock_movers": "（股票数据获取失败）"}

    if not results:
        return {"stock_movers": "（股票数据暂不可用）"}

    movers = _sector_movers_table(results)
    elapsed = time.perf_counter() - t0
    print(f"[nasdaq] fetch_stock_movers: {elapsed:.2f}s → {len(results)} tickers ({label})")
    return {"stock_movers": movers}


# ─── 报告生成节点 ──────────────────────────────────────────────────────────────

_PREMARKET_SYSTEM = """你是专业美股分析师，擅长整理纳斯达克100盘前动态。
根据以下新闻摘要，用中文生成盘前日报的叙述部分。

严格要求：
- 总字数不超过800字
- 信息客观准确，不要编造数据
- 只生成叙述部分，不要包含股票数据表格（表格将单独附加）
- 使用以下固定格式，不要偏离

输出格式：
【纳斯达克100盘前日报】{date}

📊 市场概况
（2-3句：纳指期货方向、整体情绪）

🔥 盘前三大热点
1.
2.
3.

⚠️ 风险提示
（1-2点关键风险或待关注事件）"""

_AFTERHOURS_SYSTEM = """你是专业美股分析师，擅长整理纳斯达克100盘后动态。
根据以下新闻摘要，用中文生成盘后日报的叙述部分。

严格要求：
- 总字数不超过800字
- 信息客观准确，不要编造数据
- 只生成叙述部分，不要包含股票数据表格（表格将单独附加）
- 使用以下固定格式，不要偏离

输出格式：
【纳斯达克100盘后日报】{date}

📊 收盘概况
（2-3句：纳指收盘涨跌幅、当日整体走势）

🔥 盘后三大焦点
1.（财报结果或重大公告）
2.（盘后异动个股及原因）
3.（明日前瞻或关键数据）

⚠️ 关注事项
（1-2点：明日待关注风险或催化剂）"""


async def _generate_narrative(state: NasdaqReportState, system_prompt: str) -> str:
    """调用 LLM 生成叙述部分（不含股票表格），返回叙述字符串。"""
    t0 = time.perf_counter()
    articles = state["raw_articles"]

    context_parts = [
        f"标题：{a['title']}\n摘要：{a['body'][:200]}"
        for a in articles[:20]
        if a.get("title") or a.get("body")
    ]
    context = "\n\n".join(context_parts) if context_parts else "（暂无搜索结果）"

    llm = ChatAnthropic(
        model=GENERATE_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=900,
        max_retries=3,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"今日新闻摘要如下：\n\n{context}"),
    ]

    try:
        narrative = await (llm | StrOutputParser()).ainvoke(messages)
    except Exception as e:
        print(f"[nasdaq] _generate_narrative failed: {e}")
        return "（今日叙述生成失败，请稍后重试）"

    print(f"[nasdaq] _generate_narrative: {time.perf_counter() - t0:.2f}s → {len(narrative)} chars")
    return narrative


async def generate_report(state: NasdaqReportState) -> dict:
    """盘前日报：LLM生成叙述 + 程序拼接板块涨跌表格。"""
    t0 = time.perf_counter()
    prompt = _PREMARKET_SYSTEM.replace("{date}", state.get("date") or "")
    narrative = await _generate_narrative(state, prompt)
    stock_movers = state.get("stock_movers", "（数据加载中）")

    report = (
        f"{narrative}\n\n"
        f"---\n\n"
        f"📈 板块涨跌榜\n\n"
        f"{stock_movers}\n\n"
        f"来源：Reuters/CNBC/MarketWatch"
    )

    print(f"[nasdaq] generate_report: {time.perf_counter() - t0:.2f}s → {len(report)} chars total")
    return {"report_content": report}


async def generate_afterhours_report(state: NasdaqReportState) -> dict:
    """盘后日报：LLM生成叙述 + 程序拼接板块涨跌表格。"""
    t0 = time.perf_counter()
    prompt = _AFTERHOURS_SYSTEM.replace("{date}", state.get("date") or "")
    narrative = await _generate_narrative(state, prompt)
    stock_movers = state.get("stock_movers", "（数据加载中）")

    report = (
        f"{narrative}\n\n"
        f"---\n\n"
        f"📈 板块涨跌榜\n\n"
        f"{stock_movers}\n\n"
        f"来源：Reuters/CNBC/MarketWatch"
    )

    print(f"[nasdaq] generate_afterhours_report: {time.perf_counter() - t0:.2f}s → {len(report)} chars total")
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
