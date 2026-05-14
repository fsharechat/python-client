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

from config import ANTHROPIC_API_KEY, GENERATE_MODEL, NOTIFY_MOBILES
from nasdaq_agent.state import NasdaqReportState
from nasdaq_agent.tickers import NASDAQ100_TICKERS, NASDAQ100_SECTOR_MAP, SECTOR_ORDER

NOTIFY_URL = "https://backend-http.fsharechat.cn/imopenapi/pushNotificationByMobile"

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

# Yahoo Finance 非官方 JSON 接口（期货/个股盘前盘后，国际网络访问）
_YF_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
_YF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
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

def _sector_movers_table(all_results: list[tuple], price_label: str = "价格") -> str:
    """
    all_results: [(sym, price, chg), ...] 全量行情数据
    price_label: 价格列标题（盘前报告传"盘前价"，盘后报告传"收盘价"）
    按 SECTOR_ORDER 分组，每板块展示实际上涨（chg>0）前10和实际下跌（chg<0）前10。
    """
    buckets: dict[str, list[tuple]] = {s: [] for s in SECTOR_ORDER}
    for sym, price, chg in all_results:
        sector = NASDAQ100_SECTOR_MAP.get(sym)
        if sector is None:
            print(f"[nasdaq] warning: {sym} not in NASDAQ100_SECTOR_MAP, skipping")
            continue
        buckets[sector].append((sym, price, chg))

    def _table(rows: list[tuple]) -> str:
        header = f"| 代码 | {price_label} | 涨跌幅 |\n|:---|---:|---:|"
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
        gainers = sorted([s for s in stocks if s[2] > 0], key=lambda x: x[2], reverse=True)[:10]
        losers = sorted([s for s in stocks if s[2] < 0], key=lambda x: x[2])[:10]
        if gainers:
            parts.append(f"▎{sector}（涨幅前{len(gainers)}）\n\n{_table(gainers)}")
        if losers:
            parts.append(f"▎{sector}（跌幅前{len(losers)}）\n\n{_table(losers)}")

    return "\n\n".join(parts)


def _top_movers_summary(results: list[tuple]) -> str:
    """全市场涨跌幅前5摘要，注入 LLM prompt 供交叉验证。"""
    if not results:
        return ""
    gainers = sorted([r for r in results if r[2] > 0], key=lambda x: x[2], reverse=True)[:5]
    losers = sorted([r for r in results if r[2] < 0], key=lambda x: x[2])[:5]
    parts = ["【实际行情数据（请在叙述中引用具体数字）】"]
    if gainers:
        parts.append("涨幅前5：" + "、".join(f"{sym} {chg:+.2f}%（${price:.2f}）" for sym, price, chg in gainers))
    if losers:
        parts.append("跌幅前5：" + "、".join(f"{sym} {chg:+.2f}%（${price:.2f}）" for sym, price, chg in losers))
    return "\n".join(parts)


async def _fetch_yahoo_index_data(report_type: str) -> str:
    """
    Yahoo Finance 获取指数/期货行情。
    盘前：NQ=F（纳指100期货）+ ES=F（标普500期货），近24h连续交易，反映盘前方向
    盘后：^NDX（纳斯达克100指数） + ^GSPC（标普500指数），收盘实际值
    """
    if report_type == "premarket":
        symbols = "NQ=F,ES=F"
        name_map = {"NQ=F": "纳指100期货(NQ)", "ES=F": "标普500期货(ES)"}
        order = ["NQ=F", "ES=F"]
        val_label = "盘前价"
    else:
        symbols = "^NDX,^GSPC"
        name_map = {"^NDX": "纳斯达克100指数", "^GSPC": "标普500指数"}
        order = ["^NDX", "^GSPC"]
        val_label = "收盘价"

    try:
        async with httpx.AsyncClient(headers=_YF_HEADERS, timeout=10, trust_env=False) as client:
            resp = await client.get(_YF_QUOTE_URL, params={"symbols": symbols})
        quotes = resp.json().get("quoteResponse", {}).get("result", [])

        row_map: dict[str, str] = {}
        for q in quotes:
            sym = q.get("symbol", "")
            price = q.get("regularMarketPrice")
            chg_pct = q.get("regularMarketChangePercent")
            chg_pts = q.get("regularMarketChange")
            if sym not in name_map or price is None or chg_pct is None:
                continue
            sign = "+" if chg_pct >= 0 else ""
            pts_sign = "+" if (chg_pts or 0) >= 0 else ""
            row_map[sym] = f"| {name_map[sym]} | {price:,.2f} | {pts_sign}{chg_pts:.2f} | {sign}{chg_pct:.2f}% |"

        rows = [row_map[s] for s in order if s in row_map]
        if not rows:
            return ""
        col_label = "期货价" if report_type == "premarket" else val_label
        header = f"| 指数/期货 | {col_label} | 涨跌点 | 涨跌幅 |\n|:---|---:|---:|---:|"
        print(f"[nasdaq] Yahoo index ({report_type}): {len(rows)} rows fetched")
        return "📊 主要指数\n\n" + header + "\n" + "\n".join(rows)

    except Exception as e:
        print(f"[nasdaq] Yahoo index fetch failed: {e}")
        return ""


async def _fetch_em_index_data(report_type: str) -> str:
    """东方财富指数兜底：盘前/盘后均用 QQQ/SPY ETF（对应关系明确，规避综合指数混淆）。"""
    secids = "105.QQQ,106.SPY"
    name_map = {"QQQ": "纳指100ETF(QQQ)", "SPY": "标普500ETF(SPY)"}
    order = ["QQQ", "SPY"]
    val_label = "盘前价" if report_type == "premarket" else "收盘价"

    params = {"fltt": "2", "invt": "2", "fields": "f12,f2,f3,f4", "secids": secids}
    try:
        async with httpx.AsyncClient(headers=_EM_HEADERS, timeout=10, trust_env=False, follow_redirects=True) as client:
            resp = await client.get(_EM_URL, params=params)
        diff = resp.json().get("data", {}).get("diff", [])
        items = diff if isinstance(diff, list) else list(diff.values())

        row_map: dict[str, str] = {}
        for item in items:
            code = item.get("f12", "")
            val, chg_pct, chg_pts = item.get("f2"), item.get("f3"), item.get("f4")
            if code not in name_map or val in (None, "-") or chg_pct in (None, "-"):
                continue
            try:
                val_f, pct_f = float(val), float(chg_pct)
                pts_f = float(chg_pts) if chg_pts not in (None, "-") else 0.0
                sign = "+" if pct_f >= 0 else ""
                row_map[code] = f"| {name_map[code]} | {val_f:,.2f} | {sign}{pts_f:.2f} | {sign}{pct_f:.2f}% |"
            except (ValueError, TypeError):
                pass

        rows = [row_map[c] for c in order if c in row_map]
        if not rows:
            return ""
        header = f"| 指数/ETF | {val_label} | 涨跌点 | 涨跌幅 |\n|:---|---:|---:|---:|"
        return "📊 主要指数\n\n" + header + "\n" + "\n".join(rows)
    except Exception as e:
        print(f"[nasdaq] EM index fetch failed: {e}")
        return ""


async def _fetch_index_data(report_type: str) -> str:
    """指数行情：Yahoo Finance（NQ=F/ES=F）为主，东方财富为兜底。"""
    result = await _fetch_yahoo_index_data(report_type)
    if result:
        return result
    print("[nasdaq] Yahoo index failed, falling back to East Money")
    return await _fetch_em_index_data(report_type)


async def _fetch_yahoo_stock_data(report_type: str) -> dict[str, tuple]:
    """
    Yahoo Finance 批量获取 NASDAQ100 成分股行情。
    盘前取 preMarketPrice/preMarketChangePercent，
    盘后取 postMarketPrice/postMarketChangePercent，
    无扩展盘数据时退回 regularMarket 字段。
    返回 {sym: (price, chg_pct)}，失败返回空字典。
    """
    symbols = ",".join(NASDAQ100_TICKERS)
    try:
        async with httpx.AsyncClient(headers=_YF_HEADERS, timeout=20, trust_env=False) as client:
            resp = await client.get(_YF_QUOTE_URL, params={"symbols": symbols})

        result: dict[str, tuple] = {}
        for q in resp.json().get("quoteResponse", {}).get("result", []):
            sym = q.get("symbol", "")
            if not sym:
                continue
            reg_price = q.get("regularMarketPrice")
            reg_chg = q.get("regularMarketChangePercent", 0.0)
            if report_type == "premarket":
                price = q.get("preMarketPrice") or reg_price
                chg = q.get("preMarketChangePercent") or reg_chg
            elif report_type == "afterhours":
                price = q.get("postMarketPrice") or reg_price
                chg = q.get("postMarketChangePercent") or reg_chg
            else:
                price, chg = reg_price, reg_chg
            if price is not None:
                result[sym] = (float(price), float(chg or 0))

        print(f"[nasdaq] Yahoo stocks: {len(result)} tickers fetched")
        return result
    except Exception as e:
        print(f"[nasdaq] Yahoo stock fetch failed: {e}")
        return {}


def _merge_stock_results(
    em_results: list[tuple],
    yf_results: dict[str, tuple],
) -> list[tuple]:
    """
    合并东方财富（主）和 Yahoo Finance（补）行情数据。
    - 东方财富有数据：使用东方财富，并与 Yahoo 交叉验证（差异>2% 打印告警）
    - 东方财富缺失：用 Yahoo Finance 补充
    - 两者都缺失：跳过
    """
    em_map = {sym: (price, chg) for sym, price, chg in em_results}
    merged: list[tuple] = []

    for sym in NASDAQ100_TICKERS:
        if sym in em_map:
            em_price, em_chg = em_map[sym]
            if sym in yf_results:
                yf_price, yf_chg = yf_results[sym]
                if abs(em_chg - yf_chg) > 2.0:
                    print(f"[nasdaq] ⚠ 数据差异 {sym}: 东方财富 {em_chg:+.2f}% vs Yahoo {yf_chg:+.2f}%")
            merged.append((sym, em_price, em_chg))
        elif sym in yf_results:
            yf_price, yf_chg = yf_results[sym]
            print(f"[nasdaq] {sym}: 东方财富无数据，Yahoo Finance 补充 {yf_chg:+.2f}%")
            merged.append((sym, yf_price, yf_chg))

    return merged


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
    import asyncio
    import traceback
    report_type = state.get("report_type") or "premarket"
    t0 = time.perf_counter()
    print(f"[nasdaq] fetch_stock_movers: [{report_type}] 并发拉取 东方财富+Yahoo Finance ...")

    # 三路并发：指数（Yahoo主/EM备）、东方财富个股、Yahoo个股
    index_task = asyncio.create_task(_fetch_index_data(report_type))
    yf_task = asyncio.create_task(_fetch_yahoo_stock_data(report_type))

    try:
        em_results, em_label = await _fetch_stock_data()
    except Exception as e:
        print(f"[nasdaq] 东方财富个股失败: {e}")
        print(traceback.format_exc())
        em_results = []

    index_summary = await index_task
    yf_results = await yf_task

    merged = _merge_stock_results(em_results, yf_results)
    elapsed = time.perf_counter() - t0
    print(f"[nasdaq] fetch_stock_movers: {elapsed:.2f}s | EM={len(em_results)} YF={len(yf_results)} 合并={len(merged)}")
    return {"index_summary": index_summary, "stock_results": merged}


# ─── 报告生成节点 ──────────────────────────────────────────────────────────────

_PREMARKET_SYSTEM = """你是专业美股分析师，擅长整理纳斯达克100盘前动态。
根据提供的【实际行情数据】和新闻摘要，用中文生成盘前日报的叙述部分。

严格要求：
- 总字数不超过800字
- 【实际行情数据】中的涨跌幅数字必须在市场概况中引用，不得与实际数据矛盾
- 只生成叙述部分，不要包含股票数据表格（表格将单独附加）
- 使用以下固定格式，不要偏离

输出格式：
【纳斯达克100盘前日报】{date}

📊 市场概况
（2-3句：纳指期货/QQQ盘前方向、整体情绪，引用实际涨跌幅数字）

🔥 盘前三大热点
1.
2.
3.

⚠️ 风险提示
（1-2点关键风险或待关注事件）"""

_AFTERHOURS_SYSTEM = """你是专业美股分析师，擅长整理纳斯达克100盘后动态。
根据提供的【实际行情数据】和新闻摘要，用中文生成盘后日报的叙述部分。

严格要求：
- 总字数不超过800字
- 【实际行情数据】中的涨跌幅数字必须在收盘概况中引用，不得与实际数据矛盾
- 只生成叙述部分，不要包含股票数据表格（表格将单独附加）
- 使用以下固定格式，不要偏离

输出格式：
【纳斯达克100盘后日报】{date}

📊 收盘概况
（2-3句：纳指收盘涨跌幅、当日整体走势，引用实际涨跌幅数字）

🔥 盘后三大焦点
1.（财报结果或重大公告）
2.（盘后异动个股及原因）
3.（明日前瞻或关键数据）

⚠️ 关注事项
（1-2点：明日待关注风险或催化剂）"""


async def _generate_narrative(state: NasdaqReportState, system_prompt: str, movers_summary: str = "") -> str:
    """调用 LLM 生成叙述部分（不含股票表格），movers_summary 注入实际行情供交叉验证。"""
    t0 = time.perf_counter()
    articles = state["raw_articles"]

    context_parts = [
        f"标题：{a['title']}\n摘要：{a['body'][:200]}"
        for a in articles[:20]
        if a.get("title") or a.get("body")
    ]
    news_context = "\n\n".join(context_parts) if context_parts else "（暂无搜索结果）"

    human_content = (
        f"{movers_summary}\n\n今日新闻摘要如下：\n\n{news_context}"
        if movers_summary else
        f"今日新闻摘要如下：\n\n{news_context}"
    )

    llm = ChatAnthropic(
        model=GENERATE_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=900,
        max_retries=3,
    )

    try:
        narrative = await (llm | StrOutputParser()).ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ])
    except Exception as e:
        print(f"[nasdaq] _generate_narrative failed: {e}")
        return "（今日叙述生成失败，请稍后重试）"

    print(f"[nasdaq] _generate_narrative: {time.perf_counter() - t0:.2f}s → {len(narrative)} chars")
    return narrative


async def generate_report(state: NasdaqReportState) -> dict:
    """盘前日报：LLM生成叙述（含实际行情交叉验证）+ 程序拼接板块涨跌表格。"""
    t0 = time.perf_counter()
    stock_results: list = state.get("stock_results") or []
    movers_summary = _top_movers_summary(stock_results)
    prompt = _PREMARKET_SYSTEM.replace("{date}", state.get("date") or "")
    narrative = await _generate_narrative(state, prompt, movers_summary)

    index_summary = state.get("index_summary", "")
    movers_table = _sector_movers_table(stock_results, price_label="盘前价") if stock_results else "（股票数据暂不可用）"

    sections = [narrative, "---"]
    if index_summary:
        sections += [index_summary, "---"]
    sections += ["📈 板块涨跌榜", movers_table, "来源：Reuters/CNBC/MarketWatch"]
    report = "\n\n".join(sections)

    print(f"[nasdaq] generate_report: {time.perf_counter() - t0:.2f}s → {len(report)} chars total")
    return {"report_content": report}


async def generate_afterhours_report(state: NasdaqReportState) -> dict:
    """盘后日报：LLM生成叙述（含实际行情交叉验证）+ 程序拼接板块涨跌表格。"""
    t0 = time.perf_counter()
    stock_results: list = state.get("stock_results") or []
    movers_summary = _top_movers_summary(stock_results)
    prompt = _AFTERHOURS_SYSTEM.replace("{date}", state.get("date") or "")
    narrative = await _generate_narrative(state, prompt, movers_summary)

    index_summary = state.get("index_summary", "")
    movers_table = _sector_movers_table(stock_results, price_label="收盘价") if stock_results else "（股票数据暂不可用）"

    sections = [narrative, "---"]
    if index_summary:
        sections += [index_summary, "---"]
    sections += ["📈 板块涨跌榜", movers_table, "来源：Reuters/CNBC/MarketWatch"]
    report = "\n\n".join(sections)

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
