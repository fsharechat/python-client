"""
Nasdaq 100 日报 Agent 各节点实现（全 async）。

搜索使用 httpx 异步抓取 DuckDuckGo HTML。
股票行情使用东方财富 push2 API（国内服务器直连稳定，无需认证，单次批量拉取）。
盘前：4路搜索 + 行情 → generate_report → send_notification
盘后：4路盘后搜索 + 行情 → generate_afterhours_report → send_notification
盘中：3路开盘搜索 + 行情 → generate_intraday_report → send_notification
"""

import json
import time

import httpx
from bs4 import BeautifulSoup
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from config import ANTHROPIC_API_KEY, FINNHUB_API_KEY, GENERATE_MODEL, NOTIFY_MOBILES
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

# Finnhub 实时行情接口（盘前/盘后指数与七姐妹）
_FH_QUOTE_URL = "https://finnhub.io/api/v1/quote"

# 美股七姐妹（盘前日报单独展示）
MAG7 = ["AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA"]


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
        print(f"[nasdaq] DDG raw results: {json.dumps(results, ensure_ascii=False)}")
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


async def search_opening_movers(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    query = f"Nasdaq stocks gap up gap down market open {state['date']}"
    print(f"[nasdaq] search_opening_movers query: {query}")
    results = await _ddg_search(query)
    print(f"[nasdaq] search_opening_movers: {time.perf_counter() - t0:.2f}s → {len(results)} results")
    return {"raw_articles": results}


async def search_morning_economics(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    query = f"US economic data release morning CPI jobs {state['date']}"
    print(f"[nasdaq] search_morning_economics query: {query}")
    results = await _ddg_search(query)
    print(f"[nasdaq] search_morning_economics: {time.perf_counter() - t0:.2f}s → {len(results)} results")
    return {"raw_articles": results}


async def search_opening_news(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    query = f"stock market open recap early trading {state['date']}"
    print(f"[nasdaq] search_opening_news query: {query}")
    results = await _ddg_search(query)
    print(f"[nasdaq] search_opening_news: {time.perf_counter() - t0:.2f}s → {len(results)} results")
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


def _build_full_market_context(yf_data: dict, stock_results: list) -> str:
    """
    汇总 yfinance 盘前数据 + 东方财富涨跌幅，构建注入 LLM 的完整行情上下文。
    yf_data: {sym: {"price", "chg_pct", "chg_pts"}}
    stock_results: [(sym, price, chg_pct), ...]  来自东方财富批量接口
    """
    parts = ["【实际行情数据（请在叙述中引用具体数字）】"]

    # 主要指数盘前
    name_map = {"QQQ": "纳指100ETF(QQQ)", "SPY": "标普500ETF(SPY)", "RSP": "标普500等权ETF(RSP)"}
    index_lines = []
    for sym in ["QQQ", "SPY", "RSP"]:
        if sym in yf_data:
            q = yf_data[sym]
            s = "+" if q["chg_pct"] >= 0 else ""
            index_lines.append(f"  {name_map[sym]}: ${q['price']:.2f} {s}{q['chg_pct']:.4f}%")
    if index_lines:
        parts.append("📊 主要指数（盘前）\n" + "\n".join(index_lines))

    # 七姐妹盘前
    mag7_lines = []
    for sym in MAG7:
        if sym in yf_data:
            q = yf_data[sym]
            s = "+" if q["chg_pct"] >= 0 else ""
            mag7_lines.append(f"  {sym}: ${q['price']:.2f} {s}{q['chg_pct']:.4f}%")
    if mag7_lines:
        parts.append("💎 七姐妹盘前\n" + "\n".join(mag7_lines))

    # 纳斯达克100涨跌幅前5（东方财富）
    if stock_results:
        gainers = sorted([r for r in stock_results if r[2] > 0], key=lambda x: x[2], reverse=True)[:5]
        losers  = sorted([r for r in stock_results if r[2] < 0], key=lambda x: x[2])[:5]
        if gainers:
            parts.append("涨幅前5：" + "、".join(f"{sym} {chg:+.2f}%（${price:.2f}）" for sym, price, chg in gainers))
        if losers:
            parts.append("跌幅前5：" + "、".join(f"{sym} {chg:+.2f}%（${price:.2f}）" for sym, price, chg in losers))

    return "\n\n".join(parts)


async def _fetch_finnhub_quotes(symbols: list[str]) -> dict[str, dict]:
    """
    Finnhub 顺序查询行情，1次/秒，避免触发限速。
    返回 {sym: {"price": float, "chg_pct": float, "chg_pts": float}}。
    """
    import asyncio
    result = {}
    async with httpx.AsyncClient(timeout=10) as client:
        for i, sym in enumerate(symbols):
            try:
                resp = await client.get(_FH_QUOTE_URL, params={"symbol": sym, "token": FINNHUB_API_KEY})
                d = resp.json()
                print(f"[nasdaq] Finnhub {sym} raw: {json.dumps(d)}")
                price = d.get("c")
                if price and float(price) > 0:
                    result[sym] = {
                        "price": float(price),
                        "chg_pct": float(d.get("dp") or 0),
                        "chg_pts": float(d.get("d") or 0),
                    }
            except Exception as e:
                print(f"[nasdaq] Finnhub {sym}: {e}")
            if i < len(symbols) - 1:
                await asyncio.sleep(1)
    print(f"[nasdaq] Finnhub: {len(result)}/{len(symbols)} symbols fetched")
    return result


async def _fetch_em_index_data(report_type: str) -> str:
    """东方财富指数兜底：盘前/盘后均用 QQQ/SPY ETF（对应关系明确，规避综合指数混淆）。"""
    secids = "105.QQQ,106.SPY,106.RSP"
    name_map = {"QQQ": "纳指100ETF(QQQ)", "SPY": "标普500ETF(SPY)", "RSP": "标普500等权ETF(RSP)"}
    order = ["QQQ", "SPY", "RSP"]
    val_label = "盘前价" if report_type == "premarket" else "收盘价"

    params = {"fltt": "2", "invt": "2", "fields": "f12,f2,f3,f4", "secids": secids}
    try:
        async with httpx.AsyncClient(headers=_EM_HEADERS, timeout=10, trust_env=False, follow_redirects=True) as client:
            resp = await client.get(_EM_URL, params=params)
        raw = resp.json()
        print(f"[nasdaq] EM index raw: {json.dumps(raw, ensure_ascii=False)}")
        diff = raw.get("data", {}).get("diff", [])
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


def _fetch_yf_premarket_sync(symbols: list[str]) -> dict[str, dict]:
    """
    同步顺序获取 yfinance 盘前行情（避免 curl_cffi 多线程 TLS 问题）。
    preMarketChangePercent 已是百分比格式（1.38 = 1.38%）。
    返回 {sym: {"price": float, "chg_pct": float, "chg_pts": float}}
    """
    import yfinance as yf
    result = {}
    for i, sym in enumerate(symbols):
        try:
            info = yf.Ticker(sym).info
            print(f"[nasdaq] yfinance {sym} raw: preMarketPrice={info.get('preMarketPrice')} "
                  f"preMarketChangePercent={info.get('preMarketChangePercent')} "
                  f"regularMarketPrice={info.get('regularMarketPrice')}")
            pre_p = info.get("preMarketPrice")
            reg_p = info.get("regularMarketPrice")
            pre_c = info.get("preMarketChangePercent")
            if pre_p and float(pre_p) > 0:
                price = float(pre_p)
                reg = float(reg_p or price)
                chg_pts = price - reg
                chg_pct = float(pre_c) if pre_c is not None else (chg_pts / reg * 100 if reg else 0)
                result[sym] = {"price": price, "chg_pct": chg_pct, "chg_pts": chg_pts}
        except Exception as e:
            print(f"[nasdaq] yfinance {sym} failed: {e}")
        if i < len(symbols) - 1:
            time.sleep(1)
    print(f"[nasdaq] yfinance premarket: {len(result)}/{len(symbols)} symbols fetched")
    return result


async def _fetch_yf_premarket(symbols: list[str]) -> dict[str, dict]:
    """异步包装：在单线程 executor 中顺序调用 yfinance。"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_yf_premarket_sync, symbols)


async def _fetch_premarket_index_individual() -> str:
    """盘前指数兜底：Finnhub（昨收盘价）→ 东方财富。"""
    name_map = {"QQQ": "纳指100ETF(QQQ)", "SPY": "标普500ETF(SPY)", "RSP": "标普500等权ETF(RSP)"}
    order = ["QQQ", "SPY", "RSP"]

    quotes = await _fetch_finnhub_quotes(order)
    rows = []
    for sym in order:
        if sym not in quotes:
            continue
        q = quotes[sym]
        s = "+" if q["chg_pct"] >= 0 else ""
        ps = "+" if q["chg_pts"] >= 0 else ""
        rows.append(f"| {name_map[sym]} | {q['price']:,.2f} | {ps}{q['chg_pts']:.2f} | {s}{q['chg_pct']:.2f}% |")

    if rows:
        print(f"[nasdaq] premarket index Finnhub: {len(rows)} rows")
        header = "| 指数/ETF | 盘前价 | 涨跌点 | 涨跌幅 |\n|:---|---:|---:|---:|"
        return "📊 主要指数\n\n" + header + "\n" + "\n".join(rows)

    print("[nasdaq] premarket index Finnhub failed, falling back to East Money")
    return await _fetch_em_index_data("premarket")


async def _fetch_premarket_mag7() -> list[tuple]:
    """通过 Finnhub 获取七姐妹盘前价，返回 [(sym, price, chg_pct), ...]。"""
    quotes = await _fetch_finnhub_quotes(MAG7)
    results = [(sym, quotes[sym]["price"], quotes[sym]["chg_pct"]) for sym in MAG7 if sym in quotes]
    print(f"[nasdaq] Mag7 Finnhub: {len(results)}/{len(MAG7)}")
    return results


def _mag7_table(results: list[tuple], price_label: str = "盘前价") -> str:
    """美股七姐妹价格表。"""
    if not results:
        return "（七姐妹数据暂不可用）"
    header = f"| 代码 | {price_label} | 涨跌幅 |\n|:---|---:|---:|"
    rows = [
        f"| {sym} | ${price:.2f} | {'+'if chg>=0 else ''}{chg:.2f}% |"
        for sym, price, chg in results
    ]
    return header + "\n" + "\n".join(rows)


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

    print(f"[nasdaq] East Money HTTP {resp.status_code}")

    if resp.status_code != 200:
        raise RuntimeError(f"East Money API returned HTTP {resp.status_code}")

    raw = resp.json()
    diff = raw.get("data", {}).get("diff", [])
    items = diff if isinstance(diff, list) else list(diff.values())
    print(f"[nasdaq] East Money stocks raw ({len(items)} items): {json.dumps(items, ensure_ascii=False)}")

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


def _build_fh_index_table(fh_data: dict, report_type: str) -> str:
    """从 Finnhub 行情数据构建 QQQ/SPY/RSP 指数表格字符串。"""
    name_map = {"QQQ": "纳指100ETF(QQQ)", "SPY": "标普500ETF(SPY)", "RSP": "标普500等权ETF(RSP)"}
    order = ["QQQ", "SPY", "RSP"]
    if report_type == "premarket":
        val_label = "盘前价"
    elif report_type == "intraday":
        val_label = "实时价"
    else:
        val_label = "收盘价"
    rows = []
    for sym in order:
        if sym not in fh_data:
            continue
        q = fh_data[sym]
        s = "+" if q["chg_pct"] >= 0 else ""
        ps = "+" if q["chg_pts"] >= 0 else ""
        rows.append(f"| {name_map[sym]} | {q['price']:,.2f} | {ps}{q['chg_pts']:.2f} | {s}{q['chg_pct']:.2f}% |")
    if not rows:
        return ""
    header = f"| 指数/ETF | {val_label} | 涨跌点 | 涨跌幅 |\n|:---|---:|---:|---:|"
    print(f"[nasdaq] Finnhub index ({report_type}): {len(rows)} rows")
    return "📊 主要指数\n\n" + header + "\n" + "\n".join(rows)


async def fetch_stock_movers(state: NasdaqReportState) -> dict:
    import asyncio
    import traceback
    report_type = state.get("report_type") or "premarket"
    t0 = time.perf_counter()
    print(f"[nasdaq] fetch_stock_movers: [{report_type}] 拉取行情...")

    if report_type in ("afterhours", "intraday"):
        # 并发：Finnhub（102只成分股 + QQQ/SPY/RSP）与 EM 批量同时发起
        fh_task = asyncio.create_task(_fetch_finnhub_quotes(NASDAQ100_TICKERS + ["QQQ", "SPY", "RSP"]))

    try:
        em_results, _ = await _fetch_stock_data()
    except Exception as e:
        print(f"[nasdaq] 东方财富个股失败: {e}")
        print(traceback.format_exc())
        em_results = []

    if report_type in ("afterhours", "intraday"):
        fh_data = await fh_task

        # 个股合并：Finnhub 主，EM 备
        em_map = {sym: (price, chg) for sym, price, chg in em_results}
        stock_results = []
        fh_count = 0
        for sym in NASDAQ100_TICKERS:
            if sym in fh_data:
                q = fh_data[sym]
                stock_results.append((sym, q["price"], q["chg_pct"]))
                fh_count += 1
            elif sym in em_map:
                stock_results.append((sym, em_map[sym][0], em_map[sym][1]))
        print(f"[nasdaq] {report_type} stocks: Finnhub={fh_count} EM_fallback={len(stock_results) - fh_count}")

        # 指数：Finnhub 主，EM 备
        index_summary = _build_fh_index_table(fh_data, report_type)
        if not index_summary:
            print("[nasdaq] Finnhub index empty, falling back to East Money")
            index_summary = await _fetch_em_index_data(report_type)
    else:
        stock_results = em_results
        index_summary = ""

    elapsed = time.perf_counter() - t0
    print(f"[nasdaq] fetch_stock_movers: {elapsed:.2f}s | stocks={len(stock_results if report_type in ('afterhours', 'intraday') else em_results)}")
    return {"index_summary": index_summary, "stock_results": stock_results if report_type in ("afterhours", "intraday") else em_results}


# ─── 报告生成节点 ──────────────────────────────────────────────────────────────

_PREMARKET_SYSTEM = """你是专业美股分析师，擅长整理纳斯达克100盘前动态。
根据提供的【实际行情数据】和新闻摘要，用中文生成盘前日报的叙述部分。

严格要求：
- 总字数不超过1500字
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
- 总字数不超过1500字
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

_INTRADAY_SYSTEM = """你是专业美股分析师，擅长整理纳斯达克100开盘动态。
根据提供的【实际行情数据】和新闻摘要，用中文生成开盘日报的叙述部分。

严格要求：
- 总字数不超过1500字
- 【实际行情数据】中的涨跌幅数字必须在开盘走势中引用，不得与实际数据矛盾
- 只生成叙述部分，不要包含股票数据表格（表格将单独附加）
- 使用以下固定格式，不要偏离

输出格式：
【纳斯达克100开盘日报】{date}

📊 开盘走势
（2-3句：QQQ/SPY实际开盘方向、与盘前预期对比、开盘15分钟整体氛围，引用实际涨跌幅数字）

🔥 开盘三大热点
1.
2.
3.

⚠️ 盘中关注
（1-2点：今日盘中重要时间节点或待关注催化剂）"""


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
        max_tokens=4048,
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
    """盘前日报：先获取 yfinance 完整盘前数据，再连同 EM 数据一起交给 LLM 生成汇报。"""
    t0 = time.perf_counter()
    stock_results: list = state.get("stock_results") or []
    prompt = _PREMARKET_SYSTEM.replace("{date}", state.get("date") or "")

    # Step 1：获取 yfinance 盘前数据（QQQ/SPY/RSP + Mag7）
    yf_data = await _fetch_yf_premarket(["QQQ", "SPY", "RSP"] + MAG7)

    # Step 2：构建完整行情上下文（指数 + 七姐妹 + EM 涨跌幅），交给 LLM 生成叙述
    market_context = _build_full_market_context(yf_data, stock_results)
    narrative = await _generate_narrative(state, prompt, market_context)

    # Step 3：拼接指数表（yfinance 主路径，失败则 Finnhub/EM 兜底）
    name_map = {"QQQ": "纳指100ETF(QQQ)", "SPY": "标普500ETF(SPY)", "RSP": "标普500等权ETF(RSP)"}
    index_rows = []
    for sym in ["QQQ", "SPY", "RSP"]:
        if sym not in yf_data:
            continue
        q = yf_data[sym]
        s, ps = ("+" if q["chg_pct"] >= 0 else ""), ("+" if q["chg_pts"] >= 0 else "")
        index_rows.append(f"| {name_map[sym]} | {q['price']:,.2f} | {ps}{q['chg_pts']:.2f} | {s}{q['chg_pct']:.4f}% |")

    if index_rows:
        header = "| 指数/ETF | 盘前价 | 涨跌点 | 涨跌幅 |\n|:---|---:|---:|---:|"
        premarket_index = "📊 主要指数\n\n" + header + "\n" + "\n".join(index_rows)
        print(f"[nasdaq] premarket index yfinance: {len(index_rows)} rows")
    else:
        print("[nasdaq] yfinance index empty, falling back to Finnhub/EM")
        premarket_index = await _fetch_premarket_index_individual()

    # Step 4：拼接七姐妹表（yfinance 主路径，失败则 Finnhub 兜底）
    mag7_results = [(sym, yf_data[sym]["price"], yf_data[sym]["chg_pct"])
                    for sym in MAG7 if sym in yf_data]
    if mag7_results:
        print(f"[nasdaq] Mag7 yfinance: {len(mag7_results)}/{len(MAG7)}")
    else:
        print("[nasdaq] yfinance Mag7 empty, falling back to Finnhub")
        mag7_results = await _fetch_premarket_mag7()

    mag7_tbl = _mag7_table(mag7_results)

    sections = [narrative, "---"]
    if premarket_index:
        sections += [premarket_index, "---"]
    sections += ["💎 美股七姐妹盘前", mag7_tbl, "来源：Reuters/CNBC/MarketWatch"]
    report = "\n\n".join(sections)

    print(f"[nasdaq] generate_report: {time.perf_counter() - t0:.2f}s → {len(report)} chars total")
    return {"report_content": report}


async def generate_afterhours_report(state: NasdaqReportState) -> dict:
    """盘后日报：LLM生成叙述（含实际行情交叉验证）+ 程序拼接板块涨跌表格。"""
    t0 = time.perf_counter()
    stock_results: list = state.get("stock_results") or []
    movers_summary = _build_full_market_context({}, stock_results)
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


async def generate_intraday_report(state: NasdaqReportState) -> dict:
    """开盘日报：LLM生成叙述（含实际行情）+ 程序拼接指数表、七姐妹表、板块涨跌表。"""
    t0 = time.perf_counter()
    stock_results: list = state.get("stock_results") or []
    movers_summary = _build_full_market_context({}, stock_results)
    prompt = _INTRADAY_SYSTEM.replace("{date}", state.get("date") or "")
    narrative = await _generate_narrative(state, prompt, movers_summary)

    index_summary = state.get("index_summary", "")

    # MAG7 均在 NASDAQ100，从 stock_results 过滤
    mag7_results = [(sym, price, chg) for sym, price, chg in stock_results if sym in MAG7]
    mag7_tbl = _mag7_table(mag7_results, price_label="实时价")

    movers_table = _sector_movers_table(stock_results, price_label="实时价") if stock_results else "（股票数据暂不可用）"

    sections = [narrative, "---"]
    if index_summary:
        sections += [index_summary, "---"]
    sections += [
        "💎 美股七姐妹开盘",
        mag7_tbl,
        "---",
        "📈 板块涨跌榜（开盘15分钟）",
        movers_table,
        "来源：Reuters/CNBC/MarketWatch",
    ]
    report = "\n\n".join(sections)

    print(f"[nasdaq] generate_intraday_report: {time.perf_counter() - t0:.2f}s → {len(report)} chars total")
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
