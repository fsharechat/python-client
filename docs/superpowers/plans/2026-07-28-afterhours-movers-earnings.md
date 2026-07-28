# 盘后日报「异动个股解读 + 财报速览」Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为盘后日报新增两块个股层面的解读内容——全市场涨跌幅前5+5的涨跌原因（网络搜索），以及当日公布财报的纳指100成分股的关键财务数据（净利润/EPS/capex/FCF），两者在同一股票上重叠时合并展示。

**Architecture:** LangGraph 盘后图新增两个并行节点：`analyze_top_movers`（依赖 `fetch_stock_movers` 的行情数据，对涨跌前5+5做 DDG 新闻搜索）和 `find_earnings_reporters`（独立于行情数据，查询 Finnhub 财报日历+财务数据）。两者输出汇入 `generate_afterhours_report`，在其中合并、调用一次新的 LLM 生成精简条目文本，插入现有报告拼接流程。

**Tech Stack:** Python 3.11 async / httpx / LangGraph / langchain-anthropic（`ChatAnthropic`）/ Finnhub REST API（`/calendar/earnings`、`/stock/earnings`、`/stock/financials-reported`）。

## Global Constraints

- 本项目**无自动化测试框架**（无 pytest，无 CI）——文档 `CLAUDE.md` 明确规定"手动 curl 测试"。本计划的验证步骤一律使用可运行的 Python 脚本（`python3 -c` 或 scratch 脚本）+ `assert` 断言做手动验证，不引入 pytest。
- 所有新增网络调用（DDG 搜索、Finnhub 请求）失败时必须捕获异常、记录 `print(f"[nasdaq] ...")` 日志并优雅降级（返回空列表/`None`），不得让异常向上抛出中断整个报告生成流程——这是贯穿现有 `nasdaq_agent/nodes.py` 的既定风格，新代码必须遵循。
- 不编造数据：财报字段解析不到时留空，不得用估算值填充；LLM prompt 中必须明确要求"无法解释的涨跌不编原因、未提供的财务数字不编数字"。
- 只改动盘后日报路径（`report_type == "afterhours"`），不影响盘前 (`build_nasdaq_graph`) / 盘中 (`build_intraday_graph`) 图。
- 复用现有 helper：`_ddg_search`（搜索）、`_fetch_finnhub_quotes` 的节流模式（1次/秒）、`ChatAnthropic` + `GENERATE_MODEL` 的调用方式（参考 `_generate_narrative`）。
- Finnhub 请求需要 `FINNHUB_API_KEY`（`config.py` 已有，来自 `.env`），若未配置则所有 Track B 请求会失败并被优雅降级为空列表——这是预期行为，不需要额外处理。

---

## 文件改动总览

| 文件 | 改动 |
|---|---|
| `nasdaq_agent/state.py` | 新增 `movers_analysis`、`earnings_analysis` 两个状态字段 |
| `nasdaq_agent/nodes.py` | 新增 Track A/B 节点函数、Finnhub 财报数据 helper、合并函数、新 LLM prompt 与生成函数；修改 `generate_afterhours_report` |
| `nasdaq_agent/graph.py` | 修改 `build_afterhours_graph()`：新增两个节点、调整边 |

---

### Task 1: 状态字段扩展

**Files:**
- Modify: `nasdaq_agent/state.py:1-13`

**Interfaces:**
- Produces: `NasdaqReportState` 新增两个 key：`movers_analysis: list[dict]`、`earnings_analysis: list[dict]`。后续所有任务读写这两个 key。

- [ ] **Step 1: 修改 `NasdaqReportState`**

将 `nasdaq_agent/state.py` 现有内容：

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict


class NasdaqReportState(TypedDict):
    date: str
    report_type: str    # "premarket" | "afterhours" | "intraday"，service.py 初始化时写入
    raw_articles: Annotated[list[dict], operator.add]  # reducer: accumulated by all parallel search nodes
    index_summary: str  # 由 fetch_stock_movers 写入：指数行情 markdown 表格（盘前用QQQ/SPY，盘后用NDX/SPX）
    stock_results: list # 由 fetch_stock_movers 写入：原始 [(sym, price, chg), ...] 供报告节点按场景格式化
    report_content: str
    send_status: str
```

改为：

```python
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
```

- [ ] **Step 2: 手动验证 TypedDict 可正常构造**

```bash
python3 -c "
from nasdaq_agent.state import NasdaqReportState
s: NasdaqReportState = {
    'date': '2026-07-28', 'report_type': 'afterhours', 'raw_articles': [],
    'index_summary': '', 'stock_results': [], 'movers_analysis': [], 'earnings_analysis': [],
    'report_content': '', 'send_status': '',
}
assert s['movers_analysis'] == []
assert s['earnings_analysis'] == []
print('OK: state fields verified')
"
```

Expected: `OK: state fields verified`

- [ ] **Step 3: Commit**

```bash
git add nasdaq_agent/state.py
git commit -m "feat: 盘后日报状态新增 movers_analysis/earnings_analysis 字段"
```

---

### Task 2: Track A — 涨跌前5+5个股新闻搜索节点

**Files:**
- Modify: `nasdaq_agent/nodes.py`（在 `fetch_stock_movers` 定义之后、`# ─── 报告生成节点 ───` 分隔注释之前插入，即第514行附近）

**Interfaces:**
- Consumes: `state["stock_results"]`（`list[tuple[str, float, float]]`，来自 `fetch_stock_movers`）、`state["date"]`、既有 `_ddg_search(query: str, max_results: int = 5) -> list[dict]`
- Produces: 节点 `analyze_top_movers(state) -> dict`，返回 `{"movers_analysis": list[dict]}`；每个 dict 形如 `{"symbol": str, "price": float, "chg": float, "news_snippets": list[dict]}`。后续 Task 5 的 `_merge_movers_and_earnings` 依赖此结构。

- [ ] **Step 1: 在 `nodes.py` 插入 `_select_top_movers` 与 `analyze_top_movers`**

在 `fetch_stock_movers` 函数结尾（约第513行 `return {"index_summary": ...}` 之后）、`# ─── 报告生成节点 ───` 注释之前插入：

```python
def _select_top_movers(stock_results: list[tuple], n: int = 5) -> list[tuple]:
    """按涨跌幅排序，返回全市场涨幅前n + 跌幅前n（最多2n条），元素为 (sym, price, chg)。"""
    gainers = sorted([r for r in stock_results if r[2] > 0], key=lambda x: x[2], reverse=True)[:n]
    losers = sorted([r for r in stock_results if r[2] < 0], key=lambda x: x[2])[:n]
    return gainers + losers


async def analyze_top_movers(state: NasdaqReportState) -> dict:
    """盘后专用：对全市场涨跌幅前5+5个股做新闻搜索，找涨跌原因。"""
    import asyncio
    t0 = time.perf_counter()
    stock_results: list = state.get("stock_results") or []
    date = state.get("date") or ""
    top_movers = _select_top_movers(stock_results, n=5)

    async def _search_one(sym: str, chg: float) -> list[dict]:
        direction = "up" if chg > 0 else "down"
        query = f"{sym} stock why {direction} {date}"
        return await _ddg_search(query, max_results=5)

    tasks = [_search_one(sym, chg) for sym, price, chg in top_movers]
    results = await asyncio.gather(*tasks)

    movers_analysis = []
    for (sym, price, chg), snippets in zip(top_movers, results):
        if not snippets:
            print(f"[nasdaq] analyze_top_movers: {sym} no DDG results, skipped")
            continue
        movers_analysis.append({"symbol": sym, "price": price, "chg": chg, "news_snippets": snippets})

    print(f"[nasdaq] analyze_top_movers: {time.perf_counter() - t0:.2f}s → {len(movers_analysis)}/{len(top_movers)} movers with news")
    return {"movers_analysis": movers_analysis}
```

- [ ] **Step 2: 验证 `_select_top_movers` 纯函数逻辑（无需网络）**

```bash
python3 -c "
from nasdaq_agent.nodes import _select_top_movers

sample = [
    ('A', 10.0, 8.5), ('B', 20.0, 5.0), ('C', 30.0, -6.0),
    ('D', 40.0, 3.0), ('E', 50.0, -9.0), ('F', 60.0, 1.0),
    ('G', 70.0, -1.0), ('H', 80.0, 7.0), ('I', 90.0, -4.0), ('J', 100.0, 2.0),
]
result = _select_top_movers(sample, n=2)
symbols = [r[0] for r in result]
assert symbols == ['A', 'H', 'E', 'C'], symbols
print('OK: _select_top_movers verified ->', symbols)
"
```

Expected: `OK: _select_top_movers verified -> ['A', 'H', 'E', 'C']`（涨幅前2: A(8.5) H(7.0)；跌幅前2: E(-9.0) C(-6.0)）

- [ ] **Step 3: 验证 `analyze_top_movers` 节点可运行（真实网络请求）**

```bash
python3 -c "
import asyncio
from nasdaq_agent.nodes import analyze_top_movers

async def main():
    state = {
        'date': '2026-07-28',
        'stock_results': [('NVDA', 180.0, 6.2), ('AAPL', 220.0, -3.1), ('MSFT', 500.0, 1.0)],
    }
    result = await analyze_top_movers(state)
    assert 'movers_analysis' in result
    assert isinstance(result['movers_analysis'], list)
    print('OK: analyze_top_movers ->', len(result['movers_analysis']), 'entries')

asyncio.run(main())
"
```

Expected: `OK: analyze_top_movers -> N entries`（N 取决于当天 DDG 实际搜索结果，0也算通过——只要不报错）

- [ ] **Step 4: Commit**

```bash
git add nasdaq_agent/nodes.py
git commit -m "feat: 新增 analyze_top_movers 节点，搜索涨跌前5+5个股新闻原因"
```

---

### Task 3: Track B — Finnhub 财报日历 helper

**Files:**
- Modify: `nasdaq_agent/nodes.py`（紧跟 Task 2 新增代码之后插入）

**Interfaces:**
- Consumes: `FINNHUB_API_KEY`（已从 `config.py` 导入）、`NASDAQ100_TICKERS`（已从 `nasdaq_agent.tickers` 导入）
- Produces: `_fetch_finnhub_earnings_calendar(date: str) -> list[str]`，返回当日公布财报且属于纳指100成分股的 symbol 列表（已排序，可能为空）。Task 4 的 `find_earnings_reporters` 节点依赖此函数。

- [ ] **Step 1: 插入财报日历 URL 常量与 helper 函数**

在 `nodes.py` 顶部 URL 常量区（`_FH_QUOTE_URL = "https://finnhub.io/api/v1/quote"` 那一行，约第40行）之后新增：

```python
_FH_EARNINGS_CALENDAR_URL = "https://finnhub.io/api/v1/calendar/earnings"
_FH_EARNINGS_URL = "https://finnhub.io/api/v1/stock/earnings"
_FH_FINANCIALS_REPORTED_URL = "https://finnhub.io/api/v1/stock/financials-reported"
```

在 Task 2 新增的 `analyze_top_movers` 函数之后插入：

```python
async def _fetch_finnhub_earnings_calendar(date: str) -> list[str]:
    """查询 Finnhub 当日财报日历，返回其中属于纳指100成分股的 symbol 列表。"""
    ticker_set = set(NASDAQ100_TICKERS)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(_FH_EARNINGS_CALENDAR_URL, params={
                "from": date, "to": date, "token": FINNHUB_API_KEY,
            })
        data = resp.json()
        print(f"[nasdaq] Finnhub earnings calendar raw: {json.dumps(data)[:500]}")
        entries = data.get("earningsCalendar", [])
        symbols = sorted({e["symbol"] for e in entries if e.get("symbol") in ticker_set})
        print(f"[nasdaq] Finnhub earnings calendar: {len(symbols)} NASDAQ100 matches: {symbols}")
        return symbols
    except Exception as e:
        print(f"[nasdaq] Finnhub earnings calendar failed: {e}")
        return []
```

- [ ] **Step 2: 验证 helper 可运行（真实网络请求，需要 `.env` 中配置 `FINNHUB_API_KEY`）**

```bash
python3 -c "
import asyncio
from nasdaq_agent.nodes import _fetch_finnhub_earnings_calendar

async def main():
    symbols = await _fetch_finnhub_earnings_calendar('2026-07-28')
    assert isinstance(symbols, list)
    print('OK: earnings calendar ->', symbols)

asyncio.run(main())
"
```

Expected: `OK: earnings calendar -> [...]`（当日若无纳指100公司发财报，返回空列表也算通过——只要不报错）

- [ ] **Step 3: Commit**

```bash
git add nasdaq_agent/nodes.py
git commit -m "feat: 新增 Finnhub 财报日历查询 helper"
```

---

### Task 4: Track B — EPS/财务数据 helper 与 `find_earnings_reporters` 节点

**Files:**
- Modify: `nasdaq_agent/nodes.py`（紧跟 Task 3 新增代码之后插入）

**Interfaces:**
- Consumes: `_fetch_finnhub_earnings_calendar`（Task 3）
- Produces: 节点 `find_earnings_reporters(state) -> dict`，返回 `{"earnings_analysis": list[dict]}`；每个 dict 形如 `{"symbol": str, "eps_actual": float|None, "eps_estimate": float|None, "period": str|None, "net_income": float|None, "capex": float|None, "operating_cf": float|None, "fcf": float|None}`。Task 5 的 `_merge_movers_and_earnings` 依赖此结构。

- [ ] **Step 1: 插入 EPS helper、财务报表解析 helper、节点函数**

紧跟 Task 3 的 `_fetch_finnhub_earnings_calendar` 之后插入：

```python
async def _fetch_finnhub_eps(symbol: str) -> dict | None:
    """最近一期 EPS 实际值 vs 预期值（Finnhub /stock/earnings，按时间倒序，取第一条）。"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(_FH_EARNINGS_URL, params={"symbol": symbol, "token": FINNHUB_API_KEY})
        data = resp.json()
        if not data:
            return None
        latest = data[0]
        return {
            "eps_actual": latest.get("actual"),
            "eps_estimate": latest.get("estimate"),
            "period": latest.get("period"),
        }
    except Exception as e:
        print(f"[nasdaq] Finnhub EPS {symbol} failed: {e}")
        return None


_NET_INCOME_CONCEPTS = {"NetIncomeLoss", "ProfitLoss"}
_OPERATING_CF_CONCEPTS = {
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
}
_CAPEX_CONCEPTS = {
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsForCapitalImprovements",
}


def _find_concept_value(items: list[dict], concepts: set[str]) -> float | None:
    """从 Finnhub financials-reported 的 ic/cf 数组中按 concept 标签找数值。"""
    for item in items:
        if item.get("concept") in concepts:
            try:
                return float(item.get("value"))
            except (TypeError, ValueError):
                continue
    return None


async def _fetch_finnhub_financials(symbol: str) -> dict | None:
    """最新一期利润表净利润 + 现金流量表经营现金流/资本支出，算出FCF。"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(_FH_FINANCIALS_REPORTED_URL, params={
                "symbol": symbol, "freq": "quarterly", "token": FINNHUB_API_KEY,
            })
        data = resp.json()
        reports = data.get("data", [])
        if not reports:
            return None
        report = reports[0].get("report", {})
        ic_items = report.get("ic", [])
        cf_items = report.get("cf", [])

        net_income = _find_concept_value(ic_items, _NET_INCOME_CONCEPTS)
        operating_cf = _find_concept_value(cf_items, _OPERATING_CF_CONCEPTS)
        capex = _find_concept_value(cf_items, _CAPEX_CONCEPTS)
        fcf = (operating_cf - abs(capex)) if operating_cf is not None and capex is not None else None

        return {"net_income": net_income, "capex": capex, "operating_cf": operating_cf, "fcf": fcf}
    except Exception as e:
        print(f"[nasdaq] Finnhub financials-reported {symbol} failed: {e}")
        return None


async def find_earnings_reporters(state: NasdaqReportState) -> dict:
    """盘后专用：查询当日公布财报的纳指100成分股，抓取EPS与关键财务数据。"""
    import asyncio
    t0 = time.perf_counter()
    date = state.get("date") or ""

    symbols = await _fetch_finnhub_earnings_calendar(date)
    if not symbols:
        print(f"[nasdaq] find_earnings_reporters: {time.perf_counter() - t0:.2f}s → 0 reporters")
        return {"earnings_analysis": []}

    earnings_analysis = []
    for i, sym in enumerate(symbols):
        eps = await _fetch_finnhub_eps(sym)
        financials = await _fetch_finnhub_financials(sym)
        if eps is None and financials is None:
            print(f"[nasdaq] find_earnings_reporters: {sym} no data, skipped")
            continue
        entry = {"symbol": sym}
        entry.update(eps or {})
        entry.update(financials or {})
        earnings_analysis.append(entry)
        if i < len(symbols) - 1:
            await asyncio.sleep(1)

    print(f"[nasdaq] find_earnings_reporters: {time.perf_counter() - t0:.2f}s → {len(earnings_analysis)}/{len(symbols)} reporters with data")
    return {"earnings_analysis": earnings_analysis}
```

- [ ] **Step 2: 验证 `_find_concept_value` 纯函数逻辑（无需网络）**

```bash
python3 -c "
from nasdaq_agent.nodes import _find_concept_value, _NET_INCOME_CONCEPTS, _CAPEX_CONCEPTS

items = [
    {'concept': 'Revenues', 'value': '1000000'},
    {'concept': 'NetIncomeLoss', 'value': '250000.5'},
]
assert _find_concept_value(items, _NET_INCOME_CONCEPTS) == 250000.5
assert _find_concept_value(items, _CAPEX_CONCEPTS) is None
print('OK: _find_concept_value verified')
"
```

Expected: `OK: _find_concept_value verified`

- [ ] **Step 3: 验证 `find_earnings_reporters` 节点可运行（真实网络请求）**

```bash
python3 -c "
import asyncio
from nasdaq_agent.nodes import find_earnings_reporters

async def main():
    state = {'date': '2026-07-28'}
    result = await find_earnings_reporters(state)
    assert 'earnings_analysis' in result
    assert isinstance(result['earnings_analysis'], list)
    print('OK: find_earnings_reporters ->', result['earnings_analysis'])

asyncio.run(main())
"
```

Expected: `OK: find_earnings_reporters -> [...]`（空列表也算通过——只要不报错；建议挑一个已知有纳指100公司发财报的日期复测一次，确认非空场景数据格式正确）

- [ ] **Step 4: Commit**

```bash
git add nasdaq_agent/nodes.py
git commit -m "feat: 新增 find_earnings_reporters 节点，抓取当日财报公司EPS与财务数据"
```

---

### Task 5: 合并 Track A / Track B 结果

**Files:**
- Modify: `nasdaq_agent/nodes.py`（紧跟 Task 4 新增代码之后插入）

**Interfaces:**
- Consumes: `movers_analysis: list[dict]`（Task 2 输出结构）、`earnings_analysis: list[dict]`（Task 4 输出结构）
- Produces: `_merge_movers_and_earnings(movers_analysis, earnings_analysis) -> tuple[list[dict], list[dict]]`，返回 `(merged_movers, earnings_only)`。`merged_movers` 中命中财报的条目会带上 `"earnings"` key。Task 6 依赖此函数。

- [ ] **Step 1: 插入合并函数**

```python
def _merge_movers_and_earnings(movers_analysis: list[dict], earnings_analysis: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    合并异动个股（Track A）与财报公司（Track B）结果。
    返回 (merged_movers, earnings_only)：
    - merged_movers：movers_analysis 的每条，若 symbol 命中 earnings_analysis 则附加 "earnings" 字段
    - earnings_only：earnings_analysis 中未出现在 movers_analysis 里的条目（发了财报但不在涨跌前5+5）
    """
    earnings_map = {e["symbol"]: e for e in earnings_analysis}
    mover_symbols = {m["symbol"] for m in movers_analysis}

    merged_movers = []
    for m in movers_analysis:
        entry = dict(m)
        if m["symbol"] in earnings_map:
            entry["earnings"] = earnings_map[m["symbol"]]
        merged_movers.append(entry)

    earnings_only = [e for e in earnings_analysis if e["symbol"] not in mover_symbols]
    return merged_movers, earnings_only
```

- [ ] **Step 2: 验证合并逻辑（无需网络）**

```bash
python3 -c "
from nasdaq_agent.nodes import _merge_movers_and_earnings

movers = [
    {'symbol': 'NVDA', 'price': 180.0, 'chg': 8.2, 'news_snippets': [{'title': 't', 'body': 'b'}]},
    {'symbol': 'XYZ', 'price': 50.0, 'chg': -5.1, 'news_snippets': [{'title': 't2', 'body': 'b2'}]},
]
earnings = [
    {'symbol': 'NVDA', 'eps_actual': 1.2, 'eps_estimate': 1.0, 'net_income': 1e10, 'capex': 3e9, 'fcf': 5e9},
    {'symbol': 'AAPL', 'eps_actual': 2.1, 'eps_estimate': 2.0, 'net_income': 2e10, 'capex': 4e9, 'fcf': 8e9},
]
merged_movers, earnings_only = _merge_movers_and_earnings(movers, earnings)

assert len(merged_movers) == 2
nvda = next(m for m in merged_movers if m['symbol'] == 'NVDA')
assert 'earnings' in nvda and nvda['earnings']['eps_actual'] == 1.2
xyz = next(m for m in merged_movers if m['symbol'] == 'XYZ')
assert 'earnings' not in xyz

assert len(earnings_only) == 1 and earnings_only[0]['symbol'] == 'AAPL'
print('OK: _merge_movers_and_earnings verified')
"
```

Expected: `OK: _merge_movers_and_earnings verified`

- [ ] **Step 3: Commit**

```bash
git add nasdaq_agent/nodes.py
git commit -m "feat: 新增 _merge_movers_and_earnings 合并异动个股与财报数据"
```

---

### Task 6: 新 LLM 生成函数并接入 `generate_afterhours_report`

**Files:**
- Modify: `nasdaq_agent/nodes.py:678-696`（`generate_afterhours_report` 函数）；在其之前插入新 prompt 常量与生成函数

**Interfaces:**
- Consumes: `_merge_movers_and_earnings`（Task 5）、`ChatAnthropic`/`GENERATE_MODEL`/`ANTHROPIC_API_KEY`（已导入）
- Produces: `_generate_movers_insight(merged_movers, earnings_only) -> str`（均为空时返回空字符串）；修改后的 `generate_afterhours_report(state) -> dict` 在报告中插入新 section。

- [ ] **Step 1: 在 `_AFTERHOURS_SYSTEM` 常量之后插入新 prompt 与生成函数**

在 `nodes.py` 中 `_AFTERHOURS_SYSTEM = """..."""`（约541-562行）结束之后、`_INTRADAY_SYSTEM` 定义之前插入：

```python
_MOVERS_INSIGHT_SYSTEM = """你是专业美股分析师。根据提供的个股新闻摘要和财报数据，用中文生成两个部分的精简条目。

严格要求：
- 总字数不超过800字
- 每只股票1-2句话，简明扼要
- 若某只股票的新闻摘要无法解释其涨跌原因，跳过该股，不要编造原因
- 财报数据（净利润/EPS/capex/FCF）若提供了就必须引用具体数字，不得编造未提供的数字
- 严格使用以下格式，若某部分没有对应股票数据则完全省略该部分（包括标题）

输出格式：
🔍 异动个股解读
• {代码} {涨跌幅}：（1-2句涨跌原因，若有财报数据则一并给出净利润/EPS/capex/FCF）

📑 今日财报速览
• {代码}：（净利润/EPS实际vs预期/capex/FCF关键数字）"""


async def _generate_movers_insight(merged_movers: list[dict], earnings_only: list[dict]) -> str:
    """调用 LLM 生成异动个股解读 + 财报速览。均为空时返回空字符串。"""
    if not merged_movers and not earnings_only:
        return ""

    t0 = time.perf_counter()

    def _fmt_movers(entries: list[dict]) -> str:
        lines = []
        for e in entries:
            snippets = "; ".join(s.get("body", "")[:150] for s in e.get("news_snippets", [])[:3])
            line = f"{e['symbol']} {e['chg']:+.2f}% (${e['price']:.2f})：新闻摘要：{snippets}"
            if "earnings" in e:
                line += f" | 财报：{e['earnings']}"
            lines.append(line)
        return "\n".join(lines)

    def _fmt_earnings_only(entries: list[dict]) -> str:
        return "\n".join(f"{e['symbol']}：{e}" for e in entries)

    parts = []
    if merged_movers:
        parts.append("【异动个股新闻与财报】\n" + _fmt_movers(merged_movers))
    if earnings_only:
        parts.append("【今日财报公司（非异动榜）】\n" + _fmt_earnings_only(earnings_only))
    human_content = "\n\n".join(parts)

    llm = ChatAnthropic(
        model=GENERATE_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=2048,
        max_retries=3,
    )

    try:
        insight = await (llm | StrOutputParser()).ainvoke([
            SystemMessage(content=_MOVERS_INSIGHT_SYSTEM),
            HumanMessage(content=human_content),
        ])
    except Exception as e:
        print(f"[nasdaq] _generate_movers_insight failed: {e}")
        return ""

    print(f"[nasdaq] _generate_movers_insight: {time.perf_counter() - t0:.2f}s → {len(insight)} chars")
    return insight
```

- [ ] **Step 2: 修改 `generate_afterhours_report`**

将现有（第678-696行）：

```python
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
```

改为：

```python
async def generate_afterhours_report(state: NasdaqReportState) -> dict:
    """盘后日报：LLM生成叙述（含实际行情交叉验证）+ 异动个股解读/财报速览 + 程序拼接板块涨跌表格。"""
    t0 = time.perf_counter()
    stock_results: list = state.get("stock_results") or []
    movers_summary = _build_full_market_context({}, stock_results)
    prompt = _AFTERHOURS_SYSTEM.replace("{date}", state.get("date") or "")
    narrative = await _generate_narrative(state, prompt, movers_summary)

    merged_movers, earnings_only = _merge_movers_and_earnings(
        state.get("movers_analysis") or [], state.get("earnings_analysis") or []
    )
    insight = await _generate_movers_insight(merged_movers, earnings_only)

    index_summary = state.get("index_summary", "")
    movers_table = _sector_movers_table(stock_results, price_label="收盘价") if stock_results else "（股票数据暂不可用）"

    sections = [narrative, "---"]
    if insight:
        sections += [insight, "---"]
    if index_summary:
        sections += [index_summary, "---"]
    sections += ["📈 板块涨跌榜", movers_table, "来源：Reuters/CNBC/MarketWatch"]
    report = "\n\n".join(sections)

    print(f"[nasdaq] generate_afterhours_report: {time.perf_counter() - t0:.2f}s → {len(report)} chars total")
    return {"report_content": report}
```

- [ ] **Step 3: 验证 `_generate_movers_insight` 空输入返回空字符串（无需网络）**

```bash
python3 -c "
import asyncio
from nasdaq_agent.nodes import _generate_movers_insight

async def main():
    result = await _generate_movers_insight([], [])
    assert result == ''
    print('OK: _generate_movers_insight empty-input verified')

asyncio.run(main())
"
```

Expected: `OK: _generate_movers_insight empty-input verified`

- [ ] **Step 4: 验证 `_generate_movers_insight` 非空输入可调用 LLM（真实API请求，需要 `ANTHROPIC_API_KEY`）**

```bash
python3 -c "
import asyncio
from nasdaq_agent.nodes import _generate_movers_insight

async def main():
    merged_movers = [{
        'symbol': 'NVDA', 'price': 180.0, 'chg': 8.2,
        'news_snippets': [{'title': 'NVDA surges on AI demand', 'body': 'Nvidia shares jumped after strong data center demand forecast.'}],
        'earnings': {'eps_actual': 1.2, 'eps_estimate': 1.0, 'net_income': 1.5e10, 'capex': 3e9, 'fcf': 6e9},
    }]
    result = await _generate_movers_insight(merged_movers, [])
    assert isinstance(result, str) and len(result) > 0
    print('OK: LLM insight generated:')
    print(result)

asyncio.run(main())
"
```

Expected: 打印非空的中文条目文本，包含 `🔍 异动个股解读`

- [ ] **Step 5: Commit**

```bash
git add nasdaq_agent/nodes.py
git commit -m "feat: generate_afterhours_report 接入异动个股解读/财报速览 section"
```

---

### Task 7: 图结构接入新节点

**Files:**
- Modify: `nasdaq_agent/graph.py:1-29`（import 区）、`nasdaq_agent/graph.py:61-87`（`build_afterhours_graph`）

**Interfaces:**
- Consumes: `analyze_top_movers`、`find_earnings_reporters`（Task 2、Task 4 定义的节点函数）
- Produces: `build_afterhours_graph()` 编译出的图新增 `analyze_top_movers`、`find_earnings_reporters` 两个节点，均汇入 `generate_afterhours_report`。

- [ ] **Step 1: 修改 import 列表**

将 `nasdaq_agent/graph.py` 第12-29行：

```python
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
    send_notification,
)
```

改为：

```python
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
```

- [ ] **Step 2: 修改 `build_afterhours_graph()`**

将现有（第61-87行）：

```python
def build_afterhours_graph():
    builder = StateGraph(NasdaqReportState)

    builder.add_node("search_earnings_results", search_earnings_results)
    builder.add_node("search_afterhours_movers", search_afterhours_movers)
    builder.add_node("search_closing_summary", search_closing_summary)
    builder.add_node("search_tomorrow_preview", search_tomorrow_preview)
    builder.add_node("fetch_stock_movers", fetch_stock_movers)
    builder.add_node("generate_afterhours_report", generate_afterhours_report)
    builder.add_node("send_notification", send_notification)

    builder.add_edge(START, "search_earnings_results")
    builder.add_edge(START, "search_afterhours_movers")
    builder.add_edge(START, "search_closing_summary")
    builder.add_edge(START, "search_tomorrow_preview")
    builder.add_edge(START, "fetch_stock_movers")

    builder.add_edge("search_earnings_results", "generate_afterhours_report")
    builder.add_edge("search_afterhours_movers", "generate_afterhours_report")
    builder.add_edge("search_closing_summary", "generate_afterhours_report")
    builder.add_edge("search_tomorrow_preview", "generate_afterhours_report")
    builder.add_edge("fetch_stock_movers", "generate_afterhours_report")

    builder.add_edge("generate_afterhours_report", "send_notification")
    builder.add_edge("send_notification", END)

    return builder.compile()
```

改为：

```python
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
    builder.add_edge(START, "find_earnings_reporters")

    builder.add_edge("fetch_stock_movers", "analyze_top_movers")

    builder.add_edge("search_earnings_results", "generate_afterhours_report")
    builder.add_edge("search_afterhours_movers", "generate_afterhours_report")
    builder.add_edge("search_closing_summary", "generate_afterhours_report")
    builder.add_edge("search_tomorrow_preview", "generate_afterhours_report")
    builder.add_edge("analyze_top_movers", "generate_afterhours_report")
    builder.add_edge("find_earnings_reporters", "generate_afterhours_report")

    builder.add_edge("generate_afterhours_report", "send_notification")
    builder.add_edge("send_notification", END)

    return builder.compile()
```

- [ ] **Step 3: 验证图可正常编译**

```bash
python3 -c "
from nasdaq_agent.graph import build_afterhours_graph

graph = build_afterhours_graph()
node_names = set(graph.get_graph().nodes.keys())
assert 'analyze_top_movers' in node_names, node_names
assert 'find_earnings_reporters' in node_names, node_names
print('OK: build_afterhours_graph compiled with new nodes ->', node_names)
"
```

Expected: `OK: build_afterhours_graph compiled with new nodes -> {...}`（包含 `analyze_top_movers`、`find_earnings_reporters`）

- [ ] **Step 4: Commit**

```bash
git add nasdaq_agent/graph.py
git commit -m "feat: 盘后图接入 analyze_top_movers/find_earnings_reporters 节点"
```

---

### Task 8: 端到端手动验证

**Files:**
- 无代码改动，纯验证步骤

**Interfaces:**
- Consumes: `service.py` 现有的 `POST /nasdaq/trigger/afterhours` 接口（无需改动，已经调用 `build_afterhours_graph()` 编译出的图）

- [ ] **Step 1: 启动服务**

```bash
python service.py
```

Expected: 日志显示服务在 `http://localhost:8000` 启动，无报错。

- [ ] **Step 2: 手动触发盘后日报**

```bash
curl -X POST http://localhost:8000/nasdaq/trigger/afterhours
```

Expected: HTTP 响应立即返回（异步任务已提交），后台日志陆续输出：
```
[nasdaq] analyze_top_movers: X.XXs → N/10 movers with news
[nasdaq] find_earnings_reporters: X.XXs → N/M reporters with data
[nasdaq] _generate_movers_insight: X.XXs → N chars
[nasdaq] generate_afterhours_report: X.XXs → N chars total
[nasdaq] send_notification: ... → ok:200
```

- [ ] **Step 3: 检查推送内容格式**

确认推送到 `NOTIFY_MOBILES` 的报告文本中：
- 若涨跌前5+5有可用新闻，出现 `🔍 异动个股解读` section，每条引用了实际涨跌幅数字
- 若当日有纳指100成分股发财报，出现 `📑 今日财报速览` 或对应股票在异动解读中带财报数字；数字应与 Finnhub 返回的原始日志数据一致（对照 Step 2 的 `[nasdaq] Finnhub financials-reported {symbol} raw` 之类的日志，若需要更详细排查可临时加日志）
- 报告结构未破坏：主叙述 → （异动解读/财报速览，可能省略）→ 主要指数（若有）→ 板块涨跌榜 → 来源

- [ ] **Step 4: 补充验证财报场景（建议）**

由于财报日历依赖当日真实市场事件，建议额外找一个**已知**有纳指100成分股当日盘后发财报的历史交易日，用以下方式单独验证 `find_earnings_reporters` 返回非空且字段正确（不通过完整服务，直接跑 Task 4 Step 3 的验证脚本，把日期换成该历史日期）：

```bash
python3 -c "
import asyncio
from nasdaq_agent.nodes import find_earnings_reporters

async def main():
    state = {'date': '2026-XX-XX'}  # 替换为已知有财报公布的日期
    result = await find_earnings_reporters(state)
    print(result['earnings_analysis'])

asyncio.run(main())
"
```

确认返回的每条记录里 `net_income`/`capex`/`fcf`/`eps_actual` 数值与该公司公开财报数据大致吻合（可用公开财经网站核对，不要求完全一致，但不应为明显错误的数量级）。

---

## Self-Review 记录

- **Spec 覆盖**：spec 中的 Track A（涨跌前5+5新闻搜索）→ Task 2；Track B（Finnhub财报日历+EPS+financials-reported）→ Task 3/4；合并逻辑 → Task 5；LLM 生成新 section 并插入报告 → Task 6；图结构调整 → Task 7；测试 → Task 8。State 字段 → Task 1。spec 中列出的"不在本次范围内"事项（盘前/盘中改动、历史财报趋势对比、数据源切换）均未在任务中出现，范围一致。
- **占位符扫描**：无 TBD/TODO，所有代码块为完整可运行代码。
- **类型一致性**：`movers_analysis`/`earnings_analysis` 的字段名和结构在 Task 2/4（生产者）与 Task 5/6（消费者）中保持一致（`symbol`/`price`/`chg`/`news_snippets`；`symbol`/`eps_actual`/`eps_estimate`/`period`/`net_income`/`capex`/`operating_cf`/`fcf`）；`_merge_movers_and_earnings` 返回的 `merged_movers`/`earnings_only` 在 Task 6 的 `_generate_movers_insight` 中的使用方式与其定义匹配。
