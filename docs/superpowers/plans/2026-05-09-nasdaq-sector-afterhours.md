# 纳斯达克100板块涨跌+盘后日报 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将盘前日报的涨跌榜改为按11个板块分组展示，并新增结构对称的盘后日报（含定时调度和手动触发接口）。

**Architecture:** 在 `nasdaq_agent/tickers.py` 新增板块映射；重构 `nodes.py` 的行情节点（按板块分组）和报告生成节点（LLM叙述与表格程序拼接解耦）；新增4个盘后搜索节点和盘后图构建函数；在 `service.py` 中新增盘后图初始化、ET 16:30 调度任务和手动触发接口。

**Tech Stack:** Python 3.11, LangGraph, FastAPI, APScheduler, httpx, langchain-anthropic

> **注意：** 本项目无自动化测试套件（见 CLAUDE.md），所有验证步骤均为手动执行。

---

## 文件一览

| 文件 | 操作 | 职责 |
|------|------|------|
| `nasdaq_agent/tickers.py` | 修改 | 新增 `NASDAQ100_SECTOR_MAP` 和 `SECTOR_ORDER` |
| `nasdaq_agent/nodes.py` | 修改 | 重构行情节点（板块分组）、重构报告生成（解耦）、新增盘后搜索节点和报告函数 |
| `nasdaq_agent/graph.py` | 修改 | 新增 `build_afterhours_graph()` |
| `service.py` | 修改 | 新增盘后图、ET 16:30 调度任务、`POST /nasdaq/trigger/afterhours` |

---

## Task 1: 新增板块映射（`tickers.py`）

**Files:**
- Modify: `nasdaq_agent/tickers.py`

- [ ] **Step 1: 在文件末尾追加 `NASDAQ100_SECTOR_MAP` 和 `SECTOR_ORDER`**

将以下内容追加到 `nasdaq_agent/tickers.py` 现有内容之后：

```python
# ticker → 板块中文名
NASDAQ100_SECTOR_MAP: dict[str, str] = {
    # 半导体
    "NVDA": "半导体", "AMD": "半导体", "AVGO": "半导体", "QCOM": "半导体",
    "INTC": "半导体", "TXN": "半导体", "AMAT": "半导体", "LRCX": "半导体",
    "KLAC": "半导体", "NXPI": "半导体", "MRVL": "半导体", "MCHP": "半导体",
    "ADI": "半导体", "ON": "半导体", "MU": "半导体", "ASML": "半导体",
    "ARM": "半导体", "GFS": "半导体", "SNPS": "半导体", "CDNS": "半导体",
    # 大型科技
    "AAPL": "大型科技", "MSFT": "大型科技", "GOOGL": "大型科技",
    "GOOG": "大型科技", "META": "大型科技", "AMZN": "大型科技",
    # 软件/SaaS
    "ADBE": "软件/SaaS", "INTU": "软件/SaaS", "PANW": "软件/SaaS",
    "CRWD": "软件/SaaS", "WDAY": "软件/SaaS", "DDOG": "软件/SaaS",
    "ZS": "软件/SaaS", "FTNT": "软件/SaaS", "OKTA": "软件/SaaS",
    "TEAM": "软件/SaaS", "CTSH": "软件/SaaS", "ROP": "软件/SaaS",
    "ANSS": "软件/SaaS", "CSGP": "软件/SaaS",
    # 互联网/电商
    "BKNG": "互联网/电商", "ABNB": "互联网/电商", "EBAY": "互联网/电商",
    "MELI": "互联网/电商", "PDD": "互联网/电商", "DASH": "互联网/电商",
    "RBLX": "互联网/电商", "ZM": "互联网/电商", "MTCH": "互联网/电商",
    "TTD": "互联网/电商", "APP": "互联网/电商",
    # 医疗健康
    "ISRG": "医疗健康", "VRTX": "医疗健康", "GILD": "医疗健康",
    "REGN": "医疗健康", "BIIB": "医疗健康", "IDXX": "医疗健康",
    "DXCM": "医疗健康", "ILMN": "医疗健康", "MRNA": "医疗健康",
    "ALGN": "医疗健康", "GEHC": "医疗健康",
    # 消费/零售
    "COST": "消费/零售", "SBUX": "消费/零售", "ORLY": "消费/零售",
    "DLTR": "消费/零售", "ROST": "消费/零售", "MDLZ": "消费/零售",
    "MNST": "消费/零售", "KDP": "消费/零售", "KHC": "消费/零售",
    # 媒体/娱乐
    "NFLX": "媒体/娱乐", "CHTR": "媒体/娱乐", "WBD": "媒体/娱乐",
    "SIRI": "媒体/娱乐", "EA": "媒体/娱乐", "TTWO": "媒体/娱乐",
    # 工业/物流
    "ADP": "工业/物流", "PAYX": "工业/物流", "PCAR": "工业/物流",
    "ODFL": "工业/物流", "CSX": "工业/物流", "FAST": "工业/物流",
    "VRSK": "工业/物流", "CTAS": "工业/物流", "CPRT": "工业/物流",
    "LIN": "工业/物流",
    # 金融科技
    "PYPL": "金融科技", "COIN": "金融科技",
    # 新能源/公用
    "CEG": "新能源/公用", "ENPH": "新能源/公用", "EXC": "新能源/公用",
    "XEL": "新能源/公用", "BKR": "新能源/公用", "FANG": "新能源/公用",
    "AEP": "新能源/公用",
    # 新兴科技/AI
    "TSLA": "新兴科技/AI", "PLTR": "新兴科技/AI",
    "AXON": "新兴科技/AI", "SMCI": "新兴科技/AI",
}

# 板块展示顺序
SECTOR_ORDER = [
    "半导体", "大型科技", "软件/SaaS", "互联网/电商", "医疗健康",
    "消费/零售", "媒体/娱乐", "工业/物流", "金融科技", "新能源/公用", "新兴科技/AI",
]
```

- [ ] **Step 2: 验证板块映射覆盖全部100只股票**

在项目根目录运行：

```bash
cd /path/to/python_client
source .venv/bin/activate
python -c "
from nasdaq_agent.tickers import NASDAQ100_TICKERS, NASDAQ100_SECTOR_MAP
missing = [t for t in NASDAQ100_TICKERS if t not in NASDAQ100_SECTOR_MAP]
extra = [t for t in NASDAQ100_SECTOR_MAP if t not in NASDAQ100_TICKERS]
print('Missing from map:', missing)
print('Extra in map:', extra)
print('Total mapped:', len(NASDAQ100_SECTOR_MAP))
"
```

预期输出：
```
Missing from map: []
Extra in map: []
Total mapped: 100
```

- [ ] **Step 3: Commit**

```bash
git add nasdaq_agent/tickers.py
git commit -m "feat: 新增纳斯达克100板块映射 NASDAQ100_SECTOR_MAP"
```

---

## Task 2: 重构行情节点为板块分组（`nodes.py`）

**Files:**
- Modify: `nasdaq_agent/nodes.py`

- [ ] **Step 1: 在文件顶部导入新增的板块常量**

在 `nasdaq_agent/nodes.py` 第19行，将：
```python
from nasdaq_agent.tickers import NASDAQ100_TICKERS
```
替换为：
```python
from nasdaq_agent.tickers import NASDAQ100_TICKERS, NASDAQ100_SECTOR_MAP, SECTOR_ORDER
```

- [ ] **Step 2: 新增板块分组表格生成函数**

在 `_fetch_stock_data` 函数定义之前（约第103行），插入以下函数：

```python
def _sector_movers_table(all_results: list[tuple]) -> str:
    """
    all_results: [(sym, price, chg), ...] 全量行情数据
    按 SECTOR_ORDER 分组，每板块展示涨幅前N和跌幅前N（N=min(10, 板块内有数据的股票数）。
    """
    # 按板块分桶
    buckets: dict[str, list[tuple]] = {s: [] for s in SECTOR_ORDER}
    for sym, price, chg in all_results:
        sector = NASDAQ100_SECTOR_MAP.get(sym, "其他")
        if sector not in buckets:
            buckets[sector] = []
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
        n = min(10, len(stocks_sorted))
        gainers = stocks_sorted[:n]
        losers = stocks_sorted[-n:][::-1]
        parts.append(f"▎{sector}（涨幅前{n}）\n\n{_table(gainers)}")
        parts.append(f"▎{sector}（跌幅前{n}）\n\n{_table(losers)}")

    return "\n\n".join(parts)
```

- [ ] **Step 3: 替换 `fetch_stock_movers` 函数体**

将现有的 `fetch_stock_movers` 函数（约第145-176行）整体替换为：

```python
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
```

- [ ] **Step 4: 验证板块分组格式**

```bash
python -c "
import asyncio
from nasdaq_agent.nodes import fetch_stock_movers
result = asyncio.run(fetch_stock_movers({'date': '2026-05-09', 'raw_articles': [], 'stock_movers': '', 'report_content': '', 'send_status': ''}))
print(result['stock_movers'][:1000])
"
```

预期：输出包含 `▎半导体（涨幅前10）` 等板块标题和 markdown 表格。

- [ ] **Step 5: Commit**

```bash
git add nasdaq_agent/nodes.py
git commit -m "feat: fetch_stock_movers 改为按板块分组展示涨跌榜"
```

---

## Task 3: 重构报告生成节点（LLM叙述与表格解耦）

**Files:**
- Modify: `nasdaq_agent/nodes.py`

- [ ] **Step 1: 删除旧常量，新增盘前叙述模板**

将现有的 `MAX_REPORT_CHARS = 2048` 和 `_REPORT_SYSTEM` 常量（约第23行和第181-206行）替换为：

```python
MAX_NARRATIVE_CHARS = 1200  # 只限制LLM叙述部分长度

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
```

- [ ] **Step 2: 新增共享 LLM 调用辅助函数 `_generate_narrative`**

在 `_PREMARKET_SYSTEM` 常量之后，插入：

```python
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

    narrative = await (llm | StrOutputParser()).ainvoke(messages)

    if len(narrative) > MAX_NARRATIVE_CHARS:
        narrative = narrative[:MAX_NARRATIVE_CHARS - 3] + "..."

    print(f"[nasdaq] _generate_narrative: {time.perf_counter() - t0:.2f}s → {len(narrative)} chars")
    return narrative
```

- [ ] **Step 3: 替换 `generate_report` 函数**

将现有 `generate_report` 函数整体替换为：

```python
async def generate_report(state: NasdaqReportState) -> dict:
    """盘前日报：LLM生成叙述 + 程序拼接板块涨跌表格。"""
    t0 = time.perf_counter()
    prompt = _PREMARKET_SYSTEM.replace("{date}", state["date"])
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
```

- [ ] **Step 4: 验证盘前日报生成（端到端，需服务运行或直接调用图）**

```bash
python -c "
import asyncio
from datetime import date
from nasdaq_agent.graph import build_nasdaq_graph

graph = build_nasdaq_graph()
initial = {'date': date.today().isoformat(), 'raw_articles': [], 'stock_movers': '', 'report_content': '', 'send_status': ''}
result = asyncio.run(graph.ainvoke(initial))
print('=== 报告内容 ===')
print(result['report_content'][:2000])
print('...')
print(f'总长度: {len(result[\"report_content\"])} chars')
"
```

预期：报告包含叙述部分（市场概况/热点/风险）+ 分割线 + 各板块表格。

- [ ] **Step 5: Commit**

```bash
git add nasdaq_agent/nodes.py
git commit -m "refactor: 报告生成解耦LLM叙述与板块表格拼接"
```

---

## Task 4: 新增盘后搜索节点和报告生成函数（`nodes.py`）

**Files:**
- Modify: `nasdaq_agent/nodes.py`

- [ ] **Step 1: 新增盘后叙述模板常量**

在 `_PREMARKET_SYSTEM` 之后追加：

```python
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
```

- [ ] **Step 2: 新增4个盘后搜索节点**

在 `search_futures` 函数之后（约第98行之后）追加：

```python
async def search_earnings_results(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    query = f"tech stocks earnings results after close beat miss {state['date']}"
    print(f"[nasdaq] search_earnings_results query: {query}")
    results = await _ddg_search(query)
    print(f"[nasdaq] search_earnings_results: {time.perf_counter() - t0:.2f}s → {len(results)} results")
    return {"raw_articles": results}


async def search_afterhours_movers(state: NasdaqReportState) -> dict:
    t0 = time.perf_counter()
    query = f"Nasdaq after hours movers stock gains losses premarket {state['date']}"
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
```

- [ ] **Step 3: 新增 `generate_afterhours_report` 函数**

在 `generate_report` 函数之后追加：

```python
async def generate_afterhours_report(state: NasdaqReportState) -> dict:
    """盘后日报：LLM生成叙述 + 程序拼接板块涨跌表格。"""
    t0 = time.perf_counter()
    prompt = _AFTERHOURS_SYSTEM.replace("{date}", state["date"])
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
```

- [ ] **Step 4: Commit**

```bash
git add nasdaq_agent/nodes.py
git commit -m "feat: 新增盘后搜索节点和 generate_afterhours_report"
```

---

## Task 5: 新增盘后图（`graph.py`）

**Files:**
- Modify: `nasdaq_agent/graph.py`

- [ ] **Step 1: 更新 import，新增盘后图构建函数**

将 `nasdaq_agent/graph.py` 整体替换为：

```python
"""
Nasdaq 100 日报 LangGraph 状态机。

build_nasdaq_graph():    盘前图（4路新闻搜索 + 行情 → 盘前报告 → 通知）
build_afterhours_graph(): 盘后图（4路盘后搜索 + 行情 → 盘后报告 → 通知）
"""

from langgraph.graph import END, START, StateGraph

from nasdaq_agent.state import NasdaqReportState
from nasdaq_agent.nodes import (
    fetch_stock_movers,
    generate_report,
    generate_afterhours_report,
    search_earnings,
    search_futures,
    search_macro_news,
    search_tech_news,
    search_earnings_results,
    search_afterhours_movers,
    search_closing_summary,
    search_tomorrow_preview,
    send_notification,
)


def build_nasdaq_graph():
    builder = StateGraph(NasdaqReportState)

    builder.add_node("search_tech", search_tech_news)
    builder.add_node("search_macro", search_macro_news)
    builder.add_node("search_earnings", search_earnings)
    builder.add_node("search_futures", search_futures)
    builder.add_node("fetch_stock_movers", fetch_stock_movers)
    builder.add_node("generate_report", generate_report)
    builder.add_node("send_notification", send_notification)

    builder.add_edge(START, "search_tech")
    builder.add_edge(START, "search_macro")
    builder.add_edge(START, "search_earnings")
    builder.add_edge(START, "search_futures")
    builder.add_edge(START, "fetch_stock_movers")

    builder.add_edge("search_tech", "generate_report")
    builder.add_edge("search_macro", "generate_report")
    builder.add_edge("search_earnings", "generate_report")
    builder.add_edge("search_futures", "generate_report")
    builder.add_edge("fetch_stock_movers", "generate_report")

    builder.add_edge("generate_report", "send_notification")
    builder.add_edge("send_notification", END)

    return builder.compile()


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

- [ ] **Step 2: 验证两个图可编译**

```bash
python -c "
from nasdaq_agent.graph import build_nasdaq_graph, build_afterhours_graph
g1 = build_nasdaq_graph()
g2 = build_afterhours_graph()
print('盘前图节点:', list(g1.nodes))
print('盘后图节点:', list(g2.nodes))
print('两个图编译成功')
"
```

预期：打印两组节点列表，无报错。

- [ ] **Step 3: Commit**

```bash
git add nasdaq_agent/graph.py
git commit -m "feat: 新增 build_afterhours_graph 盘后日报图"
```

---

## Task 6: 扩展 `service.py`（盘后图、调度、接口）

**Files:**
- Modify: `service.py`

- [ ] **Step 1: 更新 import，新增 `build_afterhours_graph`**

将 `service.py` 第29行：
```python
from nasdaq_agent.graph import build_nasdaq_graph
```
替换为：
```python
from nasdaq_agent.graph import build_nasdaq_graph, build_afterhours_graph
```

- [ ] **Step 2: 新增 `_run_afterhours_report` runner 函数**

在 `_run_nasdaq_report` 函数（约第34-39行）之后追加：

```python
async def _run_afterhours_report(afterhours_graph) -> None:
    today = date_type.today().isoformat()
    print(f"[nasdaq] Starting afterhours report for {today} ...")
    initial = {"date": today, "raw_articles": [], "stock_movers": "", "report_content": "", "send_status": ""}
    await afterhours_graph.ainvoke(initial)
    print("[nasdaq] Afterhours report complete.")
```

- [ ] **Step 3: 在 `lifespan` 中初始化盘后图并添加调度任务**

找到 `lifespan` 函数中以下代码块（约第60-75行）：
```python
    nasdaq_graph = build_nasdaq_graph()
    app_state["nasdaq_graph"] = nasdaq_graph

    scheduler = AsyncIOScheduler(timezone="America/New_York")
    # 每个工作日 ET 8:00 AM（自动处理夏/冬令时）
    scheduler.add_job(
        _run_nasdaq_report,
        "cron",
        day_of_week="mon-fri",
        hour=8,
        minute=0,
        args=[nasdaq_graph],
    )
    scheduler.start()
    app_state["scheduler"] = scheduler
    print("Nasdaq scheduler started (ET 08:00, Mon–Fri).")
```

替换为：

```python
    nasdaq_graph = build_nasdaq_graph()
    app_state["nasdaq_graph"] = nasdaq_graph

    afterhours_graph = build_afterhours_graph()
    app_state["afterhours_graph"] = afterhours_graph

    scheduler = AsyncIOScheduler(timezone="America/New_York")
    scheduler.add_job(
        _run_nasdaq_report,
        "cron",
        day_of_week="mon-fri",
        hour=8,
        minute=0,
        args=[nasdaq_graph],
    )
    scheduler.add_job(
        _run_afterhours_report,
        "cron",
        day_of_week="mon-fri",
        hour=16,
        minute=30,
        args=[afterhours_graph],
    )
    scheduler.start()
    app_state["scheduler"] = scheduler
    print("Nasdaq scheduler started (ET 08:00 premarket / ET 16:30 afterhours, Mon–Fri).")
```

- [ ] **Step 4: 新增 `POST /nasdaq/trigger/afterhours` 接口**

在 `trigger_nasdaq_report` 函数（约第195-200行）之后追加：

```python
@app.post("/nasdaq/trigger/afterhours")
async def trigger_afterhours_report():
    """立即触发一次纳斯达克盘后日报（用于测试或手动补发）。"""
    afterhours_graph = app_state["afterhours_graph"]
    asyncio.create_task(_run_afterhours_report(afterhours_graph))
    return {"status": "triggered", "date": date_type.today().isoformat()}
```

- [ ] **Step 5: 验证服务启动和接口可访问**

```bash
# 终端1：启动服务
python service.py

# 终端2：验证健康检查
curl http://localhost:8000/health
# 预期: {"status":"ok"}

# 验证盘后触发接口存在
curl -X POST http://localhost:8000/nasdaq/trigger/afterhours
# 预期: {"status":"triggered","date":"2026-05-09"}

# 验证 /docs 中新接口可见
curl http://localhost:8000/docs | grep afterhours
```

- [ ] **Step 6: Commit**

```bash
git add service.py
git commit -m "feat: service.py 新增盘后日报图、ET 16:30 调度和 /nasdaq/trigger/afterhours 接口"
```

---

## Task 7: 更新文档标注实现完成

**Files:**
- Modify: `docs/superpowers/specs/2026-05-09-nasdaq-sector-afterhours-design.md`

- [ ] **Step 1: 更新设计文档状态**

将文件第3行：
```
**状态**：已批准，待实现
```
替换为：
```
**状态**：已实现
```

- [ ] **Step 2: Commit 设计文档和实现计划**

```bash
git add docs/superpowers/specs/2026-05-09-nasdaq-sector-afterhours-design.md
git add docs/superpowers/plans/2026-05-09-nasdaq-sector-afterhours.md
git commit -m "docs: 新增纳斯达克板块涨跌+盘后日报设计文档和实现计划"
```
