# 盘中开盘日报 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在美股开盘15分钟后（ET 09:45，周一至周五）自动推送盘中开盘日报，包含实际开盘行情与开盘后新闻。

**Architecture:** 新增 `report_type="intraday"`，复用现有 `fetch_stock_movers` 的 Finnhub+东方财富双源策略（与盘后报告完全一致）；新增3路开盘新闻搜索节点和独立 LangGraph 图；`service.py` 增加 ET 09:45 定时任务与手动触发接口。

**Tech Stack:** Python 3.11, LangGraph, APScheduler, Finnhub API, 东方财富 push2 API, DuckDuckGo HTML Search, FastAPI

---

## 文件改动地图

| 文件 | 类型 | 改动内容 |
|---|---|---|
| `nasdaq_agent/nodes.py` | 修改 | 新增3个搜索函数、`_INTRADAY_SYSTEM`、`generate_intraday_report()`；修改 `_mag7_table`、`_build_fh_index_table`、`fetch_stock_movers` |
| `nasdaq_agent/graph.py` | 修改 | 新增 `build_intraday_graph()`，更新 import |
| `service.py` | 修改 | 新增 runner、intraday graph 构建、ET 09:45 调度任务、`/nasdaq/trigger/intraday` 接口 |

> **注意**：本项目无自动化测试套件（见 CLAUDE.md）。验证步骤均为手动 curl 触发接口观察日志。

---

## Task 1：新增3路开盘新闻搜索节点

**Files:**
- Modify: `nasdaq_agent/nodes.py`（在现有 `search_tomorrow_preview` 函数之后追加）

- [ ] **Step 1：在 `nasdaq_agent/nodes.py` 中，找到 `search_tomorrow_preview` 函数末尾，追加以下3个函数**

```python
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
```

- [ ] **Step 2：验证语法无误**

```bash
python -c "from nasdaq_agent.nodes import search_opening_movers, search_morning_economics, search_opening_news; print('OK')"
```

期望输出：`OK`

- [ ] **Step 3：提交**

```bash
git add nasdaq_agent/nodes.py
git commit -m "feat: 盘中开盘日报新增3路新闻搜索节点"
```

---

## Task 2：修改 `_mag7_table`、`_build_fh_index_table`、`fetch_stock_movers`

**Files:**
- Modify: `nasdaq_agent/nodes.py`

### 2a：`_mag7_table` 加 `price_label` 参数

- [ ] **Step 1：找到 `_mag7_table` 函数，将其替换为以下版本（加 `price_label` 参数，header 改为动态）**

原代码（约第362行）：
```python
def _mag7_table(results: list[tuple]) -> str:
    """美股七姐妹盘前价格表。"""
    if not results:
        return "（七姐妹数据暂不可用）"
    header = "| 代码 | 盘前价 | 涨跌幅 |\n|:---|---:|---:|"
    rows = [
        f"| {sym} | ${price:.2f} | {'+'if chg>=0 else ''}{chg:.2f}% |"
        for sym, price, chg in results
    ]
    return header + "\n" + "\n".join(rows)
```

替换为：
```python
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
```

### 2b：`_build_fh_index_table` 加 intraday 标签分支

- [ ] **Step 2：找到 `_build_fh_index_table` 函数中的标签逻辑（约第419行），将其修改如下**

原代码：
```python
    val_label = "盘前价" if report_type == "premarket" else "收盘价"
```

替换为：
```python
    if report_type == "premarket":
        val_label = "盘前价"
    elif report_type == "intraday":
        val_label = "实时价"
    else:
        val_label = "收盘价"
```

### 2c：`fetch_stock_movers` 的 intraday 走盘后分支

- [ ] **Step 3：找到 `fetch_stock_movers` 函数中的分支判断（约第438行），将条件改为同时覆盖 intraday**

原代码：
```python
    if report_type == "afterhours":
```

替换为：
```python
    if report_type in ("afterhours", "intraday"):
```

- [ ] **Step 4：验证语法无误**

```bash
python -c "from nasdaq_agent.nodes import fetch_stock_movers, _mag7_table, _build_fh_index_table; print('OK')"
```

期望输出：`OK`

- [ ] **Step 5：提交**

```bash
git add nasdaq_agent/nodes.py
git commit -m "feat: 盘中开盘日报适配行情节点（实时价标签、intraday分支）"
```

---

## Task 3：新增 `_INTRADAY_SYSTEM` 和 `generate_intraday_report`

**Files:**
- Modify: `nasdaq_agent/nodes.py`（在 `_AFTERHOURS_SYSTEM` 常量之后追加常量，在 `generate_afterhours_report` 之后追加函数）

- [ ] **Step 1：在 `_AFTERHOURS_SYSTEM` 常量之后，追加 `_INTRADAY_SYSTEM`**

```python
_INTRADAY_SYSTEM = """你是专业美股分析师，擅长整理纳斯达克100开盘动态。
根据提供的【实际行情数据】和新闻摘要，用中文生成开盘日报的叙述部分。

严格要求：
- 总字数不超过800字
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
```

- [ ] **Step 2：在 `generate_afterhours_report` 函数之后，追加 `generate_intraday_report`**

```python
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
```

- [ ] **Step 3：验证语法无误**

```bash
python -c "from nasdaq_agent.nodes import generate_intraday_report, _INTRADAY_SYSTEM; print('OK')"
```

期望输出：`OK`

- [ ] **Step 4：提交**

```bash
git add nasdaq_agent/nodes.py
git commit -m "feat: 新增盘中开盘日报生成节点与系统提示"
```

---

## Task 4：新增 `build_intraday_graph`

**Files:**
- Modify: `nasdaq_agent/graph.py`

- [ ] **Step 1：在 `graph.py` 的 import 语句中，追加3个新搜索函数和 `generate_intraday_report` 的导入**

找到现有 import 块（约第11行），在末尾追加：
```python
from nasdaq_agent.nodes import (
    fetch_stock_movers,
    generate_report,
    generate_afterhours_report,
    generate_intraday_report,       # 新增
    search_earnings,
    search_futures,
    search_macro_news,
    search_tech_news,
    search_earnings_results,
    search_afterhours_movers,
    search_closing_summary,
    search_tomorrow_preview,
    search_opening_movers,          # 新增
    search_morning_economics,       # 新增
    search_opening_news,            # 新增
    send_notification,
)
```

- [ ] **Step 2：在 `build_afterhours_graph` 函数之后，追加 `build_intraday_graph`**

```python
def build_intraday_graph():
    builder = StateGraph(NasdaqReportState)

    builder.add_node("search_opening_movers", search_opening_movers)
    builder.add_node("search_morning_economics", search_morning_economics)
    builder.add_node("search_opening_news", search_opening_news)
    builder.add_node("fetch_stock_movers", fetch_stock_movers)
    builder.add_node("generate_intraday_report", generate_intraday_report)
    builder.add_node("send_notification", send_notification)

    builder.add_edge(START, "search_opening_movers")
    builder.add_edge(START, "search_morning_economics")
    builder.add_edge(START, "search_opening_news")
    builder.add_edge(START, "fetch_stock_movers")

    builder.add_edge("search_opening_movers", "generate_intraday_report")
    builder.add_edge("search_morning_economics", "generate_intraday_report")
    builder.add_edge("search_opening_news", "generate_intraday_report")
    builder.add_edge("fetch_stock_movers", "generate_intraday_report")

    builder.add_edge("generate_intraday_report", "send_notification")
    builder.add_edge("send_notification", END)

    return builder.compile()
```

- [ ] **Step 3：验证 graph 可以编译**

```bash
python -c "from nasdaq_agent.graph import build_intraday_graph; g = build_intraday_graph(); print('OK')"
```

期望输出：`OK`

- [ ] **Step 4：提交**

```bash
git add nasdaq_agent/graph.py
git commit -m "feat: 新增盘中开盘日报 LangGraph 状态机"
```

---

## Task 5：在 service.py 接入定时任务与触发接口

**Files:**
- Modify: `service.py`

- [ ] **Step 1：在 `service.py` 顶部 import 中，追加 `build_intraday_graph`**

找到现有：
```python
from nasdaq_agent.graph import build_nasdaq_graph, build_afterhours_graph
```

替换为：
```python
from nasdaq_agent.graph import build_nasdaq_graph, build_afterhours_graph, build_intraday_graph
```

- [ ] **Step 2：在 `_run_afterhours_report` 函数之后，追加 `_run_intraday_report`**

```python
async def _run_intraday_report(intraday_graph) -> None:
    today = date_type.today().isoformat()
    print(f"[intraday] Starting opening report for {today} ...")
    initial = {
        "date": today,
        "report_type": "intraday",
        "raw_articles": [],
        "index_summary": "",
        "stock_results": [],
        "report_content": "",
        "send_status": "",
    }
    try:
        await intraday_graph.ainvoke(initial)
        print("[intraday] Opening report complete.")
    except Exception as e:
        print(f"[intraday] Opening report failed: {e}")
```

- [ ] **Step 3：在 `lifespan` 函数中，找到 `afterhours_graph` 的构建和调度代码之后，追加 intraday graph 构建与调度任务**

找到（约第77行）：
```python
    afterhours_graph = build_afterhours_graph()
    app_state["afterhours_graph"] = afterhours_graph
```

在其后追加：
```python
    intraday_graph = build_intraday_graph()
    app_state["intraday_graph"] = intraday_graph
```

找到（约第89行）已有的 `scheduler.add_job` 两个调用，在其后追加：
```python
    scheduler.add_job(
        _run_intraday_report,
        "cron",
        day_of_week="mon-fri",
        hour=9,
        minute=45,
        args=[intraday_graph],
    )
```

同时找到 scheduler 日志（约第99行），将其替换：

原代码：
```python
    print("Nasdaq scheduler started (ET 08:00 premarket / ET 16:30 afterhours, Mon–Fri).")
```

替换为：
```python
    print("Nasdaq scheduler started (ET 08:00 premarket / ET 09:45 intraday / ET 16:30 afterhours, Mon–Fri).")
```

- [ ] **Step 4：在 `/nasdaq/trigger/afterhours` 接口之后，追加手动触发接口**

```python
@app.post("/nasdaq/trigger/intraday")
async def trigger_intraday_report():
    """立即触发一次纳斯达克开盘日报（用于测试或手动补发）。"""
    intraday_graph = app_state["intraday_graph"]
    asyncio.create_task(_run_intraday_report(intraday_graph))
    return {"status": "triggered", "date": date_type.today().isoformat()}
```

- [ ] **Step 5：同步更新文件顶部 docstring，追加新接口说明**

找到（约第9行）：
```python
  POST /nasdaq/trigger/afterhours  – manually trigger Nasdaq afterhours report
```

在其后追加：
```python
  POST /nasdaq/trigger/intraday    – manually trigger Nasdaq opening report
```

- [ ] **Step 6：验证 service.py 语法无误**

```bash
python -c "import service; print('OK')"
```

期望输出：`OK`

- [ ] **Step 7：提交**

```bash
git add service.py
git commit -m "feat: service.py 接入盘中开盘日报定时任务（ET 09:45）与手动触发接口"
```

---

## Task 6：端到端手动验证

**Files:** 无（纯验证）

- [ ] **Step 1：启动服务**

```bash
source .venv/bin/activate
python service.py
```

期望日志包含：
```
Nasdaq scheduler started (ET 08:00 premarket / ET 09:45 intraday / ET 16:30 afterhours, Mon–Fri).
```

- [ ] **Step 2：手动触发开盘日报**

新开终端：
```bash
curl -s -X POST http://localhost:8000/nasdaq/trigger/intraday | python3 -m json.tool
```

期望响应：
```json
{
    "status": "triggered",
    "date": "2026-05-21"
}
```

- [ ] **Step 3：观察服务日志，验证流程正常**

期望日志顺序（约2分钟内完成）：
```
[intraday] Starting opening report for 2026-05-21 ...
[nasdaq] search_opening_movers query: Nasdaq stocks gap up gap down market open 2026-05-21
[nasdaq] search_morning_economics query: US economic data release morning CPI jobs 2026-05-21
[nasdaq] search_opening_news query: stock market open recap early trading 2026-05-21
[nasdaq] fetch_stock_movers: [intraday] 拉取行情...
[nasdaq] Finnhub QQQ raw: ...
...
[nasdaq] generate_intraday_report: ...s → ... chars total
[nasdaq] send_notification: ...s → ok:200 ...
[intraday] Opening report complete.
```

- [ ] **Step 4：验证推送内容格式**

检查 FshareChat 通知，确认格式包含：
- `【纳斯达克100开盘日报】` 标题
- `📊 开盘走势` 章节
- `🔥 开盘三大热点`
- `⚠️ 盘中关注`
- `📊 主要指数` 表格（标签为"实时价"）
- `💎 美股七姐妹开盘` 表格（标签为"实时价"）
- `📈 板块涨跌榜（开盘15分钟）`

- [ ] **Step 5：最终提交（若有遗漏调整）**

```bash
git add -p  # 仅添加遗漏的小调整
git commit -m "fix: 盘中日报验证后微调"
```
