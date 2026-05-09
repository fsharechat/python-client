# 纳斯达克100盘前/盘后日报按板块涨跌统计 — 设计文档

**日期**：2026-05-09  
**状态**：已实现

## 需求概述

1. **盘前日报改造**：将现有整体涨跌前十替换为按板块分组的涨跌榜。
2. **新增盘后日报**：结构与盘前对称，搜索内容改为盘后财报/异动/收盘总结/明日前瞻，同样按板块展示涨跌。

---

## 文件改动范围

| 文件 | 改动 |
|------|------|
| `nasdaq_agent/tickers.py` | 新增 `NASDAQ100_SECTOR_MAP`（ticker → 板块中文名） |
| `nasdaq_agent/nodes.py` | 改造行情节点；重构报告生成（LLM叙述与表格解耦）；新增4个盘后搜索节点和盘后报告模板 |
| `nasdaq_agent/graph.py` | 新增 `build_afterhours_graph()` |
| `nasdaq_agent/state.py` | 无改动 |
| `service.py` | 新增盘后图初始化、调度任务、`POST /nasdaq/trigger/afterhours` 接口 |

---

## 板块定义（11个板块，覆盖全部100只成分股）

| 板块名 | 股票数 | 成分股 |
|--------|--------|--------|
| 半导体 | 20 | NVDA, AMD, AVGO, QCOM, INTC, TXN, AMAT, LRCX, KLAC, NXPI, MRVL, MCHP, ADI, ON, MU, ASML, ARM, GFS, SNPS, CDNS |
| 大型科技 | 6 | AAPL, MSFT, GOOGL, GOOG, META, AMZN |
| 软件/SaaS | 14 | ADBE, INTU, PANW, CRWD, WDAY, DDOG, ZS, FTNT, OKTA, TEAM, CTSH, ROP, ANSS, CSGP |
| 互联网/电商 | 11 | BKNG, ABNB, EBAY, MELI, PDD, DASH, RBLX, ZM, MTCH, TTD, APP |
| 医疗健康 | 11 | ISRG, VRTX, GILD, REGN, BIIB, IDXX, DXCM, ILMN, MRNA, ALGN, GEHC |
| 消费/零售 | 9 | COST, SBUX, ORLY, DLTR, ROST, MDLZ, MNST, KDP, KHC |
| 媒体/娱乐 | 6 | NFLX, CHTR, WBD, SIRI, EA, TTWO |
| 工业/物流 | 10 | ADP, PAYX, PCAR, ODFL, CSX, FAST, VRSK, CTAS, CPRT, LIN |
| 金融科技 | 2 | PYPL, COIN |
| 新能源/公用 | 7 | CEG, ENPH, EXC, XEL, BKR, FANG, AEP |
| 新兴科技/AI | 4 | TSLA, PLTR, AXON, SMCI |

**边界处理**：板块股票数 N < 10 时，涨/跌各展示 min(10, N) 支。金融科技仅2只，按涨跌幅排序全部展示。

---

## 架构设计

### 1. 行情节点重构（`fetch_stock_movers`）

盘前盘后共用同一节点，东方财富 API 返回的是当前时段价格（盘前时段=盘前价，盘后时段=盘后价），无需区分。

输出格式（`stock_movers` 字段）：

```
▎半导体（涨幅前10）
| 代码 | 价格 | 涨跌幅 |
|:---|---:|---:|
| NVDA | $950.00 | +3.50% |
...

▎半导体（跌幅前10）
| 代码 | 价格 | 涨跌幅 |
...

▎大型科技（涨幅前6）
...
```

### 2. 报告生成重构（LLM叙述与表格解耦）

**旧做法**：`stock_movers` 塞入 LLM system prompt，LLM 负责在输出中原样复现表格 → 容易被截断或改写。

**新做法**：
- LLM 只生成叙述部分（市场概况、三大热点、风险提示），目标 800-1000 字，`max_tokens=800`
- `generate_report` 节点将 LLM 叙述与 `state["stock_movers"]` 表格**程序拼接**后写入 `report_content`
- `MAX_REPORT_CHARS` 改为只限制 LLM 叙述，全报告不做截断（全报告预计 3000-6000 字）

拼接结构：
```
{llm_narrative}

---

📈 板块涨跌榜

{stock_movers}

来源：Reuters/CNBC/MarketWatch
```

### 3. 盘前/盘后 LLM 叙述部分格式

**盘前叙述模板**（LLM 只生成此部分）：
```
【纳斯达克100盘前日报】{date}

📊 市场概况
（2-3句：纳指期货方向、整体情绪）

🔥 盘前三大热点
1.
2.
3.

⚠️ 风险提示
（1-2点）
```

**盘后叙述模板**（LLM 只生成此部分）：
```
【纳斯达克100盘后日报】{date}

📊 收盘概况
（2-3句：纳指收盘涨跌、当日走势）

🔥 盘后三大焦点
1.（财报结果或重大公告）
2.（盘后异动个股）
3.（明日前瞻/关键数据）

⚠️ 关注事项
（1-2点：明日待关注风险或催化剂）
```

### 4. 盘后搜索节点（4路并行）

| 节点名 | 搜索关键词方向 |
|--------|--------------|
| `search_earnings_results` | `tech stocks earnings results after close {date}` |
| `search_afterhours_movers` | `Nasdaq after hours movers stock gains losses {date}` |
| `search_closing_summary` | `Nasdaq 100 stock market closing recap today {date}` |
| `search_tomorrow_preview` | `stock market outlook tomorrow economic calendar preview {date}` |

### 5. 图结构

**盘前图**（现有，`build_nasdaq_graph`，无结构变化）：
```
START → search_tech ─────────┐
      → search_macro          ├─→ generate_report → send_notification → END
      → search_earnings ──────┤
      → search_futures ───────┤
      → fetch_stock_movers ───┘
```

**盘后图**（新增，`build_afterhours_graph`）：
```
START → search_earnings_results ─┐
      → search_afterhours_movers  ├─→ generate_afterhours_report → send_notification → END
      → search_closing_summary ───┤
      → search_tomorrow_preview ──┤
      → fetch_stock_movers ───────┘
```

两个图共用 `fetch_stock_movers` 和 `send_notification` 节点，各自使用不同的报告生成函数。

---

## 报告生成函数拆分

```python
async def _generate_narrative(state, system_prompt) -> str:
    # 共享 LLM 调用逻辑，返回叙述字符串

async def generate_report(state) -> dict:
    # 盘前：调用 _generate_narrative + 拼接 stock_movers

async def generate_afterhours_report(state) -> dict:
    # 盘后：调用 _generate_narrative + 拼接 stock_movers
```

---

## 定时触发与接口设计（`service.py` 扩展）

调度器已在 `service.py` 的 `lifespan` 中初始化（`AsyncIOScheduler`，`America/New_York` 时区）。本次在同一 `lifespan` 中新增盘后图和调度任务，保持一致。

### 触发时间

| 日报 | 触发时间（美东） | 北京时间参考 |
|------|-----------------|-------------|
| 盘前日报 | 每周一至周五 08:00 AM ET（已有） | 冬季 21:00 / 夏季 20:00 |
| 盘后日报 | 每周一至周五 04:30 PM ET（新增） | 冬季次日 05:30 / 夏季次日 04:30 |

### `lifespan` 新增内容

```python
afterhours_graph = build_afterhours_graph()
app_state["afterhours_graph"] = afterhours_graph

scheduler.add_job(
    _run_afterhours_report,
    "cron",
    day_of_week="mon-fri",
    hour=16,
    minute=30,
    args=[afterhours_graph],
)
```

### 新增接口

```
POST /nasdaq/trigger/afterhours
```

立即触发一次盘后日报（用于测试或手动补发），与现有 `POST /nasdaq/trigger`（盘前）对称。

```python
@app.post("/nasdaq/trigger/afterhours")
async def trigger_afterhours_report():
    afterhours_graph = app_state["afterhours_graph"]
    asyncio.create_task(_run_afterhours_report(afterhours_graph))
    return {"status": "triggered", "date": date_type.today().isoformat()}
```

---

## 不在本次范围内

- 发送目标手机号/通知渠道：维持现有配置
- 美股节假日过滤：不涉及（节假日美股不开盘，行情数据为零或无效，报告会自动标注数据不可用）
- 回测或历史数据：不涉及
