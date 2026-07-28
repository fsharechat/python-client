# 盘后日报「异动个股解读 + 财报速览」设计规格

**日期**：2026-07-28
**状态**：已确认

## 背景

现有盘后日报（`nasdaq_agent/generate_afterhours_report`）只输出市场整体叙述 + 板块涨跌幅表格，用户看不到"某只股票为什么涨/跌""财报公司具体数据如何"这类个股层面的信息。本次为盘后日报新增两块个股层面的解读内容，不涉及盘前/盘中日报。

## 目标

- 对当日全市场涨跌幅最大的股票，给出网络搜索得到的涨跌原因。
- 对当日公布财报的纳指100成分股，给出净利润、EPS（实际vs预期）、capex、FCF 等关键财务数据。
- 两条信息在同一只股票上重叠时合并展示，避免重复。

## 技术方案：两条独立 Track + 合并展示

### 新增状态字段（`nasdaq_agent/state.py`）

```python
movers_analysis: list[dict]    # Track A 输出：[{symbol, price, chg, news_snippets}, ...]
earnings_analysis: list[dict]  # Track B 输出：[{symbol, eps_actual, eps_estimate, net_income, capex, fcf, period}, ...]
```

均为普通字段（单一节点写入，无需 `operator.add` reducer）。

### 数据流

```
START
  ├─ search_earnings_results   (并行，现有)
  ├─ search_afterhours_movers  (并行，现有)
  ├─ search_closing_summary    (并行，现有)
  ├─ search_tomorrow_preview   (并行，现有)
  ├─ fetch_stock_movers ──→ analyze_top_movers        (Track A，依赖行情数据)
  └─ find_earnings_reporters                           (Track B，只需日期，与其他节点并行)
         ↓ (全部汇聚)
  generate_afterhours_report   LLM生成叙述 + 合并A/B + 程序拼接表格
         ↓
  send_notification            复用现有节点
         ↓
  END
```

### Track A：`analyze_top_movers`（新增节点）

- 输入：`state["stock_results"]`（`fetch_stock_movers` 输出的全量 `[(sym, price, chg), ...]`）。
- 逻辑：按 `chg` 排序，取全市场涨幅前5 + 跌幅前5（共最多10只）。
- 对这10只股票并发执行 `_ddg_search(f"{sym} stock why {'up' if chg>0 else 'down'} {date}")`（复用现有 `_ddg_search` helper，`asyncio.gather`）。
- 硬过滤（节点内代码判断）：DDG 返回空列表 → 跳过该股，不写入 `movers_analysis`。
- 语义过滤（交给 LLM 判断）：摘要虽非空但与涨跌明显无关的情况，不在节点内用规则判断（不可靠），而是在 `_MOVERS_INSIGHT_SYSTEM` prompt 中明确要求"若提供的新闻摘要无法解释该股票涨跌原因，跳过该股，不要编造原因"，由生成阶段的 LLM 决定是否为该股票生成条目。
- 输出：`{"movers_analysis": [{"symbol": sym, "price": price, "chg": chg, "news_snippets": [...]}, ...]}`（只要有非空搜索结果就写入，最终是否呈现由生成阶段决定）

### Track B：`find_earnings_reporters`（新增节点）

- 输入：仅 `state["date"]`，不依赖行情数据，可与 `fetch_stock_movers` 等 START 级节点并行。
- 逻辑：
  1. 调用 Finnhub `/calendar/earnings?from={date}&to={date}`，取当日财报日历。
  2. 筛选出 `symbol` 属于 `NASDAQ100_TICKERS` 的条目。
  3. 若为空列表，Track B 直接返回空结果（当日无纳指100成分股发财报是常态）。
  4. 对每个匹配到的 symbol：
     - 调用 Finnhub `/stock/earnings?symbol={sym}` 取最近一期 EPS 实际值 vs 预期值（`actual`/`estimate`/`period`）。
     - 调用 Finnhub `/stock/financials-reported?symbol={sym}`，从最新一期 `report.ic`（利润表）解析净利润（`NetIncomeLoss` concept），从 `report.cf`（现金流量表）解析经营性现金流（`NetCashProvidedByUsedInOperatingActivities`）和资本支出（`PaymentsToAcquirePropertyPlantAndEquipment`），`FCF = 经营性现金流 - capex`。
     - 任一字段解析失败（concept 未匹配到、接口报错）→ 该字段留空，不编造数字，沿用现有降级风格（记录日志、继续执行）。
  5. 请求节流：复用 `_fetch_finnhub_quotes` 的顺序+1秒间隔模式，避免触发 Finnhub 免费额度限速。
- 输出：`{"earnings_analysis": [{"symbol": sym, "eps_actual": ..., "eps_estimate": ..., "net_income": ..., "capex": ..., "fcf": ..., "period": ...}, ...]}`

### 合并与报告生成（`generate_afterhours_report` 内新增逻辑）

1. 以 symbol 为 key，将 `movers_analysis` 与 `earnings_analysis` 合并：
   - 同时出现在两边的 symbol → 单条目，新闻原因 + 财报数据一起呈现。
   - 只在 `movers_analysis` → 只有新闻原因。
   - 只在 `earnings_analysis` → 归入独立的"财报速览"分组（不在异动个股解读里出现）。
2. 新增一次 LLM 调用（新 system prompt `_MOVERS_INSIGHT_SYSTEM`），输入合并后的结构化数据，输出**精简条目式**文本，每只股票1-2句：

```
🔍 异动个股解读
• NVDA +8.2%：Q3营收超预期，数据中心业务同比+120%；净利润$xx亿，EPS $x.xx（预期$x.xx），FCF $xx亿
• XXX -5.1%：分析师下调评级，云业务增速放缓担忧

📑 今日财报速览
• AAPL：净利润$xx亿，EPS $x.xx（预期$x.xx），capex $xx亿，FCF $xx亿
```

   - 若 Track A 和 Track B 合并结果均为空（当日无匹配新闻也无财报公布）→ 两个 section 都不生成，报告退化为现状。
   - 若只有一个 section 有内容 → 只附加那一个。
3. 插入位置：主叙述之后、板块涨跌榜（`_sector_movers_table`）之前。
4. 字数预算：主叙述维持1500字上限；新增两个 section 合计预算约500-800字。

### 容错

延续项目现有风格：DDG 搜索超时/无结果、Finnhub 请求失败、financials-reported 字段解析失败，均只记录日志（`print(f"[nasdaq] ...")`）并跳过对应股票/字段，不中断整体报告生成流程，不影响现有行情表格和板块涨跌榜。

## 改动清单

### `nasdaq_agent/state.py`
1. 新增字段：`movers_analysis: list[dict]`、`earnings_analysis: list[dict]`

### `nasdaq_agent/nodes.py`
1. 新增节点函数：`analyze_top_movers(state)`（Track A）
2. 新增节点函数：`find_earnings_reporters(state)`（Track B），含 Finnhub 财报日历 + `/stock/earnings` + `/stock/financials-reported` 调用与解析
3. 新增常量：`_MOVERS_INSIGHT_SYSTEM`
4. 新增函数：`_merge_movers_and_earnings(movers_analysis, earnings_analysis) -> dict`（合并逻辑）
5. 修改 `generate_afterhours_report`：合并数据 → 调用新 LLM prompt 生成两个 section → 插入报告拼接顺序

### `nasdaq_agent/graph.py`
1. 修改 `build_afterhours_graph()`：新增 `analyze_top_movers`、`find_earnings_reporters` 节点，调整边：
   - `fetch_stock_movers` → `analyze_top_movers` → `generate_afterhours_report`
   - `START` → `find_earnings_reporters` → `generate_afterhours_report`

### `service.py`
无需改动（沿用现有 `_run_afterhours_report` / `/nasdaq/trigger/afterhours`）。

## 测试

沿用项目"无自动化测试"约定，手动触发验证：

```bash
curl -X POST http://localhost:8000/nasdaq/trigger/afterhours
```

检查日志中 `[nasdaq] analyze_top_movers ...` 和 `[nasdaq] find_earnings_reporters ...` 的输出，以及最终推送内容中新增 section 的格式与数据准确性。建议挑选一个已知有纳指100成分股当日发财报的交易日测试 Track B。

## 不在本次范围内

- 盘前日报、盘中日报的任何改动
- Track A 的股票范围调整（超出全市场前5+5）
- 财报数据源切换为 financialdatasets.ai 或 yfinance
- 历史财报趋势对比（同比/环比），仅取最新一期
