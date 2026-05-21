# 盘中开盘日报设计规格

**日期**：2026-05-21  
**状态**：已确认

## 背景

现有系统在 ET 08:00（盘前日报）和 ET 16:30（盘后日报）各发送一次美股日报。本次新增**盘中开盘日报**，在美股正式开盘15分钟后（ET 09:45）发送，抓取实际开盘行情与盘后第一批新闻。

## 目标

- 向用户推送开盘15分钟的实际走势（对比盘前预期）
- 汇总开盘后公布的经济数据与重大公司公告
- 与现有盘前/盘后日报形成完整的全天三段式覆盖

## 技术方案：方案A（最小改动，复用盘后管线）

### 新增报告类型

`report_type = "intraday"`，驱动现有 `fetch_stock_movers` 走盘后分支（Finnhub 主力 + 东方财富兜底）。

### 数据流

```
START
  ├─ search_opening_movers      (并行) 开盘异动股搜索
  ├─ search_morning_economics   (并行) 开盘后经济数据/公告搜索
  ├─ search_opening_news        (并行) 开盘综合新闻搜索
  └─ fetch_stock_movers         (并行) report_type="intraday" → Finnhub+EM双源
         ↓ (全部汇聚)
  generate_intraday_report      LLM生成叙述 + 程序拼接表格
         ↓
  send_notification             复用现有节点
         ↓
  END
```

### 行情数据策略

与盘后报告完全一致：
- `fetch_stock_movers` 内部判断：`if report_type in ("afterhours", "intraday")` 触发 Finnhub+EM 双源并发
- Finnhub 顺序查询全量 NASDAQ100（101只）+ QQQ/SPY，限速1次/秒，约103秒；与 East Money 批量请求并发执行
- 个股：Finnhub 为主，East Money 为备
- 指数（QQQ/SPY）：Finnhub 为主，East Money 为备
- 价格列标签：`"实时价"`（区别于盘前 `"盘前价"` 和盘后 `"收盘价"`）

### 新闻搜索节点（3路并发）

| 节点 | 查询语句 |
|---|---|
| `search_opening_movers` | `"Nasdaq stocks gap up gap down market open {date}"` |
| `search_morning_economics` | `"US economic data release morning CPI jobs {date}"` |
| `search_opening_news` | `"stock market open recap early trading {date}"` |

注：盘前日报用4路搜索，盘中日报用3路（开盘期新闻较集中，3路已足够覆盖）。

### LLM 系统提示（`_INTRADAY_SYSTEM`）

```
你是专业美股分析师，擅长整理纳斯达克100开盘动态。
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
（1-2点：今日盘中重要时间节点或待关注催化剂）
```

### 报告结构（程序拼接）

```
{LLM叙述}
---
📊 主要指数（实时价）
{QQQ/SPY 实时价表格，来自 index_summary}
---
💎 美股七姐妹开盘
{MAG7 实时价表格，price_label="实时价"}
📈 板块涨跌榜（开盘15分钟）
{_sector_movers_table(stock_results, price_label="实时价")}
来源：Reuters/CNBC/MarketWatch
```

注：MAG7 数据从 `stock_results` 中过滤（MAG7 均在 NASDAQ100 内），无需单独获取。

## 改动清单

### `nasdaq_agent/nodes.py`

1. 新增搜索函数：`search_opening_movers`、`search_morning_economics`、`search_opening_news`
2. 新增常量：`_INTRADAY_SYSTEM`
3. 新增函数：`generate_intraday_report()`
4. 修改 `fetch_stock_movers`：`if report_type == "afterhours"` → `if report_type in ("afterhours", "intraday")`
5. 修改 `_build_fh_index_table`：标签逻辑加 `"intraday"` → `"实时价"` 分支
6. 修改 `_mag7_table`：加 `price_label: str = "盘前价"` 参数，盘中报告传 `"实时价"`；MAG7数据从 `stock_results` 过滤获取（MAG7均在NASDAQ100内）

### `nasdaq_agent/graph.py`

1. 新增函数：`build_intraday_graph()`，导入新增节点函数

### `service.py`

1. 新增：`_run_intraday_report()` 异步函数
2. 在 `lifespan` 中构建并存储 `intraday_graph`
3. 新增调度任务：ET 09:45，周一至周五
4. 新增接口：`POST /nasdaq/trigger/intraday`

## 调度时间

| 报告 | 时间（ET）| 说明 |
|---|---|---|
| 盘前日报 | 08:00 | 现有，不变 |
| 开盘日报 | 09:45 | 新增 |
| 盘后日报 | 16:30 | 现有，不变 |

## 不在本次范围内

- MAG7 单独 Finnhub 查询（已在 stock_results 内，直接过滤即可）
- 盘前/盘后报告的任何改动
- 新的数据源接入
