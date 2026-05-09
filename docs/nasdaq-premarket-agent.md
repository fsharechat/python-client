# 纳斯达克100盘前日报 Agent 原理说明

## 概述

本模块（`nasdaq_agent/`）基于 **LangGraph 状态机**实现，每个工作日美东时间 8:00 AM 自动运行，从多个渠道并行采集信息，生成中文盘前日报，通过飞享IM推送接口发送到指定手机。

---

## 整体架构

```
START
  ├── search_tech_news    ─┐
  ├── search_macro_news   ─┤
  ├── search_earnings     ─┼──► generate_report ──► send_notification ──► END
  ├── search_futures      ─┤
  └── fetch_stock_movers  ─┘
```

五个节点从 START **并行启动**，全部完成后汇入 `generate_report`（LangGraph fan-out / fan-in 机制）。

---

## 模块文件说明

```
nasdaq_agent/
├── __init__.py
├── state.py      # LangGraph 状态定义
├── tickers.py    # 纳斯达克100成分股列表（100支）
├── nodes.py      # 各节点函数实现
└── graph.py      # 状态机图构建
```

---

## LangGraph 状态设计

```python
class NasdaqReportState(TypedDict):
    date: str                                      # 日期，如 "2026-05-08"
    raw_articles: Annotated[list[dict], operator.add]  # 四路搜索结果合并（reducer）
    stock_movers: str                              # 涨跌幅榜（markdown 格式）
    report_content: str                            # 最终生成的中文日报
    send_status: str                               # 通知发送状态
```

`raw_articles` 使用 `operator.add` 作为 reducer，四个搜索节点各自返回的文章列表会自动合并，无需加锁或手动聚合。

---

## 各节点详解

### 1. 并行搜索节点（4个）

使用 `httpx.AsyncClient` 异步抓取 DuckDuckGo HTML 页面（`html.duckduckgo.com/html/`），用 BeautifulSoup 解析结果。

| 节点 | 搜索主题 |
|------|---------|
| `search_tech_news` | 纳斯达克100科技股盘前动态 |
| `search_macro_news` | 美联储利率、宏观经济数据 |
| `search_earnings` | 科技股财报预告与盈利情况 |
| `search_futures` | QQQ/纳指期货盘前走势 |

选用 DuckDuckGo 的原因：免费、无需 API Key、无严格限速。

**为何不用 ddgs 库？**  
`ddgs`（原 `duckduckgo-search`）底层依赖 `primp`，一个 Rust 实现的 HTTP 客户端，自带 TLS 实现，与部分 Linux 服务器的 OpenSSL 不兼容，会触发 Segmentation fault。改用 `httpx` 直接抓取 HTML 彻底规避了此问题。

### 2. 股票行情节点（fetch_stock_movers）

调用**东方财富 push2 批量行情接口**获取纳斯达克100全部成分股的实时价格和涨跌幅。

**接口信息：**

```
GET https://push2.eastmoney.com/api/qt/ulist.np/get
参数：
  fltt=2          价格保留原始精度
  invt=2
  fields=f12,f2,f3  f12=股票代码，f2=最新价，f3=涨跌幅%
  secids=105.AAPL,105.MSFT,...  纳斯达克前缀为 105
```

**选用东方财富的原因：**
- Yahoo Finance 对国内服务器 IP 触发 429 限速（Too Many Requests）
- `yfinance` 库底层同样依赖 `primp`，存在 TLS 兼容问题
- 东方财富 push2 接口国内直连稳定，无需认证，单次请求返回全部100支股票数据

**盘前数据说明：**  
美股盘前时段（ET 4:00–9:30 AM），东方财富 `f2` 字段直接反映盘前最新成交价，`f3` 为相对前收盘的涨跌幅。

输出格式（markdown 表格）：

```markdown
**涨幅前十（实时行情）**

| 代码 | 价格 | 涨跌幅 |
|:---|---:|---:|
| NVDA | $211.50 | +1.77% |
...

**跌幅前十（实时行情）**
...
```

### 3. 报告生成节点（generate_report）

收集到所有搜索文章和股票数据后，调用 Claude（`claude-sonnet-4-6`）生成中文日报。

- 取前 20 条搜索结果，每条摘要截断为 200 字，避免 token 超限
- 股票涨跌榜直接嵌入 System Prompt 的固定格式模板中
- 最终报告限制在 **2048 字符**以内（通知接口限制）

日报固定格式：
```
【纳斯达克100盘前日报】{date}

📊 市场概况
📊 盘前三大热点
📈 盘前涨跌幅前十
⚠️ 风险提示
```

### 4. 通知发送节点（send_notification）

通过飞享IM开放平台接口将日报推送到指定手机号：

```http
POST https://backend-http.fsharechat.cn/imopenapi/pushNotificationByMobile
Content-Type: application/json

{
  "mobiles": [13900000001],
  "content": {
    "content": {
      "type": 1,
      "searchableContent": "<日报内容>"
    }
  }
}
```

---

## 调度机制

使用 **APScheduler（AsyncIOScheduler）** 集成在 FastAPI `lifespan` 中，随服务启动自动注册定时任务：

```python
scheduler = AsyncIOScheduler(timezone="America/New_York")
scheduler.add_job(
    _run_nasdaq_report,
    "cron",
    day_of_week="mon-fri",
    hour=8,
    minute=0,
)
```

- `timezone="America/New_York"` 自动处理美国夏令时（EDT/EST），无需手动维护 UTC 偏移
- 每个工作日 ET 8:00 AM 触发，对应北京时间：夏令时 20:00 / 冬令时 21:00

---

## 手动触发接口

开发调试或手动补发时，可调用以下接口立即执行一次：

```bash
curl --noproxy localhost -X POST http://localhost:8000/nasdaq/trigger
# 响应: {"status": "triggered", "date": "2026-05-08"}
```

任务在后台异步执行，接口立即返回。

---

## 关键设计决策

### 为何全部节点使用 async def？

LangGraph 在执行并行节点时：
- 同步节点（`def`）→ 用 `ThreadPoolExecutor` 并发，多线程下 `primp`/`curl_cffi` 会 segfault
- 异步节点（`async def`）→ 用 asyncio 并发，无线程，彻底避免上述问题

所有节点改为 `async def` 后，通过 `graph.ainvoke()` 调用，5个并行节点通过 asyncio 事件循环协调，无任何线程竞争。

### 为何 httpx 客户端设置 trust_env=False？

东方财富接口请求中加入 `trust_env=False`，防止服务器上配置的 HTTP 代理（如透明代理或 VPN）干扰对东方财富的直连请求，确保数据稳定获取。

---

## 日志输出示例

```
[nasdaq] Starting daily report for 2026-05-08 ...
[nasdaq] search_tech query: Nasdaq 100 technology stocks premarket 2026-05-08
[nasdaq] search_macro query: Federal Reserve interest rate US economy market 2026-05-08
[nasdaq] search_earnings query: tech stocks earnings results premarket today 2026-05-08
[nasdaq] search_futures query: QQQ Nasdaq futures premarket market open 2026-05-08
[nasdaq] fetch_stock_movers: fetching 100 tickers via East Money ...
[nasdaq] East Money HTTP 200, body[:200]: {"data":{"diff":[...
[nasdaq] search_tech: 3.21s → 5 results
[nasdaq] search_macro: 2.87s → 5 results
[nasdaq] search_earnings: 3.54s → 4 results
[nasdaq] search_futures: 2.96s → 5 results
[nasdaq] fetch_stock_movers: 1.43s → 98 tickers (实时行情)
[nasdaq] generate_report: 8.12s → 1876 chars
[nasdaq] send_notification: 0.34s → ok:200
[nasdaq] Daily report complete.
```

---

## 依赖

| 库 | 用途 |
|----|------|
| `langgraph` | 状态机编排，fan-out/fan-in 并行执行 |
| `langchain-anthropic` | 调用 Claude 生成日报 |
| `httpx` | 异步 HTTP 客户端（搜索 + 行情 + 通知） |
| `beautifulsoup4` + `lxml` | 解析 DuckDuckGo HTML 搜索结果 |
| `apscheduler` | 定时任务调度 |
| `pytz` | 时区处理（美东时间 DST） |
