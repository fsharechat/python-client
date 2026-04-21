# 智能客服编排层 (Orchestrator)

统一入口，自动将用户请求路由到对应的下游 Agent。

```
用户请求
   │
   ▼
Orchestrator :8001
   │
   ├─ 产品FAQ ──► Python Q&A Agent :8000
   └─ 订单/退款 ► Java LangChain4j Agent :8080
```

## 依赖服务

| 服务 | 端口 | 说明 |
|------|------|------|
| Python Q&A Agent | 8000 | 飞享IM 产品问答（本项目根目录） |
| Java Order Agent | 8080 | 订单查询、退款处理（`smart-customer-service`） |
| Anthropic API | — | 意图分类（Claude Haiku） |

---

## 安装

### 1. 前置要求

- Python 3.10+
- 已配置好 Python Q&A Agent（根目录 `service.py` 可正常启动）
- 已配置好 Java Order Agent（`smart-customer-service` 可正常启动）

### 2. 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

`httpx` 是 Orchestrator 新增的依赖，其余均与主项目共用。

### 3. 配置环境变量

在项目根目录的 `.env` 文件中追加以下配置（参考 `.env.example`）：

```env
# Anthropic API Key（已存在，无需重复添加）
ANTHROPIC_API_KEY=sk-ant-api03-...

# 下游服务地址（默认值如下，按实际情况修改）
PYTHON_QA_AGENT_URL=http://localhost:8000
JAVA_ORDER_AGENT_URL=http://localhost:8080

# Orchestrator 端口（默认 8001）
ORCHESTRATOR_PORT=8001

# 意图分类模型（默认使用 Haiku，速度快、成本低）
ORCHESTRATOR_MODEL=claude-haiku-4-5-20251001
```

---

## 启动

三个服务需分别在独立终端中启动。

### 终端 1 — Python Q&A Agent

```bash
cd /path/to/python_client
source .venv/bin/activate
python service.py
# 监听 http://localhost:8000
```

### 终端 2 — Java Order Agent

```bash
cd /path/to/smart-customer-service
mvn spring-boot:run
# 监听 http://localhost:8080
```

### 终端 3 — Orchestrator

```bash
cd /path/to/python_client
source .venv/bin/activate
python -m orchestrator.service
# 监听 http://localhost:8001
```

启动成功后输出：

```
Orchestrator ready.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

---

## API

### POST /chat

向 Orchestrator 发送用户消息，自动路由并返回回答。

**请求**

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "飞享IM支持哪些平台？"}'
```

**请求体**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `message` | string | 是 | 用户输入 |
| `session_id` | string | 否 | 会话 ID，用于与 Java Agent 保持多轮对话 |

**响应体**

```json
{
  "message": "飞享IM支持哪些平台？",
  "answer": "飞享IM 支持 Android、iOS、Windows、macOS 和 Web 端...",
  "intent": "faq",
  "source": "faq_agent",
  "session_id": ""
}
```

| 字段 | 说明 |
|------|------|
| `intent` | `faq` — 产品问答；`order_ops` — 订单/退款操作 |
| `source` | `faq_agent` — 来自 Python Agent；`order_agent` — 来自 Java Agent |

### GET /health

检查 Orchestrator 及下游服务的可用状态。

```bash
curl http://localhost:8001/health
```

```json
{
  "status": "ok",
  "downstream": {
    "faq_agent": "ok",
    "order_agent": "ok"
  }
}
```

下游状态值：`ok` / `degraded` / `unreachable`

---

## 路由规则

Orchestrator 使用 Claude Haiku 对用户输入做意图分类：

| 用户输入示例 | 路由目标 |
|-------------|---------|
| 飞享IM 支持哪些平台？ | `faq_agent` |
| 怎么部署飞享IM 服务端？ | `faq_agent` |
| 我的订单 ORD2024001 到哪里了？ | `order_agent` |
| 我要申请退款 | `order_agent` |
| 这个商品还有货吗？ | `order_agent` |
| 我的退款进度怎么样？ | `order_agent` |

---

## 验证测试

```bash
# FAQ 路由
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "飞享IM支持哪些平台？"}'
# 期望: intent="faq", source="faq_agent"

# 订单路由
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我的订单ORD2024001到哪里了？", "session_id": "test-001"}'
# 期望: intent="order_ops", source="order_agent"

# 退款路由
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我要申请退款，订单号ORD2024002"}'
# 期望: intent="order_ops", source="order_agent"

# 健康检查
curl http://localhost:8001/health
```

---

## 目录结构

```
orchestrator/
├── README.md       本文档
├── __init__.py
├── config.py       环境变量读取与默认值
├── clients.py      调用下游 Agent 的 HTTP 客户端
├── graph.py        LangGraph 状态机（分类 → 路由 → 调用）
└── service.py      FastAPI 服务入口
```
