# 飞享IM 智能问答服务

基于 **Python + LangChain + LangGraph + Claude** 构建的 RAG 智能问答系统，专门回答与[飞享IM](https://fsharechat.cn)相关的问题。

---

## 目录

- [项目结构](#项目结构)
- [技术架构](#技术架构)
- [环境要求](#环境要求)
- [安装](#安装)
- [配置](#配置)
- [运行](#运行)
- [测试](#测试)
- [API 文档](#api-文档)

---

## 项目结构

```
python_client/
├── config.py          # 全局配置（模型、路径、知识库内容）
├── ingest.py          # 数据摄入（爬取官网 + 静态知识 → ChromaDB 向量库）
├── graph.py           # LangGraph 工作流（问答核心逻辑）
├── service.py         # FastAPI HTTP 服务（/ask、/stream 接口）
├── cli.py             # 命令行交互界面
├── requirements.txt   # Python 依赖
├── .env.example       # 环境变量模板
└── README.md
```

### 各文件职责

| 文件 | 职责 |
|------|------|
| `config.py` | 统一管理 API Key、模型名、向量库路径、爬取 URL、静态知识文本 |
| `ingest.py` | 爬取飞享IM官网页面，与静态知识合并后切片，写入 ChromaDB 向量库 |
| `graph.py` | 定义 LangGraph 状态机：classify → retrieve → grade_docs → generate/fallback/reject |
| `service.py` | FastAPI 应用，暴露 REST 接口，支持普通响应和 SSE 流式响应 |
| `cli.py` | 纯命令行对话入口，无需启动 HTTP 服务 |

---

## 技术架构

### 整体流程

```
用户问题
    │
    ▼
┌─────────────┐
│  classify   │  Claude 判断问题是否与飞享IM相关
└──────┬──────┘
       │
  ┌────┴────┐
  │         │
on_topic  off_topic
  │         │
  ▼         ▼
┌────────┐ ┌────────┐
│retrieve│ │ reject │ → Claude Sonnet 4.6 直接回答通用问题
└───┬────┘ └────────┘
    │
    ▼
┌───────────┐
│ grade_docs│  Claude 逐块过滤不相关文档
└─────┬─────┘
      │
 ┌────┴────┐
 │         │
有相关内容  无相关内容
 │         │
 ▼         ▼
┌────────┐ ┌──────────┐
│generate│ │ fallback │ → 建议访问官网
└────────┘ └──────────┘
    │
    ▼
  答案
```

### 关键技术选型

| 组件 | 技术 | 说明 |
|------|------|------|
| LLM | Claude Opus 4.7（Anthropic） | 问题分类、文档评分、答案生成，启用 adaptive thinking |
| 向量嵌入 | `BAAI/bge-small-zh-v1.5`（本地） | 中文优化嵌入模型，无需额外 API Key |
| 向量数据库 | ChromaDB（本地持久化） | 存储飞享IM知识库的文档向量 |
| 工作流编排 | LangGraph | 状态机驱动的多节点 RAG 流程 |
| Web 框架 | FastAPI | REST + SSE 流式接口 |
| 知识来源 | 官网爬取 + 静态知识库 | 双源融合，保证覆盖率 |

---

## 环境要求

- Python 3.11+
- 网络访问（首次运行需下载嵌入模型，约 100MB）
- Anthropic API Key（[申请地址](https://console.anthropic.com/)）

---

## 安装

```bash
# 1. 克隆或进入项目目录
cd python_client

# 2. 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

> 首次安装 `sentence-transformers` 时会自动下载嵌入模型，需保持网络畅通。

---

## 配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env，填入你的 Anthropic API Key
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx
```

其他可选配置项在 `config.py` 中修改：

```python
CLAUDE_MODEL = "claude-opus-4-7"      # 使用的 Claude 模型
CHROMA_PERSIST_DIR = "./chroma_db"    # 向量库存储目录
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 嵌入模型
```

---

## 运行

### 方式一：命令行对话（推荐快速体验）

```bash
python cli.py
```

首次运行会自动构建向量库（约 1-2 分钟），之后直接进入对话：

```
加载知识库...

============================================================
  飞享IM 智能问答助手（输入 q 退出）
============================================================

您好，请问有什么关于飞享IM的问题？
> 飞享IM支持哪些部署方式？

思考中...

助手：飞享IM支持以下三种部署方式：
1. 一键脚本部署 - 平台提供脚本，适合快速上手
2. Docker 容器化 - 使用 Docker Compose 编排，简单快捷
3. Kubernetes - 生产级容器编排，适合大规模部署
...
```

### 方式二：HTTP 服务

```bash
python service.py
```

服务启动后监听 `http://0.0.0.0:8000`，可通过以下方式访问：

#### 后台运行（生产/服务器环境）

```bash
# 启动（日志写入 service.log）
nohup python3.11 service.py > service.log 2>&1 &

# 查看实时日志
tail -f service.log

# 查看进程 PID
pgrep -fa "service.py"

# 重启（停止旧进程后重新启动）
pkill -f "service.py" && sleep 1 && nohup python3.11 service.py > service.log 2>&1 &
```

```bash
# 普通问答
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "飞享IM的音视频通话基于什么技术？"}'

# 流式问答（SSE）
curl -N -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "飞享IM有哪些核心功能？"}'

# 健康检查
curl http://localhost:8000/health
```

### 方式三：单独构建向量库

```bash
# 仅执行数据摄入，不启动服务
python ingest.py
```

---

## 测试

### 手动测试用例

以下用例覆盖工作流的所有分支：

#### 1. 正常问答（on_topic → generate）

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "飞享IM支持哪些客户端平台？"}'
```

预期：返回 Android、iOS、Windows、Mac、Web 客户端信息，`route` 为 `on_topic`。

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "飞享IM如何部署？需要什么服务器配置？"}'
```

预期：返回部署方式和服务器配置要求。

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "飞享IM的音视频通话是基于什么技术实现的？"}'
```

预期：返回 WebRTC 相关说明。

#### 2. 超出知识库范围（on_topic → fallback）

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "飞享IM的月活用户数量是多少？"}'
```

预期：返回 fallback 提示，建议访问官网。

#### 3. 无关问题（off_topic → reject）

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "今天天气怎么样？"}'
```

预期：由 Claude Sonnet 4.6 直接回答通用问题，`route` 为 `off_topic`。

#### 4. 流式输出测试

```bash
curl -N -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "飞享IM的后端技术栈有哪些？"}'
```

预期：SSE 事件流逐步输出答案，最后收到 `data: [DONE]`。

#### 5. 健康检查

```bash
curl http://localhost:8000/health
# 预期: {"status": "ok"}
```

### 自动化测试（pytest）

安装测试依赖后运行：

```bash
pip install pytest httpx

pytest tests/ -v
```

> 当前项目不含 `tests/` 目录，如需添加，建议使用 `httpx.AsyncClient` 测试 FastAPI 接口，并 mock `ChatAnthropic` 以避免实际调用 API。

---

## API 文档

服务启动后访问交互式文档：

- Swagger UI：`http://localhost:8000/docs`
- ReDoc：`http://localhost:8000/redoc`

### POST /ask

同步问答接口。

**请求体**

```json
{
  "question": "飞享IM支持哪些部署方式？"
}
```

**响应**

```json
{
  "question": "飞享IM支持哪些部署方式？",
  "answer": "飞享IM支持三种部署方式：一键脚本部署、Docker 容器化和 Kubernetes...",
  "route": "on_topic"
}
```

### POST /stream

流式问答接口，返回 Server-Sent Events。

**请求体**：同 `/ask`

**响应**（SSE 事件流）

```
data: 飞享IM

data: 支持以下

data: 三种部署方式...

data: [DONE]
```

### GET /health

服务存活检查，返回 `{"status": "ok"}`。
