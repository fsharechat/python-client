import os
from dotenv import load_dotenv

load_dotenv()

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-opus-4-7"

# Document cache (BM25 retriever — no embedding model required)
DOCS_PERSIST_PATH = "./docs_cache.json"

# Data sources — pages to scrape for 飞享IM knowledge base
FSHARECHAT_URLS = [
    "https://fsharechat.cn",
    "https://fsharechat.cn/about",
    "https://fsharechat.cn/price",
    "https://fsharechat.cn/download",
]

# Additional static knowledge about 飞享IM (fallback when scraping is incomplete)
FSHARECHAT_STATIC_KNOWLEDGE = """
# 飞享IM (FshareChat) 产品知识库

## 产品定位
飞享IM是一款技术自主可控的即时通讯系统，专为企业私有化部署设计。
口号：让团队沟通更高效，数据更安全。

## 核心功能

### 即时消息
- 支持文字、图片、语音、视频、文件等多种消息格式
- 消息多端实时同步（手机、电脑、网页同步）
- 消息转发、撤回、引用回复
- 消息已读未读状态显示
- @提及功能

### 音视频通话
- 一对一音频通话
- 一对一视频通话
- 多人视频会议
- 基于 WebRTC 技术开发，自主可控
- 支持屏幕共享

### 群组功能
- 创建群组，支持大群
- 群组成员管理（拉人、踢人）
- 群主权限管理
- 群公告功能
- 群文件共享

### 好友系统
- 添加好友、删除好友
- 好友分组管理
- 在线状态显示

## 技术架构

### 后端技术栈
- SpringBoot 微服务框架
- tio 高性能网络框架（处理 IM 长连接）
- Dubbo RPC 服务间通信
- MySQL 数据存储
- Redis 缓存
- MinIO 对象存储（图片、文件）

### 前端技术栈
- Vue.js 前端框架
- Electron 跨平台桌面客户端

### 通信协议
- WebSocket 长连接
- WebRTC 音视频

## 部署方案

### 支持部署方式
1. **一键脚本部署** - 平台提供脚本，快速完成部署
2. **Docker 容器化** - Docker Compose 编排，简单快捷
3. **Kubernetes** - 生产级容器编排，适合大规模部署

### 系统要求
- Linux 服务器（推荐 CentOS 7+ 或 Ubuntu 18+）
- 最低 2核4G 内存（小规模使用）
- 推荐 4核8G 以上（企业生产环境）

## 客户端支持
- Android 手机客户端
- iOS 手机客户端
- Windows 桌面客户端
- Mac 桌面客户端
- Web 网页客户端

## 产品优势
1. **数据自主可控** - 私有化部署，数据不出企业内网
2. **安全性高** - 消息加密传输，保障通讯安全
3. **开源可定制** - 支持二次开发，满足个性化需求
4. **技术自主** - 核心技术自研，不依赖第三方 IM 服务
5. **扩展性好** - 微服务架构，支持横向扩展

## 适用场景
- 企业内部即时通讯
- 政务内网通讯
- 金融机构内部沟通
- 医疗行业安全通讯
- 教育机构内部协作

## 官网
- 官网：https://fsharechat.cn
- GitHub：开源项目（支持二次开发）
"""
