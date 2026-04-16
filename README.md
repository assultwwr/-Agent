# 🤖 智扫通 - 扫地机器人智能客服 Agent

基于 **RAG (检索增强生成)** 和 **ReAct Agent** 架构的智能客服系统，专为扫地机器人产品打造。

## 一、核心特性

- **RAG 知识库**：支持 PDF/TXT/DOCX 格式文档自动解析、向量化存储与智能检索
- **智能 Agent**：基于 LangGraph + MCP 协议的工具调用系统
- **多模型支持**：兼容 Ollama 本地模型与阿里云 API
- **向量数据库同步**：文件增删改自动同步至 Milvus，保持知识库一致性
- **混合检索**：BM25 + 向量相似度 + Rerank 重排序，提升检索准确率
- **聊天历史管理**：MongoDB 持久化存储，支持多轮对话上下文
- **高德地图集成**：天气查询与自动位置定位服务
- **容器化部署**：Docker Compose 一键启动

## 二、系统架构

```
┌─────────────┐     HTTP     ┌──────────────┐     FastAPI     ┌────────────────┐
│  Streamlit  │ ◄──────────► │   Backend    │ ◄──────────────► │  Agent Core    │
│  Frontend   │   (8501)     │  FastAPI     │    (8000)        │  LangGraph     │
└─────────────┘              └──────────────┘                   └────────────────┘
                                                                    │
                                                    ┌───────────────┼───────────────┐
                                                    ▼               ▼               ▼
                                             ┌──────────┐   ┌──────────┐   ┌──────────────┐
                                             │  Milvus  │   │ MongoDB  │   │  Ollama/     │
                                             │ (向量库) │   │ (历史)   │   │  阿里云      │
                                             └──────────┘   └──────────┘   └──────────────┘
```

## 三、技术栈

| 类别 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| 前端框架 | Streamlit |
| Agent 框架 | LangGraph + LangChain |
| 向量数据库 | Milvus |
| 聊天历史 | MongoDB |
| 模型服务 | Ollama (qwen2.5:7b, qwen3-embedding) / 阿里云 |
| 重排序模型 | BAAI-bge-reranker-v2-m3 |
| 部署方案 | Docker Compose |

## 四、快速开始

### 前置要求

#### Docker 部署方式
- **Docker & Docker Compose**（推荐）
- **Ollama** 已安装并运行模型，或使用阿里云 API Key
- **Rerank 模型**：需手动下载（见下方说明）

#### 本地开发方式
- **Python 3.11+**
- **MongoDB** 已安装并运行
- **Milvus** 已安装并运行
- **Ollama** 已安装并运行模型，或使用阿里云 API Key
- **Rerank 模型**：需手动下载（见下方说明）

### 1. 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填写实际配置
# - 高德地图 API Key（天气查询功能）
# - Ollama 服务地址或阿里云配置
# - MongoDB / Milvus 连接地址
```

### 2. 下载 Rerank 模型（重要）

由于模型文件较大（约2GB），GitHub不包含此文件，需手动下载：

#### 方式一：从 HuggingFace 下载（推荐）

```bash
# 安装 git-lfs（如果未安装）
# Windows: https://git-lfs.com
# macOS: brew install git-lfs
# Linux: sudo apt-get install git-lfs

# 下载模型到项目目录
cd model
git lfs install
git clone https://huggingface.co/BAAI/bge-reranker-v2-m3 BAAI-bge-reranker-v2-m3
```

#### 方式二：手动下载

1. 访问：https://huggingface.co/BAAI/bge-reranker-v2-m3
2. 下载所有文件到 `model/BAAI-bge-reranker-v2-m3/` 目录
3. 确保包含以下关键文件：
   - `model.safetensors` （模型权重）
   - `config.json`
   - `tokenizer.json`
   - `tokenizer_config.json`

#### 方式三：使用阿里云在线模型（无需下载）

修改 `.env` 文件：
```env
RERANK_MODEL_PATH=BAAI/bge-reranker-v2-m3  # 使用 HuggingFace 在线模型
```

### 3. 准备知识库文件

将文档放入 `data/` 目录：
```
data/
├── 选购指南.txt
├── 故障排除.docx
├── 扫地机器人100问.pdf
└── ...
```

**支持格式**：`.txt`, `.docx`, `.pdf`

### 3. 启动服务

```bash
# 构建并启动所有服务
docker-compose build
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 4. 访问应用

- **前端界面**：http://localhost:8501
- **后端 API**：http://localhost:8000
- **健康检查**：http://localhost:8000/health

## 五、知识库同步机制

系统会自动维护 `data/` 文件夹与 Milvus 向量库的同步：

| 操作 | 行为 |
|------|------|
| **添加文件** | 重启容器后自动解析并向量化 |
| **修改文件** | 根据 MD5 检测变更，自动更新向量数据 |
| **删除文件** | 重启容器后自动清理对应向量记录 |

> ⚠️ **同步触发时机**：容器启动时自动执行（如 `docker-compose up`、`docker restart backend`）

## 六、常用操作

### 重启后端服务（触发知识库同步）
```bash
# 重启后端服务（触发知识库同步）
docker-compose restart backend
# 或使用容器名
docker restart agent-backend
```

### 查看日志
```bash
# 实时查看所有服务日志
docker-compose logs -f

# 仅查看后端日志
docker-compose logs -f backend

# 查看同步相关日志
docker-compose logs backend | grep -E "文件同步|加载知识库"
```

### 手动触发全量加载
```bash
# 删除 md5.text 缓存文件
docker exec agent-backend rm -f /app/md5.text

# 重启后端服务
docker-compose restart backend
# 或
docker restart agent-backend
```

### 清理向量数据
```python
# 进入容器执行
docker exec agent-backend python -c "
from pymilvus import MilvusClient
c = MilvusClient('http://milvus:19530')
c.drop_collection('agent')
print('Collection 已删除')
"
```

## 七、项目结构

```
Agent/
├── agent/              # Agent 核心逻辑
│   ├── react_agent.py  # ReAct Agent 实现
│   └── tools/          # MCP 工具集
├── api/                # FastAPI 路由
│   ├── app.py          # 主应用入口
│   └── routes/         # API 路由
├── config/             # YAML 配置文件
├── data/               # 知识库文件目录
├── frontend/           # Streamlit 前端
├── model/              # 模型工厂
│   └── factory.py      # 模型初始化
├── prompts/            # Prompt 模板
├── rag/                # RAG 服务
│   ├── rag_service.py  # RAG 检索与总结
│   └── vector_store.py # 向量存储与同步
├── utils/              # 工具类
├── Dockerfile.backend  # 后端镜像
├── Dockerfile.frontend # 前端镜像
├── docker-compose.yml  # 容器编排
└── requirements.txt    # Python 依赖
```

## 八、API 接口

### 健康检查
```bash
curl http://localhost:8000/health
```

### 聊天接口
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "小户型适合哪些扫地机器人？",
    "session_id": "test_user_001"
  }'
```

## 九、配置说明

### 环境变量优先级

配置加载顺序（从高到低）：
1. **环境变量**（`.env` 文件或系统环境变量）
2. **YAML 配置文件**（`config/*.yml`）
3. **代码默认值**

### agent.yml
```yaml
# 外部数据文件路径
external_data_path: data/external/records.csv

# 高德地图 API Key（建议通过环境变量 AMAP_API_KEY 配置）
# amap_api_key: your_api_key_here
```

### rag.yml（检索配置）
```yaml
# 模型提供商选择: ollama 或 aliyun
chat_model_provider: ollama
chat_model_name: qwen2.5:7b

embedding_model_provider: ollama
embedding_model_name: qwen3-embedding:latest

# Rerank 模型配置
rerank_model_path: model/BAAI-bge-reranker-v2-m3
rerank_device: cpu  # 有 GPU 可改为 cuda

# Ollama 服务地址（建议通过环境变量配置）
ollama_base_url: http://localhost:11434
```

### milvus.yml（向量库配置）
```yaml
collection_name: agent  # Collection 名称
uri: http://localhost:19530  # Milvus 地址
k: 3  # 检索返回数量
data_path: data  # 知识库文件夹
```

### .env 文件配置要点

**关键配置项：**
- `AMAP_API_KEY`: 高德地图 API Key
- `OLLAMA_BASE_URL`: Ollama 服务地址
- `MONGODB_URI`: MongoDB 连接地址
- `MILVUS_URI`: Milvus 连接地址
- `CHAT_MODEL_PROVIDER`: 聊天模型提供商 (ollama/aliyun)
- `EMBEDDING_MODEL_PROVIDER`: 嵌入模型提供商 (ollama/aliyun)

**不同操作系统的配置差异：**

| 配置项 | Windows/Mac (Docker) | Linux (Docker) | 本地开发 |
|--------|---------------------|----------------|----------|
| OLLAMA_BASE_URL | http://host.docker.internal:11434 | http://172.17.0.1:11434 | http://localhost:11434 |
| MONGODB_URI | mongodb://host.docker.internal:27017 | mongodb://172.17.0.1:27017 | mongodb://localhost:27017 |
| MILVUS_URI | http://host.docker.internal:19530 | http://172.17.0.1:19530 | http://localhost:19530 |

**Linux 系统特殊说明：**

Docker 在 Linux 上不支持 `host.docker.internal`，需要使用宿主机 IP：

```bash
# 查看宿主机 IP（通常是 Docker 网桥网关）
ip route | grep default | awk '{print $3}'
# 输出示例: 172.17.0.1

# 然后在 .env 中修改对应配置
MONGODB_URI=mongodb://172.17.0.1:27017
MILVUS_URI=http://172.17.0.1:19530
OLLAMA_BASE_URL=http://172.17.0.1:11434
```

## 十、故障排查

### 1. 知识库文件未加载
```bash
# 检查文件是否存在于容器内
docker exec agent-backend ls -l /app/data/

# 查看加载日志
docker-compose logs backend | grep "加载知识库"
# 或
docker logs agent-backend | grep "加载知识库"
```

### 2. 同步功能未生效
```bash
# 确认代码已更新到容器内
docker exec agent-backend grep "sync_documents" /app/rag/vector_store.py

# 重新构建镜像并重启
docker-compose build backend
docker-compose up -d backend
```

### 3. 数据库连接失败
```bash
# 检查 MongoDB 连接（根据实际配置调整 URI）
docker exec agent-backend python -c "
from pymongo import MongoClient
import os
uri = os.getenv('MONGODB_URI', 'mongodb://host.docker.internal:27017')
c = MongoClient(uri)
print(c.list_database_names())
"
```

## 十一、开发指南

### 本地运行（非 Docker）
```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export MONGODB_URI=mongodb://localhost:27017
export MILVUS_URI=http://localhost:19530

# 启动后端
uvicorn api.app:app --reload --port 8000

# 启动前端
streamlit run frontend/app.py
```

### 添加新工具（MCP）
1. 在 `agent/tools/mcp_server.py` 中定义工具
2. 在 `react_agent.py` 中注册工具
3. 重启服务

## 十二、完整部署指南

### 场景一：Docker Compose 部署（推荐）

**适用人群**：生产环境、快速部署

**步骤：**

1. **准备环境**
   ```bash
   # 确保已安装 Docker 和 Docker Compose
   docker --version
   docker-compose --version
   
   # 克隆项目
   git clone https://github.com/your-username/Agent.git
   cd Agent
   ```

2. **配置环境变量**
   ```bash
   cp .env.example .env
   # 编辑 .env，根据操作系统修改连接地址
   ```

3. **准备外部服务**
   - 安装并启动 Ollama，拉取模型：`ollama pull qwen2.5:7b`
   - 安装并启动 MongoDB
   - 安装并启动 Milvus

4. **启动服务**
   ```bash
   docker-compose build
   docker-compose up -d
   ```

5. **验证部署**
   ```bash
   # 检查服务状态
   docker-compose ps
   
   # 查看日志
   docker-compose logs -f backend
   
   # 访问健康检查
   curl http://localhost:8000/health
   ```

### 场景二：GitHub 下载 + 本地部署

**适用人群**：开发调试、学习研究

**步骤：**

1. **下载项目**
   ```bash
   # 方式1：Git 克隆
   git clone https://github.com/your-username/Agent.git
   cd Agent
   
   # 方式2：下载 ZIP 包
   # 访问 GitHub 页面 -> Code -> Download ZIP
   # 解压到指定目录
   ```

2. **安装 Python 依赖**
   ```bash
   # 建议使用虚拟环境
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   
   pip install -r requirements.txt
   ```

3. **安装并启动基础服务**
   
   **MongoDB:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install mongodb
   sudo systemctl start mongod
   
   # macOS
   brew install mongodb-community
   brew services start mongodb-community
   
   # Windows: 下载安装包安装
   ```
   
   **Milvus:**
   ```bash
   # 使用 Docker 快速启动
   docker run -d --name milvus-standalone \
     -p 19530:19530 \
     -p 9091:9091 \
     milvusdb/milvus:v2.4.0 \
     milvus run standalone
   ```
   
   **Ollama:**
   ```bash
   # 安装 Ollama (https://ollama.ai)
   # 拉取模型
   ollama pull qwen2.5:7b
   ollama pull qwen3-embedding:latest
   ```

4. **配置环境变量**
   ```bash
   cp .env.example .env
   # 编辑 .env，使用 localhost 连接本地服务
   ```

5. **启动应用**
   ```bash
   # 终端1：启动后端
   uvicorn api.app:app --reload --port 8000
   
   # 终端2：启动前端
   streamlit run frontend/app.py
   ```

6. **访问应用**
   - 前端：http://localhost:8501
   - 后端 API：http://localhost:8000
   - 健康检查：http://localhost:8000/health

### 场景三：完全 Docker 化部署（含数据库）

**适用人群**：无外部依赖、隔离环境

**步骤：**

1. **修改 docker-compose.yml**
   ```yaml
   # 取消注释 mongodb 和 milvus 服务
   # 修改 backend 的 depends_on 配置
   ```

2. **修改 .env**
   ```env
   MONGODB_URI=mongodb://mongodb:27017
   MILVUS_URI=http://milvus:19530
   OLLAMA_BASE_URL=http://host.docker.internal:11434  # 仍需外部 Ollama
   ```

3. **启动所有服务**
   ```bash
   docker-compose up -d
   ```

### 常见问题

**Q1: Docker 启动后无法连接数据库？**

A: 检查 `.env` 中的连接地址是否正确：
- Windows/Mac: 使用 `host.docker.internal`
- Linux: 使用宿主机 IP（如 `172.17.0.1`）

**Q2: 知识库文件未加载？**

A: 
```bash
# 检查文件是否在 data/ 目录
ls -l data/

# 查看加载日志
docker-compose logs backend | grep "加载知识库"

# 手动触发加载
docker exec agent-backend rm -f /app/md5.text
docker-compose restart backend
```

**Q3: Ollama 模型拉取失败？**

A:
```bash
# 检查 Ollama 服务是否运行
ollama list

# 重新拉取模型
ollama pull qwen2.5:7b
ollama pull qwen3-embedding:latest
```

**Q4: 端口被占用？**

A: 修改 `docker-compose.yml` 或 `.env` 中的端口配置：
```env
BACKEND_PORT=8001
FRONTEND_PORT=8502
```

## 📄 License

MIT License

---

**项目版本**: 1.0.0  
**最后更新**: 2026-04-13
