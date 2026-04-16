# 部署检查清单

在部署智扫通智能客服系统之前，请确保完成以下检查项。

## 📋 Docker Compose 部署检查清单

### 1. 基础环境
- [ ] Docker 已安装（版本 >= 20.10）
  ```bash
  docker --version
  ```
  
- [ ] Docker Compose 已安装（版本 >= 2.0）
  ```bash
  docker-compose --version
  ```

### 2. 外部服务
- [ ] Ollama 已安装并运行
  ```bash
  ollama list  # 查看已安装的模型
  ollama pull qwen2.5:7b  # 拉取聊天模型
  ollama pull qwen3-embedding:latest  # 拉取嵌入模型
  ```

- [ ] MongoDB 已安装并运行（端口 27017）
  ```bash
  # 测试连接
  mongosh --eval "db.adminCommand('ping')"
  ```

- [ ] Milvus 已安装并运行（端口 19530）
  ```bash
  # 使用 Docker 快速启动
  docker run -d --name milvus-standalone \
    -p 19530:19530 -p 9091:9091 \
    milvusdb/milvus:v2.4.0 milvus run standalone
  ```

### 3. 配置文件
- [ ] 已复制 `.env.example` 为 `.env`
  ```bash
  cp .env.example .env  # Linux/Mac
  copy .env.example .env  # Windows
  ```

- [ ] 已配置高德地图 API Key（如需天气查询功能）
  - 访问 https://console.amap.com/dev/key/app 注册应用
  - 在 `.env` 中设置 `AMAP_API_KEY=your_key`

- [ ] 已根据操作系统修改连接地址
  
  **Windows/Mac:**
  ```env
  MONGODB_URI=mongodb://host.docker.internal:27017
  MILVUS_URI=http://host.docker.internal:19530
  OLLAMA_BASE_URL=http://host.docker.internal:11434
  ```
  
  **Linux:**
  ```bash
  # 先获取宿主机 IP
  ip route | grep default | awk '{print $3}'
  # 假设输出 172.17.0.1，则配置：
  ```
  ```env
  MONGODB_URI=mongodb://172.17.0.1:27017
  MILVUS_URI=http://172.17.0.1:19530
  OLLAMA_BASE_URL=http://172.17.0.1:11434
  ```

### 4. 知识库文件
- [ ] 已在 `data/` 目录放置知识库文件
  ```bash
  ls -l data/
  # 支持格式: .txt, .pdf, .docx
  ```

### 5. 启动服务
- [ ] 构建镜像
  ```bash
  docker-compose build
  ```

- [ ] 启动服务
  ```bash
  docker-compose up -d
  ```

- [ ] 检查服务状态
  ```bash
  docker-compose ps
  # 所有服务应显示 "Up" 状态
  ```

- [ ] 查看日志确认无错误
  ```bash
  docker-compose logs -f backend
  # 应看到 "知识库加载完成" 等日志
  ```

### 6. 验证部署
- [ ] 健康检查通过
  ```bash
  curl http://localhost:8000/health
  # 应返回 {"status": "healthy", ...}
  ```

- [ ] 前端可访问
  ```
  浏览器打开: http://localhost:8501
  ```

- [ ] 后端 API 可访问
  ```bash
  curl http://localhost:8000/docs  # Swagger UI
  ```

---

## 📋 本地开发部署检查清单

### 1. Python 环境
- [ ] Python 3.11+ 已安装
  ```bash
  python --version
  ```

- [ ] 创建虚拟环境（推荐）
  ```bash
  python -m venv venv
  source venv/bin/activate  # Linux/Mac
  venv\Scripts\activate     # Windows
  ```

- [ ] 安装依赖
  ```bash
  pip install -r requirements.txt
  ```

### 2. 数据库服务
- [ ] MongoDB 已启动
  ```bash
  # 检查服务状态
  systemctl status mongod  # Linux
  brew services list | grep mongodb  # macOS
  ```

- [ ] Milvus 已启动
  ```bash
  # 使用 Docker
  docker ps | grep milvus
  ```

### 3. 模型服务
- [ ] Ollama 已安装并运行
  ```bash
  ollama serve  # 启动服务（后台运行）
  ollama list   # 查看模型
  ```

- [ ] 所需模型已拉取
  ```bash
  ollama pull qwen2.5:7b
  ollama pull qwen3-embedding:latest
  ```

### 4. 配置文件
- [ ] 已配置 `.env` 文件
  ```env
  MONGODB_URI=mongodb://localhost:27017
  MILVUS_URI=http://localhost:19530
  OLLAMA_BASE_URL=http://localhost:11434
  ```

### 5. 启动应用
- [ ] 启动后端（终端1）
  ```bash
  uvicorn api.app:app --reload --port 8000
  ```

- [ ] 启动前端（终端2）
  ```bash
  streamlit run frontend/app.py
  ```

- [ ] 访问应用
  - 前端: http://localhost:8501
  - 后端: http://localhost:8000

---

## 🔧 故障排查

### 问题1：Docker 容器无法启动
```bash
# 查看详细日志
docker-compose logs backend

# 常见原因：
# 1. 端口被占用 -> 修改 docker-compose.yml 中的端口映射
# 2. 环境变量错误 -> 检查 .env 文件
# 3. 依赖服务未启动 -> 确认 MongoDB/Milvus/Ollama 运行中
```

### 问题2：知识库文件未加载
```bash
# 检查文件是否存在
docker exec agent-backend ls -l /app/data/

# 查看加载日志
docker-compose logs backend | grep "加载知识库"

# 手动触发重新加载
docker exec agent-backend rm -f /app/md5.text
docker-compose restart backend
```

### 问题3：数据库连接失败
```bash
# 测试 MongoDB 连接
docker exec agent-backend python -c "
from pymongo import MongoClient
import os
uri = os.getenv('MONGODB_URI')
print(f'Connecting to: {uri}')
c = MongoClient(uri)
print(c.list_database_names())
"

# 测试 Milvus 连接
docker exec agent-backend python -c "
from pymilvus import MilvusClient
import os
uri = os.getenv('MILVUS_URI')
print(f'Connecting to: {uri}')
c = MilvusClient(uri)
print(c.list_collections())
"
```

### 问题4：Ollama 连接失败
```bash
# 检查 Ollama 服务
curl http://localhost:11434/api/tags

# Docker 内测试
docker exec agent-backend curl http://host.docker.internal:11434/api/tags
# 或 Linux
docker exec agent-backend curl http://172.17.0.1:11434/api/tags
```

---

## ✅ 部署成功标志

- [x] `docker-compose ps` 显示所有服务状态为 "Up"
- [x] `curl http://localhost:8000/health` 返回 `{"status": "healthy"}`
- [x] 浏览器访问 http://localhost:8501 能看到聊天界面
- [x] 发送测试消息能收到回复
- [x] 日志中显示 "知识库加载完成"
- [x] 向量库中有文档数据（通过健康检查接口查看）

---

**最后更新**: 2026-04-16
