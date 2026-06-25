# VisualSearch.Image

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

基于 DINOv2 的轻量级视觉搜索服务，用于图片向量化提取与相似图像检索。
实现类似淘宝京东抖音图片识别，以图识图，图像搜索，图像匹对功能。

## 特性

- 🚀 **高性能向量检索**：基于 DINOv2 模型提取图片特征向量，支持海量图片快速检索
- 💾 **轻量存储**：SQLite 本地数据库，无需额外部署数据库服务
- 🔄 **增量索引**：自动检测文件变更（大小/修改时间），增量更新索引，节省计算资源
- 🗂️ **多图库支持**：可同时监控多个图片目录
- 🖥️ **内置管理台**：提供 Web 管理界面，支持索引状态查看、手动扫描、图片搜索测试
- 🔌 **易集成**：提供 RESTful API，支持 Docker 和 systemd 部署
- 🏷️ **分类管理**：支持按图片分类进行过滤检索

## 技术栈

- **Python 3.11 / 3.12**
- **FastAPI** - Web 框架
- **DINOv2** - 视觉特征提取模型
- **SQLite** - 向量索引存储
- **Uvicorn** - ASGI 服务器
- **Docker** - 容器化部署

## 快速开始

### 方式一：Windows 原生运行

```powershell
# 1. 克隆项目
git clone https://github.com/yourusername/VisualSearch.Image.git
cd VisualSearch.Image

# 2. 启动服务（首次运行自动创建虚拟环境并安装依赖）
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Start-VisualSearch.ps1

# 3. 访问管理台
# http://127.0.0.1:8100/admin
```

### 方式二：Docker 运行

```bash
# 1. 复制环境变量模板
cp .env.example .env

# 2. 修改 .env 中的图库路径
# VISUAL_SEARCH_UPLOAD_ROOT=/path/to/your/images

# 3. 启动服务
docker compose up -d --build

# 4. 查看日志
docker compose logs -f visual-search
```

### 方式三：Linux systemd 部署

```bash
# 1. 创建目录并拷贝代码
sudo mkdir -p /opt/visualsearch/image
sudo cp -r service /opt/visualsearch/image/

# 2. 创建虚拟环境
sudo python3 -m venv /opt/visualsearch/image/venv
sudo /opt/visualsearch/image/venv/bin/pip install -r /opt/visualsearch/image/service/requirements.txt

# 3. 配置环境变量
sudo cp deploy/visual-search.env.example /etc/visualsearch/image.env

# 4. 安装 systemd 服务
sudo cp deploy/visualsearch-image.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now visualsearch-image
```

## 配置说明

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `VISUAL_SEARCH_DEVICE` | 运行设备：cpu / cuda | `cpu` |
| `VISUAL_SEARCH_STORE_BACKEND` | 存储后端 | `sqlite` |
| `VISUAL_SEARCH_SQLITE_PATH` | SQLite 数据库路径 | `/data/visual-search.db` |
| `VISUAL_SEARCH_CATALOG_ROOT` | 单图库根目录 | - |
| `VISUAL_SEARCH_CATALOG_ROOTS` | 多图库根目录（分号分隔） | - |
| `VISUAL_SEARCH_API_KEY` | API 认证密钥（可选） | - |
| `VISUAL_SEARCH_MODEL_ID` | DINOv2 模型名称 | `facebook/dinov2-base` |
| `VISUAL_SEARCH_MODEL_VERSION` | 模型版本标识 | `dinov2-base-v1` |
| `HF_HOME` | 模型缓存目录 | `/models` |
| `HF_ENDPOINT` | HuggingFace 镜像地址 | `https://hf-mirror.com` |
| `VISUAL_SEARCH_PUBLIC_IMAGE_BASE_URL` | 图片访问 URL 前缀 | `/v1/catalog/image?path=` |

## API 接口

### 健康检查

```http
GET /health/live
GET /health/ready
```

### 图片检索

```http
POST /v1/search
Content-Type: application/json

{
    "image_base64": "data:image/jpeg;base64,...",
    "limit": 10,
    "category": "product"  // 可选，按分类过滤
}
```

### 索引管理

```http
# 扫描图库（增量更新）
POST /v1/catalog/scan
{
    "sync_delete": true  // 是否同步删除已移除的图片
}

# 查询扫描状态
GET /v1/catalog/scan/{jobId}

# 获取最新扫描任务
GET /v1/catalog/scan/latest

# 获取索引统计信息
GET /v1/catalog/stats
```

### 资产管理

```http
# 分页查询索引图片
GET /v1/catalog/assets?offset=0&limit=50

# 获取图片文件
GET /v1/catalog/image?path=relative/path/to/image.jpg

# 删除指定图片索引
DELETE /v1/index/products/{productId}
```

### 单张索引

```http
POST /v1/index/image
Content-Type: application/json

{
    "image_path": "relative/path/to/image.jpg",
    "product_id": "optional_id",
    "category": "product"
}
```

## 项目结构

```
VisualSearch.Image/
├── service/                    # FastAPI 核心服务
│   ├── app/
│   │   ├── api/               # API 路由
│   │   │   ├── v1/            # v1 版本接口
│   │   │   └── admin.py       # 管理台页面
│   │   ├── core/              # 配置与依赖
│   │   │   ├── config.py      # 环境变量配置
│   │   │   └── dependencies.py
│   │   ├── models/            # 数据模型（Pydantic/SQLAlchemy）
│   │   ├── services/          # 业务逻辑
│   │   │   ├── embedding.py   # DINOv2 特征提取
│   │   │   ├── indexer.py     # 索引构建
│   │   │   ├── searcher.py    # 相似检索
│   │   │   └── scanner.py     # 目录扫描
│   │   ├── storage/           # 存储层
│   │   │   └── sqlite.py      # SQLite 操作
│   │   ├── templates/         # 管理台 HTML 模板
│   │   └── main.py            # FastAPI 入口
│   ├── tools/                 # 工具脚本
│   │   ├── batch_import.py    # 批量导入
│   │   └── evaluate.py        # 检索评估
│   └── requirements.txt       # Python 依赖
├── scripts/                   # Windows 启停脚本
│   ├── Start-VisualSearch.ps1
│   └── Stop-VisualSearch.ps1
├── deploy/                    # Linux systemd 配置
│   ├── visualsearch-image.service
│   └── visual-search.env.example
├── data/                      # 默认数据目录
│   ├── sqlite/               # SQLite 数据库
│   └── models/               # 模型缓存
├── docker-compose.yml        # CPU 部署配置
├── docker-compose.gpu.yml    # GPU 部署覆盖配置
└── .env.example              # 环境变量模板
```

## 数据库

SQLite 文件位于 `data/sqlite/visual-search.db`，主要表结构：

### visual_assets（图片索引表）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INTEGER | 主键 |
| `product_id` | TEXT | 商品/图片唯一标识 |
| `product_code` | TEXT | 商品编码 |
| `file_id` | TEXT | 文件标识 |
| `file_path` | TEXT | 文件相对路径 |
| `file_size` | INTEGER | 文件大小（字节） |
| `file_mtime` | REAL | 最后修改时间戳 |
| `category` | TEXT | 分类标签 |
| `embedding` | BLOB | 特征向量（二进制） |
| `created_at` | REAL | 创建时间戳 |
| `updated_at` | REAL | 更新时间戳 |

### visual_scan_jobs（扫描任务表）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INTEGER | 主键 |
| `job_id` | TEXT | 任务唯一标识 |
| `status` | TEXT | 状态：pending/running/completed/failed |
| `total_files` | INTEGER | 总文件数 |
| `processed_files` | INTEGER | 已处理文件数 |
| `synced_delete` | INTEGER | 是否同步删除 |
| `started_at` | REAL | 开始时间 |
| `completed_at` | REAL | 完成时间 |
| `error_message` | TEXT | 错误信息 |

使用 [DB Browser for SQLite](https://sqlitebrowser.org/) 可便捷查看和管理。

### 常用查询

```sql
-- 查看索引总数
SELECT COUNT(*) FROM visual_assets;

-- 查看最近索引的图片
SELECT product_id, file_path, category, file_size, updated_at
FROM visual_assets
ORDER BY updated_at DESC
LIMIT 100;

-- 按分类统计
SELECT category, COUNT(*) AS count
FROM visual_assets
GROUP BY category
ORDER BY count DESC;
```

## 性能建议

- 🔹 初次索引时 CPU 占用较高，建议在低峰期执行
- 🔹 生产环境建议设置 `VISUAL_SEARCH_DEVICE=cpu`，单 worker 运行（避免重复加载模型）
- 🔹 SQLite 和数据目录建议挂载到持久化存储，避免随容器重启丢失
- 🔹 大量图片（>10万）建议定期备份数据库文件
- 🔹 管理台不要直接暴露公网，建议通过内网或 VPN 访问

## 从已有项目迁移

若从原 `CrossCart.VisualSearch` 迁移，只需：

1. 重命名项目目录
2. 更新服务名称和配置文件中的路径引用
3. SQLite 数据库可直接复用，无需迁移数据

## 开源协议

[MIT License](LICENSE)

## 贡献指南

欢迎提交 Issue 和 Pull Request。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交变更 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交 Pull Request

---

**⭐ 如果这个项目对你有帮助，请给一个 Star！**