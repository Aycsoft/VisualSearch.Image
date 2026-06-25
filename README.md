# VisualSearch.Image

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

基于 DINOv2 的轻量级视觉搜索服务，用于图片向量化提取与相似图像检索。

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
