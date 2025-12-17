# 本地 AI 智能文献与图像管理助手 (Local Multimodal AI Agent)
## 1. 项目简介 (Project Introduction)
本项目是一个基于 Python 的本地多模态 AI 智能助手，旨在解决本地大量文献和图像素材管理困难的问题。不同于传统的文件名搜索，本项目利用多模态神经网络技术，实现对内容的**语义搜索**和**自动分类**。本项目可以帮助各位同学理解多模态大模型的实际应用，并且可以在实际日常学习生活中帮助各位同学管理自己的本地知识库。希望各位同学可以不局限于本次作业规定的内容，通过自己的构建、扩展和维护实现自己的本地AI助手。

项目可使用本地化部署，也可以调用云端大模型 API 以获得更强的性能。

## ✨ 核心功能

### 📚 智能文献管理

#### 1. 语义搜索（文件级）
```bash
python main.py search_paper "Image Segmentation"
```
返回最相关的论文文件列表及相似度评分。

#### 2. 细粒度语义搜索（片段+页码）
```bash
python main.py semantic_search "扩散模型在图像修复任务中的数学逻辑"
```
返回具体论文片段和精确页码。支持 OCR 识别扫描版 PDF。

#### 3. 单文件分类
```bash
python main.py add_paper paper.pdf --topics "图像分割,目标检测,NLP"
```
自动分类论文并移动到对应主题文件夹。

#### 4. 批量整理（预定义主题）
```bash
python main.py batch_organize "C:\Downloads" --topics "图像分割,生成式异常检测,目标检测"
```
一键整理混乱文件夹，按预定义主题分类所有 PDF。

#### 5. 自动整理（智能主题发现）
```bash
python main.py auto_organize "C:\Downloads" --n_clusters 4
```
无需预定义主题，系统自动识别论文主题并分类。

### 🖼️ 智能图像管理

#### 6. 以文搜图
```bash
python main.py search_image "海边"
```
使用自然语言描述搜索本地图片库。

## 🛠️ 技术架构

### 核心技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 文本嵌入 | SentenceTransformers (all-MiniLM-L6-v2) | 384维向量，速度快 |
| 图像嵌入 | CLIP (ViT-B/32) | 图文联合表示学习 |
| 向量数据库 | ChromaDB | 嵌入式向量检索 |
| PDF处理 | pdfplumber + pytesseract | 文本提取 + OCR |
| 聚类 | scikit-learn K-Means | 自动主题发现 |

### 系统特点

- ✅ **本地化部署**：完全离线运行，数据隐私安全
- ✅ **模块化设计**：易于扩展和替换模型
- ✅ **多维度检索**：支持文件级和片段级搜索
- ✅ **智能 OCR**：自动处理扫描版 PDF
- ✅ **自动主题**：无需人工定义分类

## 📦 环境要求

- **操作系统**: Windows / macOS / Linux
- **Python**: 3.8+
- **内存**: 建议 8GB+
- **磁盘**: 至少 2GB（模型缓存）

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**核心依赖**：
```
sentence-transformers
chromadb
pdfplumber
torch
torchvision
ftfy
regex
Pillow
numpy
scikit-learn
pytesseract
pdf2image
```

### 2. 安装 OCR 引擎（可选，用于扫描版 PDF）

**Windows**:
1. 下载 Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
2. 安装到默认路径 `C:\Program Files\Tesseract-OCR`
3. 确保安装了中英文语言包

**macOS**:
```bash
brew install tesseract tesseract-lang
```

**Linux**:
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim
```

### 3. 准备数据

```
项目根目录/
├── data/
│   ├── papers/          # 论文存储目录（按主题分类）
│   └── images/          # 图片存储目录
```

## 📖 使用指南

### 命令列表

| 命令 | 功能 | 示例 |
|------|------|------|
| `add_paper` | 添加并分类单个论文 | `python main.py add_paper paper.pdf --topics "CV,NLP"` |
| `search_paper` | 语义搜索论文（文件级） | `python main.py search_paper "Image Segmentation"` |
| `semantic_search` | 细粒度搜索（片段+页码） | `python main.py semantic_search "扩散模型在图像修复任务中的数学逻辑"` |
| `batch_organize` | 批量整理（预定义主题） | `python main.py batch_organize ./pdfs --topics "CV,NLP"` |
| `auto_organize` | 自动整理（智能主题） | `python main.py auto_organize ./pdfs --n_clusters 3` |
| `search_image` | 以文搜图 | `python main.py search_image "火车"` |

## 📁 项目结构

```
多模态第二次作业/
├── main.py                          # 统一命令行入口
├── requirements.txt                 # Python 依赖
├── README.md                        # 项目文档
├── data/
│   ├── papers/                      # 论文存储（按主题分类）
│   └── images/                      # 图片存储
└── src/
    ├── document_manager/            # 文档管理模块
    │   ├── add_paper.py            # 单文件分类
    │   ├── search_paper.py         # 文件级语义搜索
    │   ├── semantic_search.py      # 片段级细粒度搜索
    │   ├── batch_organize.py       # 批量整理（预定义主题）
    │   └── auto_organize.py        # 自动整理（智能主题）
    └── image_manager/               # 图像管理模块
        └── search_image.py         # 以文搜图
```

## ❓ 常见问题

**Q: 为什么搜索结果只显示 URL？**  
A: PDF 是扫描版，需要安装 Tesseract OCR。系统会在文本提取失败时自动回退到 OCR。

**Q: 相似度分数都很低（< 0.3）？**  
A: 可能使用了 SimpleTextEncoder 简易编码器。请确保下载了 `all-MiniLM-L6-v2` 模型。

**Q: 自动主题命名不够准确？**  
A: 可以手动指定主题数量，或使用 `batch_organize` 手动定义主题名称。

---

## 📝 作业提交清单

本项目已完整实现所有作业要求：

✅ **项目简介与核心功能**：详见上方功能介绍  
✅ **环境配置与依赖**：详见"快速开始"章节  
✅ **详细使用说明**：每个命令都有完整示例  
✅ **技术选型说明**：详见"技术架构"章节  
✅ **统一入口**：`main.py` 支持所有功能  
✅ **命令行接口**：所有功能均可一键调用

### 📸 功能演示

#### 1. 语义搜索论文（文件级）

**命令**：
```bash
python main.py search_paper "Image Segmentation"
```

**输出示例**：
```
Indexed paper: 2404.17900v1.pdf
Indexed paper: gcnet.pdf
Indexed paper: rdrnet.pdf
Indexed paper: unet.pdf
Found 3 papers:
1. unet.pdf (Similarity: 0.6473)
   Path: data/papers\unet.pdf
2. gcnet.pdf (Similarity: 0.7070)
   Path: data/papers\gcnet.pdf
3. rdrnet.pdf (Similarity: 0.7103)
   Path: data/papers\rdrnet.pdf
...
```

#### 2. 细粒度搜索（片段+页码）

**命令**：
```bash
python main.py semantic_search "扩散模型在图像修复任务中的数学逻辑"
```

**输出示例**：
```
Most relevant passage:

File: 2404.17900v1.pdf
Page: 3
Similarity: 0.8234

Content (passage):
en,weestablish images, we discuss the design of ϵ ϕ(x t,y,t). According to amaskednoisyobservationmodeltodescribetherelationship Eqn.(3)andEqn.(8a),wecanaccuratelyestimateϵas: betweenyandx asthefollowing: √ √ 0 x − α x x − α y ϵm(x ,y,t)= t√ t 0 = √t t (10) y =(1−m)⊙x 0+m⊙(x 0+n) (7) ϕ t 1−α t 1−α t .
```

#### 3. 单文件分类

**命令**：
```bash
python main.py add_paper new_paper.pdf --topics "图像分割"
```

**输出示例**：
```
Classified paper as: 图像分割
Moved new_paper.pdf to data/papers/图像分割/new_paper.pdf
```

#### 4. 批量整理（预定义主题）

**命令**：
```bash
python main.py batch_organize "c:\Users\李腾\Desktop\多模态第二次作业\data\papers" --topics "Image Segmentation,Anomaly Detection"
```

**输出示例**：
```
Loading classification model...
Loading from: C:\Users\李腾\.cache\torch\sentence_transformers\sentence-transformers_all-MiniLM-L6-v2

Found 4 PDF files to organize
Topics: Image Segmentation, Anomaly Detection
------------------------------------------------------------

Processing: 2404.17900v1.pdf
  Classified as: Anomaly Detection (confidence: 0.4044)
  Moved to: data/papers\Anomaly Detection\2404.17900v1.pdf

Processing: gcnet.pdf
  Classified as: Image Segmentation (confidence: 0.2930)
  Moved to: data/papers\Image Segmentation\gcnet.pdf

Processing: rdrnet.pdf
  Classified as: Image Segmentation (confidence: 0.2897)
  Moved to: data/papers\Image Segmentation\rdrnet.pdf

Processing: unet.pdf
  Classified as: Image Segmentation (confidence: 0.3527)
  Moved to: data/papers\Image Segmentation\unet.pdf

============================================================
Batch organization complete!
  Successfully organized: 4 files
  Skipped: 0 files
============================================================

```

#### 5. 自动整理（智能主题发现）

**命令**：
```bash
python main.py auto_organize "F:\02论文"
```

**输出示例**：
```
Loading embedding model...

Found 15 PDF files
Extracting text and generating embeddings...
  Processing: 1-s2.0-S0952197625008486-main.pdf
  Processing: 2025.acl-demo.27.pdf
  Processing: 2404.17900v1.pdf
  Processing: 2502.14913v1.pdf
  Processing: 2508.14597v1.pdf
  Processing: 3e9fb7ae53c71b9f00dba7c03f46ca1353ea.pdf
  Processing: electronics-08-01131-v2.pdf
  Processing: forests-15-00689.pdf
  Processing: Research_on_Visual_Fault_Diagnosis_Technology_for_Power_Transformers_Based_on_Diffusion_Model_Sample_Augmentation.pdf
  Processing: s10489-024-05341-0.pdf
  Processing: s42408-022-00165-0.pdf
  Processing: sensors-23-05702.pdf
  Processing: transform.pdf
  Processing: 山火多模态检测.pdf
  Processing: 山火多模态检测1.pdf

Successfully processed 15 files
Auto-detected 2 clusters

Performing clustering...

Generating topic names...
  Cluster 0: smoke_detection (10 papers)
  Cluster 1: anomaly_diffusion (5 papers)

Organizing files...
  1-s2.0-S0952197625008486-main.pdf → smoke_detection/
  2025.acl-demo.27.pdf → anomaly_diffusion/
  2404.17900v1.pdf → anomaly_diffusion/
  2502.14913v1.pdf → anomaly_diffusion/
  2508.14597v1.pdf → smoke_detection/
  3e9fb7ae53c71b9f00dba7c03f46ca1353ea.pdf → smoke_detection/
  electronics-08-01131-v2.pdf → smoke_detection/
  forests-15-00689.pdf → smoke_detection/
  Research_on_Visual_Fault_Diagnosis_Technology_for_Power_Transformers_Based_on_Diffusion_Model_Sample_Augmentation.pdf → anomaly_diffusion/
  s10489-024-05341-0.pdf → anomaly_diffusion/
  s42408-022-00165-0.pdf → smoke_detection/
  sensors-23-05702.pdf → smoke_detection/
  transform.pdf → smoke_detection/
  山火多模态检测.pdf → smoke_detection/
  山火多模态检测1.pdf → smoke_detection/

============================================================
Auto-organization complete!
  Organized 15 files into 2 topics:
    - smoke_detection: 10 files
    - anomaly_diffusion: 5 files
============================================================
```

#### 6. 以文搜图

**命令**：
```bash
python main.py search_image "海边"
```

**输出示例**：
```
Found 1 images:
1. bj.jpg (Similarity: 0.7771)
   Path: data/images\bj.jpg   
...
```

### 📂 整理后的文件夹结构

```
data/papers/
├── 图像分割/
│   ├── gcnet.pdf
│   ├── rdrnet.pdf
│   └── unet_paper.pdf
├── 生成式异常检测/
│   ├── 2404.17900v1.pdf
│   ├── Research_on_Visual_Fault_Diagnosis.pdf
│   └── s10489-024-05341-0.pdf
├── segmentation_network/     # 自动发现的主题
│   └── ...
└── detection_transformer/     # 自动发现的主题
    └── ...
```

---

## 🎬 演示视频

- 将演示视频放在 `docs/demo.mp4`



- [▶️ 点此查看演示视频]
(
链接: https://pan.baidu.com/s/14Ry2uuyXJCZxnxqIDahMCw?pwd=jy21 提取码: jy21)

---


## 📄 许可证

本项目仅用于学习目的。


