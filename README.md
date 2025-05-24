# 基于LLM的文献筛选系统

## 项目简介

这是一个基于大语言模型（LLM）的学术论文分析和筛选系统，用于从大量学术文献中识别和分类与机器学习、深度学习和大语言模型相关的金融研究论文。

## 两阶段处理流程

### 第一阶段：技术方法筛选
通过预定义的关键词库和LLM智能分析，从海量学术论文中筛选出使用机器学习、深度学习或大语言模型技术的金融研究论文。该阶段主要识别论文所采用的技术方法，确保筛选出的论文确实运用了相关的AI/ML技术。

### 第二阶段：应用领域分类
对第一阶段筛选出的论文进行进一步的领域细分，将其分类到具体的金融应用场景中，包括衍生品定价、金融风险管理、投资组合管理。


## 项目架构

```
submit/
├── 📁 utils/                      # 核心工具模块
│   ├── data_loader.py             # 数据加载器
│   ├── cache_manager.py           # 第一阶段缓存管理
│   ├── llm_processor.py           # 第一阶段LLM处理器
│   ├── stage2_cache_manager.py    # 第二阶段缓存管理
│   └── stage2_llm_processor.py    # 第二阶段LLM处理器
├── 📁 prompts/                    # 提示词模板
│   ├── default.json               # 第一阶段默认提示词
│   └── stage2_prompts.json        # 第二阶段提示词
├── 📁 raw_data/                   # 原始数据目录
│   ├── CNKI/                      # 中国知网数据
│   └── WOS/                       # Web of Science数据
├── 📁 data/                       # 处理数据和缓存
├── 📁 cache/                      # 缓存文件
├── stage1_keywords.py             # 第一阶段：关键词筛选服务器
├── stage2_domain_filter.py        # 第二阶段：领域筛选服务器
├── final_result.py                # 第三阶段：结果统计查看服务器
└── keywords.py                    # 关键词定义
```


## 环境安装

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/AI-Optimization-Lab/FinRiskLLMReview
```

2. **安装依赖管理器**
```bash
pip install uv
```

3. **创建虚拟环境并安装依赖**
```bash
uv venv
uv sync
```


## 使用指南

### 1. 数据准备
在 `raw_data/` 目录下按以下结构放置数据文件：
```
raw_data/
├── CNKI/
│   ├── risk_ml.xls        # 金融风险-机器学习
│   ├── risk_dl.xls        # 金融风险-深度学习
│   ├── risk_llm.xls       # 金融风险-大语言模型
│   ├── pricing_ml.xls     # 衍生品定价-机器学习
│   ├── pricing_dl.xls     # 衍生品定价-深度学习
│   ├── portfolio_ml.xls   # 投资组合-机器学习
│   ├── portfolio_dl.xls   # 投资组合-深度学习
│   └── portfolio_llm.xls  # 投资组合-大语言模型
└── WOS/
    └── (类似结构的WOS数据文件)
```

### 2. 第一阶段：关键词筛选
```bash
uv run streamlit run stage1_keywords.py
```


### 3. 第二阶段：领域筛选
```bash
uv run streamlit run stage2_domain_filter.py
```


### 4. 第三阶段：结果统计
```bash
uv run streamlit run final_result.py
```



### API密钥配置
需要在界面中配置DeepSeek API密钥



## 数据格式说明

### 输入数据格式
支持Excel (.xls/.xlsx) 和CSV (.csv) 格式，要求包含以下列：
- **标题列**: "Title"、"题名"、"Article Title" 等
- **摘要列**: "Abstract"、"摘要"、"Summary" 等  
- **年份列**: "Year"、"年"、"Publication Year" 等

### 输出数据格式
处理结果包含原始数据加上以下扩展字段：
- `stage`: 处理阶段标识 (1或2)
- `matched_keywords`: 匹配的关键词列表
- `application_domains`: 应用领域分类结果
- `processing_timestamp`: 处理时间戳




