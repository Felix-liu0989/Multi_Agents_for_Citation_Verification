# 多维度论文分类系统 (Multi-Dimensional Paper Classification System)

一个基于大语言模型的多维度论文分类系统，能够自动构建分类体系、富化节点信息，并对学术论文进行细粒度分类。

## 🚀 功能特性

- **多维度分类**：支持从任务、数据集、方法论、评估方法、应用领域等多个维度对论文进行分类
- **自动建图**：基于 LLM 自动构建和扩展分类体系的 DAG（有向无环图）
- **节点富化**：为每个分类节点生成关键短语和描述句子，增强语义理解
- **细粒度分类**：将论文递归下沉到最合适的子节点，实现精准分类
- **模块化设计**：清晰的模块划分，便于扩展和维护

## 📁 项目结构

```
multi_dims/
├── __init__.py          # 包初始化
├── builder.py           # DAG 构建和扩展
├── enricher.py          # 节点富化（生成短语和句子）
├── classifier.py        # 论文分类（五维顶层分类）
├── pipeline.py          # 流水线编排
└── taxo.py             # 核心 Node 和 DAG 类定义
```

## 🏗️ 系统架构

### 完整流水线

```
数据加载 → 五维分类 → 建图 → 扩展DAG → 富化 → 细粒度分类 → 保存结果
```

### 核心模块

1. **建图模块 (`builder.py`)**
   - `build_dags()`: 根据主题和维度创建多个 DAG 根节点
   - `expand_all_dags()`: 使用 BFS 调用 LLM 扩展各维度的子节点
   - `update_roots_with_labels()`: 将五维分类结果写回根节点

2. **富化模块 (`enricher.py`)**
   - `enrich_all_dags()`: 为所有节点生成常识短语和句子
   - 支持保存到 JSON 文件便于后续使用

3. **分类模块 (`classifier.py`)**
   - `label_papers_by_type()`: 顶层五维布尔分类
   - 处理 LLM 返回格式的鲁棒性解析

4. **流水线模块 (`pipeline.py`)**
   - `run()`: 基础建图→富化流水线
   - `run_dag_to_classifier()`: 完整分类流水线

## 🛠️ 安装和配置

### 环境要求

```bash
conda activate -n taxo python==3.10
pip install -r requirements.txt
```

### 配置参数

```python
class Args:
    def __init__(self):
        self.topic = "natural language processing"  # 研究主题
        self.dimensions = [                         # 分类维度
            "tasks", 
            "datasets", 
            "methodologies", 
            "evaluation_methods", 
            "real_world_domains"
        ]
        self.llm = 'gpt'                           # 使用的LLM
        self.init_levels = 2                       # 初始层级
        self.max_density = 5                       # 最大节点密度
        self.max_depth = 3                         # 最大深度
        self.length = 512                          # 文本长度
        self.dim = 768                             # 向量维度
```

## 🚀 使用方法

### 基础使用

```python
from multi_dims.pipeline import run
from model_definitions import initializeLLM

# 初始化参数
args = Args()
args = initializeLLM(args)

# 运行基础流水线（建图 + 富化）
report = run(args)
```

### 完整分类流水线

```python
from multi_dims.pipeline import run_dag_to_classifier
from datasets import load_dataset

# 加载数据集
ds = load_dataset("your_dataset_path", split="train")

# 准备论文集合
paper_collection = {}
for i, paper in enumerate(ds):
    paper_collection[i] = Paper(
        i, 
        paper['title'], 
        paper['abstract'], 
        label_opts=args.dimensions, 
        internal=True
    )

# 运行完整分类流水线
roots, dags = run_dag_to_classifier(args, paper_collection)
```

## 📊 输出文件

系统会在 `runs/{topic_name}/` 目录下生成以下文件：

### 分类结果
- `paper_labels.json`: 每篇论文的细粒度标签
- `grouped_papers.json`: 按维度分组的论文
- `paper_meta.json`: 论文元信息（标题、摘要）

### 分类体系
- `final_taxo_*.json`: 各维度的分类体系树（JSON格式）
- `final_taxo_*.txt`: 各维度的分类体系树（可读格式）

### 富化信息
- `enriched_phrases.json`: 节点关键短语
- `enriched_sentences.json`: 节点描述句子

### 统计报告
- `pipeline_report.json`: 流水线运行统计

## 📝 输出格式示例

### 论文标签 (`paper_labels.json`)
```json
{
  "0": {
    "tasks": ["text_classification", "sentiment_analysis"],
    "datasets": ["imdb", "sst"],
    "methodologies": ["transformer", "bert"]
  }
}
```

### 分组论文 (`grouped_papers.json`)
```json
{
  "tasks": [
    {
      "paper_id": 0,
      "title": "BERT for Sentiment Analysis",
      "abstract": "This paper presents..."
    }
  ]
}
```

### 分类体系 (`final_taxo_tasks.json`)
```json
{
  "label": "tasks",
  "level": 0,
  "papers": {...},
  "children": {
    "text_classification": {
      "label": "text_classification",
      "level": 1,
      "children": {...}
    }
  }
}
```

## 🔧 核心算法

### 1. DAG 构建
- 使用 BFS 遍历扩展节点
- 支持宽度扩展（`expandNodeWidth`）和深度扩展（`expandNodeDepth`）
- 动态调整扩展策略

### 2. 论文分类
- 顶层五维布尔分类确定论文所属维度
- 递归下沉到子节点实现细粒度分类
- 使用队列管理分类过程

### 3. 节点富化
- 为每个节点生成 20 个关键短语
- 生成 10 个描述句子增强语义理解
- 支持批量处理和增量更新

## 🎯 应用场景

1. **文献综述生成**：基于分类结果生成结构化的相关工作
2. **知识图谱构建**：构建学术领域的知识图谱
3. **论文推荐系统**：基于多维度分类进行精准推荐
4. **研究趋势分析**：分析不同维度的研究热点和发展趋势

## 🐛 常见问题

### JSON 解析错误
- **问题**：LLM 返回 `True/False` 而非 `true/false`
- **解决**：使用 `_safe_json_load()` 进行格式转换

### API 调用格式错误
- **问题**：`messages` 参数格式不正确
- **解决**：使用 `constructPrompt()` 生成正确格式

### 标签重复问题
- **问题**：同一论文被多次分类导致标签重复
- **解决**：使用 `list(set(p.labels[dim]))` 去重


## TODO List
- 利用文献树生成大纲
- 根据大纲为每个部分迭代生成
- 实现几个部分的异步并行
- 评估指标跑通

- 大纲中包含两到三个要点
- 生成的大纲也是一个有向无环图
- 生成的内容例子如下：task-methodology-eval三元组
- 根据三元组生成summary，（以下可选）生成三次，打分
- 挂靠在某个要点下
- 迭代对比

