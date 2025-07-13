#!/usr/bin/env python3
"""
DAG可视化使用示例
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from multi_dims.visualizer import visualize_dags, DAGVisualizer
from multi_dims.pipeline import run_dag_to_classifier
from multi_dims.model_definitions import initializeLLM
from multi_dims.paper import Paper
from datasets import load_dataset
from tqdm import tqdm
from itertools import islice
import json


def demo_visualization():
    """演示如何使用DAG可视化工具"""
    
    # 1. 设置参数
    class Args:
        def __init__(self):
            self.topic = "natural language processing"
            self.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods", "real_world_domains"]
            self.llm = 'gpt'
            self.init_levels = 2
            self.max_density = 5   
            self.max_depth = 3
            self.dataset = "Reasoning"
            self.data_dir = f"datasets/multi_dim/{self.dataset.lower().replace(' ', '_')}/"
            self.internal = f"{self.dataset}.txt"
            self.external = f"{self.dataset}_external.txt"
            self.groundtruth = "groundtruth.txt"
            self.length = 512
            self.dim = 768
            self.iters = 4
    
    args = Args()
    args = initializeLLM(args)
    
    # 2. 创建测试数据集（这里使用小样本）
    os.makedirs(args.data_dir, exist_ok=True)
    
    # 模拟一些论文数据
    sample_papers = [
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": "We introduce BERT, a new method for pre-training language representations which obtains state-of-the-art results on a wide array of natural language processing tasks."
        },
        {
            "title": "GPT-3: Language Models are Few-Shot Learners", 
            "abstract": "We show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches."
        },
        {
            "title": "Attention Is All You Need",
            "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
        },
        {
            "title": "ImageNet Classification with Deep Convolutional Neural Networks",
            "abstract": "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes."
        },
        {
            "title": "Deep Residual Learning for Image Recognition",
            "abstract": "We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously."
        }
    ]
    
    # 3. 构建internal_collection
    internal_collection = {}
    
    with open(os.path.join(args.data_dir, 'internal.txt'), 'w') as f:
        for i, paper_data in enumerate(sample_papers):
            temp_dict = {"Title": paper_data['title'], "Abstract": paper_data['abstract']}
            formatted_dict = json.dumps(temp_dict)
            f.write(f'{formatted_dict}\n')
            internal_collection[i] = Paper(i, paper_data['title'], paper_data['abstract'], 
                                         label_opts=args.dimensions, internal=True)
    
    print(f"创建了 {len(internal_collection)} 篇论文的测试数据集")
    
    # 4. 运行DAG构建和分类
    print("正在构建DAG结构...")
    roots, dags = run_dag_to_classifier(args, internal_collection)
    
    # 5. 可视化DAG
    print("正在生成可视化...")
    visualizer = visualize_dags(roots, dags, output_dir="dag_visualizations",topic=args.topic)
    
    # # 6. 单独生成某个维度的Mermaid图表
    # print("\n生成单个维度的Mermaid图表:")
    # tasks_mermaid = visualizer.generate_mermaid_diagram("tasks", max_depth=2)
    # print("Tasks维度的Mermaid代码:")
    # print(tasks_mermaid)
    
    # 7. 生成文本树
    # print("\n生成文本树结构:")
    # tasks_tree = visualizer.generate_text_tree("tasks", max_depth=3)
    # print("Tasks维度的文本树:")
    # print(tasks_tree)
    
    return roots, dags, visualizer


def create_interactive_mermaid_html(mermaid_code, output_file="dag_visualization.html"):
    """
    创建交互式的Mermaid HTML文件
    
    Args:
        mermaid_code: Mermaid图表代码
        output_file: 输出HTML文件名
    """
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>DAG可视化</title>
    <script src="https://unpkg.com/mermaid@10.6.1/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .mermaid {{
            text-align: center;
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 20px;
            margin: 20px 0;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>DAG结构可视化</h1>
        <div class="mermaid">
{mermaid_code}
        </div>
    </div>
    
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                useMaxWidth: true,
                htmlLabels: true
            }}
        }});
    </script>
</body>
</html>"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"交互式HTML文件已保存到: {output_file}")


def visualize_specific_dimension(roots, dags, dimension="tasks"):
    """
    可视化特定维度的DAG
    
    Args:
        roots: 根节点字典
        dags: DAG字典
        dimension: 要可视化的维度
    """
    visualizer = DAGVisualizer(roots, dags)
    
    if dimension not in visualizer.dimensions:
        print(f"维度 {dimension} 不存在。可用维度: {visualizer.dimensions}")
        return
    
    # 生成Mermaid图表
    mermaid_code = visualizer.generate_mermaid_diagram(dimension, max_depth=3)
    
    # 保存Mermaid文件
    with open(f"{dimension}_dag.mmd", 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    
    # 创建HTML文件
    create_interactive_mermaid_html(mermaid_code, f"{dimension}_dag.html")
    
    # 生成NetworkX图表
    visualizer.plot_networkx_graph(dimension, f"{dimension}_networkx.png")
    
    # 打印文本树
    print(f"\n{dimension.upper()} 维度的文本树结构:")
    print(visualizer.generate_text_tree(dimension))
    
    print(f"\n{dimension} 维度的可视化文件已生成:")
    print(f"  - {dimension}_dag.mmd (Mermaid源码)")
    print(f"  - {dimension}_dag.html (交互式HTML)")
    print(f"  - {dimension}_networkx.png (NetworkX图表)")


if __name__ == "__main__":
    print("DAG可视化演示")
    print("=" * 50)
    
    # 运行演示
    roots, dags, visualizer = demo_visualization()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    # print("生成的文件:")
    # print("  - dag_visualizations/ 目录包含所有可视化文件")
    # print("  - 每个维度都有对应的 .mmd、.txt 和 .png 文件")
    
    # 演示单个维度可视化
    # print("\n正在为 'tasks' 维度生成额外的可视化...")
    # visualize_specific_dimension(roots, dags, "tasks") 