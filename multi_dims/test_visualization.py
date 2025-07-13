#!/usr/bin/env python3
"""
测试修复后的可视化功能
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from multi_dims.visualizer import DAGVisualizer
from multi_dims.taxo import Node, DAG
from multi_dims.example_visualization import create_interactive_mermaid_html


def create_test_dag():
    """创建一个测试DAG结构"""
    # 创建根节点
    root = Node(
        id=0,
        label="natural language processing tasks",
        dimension="tasks",
        description="Root node for NLP tasks"
    )
    
    # 创建子节点
    nlu_node = Node(
        id=1,
        label="natural language understanding (NLU) tasks",
        dimension="tasks", 
        description="Tasks related to understanding natural language"
    )
    
    nlg_node = Node(
        id=2,
        label="natural language generation (NLG) tasks",
        dimension="tasks",
        description="Tasks related to generating natural language"
    )
    
    # 创建孙子节点
    semantic_node = Node(
        id=3,
        label="semantic parsing tasks",
        dimension="tasks",
        description="Tasks for parsing semantic structures"
    )
    
    discourse_node = Node(
        id=4,
        label="discourse understanding tasks", 
        dimension="tasks",
        description="Tasks for understanding discourse"
    )
    
    data_to_text_node = Node(
        id=5,
        label="data-to-text generation",
        dimension="tasks",
        description="Generate text from structured data"
    )
    
    text_to_text_node = Node(
        id=6,
        label="text-to-text generation",
        dimension="tasks", 
        description="Generate text from input text"
    )
    
    # 建立层次关系
    root.add_child("nlu", nlu_node)
    root.add_child("nlg", nlg_node)
    
    nlu_node.add_child("semantic", semantic_node)
    nlu_node.add_child("discourse", discourse_node)
    
    nlg_node.add_child("data_to_text", data_to_text_node)
    nlg_node.add_child("text_to_text", text_to_text_node)
    
    # 模拟论文分配
    root.papers = {0: "paper0", 1: "paper1", 2: "paper2", 3: "paper3", 4: "paper4"}
    nlu_node.papers = {0: "paper0", 1: "paper1"}
    nlg_node.papers = {2: "paper2", 3: "paper3"}
    semantic_node.papers = {0: "paper0"}
    discourse_node.papers = {1: "paper1"}
    data_to_text_node.papers = {2: "paper2"}
    text_to_text_node.papers = {3: "paper3"}
    
    # 创建DAG
    dag = DAG(root, "tasks")
    
    return {"tasks": root}, {"tasks": dag}


def test_mermaid_generation():
    """测试Mermaid图表生成"""
    print("🧪 测试Mermaid图表生成...")
    
    roots, dags = create_test_dag()
    visualizer = DAGVisualizer(roots, dags)
    
    # 生成Mermaid代码
    mermaid_code = visualizer.generate_mermaid_diagram("tasks", max_depth=3)
    
    print("✅ 生成的Mermaid代码:")
    print("=" * 50)
    print(mermaid_code)
    print("=" * 50)
    
    # 保存到文件
    with open("test_tasks_dag.mmd", 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    
    # 创建HTML文件
    create_interactive_mermaid_html(mermaid_code, "test_tasks_dag.html")
    
    print("✅ 文件已生成:")
    print("  - test_tasks_dag.mmd")
    print("  - test_tasks_dag.html")
    
    return mermaid_code


def test_text_tree():
    """测试文本树生成"""
    print("\n🧪 测试文本树生成...")
    
    roots, dags = create_test_dag()
    visualizer = DAGVisualizer(roots, dags)
    
    # 生成文本树
    tree_text = visualizer.generate_text_tree("tasks", max_depth=3)
    
    print("✅ 生成的文本树:")
    print("=" * 50)
    print(tree_text)
    print("=" * 50)
    
    # 保存到文件
    with open("test_tasks_tree.txt", 'w', encoding='utf-8') as f:
        f.write(tree_text)
    
    print("✅ 文件已生成: test_tasks_tree.txt")
    
    return tree_text


def test_summary_stats():
    """测试统计信息生成"""
    print("\n🧪 测试统计信息生成...")
    
    roots, dags = create_test_dag()
    visualizer = DAGVisualizer(roots, dags)
    
    # 生成统计信息
    stats = visualizer.generate_summary_stats()
    
    print("✅ 生成的统计信息:")
    print("=" * 50)
    for dimension, stat in stats.items():
        print(f"{dimension.upper()} 维度:")
        print(f"  节点总数: {stat['total_nodes']}")
        print(f"  最大深度: {stat['max_depth']}")
        print(f"  论文总数: {stat['total_papers']}")
        print(f"  平均每节点论文数: {stat['avg_papers_per_node']:.2f}")
    print("=" * 50)
    
    return stats


def analyze_html_issues():
    """分析HTML问题"""
    print("\n🔍 分析之前的HTML问题...")
    
    issues = [
        "1. 节点名称包含特殊字符 (括号、空格) 导致解析错误",
        "2. 节点标识符使用了标签内容而不是唯一ID",
        "3. 没有正确应用定义的CSS样式类",
        "4. Mermaid版本过旧，可能不支持某些特性",
        "5. 缺少节点详细信息显示"
    ]
    
    fixes = [
        "✅ 使用 node_ID 作为节点标识符",
        "✅ 清理节点标签，移除特殊字符",
        "✅ 正确应用样式类到对应级别的节点",
        "✅ 升级到Mermaid 10.6.1版本",
        "✅ 添加节点层级和论文数信息"
    ]
    
    print("❌ 之前的问题:")
    for issue in issues:
        print(f"   {issue}")
    
    print("\n✅ 修复方案:")
    for fix in fixes:
        print(f"   {fix}")


if __name__ == "__main__":
    print("🚀 DAG可视化修复测试")
    print("=" * 60)
    
    # 分析问题
    analyze_html_issues()
    
    # 运行测试
    mermaid_code = test_mermaid_generation()
    tree_text = test_text_tree()
    stats = test_summary_stats()
    
    print("\n🎉 测试完成！")
    print("=" * 60)
    print("生成的测试文件:")
    print("  - test_tasks_dag.mmd (Mermaid源码)")
    print("  - test_tasks_dag.html (修复后的HTML)")
    print("  - test_tasks_tree.txt (文本树)")
    print("\n💡 建议:")
    print("  1. 打开 test_tasks_dag.html 查看可视化效果")
    print("  2. 对比之前的HTML文件，查看改进效果")
    print("  3. 检查节点标识符和样式是否正确应用") 