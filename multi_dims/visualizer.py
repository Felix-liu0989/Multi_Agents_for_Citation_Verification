import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from typing import Dict, List, Set
import os


class DAGVisualizer:
    """DAG可视化工具类"""
    
    def __init__(self, roots: Dict, dags: Dict):
        """
        初始化可视化工具
        
        Args:
            roots: 根节点字典 {dimension: root_node}
            dags: DAG字典 {dimension: dag_instance}
        """
        self.roots = roots
        self.dags = dags
        self.dimensions = list(roots.keys())
    
    def generate_mermaid_diagram(self, dimension: str, max_depth: int = 3) -> str:
        """
        生成指定维度的Mermaid图表代码
        
        Args:
            dimension: 要可视化的维度
            max_depth: 最大深度层级
            
        Returns:
            str: Mermaid图表代码
        """
        if dimension not in self.roots:
            raise ValueError(f"维度 {dimension} 不存在")
        
        root = self.roots[dimension]
        mermaid_code = ["graph TD"]
        
        # 收集所有节点和边
        nodes_info = []
        edges = []
        visited = set()
        node_levels = []  # 存储节点级别信息
        
        def traverse_node(node, parent_id=None):
            if node.id in visited:
                return
            visited.add(node.id)
            
            # 节点信息 - 使用节点ID作为标识符，避免特殊字符问题
            node_id = f"node_{node.id}"
            paper_count = len(node.papers) if hasattr(node, 'papers') else 0
            description = node.description[:30] + "..." if node.description and len(node.description) > 30 else (node.description or "")
            
            # 清理节点标签，移除特殊字符
            clean_label = node.label.replace('"', "'").replace('\n', ' ')
            node_info = f'{node_id}["{clean_label}<br/>Level: {node.level}<br/>Papers: {paper_count}"]'
            nodes_info.append(node_info)
            
            # 记录节点级别以便后续应用样式
            node_levels.append((node_id, node.level))
            
            # 添加边
            if parent_id is not None:
                parent_node_id = f"node_{parent_id}"
                edges.append(f"    {parent_node_id} --> {node_id}")
            
            # 遍历子节点
            if node.level < max_depth:
                for child_label, child_node in node.children.items():
                    traverse_node(child_node, node.id)
        
        # 从根节点开始遍历
        traverse_node(root)
        
        # 生成完整的Mermaid代码
        mermaid_code.extend(edges)
        
        # 添加样式定义
        mermaid_code.extend([
            "",
            "    classDef level0 fill:#ff9999,stroke:#333,stroke-width:3px",
            "    classDef level1 fill:#99ccff,stroke:#333,stroke-width:2px", 
            "    classDef level2 fill:#99ff99,stroke:#333,stroke-width:1px",
            "    classDef level3 fill:#ffff99,stroke:#333,stroke-width:1px"
        ])
        
        # 应用样式类到节点
        level_groups = {}
        for node_id, level in node_levels:
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node_id)
        
        for level, nodes in level_groups.items():
            if level <= 3:  # 只为前4层应用样式
                nodes_str = ",".join(nodes)
                mermaid_code.append(f"    class {nodes_str} level{level}")
        
        return "\n".join(mermaid_code)
    
    def save_mermaid_diagrams(self, output_dir: str = "visualizations"):
        """
        为所有维度生成并保存Mermaid图表
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for dimension in self.dimensions:
            mermaid_code = self.generate_mermaid_diagram(dimension)
            output_path = os.path.join(output_dir, f"{dimension}_dag.mmd")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(mermaid_code)
            
            print(f"已保存 {dimension} 维度的Mermaid图表: {output_path}")
    
    def create_networkx_graph(self, dimension: str) -> nx.DiGraph:
        """
        创建NetworkX图对象
        
        Args:
            dimension: 维度名称
            
        Returns:
            nx.DiGraph: NetworkX图对象
        """
        if dimension not in self.roots:
            raise ValueError(f"维度 {dimension} 不存在")
        
        G = nx.DiGraph()
        root = self.roots[dimension]
        visited = set()
        
        def add_node_to_graph(node):
            if node.id in visited:
                return
            visited.add(node.id)
            
            # 添加节点
            paper_count = len(node.papers) if hasattr(node, 'papers') else 0
            G.add_node(node.id, 
                      label=node.label, 
                      level=node.level,
                      papers=paper_count,
                      description=node.description or "")
            
            # 添加边并递归处理子节点
            for child_label, child_node in node.children.items():
                G.add_edge(node.id, child_node.id)
                add_node_to_graph(child_node)
        
        add_node_to_graph(root)
        return G
    
    def plot_networkx_graph(self, dimension: str, save_path: str = None, figsize=(12, 8)):
        """
        使用NetworkX和matplotlib绘制图表
        
        Args:
            dimension: 维度名称
            save_path: 保存路径
            figsize: 图表尺寸
        """
        G = self.create_networkx_graph(dimension)
        
        plt.figure(figsize=figsize)
        
        # 使用层次布局
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 根据level分层着色
        node_colors = []
        for node_id in G.nodes():
            level = G.nodes[node_id]['level']
            if level == 0:
                node_colors.append('#ff9999')
            elif level == 1:
                node_colors.append('#99ccff')
            elif level == 2:
                node_colors.append('#99ff99')
            else:
                node_colors.append('#ffff99')
        
        # 绘制图
        nx.draw(G, pos, 
                node_color=node_colors,
                node_size=1000,
                with_labels=False,
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                alpha=0.7)
        
        # 添加标签
        labels = {node_id: G.nodes[node_id]['label'][:10] + "..." if len(G.nodes[node_id]['label']) > 10 
                 else G.nodes[node_id]['label'] for node_id in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(f"{dimension.upper()} Dimension DAG", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()
    
    
    def generate_text_tree(self, dimension: str, max_depth: int = 3) -> str:
        """
        生成文本树状结构
        
        Args:
            dimension: 维度名称
            max_depth: 最大深度
            
        Returns:
            str: 文本树状结构
        """
        if dimension not in self.roots:
            raise ValueError(f"维度 {dimension} 不存在")
        
        root = self.roots[dimension]
        lines = []
        visited = set()
        
        def traverse_node(node, prefix="", is_last=True):
            if node.id in visited or node.level > max_depth:
                return
            visited.add(node.id)
            
            # 当前节点信息
            paper_count = len(node.papers) if hasattr(node, 'papers') else 0
            connector = "└── " if is_last else "├── "
            node_info = f"{prefix}{connector}{node.label} (Level: {node.level}, Papers: {paper_count})"
            lines.append(node_info)
            
            # 子节点前缀
            child_prefix = prefix + ("    " if is_last else "│   ")
            
            # 遍历子节点
            children = list(node.children.items())
            for i, (child_label, child_node) in enumerate(children):
                is_last_child = (i == len(children) - 1)
                traverse_node(child_node, child_prefix, is_last_child)
        
        traverse_node(root)
        return "\n".join(lines)
    
    def generate_text_tree_with_papers_only(self, dimension: str, max_depth: int = 3) -> str:
        """
        生成文本树状结构，只显示包含论文的节点
        
        Args:
            dimension: 维度名称
            max_depth: 最大深度
            
        Returns:
            str: 文本树状结构
        """
        if dimension not in self.roots:
            raise ValueError(f"维度 {dimension} 不存在")
        
        root = self.roots[dimension]
        lines = []
        visited = set()
        
        def traverse_node(node, prefix="", is_last=True):
            if node.id in visited or node.level > max_depth:
                return
            visited.add(node.id)
            
            # 检查当前节点是否包含论文
            paper_count = len(node.papers) if hasattr(node, 'papers') else 0
            
            # 如果当前节点没有论文，检查是否存在有论文的子节点
            has_papers_in_subtree = paper_count > 0
            if not has_papers_in_subtree and hasattr(node, 'children'):
                # 递归检查子树
                has_papers_in_subtree = _has_papers_in_subtree(node,visited,max_depth)
            
            # 如果当前节点和子树都没有论文，跳过
            if not has_papers_in_subtree:
                return
            
            # 有论文挂靠在节点下，则显示当前节点信息
            connector = "└── " if is_last else "├── "
            node_info = f"{prefix}{connector}{node.label} (Level: {node.level}, Papers: {paper_count})"
            lines.append(node_info)
            
            # 子节点前缀
            child_prefix = prefix + ("    " if is_last else "│   ")
            
            # 获取有论文的节点
            children_with_papers = []
            if hasattr(node, 'children'):
                for child_label, child_node in node.children.items():
                    child_paper_count = len(child_node.papers) if hasattr(child_node, 'papers') else 0
                    if child_paper_count > 0:
                        children_with_papers.append((child_label,child_node))
                        
            # 遍历有论文的子节点
            for i, (child_label, child_node) in enumerate(children_with_papers):
                is_last_child = (i == len(children_with_papers) - 1)
                traverse_node(child_node, child_prefix, is_last_child)
            
            
                

        def _has_papers_in_subtree(node,temp_visited,max_depth):
            '''
            检查子树中是否有包含论文的节点
            '''
            if node.id in temp_visited or node.level > max_depth:
                return False
            temp_visited.add(node.id)
            
            # 检查当前节点是否包含论文
            paper_count = len(node.papers) if hasattr(node, 'papers') else 0
            if paper_count > 0:
                return True
            
            # 递归检查子节点
            if hasattr(node, 'children'):
                for child_node in node.children.values():
                    if _has_papers_in_subtree(child_node,temp_visited,max_depth):
                        return True
            return False
        
        traverse_node(root)
        return "\n".join(lines)
        
    
    def save_text_trees(self, output_dir: str = "visualizations", topic: str = "topic"):
        """
        保存所有维度的文本树状结构
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(os.path.join(output_dir, topic), exist_ok=True)
        arxiv_trees = ""
        for dimension in self.dimensions:
            tree_text = self.generate_text_tree_with_papers_only(dimension)
            output_path = os.path.join(output_dir, f"{topic}/{dimension}_tree.txt")
            print(f"{dimension} dimension tree structure:")
            print(tree_text)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"{dimension.upper()} dimension tree structure:\n")
                f.write("=" * 50 + "\n\n")
                f.write(tree_text)
            arxiv_trees += f"{dimension.upper()} dimension tree structure:\n"
            arxiv_trees += "=" * 50 + "\n\n"
            arxiv_trees += tree_text
            
            print(f"Saved {dimension} dimension text tree: {output_path}")
        
        return arxiv_trees
    
    def generate_summary_stats(self) -> Dict:
        """
        生成DAG统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = {}
        
        for dimension in self.dimensions:
            root = self.roots[dimension]
            
            # 统计节点数量
            node_count = 0
            max_depth = 0
            total_papers = 0
            visited = set()
            
            def count_nodes(node):
                nonlocal node_count, max_depth, total_papers
                if node.id in visited:
                    return
                visited.add(node.id)
                
                node_count += 1
                max_depth = max(max_depth, node.level)
                if hasattr(node, 'papers'):
                    total_papers += len(node.papers)
                
                for child_node in node.children.values():
                    count_nodes(child_node)
            
            count_nodes(root)
            
            stats[dimension] = {
                "total_nodes": node_count,
                "max_depth": max_depth,
                "total_papers": total_papers,
                "avg_papers_per_node": total_papers / node_count if node_count > 0 else 0
            }
        
        return stats
    
    def print_summary(self):
        """打印DAG摘要信息"""
        stats = self.generate_summary_stats()
        
        print("\n" + "="*60)
        print("DAG 结构摘要")
        print("="*60)
        
        for dimension, stat in stats.items():
            print(f"\n{dimension.upper()} 维度:")
            print(f"  节点总数: {stat['total_nodes']}")
            print(f"  最大深度: {stat['max_depth']}")
            print(f"  论文总数: {stat['total_papers']}")
            print(f"  平均每节点论文数: {stat['avg_papers_per_node']:.2f}")
        
        print("\n" + "="*60)


# 使用示例函数
def visualize_dags(roots, dags, topic, output_dir="visualizations"):
    """
    一键可视化所有DAG
    
    Args:
        roots: 根节点字典
        dags: DAG字典
        output_dir: 输出目录
    """
    visualizer = DAGVisualizer(roots, dags)
    
    # 打印摘要
    visualizer.print_summary()
    
    # 保存所有可视化
    # visualizer.save_mermaid_diagrams(output_dir)
    arxiv_trees = visualizer.save_text_trees(output_dir,topic)
    
    # # 为每个维度生成NetworkX图表
    # for dimension in visualizer.dimensions:
    #     save_path = os.path.join(output_dir, f"{topic}_{dimension}_networkx.png")
    #     visualizer.plot_networkx_graph(dimension, save_path)
    
    print(f"\n所有可视化文件已保存到: {output_dir}")
    return visualizer,arxiv_trees





def visualize_tree_simple_filter(root, max_depth=10):
    """
    简化版：只显示直接包含论文的节点（不考虑子树）
    
    Args:
        root: 根节点
        max_depth: 最大显示深度
    
    Returns:
        str: 树形结构字符串
    """
    lines = []
    visited = set()
    
    def traverse_node(node, prefix="", is_last=True):
        if node.id in visited or node.level > max_depth:
            return
        visited.add(node.id)
        
        # 检查当前节点是否包含论文
        paper_count = len(node.papers) if hasattr(node, 'papers') else 0
        
        # 如果没有论文，直接跳过
        if paper_count == 0:
            return
        
        # 显示当前节点信息
        connector = "└── " if is_last else "├── "
        node_info = f"{prefix}{connector}{node.label} (Level: {node.level}, Papers: {paper_count})"
        lines.append(node_info)
        
        # 子节点前缀
        child_prefix = prefix + ("    " if is_last else "│   ")
        
        # 获取有论文的子节点
        children_with_papers = []
        if hasattr(node, 'children'):
            for child_label, child_node in node.children.items():
                child_paper_count = len(child_node.papers) if hasattr(child_node, 'papers') else 0
                if child_paper_count > 0:
                    children_with_papers.append((child_label, child_node))
        
        # 遍历有论文的子节点
        for i, (child_label, child_node) in enumerate(children_with_papers):
            is_last_child = (i == len(children_with_papers) - 1)
            traverse_node(child_node, child_prefix, is_last_child)
    
    traverse_node(root)
    return "\n".join(lines)


if __name__ == "__main__":
    # 示例用法
    print("DAG可视化工具已准备就绪！")
    print("使用方法:")
    print("from visualizer import visualize_dags")
    print("visualizer = visualize_dags(roots, dags)") 