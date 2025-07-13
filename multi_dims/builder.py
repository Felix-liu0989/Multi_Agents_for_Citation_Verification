# flake8: noqa

from __future__ import annotations
from typing import List, Dict, Any, Optional
from .taxo import Node,DAG
from collections import deque
import json
from .model_definitions import promptLLM,constructPrompt,initializeLLM
from .prompts import multi_dim_prompt,NodeListSchema
from .utils import clean_json_string

__all__ = [
    "build_dags",
    "update_roots_with_labels",
    "expand_all_dags",
    "build_single_topic_dag",
]

def build_dags(args):
    """
    根据args中的topic和dimensions构建多颗DAG
    Parameters
    ----------
    args : Any
        需要至少包含以下字段::
            topic        : str   # 研究主题名称，如 "natural language processing"
            dimensions   : list  # 维度列表，例如 ["tasks", "datasets", ...]

    Returns
    -------
    roots : dict[str, Node]
        维度名 -> 根节点
    dags : dict[str, DAG]
        维度名 -> 对应 DAG 实例（只包含根节点，后续可增扩）
    id2node : dict[int, Node]
        全局 node.id -> Node  的映射
    label2node : dict[str, Node]
        全局 node.label -> Node 的映射
    """
    
    if not hasattr(args, "topic") or not hasattr(args, "dimensions"):
        raise ValueError("args must contain topic and dimensions")
    
    topic_prefix = args.topic.replace(' ', '_').lower()
    roots: Dict[str, Node] = {}
    dags: Dict[str, DAG] = {}
    id2node: Dict[int, Node] = {}
    label2node: Dict[str, Node] = {}
    
    cur_id = 0
    
    for dim in args.dimensions:
        mod_topic = topic_prefix + f"_{dim}"
        label = f"{topic_prefix}_{dim}"
        root = Node(
            id = cur_id,
            label = label,
            dimension = dim
        )
        roots[dim] = root
        dags[dim] = DAG(root, dim)
        id2node[cur_id] = root
        label2node[label] = root
        cur_id += 1
        
    return roots, dags, id2node, label2node


def build_topic_dags(args):
    """
    根据args中多个topic构建多颗DAG
    Parameters
    ----------
    args : Any
        需要至少包含以下字段::
            topic        : str   # 研究主题名称，如 "natural language processing"
            dimensions   : list  # 维度列表，例如 ["tasks", "datasets", ...]

    Returns
    -------
    roots : dict[str, Node]
        维度名 -> 根节点
    dags : dict[str, DAG]
        维度名 -> 对应 DAG 实例（只包含根节点，后续可增扩）
    id2node : dict[int, Node]
        全局 node.id -> Node  的映射
    label2node : dict[str, Node]
        全局 node.label -> Node 的映射
    """
    
    if not hasattr(args, "topic") or not hasattr(args, "dimensions"):
        raise ValueError("args must contain topic and dimensions")
    
    topic_prefix = args.topic.replace(' ', '_').lower()
    roots: Dict[str, Node] = {}
    dags: Dict[str, DAG] = {}
    id2node: Dict[int, Node] = {}
    label2node: Dict[str, Node] = {}
    
    cur_id = 0
    
    for dim in args.dimensions:
        # label = f"{topic_prefix}_{dim}"
        label = dim
        root = Node(
            id = cur_id,
            label = label,
            dimension = dim
        )
        roots[dim] = root
        dags[dim] = DAG(root, dim)
        id2node[cur_id] = root
        label2node[label] = root
        cur_id += 1
        
    return roots, dags, id2node, label2node

    

def update_roots_with_labels(
    roots: Dict[str, Node],
    outputs: List[Dict[str, bool]],
    internal_collection: Dict[int, Any],
    args,
):
    """把五维 LLM 分类结果写回各根节点的 ``papers`` 字典。

    Parameters
    ----------
    roots : dict[str, Node]
        build_dags 返回的根节点字典。
    outputs : list[dict]
        LLM 批量返回的布尔字典列表，长度 = 论文数。
        例如::
            {
                "tasks"             : True,
                "datasets"          : False,
                "methodologies"     : True,
                ...
            }
    internal_collection : dict[int, Paper]
        论文 id -> Paper 对象。
    args : Any
        需要包含 ``dimensions`` 字段。
    """
    
    if len(outputs) != len(internal_collection):
        raise ValueError("outputs and internal_collection must have the same length")
    
    for r in roots.values():
        r.papers = {}
    
    # type_dist = {dim:[] for dim in args.dimensions}
    
    for pid, out in enumerate(outputs):
        paper = internal_collection[pid]
        # 为后续细粒度分类预留标签容器
        paper.labels = {}
        for dim in args.dimensions:
            if out.get(dim):
                # 写回根节点
                roots[dim].papers[pid] = paper
                # 每个维度初始化空列表，后续子标签会 append
                paper.labels[dim] = [] 
    
            
def _expand_single_dag(
    dag: DAG,
    args,
    id2node:Dict[int,Node],
    label2node:Dict[str,Node],
):
    added = 0
    queue = deque([node for node in id2node.values()
            if node.dimension == dag.dimension])
    # deque([node for id, node in id2node.items()])
    while queue:
        curr_node = queue.popleft()
        # 构造prompt
        sys_ins,main_p,json_fmt = multi_dim_prompt(curr_node)
        prompts = [constructPrompt(args,sys_ins,main_p + "\n\n" + json_fmt)]
        # print(len(prompts))
        raw = promptLLM(
            args = args,
            prompts = prompts,
            schema = NodeListSchema,
            max_new_tokens = 3000,
            json_mode = False,
            temperature = 0.1,
            top_p = 0.99,
        )
        raw = clean_json_string(raw[0]) if "```" in raw[0] else raw[0].strip()
        outputs = json.loads(raw) if "```" in raw else json.loads(raw.strip())
        outputs = outputs['root_topic'] if 'root_topic' in outputs else next(iter(outputs.values())) if curr_node.label not in outputs else outputs[curr_node.label]
        
        for key,val in outputs.items():
            key = key.replace(' ', '_').lower()
            dim = curr_node.dimension
            
            # 判断是否需要新节点
            need_new_node = key not in label2node or \
                label2node[key].dimension != dim
            
            if need_new_node:
                child = Node(
                    id = len(id2node),
                    label = key,
                    dimension = dim,
                    description = val['description'],
                    parents = [curr_node],
                )
                curr_node.add_child(key,child)
                id2node[child.id] = child
                full_label = key + f"_{dim}"
                label2node[full_label] = child
                added += 1
            else:
                child = label2node[key]
                child.add_parent(curr_node)
                
            # 深度控制
            if child.level < args.init_levels:
                queue.append(child)
            
    return added

def expand_all_dags(
    dags: Dict[str, DAG],
    args,
    id2node: Dict[int, Node],
    label2node: Dict[str, Node],
):
    """
    对 `dags` 中所有维度执行扩展。
    返回 {dim: 新增节点数量}
    """
    stats = {}
    for dim, dag in dags.items():
        added = _expand_single_dag(dag, args, id2node, label2node)
        stats[dim] = added
    return stats

def build_single_topic_dag(args):
    """
    构建单主题DAG（最粗粒度）
    
    Parameters
    ----------
    args : Any
        需要至少包含以下字段::
            topic : str   # 研究主题名称，如 "natural language processing"

    Returns
    -------
    root : Node
        单个根节点
    dag : DAG
        单个DAG实例
    id2node : dict[int, Node]
        全局 node.id -> Node  的映射
    label2node : dict[str, Node]
        全局 node.label -> Node 的映射
    """
    
    if not hasattr(args, "topic"):
        raise ValueError("args must contain topic")
    
    topic_label = args.topic.replace(' ', '_').lower()
    
    # 创建单个根节点
    root = Node(
        id=0,
        label=topic_label,
        dimension="topic"  # 使用"topic"作为维度名
    )
    
    # 创建单个DAG
    dag = DAG(root, "topic")
    
    # 创建映射字典
    id2node = {0: root}
    label2node = {topic_label: root}
    
    return root, dag, id2node, label2node

if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.topic = "natural language processing"
            self.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods", "real_world_domains"]
            self.llm = 'gpt'
            self.init_levels = 2
            
    args = Args()
    root, dag, id2node, label2node = build_single_topic_dag(args)
    print(root)
    print(dag)
    print(id2node)
    print(label2node)
    # class Args:
    #     def __init__(self):
    #         self.topic = "natural language processing"
    #         self.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods", "real_world_domains"]
    #         self.llm = 'gpt'
    #         self.init_levels = 2

    #         self.dataset = "Reasoning"
    #         self.data_dir = f"datasets/multi_dim/{self.dataset.lower().replace(' ', '_')}/"
    #         self.internal = f"{self.dataset}.txt"
    #         self.external = f"{self.dataset}_external.txt"
    #         self.groundtruth = "groundtruth.txt"
            
    #         self.length = 512
    #         self.dim = 768

    #         self.iters = 4

    # args = Args()
    # args = initializeLLM(args)
    # roots, dags, id2node, label2node = build_dags(args)
    # # print(roots)
    
    # expand_stats = expand_all_dags(dags, args, id2node, label2node)
    # print(expand_stats)
    # print(expand_stats['tasks'])
    # print(expand_stats['datasets'])
    # print(expand_stats['methodologies'])
    # print(expand_stats['evaluation_methods'])
    # print(expand_stats['real_world_domains'])