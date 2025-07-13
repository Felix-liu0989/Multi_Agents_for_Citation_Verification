from .prompts import type_cls_system_instruction, topic_cls_main_prompt, TypeClsSchema, TopicsSchema, topic_cls_system_instruction, type_cls_main_prompt
from .model_definitions import promptLLM, constructPrompt
from typing import Dict, List
import json, re
from .utils import clean_json_string
from .classification import classify_prompt
from .taxo import DAG, Node
import json_repair
def label_papers_by_topic(
    args,
    paper_collection,
    topics,
    batch_size = 1
):
    paper_ids = list(paper_collection.keys())
    results = []
    for i in range(0, len(paper_ids), batch_size):
        batch_ids = paper_ids[i:i+batch_size]
        batch_papers = {pid:paper_collection[pid] for pid in batch_ids}
        prompts = [constructPrompt(args, topic_cls_system_instruction, topic_cls_main_prompt(paper,topics)) for paper in batch_papers.values()]
        outputs = promptLLM(
            args,
            prompts,
            schema=TopicsSchema,
            max_new_tokens=1500,
            json_mode=False,
            temperature=0.1,
            top_p=0.99,
        )
        outputs = [_safe_json_load(c) for c in outputs]
        
        for pid, out in zip(batch_ids, outputs):
            paper_collection[pid].labels = out
        results.extend(outputs)
        
    return results

def label_papers_by_type(
    args,
    paper_collection,
    batch_size = 1
):
    paper_ids = list(paper_collection.keys())
    
    results = []
    for i in range(0, len(paper_ids), batch_size):
        batch_ids = paper_ids[i:i+batch_size]
        batch_papers = {pid:paper_collection[pid] for pid in batch_ids}
        prompts = [constructPrompt(args, type_cls_system_instruction, type_cls_main_prompt(paper)) for paper in batch_papers.values()]
        outputs = promptLLM(
            args,
            prompts,
            schema=TypeClsSchema,
            max_new_tokens=1500,
            json_mode=False,
            temperature=0.1,
            top_p=0.99,
        )
        outputs = [_safe_json_load(c) for c in outputs]
        
        for pid, out in zip(batch_ids, outputs):
            paper_collection[pid].labels = out
        results.extend(outputs)
        
    return results


def assign_papers_to_dag(
    args,
    dag: DAG,
    label2node: Dict[str,Node],
    start_node: Node | None = None,
    max_depth: int = 3,
    batch_size: int = 1
):
    """
    递归地把论文分配到dag的子节点，直到max_depth
    """
    visited = set()
    queue = [(dag.root if start_node is None else start_node, 0)]
    
    while queue:
        node,depth = queue.pop(0)
        

# ---------------------------------------------------------------------------
# 内部工具：更健壮地解析 LLM 返回的伪 JSON
# ---------------------------------------------------------------------------

def _safe_json_load(raw: str):
    """尝试解析带有 Python 布尔值的伪 JSON。

    1. 去掉 ```json ...``` 包装 → clean_json_string
    2. 将 `True/False` 正则替换为合法 JSON 的 `true/false`
    3. 调用 json.loads 返回对象
    """
    s = clean_json_string(raw) if "```" in raw else raw.strip()
    # 替换大小写布尔值
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    return json_repair.loads(s)

