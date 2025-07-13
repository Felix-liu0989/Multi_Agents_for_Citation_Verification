"""Multi-dim pipeline orchestrator.

当前版本仅实现两步：
1. 构建/扩展多维 DAG（builder.build_dags + builder.expand_all_dags）
2. 共识富化（enricher.enrich_all_dags）

后续可继续在此串入分类、评估等阶段。
"""

from __future__ import annotations
from .model_definitions import initializeLLM, promptLLM, constructPrompt
import os
import json
from typing import Any, Dict
from . import builder,enricher,classifier
from datasets import load_dataset
from tqdm import tqdm
from .paper import Paper
import os
from itertools import islice
from collections import deque
from contextlib import redirect_stdout
from .expansion import expandNodeWidth, expandNodeDepth
from pathlib import Path
__all__ = ["run"]


def _default_output_dir(args) -> str:
    """推导输出目录。"""
    base = getattr(args, "output_dir", None)
    if base is None:
        base = f"runs/{args.topic.replace(' ', '_').lower()}"
    os.makedirs(base, exist_ok=True)
    return base


def _dump_stats(path: str, stats: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def run(args):  # noqa: C901 – keep simple for now
    """执行 *建图 → 富化* 流水线。

    Parameters
    ----------
    args : Namespace or argparse.Namespace-like object
        需包含至少：
        topic、dimensions(list[str])、init_levels、llm（或已初始化）、output_dir(optional)

    Returns
    -------
    Dict[str, Any]
        运行报告，包括 *expand_stats*、*enrich_counts* 等。
    """

    # 1. 构建根节点与初始化 DAG
    roots, dags, id2node, label2node = builder.build_dags(args)

    # 2. 扩展（BFS 调 LLM）
    expand_stats = builder.expand_all_dags(dags, args, id2node, label2node)

    # 3. 富化
    output_dir = _default_output_dir(args)
    # mypy/pylint 类型协变问题：enricher 接受 Dict[int, object]，此处向上转型
    enriched_phrases, enriched_sentences = enricher.enrich_all_dags(
        args, 
        dags, 
        id2node, 
        save_dir=output_dir
    )

    # 4. 统计汇总
    enrich_counts = {dim: {
        "phrases": len(enriched_phrases[dim]),
        "sentences": len(enriched_sentences[dim])
    } for dim in dags}

    report = {
        "expand_stats": expand_stats,
        "enrich_counts": enrich_counts,
        "output_dir": output_dir,
    }

    _dump_stats(os.path.join(output_dir, "pipeline_report.json"), report)

    # 控制台简报
    print("=== DAG 扩展统计 ===")
    for dim, num in expand_stats.items():
        print(f"{dim}: +{num} nodes")
    print("=== 富化结果统计 ===")
    for dim, cnt in enrich_counts.items():
        print(f"{dim}: {cnt['phrases']} phrases, {cnt['sentences']} sentences")
    print(f"中间产物 & 报告已保存至: {output_dir}")

    return report


def run_for_related_work_sections(args):
    roots, dags, id2node, label2node = builder.build_dags(args)
    return roots,dags,id2node,label2node

def run_simple_dag(
    args,
    paper_collection
):
    root, dag, id2node, label2node = builder.build_single_topic_dag(args)
    
    # 将所有论文挂靠在根节点下
    for paper_id, paper in paper_collection.items():
        root.papers[paper_id] = paper
        # 设置论文的标签
        if hasattr(paper, 'labels'):
            paper.labels = {args.topic: [root.label]}
        else:
            paper.labels = {args.topic: [root.label]}
    
    print(f"Successfully attached {len(paper_collection)} papers to topic: {args.topic}")
    print(f"Root node '{root.label}' now contains {len(root.papers)} papers")
    
    return root, dag

def run_dag_to_classifier(
    args,
    paper_collection
):
    """
    建图 + 五维分类 + 做细粒度分类
    """
    # 创建根节点+空DAG
    roots, dags, id2node, label2node = builder.build_dags(args)
    
    # 顶层五维分类
    outputs = classifier.label_papers_by_type(
        args,
        paper_collection
    )
    # 把分类结果写回根节点
    builder.update_roots_with_labels(
        roots,
        outputs,
        paper_collection,
        args
    )
    grouped = {dim:[
        {
            "paper_id":pid,
            "title":paper.title,
            "abstract":paper.abstract
        }
        for pid,paper in roots[dim].papers.items()
    ] for dim in args.dimensions}
    # 扩展DAG
    expand_stats = builder.expand_all_dags(
        dags,
        args,
        id2node, 
        label2node
    )
    
    # 细粒度分类 递归地将论文下沉到子节点
    # for dag in dags.values():
    #     dag.classify_dag(
    #         args,
    #         label2node
    #     )
    
    visited = set()
    queue = deque([roots[r] for r in roots])

    while queue:
        curr_node = queue.popleft()
        print(f'VISITING {curr_node.label} ({curr_node.dimension}) AT LEVEL {curr_node.level}. WE HAVE {len(queue)} NODES LEFT IN THE QUEUE!')
        
        if len(curr_node.children) > 0:
            if curr_node.id in visited:
                continue
            visited.add(curr_node.id)

            # classify
            curr_node.classify_node(args, label2node, visited)

            # sibling expansion if needed
            new_sibs = expandNodeWidth(args, curr_node, id2node, label2node)
            print(f'(WIDTH EXPANSION) new children for {curr_node.label} ({curr_node.dimension}) are: {str((new_sibs))}')

            # re-classify and re-do process if necessary
            if len(new_sibs) > 0:
                curr_node.classify_node(args, label2node, visited)
            
            # add children to queue if constraints are met
            for child_label, child_node in curr_node.children.items():
                c_papers = label2node[child_label + f"_{curr_node.dimension}"].papers
                if (child_node.level < args.max_depth) and (len(c_papers) > args.max_density):
                    queue.append(child_node)
        else:
            # no children -> perform depth expansion
            
            # new_children, success = expandNodeDepth(args, curr_node, id2node, label2node)
            try:
                expand_result = expandNodeDepth(args, curr_node, id2node, label2node)
                if expand_result is None or len(expand_result) !=2:
                    new_children,success = [],False
                else:
                    new_children,success = expand_result
            except Exception as e:
                new_children,success = [],False
                print(f"Error in depth expansion for {curr_node.label} ({curr_node.dimension}): {e}")
                    
            args.llm = 'gpt'
            print(f'(DEPTH EXPANSION) new {len(new_children)} children for {curr_node.label} ({curr_node.dimension}) are: {str((new_children))}')
            if (len(new_children) > 0) and success:
                queue.append(curr_node)
            # classify
            # curr_node.classify_node(args, label2node, visited)
            # add children to queue if constraints are met
            # for child_label, child_node in curr_node.children.items():
            #     c_papers = label2node[child_label + f"_{curr_node.dimension}"].papers
            #     if (child_node.level < args.max_depth) and (len(c_papers) > args.min_density):
            #         queue.append(child_node)
            
    
    proj_root = Path(__file__).parent.parent.parent
    out_dir = str(proj_root / f"multi_dim_cls_results/{args.topic.replace(' ', '_')}")
    os.makedirs(out_dir, exist_ok=True)
    
    # enricher.enrich_all_dags(args, dags, id2node, save_dir=out_dir)
    
    for p in paper_collection.values():
      for dim in p.labels:
          p.labels[dim] = list(set(p.labels[dim]))
          
    json.dump(
        {pid: [{"labels":p.labels,"title":p.title,"abstract":p.abstract}] for pid, p in paper_collection.items()},
        open(os.path.join(out_dir, "paper_labels.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=4
    )
    json.dump(grouped,
          open(os.path.join(out_dir, "grouped_papers.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=4)

    # json.dump(expand_stats, open(os.path.join(out_dir, "expand_stats.json"), "w"), indent=2)
    for dim in args.dimensions:
        with open(f'{out_dir}/final_taxo_{dim}.txt', 'w') as f:
            with redirect_stdout(f):
                taxo_dict = roots[dim].display(0, indent_multiplier=2)

        with open(f'{out_dir}/final_taxo_{dim}.json', 'w', encoding='utf-8') as f:
            json.dump(taxo_dict, f, ensure_ascii=False, indent=4)
            
    paper_meta = {
       pid: {
           "title": p.title,
           "abstract": p.abstract
       }
       for pid, p in paper_collection.items()
    }
    with open(f"{out_dir}/paper_meta.json", "w", encoding="utf-8") as f:
        json.dump(paper_meta, f, ensure_ascii=False, indent=2)
        print("Pipeline 完成！已保存至", out_dir)
    return roots, dags
    




if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.topic = "natural language processing"
            self.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods", "real_world_domains"]
            self.llm = 'gpt'
            self.init_levels = 2

            self.dataset = "Reasoning"
            self.data_dir = f"datasets/multi_dim/{self.dataset.lower().replace(' ', '_')}/"
            self.internal = f"{self.dataset}.txt"
            self.external = f"{self.dataset}_external.txt"
            self.groundtruth = "groundtruth.txt"
            self.max_density = 5   
            self.max_depth   = 3
            self.length = 512
            self.dim = 768
            self.iters = 4
            
    args = Args()
    args = initializeLLM(args)
    
    ds = load_dataset("/home/liujian/project/2025-07/taxoadapt-main/datasets/EMNLP/EMNLP2024-papers",
            split="train")
    
    internal_collection = {}
    
    with open(os.path.join(args.data_dir, 'internal.txt'), 'w') as i:
        internal_count = 0
        id = 0
        for p in tqdm(islice(ds,10)):
            temp_dict = {"Title": p['title'], "Abstract": p['abstract']}
            formatted_dict = json.dumps(temp_dict)
            i.write(f'{formatted_dict}\n')
            internal_collection[id] = Paper(id, p['title'], p['abstract'], label_opts=args.dimensions, internal=True)
            internal_count += 1
            id += 1
    print(f'Internal: {internal_count}')
    
    roots,dags = run_dag_to_classifier(
        args,
        internal_collection
    )