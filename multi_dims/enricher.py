from .taxo import DAG, Node  # noqa: F401 – imported for type hints / consistency

import os
import json
from typing import Dict, Tuple, List

__all__ = [
    "enrich_all_dags",
]


def enrich_all_dags(
    args,
    dags: Dict[str, "DAG"],
    id2node: Dict[int, "Node"],
    save_dir: str | None = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """批量富化多维 DAG。

    该函数遍历 *dags* 字典中的每一个 :class:`~taxo.DAG`，
    调用已有的 ``DAG.enrich_dag`` 方法生成节点的 *20 个短语* 与 *10 句论文式样句子*，
    并按维度聚合返回。

    Parameters
    ----------
    args : Namespace
        命令行参数或 Notebook 中的 ``args``，需至少包含调用 LLM 的相关配置。
    dags : Dict[str, DAG]
        由 :pymeth:`multi_dims.builder.build_dags` 创建的、以维度名为键的 DAG 字典。
    id2node : Dict[int, Node]
        ``node_id → Node`` 的映射，用于在 ``DAG.enrich_dag`` 内部写回节点。
    save_dir : str | None, optional
        若指定，则把 ``enriched_phrases.json`` 和 ``enriched_sentences.json``
        持久化到该目录下。目录不存在时会自动创建。

    Returns
    -------
    Tuple[Dict[str, List[str]], Dict[str, List[str]]]
        ``(enriched_phrases, enriched_sentences)`` 两个字典，键为 *dimension*，
        值分别是聚合去重后的短语列表与句子列表。
    """
    enriched_phrases: Dict[str, List[str]] = {dim: [] for dim in dags}
    enriched_sentences: Dict[str, List[str]] = {dim: [] for dim in dags}

    for dim, dag in dags.items():
        # 调用 taxo.DAG 自带的 enrich_dag
        phrases, sentences = dag.enrich_dag(args, id2node)
        # 使用 set 进行简单去重后再写回
        enriched_phrases[dim].extend([p for p in phrases if p not in enriched_phrases[dim]])
        enriched_sentences[dim].extend([s for s in sentences if s not in enriched_sentences[dim]])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        _save_json(os.path.join(save_dir, "enriched_phrases.json"), enriched_phrases)
        _save_json(os.path.join(save_dir, "enriched_sentences.json"), enriched_sentences)

    return enriched_phrases, enriched_sentences


def _save_json(path: str, data: dict) -> None:
    """线程安全地写入 JSON 文件（UTF-8、缩进 4）。"""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, path)
