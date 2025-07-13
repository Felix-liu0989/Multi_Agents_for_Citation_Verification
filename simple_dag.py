import json
import os
import sys
from pathlib import Path
from multi_dims.model_definitions import initializeLLM, promptLLM, constructPrompt
from multi_dims.prompts import topic_cls_main_prompt,TopicsSchema
from multi_dims.taxo import DAG, Node
from multi_dims.paper import Paper_simple
from pydantic import BaseModel
from multi_dims.classifier import label_papers_by_topic
from datasets import load_dataset
from tqdm import tqdm
from itertools import islice
from multi_dims.builder import build_dags, build_single_topic_dag, update_roots_with_labels

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

def create_test_papers():
    """创建测试论文"""
    papers = [
        Paper_simple(
            paper_id=1,
            title="Attention Is All You Need",
            abstract="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms..."
        ),
        Paper_simple(
            paper_id=2,
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            abstract="We introduce BERT, a new language representation model which obtains state-of-the-art results on eleven natural language processing tasks..."
        ),
        Paper_simple(
            paper_id=3,
            title="ImageNet Classification with Deep Convolutional Neural Networks",
            abstract="We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet..."
        ),
        Paper_simple(
            paper_id=4,
            title="Generative Adversarial Networks",
            abstract="We propose a new framework for estimating generative models via an adversarial process..."
        ),
        Paper_simple(
            paper_id=5,
            title="ResNet: Deep Residual Learning for Image Recognition",
            abstract="Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks..."
        )
    ]
    return papers

parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))





ds = load_dataset("/home/liujian/project/2025-07/taxoadapt-main/datasets/EMNLP/EMNLP2024-papers",
            split="train")
    
internal_collection = {}


internal_count = 0
id = 0
for p in tqdm(islice(ds,10)):
    temp_dict = {"Title": p['title'], "Abstract": p['abstract']}
    formatted_dict = json.dumps(temp_dict)
    internal_collection[id] = Paper_simple(id, p['title'], p['abstract'], internal=True)
    internal_count += 1
    id += 1
print(f'Internal: {internal_count}')

if __name__ == "__main__":
    
    # # 测试关键词分类器
    # print("=== 测试关键词分类器 ===")
    # test_keyword_classifier()
    
    # # 测试LLM分类器
    # print("\n=== 测试LLM分类器 ===")
    def test_llm_classifier():
        """测试LLM分类器"""
        papers = create_test_papers()
        
        # 创建Args对象
        args = Args()
        
        args = initializeLLM(args)
        # 定义两个主题
        
        topic1 = "Natural Language Processing"
        topic2 = "Computer Vision"
        args.dimensions = [topic1, topic2]
        roots,dags,id2node,label2node = build_dags(args)
        results = label_papers_by_topic(
            args, 
            internal_collection,
            topic1,
            topic2
        )
        print(results)
        update_roots_with_labels(roots, results, internal_collection, args)
        grouped = {dim:[
        {
            "paper_id":pid,
            "title":paper.title,
            "abstract":paper.abstract
        }
        for pid,paper in roots[dim].papers.items()
    ] for dim in args.dimensions}
        print(grouped)
    test_llm_classifier()