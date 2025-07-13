# Imports
import json
import math
import os
from typing import Callable, Optional
import datetime
from multi_dims.model_definitions import initializeLLM, promptLLM, constructPrompt
from multi_dims.pipeline import run_dag_to_classifier
from bertopic import BERTopic
from dotenv import load_dotenv
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from multi_dims.paper import Paper
from multi_dims.visualizer import visualize_dags
from multi_dims.builder import build_dags, update_roots_with_labels
from multi_dims.classifier import label_papers_by_topic
from citegeist.utils.citations import (
    filter_citations,
    get_arxiv_abstract,
    get_arxiv_citation,
    get_arxiv_title,
    process_arxiv_paper_with_embeddings,
)
from citegeist.utils.filtering import (
    select_diverse_pages_for_top_b_papers,
    select_diverse_papers_with_weighted_similarity,
)
from citegeist.utils.llm_clients import create_client
from citegeist.utils.llm_clients.deepseek_client import DeepSeekClient
from citegeist.utils.prompts import (
    generate_brief_topic_prompt,
    generate_question_answer_prompt,
    generate_related_work_prompt,
    generate_summary_prompt_question_with_page_content,
    generate_summary_prompt_with_page_content,
    generate_related_work_outline_prompt,
    generate_related_work_prompt_with_arxiv_trees,
    generate_related_work_revision_prompt,
    genrate_original_related_work_feedback_prompt,
    generate_related_work_outline_prompt_various_1    
)
import json_repair
from pathlib import Path

# Load environment variables
load_dotenv()



class Args:
    def __init__(self):
        self.topic = "topic"
        self.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
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
global args
args = Args()
args = initializeLLM(args)

args1 = Args()
args1 = initializeLLM(args1)

args2 = Args()
args2 = initializeLLM(args2)




class Generator:
    """Main generator class for Citegeist."""

    def __init__(
        self,
        llm_provider: str,
        database_uri: str,  # path to local milvus DB file or remote hosted Milvus DB
        database_token: Optional[str] = None,  # This only has to be set when authentication is required for the DB
        sentence_embedding_model_name: str = "/home/liujian/project/2025-07/sentence-transformers/all-mpnet-base-v2",
        topic_model_name: str = "/home/liujian/project/2025-07/BERTopic_ArXiv",
        **llm_kwargs,
    ):
        """
        Initialize the Generator with configuration.

        Args:
            llm_provider: LLM provider name ('azure', 'openai', 'anthropic').
                          Falls back to environment variable LLM_PROVIDER, then to 'azure'
            sentence_embedding_model_name: Name of the sentence transformer embedding model
            topic_model_name: Name of the BERTopic model
            database_uri: Path to the Milvus database
            database_token: Optional token for accessing Milvus database
            **llm_kwargs: Provider-specific configuration arguments for the LLM client
        """
        # Initialize core models
        self.topic_model = BERTopic.load(topic_model_name,embedding_model=sentence_embedding_model_name)
        self.sentence_embedding_model = SentenceTransformer(sentence_embedding_model_name)
        if database_token is None:
            self.db_client = MilvusClient(uri=database_uri)
        else:
            self.db_client = MilvusClient(uri=database_uri, token=database_token)

        # Set up LLM client
        self.llm_provider = llm_provider

        # Create LLM client (falls back to value of LLM_PROVIDER in env variables, and finally falls back to azure)
        self.llm_client = create_client(self.llm_provider, **llm_kwargs)

        # Store API version for Azure compatibility
        self.api_version = os.getenv("AZURE_API_VERSION", "2023-05-15")
    def test_args(self):
        args.topic = "natural language processing"
        return args.topic
    
    # def __del__(self):
    #     # Close out MilvusClient
    #     self.db_client.close()
    
    # def reset_system_state():
    #     """
    #     在处理新文献之前，重置所有动态状态
    #     """
    #     global current_dimensions,dime
    
    def get_arxiv_dim_test_distribution(
        self,
        abstract: str,
        breadth: int,
        depth: int,
        diversity: float,
        status_callback: Optional[Callable] = None,
    ):
        if status_callback:
            status_callback(1, "Initializing")
        print("摘要：", abstract)
        print("向量化摘要并在milvus库中检索...")
        # 向量化摘要并在milvus库中检索  
        embedded_abstract = self.sentence_embedding_model.encode(abstract)
        topic = self.topic_model.transform(abstract)
        topic_id = topic[0][0]
        
        # 检索milvus库中与摘要最相似的论文
        if status_callback:
            status_callback(2, "Querying Vector DB for matches (this may take a while)")
            
        query_data: list[list[dict]] = self.db_client.search(
            collection_name = "abstracts",
            data = [embedded_abstract],
            limit = 6 * breadth,
            anns_field = "embedding",
            search_params = {"metric_type": "COSINE", "params": {}},
            output_fields = ["embedding"],
        )
        
        if status_callback:
            status_callback(3, f"Retrieved {len(query_data[0])} papers from the DB")
            
        # 清理DB响应数据
        papers_data: list[dict] = query_data[0]
        for obj in papers_data:
            obj["embedding"] = obj["entity"]["embedding"]
            obj.pop("entity")
            
        # 选择一个长列表的论文
        selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
            paper_data = papers_data,
            k = 3 * breadth,
            diversity_weight = diversity)
        
        if status_callback:
            status_callback(4, f"Selected {len(selected_papers)} papers for the longlist, retrieving full text(s) (this might take a while)")
            
        # 生成每个论文的页面嵌入
        page_embeddings: list[list[dict]] = []
        for paper in selected_papers:
            arxiv_id = paper["id"]
            result = process_arxiv_paper_with_embeddings(arxiv_id, self.topic_model)
            if result:
                page_embeddings.append(result)
                
        if status_callback:
            status_callback(5, f"Generated page embeddings for {len(page_embeddings)} papers")
            
        # 生成一个短列表的论文（最多k页每篇，最多b篇）
        relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
            paper_embeddings=page_embeddings,
            input_string=abstract,
            topic_model=self.topic_model,
            k=depth,
            b=breadth,
            diversity_weight=diversity,
            skip_first=False,
        )

        if status_callback:
            status_callback(6, f"Selected {len(relevant_pages)} papers for the shortlist")
        
        internal_collection = {}
        data = []
        id = 0
        # Generate summaries for individual papers (taking all relevant pages into account)
        for obj in relevant_pages:
            # Because paper_id != arXiv_id -> retrieve arXiv id/
            arxiv_id = papers_data[obj["paper_id"]]["id"]
            arxiv_abstract = get_arxiv_abstract(arxiv_id)
            text_segments = obj["text"]
            
            title = get_arxiv_title(arxiv_id)
            
            # Create prompt
            prompt = generate_summary_prompt_with_page_content(
                abstract_source_paper=abstract,
                abstract_to_be_cited=arxiv_abstract,
                page_text_to_be_cited=text_segments,
                sentence_count=5,
            )
            
            internal_collection[id] = Paper(
                id, 
                title, 
                arxiv_abstract, 
                label_opts=["tasks", "datasets", "methodologies", "evaluation_methods"], 
                internal=True
            )
            temp_dict = {"Title": title, "Abstract": arxiv_abstract}
            data.append(temp_dict)            
            # Use the appropriate LLM client based on the provider
            response: str = self.llm_client.get_completion(prompt)
            obj["summary"] = response
            obj["citation"] = get_arxiv_citation(arxiv_id)
            internal_collection[id].summary = response
            internal_collection[id].citations = get_arxiv_citation(arxiv_id)
            id += 1
            
        with open("internal_collection.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        if status_callback:
            status_callback(7, "Generated summaries of papers (and their pages)")
        
        # 生成本篇文章的topic
        topic_prompt = generate_brief_topic_prompt(abstract)
        topic = self.llm_client.get_completion(topic_prompt)
        print("topic:")
        print(topic)
        
        # 生成文献树
        args.topic = topic
        roots,dags = run_dag_to_classifier(
            args,
            internal_collection
        )
        
        print("正在生成可视化...")
        
        proj_root = Path(__file__).parent.parent.parent
        dir = str(proj_root / "multi_dim_literature_visualizations")
        visualizer,arxiv_trees = visualize_dags(roots, dags, output_dir=dir,topic=args.topic)
        
        ## 为生成related work，先生成大纲
        print("正在生成related work大纲...")
        outline_prompt = generate_related_work_outline_prompt(abstract,arxiv_trees)
        outline = self.llm_client.get_completion(outline_prompt)
        outline = json_repair.loads(outline)
        print("outline:")
        print(outline)
        
        topic1 = outline["outline"][0]
        topic2 = outline["outline"][1]
        
        args1.topic = topic1
        
        args2.topic = topic2
        
        
        args.dimensions = [topic1, topic2]
        roots,dags,id2node,label2node = build_dags(args)
        results = label_papers_by_topic(
            args, 
            internal_collection,
            [topic1, topic2]
        )
        print(results)
        
        update_roots_with_labels(roots, results, internal_collection, args)
        grouped = {dim:[
        {
            "paper_id":pid,
            "title":paper.title,
            "abstract":paper.abstract,
            "summary":paper.summary,
            "citations":paper.citations
        }
        for pid,paper in roots[dim].papers.items()
    ] for dim in args.dimensions}
        
        dim_1 = args.dimensions[0]
        dim_2 = args.dimensions[1]
        grouped_dim_1 = grouped[dim_1]
        grouped_dim_2 = grouped[dim_2]
        
        # for dim in args.dimensions:
        #     for paper in grouped[dim]:
                
            
        # 生成related work
        prompt = generate_related_work_prompt_with_arxiv_trees(abstract,args.dimensions,grouped)
        related_work = self.llm_client.get_completion(prompt)
        print("related_work:")
        print(related_work)
        
        
            
        filtered_citations: list[str] = filter_citations(
            related_works_section=related_work, citation_strings=[obj["citation"] for obj in relevant_pages]
        )

        print("filtered_citations:")
        print(filtered_citations)
        os.makedirs(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}", exist_ok=True)
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/related_work_with_citations.txt", "w", encoding="utf-8") as f:
            f.write("="*50 + "\n" + "related_work" + "\n" + "="*50 + "\n" + related_work + "\n" + "="*50 + "\n" + "citations" + "\n" + "="*50 + "\n" + str(filtered_citations))
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/grouped_dim_1.json", "w", encoding="utf-8") as f:
            json.dump(grouped_dim_1,f,ensure_ascii=False,indent=4)
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/grouped_dim_2.json", "w", encoding="utf-8") as f:
            json.dump(grouped_dim_2,f,ensure_ascii=False,indent=4)
        if status_callback:
            status_callback(8, f"Generated related work section with {len(filtered_citations)} citations")
        
        return {"related_works": related_work, "citations": filtered_citations, "selected_papers": relevant_pages}
    
    
    def get_arxiv_dim_various_topics(
            self,
            abstract: str,
            breadth: int,
            depth: int,
            diversity: float,
            status_callback: Optional[Callable] = None,
    ):
        if status_callback:
            status_callback(1, "Initializing")
        print("摘要：", abstract)
        print("向量化摘要并在milvus库中检索...")
        # 向量化摘要并在milvus库中检索  
        embedded_abstract = self.sentence_embedding_model.encode(abstract)
        topic = self.topic_model.transform(abstract)
        topic_id = topic[0][0]
        
        # 检索milvus库中与摘要最相似的论文
        if status_callback:
            status_callback(2, "Querying Vector DB for matches (this may take a while)")
            
        query_data: list[list[dict]] = self.db_client.search(
            collection_name = "abstracts",
            data = [embedded_abstract],
            limit = 6 * breadth,
            anns_field = "embedding",
            search_params = {"metric_type": "COSINE", "params": {}},
            output_fields = ["embedding"],
        )
        
        if status_callback:
            status_callback(3, f"Retrieved {len(query_data[0])} papers from the DB")
            
        # 清理DB响应数据
        papers_data: list[dict] = query_data[0]
        for obj in papers_data:
            obj["embedding"] = obj["entity"]["embedding"]
            obj.pop("entity")
            
        # 选择一个长列表的论文
        selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
            paper_data = papers_data,
            k = 3 * breadth,
            diversity_weight = diversity)
        
        if status_callback:
            status_callback(4, f"Selected {len(selected_papers)} papers for the longlist, retrieving full text(s) (this might take a while)")
            
        # 生成每个论文的页面嵌入
        page_embeddings: list[list[dict]] = []
        for paper in selected_papers:
            arxiv_id = paper["id"]
            result = process_arxiv_paper_with_embeddings(arxiv_id, self.topic_model)
            if result:
                page_embeddings.append(result)
                
        if status_callback:
            status_callback(5, f"Generated page embeddings for {len(page_embeddings)} papers")
            
        # 生成一个短列表的论文（最多k页每篇，最多b篇）
        relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
            paper_embeddings=page_embeddings,
            input_string=abstract,
            topic_model=self.topic_model,
            k=depth,
            b=breadth,
            diversity_weight=diversity,
            skip_first=False,
        )

        if status_callback:
            status_callback(6, f"Selected {len(relevant_pages)} papers for the shortlist")
        
        internal_collection = {}
        data = []
        id = 0
        # Generate summaries for individual papers (taking all relevant pages into account)
        for obj in relevant_pages:
            # Because paper_id != arXiv_id -> retrieve arXiv id/
            arxiv_id = papers_data[obj["paper_id"]]["id"]
            arxiv_abstract = get_arxiv_abstract(arxiv_id)
            text_segments = obj["text"]
            obj["cite_ids"] = [id]
            title = get_arxiv_title(arxiv_id)
            
            # Create prompt
            prompt = generate_summary_prompt_with_page_content(
                abstract_source_paper=abstract,
                abstract_to_be_cited=arxiv_abstract,
                page_text_to_be_cited=text_segments,
                sentence_count=5,
            )
            
            internal_collection[id] = Paper(
                id, 
                title, 
                arxiv_abstract, 
                label_opts=["tasks", "datasets", "methodologies", "evaluation_methods"], 
                internal=True
            )
            temp_dict = {"Title": title, "Abstract": arxiv_abstract}
            data.append(temp_dict)            
            # Use the appropriate LLM client based on the provider
            response: str = self.llm_client.get_completion(prompt)
            obj["summary"] = response
            obj["citation"] = get_arxiv_citation(arxiv_id)
            internal_collection[id].summary = response
            internal_collection[id].citations = get_arxiv_citation(arxiv_id)
            id += 1
            
        with open("internal_collection.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        if status_callback:
            status_callback(7, "Generated summaries of papers (and their pages)")
        
        # 生成本篇文章的topic
        topic_prompt = generate_brief_topic_prompt(abstract)
        topic = self.llm_client.get_completion(topic_prompt)
        print("topic:")
        print(topic)
        
        # 生成文献树
        args.topic = topic
        roots,dags = run_dag_to_classifier(
            args,
            internal_collection
        )
        
        print("正在生成可视化...")
        
        proj_root = Path(__file__).parent.parent.parent
        dir = str(proj_root / "multi_dim_literature_visualizations")
        visualizer,arxiv_trees = visualize_dags(roots, dags, output_dir=dir,topic=args.topic)
        
        ## 为生成related work，先生成大纲
        print("正在生成related work大纲...")
        outline_prompt = generate_related_work_outline_prompt_various_1(abstract,arxiv_trees)
        outline = self.llm_client.get_completion(outline_prompt)
        outline = json_repair.loads(outline)
        
        # outline_prompt = generate_related_work_outline_prompt(abstract,arxiv_trees)
        # outline = self.llm_client.get_completion(outline_prompt)
        # outline = json_repair.loads(outline)
        print("outline:")
        print(outline)
        
        subsection_titles = outline["outline"]
        # topic1 = outline["outline"][0]
        # topic2 = outline["outline"][1]
        
        # refine_outline_prompt = generate_related_work_outline_prompt_various(abstract,arxiv_trees,topic1,topic2)
        # refine_outline = self.llm_client.get_completion(refine_outline_prompt)
        # refine_outline = json_repair.loads(refine_outline)
        
        # if refine_outline == True:
        #     subsection_titles = [topic1,topic2]
        # else:
        #     subsection_titles = refine_outline["optimized_outline"]
            
        
        args.dimensions = subsection_titles
        roots,dags,id2node,label2node = build_dags(args)
        results = label_papers_by_topic(
            args, 
            internal_collection,
            subsection_titles
        )
        print(results)
        
        update_roots_with_labels(roots, results, internal_collection, args)
        grouped = {dim:[
        {
            "paper_id":pid,
            "title":paper.title,
            "abstract":paper.abstract,
            "summary":paper.summary,
            "citations":paper.citations
        }
        for pid,paper in roots[dim].papers.items()
    ] for dim in args.dimensions}
        
        dim_1 = args.dimensions[0]
        dim_2 = args.dimensions[1]
        grouped_dim_1 = grouped[dim_1]
        grouped_dim_2 = grouped[dim_2]
        
        
        
        # 生成related work
        prompt = generate_related_work_prompt_with_arxiv_trees(abstract,args.dimensions,grouped)
        related_work = self.llm_client.get_completion(prompt)
        print("related_work:")
        print(related_work)
        
        client = DeepSeekClient(
            api_key = os.environ.get("DEEPSEEK_API_KEY", ""),
            model_name = "deepseek-chat"
        )
        
        feedback_prompt = genrate_original_related_work_feedback_prompt(related_work)
        feedback = self.llm_client.get_completion(feedback_prompt)
        print("feedback:")
        print(feedback)
        
        prompt_for_revision = generate_related_work_revision_prompt(abstract,related_work,feedback,args.dimensions)
        related_work_revision = client.get_completion(prompt_for_revision)
        print("related_work_revision:")
        print(related_work_revision)
        
            
        filtered_citations: list[str] = filter_citations(
            related_works_section=related_work, citation_strings=[obj["citation"] for obj in relevant_pages]
        )
        
        
        print("filtered_citations:")
        print(filtered_citations)
        date = datetime.date.today()
        os.makedirs(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}", exist_ok=True)
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/related_work_with_citations.txt", "w", encoding="utf-8") as f:
            f.write("="*50 + "\n" + "related_work" + "\n" + "="*50 + "\n" + related_work + "\n" + "="*50 + "\n" + "citations" + "\n" + "="*50 + "\n" + str(filtered_citations))
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/grouped_dim_1.json", "w", encoding="utf-8") as f:
            json.dump(grouped_dim_1,f,ensure_ascii=False,indent=4)
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/grouped_dim_2.json", "w", encoding="utf-8") as f:
            json.dump(grouped_dim_2,f,ensure_ascii=False,indent=4)
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/related_work_revision.txt", "w", encoding="utf-8") as f:
            f.write("="*50 + "\n" + "related_work_revision" + "\n" + "="*50 + "\n" + related_work_revision + "\n" + "="*50 + "\n" + "citations" + "\n" + "="*50 + "\n" + str(filtered_citations))
        if status_callback:
            status_callback(8, f"Generated related work section with {len(filtered_citations)} citations")
        args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
        final= {"related_works": related_work_revision, "citations": filtered_citations, "selected_papers": relevant_pages}
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/final.json", "w", encoding="utf-8") as f:
            json.dump(final,f,ensure_ascii=False,indent=4)
            
        args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
        return final
    
    def get_arxiv_dim_test(
        self,
        abstract: str,
        breadth: int,
        depth: int,
        diversity: float,
        status_callback: Optional[Callable] = None,
    ):
        if status_callback:
            status_callback(1, "Initializing")
        print("摘要：", abstract)
        print("向量化摘要并在milvus库中检索...")
        # 向量化摘要并在milvus库中检索  
        embedded_abstract = self.sentence_embedding_model.encode(abstract)
        topic = self.topic_model.transform(abstract)
        topic_id = topic[0][0]
        
        # 检索milvus库中与摘要最相似的论文
        if status_callback:
            status_callback(2, "Querying Vector DB for matches (this may take a while)")
            
        query_data: list[list[dict]] = self.db_client.search(
            collection_name = "abstracts",
            data = [embedded_abstract],
            limit = 6 * breadth,
            anns_field = "embedding",
            search_params = {"metric_type": "COSINE", "params": {}},
            output_fields = ["embedding"],
        )
        
        if status_callback:
            status_callback(3, f"Retrieved {len(query_data[0])} papers from the DB")
            
        # 清理DB响应数据
        papers_data: list[dict] = query_data[0]
        for obj in papers_data:
            obj["embedding"] = obj["entity"]["embedding"]
            obj.pop("entity")
            
        # 选择一个长列表的论文
        selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
            paper_data = papers_data,
            k = 3 * breadth,
            diversity_weight = diversity)
        
        if status_callback:
            status_callback(4, f"Selected {len(selected_papers)} papers for the longlist, retrieving full text(s) (this might take a while)")
            
        # 生成每个论文的页面嵌入
        page_embeddings: list[list[dict]] = []
        for paper in selected_papers:
            arxiv_id = paper["id"]
            result = process_arxiv_paper_with_embeddings(arxiv_id, self.topic_model)
            if result:
                page_embeddings.append(result)
                
        if status_callback:
            status_callback(5, f"Generated page embeddings for {len(page_embeddings)} papers")
            
        # 生成一个短列表的论文（最多k页每篇，最多b篇）
        relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
            paper_embeddings=page_embeddings,
            input_string=abstract,
            topic_model=self.topic_model,
            k=depth,
            b=breadth,
            diversity_weight=diversity,
            skip_first=False,
        )

        if status_callback:
            status_callback(6, f"Selected {len(relevant_pages)} papers for the shortlist")
        
        internal_collection = {}
        data = []
        id = 0
        # Generate summaries for individual papers (taking all relevant pages into account)
        for obj in relevant_pages:
            # Because paper_id != arXiv_id -> retrieve arXiv id/
            arxiv_id = papers_data[obj["paper_id"]]["id"]
            arxiv_abstract = get_arxiv_abstract(arxiv_id)
            text_segments = obj["text"]
            obj["cite_ids"] = [id]
            title = get_arxiv_title(arxiv_id)
            
            # Create prompt
            prompt = generate_summary_prompt_with_page_content(
                abstract_source_paper=abstract,
                abstract_to_be_cited=arxiv_abstract,
                page_text_to_be_cited=text_segments,
                sentence_count=5,
            )
            
            internal_collection[id] = Paper(
                id, 
                title, 
                arxiv_abstract, 
                label_opts=["tasks", "datasets", "methodologies", "evaluation_methods"], 
                internal=True
            )
            temp_dict = {"Title": title, "Abstract": arxiv_abstract}
            data.append(temp_dict)            
            # Use the appropriate LLM client based on the provider
            response: str = self.llm_client.get_completion(prompt)
            obj["summary"] = response
            obj["citation"] = get_arxiv_citation(arxiv_id)
            internal_collection[id].summary = response
            internal_collection[id].citations = get_arxiv_citation(arxiv_id)
            id += 1
            
        with open("internal_collection.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        if status_callback:
            status_callback(7, "Generated summaries of papers (and their pages)")
        
        # 生成本篇文章的topic
        topic_prompt = generate_brief_topic_prompt(abstract)
        topic = self.llm_client.get_completion(topic_prompt)
        print("topic:")
        print(topic)
        
        # 生成文献树
        args.topic = topic
        roots,dags = run_dag_to_classifier(
            args,
            internal_collection
        )
        
        print("正在生成可视化...")
        
        proj_root = Path(__file__).parent.parent.parent
        dir = str(proj_root / "multi_dim_literature_visualizations")
        visualizer,arxiv_trees = visualize_dags(roots, dags, output_dir=dir,topic=args.topic)
        
        ## 为生成related work，先生成大纲
        print("正在生成related work大纲...")
        outline_prompt = generate_related_work_outline_prompt(abstract,arxiv_trees)
        outline = self.llm_client.get_completion(outline_prompt)
        outline = json_repair.loads(outline)
        print("outline:")
        print(outline)
        
        topic1 = outline["outline"][0]
        topic2 = outline["outline"][1]
        
        args.dimensions = [topic1, topic2]
        roots,dags,id2node,label2node = build_dags(args)
        results = label_papers_by_topic(
            args,   
            internal_collection,
            [topic1, topic2]    
        )
        print(results)
        
        update_roots_with_labels(roots, results, internal_collection, args)
        grouped = {dim:[
        {
            "paper_id":pid,
            "title":paper.title,
            "abstract":paper.abstract,
            "summary":paper.summary,
            "citations":paper.citations
        }
        for pid,paper in roots[dim].papers.items()
    ] for dim in args.dimensions}
        
        dim_1 = args.dimensions[0]
        dim_2 = args.dimensions[1]
        grouped_dim_1 = grouped[dim_1]
        grouped_dim_2 = grouped[dim_2]
        
        
        
        # 生成related work
        prompt = generate_related_work_prompt_with_arxiv_trees(abstract,args.dimensions,grouped)
        related_work = self.llm_client.get_completion(prompt)
        print("related_work:")
        print(related_work)
        
        client = DeepSeekClient(
            api_key = os.environ.get("DEEPSEEK_API_KEY", ""),
            model_name = "deepseek-chat"
        )
        
        feedback_prompt = genrate_original_related_work_feedback_prompt(related_work)
        feedback = self.llm_client.get_completion(feedback_prompt)
        print("feedback:")
        print(feedback)
        
        prompt_for_revision = generate_related_work_revision_prompt(abstract,related_work,feedback,args.dimensions)
        related_work_revision = client.get_completion(prompt_for_revision)
        related_work_revision = related_work_revision.replace("```json", "").replace("```", "")
        related_work_revision = json_repair.loads(related_work_revision)
        revised_related_work,cited_ids = related_work_revision["related_work"],related_work_revision["cite_ids"]
        print("revised_related_work:")
        print(revised_related_work)
        print("cited_ids:")
        print(cited_ids)
            
        filtered_citations: list[str] = filter_citations(
            related_works_section=revised_related_work, citation_strings=[obj["citation"] for obj in relevant_pages]
        )
        args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
        print("filtered_citations:")
        print(filtered_citations)
        date = datetime.date.today()
        os.makedirs(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}", exist_ok=True)
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/related_work_with_citations.txt", "w", encoding="utf-8") as f:
            f.write("="*50 + "\n" + "related_work" + "\n" + "="*50 + "\n" + related_work + "\n" + "="*50 + "\n" + "citations" + "\n" + "="*50 + "\n" + str(filtered_citations))
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/grouped_dim_1.json", "w", encoding="utf-8") as f:
            json.dump(grouped_dim_1,f,ensure_ascii=False,indent=4)
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/grouped_dim_2.json", "w", encoding="utf-8") as f:
            json.dump(grouped_dim_2,f,ensure_ascii=False,indent=4)
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/related_work_revision.txt", "w", encoding="utf-8") as f:
            f.write("="*50 + "\n" + "related_work_revision" + "\n" + "="*50 + "\n" + revised_related_work + "\n" + "="*50 + "\n" + "citations" + "\n" + "="*50 + "\n" + str(filtered_citations))
        if status_callback:
            status_callback(8, f"Generated related work section with {len(filtered_citations)} citations")
        
        return {"related_works": revised_related_work, "cited_ids": cited_ids, "citations": filtered_citations, "selected_papers": relevant_pages}
    
    def generate_related_work(
        self,
        abstract: str,
        breadth: int,
        depth: int,
        diversity: float,
        status_callback: Optional[Callable] = None,
    ) -> dict[str, str | list[str] | list[dict]]:
        """
        Generate a related work section based on an abstract.

        Args:
            abstract: The input abstract text
            breadth: Number of papers to consider
            depth: Number of pages to extract from each paper
            diversity: Diversity factor for paper selection (0-1)
            status_callback: Callback function that will update jobs according to the function progress

        Returns:
            Dictionary with 'related_works' text and 'citations' list
        """
        if status_callback:
            status_callback(1, "Initializing")

        embedded_abstract = self.sentence_embedding_model.encode(abstract)
        # topic = self.topic_model.transform(abstract)
        # topic_id = topic[0][0]

        # Query Milvus Vector DB
        if status_callback:
            status_callback(2, "Querying Vector DB for matches (this may take a while)")

        query_data: list[list[dict]] = self.db_client.search(
            collection_name="abstracts",
            data=[embedded_abstract],
            limit=6 * breadth,
            anns_field="embedding",
            # filter = f'topic == {topic_id}',
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["embedding"],
        )

        if status_callback:
            status_callback(3, f"Retrieved {len(query_data[0])} papers from the DB")

        # Clean DB response data
        papers_data: list[dict] = query_data[0]
        for obj in papers_data:
            obj["embedding"] = obj["entity"]["embedding"]
            obj.pop("entity")

        # Select a longlist of papers
        selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
            paper_data=papers_data, k=3 * breadth, diversity_weight=diversity
        )

        if status_callback:
            status_callback(
                4,
                f"Selected {len(selected_papers)} papers for the longlist, retrieving full text(s)"
                f" (this might take a while)",
            )

        # Generate embeddings of each page of every paper in the longlist
        page_embeddings: list[list[dict]] = []
        for paper in selected_papers:
            arxiv_id = paper["id"]
            result = process_arxiv_paper_with_embeddings(arxiv_id, self.topic_model)
            if result:
                page_embeddings.append(result)

        if status_callback:
            status_callback(5, f"Generated page embeddings for {len(page_embeddings)} papers")

        # Generate shortlist of papers (at most k pages per paper, at most b papers in total)
        relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
            paper_embeddings=page_embeddings,
            input_string=abstract,
            topic_model=self.topic_model,
            k=depth,
            b=breadth,
            diversity_weight=diversity,
            skip_first=False,
        )

        if status_callback:
            status_callback(6, f"Selected {len(relevant_pages)} papers for the shortlist")

        internal_collection = {}
        data = []
        # Generate summaries for individual papers (taking all relevant pages into account)
        for obj in relevant_pages:
            # Because paper_id != arXiv_id -> retrieve arXiv id/
            arxiv_id = papers_data[obj["paper_id"]]["id"]
            arxiv_abstract = get_arxiv_abstract(arxiv_id)
            text_segments = obj["text"]
            
            title = get_arxiv_title(arxiv_id)
            
            # Create prompt
            prompt = generate_summary_prompt_with_page_content(
                abstract_source_paper=abstract,
                abstract_to_be_cited=arxiv_abstract,
                page_text_to_be_cited=text_segments,
                sentence_count=5,
            )
            internal_collection[arxiv_id] = Paper(
                arxiv_id, 
                title, 
                arxiv_abstract, 
                label_opts=["tasks", "datasets", "methodologies", "evaluation_methods"], 
                internal=True
            )
            temp_dict = {"Title": title, "Abstract": arxiv_abstract}
            data.append(temp_dict)
            # Use the appropriate LLM client based on the provider
            response: str = self.llm_client.get_completion(prompt)
            obj["summary"] = response
            obj["citation"] = get_arxiv_citation(arxiv_id)
            
            
            
        if status_callback:
            status_callback(7, "Generated summaries of papers (and their pages)")
            
        # Generate the final related works section text
        prompt = generate_related_work_prompt(
            source_abstract=abstract, data=relevant_pages, paragraph_count=math.ceil(breadth / 2), add_summary=False
        )

        # Use the appropriate LLM client based on provider
        related_works_section: str = self.llm_client.get_completion(prompt)

        filtered_citations: list[str] = filter_citations(
            related_works_section=related_works_section, citation_strings=[obj["citation"] for obj in relevant_pages]
        )

        if status_callback:
            status_callback(8, f"Generated related work section with {len(filtered_citations)} citations")

        return {"related_works": related_works_section, "citations": filtered_citations, "selected_papers": relevant_pages}

    def generate_related_work_from_paper(
        self,
        pages: list[str],
        breadth: int,
        depth: int,
        diversity: float,
        status_callback: Optional[Callable] = None,
    ) -> dict[str, str | list[str]]:
        """
        Generate a related work section based on a full paper.

        Args:
            pages: List of paper pages
            breadth: Number of papers to consider
            depth: Number of pages to extract from each paper
            diversity: Diversity factor for paper selection (0-1)
            status_callback: Callback function that will update jobs according to the function progress

        Returns:
            Dictionary with 'related_works' text and 'citations' list
        """
        if status_callback:
            status_callback(1, "Initializing.")

        # Create embeddings for all pages
        page_embeddings = [self.sentence_embedding_model.encode(page) for page in pages]

        # Query Milvus Vector DB for each page
        if status_callback:
            status_callback(2, "Querying Vector DB for matches (this may take a while)")

        all_query_data: list[list[dict]] = []
        for embedding in page_embeddings:
            query_result = self.db_client.search(
                collection_name="abstracts",
                data=[embedding],
                limit=6 * breadth,
                anns_field="embedding",
                # filter = f'topic == {topic_id}',  # Could potentially use topic_ids here
                search_params={"metric_type": "COSINE", "params": {}},
                output_fields=["embedding"],
            )
            all_query_data.extend(query_result)

        if status_callback:
            status_callback(3, f"Retrieved papers from DB for {len(all_query_data)} pages")

        # Aggregate similarity scores for papers that appear multiple times
        paper_scores: dict[str, float] = {}
        paper_data: dict[str, dict] = {}

        for page_results in all_query_data:
            for result in page_results:
                paper_id = result["id"]
                similarity_score = result["distance"]  # Assuming this is the similarity score

                if paper_id in paper_scores:
                    paper_scores[paper_id] += similarity_score
                else:
                    paper_scores[paper_id] = similarity_score
                    paper_data[paper_id] = {"id": paper_id, "embedding": result["entity"]["embedding"]}

        # Convert aggregated results back to format expected by select_diverse_papers
        # Sort papers by aggregated score and take top 6*breadth papers
        top_paper_ids = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)[: 6 * breadth]

        # Convert back to original format expected by select_diverse_papers
        # Each entry should be a list with one dict per query result
        aggregated_query_data = [
            {"id": paper_id, "embedding": paper_data[paper_id]["embedding"], "distance": score}
            for paper_id, score in top_paper_ids
        ]

        # Select a longlist of papers using aggregated scores
        selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
            paper_data=aggregated_query_data, k=3 * breadth, diversity_weight=diversity
        )

        if status_callback:
            status_callback(
                4,
                f"Selected {len(selected_papers)} papers for the longlist, retrieving full text(s)"
                f" (this might take a while)",
            )

        # Generate embeddings of each page of every paper in the longlist
        page_embeddings_papers: list[list[dict]] = []
        for paper in selected_papers:
            arxiv_id = paper["id"]
            result = process_arxiv_paper_with_embeddings(arxiv_id, self.topic_model)
            if result:
                page_embeddings_papers.append(result)

        if status_callback:
            status_callback(5, f"Generated page embeddings for {len(page_embeddings)} papers")

        # Generate shortlist of papers using first page as reference
        # (you might want to modify this to consider all input pages)
        relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
            paper_embeddings=page_embeddings_papers,
            input_string=pages[0],  # Using first page as reference
            topic_model=self.topic_model,
            k=depth,
            b=breadth,
            diversity_weight=diversity,
            skip_first=False,
        )

        if status_callback:
            status_callback(6, f"Selected {len(relevant_pages)} papers for the shortlist")

        # Generate summaries for individual papers
        for obj in relevant_pages:
            arxiv_id = aggregated_query_data[obj["paper_id"]]["id"]
            arxiv_abstract = get_arxiv_abstract(arxiv_id)
            text_segments = obj["text"]
            # Create prompt
            prompt = generate_summary_prompt_with_page_content(
                abstract_source_paper=pages[0],  # Using first page as reference
                abstract_to_be_cited=arxiv_abstract,
                page_text_to_be_cited=text_segments,
                sentence_count=5,
            )

            # Use the appropriate LLM client
            response: str = self.llm_client.get_completion(prompt)
            obj["summary"] = response
            obj["citation"] = get_arxiv_citation(arxiv_id)

        if status_callback:
            status_callback(7, "Generated summaries of papers (and their pages)")

        # Generate the final related works section text
        prompt = generate_related_work_prompt(
            source_abstract=pages[0],  # Using first page as reference
            data=relevant_pages,
            paragraph_count=math.ceil(breadth / 2),
            add_summary=False,
        )

        # Use the appropriate LLM client
        related_works_section: str = self.llm_client.get_completion(prompt)

        filtered_citations: list[str] = filter_citations(
            related_works_section=related_works_section, citation_strings=[obj["citation"] for obj in relevant_pages]
        )

        if status_callback:
            status_callback(8, f"Generated related work section with {len(filtered_citations)} citations")

        return {"related_works": related_works_section, "citations": filtered_citations}

    def generate_answer_to_scientific_question(
        self,
        question: str,
        breadth: int,
        depth: int,
        diversity: float,
        status_callback: Optional[Callable] = None,
    ) -> dict[str, str | list[str]]:
        """
        Generate an answer to a scientific question.

        Args:
            question: The input question text
            breadth: Number of papers to consider
            depth: Number of pages to extract from each paper
            diversity: Diversity factor for paper selection (0-1)
            status_callback: Callback function that will update jobs according to the function progress

        Returns:
            Dictionary with 'question_answer' text and 'citations' list
        """
        if status_callback:
            status_callback(1, "Initializing.")

        embedded_abstract = self.sentence_embedding_model.encode(question)
        # topic = self.topic_model.transform(question)
        # topic_id = topic[0][0]

        # Query Milvus Vector DB
        if status_callback:
            status_callback(2, "Querying Vector DB for matches (this may take a while)")

        query_data: list[list[dict]] = self.db_client.search(
            collection_name="abstracts",
            data=[embedded_abstract],
            limit=6 * breadth,
            anns_field="embedding",
            # filter = f'topic == {topic_id}',
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["embedding"],
        )

        if status_callback:
            status_callback(3, f"Retrieved {len(query_data[0])} papers from the DB")

        # Clean DB response data
        papers_data: list[dict] = query_data[0]
        for obj in papers_data:
            obj["embedding"] = obj["entity"]["embedding"]
            obj.pop("entity")

        # Select a longlist of papers
        selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
            paper_data=papers_data, k=3 * breadth, diversity_weight=diversity
        )

        if status_callback:
            status_callback(
                4,
                f"Selected {len(selected_papers)} papers for the longlist, retrieving full text(s)"
                f" (this might take a while)",
            )

        # Generate embeddings of each page of every paper in the longlist
        page_embeddings: list[list[dict]] = []
        for paper in selected_papers:
            arxiv_id = paper["id"]
            result = process_arxiv_paper_with_embeddings(arxiv_id, self.topic_model)
            if result:
                page_embeddings.append(result)

        if status_callback:
            status_callback(5, f"Generated page embeddings for {len(page_embeddings)} papers")

        # Generate shortlist of papers (at most k pages per paper, at most b papers in total)
        relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
            paper_embeddings=page_embeddings,
            input_string=question,
            topic_model=self.topic_model,
            k=depth,
            b=breadth,
            diversity_weight=diversity,
            skip_first=False,
        )

        if status_callback:
            status_callback(6, f"Selected {len(relevant_pages)} papers for the shortlist")

        # Generate summaries for individual papers (taking all relevant pages into account)
        for obj in relevant_pages:
            # Because paper_id != arXiv_id -> retrieve arXiv id/
            arxiv_id = papers_data[obj["paper_id"]]["id"]
            arxiv_abstract = get_arxiv_abstract(arxiv_id)
            text_segments = obj["text"]
            # Create prompt
            prompt = generate_summary_prompt_question_with_page_content(
                question=question, abstract_to_be_considered=arxiv_abstract, page_text_to_be_cited=text_segments
            )

            # Use the appropriate LLM client
            response: str = self.llm_client.get_completion(prompt)
            obj["summary"] = response
            obj["citation"] = get_arxiv_citation(arxiv_id)

        if status_callback:
            status_callback(7, "Generated summaries of papers (and their pages)")

        # Generate the final question answer
        prompt = generate_question_answer_prompt(question=question, data=relevant_pages)

        # Use the appropriate LLM client
        question_answer: str = self.llm_client.get_completion(prompt)

        filtered_citations: list[str] = filter_citations(
            related_works_section=question_answer, citation_strings=[obj["citation"] for obj in relevant_pages]
        )

        if status_callback:
            status_callback(8, f"Generated answer to question with {len(filtered_citations)} citations")

        return {"question_answer": question_answer, "citations": filtered_citations}
