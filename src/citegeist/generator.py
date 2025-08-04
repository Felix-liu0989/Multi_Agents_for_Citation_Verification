# Imports
import json
import math
import os
import sys
import re
sys.path.append("/home/liujian/project/2025-07/A2R-code-reproduction/src")
from evaluation.agents.judge import Judge
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
    generate_related_work_outline_prompt_various_1,
    generate_related_work_revision_prompt_without_DAG,
    generate_summary_prompt_with_second_round_retrieved_content,
    process_data_for_extract_cited_sentences,
    process_data_for_classify_errors,
    process_data_for_correct_citation_errors,
)
import json_repair
from pathlib import Path
from post_hoc_rag.retriever import ArxivRetriever
from post_hoc_rag.reranker import Reranker
from post_hoc_rag.read_dataset import Reader

reader = Reader("/home/liujian/project/2025-07/A2R-code-reproduction/datasets/retrieval_data.json")
arxiv_corpus = reader.get_data()
# Load environment variables
load_dotenv()


retriever = ArxivRetriever(
    model_name="/home/liujian/project/2025-07/A2R-code-reproduction/bge-m3",
    arxiv_corpus=arxiv_corpus,
    device="cuda",
    language="en",
    faiss_index_path="/home/liujian/project/2025-07/A2R-code-reproduction/faiss_index_naive_rag"
)
reranker = Reranker(
    rerank_model_name_or_path="/home/liujian/project/2025-07/A2R-code-reproduction/bge-reranker-large",
    device="cuda",
)
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


judge_gemini = Judge(model="google/gemini-2.5-flash")
judge_deepseek = Judge(model="deepseek-chat")

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
    
    def generate_related_work_MACG(
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
        with open("relevant_pages.json", "w", encoding="utf-8") as f:
            json.dump(relevant_pages, f, indent=4, ensure_ascii=False)

        if status_callback:
            status_callback(6, f"Selected {len(relevant_pages)} papers for the shortlist")
        
        selected_papers = []
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
            idx = {
                "cite_ids":[id],
                "title":title,
                "abstract":arxiv_abstract,
                "summary":response,
                "citations":get_arxiv_citation(arxiv_id)
            }
            selected_papers.append(idx)
            id += 1
        # with open("internal_collection.json", "w", encoding="utf-8") as f:
        #     json.dump(data, f, indent=4, ensure_ascii=False)
        
        with open("selected_papers.json", "w", encoding="utf-8") as f:
            json.dump(selected_papers, f, indent=4, ensure_ascii=False)
        
        # with open("/home/liujian/internal_collection.json", "r", encoding="utf-8") as f:
        #       internal_collection = json.load(f)
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
        

        print("outline:")
        print(outline)
        
        subsection_titles = outline["outline"]
            
        
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
        related_work_with_citations = self.llm_client.get_completion(prompt)
        related_work_with_citations = related_work_with_citations.replace("```json","").replace("```","")
        related_work_with_citations = json_repair.loads(related_work_with_citations)
        print("related_work_with_citations:")
        print(related_work_with_citations)
        related_work = related_work_with_citations["related_work"]
        citations = related_work_with_citations["cite_ids"]
        print("related_work:")
        print(related_work)
        print("citations:")
        print(citations)
        
        # prompt_for_extract_cited_sentences = process_data_for_extract_cited_sentences(related_work)
        # cited_sentences = self.llm_client.get_completion(prompt_for_extract_cited_sentences)
        # cited_sentences = json_repair.loads(cited_sentences)
        # print("cited_sentences:")
        # print(cited_sentences)
        
        # judge_gemini = Judge(model="google/gemini-2.5-flash")
        # judge_deepseek = Judge(model="deepseek-chat")
        
        # ids = [i for i in citations if "paper_id" in i and i["paper_id"] is not None]
        
        # selected_papers = relevant_pages
        
        client = DeepSeekClient(
            api_key = os.environ.get("DEEPSEEK_API_KEY", ""),
            model_name = "deepseek-chat"
        )
        
        feedback_prompt = genrate_original_related_work_feedback_prompt(related_work)
        feedback = self.llm_client.get_completion(feedback_prompt)
        print("feedback:")
        print(feedback)
        
        prompt_for_revision = generate_related_work_revision_prompt(abstract,related_work,feedback,citations,args.dimensions)
        related_work_revision = client.get_completion(prompt_for_revision)
        print("related_work_revision:")
        print(related_work_revision)
        related_work_revision = related_work_revision.replace("```json","").replace("```","")
        related_work_revision_dict = json_repair.loads(related_work_revision)
        related_work_revision = related_work_revision_dict["related_work"]
        citations = related_work_revision_dict["cite_ids"]
        
        related_work_revision,validation_results,error_types = self.validate_and_correct_citations(related_work_revision,selected_papers,citations)
        
           
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
    
    def validate_and_correct_citations(self, related_work, selected_papers, citations):
        """
        验证和纠正引用错误的内联方法
        """
        # 1. 提取引用的句子
        prompt_for_extract_cited_sentences = process_data_for_extract_cited_sentences(related_work)
        cited_sentences = self.llm_client.get_completion(prompt_for_extract_cited_sentences)
        cited_sentences = json_repair.loads(cited_sentences)
        print("selected_papers:")
        print(selected_papers)
        # 2. 构建引用验证数据
        quotes_with_citation_info = self._build_citation_verification_data(
            cited_sentences, selected_papers, citations
        )
        
        # 3. 使用双模型验证
        validation_results = self._validate_citations_with_dual_models(quotes_with_citation_info,related_work,citations,selected_papers)
        
        # 4. 分类错误类型
        error_types = self._classify_citation_errors(validation_results)
        
        # 5. 纠正错误
        corrected_related_work = self._correct_citation_errors(related_work, error_types)
        
        return corrected_related_work, validation_results, error_types
    
    
    def _count_sentences(self,text):
        sentences = re.split(r"[.!?\n]+(?:\s|\n|$)", text.strip())
        sentences = [s for s in sentences if s]
        return sentences
    
    
    def _build_citation_verification_data(
        self, 
        quotes: list[str], 
        selected_papers: list[dict], 
        cite_ids: list[dict]
    ):
        """构建引用验证所需的数据结构"""
        quotes_with_citation_info = {quote: [] for quote in quotes}
        ids = [i for i in cite_ids if "paper_id" in i and i["paper_id"] is not None]
        print("ids:")
        print(ids)
        for cited_id in ids:
            for id,selected_paper in enumerate(selected_papers):
                cited_paper_id = cited_id["paper_id"]
                cited_paper_id = cited_paper_id.replace("paper_","")
                if cited_paper_id == "null":
                    continue
                if "cite_ids" not in selected_paper:
                    continue
                try:
                    if int(cited_paper_id) == int(selected_paper["cite_ids"][0]):
                        cited_id["summary"] = selected_paper["summary"]
                except Exception as e:
                    print(e)
                    continue
                
        for quote in quotes:
            for cited_id in ids:
                c_text = cited_id["citation_text"]
                year = re.findall(r'(?<!\d)\d{4}(?!\d)',c_text)
                year = year[0] if year else None
                if "summary" not in cited_id:
                    continue
                if c_text in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["summary"])
                elif c_text.split(".")[0] in quote and year in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["summary"])
                elif c_text.split("(")[0] in quote and year in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["summary"])
                elif c_text.split(",")[0] in quote and year in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["summary"])
                elif c_text.split("et")[0] in quote and year in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["summary"])
                elif c_text.split("&")[0] in quote and year in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["summary"])
                elif c_text.split("and")[0] in quote and year in quote:
                    quotes_with_citation_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["summary"])
        print("quotes_with_citation_info:")
        print(quotes_with_citation_info)
        return quotes_with_citation_info
    
    def _validate_citations_with_dual_models(self, quotes_with_citation_info,related_work,citations,selected_papers):
        """使用双模型验证引用准确性"""

        yes_gemini = []
        no_gemini = []
        yes_deepseek = []
        no_deepseek = []
        validation_results = {
                "yes_gemini": [],
                "no_gemini": [],
                "yes_deepseek": [],
                "no_deepseek": [],
                "yes_ids": [],
                "no_ids": [],
                "claim_precision": 0,
                "citation_precision": 0,
                "reference_precision": 0,
                "citation_density": 0,
                "avg_citation_per_sentence": 0
            }
        for i,quote in enumerate(quotes_with_citation_info.keys()):
            if len(quotes_with_citation_info[quote]) == 0:
                continue
            q = list(set(quotes_with_citation_info[quote]))
            source = "\n".join(q)
            print(f"source: {source}")
            print(f"quote: {quote}")
            score = judge_gemini._get_pair_score_new(source, quote)
            print(f"{i}. {score} by gemini")
            if score.lower() == "yes":
                yes_gemini.append({"id": i, "claim": quote, "source": source})
            else:
                no_gemini.append({"id": i, "claim": quote, "source": source})
            score = judge_deepseek._get_pair_score_new(source, quote)
            print(f"{i}. {score} by deepseek")
            if score.lower() == "yes":
                yes_deepseek.append({"id": i, "claim": quote, "source": source})
            else:
                no_deepseek.append({"id": i, "claim": quote, "source": source})
            quotes_with_citation_info[quote] = q
        # 计算yes_ids和no_ids,两个模型都认为才算对
        yes_gemini_ids = [(j["id"],j["source"]) for j in yes_gemini]
        yes_deepseek_ids = [(j["id"],j["source"]) for j in yes_deepseek]
        no_gemini_ids = [(j["id"],j["source"]) for j in no_gemini]
        no_deepseek_ids = [(j["id"],j["source"]) for j in no_deepseek]
        yes_ids = list(set(yes_gemini_ids) & set(yes_deepseek_ids))
        # 两个模型一个认为对，一个认为错，则算错
        no_ids = list(set(no_gemini_ids) | set(no_deepseek_ids))
        # 带引用的claim数量
        total_claims = len(quotes_with_citation_info.keys())
        # related work中总共的句子数量
        total_sentences = len(self._count_sentences(related_work))
        
        ## 1.claim_precision 正确引用的claim数量 / 所有claim数量（句子层面的）
        claim_precision = len(yes_ids) / total_claims
        claim_precision = round(claim_precision,3)
        
        ## 2.citation_precision 正确引用数量 / 所有引用数量 （引用层面的）
        correct_source = 0
        for yes_id in yes_ids:
            source = yes_id[1]
            source = source.split("\n")
            correct_source += len(source)
        
        total_citations = len(citations)
        
        citation_precision = correct_source / total_citations
        citation_precision = round(citation_precision,3)
        
        ## 3.reference_precision 被正确引用的不同论文篇数（其实就是正确的引用去重之后） / 参考文献总篇数 （信源层面的）
        correct_reference_source = set()
        for yes_id in yes_ids:
            source = yes_id[1]
            source = source.split("\n")
            for s in source:
                correct_reference_source.add(s)

        ## 全部被引用上的文章数量，将被引用的信源进行去重，然后计算数量
        unique_citations = set()
        for quote in quotes_with_citation_info.keys():
            for citation in quotes_with_citation_info[quote]:
                unique_citations.add(citation)
        reference_precision = len(correct_reference_source) / len(selected_papers) # unique_citations
        reference_precision = round(reference_precision,3)
        
        ## 4. citation_density 引用总数 ÷ 正文句子总数(引用密度层面的)
        citation_density = total_citations / total_sentences
        citation_density = round(citation_density,3)
        
        ## 5. avg_citation_per_sentence 引用总数 ÷ claim总数(引用密度层面的)
        avg_citation_per_sentence = total_citations / total_claims
        avg_citation_per_sentence = round(avg_citation_per_sentence,3)
            
        validation_results = {
            "yes_gemini": yes_gemini,
            "no_gemini": no_gemini,
            "yes_deepseek": yes_deepseek,
            "no_deepseek": no_deepseek,
            "yes_ids": yes_ids,
            "no_ids": no_ids,
            "claim_precision": claim_precision,
            "citation_precision": citation_precision,
            "reference_precision": reference_precision,
            "citation_density": citation_density,
            "avg_citation_per_sentence": avg_citation_per_sentence
        }  
            
            
        return validation_results

    def _classify_citation_errors(self, validation_results):
        """分类引用错误类型"""
        yes_ids = validation_results["yes_ids"]
        no_ids = validation_results["no_ids"]
        no_ids = [j[0] for j in no_ids]
        yes_gemini = validation_results["yes_gemini"]
        no_gemini = validation_results["no_gemini"]
        yes_deepseek = validation_results["yes_deepseek"]
        no_deepseek = validation_results["no_deepseek"]
        
        direct_contradiction = 0
        information_not_present = 0
        misrepresentation = 0
        incorrect_attribution = 0
        other = 0
        
        error_types = []
        for quote_id in no_ids:
            gemini_item = next((item for item in no_gemini if item["id"] == quote_id), None)
            deepseek_item = next((item for item in no_deepseek if item["id"] == quote_id), None)
            
            if gemini_item:
                claim = gemini_item["claim"]
                source = gemini_item["source"]
            elif deepseek_item:
                claim = deepseek_item["claim"]
                source = deepseek_item["source"]
            else:
                print(f"Warning: quote_id {quote_id} not found in either no_gemini or no_deepseek")
                continue
            
            try:
                prompt = process_data_for_classify_errors(claim,source)
                result = self.llm_client.get_completion(prompt)
                result = result.replace("```json", "").replace("```", "")
                result = json_repair.loads(result)
                print(result)
                print(result["error_type"])
                if "Direct Contradiction" in result["error_type"]:
                    direct_contradiction += 1
                elif "Information Not Present / Unsubstantiated" in result["error_type"]:
                    information_not_present += 1
                elif "Misrepresentation / Imprecise Wording" in result["error_type"]:
                    misrepresentation += 1
                elif "Incorrect Attribution" in result["error_type"]:
                    incorrect_attribution += 1
                else:
                    other += 1
                errors_count = {
                    "direct_contradiction": direct_contradiction,
                    "information_not_present": information_not_present,
                    "misrepresentation": misrepresentation,
                    "incorrect_attribution": incorrect_attribution,
                    "other": other
                }
                if isinstance(result, dict):
                    result["id"] = quote_id
                    result["claim"] = claim
                    result["source"] = source
                    result["errors_count"] = errors_count
                    error_types.append(result)
                else:
                    print(f"Error: Unexpected result format: {result}")
                    continue
                
            except Exception as e:
                print(e)
                continue
            print("-"*100)
        
        print(f"Direct Contradiction: {direct_contradiction}")
        print(f"Information Not Present / Unsubstantiated: {information_not_present}")
        print(f"Misrepresentation / Imprecise Wording: {misrepresentation}")
        print(f"Incorrect Attribution: {incorrect_attribution}")
        print(f"Other: {other}")
        return error_types

    def _correct_citation_errors(self, related_work, error_types):
        """纠正引用错误"""
        correct_count = 0
        for error in error_types:
            source = error["source"]
            claim = error["claim"]
            error_type = error["error_type"]
            error_description = error["error_description"]
            
            prompt = process_data_for_correct_citation_errors(source,claim,error_type,error_description)
            result = self.llm_client.get_completion(prompt)
            result = result.replace("```json", "").replace("```", "")
            result = json_repair.loads(result)
            error["corrected_claim"] = result["corrected_claim"]
            error["explanation"] = result["explanation"]
            error["key_changes"] = result["key_changes"]
            print(f"corrected_claim: {result['corrected_claim']}")
            print(f"explanation: {result['explanation']}")
            score = judge_gemini._get_pair_score_new(source, result["corrected_claim"])
            print(f"score: {score}")
            print("-"*100)
            if score == "Yes":
                correct_count += 1
            else:
                prompt = process_data_for_classify_errors(result["corrected_claim"],source)
                result = self.llm_client.get_completion(prompt)
                result = result.replace("```json", "").replace("```", "")
                result = json_repair.loads(result)
                error_type = result["error_type"]
                error_description = result["error_description"]
        
        
        related_work_revision = related_work
        for error in error_types:
            error_claim = error["claim"]
            error_claim = error_claim.replace("<cite>", "").replace("</cite>", "")
            if error_claim[-1] == ".":
                error_claim = error_claim[:-1]
            related_work_revision = related_work_revision.replace(error_claim,error["corrected_claim"])
        print(f"related_work_revision: {related_work_revision}")
        return related_work_revision
    
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
    
    def ablation_experiment_without_feedback_revision(
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
        if status_callback:
            status_callback(8, f"Generated related work section with {len(filtered_citations)} citations")
        args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
        final= {"related_works": related_work, "citations": filtered_citations, "selected_papers": relevant_pages}
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/final.json", "w", encoding="utf-8") as f:
            json.dump(final,f,ensure_ascii=False,indent=4)
            
        args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
        return final
    
    def ablation_experiment_without_feedback_revision_1(
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
        unique_results = set()
        retrieval_results = []
        vector_results = retriever.vector_retrieval(abstract,top_k=10)
        for result in vector_results:
            unique_key = (result['instruction'],result['output'],result['reference'])
            if unique_key not in unique_results:
                unique_results.add(unique_key)
                i = {
                    "query":abstract,
                    "instruction":result['instruction'],
                    "output":result['output'],
                    "reference":result['reference']
                }
                retrieval_results.append(i)
        reranked_results = reranker.rerank(retrieval_results,abstract,k=20)
        internal_collection = {}
        selected_papers = []
        for id,result in enumerate(reranked_results):
            title = result['instruction']
            title = title.split("\n")[0]
            print(f"title: {title}")
            print(f"output: {result['output']}")
            print(f"reference: {result['reference']}")
            print("-"*100)
            retrieved_abstract = result["output"]
            print(f"retrieved_abstract: {retrieved_abstract}")
            print(f"reference: {result['reference']}")
        
            internal_collection[id] = Paper(
                id,
                title,
                retrieved_abstract,
                label_opts=["tasks", "datasets", "methodologies", "evaluation_methods"],
                internal=True
            )
            internal_collection[id].citations = result['reference']
            prompt = generate_summary_prompt_with_second_round_retrieved_content(
                abstract_source_paper=abstract,
                abstract_to_be_cited=retrieved_abstract,
                sentence_count=5,
            )
            response: str = self.llm_client.get_completion(prompt)
            internal_collection[id].summary = response
            idx = {
                "cite_ids":[id],
                "title":title,
                "abstract":retrieved_abstract,
                "summary":response,
                "citations":result['reference']
            }
            selected_papers.append(idx)
            
        roots,dags,id2node,label2node = build_dags(args)
        results = label_papers_by_topic(
            args,
            internal_collection,
            args.dimensions
        )
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
        
        prompt = generate_related_work_prompt_with_arxiv_trees(
            abstract,
            args.dimensions,
            grouped
        )
        related_work = self.llm_client.get_completion(prompt)
        print("related_work:")
        print(related_work)
        
        filtered_citations: list[str] = filter_citations(
            related_works_section=related_work, citation_strings=[obj["citations"] for obj in selected_papers]
        )
        print("filtered_citations:")
        print(filtered_citations)
        date = datetime.date.today()
        os.makedirs(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}", exist_ok=True)
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/related_work_with_citations.txt", "w", encoding="utf-8") as f:
            f.write("="*50 + "\n" + "related_work" + "\n" + "="*50 + "\n" + related_work + "\n" + "="*50 + "\n" + "citations" + "\n" + "="*50 + "\n" + str(filtered_citations))
        
        final = {"related_works": related_work, "citations": filtered_citations, "selected_papers": selected_papers}
        return final
    
    def ablation_experiment_without_summarization(
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
        unique_results = set()
        retrieval_results = []
        vector_results = retriever.vector_retrieval(abstract,top_k=10)
        for result in vector_results:
            unique_key = (result['instruction'],result['output'],result['reference'])
            if unique_key not in unique_results:
                unique_results.add(unique_key)
                i = {
                    "query":abstract,
                    "instruction":result['instruction'],
                    "output":result['output'],
                    "reference":result['reference']
                }
                retrieval_results.append(i)
        reranked_results = reranker.rerank(retrieval_results,abstract,k=20)
        internal_collection = {}
        selected_papers = []
        for id,result in enumerate(reranked_results):
            title = result['instruction']
            title = title.split("\n")[0]
            print(f"title: {title}")
            print(f"output: {result['output']}")
            print(f"reference: {result['reference']}")
            print("-"*100)
            retrieved_abstract = result["output"]
            print(f"retrieved_abstract: {retrieved_abstract}")
            print(f"reference: {result['reference']}")
        
            internal_collection[id] = Paper(
                id,
                title,
                retrieved_abstract,
                label_opts=["tasks", "datasets", "methodologies", "evaluation_methods"],
                internal=True
            )
            internal_collection[id].citations = result['reference']
            internal_collection[id].summary = abstract
            idx = {
                "cite_ids":[id],
                "title":title,
                "abstract":retrieved_abstract,
                "summary":abstract,
                "citations":result['reference']
            }
            selected_papers.append(idx)
        # 生成本篇文章的topic
        topic_prompt = generate_brief_topic_prompt(abstract)
        topic = self.llm_client.get_completion(topic_prompt)
        args.topic = topic
        
        roots,dags,id2node,label2node = build_dags(args)
        results = label_papers_by_topic(
            args,
            internal_collection,
            args.dimensions
        )
        update_roots_with_labels(roots, results, internal_collection, args)
        
        print("正在生成可视化...")
        
        proj_root = Path(__file__).parent.parent.parent
        dir = str(proj_root / "multi_dim_literature_visualizations")
        visualizer,arxiv_trees = visualize_dags(roots, dags, output_dir=dir,topic=args.topic)
        
        ## 为生成related work，先生成大纲
        print("正在生成related work大纲...")
        outline_prompt = generate_related_work_outline_prompt_various_1(abstract,arxiv_trees)
        outline = self.llm_client.get_completion(outline_prompt)
        outline = json_repair.loads(outline)
        subsection_titles = outline["outline"]
        print("subsection_titles:")
        print(subsection_titles)
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
    
        
        prompt = generate_related_work_prompt_with_arxiv_trees(
            abstract,
            args.dimensions,
            grouped
        )
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
            related_works_section=related_work, citation_strings=[obj["citations"] for obj in selected_papers]
        )
        print("filtered_citations:")
        print(filtered_citations)
        date = datetime.date.today()
        os.makedirs(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}", exist_ok=True)
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/related_work_with_citations.txt", "w", encoding="utf-8") as f:
            f.write("="*50 + "\n" + "related_work" + "\n" + "="*50 + "\n" + related_work + "\n" + "="*50 + "\n" + "citations" + "\n" + "="*50 + "\n" + str(filtered_citations))
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/related_work_revision.txt", "w", encoding="utf-8") as f:
            f.write("="*50 + "\n" + "related_work_revision" + "\n" + "="*50 + "\n" + related_work_revision + "\n" + "="*50 + "\n" + "citations" + "\n" + "="*50 + "\n" + str(filtered_citations))
        args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
        final = {"related_works": related_work_revision, "citations": filtered_citations, "selected_papers": selected_papers}
        return final
            
            
    def ablation_experiment_without_DAG_1(
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
        unique_results = set()
        retrieval_results = []
        vector_results = retriever.vector_retrieval(abstract,top_k=10)
        for result in vector_results:
            unique_key = (result['instruction'],result['output'],result['reference'])
            if unique_key not in unique_results:
                unique_results.add(unique_key)
                i = {
                    "query":abstract,
                    "instruction":result['instruction'],
                    "output":result['output'],
                    "reference":result['reference']
                }
                retrieval_results.append(i)
        reranked_results = reranker.rerank(retrieval_results,abstract,k=20)
        internal_collection = {}
        selected_papers = []
        for id,result in enumerate(reranked_results):
            title = result['instruction']
            title = title.split("\n")[0]
            print(f"title: {title}")
            print(f"output: {result['output']}")
            print(f"reference: {result['reference']}")
            print("-"*100)
            retrieved_abstract = result["output"]
            print(f"retrieved_abstract: {retrieved_abstract}")
            print(f"reference: {result['reference']}")
        
            internal_collection[id] = Paper(
                id,
                title,
                retrieved_abstract,
                label_opts=["tasks", "datasets", "methodologies", "evaluation_methods"],
                internal=True
            )
            internal_collection[id].citations = result['reference']
            prompt = generate_summary_prompt_with_second_round_retrieved_content(
                abstract_source_paper=abstract,
                abstract_to_be_cited=retrieved_abstract,
                sentence_count=5,
            )
            response: str = self.llm_client.get_completion(prompt)
            internal_collection[id].summary = response
            idx = {
                "cite_ids":[id],
                "title":title,
                "abstract":retrieved_abstract,
                "summary":response,
                "citation":result['reference']
            }
            selected_papers.append(idx)
        
        proj_root = Path(__file__).parent.parent.parent
        dir = str(proj_root / "multi_dim_literature_visualizations")
        
        
        
        # 生成related work
        prompt = generate_related_work_prompt(abstract,selected_papers)
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
        
        prompt_for_revision = generate_related_work_revision_prompt_without_DAG(abstract,related_work,feedback)
        related_work_revision = client.get_completion(prompt_for_revision)
        print("related_work_revision:")
        print(related_work_revision)
        
            
        filtered_citations: list[str] = filter_citations(
            related_works_section=related_work, citation_strings=[obj["citation"] for obj in selected_papers]
        )
        
        
        print("filtered_citations:")
        print(filtered_citations)
        date = datetime.date.today()
        os.makedirs(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}", exist_ok=True)
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/related_work_with_citations.txt", "w", encoding="utf-8") as f:
            f.write("="*50 + "\n" + "related_work" + "\n" + "="*50 + "\n" + related_work + "\n" + "="*50 + "\n" + "citations" + "\n" + "="*50 + "\n" + str(filtered_citations))
        # with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/grouped_dim_1.json", "w", encoding="utf-8") as f:
        #     json.dump(grouped_dim_1,f,ensure_ascii=False,indent=4)
        # with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/grouped_dim_2.json", "w", encoding="utf-8") as f:
        #     json.dump(grouped_dim_2,f,ensure_ascii=False,indent=4)
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/related_work_revision.txt", "w", encoding="utf-8") as f:
            f.write("="*50 + "\n" + "related_work_revision" + "\n" + "="*50 + "\n" + related_work_revision + "\n" + "="*50 + "\n" + "citations" + "\n" + "="*50 + "\n" + str(filtered_citations))
        if status_callback:
            status_callback(8, f"Generated related work section with {len(filtered_citations)} citations")
        args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
        final= {"related_works": related_work_revision, "citations": filtered_citations, "selected_papers":selected_papers}
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/final.json", "w", encoding="utf-8") as f:
            json.dump(final,f,ensure_ascii=False,indent=4)
        # final= {"related_works": related_work_revision, "citations": filtered_citations, "selected_papers": relevant_pages}
        # with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/final.json", "w", encoding="utf-8") as f:
        #     json.dump(final,f,ensure_ascii=False,indent=4)
            
        # args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
        return final
        
        
    def ablation_experiment_without_DAG(
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
            print("response:")
            print(response)
            obj["summary"] = response
            obj["citation"] = get_arxiv_citation(arxiv_id)
            internal_collection[id].summary = response
            internal_collection[id].citations = get_arxiv_citation(arxiv_id)
            id += 1
            
        # with open("internal_collection.json", "w", encoding="utf-8") as f:
        #     json.dump(data, f, indent=4, ensure_ascii=False)
            
        if status_callback:
            status_callback(7, "Generated summaries of papers (and their pages)")
        
        # # 生成本篇文章的topic
        # topic_prompt = generate_brief_topic_prompt(abstract)
        # topic = self.llm_client.get_completion(topic_prompt)
        # print("topic:")
        # print(topic)
        
        # 生成文献树
        # args.topic = topic
        # roots,dags = run_dag_to_classifier(
        #     args,
        #     internal_collection
        # )
        
        # print("正在生成可视化...")
        
        proj_root = Path(__file__).parent.parent.parent
        dir = str(proj_root / "multi_dim_literature_visualizations")
        
        
        
        # 生成related work
        prompt = generate_related_work_prompt(abstract,relevant_pages)
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
        
        prompt_for_revision = generate_related_work_revision_prompt_without_DAG(abstract,related_work,feedback)
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
        # with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/grouped_dim_1.json", "w", encoding="utf-8") as f:
        #     json.dump(grouped_dim_1,f,ensure_ascii=False,indent=4)
        # with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/grouped_dim_2.json", "w", encoding="utf-8") as f:
        #     json.dump(grouped_dim_2,f,ensure_ascii=False,indent=4)
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/related_work_revision.txt", "w", encoding="utf-8") as f:
            f.write("="*50 + "\n" + "related_work_revision" + "\n" + "="*50 + "\n" + related_work_revision + "\n" + "="*50 + "\n" + "citations" + "\n" + "="*50 + "\n" + str(filtered_citations))
        if status_callback:
            status_callback(8, f"Generated related work section with {len(filtered_citations)} citations")
        args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
        final= {"related_works": related_work_revision, "citations": filtered_citations, "selected_papers": relevant_pages}
        with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/final.json", "w", encoding="utf-8") as f:
            json.dump(final,f,ensure_ascii=False,indent=4)
        # final= {"related_works": related_work_revision, "citations": filtered_citations, "selected_papers": relevant_pages}
        # with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/{args.topic}/{date}/final.json", "w", encoding="utf-8") as f:
        #     json.dump(final,f,ensure_ascii=False,indent=4)
            
        # args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods"]
        return final
    
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
    
    def generate_related_work_test(
        self,
        abstract: str,
        breadth: int,
        depth: int,
        diversity: float,
        selected_papers: list[dict],
        status_callback: Optional[Callable] = None,
    ) -> dict[str, str | list[str] | list[dict]]:
        
        prompt = generate_related_work_prompt(
            source_abstract=abstract, data=selected_papers, paragraph_count=math.ceil(breadth / 2), add_summary=False
        )
        related_works_section: str = self.llm_client.get_completion(prompt)
        filtered_citations: list[str] = filter_citations(
            related_works_section=related_works_section, citation_strings=[obj["citation"] for obj in selected_papers]
        )
        return {"related_works": related_works_section, "citations": filtered_citations, "selected_papers": selected_papers}
    
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