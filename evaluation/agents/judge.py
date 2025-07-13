from collections import defaultdict
import re
from citegeist.utils.llm_clients.gemini_client import GeminiClient
from evaluation.agents.prompts import CRITERIA, CRITERIA_BASED_JUDGING_PROMPT, NLI_PROMPT, OUTLINE_EVALUATION_PROMPT, LANGUAGE_EVALUATION_PROMPT, CRITICAL_EVALUATION_PROMPT, get_extraction_prompt, get_citation_extraction_prompt
import os
import logging
import json_repair
import json
logger = logging.getLogger(__name__)

# ====================== Author-Year citation evaluation ======================

class Judge():
    def __init__(
        self,
        model: str,
    ):
        self.model = model
        self.llm_client = GeminiClient(
            api_key = os.environ.get("OPENROUTER_API_KEY", ""),
            model_name = self.model
        )
        
    def __generate_prompt(self, template, paras):
        prompt = template
        for k in paras.keys():
            prompt = prompt.replace(f"[{k}]", paras[k])
        return prompt
    
    def __get_pair_score_new(self,paper_content:str,claim:str):
        max_model_len = 900000          
        max_estimate_char_len = int(max_model_len * 1.25)
        if len(paper_content) > max_estimate_char_len:
            logger.warning(...)
            paper_content = paper_content[:max_estimate_char_len]
            
        content_paras = {'SOURCE': paper_content, 'CLAIM': claim}
        prompt = self.__generate_prompt(NLI_PROMPT, content_paras)
        messages = [{"role": "user","content": prompt}]
        response = self.llm_client.get_chat_completion(messages)
        return response
        # try:
        #     response = self.llm_client.get_chat_completion(messages)
        # except Exception as e:
        #     print(f"Error: {e}")
        #     eval_maxtrix[i][j] = -1
        #     return eval_maxtrix
        # if response and  "yes" in response.lower():
        #     eval_matrix[i][j] = 1
        # return eval_matrix
    
    def __get_pair_score(self, paper_content: str, claim: str, pair_scores: list[list[int]], i: int, j: int, citation_idx: int, raw_claim: str):
        max_model_len = 900000          
        max_estimate_char_len = int(max_model_len * 1.25)
        if len(paper_content) > max_estimate_char_len:
            logger.warning(...)
            paper_content = paper_content[:max_estimate_char_len]
            
        content_paras = {'SOURCE': paper_content, 'CLAIM': claim}
        prompt = self.__generate_prompt(NLI_PROMPT, content_paras)
        messages = [{"role": "user","content": prompt}]
        response = self.llm_client.get_chat_completion(messages)
        return response

    def _make_key(self, ref_str: str):
        """根据参考文献条目或正文引用抽取 key，形式 'Author_Year'。

        同时兼容：
        1. Author (2020)
        2. (Author, 2020)
        3. Author, 2020
        """
        # 捕获 Author 与 4 位年份，二者之间允许任意字符（括号或逗号）
        m = re.search(r"([A-Z][A-Za-z]+)[^\d]{0,30}?(\d{4})", ref_str)
        return f"{m.group(1)}_{m.group(2)}" if m else None
    
    def _build_index_new(
        self,
        citations:list[str],
        cite_info:list[dict],
        selected_papers:list[dict],
    ):
        key2idx: dict[str, int] = {}
        paper_infos: list[dict] = []
        for idx, item in enumerate(selected_papers):
            if idx >= len(citations):
                break
            citation = item["citation"]
            for c in citations:
                if c == citation:
                    cite_id = item['cite_ids']
                    for cite in cite_info:
                        paper_id = cite["paper_id"]
                        key2idx[cite["citation_text"]] = cite_id
                        paper_infos.append({"title": citation, "content": item["text"]})
                        break
        return key2idx, paper_infos


    def _build_index(self, citations: list[str], selected_papers: list[dict], keys: list[str]):
        """构建 key→index 与论文信息列表。

        citations: 生成器返回的已过滤引用字符串列表。
        selected_papers: 与 citations 顺序一致的论文条目 (Generator 中的 relevant_pages)
        """
        
        key2idx: dict[str, int] = {}
        paper_infos: list[dict] = []
        for idx, item in enumerate(selected_papers):
            if idx >= len(citations):
                break
            citation = item["citation"]
            for c in citations:
                if c == citation:
                    for key in keys:
                        if key in c:
                            break
                    if not key:
                        continue
                    key2idx[key] = idx
                    page_texts = item.get("text", [])
                    full_ref = citations[idx]
                    key = self._make_key(full_ref)
                    print(f"key: {key}")
                if not key:
                    continue
                key2idx[key] = idx
                page_texts = item.get("text", [])
            if isinstance(page_texts, list):
                content = "\n".join(page_texts)
            else:
                content = str(page_texts)
            paper_infos.append({"title": citation, "content": content})
        return key2idx, paper_infos
    
    def _get_quote(self,related_text:str):
        """从 Related-Work 正文中抽取带 (Author, Year) 引用的句子以及映射关系。"""
        keys = []
        sentence_pat = re.compile(
                    r"""                      # 选项详解见下方
                    [^.!?]*?                  # A 句首 → 第一个括号前，允许出现任意非句号/问号/感叹号字符（非贪婪）
                    (?:                       #   ┐ 两种主体引用格式
                        [A-Z][A-Za-z.&\s]*?   #   │ 1) 作者名字段，可含空格、and、&、et al.
                        \(\d{4}\)             #   │    再跟 “(YYYY)”
                    |                       #   ├─ 或
                        \(\s*                 #   │ 2) 直接以“(Author,”开头
                            [A-Z][A-Za-z]+    #   │    作者姓
                            [^,]*             #   │    括号内逗号前允许额外内容
                            ,\s*\d{4}\s*      #   │    “, YYYY”
                        \)                    #   │
                    )                         #   ┘
                    [^.?!]*                   # B 括号之后到句末标点前
                    [.!?]                     # C 句末必须以 . ! ? 结束
                    """,
                    re.VERBOSE
                )
        quotes = sentence_pat.findall(related_text)
        claims = [re.sub(r"\([^)]+\)", "", sent).strip() for sent in quotes]
        return quotes,claims
        
    def _get_keys_in_claim(self, related_text: str,cite_info) -> list[str]:
        """从 Related-Work 正文中抽取带 (Author, Year) 引用的句子以及映射关系。"""
        keys = []
        sentence_pat = re.compile(
                    r"""                      # 选项详解见下方
                    [^.!?]*?                  # A 句首 → 第一个括号前，允许出现任意非句号/问号/感叹号字符（非贪婪）
                    (?:                       #   ┐ 两种主体引用格式
                        [A-Z][A-Za-z.&\s]*?   #   │ 1) 作者名字段，可含空格、and、&、et al.
                        \(\d{4}\)             #   │    再跟 “(YYYY)”
                    |                       #   ├─ 或
                        \(\s*                 #   │ 2) 直接以“(Author,”开头
                            [A-Z][A-Za-z]+    #   │    作者姓
                            [^,]*             #   │    括号内逗号前允许额外内容
                            ,\s*\d{4}\s*      #   │    “, YYYY”
                        \)                    #   │
                    )                         #   ┘
                    [^.?!]*                   # B 括号之后到句末标点前
                    [.!?]                     # C 句末必须以 . ! ? 结束
                    """,
                    re.VERBOSE
                )
        quotes = sentence_pat.findall(related_text)
        # # 两种常见引用正则
        # patterns = [
        #     r"[A-Z][A-Za-z]+[^()]*\(\d{4}\)",        # Author (2020)
        #     r"\([A-Z][A-Za-z]+[^\d]*,\s*\d{4}\)"    # (Author, 2020)
        # ]

        # for sent in quotes:
        #     prompt = self.__generate_prompt(get_citation_extraction_prompt(sent), {"SOURCE": sent})
        #     messages = [{"role": "user","content": prompt}]
        #     response = self.llm_client.get_chat_completion(messages)
        #     keys.append(response)
        keys = [item['citation_text'] for item in cite_info]
            
        return keys

    def _harvest_claims(self, related_text: str, key2idx: dict[str, int]):
        """从 Related-Work 正文中抽取带 (Author, Year) 引用的句子以及映射关系。"""
        sentence_pat = re.compile(
                    r"""                      # 选项详解见下方
                    [^.!?]*?                  # A 句首 → 第一个括号前，允许出现任意非句号/问号/感叹号字符（非贪婪）
                    (?:                       #   ┐ 两种主体引用格式
                        [A-Z][A-Za-z.&\s]*?   #   │ 1) 作者名字段，可含空格、and、&、et al.
                        \(\d{4}\)             #   │    再跟 “(YYYY)”
                    |                       #   ├─ 或
                        \(\s*                 #   │ 2) 直接以“(Author,”开头
                            [A-Z][A-Za-z]+    #   │    作者姓
                            [^,]*             #   │    括号内逗号前允许额外内容
                            ,\s*\d{4}\s*      #   │    “, YYYY”
                        \)                    #   │
                    )                         #   ┘
                    [^.?!]*                   # B 括号之后到句末标点前
                    [.!?]                     # C 句末必须以 . ! ? 结束
                    """,
                    re.VERBOSE
                )
        quotes = sentence_pat.findall(related_text)
        # print(f"quotes: {quotes}")
        raw_claims, claims, sources_ids = [], [], []
        # 两种常见引用正则
        patterns = [
            r"[A-Z][A-Za-z]+[^()]*\(\d{4}\)",        # Author (2020)
            r"\([A-Z][A-Za-z]+[^\d]*,\s*\d{4}\)"    # (Author, 2020)
        ]

        for sent in quotes:
            ids_here = set()
            for pat in patterns:
                for m in re.finditer(pat, sent):
                    key = self._make_key(m.group(0))
                    k_list = key.split("_")
                    if len(k_list) == 2:
                        author = k_list[0]
                        year = k_list[1]
                    if author in key2idx.keys():
                        ids_here.add(key2idx[key] + 1)
            if ids_here:
                raw_claims.append(sent)
                # 去掉所有圆括号内容，剩余 claim
                claims.append(re.sub(r"\([^)]+\)", "", sent).strip())
                sources_ids.append(list(ids_here))
        return raw_claims, claims, sources_ids
    
    def _harvest_claims_with_cite_ids(self, related_text: str, key2idx: dict[str, int],cite_info):
        """从 Related-Work 正文中抽取带 (Author, Year) 引用的句子以及映射关系。"""
        sentence_pat = re.compile(
                    r"""                      # 选项详解见下方
                    [^.!?]*?                  # A 句首 → 第一个括号前，允许出现任意非句号/问号/感叹号字符（非贪婪）
                    (?:                       #   ┐ 两种主体引用格式
                        [A-Z][A-Za-z.&\s]*?   #   │ 1) 作者名字段，可含空格、and、&、et al.
                        \(\d{4}\)             #   │    再跟 “(YYYY)”
                    |                       #   ├─ 或
                        \(\s*                 #   │ 2) 直接以“(Author,”开头
                            [A-Z][A-Za-z]+    #   │    作者姓
                            [^,]*             #   │    括号内逗号前允许额外内容
                            ,\s*\d{4}\s*      #   │    “, YYYY”
                        \)                    #   │
                    )                         #   ┘
                    [^.?!]*                   # B 括号之后到句末标点前
                    [.!?]                     # C 句末必须以 . ! ? 结束
                    """,
                    re.VERBOSE
                )
        quotes = sentence_pat.findall(related_text)
        # print(f"quotes: {quotes}")
        raw_claims, claims, sources_ids = [], [], []
        # 两种常见引用正则
        patterns = [
            r"[A-Z][A-Za-z]+[^()]*\(\d{4}\)",        # Author (2020)
            r"\([A-Z][A-Za-z]+[^\d]*,\s*\d{4}\)"    # (Author, 2020)
        ]

        for sent in quotes:
            ids_here = set()
            # for cite in cite_info:
            #     if cite['citation_text'] in sent:
            #         ids_here.add(cite['paper_id'])
            for pat in patterns:
                for m in re.finditer(pat, sent):
                    key = self._make_key(m.group(0))
                    k_list = key.split("_")
                    if len(k_list) == 2:
                        author = k_list[0]
                        year = k_list[1]
                    if author in key2idx.keys():
                        ids_here.add(key2idx[key] + 1)
            if ids_here:
                raw_claims.append(sent)
                # 去掉所有圆括号内容，剩余 claim
                claims.append(re.sub(r"\([^)]+\)", "", sent).strip())
                sources_ids.append(list(ids_here))
        return raw_claims, claims, sources_ids
    
    
    def citation_quality_cite_ids(self, related_json: dict):
        related_works_with_citations = related_json.get("related_works", "")
        related_works_with_citations = related_works_with_citations.replace("```json", "").replace("```", "")
        related_works_with_citations = json_repair.loads(related_works_with_citations)
        related_text = related_works_with_citations['related_work']
        cite_info = related_works_with_citations['cite_ids']
        citations = related_json.get("citations", [])
        selected_papers = related_json.get("selected_papers", [])
        
        raw_claims,claims = self._get_quote(related_text)
        for raw_claim,cite_item in zip(raw_claims,cite_info):
            cite_item['raw_claim'] = raw_claim
            
        pair_scores = [[0] for i in range(len(claims))]
        for idx,cite_item in enumerate(cite_info):
            for paper in selected_papers:
                if str(paper['cite_ids'][0]) == str(cite_item['paper_id']):
                    paper_content = paper['text']
                    cite_info[idx]['paper_content'] = "\n".join(paper_content)
    
        with open('/home/liujian/project/2025-07/A2R-code-reproduction/src/evaluation/cite_info.json', 'w') as f:
            json.dump(cite_info, f)
        total_paper_num = len(cite_info)
        raw_claims = [
    "Early efforts in this domain, such as Retrieval-Augmented Generation (RAG), augment LLMs with external knowledge retrieval to mitigate factual inaccuracies (Asai et al., 2023)",
    "However, these approaches often suffer from coarse-grained attributions, pointing to entire documents or paragraphs, which burdens users with extensive verification (Slobodkin et al., 2024)",
    "However, these approaches often suffer from coarse-grained attributions, pointing to entire documents or paragraphs, which burdens users with extensive verification (Xia et al., 2024)",
    "To address this, methods like 'Attribute First, then Generate' prioritize concise, fine-grained attributions by breaking down the generation process into content selection, sentence planning, and sequential sentence generation (Slobodkin et al., 2024).",
    "Similarly, ReClaim focuses on interleaved reference-claim generation to provide sentence-level citations, significantly improving verifiability (Xia et al., 2024).",
    "Bohnet et al. (2022) formulated Attributed Question Answering (QA) and proposed a reproducible evaluation framework, including metrics like AIS and AutoAIS, to measure attribution quality.",
    "This work provides crucial insights into how well current state-of-the-art methods perform on attribution and hints at how to build LLMs with better attribution capabilities (Bohnet et al., 2022)",
    "Li et al. (2023) offer a comprehensive survey of attribution mechanisms in LLMs, highlighting challenges such as ambiguous knowledge reservoirs and the need for comprehensive attribution, which underscores the motivation for more robust generation techniques.",
    "Furthermore, the issue of hallucination is tackled post-hoc by systems like RARR, which automatically finds attribution for existing LM outputs and post-edits them to fix unsupported content (Gao et al., 2022).",
    "Similarly, Citation-Enhanced Generation (CEG) for chatbots uses a post-hoc approach with retrieval and natural language inference to ensure all statements are supported by citations, even regenerating responses if necessary (Li et al., 2024).",
    " Rashkin et al. (2021) introduced the Attributable to Identified Sources (AIS) framework for assessing whether NLG output is supported by underlying sources, providing a common framework for measuring attribution.",
    "Huang et al. (2024) address the challenge of acquiring high-quality attribution data by proposing START, a self-improvement framework that iteratively enhances LLM attribution capabilities through self-constructed synthetic data and fine-grained preference signals.",
    "The difficulty of automatically evaluating attribution is further highlighted by AttributionBench, a comprehensive benchmark revealing that even fine-tuned state-of-the-art LLMs struggle with nuanced information and discrepancies between model access and human annotation (Li et al., 2024)",
    "Phukan et al. (2024) propose a novel method leveraging LLM hidden states for granular attribution in contextual QA, identifying verbatim copied segments and their sources without extensive retraining.",
    "Datasets like HAGRID, which is a human-LLM collaborative dataset, are being developed to foster the creation of generative information-seeking models with improved attribution capabilities (Kamalloo et al., 2023). ",
    "For instance, Chain-of-Thought (CoT) reasoning has been adapted to enhance attribution accuracy, focusing the LLM's reasoning process on generating attribution-centric outputs at various granularities (Berchansky et al., 2024)",
    "THOUGHTSCULPT, for example, employs MCTS to explore a search tree of potential solutions, allowing for iterative revision of intermediate outputs (Chi et al., 2024)",
    "The concept of process supervision and reward modeling, as seen in OmegaPRM, which uses MCTS to efficiently collect high-quality process supervision data for mathematical reasoning, further underscores the utility of MCTS and fine-grained feedback in guiding LLMs (Luo et al., 2024). ",
    "LLMRefine proposes an inference-time optimization method that uses a learned fine-grained feedback model to pinpoint defects and guide iterative refinement via simulated annealing (Xu et al., 2023)",
    "Similarly, Self-RAG enhances LLM quality and factuality through adaptive retrieval and self-reflection, using 'reflection tokens' to guide generation and critique (Asai et al., 2023)",
    "The challenge of semantic drift, where LLMs generate correct facts initially before drifting to incorrect ones, highlights the need for continuous monitoring and guidance during generation (Spataru et al., 2024).",
    "REC, a suite of fine-tuned LLM auto-evaluators, provides detailed explanations and verifiable citations for assessing generated text quality across multiple dimensions (Hsu et al., 2024).",
    "While some methods focus on cost-effective extrinsic refinement (Cai et al., 2024)",
    "The systematic exploration of task instruction and input configuration for citation text generation also provides valuable insights into refining attribution quality (Şahinuç et al., 2024)."
]
        for idx,raw_claim in enumerate(raw_claims):
            response = self.__get_pair_score_new(cite_info[idx]['paper_content'], raw_claim)
            cite_info[idx]['score'] = response
        # for idx,cite_item in enumerate(cite_info[:]):
        #     paper_content = cite_item.get('paper_content', '')
        #     raw_claim = cite_item.get('raw_claim', '')
        #     if paper_content == "" or raw_claim == "":
        #         continue
        #     response = self.__get_pair_score_new(paper_content, raw_claim)
        #     cite_item['score'] = response
        with open('/home/liujian/project/2025-07/A2R-code-reproduction/src/evaluation/cite_info.json', 'w') as f:
            json.dump(cite_info, f,indent=4)
        from evaluation.agents.judge import claim_precision, citation_precision, reference_precision, reference_coverage, citation_density, avg_citation_per_claim, print_result

        result = {
            "claim_precision": claim_precision(pair_scores)[0] / claim_precision(pair_scores)[1] if claim_precision(pair_scores)[1] else 0,
            "citation_precision": citation_precision(pair_scores)[0] / citation_precision(pair_scores)[1] if citation_precision(pair_scores)[1] else 0,
            "reference_precision": reference_precision(pair_scores, total_paper_num),
        }
        print_result(result)               
                    
                        
        
        
        
                        

    def citation_quality_author_year(self, related_json: dict):
        """使用 Judge 实例的 __get_pair_score 评估作者-年份引用格式质量。"""
        related_works_with_citations = related_json.get("related_works", "")
        related_works_with_citations = related_works_with_citations.replace("```json", "").replace("```", "")
        related_works_with_citations = json_repair.loads(related_works_with_citations)
        related_text = related_works_with_citations['related_work']
        cite_info = related_works_with_citations['cite_ids']
        
        citations = related_json.get("citations", [])
        selected_papers = related_json.get("selected_papers", [])
        
        key2idx, paper_infos = self._build_index_new(citations, cite_info, selected_papers)
        raw_claims, claims, sources_ids = self._harvest_claims_with_cite_ids(related_text, key2idx,cite_info)
        index_to_paper = {idx: paper_infos[idx]["content"] for idx in range(len(paper_infos))}
        pair_scores = [[0] * len(sources_ids[i]) for i in range(len(claims))]
        # 顺序计算，不使用多线程
        for i in range(len(claims)):
            for j, pid in enumerate(sources_ids[i]):
                citation_idx = pid - 1
                if citation_idx not in index_to_paper:
                    pair_scores[i][j] = -1
                    continue
                self.__get_pair_score(
                    index_to_paper[citation_idx],
                    claims[i],
                    pair_scores,
                    i,
                    j,
                    citation_idx,
                    raw_claims[i],
                )

        total_paper_num = len(paper_infos)
        from evaluation.agents.judge import claim_precision, citation_precision, reference_precision, reference_coverage, citation_density, avg_citation_per_claim, print_result

        result = {
            "claim_precision": claim_precision(pair_scores)[0] / claim_precision(pair_scores)[1] if claim_precision(pair_scores)[1] else 0,
            "citation_precision": citation_precision(pair_scores)[0] / citation_precision(pair_scores)[1] if citation_precision(pair_scores)[1] else 0,
            "reference_precision": reference_precision(pair_scores, total_paper_num),
            "reference_coverage": reference_coverage(claims, sources_ids, total_paper_num),
            "citation_density": citation_density(sources_ids, related_text),
            "avg_citation_per_claim": avg_citation_per_claim(claims, sources_ids),
        }
        print_result(result)
        return result 

# ===================== 评估指标函数 =====================

def claim_precision(pairs):
    total_claim_num = len(pairs)
    correct_claim_num = 0
    for row in pairs:
        if any(score != -1 for score in row):
            correct_claim_num += 1
    return correct_claim_num, total_claim_num


def citation_precision(pairs):
    total_citation_num = sum(len(row) for row in pairs)
    correct_citation_num = sum(1 for row in pairs for s in row if s != -1)
    return correct_citation_num, total_citation_num


def reference_precision(pairs, total_paper_num):
    reference_set = {s for row in pairs for s in row if s != -1}
    return len(reference_set) / total_paper_num if total_paper_num else 0


def reference_coverage(claims, sources_ids, total_paper_num):
    reference_set = {pid - 1 for lst in sources_ids for pid in lst}
    return len(reference_set) / total_paper_num if total_paper_num else 0


def count_sentences(text):
    sentences = re.split(r"[.!?\n]+(?:\s|\n|$)", text.strip())
    sentences = [s for s in sentences if s]
    return len(sentences)


def citation_density(sources_ids, survey):
    total_citation_num = sum(len(lst) for lst in sources_ids)
    total_sentence_num = count_sentences(survey)
    return total_citation_num / total_sentence_num if total_sentence_num else 0


def avg_citation_per_claim(claims, sources_ids):
    total_citation_num = sum(len(lst) for lst in sources_ids)
    return total_citation_num / len(claims) if claims else 0


def print_result(result_dict):
    print("########## Metric with Judgement ##########")
    print(f"Claim Precision: {result_dict['claim_precision']}")
    print(f"Citation Precision: {result_dict['citation_precision']}")
    print(f"Reference Precision: {result_dict['reference_precision']}")
    print(f"######## Metric without Judgement #########")
    print(f"Reference Coverage: {result_dict['reference_coverage']}")
    print(f"Citation Density: {result_dict['citation_density']}")
    print(f"Avg Citation per Claim: {result_dict['avg_citation_per_claim']}") 