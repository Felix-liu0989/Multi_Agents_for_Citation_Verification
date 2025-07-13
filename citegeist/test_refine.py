from utils.prompts import (
    generate_related_work_revision_prompt,
    generate_related_work_comparative_prompt,
    genrate_original_related_work_feedback_prompt
)
import json
from utils.llm_clients.deepseek_client import DeepSeekClient
import os
from dotenv import load_dotenv

load_dotenv()

client = DeepSeekClient(
    api_key = os.environ.get("DEEPSEEK_API_KEY", ""),
    model_name = "deepseek-chat"
)

with open("/home/liujian/project/2025-07/A2R-code-reproduction/results/related_work.txt","r") as f:
    related_work = f.read()


abstract =  "Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling: Despite their outstanding capabilities, large language models (LLMs) are prone to hallucination and producing factually incorrect information. This challenge has spurred efforts in attributed text generation, which prompts LLMs to generate content with supporting evidence. In this paper, we propose a novel framework, called Think&Cite, and formulate attributed text generation as a multi-step reasoning problem integrated with search. Specifically, we propose Self-Guided Monte Carlo Tree Search (SG-MCTS), which capitalizes on the self-reflection capability of LLMs to reflect on the intermediate states of MCTS for guiding the tree expansion process. To provide reliable and comprehensive feedback, we introduce Progress Reward Models to measure the progress of tree search from the root to the current state from two aspects, i.e., generation and attribution progress. We conduct extensive experiments on three datasets and the results show that our approach significantly outperforms baseline approaches."

# prompt = genrate_original_related_work_feedback_prompt(
#     related_work=related_work
# )
# result = client.get_completion(prompt)
# print(result)

# prompt = generate_related_work_revision_prompt(
#     source_abstract=abstract,
#     related_work=related_work,
#     feedback=result,
#     dimensions=["Attributed Text Generation Tasks and Challenges","Tree Search Methodologies for Controlled Generation"]
# )

# result = client.get_completion(prompt)

# with open("/home/liujian/project/2025-07/A2R-code-reproduction/results/related_work_revision.txt","w") as f:
#     f.write(result)
# print(result)

dim1_path = "/home/liujian/project/2025-07/A2R-code-reproduction/results/Self-guided tree search for attributed text generation/grouped_dim_1.json"
dim2_path = "/home/liujian/project/2025-07/A2R-code-reproduction/results/Self-guided tree search for attributed text generation/grouped_dim_2_0705.json"

with open(dim1_path,"r") as f:
    dim1 = json.load(f)

with open(dim2_path,"r") as f:
    dim2 = json.load(f)
print(len(dim1),len(dim2))
    
def iters_generate_related_work_comparative_prompt(dim,abstract):
    related_work_summary = ""
    for paper in dim:
        title = paper["title"]
        summary = paper["summary"]
        citation = paper["citations"]
        prompt = generate_related_work_comparative_prompt(
            source_abstract=abstract,
            related_work_summary=related_work_summary,
            current_reference_abstract=summary,
            current_reference_title=title,
            current_reference_citation=citation
        )
        result = client.get_completion(prompt)
        related_work_summary += result + "\n"
    return related_work_summary
related_work_summary = iters_generate_related_work_comparative_prompt(dim2,abstract)
print(related_work_summary)

# for paper in dim1[:1]:
#     title = paper["title"]
#     summary = paper["summary"]
#     prompt = generate_related_work_comparative_prompt(
#         source_abstract=abstract,
#         related_work_summary="",
#         current_reference_abstract=summary
#     )
#     result = client.get_completion(prompt)
#     print(result)


