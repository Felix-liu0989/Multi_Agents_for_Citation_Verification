from citegeist import Generator
import os
import json,jsonlines
import time
from citegeist.utils.infer import load_processed_ids

def process_with_checkpoint(data, output_file):
   # 加载已处理的ID
   processed_ids = load_processed_ids(output_file)
   print(f"已处理 {len(processed_ids)} 个项目，从第 {len(processed_ids)} 个开始继续...")

   # 过滤出未处理的数据
   remaining_data = []
   for id, item in enumerate(data):
      if id not in processed_ids:
         remaining_data.append(id)
   
   print(f"剩余 {len(remaining_data)} 个项目需要处理")
   for id in remaining_data:
      item = data[id]
      title = item["title"]
      abstract = item["abstract"]
      content = f"Title: {title}\nAbstract: {abstract}"
      try:
         result = generator.get_arxiv_dim_test(content, 20, 3, 0.0)
         related_work = result["related_works"]
         print(related_work)
         item["related_work"] = result
         with jsonlines.open(output_file, "a") as writer:
            writer.write(item)
         print(f"已完成第 {id} 项")
      except Exception as e:
         print(e)
         continue

start_time = time.time()
generator = Generator(
   llm_provider="gemini",  # Choice of: "azure" (OpenAI Studio), "anthropic", "gemini", "mistral", and "openai"
   api_key=os.environ.get("OPENROUTER_API_KEY"), # Here, you will need to set the respective API key
   model_name="google/gemini-2.5-flash-preview-05-20", # Choose the model that the provider supports
   database_uri=os.environ.get("MILVUS_URI", ""),  # Set the path (local) / url (remote) for the Milvus DB connection
   database_token=os.environ.get("MILVUS_TOKEN", ""),  # Optionally, also set the access token (you DON'T need to set this when using the locally hosted Milvus Database)
)

path = "/home/liujian/project/2025-07/A2R-code-reproduction/datasets/arxiv_73/scholar_copilot_eval_data_1k_related_work_result_2_sections_15_5_5_5.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
dir = "/home/liujian/project/2025-07/A2R-code-reproduction/ourworkflow"
os.makedirs(dir, exist_ok=True)
output = os.path.join(dir, "result_all.jsonl")   
process_with_checkpoint(data, output)

# test_path = "/home/liujian/project/2025-07/A2R-code-reproduction/datasets/scholar_copilot_eval_data_1k.json"
# with open(test_path, "r", encoding="utf-8") as f:
#     data = json.load(f)


# for item in data[0:5:2]:
#    title = item["title"]
#    abstract = item["abstract"]
#    print(title)
#    content = f"Title: {title}\nAbstract: {abstract}"
#    try:  
#       result = generator.get_arxiv_dim_various_topics(content, 20, 3, 0.0)
#    except Exception as e:
#       print(e)
#       continue
#    related_work = result["related_works"]
#    print(related_work)

 
   


# abstract = "Multi-Agent Collaborative Framework for RAG-Enhanced Hierarchical Literature Synthesis: Crafting a high-quality related work section is crucial for academic papers, demanding deep comparative analysis across references, logical organization, and clear articulation of the paper's novelty. However, existing automated generation methods often fall short, producing isolated summaries lacking comparative insights, coherent structure, or robust evaluation beyond model scores. To address these limitations, we propose MACG: a novel Multi-Agent Collaborative Generation framework specifically designed for producing comparative and structured related work sections. MACG employs five specialized agents working iteratively: a Summarizer generates concise, relevance-focused literature summaries; a Structurer dynamically identifies and groups research themes, constructing a thematic Directed Acyclic Graph (DAG) to organize the literature landscape; a Comparator performs in-depth comparative analysis within and across these grouped themes on critical dimensions (e.g., methodology, datasets); an Integrator synthesizes the outputs into a logically coherent narrative, emphasizing the target paper's innovation; and a FactCheck agent continuously verifies fidelity to source material. Crucially, MACG leverages grouped research themes as the backbone for structuring the section and facilitating focused comparative analysis. Our framework supports iterative refinement based on prompts or fact-checking feedback. Evaluations demonstrate that MACG significantly enhances the logical flow, comparative depth, thematic organization, and factual accuracy of generated related work sections compared to existing approaches, providing a robust solution for automating this critical academic writing task."

# abstract = "Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling: Despite their outstanding capabilities, large language models (LLMs) are prone to hallucination and producing factually incorrect information. This challenge has spurred efforts in attributed text generation, which prompts LLMs to generate content with supporting evidence. In this paper, we propose a novel framework, called Think&Cite, and formulate attributed text generation as a multi-step reasoning problem integrated with search. Specifically, we propose Self-Guided Monte Carlo Tree Search (SG-MCTS), which capitalizes on the self-reflection capability of LLMs to reflect on the intermediate states of MCTS for guiding the tree expansion process. To provide reliable and comprehensive feedback, we introduce Progress Reward Models to measure the progress of tree search from the root to the current state from two aspects, i.e., generation and attribution progress. We conduct extensive experiments on three datasets and the results show that our approach significantly outperforms baseline approaches."

# '''
# Reffexion: Language Agents with Verbal Reinforcement Learning

# Large language models (LLMs) have been increasingly used to interact with external environments (e.g., games, compilers, APIs) as goal-driven agents. However, it remains challenging for these language agents to quickly and efficiently learn from trial-and-error, as traditional reinforcement learning methods require extensive training samples and expensive model fine-tuning.  

# We propose Reflexion, a novel framework to reinforce language agents not by updating weights, but instead through linguistic feedback. Specifically, Reflexion agents verbally reflect on task feedback signals, then maintain their own reflective text in an episodic memory buffer to improve decision-making in subsequent trials. Reflexion is flexible enough to incorporate various types (scalar values or free-form language) and sources (external or internally simulated) of feedback signals, and achieves significant improvements over baseline agents across diverse tasks (sequential decision-making, coding, language reasoning).  

# For example, Reflexion achieves a 91% pass@1 accuracy on the HumanEval coding benchmark, surpassing the previous state-of-the-art GPT-4, which achieves 80%. We also conduct ablation and analysis studies using different feedback signals, feedback incorporation methods, and agent types, providing insights into how they affect performance. All code, demos, and datasets are available at https://github.com/noahshinn024/reflexion.
# '''



# print(roots)
# print(dags)
# print(visualizer)
# breadth = 5
# depth = 2
# diversity = 0.0
# result = generator.get_arxiv_dim_test(abstract, breadth, depth, diversity)
# print(result)

# result = generator.get_arxiv_dim_test(abstract, 20, 3, 0.0)
# print(result)
# with open(f"/home/liujian/project/2025-07/A2R-code-reproduction/results/MACRH_related_work_with_citations_0710.json", "w", encoding="utf-8") as f:
#    json.dump(result,f,ensure_ascii=False,indent=4)
   
   
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")