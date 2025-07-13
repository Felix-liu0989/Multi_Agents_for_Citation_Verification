# flake8: noqa
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
         result = generator.generate_related_work(content, 20, 3, 0.0)
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
dir = "/home/liujian/project/2025-07/A2R-code-reproduction/citegeist"
os.makedirs(dir, exist_ok=True)
output = os.path.join(dir, "result_citegeist.jsonl")   
process_with_checkpoint(data, output)




# from citegeist import Generator
# import os
# import json
# generator = Generator(
#    llm_provider="gemini",  # Choice of: "azure" (OpenAI Studio), "anthropic", "gemini", "mistral", and "openai"
#    api_key=os.environ.get("OPENROUTER_API_KEY"), # Here, you will need to set the respective API key
#    model_name="google/gemini-2.5-flash-preview-05-20", # Choose the model that the provider supports
#    database_uri=os.environ.get("MILVUS_URI", ""),  # Set the path (local) / url (remote) for the Milvus DB connection
#    database_token=os.environ.get("MILVUS_TOKEN", ""),  # Optionally, also set the access token (you DON'T need to set this when using the locally hosted Milvus Database)
# )


# # Define input abstract and breadth (5-20), depth (1-5), and diversity (0.0-1.0) parameters.
# abstract =  "A Decision-Theoretic Approach to Natural Language Generation: We study the problem of generating an English sentence given an underlying probabilistic grammar, a world and a communicative goal. We model the generation problem as a Markov decision process with a suitably deﬁned reward function that reﬂects the communicative goal. We then use probabilistic planning to solve the MDP and generate a sentence that, with high probability, accomplishes the communicative goal. We show empirically that our approach can generate complex sentences with a speed that generally matches or surpasses the state of the art. Further, we show that our approach is anytime and can handle complex communicative goals, including negated goals. "
# # "Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling: Despite their outstanding capabilities, large language models (LLMs) are prone to hallucination and producing factually incorrect information. This challenge has spurred efforts in attributed text generation, which prompts LLMs to generate content with supporting evidence. In this paper, we propose a novel framework, called Think&Cite, and formulate attributed text generation as a multi-step reasoning problem integrated with search. Specifically, we propose Self-Guided Monte Carlo Tree Search (SG-MCTS), which capitalizes on the self-reflection capability of LLMs to reflect on the intermediate states of MCTS for guiding the tree expansion process. To provide reliable and comprehensive feedback, we introduce Progress Reward Models to measure the progress of tree search from the root to the current state from two aspects, i.e., generation and attribution progress. We conduct extensive experiments on three datasets and the results show that our approach significantly outperforms baseline approaches."
# breadth = 30
# depth = 3
# diversity = 0.0
# result = generator.generate_related_work(abstract, breadth, depth, diversity)

# print(result)
# print(f"Related Work:\n{result['related_works']}")
# print(f"Citations:\n{result['citations']}")
# print(f"Number of selected papers: {len(result['selected_papers'])}")
# print(f"{result['selected_papers']} Selected Papers:\n")

# with open("/home/liujian/project/2025-07/A2R-code-reproduction/results/result_citegeist.txt", "w") as f:
#    f.write(f"Related Work:\n{result['related_works']}")
#    f.write(f"Citations:\n{result['citations']}")
with open("/home/liujian/project/2025-07/A2R-code-reproduction/results/result_citegeist_0713.json", "w") as f:
   json.dump(result, f, ensure_ascii=False, indent=4)