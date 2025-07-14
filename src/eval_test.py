# {"related_works": related_works_section, "citations": filtered_citations, "selected_papers": relevant_pages}
from evaluation.agents.judge import Judge
import json,jsonlines
from pathlib import Path
import re
from typing import List, Tuple
import os
from citegeist.utils.llm_clients.gemini_client import GeminiClient
from citegeist.utils.llm_clients.deepseek_client import DeepSeekClient
from citegeist.utils.infer import load_processed_ids
import re
import time
import json_repair

"""
- 30个related work最终储存在result_all_with_cite_2.jsonl中
- json2json转换
- 抽取related_work中的引用信息，并和cited_ids中的信息进行匹配，每一个quote对应一个或多个溯源信息，并去重
- 对每一个quote，使用judge进行判断，判断是否正确
- 统计正确和错误的数量
- TODO List
    - 1. 给错误归类
    - 2. 判断每个错误属于哪一类
    - 3. 对每一类错误，给出改进建议
    
- 归类
    - 1. 直接矛盾 (Direct Contradiction)
        论断中的信息与来源文本的明确陈述直接相反。
        示例： 论断说某个方法在“数据极少的情况下表现不佳”，但来源明确表示该方法旨在“在有限数据下增强能力”。或者论断说某个方法“没有解决多样性问题”，但来源明确指出其目标就是“保留策略多样性”。
    - 2. 信息缺失/无根据 (Information Not Present / Unsubstantiated)
        论断中包含来源文本完全没有提及或支持的信息。
        示例： 论断提到了一个方法（如 RPO 或 HDFlow），但提供的来源只讨论了另一个方法。或者论断做出了一个普遍性声明（如“有监督微调需要大量标注数据”），但提供的摘要中没有相应支持。
    - 3. 错误表述/措辞不精确 (Misrepresentation / Imprecise Wording)
        论断对来源中的信息进行了重新阐述，但其措辞不够精确，或偏离了来源原文的含义和侧重点。
        示例： 论断说奖励最大化强化学习“过度利用高奖励行动”，而来源的描述是其在“探索状态空间和保持多样化假设”方面存在局限性。或者论断说某个方法“仍受限于预定义动作”，但来源强调的是其“与静态预定义动作形成对比”（暗示超越了这种限制）。
    - 4. 错误归属 (Incorrect Attribution)
        论断将来源中描述的某项贡献或特性错误地归因给了不同的实体或论文。
        示例： 来源明确指出某项演示是由“Proof Flow”完成的，但论断却将其归因于“FoR”。
    
- 改进建议
    ...
"""


client = GeminiClient(
    api_key = os.environ.get("OPENROUTER_API_KEY", ""),
    model_name = "google/gemini-2.5-flash-preview-05-20"
)
# 实现cite标签的添加
with open("/home/liujian/project/2025-07/A2R-code-reproduction/ourworkflow/result_all.json", "r", encoding="utf-8") as f:
    data = json.load(f)
output_file = "/home/liujian/project/2025-07/A2R-code-reproduction/ourworkflow/result_all_with_cite_2_quotes_split_by_gemini.jsonl"


# 指标计算
def calculate_meaningful_metrics(data):
    """计算更有意义的评估指标"""
    
    total_quotes = 0
    gemini_correct = 0
    deepseek_correct = 0
    both_correct = 0
    both_wrong = 0
    gemini_only_correct = 0
    deepseek_only_correct = 0
    
    for item in data:
        yes_gemini_ids = set(yes["id"] for yes in item["yes_gemini"])
        yes_deepseek_ids = set(yes["id"] for yes in item["yes_deepseek"])
        no_gemini_ids = set(no["id"] for no in item["no_gemini"])
        no_deepseek_ids = set(no["id"] for no in item["no_deepseek"])
        
        all_quote_ids = yes_gemini_ids | yes_deepseek_ids | no_gemini_ids | no_deepseek_ids
        total_quotes += len(all_quote_ids)
        
        # 计算各种情况
        gemini_correct += len(yes_gemini_ids)
        deepseek_correct += len(yes_deepseek_ids)
        both_correct += len(yes_gemini_ids & yes_deepseek_ids)
        both_wrong += len(no_gemini_ids & no_deepseek_ids)
        gemini_only_correct += len(yes_gemini_ids - yes_deepseek_ids)
        deepseek_only_correct += len(yes_deepseek_ids - yes_gemini_ids)
    
    print("=== 更有意义的评估指标 ===")
    print(f"总引用数: {total_quotes}")
    print(f"Gemini独立准确率: {gemini_correct / total_quotes:.3f}")
    print(f"DeepSeek独立准确率: {deepseek_correct / total_quotes:.3f}")
    print(f"两模型都正确的比例: {both_correct / total_quotes:.3f}")
    print(f"两模型都错误的比例: {both_wrong / total_quotes:.3f}")
    print(f"只有Gemini正确的比例: {gemini_only_correct / total_quotes:.3f}")
    print(f"只有DeepSeek正确的比例: {deepseek_only_correct / total_quotes:.3f}")
    print(f"模型一致性: {(both_correct + both_wrong) / total_quotes:.3f}")
    
    return {
        "total_quotes": total_quotes,
        "gemini_accuracy": gemini_correct / total_quotes,
        "deepseek_accuracy": deepseek_correct / total_quotes,
        "both_correct_rate": both_correct / total_quotes,
        "both_wrong_rate": both_wrong / total_quotes,
        "gemini_only_correct_rate": gemini_only_correct / total_quotes,
        "deepseek_only_correct_rate": deepseek_only_correct / total_quotes,
        "model_agreement": (both_correct + both_wrong) / total_quotes
    }


def process_with_checkpoint_add_cite(data, output_file):
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
        related_work = item["related_work"]["related_works"]
        try:
            prompt = f"""
            The following is a related work section. Please add <cite></cite> tags to sentences with citations based on the information in the related work.
            For example:
            <cite> The neural radiance field (NeRF) (Mildenhall et al., 2021) is one of the advanced methodologies that
            represent scenes using implicit neural rendering with MLP.</cite>
            Here is the related work:
            {related_work}
            """
            result = client.get_completion(prompt)
            print(result)
            item["related_work"]["related_works"] = result
            with jsonlines.open(output_file, "a") as writer:
                writer.write(item)
            print(f"已完成第 {id} 项")
        except Exception as e:
            print(e)
            continue
        
def process_with_checkpoint_extract_cited_sentences(data, output_file):
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
        related_work = item["related_work"]["related_works"]
        try:
            prompt = f"""
            The following is a related work section. Please extract the sentences with citations (<cite>...</cite>) based on the information in the related work.
            For example:
            The neural radiance field (NeRF) <cite>(Mildenhall et al., 2021)</cite> is one of the advanced methodologies that
            represent scenes using implicit neural rendering with MLP.

            Here is the related work:
            {related_work}
            Please output the whole sentences with citations, not just the citations.
            Please output the result in the following List format:
            [
                "The neural radiance field (NeRF) <cite>(Mildenhall et al., 2021)</cite> is one of the advanced methodologies that
                represent scenes using implicit neural rendering with MLP.",
                ...
            ]
            """
            result = client.get_completion(prompt)
            result = result.replace("```json", "").replace("```", "")
            result = json_repair.loads(result)
            for i,quote in enumerate(result,1):
                print(f"{i}. {quote}")
                print("-"*100)
            item["related_work"]["quotes"] = result
            with jsonlines.open(output_file, "a") as writer:
                writer.write(item)
            print(f"已完成第 {id} 项")
        except Exception as e:
            print(e)
            continue

# process_with_checkpoint_extract_cited_sentences(data, output_file)

def jsonl2json(jsonl_file, json_file):
    with jsonlines.open(jsonl_file, "r") as reader:
        data = [line for line in reader]
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
# jsonl2json("/home/liujian/project/2025-07/A2R-code-reproduction/ourworkflow/result_all_07_14.jsonl", "/home/liujian/project/2025-07/A2R-code-reproduction/ourworkflow/result_all_with_cite_07_14.json")


def extract_cited_sentences(text):

    # 匹配所有包含 <cite>...</cite> 的完整句子
    pattern = r'([^<]*<cite>.*?</cite>[^<]*)'
    sentences_with_cite = re.findall(pattern, text)
    
    return sentences_with_cite

def process_with_checkpoint_eval(input_file, output_file):
    judge_gemini = Judge(model="google/gemini-2.5-flash-preview-05-20")
    judge_deepseek = Judge(model="deepseek-chat")
    with open(input_file,"r",encoding="utf-8") as f:
        data = json.load(f)
        
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
        related_work = item["related_work"]["related_works"]
        quotes = item["related_work"]["quotes"]
        # quotes = extract_cited_sentences(related_work)
        ids = item["related_work"]["cited_ids"]
        selected_papers = item["related_work"]["selected_papers"]
        # 添加cited_text和summary
        for cited_id in ids:
            for selected_paper in selected_papers:
                if int(cited_id["paper_id"]) == int(selected_paper["cite_ids"][0]):
                    cited_id["cited_text"] = selected_paper["text"]
                    cited_id["summary"] = selected_paper["summary"]
                    
        # 为每一个quote添加上对应的一个或者多个source            
        quotes_with_cition_info = {quote: [] for quote in quotes}
        for quote in quotes:
            for cited_id in ids:
                c_text = cited_id["citation_text"]
                if c_text in quote or c_text.split(".")[0] in quote:
                    quotes_with_cition_info[quote].append(" cited_text: " + cited_id["citation_text"] + " summary: " + cited_id["summary"])
        try:           
            yes_gemini = []
            no_gemini = []
            yes_deepseek = []
            no_deepseek = []
            for i,quote in enumerate(quotes_with_cition_info.keys()):
                print(quote)
                q = list(set(quotes_with_cition_info[quote]))
                source = "".join(q)
                # print("claim:")
                # print(quote)
                # print("source:")
                # print(source)
                
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
                    
                quotes_with_cition_info[quote] = q
                
            item["quotes_with_cition_info"] = quotes_with_cition_info
            item["yes_gemini"] = yes_gemini
            item["no_gemini"] = no_gemini
            item["yes_deepseek"] = yes_deepseek
            item["no_deepseek"] = no_deepseek
            with jsonlines.open(output_file, "a") as writer:
                writer.write(item)
            print(f"已完成第 {id} 项")
        except Exception as e:
            print(e)
            continue
        
        
def process_with_checkpoint_refine_quotes(input_file, output_file):
    with open(input_file,"r",encoding="utf-8") as f:
        data = json.load(f)
    for id,item in enumerate(data):
        yes_gemini_ids = []
        yes_deepseek_ids = []
        no_gemini_ids = []
        no_deepseek_ids = []
        yes_gemini = item["yes_gemini"]
        yes_deepseek = item["yes_deepseek"]
        no_gemini = item["no_gemini"]
        no_deepseek = item["no_deepseek"]
        for yes in yes_gemini:
            yes_gemini_ids.append(yes["id"])
        for yes in yes_deepseek:
            yes_deepseek_ids.append(yes["id"])
        for no in no_gemini:
            no_gemini_ids.append(no["id"])
        for no in no_deepseek:
            no_deepseek_ids.append(no["id"])
        
        ## 对yes_gemini和yes_deepseek进行合并，并添加yes_ids
        yes_ids = yes_gemini_ids + yes_deepseek_ids
        yes_ids = list(set(yes_ids))
        item["yes_ids"] = yes_ids
        no_ids =  no_gemini_ids + no_deepseek_ids
        no_ids = list(set(no_ids))
        item["no_ids"] = no_ids
        correct = len(yes_ids)
        incorrect = len(no_ids)
        item["citation_accuracy"] = correct / (correct + incorrect)
        item["citation_accuracy_gemini"] = len(yes_gemini_ids) / (len(yes_gemini_ids) + len(no_gemini_ids))
        item["citation_accuracy_deepseek"] = len(yes_deepseek_ids) / (len(yes_deepseek_ids) + len(no_deepseek_ids))
        item["citation_accuracy_gemini_yes"] = len(yes_gemini_ids) / len(yes_ids)
        item["citation_accuracy_deepseek_yes"] = len(yes_deepseek_ids) / len(yes_ids)
        item["citation_accuracy_gemini_no"] = len(no_gemini_ids) / len(no_ids)
        item["citation_accuracy_deepseek_no"] = len(no_deepseek_ids) / len(no_ids)
        
        
        with jsonlines.open(output_file, "a") as writer:
            writer.write(item)
        # print(f"已完成第 {id} 项")
    # 计算所有项目的平均值
    total_correct = sum(item["citation_accuracy"] for item in data)
    total_accuracy = total_correct / len(data)
    print(f"所有项目的平均准确率: {total_accuracy}")
    
    meaningful_metrics = calculate_meaningful_metrics(data)
    print(meaningful_metrics)

            
start_time = time.time()
jsonl2json("/home/liujian/project/2025-07/A2R-code-reproduction/ourworkflow/result_all_with_cite_2_quotes_split_by_gemini_eval.jsonl", "/home/liujian/project/2025-07/A2R-code-reproduction/ourworkflow/result_all_with_cite_2_quotes_split_by_gemini_eval.json")
# process_with_checkpoint_extract_cited_sentences(data, output_file)
# process_with_checkpoint_eval("/home/liujian/project/2025-07/A2R-code-reproduction/ourworkflow/result_all_with_cite_2_quotes_split_by_gemini.json", "/home/liujian/project/2025-07/A2R-code-reproduction/ourworkflow/result_all_with_cite_2_quotes_split_by_gemini_eval.jsonl")
process_with_checkpoint_refine_quotes("/home/liujian/project/2025-07/A2R-code-reproduction/ourworkflow/result_all_with_cite_2_quotes_split_by_gemini_eval.json", "/home/liujian/project/2025-07/A2R-code-reproduction/ourworkflow/result_all_with_cite_2_quotes_split_by_gemini_eval_refine.jsonl")

# 使用更有意义的指标进行评估
# with open("/home/liujian/project/2025-07/A2R-code-reproduction/ourworkflow/result_all_with_cite_2_quotes_split_by_gemini_eval.json", "r", encoding="utf-8") as f:
#     eval_data = json.load(f)
# meaningful_metrics = calculate_meaningful_metrics(eval_data)

end_time = time.time()
print(f"处理时间: {end_time - start_time} 秒")




        

