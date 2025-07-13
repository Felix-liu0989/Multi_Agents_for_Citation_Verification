import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from .prompts import process_data_for_related_work_prompt
import json
import transformers
import torch
from .llm_clients import create_client
from .llm_clients.gemini_client import GeminiClient
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()
import json_repair
import jsonlines
import re
import json
import jsonlines
import os
from tqdm import tqdm

def load_processed_ids(output_file):
    """从输出文件中加载已处理的ID"""
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            with jsonlines.open(output_file, 'r') as reader:
                for item in reader:
                    processed_ids.add(item['id'])
        except Exception as e:
            print(f"读取已处理文件时出错: {e}")
    return processed_ids


def process_with_checkpoint(data, output_file):
    """带断点续跑的处理函数"""
    
    # 加载已处理的ID
    processed_ids = load_processed_ids(output_file)
    print(f"已处理 {len(processed_ids)} 个项目，从第 {len(processed_ids)} 个开始继续...")
    
    # 过滤出未处理的数据
    remaining_data = []
    for id, item in enumerate(data):
        paper_id = item['path'].split("v1.pdf")[0].split("pdfs/")[-1]
        if paper_id not in processed_ids and id not in processed_ids:
            remaining_data.append((id, item))
    
    print(f"剩余 {len(remaining_data)} 个项目需要处理")
    account = {
        "a whole related work": 0,
        "2 subsections": 0,
        "3 subsection": 0,
        "4 subsection": 0,
        "5 subsections": 0,
        "other": 0,
    }
    # 处理剩余数据
    for id, item in tqdm(remaining_data):
        try:
            related_work = item["text"]
            prompt = process_data_for_related_work_prompt(related_work)
            client = GeminiClient(api_key=os.environ.get("OPENROUTER_API_KEY"), 
                                model_name="google/gemini-2.5-flash-preview-05-20")
            result = client.get_completion(prompt)
            result = result.replace("```json", "").replace("```", "")
            result = json_repair.loads(result)
            if isinstance(result, str) and len(result) != 0:
                account["a whole related work"] += 1
            elif isinstance(result, dict) and len(result.keys()) == 2:
                account["2 subsections"] += 1
            elif isinstance(result, dict) and len(result.keys()) == 3:
                account["3 subsection"] += 1
            elif isinstance(result, dict) and len(result.keys()) == 4:
                account["4 subsection"] += 1
            elif isinstance(result, dict) and len(result.keys()) == 5:
                account["5 subsections"] += 1
            else:
                account["other"] += 1
            # 添加必要字段
            item["related_work"] = result
            item["paper_id"] = item['path'].split("v1.pdf")[0].split("pdfs/")[-1]
            item["id"] = id
            
            # 立即保存
            with jsonlines.open(output_file, "a") as writer:
                writer.write(item)
            
            print(f"已完成第 {id} 项")
            
        except Exception as e:
            print(f"处理第 {id} 项时出错: {e}")
            # 记录错误但继续处理
            continue
    return account

def jsonl2json(jsonl_file, json_file):
    with jsonlines.open(jsonl_file, "r") as reader:
        data = [item for item in reader]
    with open(json_file, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def process_data_by_subsection_nums(data, output_file):
    for item in data:
        if isinstance(item["related_work"], str) and len(item["related_work"]) != 0:
            item["subsection_nums"] = 0
        elif isinstance(item["related_work"], dict):
            item["subsection_nums"] = len(item["related_work"].keys())
        elif isinstance(item["related_work"], list):
            item["subsection_nums"] = len(item["related_work"])
        else:
            item["subsection_nums"] = -1
    data = [item for item in data if item["subsection_nums"] != -1]
    ## 按照subsection_nums排序
    data = sorted(data, key=lambda x: x["subsection_nums"], reverse=True)
    with open(output_file, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_data(data):
    data_2 = []
    data_3 = [] 
    data_4 = []
    data_5 = []
    for item in data:
        if item["subsection_nums"] == 5:
            data_3.append(item)
            
    with open("/home/liujian/project/2025-07/A2R-code-reproduction/datasets/scholar_copilot_eval_data_1k_related_work_result_5_sections.json", "w") as f:
        json.dump(data_3, f, ensure_ascii=False, indent=4)
    
def filter_data(data):
    ## 2401.11886
    regex_24 = r"24\d{2}"
    regex_23 = r"23\d{2}"
    regex_22 = r"22\d{2}"
    regex_21 = r"21\d{2}"
    regex_20 = r"20\d{2}"
    data_24 = []
    data_23 = []
    data_22 = []
    data_21 = []
    data_20 = []
    for item in data:
        id = item["paper_id"].split(".")[0]
        if re.match(regex_24, id):
            data_24.append(item)
        elif re.match(regex_23, id):
            data_23.append(item)
        elif re.match(regex_22, id):
            data_22.append(item)
        elif re.match(regex_21, id):
            data_21.append(item)
        elif re.match(regex_20, id):
            data_20.append(item)
    return data_24, data_23, data_22, data_21, data_20





if __name__ == "__main__":
    # with open("/home/liujian/project/2025-07/A2R-code-reproduction/datasets/scholar_copilot_eval_data_1k_related_work.json", "r") as f:
    #     data = json.load(f)
    # output_file = "/home/liujian/project/2025-07/A2R-code-reproduction/datasets/scholar_copilot_eval_data_1k_related_work_result.jsonl"
    # account = process_with_checkpoint(data[:200], output_file)
    # print(account)
    
    # jsonl_file = "/home/liujian/project/2025-07/A2R-code-reproduction/datasets/scholar_copilot_eval_data_1k_related_work_result.jsonl"
    # json_file = "/home/liujian/project/2025-07/A2R-code-reproduction/datasets/scholar_copilot_eval_data_1k_related_work_result.json"
    # jsonl2json(jsonl_file, json_file)
    
    # data = json.load(open(json_file, "r"))
    # process_data_by_subsection_nums(data, json_file)
    # json_file = "/home/liujian/project/2025-07/A2R-code-reproduction/datasets/scholar_copilot_eval_data_1k_related_work_result.json"
    json_file = "/home/liujian/project/2025-07/A2R-code-reproduction/datasets/scholar_copilot_eval_data_1k_related_work_result_5_sections.json"
    data = json.load(open(json_file, "r"))
    save_data(data)
    data_24, data_23, data_22, data_21, data_20 = filter_data(data)
    print(len(data_24), len(data_23), len(data_22), len(data_21), len(data_20))
    # data_4_sections = data_24[:] + data_23[:] + data_22[:] + data_21[:] + data_20[:]
    with open("/home/liujian/project/2025-07/A2R-code-reproduction/datasets/scholar_copilot_eval_data_1k_related_work_result_5_sections_8_3_0_2_2.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    # with open("/home/liujian/project/2025-07/A2R-code-reproduction/datasets/scholar_copilot_eval_data_1k_related_work_result_2_sections_15_5_1_5_5.json", "r") as f:
    #     data = json.load(f)
    # for item in data:
    #     if item["paper_id"] == "2401.11988":
    #         data.remove(item)
    #         print(item["paper_id"])
    #     elif item["paper_id"] == "2109.13995":
    #         data.remove(item)
    #         print(item["paper_id"])
    # data.append(data_20[6])
    # print(len(data))
    # with open("/home/liujian/project/2025-07/A2R-code-reproduction/datasets/scholar_copilot_eval_data_1k_related_work_result_2_sections_15_5_5_5.json", "w") as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4)
    
            
    
            
    
   