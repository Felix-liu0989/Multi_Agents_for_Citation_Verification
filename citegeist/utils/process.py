import fitz
import os
import re
import json
import arxiv
from tqdm import tqdm
# from citegeist.utils.prompts import process_data_for_related_work_prompt

def download_pdf(paper_ids:list[str], dirpath:str):
    pdf_infos = []
    for paper_id in tqdm(paper_ids):
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
        pdf_path = paper.download_pdf(dirpath=dirpath, filename=paper_id + ".pdf")
        text = pdf_to_text(pdf_path)
        pdf_infos.append({"id": paper_id, "text": text,"path": pdf_path})
    return pdf_infos



def pdf_to_text(pdf_path):
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def extract_after_related_work_regex(text):
    """使用正则表达式提取related work之后的内容"""
    
    # 匹配多种related work格式的正则表达式
    pattern = r'(?:^|\n)\s*(?:related\s+works?|literature\s+review|background)\s*:?\s*'
    
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    
    if match:
        # 从匹配位置之后开始提取
        start_pos = match.end()
        return text[start_pos:].strip()
    else:
        return None

def process_pdf(pdf_infos:list[dict]):
    for pdf_info in pdf_infos:
        text = pdf_info["text"]
        if "related work" in text.lower():
            text = extract_after_related_work_regex(text)
            if text is not None:
                pdf_info["text"] = "related work: " + text
        else:
            pdf_infos.remove(pdf_info)
    return pdf_infos

if __name__ == "__main__":
    # with open("/home/liujian/project/2025-07/A2R-code-reproduction/datasets/scholar_copilot_eval_data_1k.json","r") as f:
    #     data = json.load(f)
    # paper_ids = []
    # for item in data[:500]:
    #     paper_id = item["paper_id"]
    #     paper_id = paper_id + "v1" # 2004.12679v1
    #     paper_ids.append(paper_id)
    # pdf_infos = download_pdf(paper_ids, "/home/liujian/project/2025-07/A2R-code-reproduction/results/pdfs")
    # pdf_infos = process_pdf(pdf_infos)
    pdf_paths = "/home/liujian/project/2025-07/A2R-code-reproduction/results/pdfs"
    pdf_infos = []
    for pdf_path in os.listdir(pdf_paths):
        pdf_infos.append({"id": pdf_path.split(".")[0].replace("v1",""), "text": pdf_to_text(os.path.join(pdf_paths, pdf_path)), "path": os.path.join(pdf_paths, pdf_path)})
    pdf_infos = process_pdf(pdf_infos)
    with open("/home/liujian/project/2025-07/A2R-code-reproduction/datasets/scholar_copilot_eval_data_1k_related_work.json", "w") as f:
        json.dump(pdf_infos, f, ensure_ascii=False, indent=4)