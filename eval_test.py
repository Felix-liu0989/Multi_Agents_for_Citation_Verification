# {"related_works": related_works_section, "citations": filtered_citations, "selected_papers": relevant_pages}
from evaluation.agents.judge import Judge
import json
from pathlib import Path

with open(
    "/home/liujian/project/2025-07/A2R-code-reproduction/results/Self-guided tree search for attributed text generation/related_work_with_citations_0708.json",
    "r",
) as f:
    data = json.load(f)

# key2idx: dict[str, int] = {}
# paper_infos: list[dict] = []


# for idx, item in enumerate(data["selected_papers"]):
#     if idx >= len(data["citations"]):
#         break
#     citation = item["citation"]
#     print(f"citation: {citation}")
#     for c in data["citations"]:
#         if c == citation:
#             print(f"c: {c}")
#             key = judge._make_key(c)
#             print(f"key: {key}")
#             if not key:
#                 continue
#             key2idx[key] = idx
#             page_texts = item.get("text", [])
    # print(f"citation: {citation}")
    # full_ref = data["citations"][idx]
    # print(f"full_ref: {full_ref}")
    # key = judge._make_key(full_ref)
    # print(f"key: {key}")
    # if not key:
    #     continue
    # key2idx[key] = idx
    # page_texts = item.get("text", [])
    # if isinstance(page_texts, list):
    #     content = "\n".join(page_texts)
    # else:
    #     content = str(page_texts)
    # paper_infos.append({"title": full_ref, "content": content})

import os

judge = Judge(model="google/gemini-2.5-flash-preview-05-20")
result = judge.citation_quality_cite_ids(data)
print(result)


print(os.environ.get("OPENROUTER_API_KEY", ""))

# ===Connection Test
# from citegeist.utils.llm_clients.gemini_client import GeminiClient
# import os

# client = GeminiClient(
#     api_key = os.environ.get("OPENROUTER_API_KEY", ""),
#     model_name = "google/gemini-2.5-flash-preview-05-20"
# )

# result = client.get_completion("Hello, world!")
# print(result)
