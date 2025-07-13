import os

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from citegeist.utils.azure_client import AzureClient
from citegeist.utils.helpers import load_api_key
from citegeist.utils.prompts import generate_related_work_score_prompt

load_dotenv()

prompting_client = AzureClient(
    endpoint=os.getenv("AZURE_ENDPOINT"),
    deployment_id=os.getenv("AZURE_PROMPTING_MODEL"),
    api_key=load_api_key(os.getenv("KEY_LOCATION")),
)

OUTPUT_ARXIX_ID_ABSTRACT_RELEVANCE_SCORE_PATH = "out/output_arxiv-ids.csv"
OUTPUT_RELATED_WORK_RELEVANCE_SCORE_PATH = "out/output_related_work_scores.csv"

papers_df = pd.read_csv("out/papers.csv")
arxiv_id_df = pd.read_csv("out/papers-arxiv-ids.csv")

# # Compare the source abstract and the abstracts of all citations (that have an arxiv id)
# output_scores_data = []
#
# for i, row in arxiv_id_df.iterrows():
#     abstract: str = papers_df['abstract'][i]
#     arxiv_ids_source: list[str] = row['arxiv_ids_source'][1:-1].split(', ')
#     source_scores = []
#     for arxiv_id in arxiv_ids_source:
#         abstract2: str = get_arxiv_abstract(arxiv_id)
#         prompt: str = generate_relevance_evaluation_prompt(abstract, abstract2)
#         response: str = prompting_client.get_completions(
#             prompt,
#             os.getenv("AZURE_PROMPTING_MODEL_VERSION")
#         )
#         rating: int = int(response)
#         source_scores.append(rating)
#
#     gpt_scores = []
#     arxiv_ids_gpt4o_mini: list[str] = row['arxiv_ids_gpt4o_mini'][1:-1].split(', ')
#     for arxiv_id in arxiv_ids_gpt4o_mini:
#         abstract2: str = get_arxiv_abstract(arxiv_id)
#         prompt: str = generate_relevance_evaluation_prompt(abstract, abstract2)
#         response: str = prompting_client.get_completions(
#             prompt,
#             os.getenv("AZURE_PROMPTING_MODEL_VERSION")
#         )
#         rating: int = int(response)
#         gpt_scores.append(rating)
#
#     output_scores_data.append({
#         'title': row['title'],
#         'source_score': "|".join([str(i) for i in source_scores]),
#         'source_length': len(source_scores),
#         'gpt4o_score': "|".join([str(i) for i in gpt_scores]),
#         'gpt4o_length': len(gpt_scores)
#     })
#
#
# pd.DataFrame(output_scores_data).to_csv(OUTPUT_ARXIX_ID_ABSTRACT_RELEVANCE_SCORE_PATH, index=False)

output_df = pd.read_csv("out/output.csv")
output_full_pdf_df = pd.read_csv("out/output_full_pdf.csv")

# Score the relevance of the related works between:
# source & source, source & gpt4o_mini, source & ours, source & ours (with full page)

# Switch logic to gemini here.
# from Gemini_Evaluator import prompt_gemini_with_backoff
# project_id = "stellar-depth-441503-q5"
# model_name = "gemini-1.5-pro-002"
# OUTPUT_RELATED_WORK_RELEVANCE_SCORE_PATH = 'out/output_related_work_scores_gemini.csv'

# Switch logic to Mistral Large here
from Gemini_Evaluator import prompt_mistral_with_backoff

project_id = "stellar-depth-441503-q5"
model_name = "mistral-large-2411"
region = "us-central1"
OUTPUT_RELATED_WORK_RELEVANCE_SCORE_PATH = "out/output_related_work_scores_mistral.csv"


output_related_work_scores_data = []

for i, row in tqdm(papers_df.iterrows()):
    abstract = papers_df["abstract"][i]
    source_related_work = papers_df["related_works"][i]
    gpt4o_mini_related_work = papers_df["related_works_gpt4o_mini"][i]
    our_related_work = output_df["related_works"][i]
    our_full_pdf_related_work = output_full_pdf_df["related_works"][i]

    prompt_source: str = generate_related_work_score_prompt(abstract, source_related_work)
    prompt_gpt4o_mini: str = generate_related_work_score_prompt(abstract, gpt4o_mini_related_work)
    prompt_our: str = generate_related_work_score_prompt(abstract, our_related_work)
    prompt_our_full_pdf: str = generate_related_work_score_prompt(abstract, our_full_pdf_related_work)

    # result_source: str = prompting_client.get_completions(
    #     prompt_source,
    #     os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    # )
    # result_source: str = prompt_gemini_with_backoff(project_id, model_name, prompt_source)
    result_source: str = prompt_mistral_with_backoff(project_id, region, model_name, prompt_source)
    rating_source: int = int(result_source)

    # result_gpt4o_mini: str = prompting_client.get_completions(
    #     prompt_gpt4o_mini,
    #     os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    # )
    # result_gpt4o_mini: str = prompt_gemini_with_backoff(project_id, model_name, prompt_gpt4o_mini)
    result_gpt4o_mini: str = prompt_mistral_with_backoff(project_id, region, model_name, prompt_gpt4o_mini)
    rating_gpt4o_mini: int = int(result_gpt4o_mini)

    # result_our: str = prompting_client.get_completions(
    #     prompt_our,
    #     os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    # )
    # result_our: str = prompt_gemini_with_backoff(project_id, model_name, prompt_our)
    result_our: str = prompt_mistral_with_backoff(project_id, region, model_name, prompt_our)
    rating_our: int = int(result_our)

    # result_our_full_pdf: str = prompting_client.get_completions(
    #     prompt_our_full_pdf,
    #     os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    # )
    # result_our_full_pdf: str = prompt_gemini_with_backoff(project_id, model_name, prompt_our_full_pdf)
    result_our_full_pdf: str = prompt_mistral_with_backoff(project_id, region, model_name, prompt_our_full_pdf)
    rating_our_full_pdf: int = int(result_our_full_pdf)

    output_related_work_scores_data.append(
        {
            "title": row["title"],
            "score_source": rating_source,
            "score_gpt4o_mini": rating_gpt4o_mini,
            "score_ours": rating_our,
            "score_our_full_pdf": rating_our_full_pdf,
        }
    )

pd.DataFrame(output_related_work_scores_data).to_csv(OUTPUT_RELATED_WORK_RELEVANCE_SCORE_PATH, index=False)
