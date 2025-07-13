import os

import pandas as pd
from dotenv import load_dotenv

from citegeist.utils.azure_client import AzureClient
from citegeist.utils.helpers import load_api_key
from citegeist.utils.prompts import generate_win_rate_evaluation_prompt

load_dotenv()

papers_df = pd.read_csv("out/papers.csv")
output_df = pd.read_csv("out/output.csv")

prompting_client = AzureClient(
    endpoint=os.getenv("AZURE_ENDPOINT"),
    deployment_id=os.getenv("AZURE_PROMPTING_MODEL"),
    api_key=load_api_key(os.getenv("KEY_LOCATION")),
)

for i, row in papers_df.iterrows():
    abstract = row["abstract"]
    source_related_works = row["related_works"]
    gpt_related_works = row["papers.csv"]

    output_row = output_df.loc[i]
    our_related_works = output_row["related_works"]

    # first pair, first round
    win_rate_prompt, order = generate_win_rate_evaluation_prompt(
        source_abstract=abstract,
        source_related_work=source_related_works,
        target_related_work=our_related_works,
        reverse_order=False,
    )

    win_rate_response: str = prompting_client.get_completions(
        win_rate_prompt, os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    )

    print(win_rate_response)
    # Modify the results to account for reordering
    winner_paper_1: str = ""
    if win_rate_response == "Section A":
        winner_paper_1 = "source"
    elif win_rate_response == "Section B":
        winner_paper_1 = "ours"
    elif win_rate_response == "Tie":
        winner_paper_1 = "tie"

    # first pair, second round
    win_rate_prompt2, order2 = generate_win_rate_evaluation_prompt(
        source_abstract=abstract,
        source_related_work=source_related_works,
        target_related_work=our_related_works,
        reverse_order=True,
    )

    win_rate_response2: str = prompting_client.get_completions(
        win_rate_prompt2, os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    )

    print(win_rate_response2)
    # Modify the results to account for reordering
    winner_paper_2: str = ""
    if win_rate_response == "Section A":
        winner_paper_2 = "ours"
    elif win_rate_response == "Section B":
        winner_paper_2 = "source"
    elif win_rate_response == "Tie":
        winner_paper_2 = "tie"

    score_first_pair = 0
    if winner_paper_1 == "source" and winner_paper_2 == "source":
        score_first_pair = -1
    elif (
        winner_paper_1 == "source"
        and winner_paper_2 == "ours"
        or winner_paper_1 == "ours"
        and winner_paper_2 == "source"
    ):
        score_first_pair = 0
    elif winner_paper_1 == "ours" and winner_paper_2 == "ours":
        score_first_pair = 1
    elif winner_paper_1 == "tie" and winner_paper_2 == "source":
        score_first_pair = -1
    elif winner_paper_1 == "tie" and winner_paper_2 == "ours":
        score_first_pair = 1
    elif winner_paper_1 == "source" and winner_paper_2 == "tie":
        score_first_pair = -1
    elif winner_paper_1 == "ours" and winner_paper_2 == "tie":
        score_first_pair = 1
    elif winner_paper_1 == "tie" and winner_paper_2 == "tie":
        score_first_pair = 0

    # second pair, first round
    win_rate_prompt3, order3 = generate_win_rate_evaluation_prompt(
        source_abstract=abstract,
        source_related_work=our_related_works,
        target_related_work=gpt_related_works,
        reverse_order=False,
    )

    win_rate_response3: str = prompting_client.get_completions(
        win_rate_prompt3, os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    )

    print(win_rate_response3)
    # Modify the results to account for reordering
    winner_paper_3: str = ""
    if win_rate_response3 == "Section A":
        winner_paper_3 = "ours"
    elif win_rate_response3 == "Section B":
        winner_paper_3 = "gpt"
    elif win_rate_response3 == "Tie":
        winner_paper_3 = "tie"

    # second pair, second round
    win_rate_prompt4, order4 = generate_win_rate_evaluation_prompt(
        source_abstract=abstract,
        source_related_work=source_related_works,
        target_related_work=our_related_works,
        reverse_order=True,
    )

    win_rate_response4: str = prompting_client.get_completions(
        win_rate_prompt4, os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    )

    print(win_rate_response4)
    # Modify the results to account for reordering
    winner_paper_4: str = ""
    if win_rate_response == "Section A":
        winner_paper_4 = "gpt"
    elif win_rate_response == "Section B":
        winner_paper_4 = "ours"
    elif win_rate_response == "Tie":
        winner_paper_4 = "tie"

    score_second_pair = 0
    if winner_paper_3 == "gpt" and winner_paper_4 == "gpt":
        score_second_pair = -1
    elif winner_paper_3 == "ours" and winner_paper_4 == "gpt" or winner_paper_3 == "gpt" and winner_paper_4 == "ours":
        score_second_pair = 0
    elif winner_paper_3 == "ours" and winner_paper_4 == "ours":
        score_second_pair = 1
    elif winner_paper_3 == "tie" and winner_paper_4 == "gpt":
        score_second_pair = -1
    elif winner_paper_3 == "tie" and winner_paper_4 == "ours":
        score_second_pair = 1
    elif winner_paper_3 == "gpt" and winner_paper_4 == "tie":
        score_second_pair = -1
    elif winner_paper_3 == "ours" and winner_paper_4 == "tie":
        score_second_pair = 1
    elif winner_paper_3 == "tie" and winner_paper_4 == "tie":
        score_second_pair = 0

    print(f"Paper {i}: {score_first_pair}, {score_second_pair}")
    print("------------")
