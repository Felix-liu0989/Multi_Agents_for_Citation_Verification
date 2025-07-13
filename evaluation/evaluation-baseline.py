# Imports
import os

from bertopic import BERTopic
from dotenv import load_dotenv
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from citegeist.utils.azure_client import AzureClient
from citegeist.utils.citations import (filter_citations, get_arxiv_abstract,
                                       get_arxiv_citation)
from citegeist.utils.helpers import load_api_key
from citegeist.utils.prompts import (generate_related_work_prompt,
                                     generate_relevance_evaluation_prompt,
                                     generate_summary_prompt)

load_dotenv()

topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
client = MilvusClient("./database.db")
prompting_client = AzureClient(
    endpoint=os.getenv("AZURE_ENDPOINT"),
    deployment_id=os.getenv("AZURE_PROMPTING_MODEL"),
    api_key=load_api_key(os.getenv("KEY_LOCATION")),
)
import pandas as pd


def run_evaluation(source_abstract: str, source_related_work: str, gpt_related_work: str) -> pd.DataFrame:
    IS_THIS_PAPER_ON_ARXIV = False

    # Initialize clients
    # topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    client = MilvusClient("./database.db")
    prompting_client = AzureClient(
        endpoint=os.getenv("AZURE_ENDPOINT"),
        deployment_id=os.getenv("AZURE_PROMPTING_MODEL"),
        api_key=load_api_key(os.getenv("KEY_LOCATION")),
    )

    embedded_abstract = embedding_model.encode(source_abstract)
    # topic = topic_model.transform(source_abstract)
    # topic_id = topic[0][0]

    # Query Milvus Vector DB
    query_data: list[list[dict]] = client.search(
        collection_name="abstracts",
        data=[embedded_abstract],
        limit=10,
        anns_field="embedding",
        # filter = f'topic == {topic_id}',
        search_params={"metric_type": "COSINE", "params": {}},
        output_fields=["embedding"],
    )

    # Clean DB response data
    query_data: list[dict] = query_data[0]
    for obj in query_data:
        obj["embedding"] = obj["entity"]["embedding"]
        obj.pop("entity")

    # Remove first entry (this is the paper we're searching for). As it has already been published, it will show up with large similarity
    if IS_THIS_PAPER_ON_ARXIV:
        query_data = query_data[1:]

    print(f"Retrieved {len(query_data)} papers from the DB.")

    relevance_ratings: dict[str, int] = {}
    # calculate average longlist relevance through prompting
    for paper in query_data:
        response: str = prompting_client.get_completions(
            generate_relevance_evaluation_prompt(
                source_abstract=source_abstract, target_abstract=get_arxiv_abstract(paper["id"])
            ),
            os.getenv("AZURE_PROMPTING_MODEL_VERSION"),
        )
        rating: int = int(response)
        relevance_ratings[paper["id"]] = rating

    avg_longlist_rating: float = sum(relevance_ratings.values()) / len(relevance_ratings)
    print(f"Average longlist relevance rating: {avg_longlist_rating}")
    print(relevance_ratings.values())

    for obj in query_data:
        arxiv_id = obj["id"]
        arxiv_abstract = get_arxiv_abstract(arxiv_id)
        response: str = prompting_client.get_completions(
            generate_summary_prompt(source_abstract, arxiv_abstract),
            os.getenv("AZURE_PROMPTING_MODEL_VERSION"),
        )
        obj["summary"] = response
        obj["citation"] = get_arxiv_citation(arxiv_id)

    related_works_section: str = prompting_client.get_completions(
        generate_related_work_prompt(source_abstract, query_data),
        os.getenv("AZURE_PROMPTING_MODEL_VERSION"),
    )

    # win_rate_prompt, order = generate_win_rate_evaluation_prompt(
    #     source_abstract=source_abstract,
    #     source_related_work=source_related_work,
    #     target_related_work=related_works_section
    # )
    #
    # win_rate_response: str = prompting_client.get_completions(
    #     win_rate_prompt,
    #     os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    # )
    #
    # print(win_rate_response)
    # # Modify the results to account for reordering
    # selected_winner_paper_source_ours: str = ''
    # if win_rate_response == 'Section A':
    #     if order[0] == 'source':
    #         selected_winner_paper_source_ours = order[0]
    #     elif order[0] == 'target':
    #         selected_winner_paper_source_ours = 'our_approach'
    # elif win_rate_response == 'Section B':
    #     if order[1] == 'source':
    #         selected_winner_paper_source_ours = order[1]
    #     elif order[1] == 'target':
    #         selected_winner_paper_source_ours = 'our_approach'
    # elif win_rate_response == 'Tie':
    #     print(f'There was a tie.')
    #     selected_winner_paper_source_ours = 'tie'
    #
    # win_rate_prompt2, order2 = generate_win_rate_evaluation_prompt(
    #     source_abstract=source_abstract,
    #     source_related_work=related_works_section,
    #     target_related_work=gpt_related_work
    # )
    #
    # win_rate_response2: str = prompting_client.get_completions(
    #     win_rate_prompt2,
    #     os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    # )
    #
    # print(win_rate_response2)
    # # Modify the results to account for reordering
    # selected_winner_paper_ours_gpt: str = ''
    # if win_rate_response2 == 'Section A':
    #     if order2[0] == 'source':
    #         selected_winner_paper_ours_gpt = 'our_approach'
    #     elif order2[0] == 'target':
    #         selected_winner_paper_ours_gpt = 'gpt'
    # elif win_rate_response2 == 'Section B':
    #     if order2[1] == 'source':
    #         selected_winner_paper_ours_gpt = 'our_approach'
    #     elif order2[1] == 'target':
    #         selected_winner_paper_ours_gpt = 'gpt'
    # elif win_rate_response2 == 'Tie':
    #     print(f'There was a tie.')
    #     selected_winner_paper_ours_gpt = 'tie'

    # Filter citations so that we only output citation strings that are actually reflected in the generated related works section
    filtered_citations: list[str] = filter_citations(
        related_works_section=related_works_section, citation_strings=[obj["citation"] for obj in query_data]
    )

    return pd.DataFrame(
        [
            {
                "shortlist_length": len(query_data),
                "shortlist_ratings": "|".join([str(i) for i in relevance_ratings.values()]),
                # 'rating_source_ours': selected_winner_paper_source_ours,
                # 'rating_ours_gpt': selected_winner_paper_ours_gpt,
                "related_works": related_works_section,
                "citations": "\n".join(filtered_citations),
            }
        ]
    )


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    input_file = "out/papers.csv"
    output_file = "out/output_baseline.csv"

    input_df = pd.read_csv(input_file)
    output_df = None
    for i, row in input_df.iterrows():
        print(f"Starting process for paper: {row['Title']}")
        row_df: pd.DataFrame = run_evaluation(
            source_abstract=row["Abstract"],
            source_related_work=row["Related Works"],
            gpt_related_work=row["Related Works (GPT-4o-mini)"],
        )
        if output_df is None:
            output_df = row_df
        else:
            output_df = pd.concat([output_df, row_df])
        output_df.to_csv(output_file, index=False)
