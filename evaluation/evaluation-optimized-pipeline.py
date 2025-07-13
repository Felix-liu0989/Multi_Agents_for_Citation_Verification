# Imports
import math
import os
import sys

import pandas as pd
from bertopic import BERTopic
from dotenv import load_dotenv
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from citegeist.utils.azure_client import AzureClient
from citegeist.utils.citations import (filter_citations, get_arxiv_abstract,
                                       get_arxiv_citation,
                                       process_arxiv_paper_with_embeddings)
from citegeist.utils.filtering import (
    select_diverse_pages_for_top_b_papers,
    select_diverse_papers_with_weighted_similarity)
from citegeist.utils.helpers import load_api_key
from citegeist.utils.prompts import (generate_related_work_prompt,
                                     generate_relevance_evaluation_prompt,
                                     generate_summary_prompt_with_page_content,
                                     generate_win_rate_evaluation_prompt)


def run_evaluation(
    source_abstract: str,
    source_related_work: str,
    gpt_related_work: str,
    breadth: int = 10,
    depth: int = 2,
    diversity: float = 0,
) -> pd.DataFrame:
    IS_THIS_PAPER_ON_ARXIV = False

    # Initialize clients
    topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")
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
        limit=6 * breadth,
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

    # Select a longlist of papers
    selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
        paper_data=query_data, k=3 * breadth, diversity_weight=diversity
    )

    print(f"Selected {len(selected_papers)} papers for the longlist.")

    relevance_ratings: dict[str, int] = {}
    # calculate average longlist relevance through prompting
    for paper in selected_papers:
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

    # Generate embeddings of each page of every paper in the longlist
    page_embeddings: list[list[dict]] = []
    for paper in selected_papers:
        arxiv_id = paper["id"]
        result = process_arxiv_paper_with_embeddings(arxiv_id, topic_model)
        if result:
            page_embeddings.append(result)

    print(f"Generated page embeddings for {len(page_embeddings)} papers.")

    # Generate shortlist of papers (at most k pages per paper, at most b papers in total)
    relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
        paper_embeddings=page_embeddings,
        input_string=source_abstract,
        topic_model=topic_model,
        k=depth,
        b=breadth,
        diversity_weight=diversity,
        skip_first=False,
    )

    print(f"Selected {len(relevant_pages)} papers for the shortlist.")

    shortlist_ratings: dict[str, int] = {}
    for obj in relevant_pages:
        # this seems to have to be selected_papers instead of query data
        arxiv_id = selected_papers[obj["paper_id"]]["id"]
        shortlist_ratings[arxiv_id] = relevance_ratings[arxiv_id]

    avg_shortlist_rating: float = sum(shortlist_ratings.values()) / len(shortlist_ratings)

    print(f"Average shortlist relevance rating: {avg_shortlist_rating}")
    print(shortlist_ratings.values())

    # Generate summaries for individual papers (taking all relevant pages into account)
    for obj in relevant_pages:
        # Because paper_id != arXiv_id -> retrieve arXiv id/
        arxiv_id = query_data[obj["paper_id"]]["id"]
        arxiv_abstract = get_arxiv_abstract(arxiv_id)
        text_segments = obj["text"]
        response: str = prompting_client.get_completions(
            generate_summary_prompt_with_page_content(
                abstract_source_paper=source_abstract,
                abstract_to_be_cited=arxiv_abstract,
                page_text_to_be_cited=text_segments,
                sentence_count=5,
            ),
            os.getenv("AZURE_PROMPTING_MODEL_VERSION"),
        )
        obj["summary"] = response
        obj["citation"] = get_arxiv_citation(arxiv_id)

    print("Generated summaries of papers (and their pages).")

    # Generate the final related works section text
    related_works_section: str = prompting_client.get_completions(
        generate_related_work_prompt(
            source_abstract=source_abstract,
            data=relevant_pages,
            paragraph_count=math.ceil(breadth / 2),
            add_summary=False,
        ),
        os.getenv("AZURE_PROMPTING_MODEL_VERSION"),
    )

    # Identify whether the source or the generated related works section is preferred
    win_rate_prompt, order = generate_win_rate_evaluation_prompt(
        source_abstract=source_abstract,
        source_related_work=source_related_work,
        target_related_work=related_works_section,
    )

    win_rate_response: str = prompting_client.get_completions(
        win_rate_prompt, os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    )

    print(win_rate_response)
    # Modify the results to account for reordering
    selected_winner_paper_source_ours: str = ""
    if win_rate_response == "Section A":
        print(f"The {order[0]} paper generated related works section was deemed the winner. (source = original)")
        selected_winner_paper_source_ours = order[0]
    elif win_rate_response == "Section B":
        print(f"The {order[1]} paper generated related works section was deemed the winner. (source = original)")
        selected_winner_paper_source_ours = order[1]
    elif win_rate_response == "Tie":
        print(f"There was a tie.")
        selected_winner_paper_source_ours = "tie"

    # Identify whether the generated (our method) or the gpt generated related works section is preferred
    win_rate_prompt2, order2 = generate_win_rate_evaluation_prompt(
        source_abstract=source_abstract, source_related_work=related_works_section, target_related_work=gpt_related_work
    )

    win_rate_response2: str = prompting_client.get_completions(
        win_rate_prompt2, os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    )

    print(win_rate_response2)
    # Modify the results to account for reordering
    selected_winner_paper_ours_gpt: str = ""
    if win_rate_response2 == "Section A":
        print(f"The {order[0]} paper generated related works section was deemed the winner. (source = original)")
        selected_winner_paper_ours_gpt = order2[0]
    elif win_rate_response2 == "Section B":
        print(f"The {order[1]} paper generated related works section was deemed the winner. (source = original)")
        selected_winner_paper_ours_gpt = order2[1]
    elif win_rate_response2 == "Tie":
        print(f"There was a tie.")
        selected_winner_paper_ours_gpt = "tie"

    # Filter citations so that we only output citation strings that are actually reflected in the generated related works section
    filtered_citations: list[str] = filter_citations(
        related_works_section=related_works_section, citation_strings=[obj["citation"] for obj in relevant_pages]
    )

    print(f"Generated related work section with {len(filtered_citations)} citations.")

    # print({
    #     'related_works': related_works_section,
    #     'citations': filtered_citations
    # })

    return pd.DataFrame(
        [
            {
                "longlist_length": len(selected_papers),
                "longlist_ratings": "|".join([str(i) for i in relevance_ratings.values()]),
                "shortlist_length": len(relevant_pages),
                "shortlist_ratings": "|".join([str(i) for i in shortlist_ratings.values()]),
                "rating_source_ours": selected_winner_paper_source_ours,
                "rating_ours_gpt": selected_winner_paper_ours_gpt,
                "related_works": related_works_section,
                "citations": "\n".join(filtered_citations),
            }
        ]
    )


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    input_file = "out/papers.csv"
    output_file = "out/output.csv"

    input_df = pd.read_csv(input_file)
    output_df = None

    breadth = 10
    depth = 2
    diversity = 0.0
    if len(sys.argv) == 5:
        breadth = int(sys.argv[1])
        depth = int(sys.argv[2])
        diversity = float(sys.argv[3])
        output_file = sys.argv[4]

    for i, row in input_df.iterrows():
        print(f"Starting process for paper: {row['title']}")

        row_df: pd.DataFrame = run_evaluation(
            source_abstract=row["abstract"],
            source_related_work=row["related_works"],
            gpt_related_work=row["related_works_gpt4o_mini"],
            breadth=breadth,
            depth=depth,
            diversity=diversity,
        )
        if output_df is None:
            output_df = row_df
        else:
            output_df = pd.concat([output_df, row_df])
    output_df.to_csv(output_file, index=False)
