# Imports
import math
import os

import pandas as pd
from bertopic import BERTopic
from dotenv import load_dotenv
from pandas import DataFrame
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from citegeist.utils.azure_client import AzureClient
from citegeist.utils.citations import (extract_text_by_page, filter_citations,
                                       get_arxiv_abstract, get_arxiv_citation,
                                       process_arxiv_paper,
                                       process_arxiv_paper_with_embeddings,
                                       remove_citations_and_supplements)
from citegeist.utils.filtering import (
    select_diverse_pages_for_top_b_papers,
    select_diverse_papers_with_weighted_similarity)
from citegeist.utils.helpers import load_api_key
from citegeist.utils.prompts import (generate_related_work_prompt,
                                     generate_relevance_evaluation_prompt,
                                     generate_summary_prompt_with_page_content,
                                     generate_win_rate_evaluation_prompt)


def run_evaluation(pages: list[str], source_abstract: str, source_related_work: str, gpt_related_work: str) -> DataFrame:
    IS_THIS_PAPER_ON_ARXIV = False
    breadth = 10
    depth = 2
    diversity = 0

    # Initialize clients
    topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    client = MilvusClient("./database.db")
    prompting_client = AzureClient(
        endpoint=os.getenv("AZURE_ENDPOINT"),
        deployment_id=os.getenv("AZURE_PROMPTING_MODEL"),
        api_key=load_api_key(os.getenv("KEY_LOCATION")),
    )

    # Create embeddings for all pages
    page_embeddings = [embedding_model.encode(page) for page in pages]

    # Query Milvus Vector DB for each page
    all_query_data: list[list[dict]] = []
    for embedding in page_embeddings:
        query_result = client.search(
            collection_name="abstracts",
            data=[embedding],
            limit=6 * breadth,
            anns_field="embedding",
            # filter = f'topic == {topic_id}',  # Could potentially use topic_ids here
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["embedding"],
        )
        all_query_data.extend(query_result)

    print(f"Retrieved papers from DB for {len(all_query_data)} pages.")

    # Aggregate similarity scores for papers that appear multiple times
    paper_scores: dict[str, float] = {}
    paper_data: dict[str, dict] = {}

    for page_results in all_query_data:
        for result in page_results:
            paper_id = result["id"]
            similarity_score = result["distance"]  # Assuming this is the similarity score

            if paper_id in paper_scores:
                paper_scores[paper_id] += similarity_score
            else:
                paper_scores[paper_id] = similarity_score
                paper_data[paper_id] = {"id": paper_id, "embedding": result["entity"]["embedding"]}

    # Convert aggregated results back to format expected by select_diverse_papers
    # Sort papers by aggregated score and take top 6*breadth papers
    top_paper_ids = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)[: 6 * breadth]

    # Convert back to original format expected by select_diverse_papers
    # Each entry should be a list with one dict per query result
    aggregated_query_data = [
        {"id": paper_id, "embedding": paper_data[paper_id]["embedding"], "distance": score}
        for paper_id, score in top_paper_ids
    ]

    # Select a longlist of papers using aggregated scores
    selected_papers: list[dict] = select_diverse_papers_with_weighted_similarity(
        paper_data=aggregated_query_data, k=3 * breadth, diversity_weight=diversity
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
    page_embeddings_papers: list[list[dict]] = []
    for paper in selected_papers:
        arxiv_id = paper["id"]
        result = process_arxiv_paper_with_embeddings(arxiv_id, topic_model)
        if result:
            page_embeddings_papers.append(result)

    print(f"Generated page embeddings for {len(page_embeddings_papers)} papers.")

    # Generate shortlist of papers using first page as reference
    # (you might want to modify this to consider all input pages)
    relevant_pages: list[dict] = select_diverse_pages_for_top_b_papers(
        paper_embeddings=page_embeddings_papers,
        input_string=pages[0],  # Using first page as reference
        topic_model=topic_model,
        k=depth,
        b=breadth,
        diversity_weight=diversity,
        skip_first=False,
    )

    print(f"Selected {len(relevant_pages)} papers for the shortlist.")

    shortlist_ratings: dict[str, int] = {}
    for obj in relevant_pages:
        # arxiv_id = aggregated_query_data[obj['paper_id']]["id"]
        arxiv_id = selected_papers[obj["paper_id"]]["id"]
        shortlist_ratings[arxiv_id] = relevance_ratings[arxiv_id]

    avg_shortlist_rating: float = sum(shortlist_ratings.values()) / len(shortlist_ratings)

    print(f"Average shortlist relevance rating: {avg_shortlist_rating}")
    print(shortlist_ratings.values())

    # Generate summaries for individual papers
    for obj in relevant_pages:
        arxiv_id = aggregated_query_data[obj["paper_id"]]["id"]
        arxiv_abstract = get_arxiv_abstract(arxiv_id)
        text_segments = obj["text"]
        response: str = prompting_client.get_completions(
            generate_summary_prompt_with_page_content(
                abstract_source_paper=pages[0],  # Using first page as reference
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
            source_abstract=pages[0],  # Using first page as reference
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

    # return {
    #     'related_works': related_works_section,
    #     'citations': filtered_citations
    # }

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
    output_file = "out/output_full_pdf.csv"

    input_df = pd.read_csv(input_file)
    output_df = None
    for i, row in input_df.iterrows():
        print(f"Starting process for paper: {row['title']}")
        arxiv_id = row["arxiv_id"]
        file_path = row["file_path"]

        raw_pages = []
        if pd.notna(arxiv_id):
            raw_pages = process_arxiv_paper(arxiv_id)
        elif pd.notna(file_path):
            raw_pages = extract_text_by_page(file_path)

        pages = remove_citations_and_supplements(raw_pages)

        row_df: pd.DataFrame = run_evaluation(
            pages=pages,
            source_abstract=row["abstract"],
            source_related_work=row["related_works"],
            gpt_related_work=row["related_works_gpt4o_mini"],
        )
        if output_df is None:
            output_df = row_df
        else:
            output_df = pd.concat([output_df, row_df])
    output_df.to_csv(output_file, index=False)
