# Imports
import os

from bertopic import BERTopic
from dotenv import load_dotenv
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from citegeist.utils.azure_client import AzureClient
from citegeist.utils.citations import (get_arxiv_abstract,
                                       process_arxiv_paper_with_embeddings)
from citegeist.utils.filtering import (
    select_diverse_pages_for_top_b_papers,
    select_diverse_papers_with_weighted_similarity)
from citegeist.utils.helpers import load_api_key
from citegeist.utils.prompts import generate_relevance_evaluation_prompt

# Load environment variables
load_dotenv()

abstract = (
    "This work provides a geometric version of the next-generation method for obtaining the basic reproduction"
    " number of an epidemiological model. More precisely, we generalize the concept of the basic reproduction"
    " number for the theory of Petri nets. The motivation comes from observing that any epidemiological model"
    " has the basic structures found in the SIR model of Kermack-McKendrick. These are three substructures,"
    " also given by three Petri nets inside, representing the susceptible population, the infection process,"
    " and the infected individuals. The five assumptions of the next-generation matrix given by van den"
    " Driessche-Watmough can be described geometrically using Petri nets. Thus, the next-generation matrix"
    " results in a matrix of flows between the infection compartments. Besides that, we construct a procedure"
    " that associates with a system of ordinary differential equations under certain assumptions, a Petri net"
    " that is minimal. In addition, we explain how Petri nets extend compartmental models to include vertical"
    " transmission."
)
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

embedded_abstract = embedding_model.encode(abstract)
topic = topic_model.transform(abstract)
topic_id = topic[0][0]

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
        generate_relevance_evaluation_prompt(source_abstract=abstract, target_abstract=get_arxiv_abstract(paper["id"])),
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
    input_string=abstract,
    topic_model=topic_model,
    k=depth,
    b=breadth,
    diversity_weight=diversity,
    skip_first=False,
)

print(f"Selected {len(relevant_pages)} papers for the shortlist.")

shortlist_ratings: dict[str, int] = {}
for obj in relevant_pages:
    arxiv_id = query_data[obj["paper_id"]]["id"]
    shortlist_ratings[arxiv_id] = relevance_ratings[arxiv_id]

avg_shortlist_rating: float = sum(shortlist_ratings.values()) / len(shortlist_ratings)

print(f"Average shortlist relevance rating: {avg_shortlist_rating}")
print(shortlist_ratings.values())
