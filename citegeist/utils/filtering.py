import numpy as np
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity


def extract_most_relevant_pages(
    paper_embeddings: list[object],
    input_string: str,
    topic_model: BERTopic,
    top_k: int = 5,
    skip_first: bool = False,
) -> list[dict]:
    """
    Extracts the most relevant pages based on cosine similarity with an input string.

    Args:
        paper_embeddings (list): A list of papers, each containing dictionaries with "text" and "embedding".
        input_string (str): The string to compare against.
        topic_model (BERTopic): The BERTopic model for generating embeddings.
        top_k (int): The number of most relevant pages to return.
        skip_first (bool): Whether to skip the first page of the input string.

    Returns:
        list: A list of dictionaries with "paper_id", "page_number", "text", and "similarity".
    """
    # Encode the input string to get its embedding
    embedding_model = topic_model.embedding_model
    input_embedding = embedding_model.embedding_model.encode(input_string)

    results = []

    for paper_idx, paper in enumerate(paper_embeddings):
        paper_id = paper_idx  # Replace with actual paper ID if available in `paper_embeddings`
        for page_number, page_data in enumerate(paper):
            if skip_first and page_number == 0:
                continue
            page_text = page_data["text"]
            page_embedding = page_data["embedding"]

            # Compute cosine similarity
            similarity = cosine_similarity([input_embedding], [page_embedding])[0][0]

            # Append results
            results.append(
                {
                    "paper_id": paper_id,  # Or use a specific ID if available
                    "page_number": page_number + 1,  # Pages are 1-indexed
                    "text": page_text,
                    "similarity": similarity,
                }
            )

    # Sort results by similarity score in descending order
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)

    # Return the top-k most relevant pages
    return results[:top_k]


def extract_most_relevant_pages_for_each_paper(
    paper_embeddings: list[dict],
    input_string: str,
    topic_model: BERTopic,
    top_k: int = 5,
    skip_first: bool = False,
) -> list[dict]:
    """
    Extracts the top-k most relevant pages for each paper based on cosine similarity with an input string.

    Args:
        paper_embeddings (list): A list of papers, each containing dictionaries with "text" and "embedding".
        input_string (str): The string to compare against.
        topic_model (BERTopic): The BERTopic model for generating embeddings.
        top_k (int): The number of most relevant pages to return for each paper.
        skip_first (bool): Whether to skip the first page of the input string.

    Returns:
        list: A list of dictionaries with "paper_id", "page_number", "text", and "similarity".
    """
    # Encode the input string to get its embedding
    embedding_model = topic_model.embedding_model.embedding_model
    input_embedding = embedding_model.encode(input_string)

    all_results = []

    # Loop through each paper
    for paper_idx, paper in enumerate(paper_embeddings):
        paper_id = paper_idx  # Replace with actual paper ID if available in `paper_embeddings`
        paper_results = []

        # Loop through each page in the paper
        for page_number, page_data in enumerate(paper):
            if skip_first and page_number == 0:
                continue
            page_text = page_data["text"]
            page_embedding = page_data["embedding"]

            # Compute cosine similarity between input string and the current page
            similarity = cosine_similarity([input_embedding], [page_embedding])[0][0]

            # Add the page's data and similarity to the paper's results
            paper_results.append(
                {
                    "paper_id": paper_id,  # Or use a specific ID if available
                    "page_number": page_number + 1,  # Pages are 1-indexed
                    "text": page_text,
                    "similarity": similarity,
                }
            )

        # Sort the pages for the current paper by similarity score in descending order
        paper_results_sorted = sorted(paper_results, key=lambda x: x["similarity"], reverse=True)

        # Add the top-k pages for the current paper to the final results
        all_results.extend(paper_results_sorted[:top_k])

    return all_results


def select_diverse_papers_with_precomputed_distances(paper_data: list, k: int) -> list[str]:
    """
    Selects `k` papers that maximize diversity while minimizing distance to the input paper.

    Args:
        paper_data (list): List of dictionaries, where each dictionary represents a paper and contains:
                           - "id": Paper ID
                           - "distance": Similarity to the input paper
                           - "embedding": Embedding of the paper
        k (int): Number of papers to select.

    Returns:
        list: IDs of the selected papers.
    """
    # Extract distances and embeddings
    distances = np.array([paper["distance"] for paper in paper_data])
    embeddings = np.array([paper["entity"]["embedding"] for paper in paper_data])

    # Start with the most similar paper
    selected_indices = [np.argmax(distances)]

    # Iteratively select papers
    for _ in range(1, k):
        max_combined_score = -np.inf
        next_index = -1

        for i in range(len(paper_data)):
            if i in selected_indices:
                continue

            # Compute the diversity score: minimum distance to selected papers
            diversity_score = min(cosine_similarity([embeddings[i]], [embeddings[j]])[0][0] for j in selected_indices)

            # Combined score: similarity to input paper and diversity
            similarity_to_input = distances[i]
            combined_score = similarity_to_input * 0.5 + (1 - diversity_score) * 0.5  # Adjust weights

            if combined_score > max_combined_score:
                max_combined_score = combined_score
                next_index = i

        selected_indices.append(next_index)

    # Return the IDs of the selected papers
    return [paper_data[i] for i in selected_indices]


def select_diverse_papers_with_weighted_similarity(
    paper_data: list[dict], k: int, diversity_weight: float = 0.25
) -> list[dict]:
    """
    Selects `k` papers that balance diversity and similarity to the input paper based on the `diversity_weight`.

    Args:
        paper_data (list): List of dictionaries, where each dictionary represents a paper and contains:
                           - "id": Paper ID
                           - "distance": Similarity to the input paper
                           - "embedding": Embedding of the paper
        k (int): Number of papers to select.
        diversity_weight (float): Weight between similarity and diversity (0 to 1).
                                  1 means prioritizing diversity, 0 means prioritizing similarity.

    Returns:
        list: List of dictionaries with the original data of the selected papers.
    """
    # Extract distances and embeddings
    distances = np.array([paper["distance"] for paper in paper_data])
    embeddings = np.array([paper["embedding"] for paper in paper_data])

    # Start with the most similar paper
    selected_indices = [np.argmax(distances)]

    # Iteratively select papers
    for _ in range(1, k):
        max_combined_score = -np.inf
        next_index = -1

        for i in range(len(paper_data)):
            if i in selected_indices:
                continue

            # Compute the diversity score: minimum distance to selected papers
            diversity_score = min(cosine_similarity([embeddings[i]], [embeddings[j]])[0][0] for j in selected_indices)

            # Combined score: similarity to input paper and diversity
            similarity_to_input = distances[i]
            combined_score = (1 - diversity_weight) * similarity_to_input + diversity_weight * (1 - diversity_score)

            if combined_score > max_combined_score:
                max_combined_score = combined_score
                next_index = i

        selected_indices.append(next_index)

    # Return the IDs of the selected papers
    return [paper_data[i] for i in selected_indices]


def select_diverse_pages_for_top_b_papers(
    paper_embeddings: list[list[dict]],
    input_string: str,
    topic_model: BERTopic,
    k: int = 5,
    b: int = 3,
    diversity_weight: float = 0.25,
    skip_first: bool = False,
) -> list[dict]:
    """
    Selects `k` pages for the top `b` papers based on average similarity to the input string,
    while balancing similarity to the input and diversity among the selected pages.

    Args:
        paper_embeddings (list): A list of papers, each containing dictionaries with "text" and "embedding".
        input_string (str): The string to compare against.
        topic_model (BERTopic): The BERTopic model for generating embeddings.
        k (int): The number of pages to select for each paper.
        b (int): The number of papers to select based on the highest average similarity.
        diversity_weight (float): Weight between similarity and diversity (0 to 1).
                                  1 means prioritizing diversity, 0 means prioritizing similarity.
        skip_first (bool): Whether to skip the first page of the input string.

    Returns:
        list: A list of dictionaries, each containing "paper_id" and "text" (list of selected page texts).
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    # Encode the input string to get its embedding
    embedding_model = topic_model.embedding_model.embedding_model
    input_embedding = embedding_model.encode(input_string)

    paper_results = []

    # Loop through each paper
    for paper_idx, paper in enumerate(paper_embeddings):
        paper_id = paper_idx  # Replace with actual paper ID if available in `paper_embeddings`
        page_candidates = []

        # Loop through each page in the paper
        for page_number, page_data in enumerate(paper):
            if skip_first and page_number == 0:
                continue

            page_text = page_data["text"]
            page_embedding = page_data["embedding"]

            # Compute cosine similarity between input string and the current page
            similarity = cosine_similarity([input_embedding], [page_embedding])[0][0]

            # Add the page's data and similarity to candidates
            page_candidates.append(
                {
                    "paper_id": paper_id,
                    "page_number": page_number + 1,  # Pages are 1-indexed
                    "text": page_text,
                    "embedding": page_embedding,
                    "similarity": similarity,
                }
            )

        # Sort pages by similarity score in descending order
        page_candidates_sorted = sorted(page_candidates, key=lambda x: x["similarity"], reverse=True)

        # Select pages iteratively based on similarity and diversity
        selected_pages = [page_candidates_sorted[0]]  # Start with the most similar page
        for _ in range(1, k):
            max_combined_score = -np.inf
            next_page = None

            for candidate in page_candidates:
                if candidate in selected_pages:
                    continue

                # Compute the diversity score: minimum similarity to selected pages
                diversity_score = min(
                    cosine_similarity(
                        [candidate["embedding"]],
                        [selected["embedding"]],
                    )[
                        0
                    ][0]
                    for selected in selected_pages
                )

                # Combined score: similarity to input string and diversity
                combined_score = (1 - diversity_weight) * candidate["similarity"] + diversity_weight * (
                    1 - diversity_score
                )

                if combined_score > max_combined_score:
                    max_combined_score = combined_score
                    next_page = candidate

            if next_page:
                selected_pages.append(next_page)

        # Compute the average similarity of the selected pages
        avg_similarity = np.mean([page["similarity"] for page in selected_pages])

        # Store the results for the current paper
        paper_results.append(
            {
                "paper_id": paper_id,
                "selected_pages": selected_pages,
                "avg_similarity": avg_similarity,
            }
        )

    # Sort papers by their average similarity in descending order
    top_b_papers = sorted(paper_results, key=lambda x: x["avg_similarity"], reverse=True)[:b]

    # Construct the final results, one object per paper
    final_results = []
    for paper in top_b_papers:
        final_results.append(
            {
                "paper_id": paper["paper_id"],
                "text": [page["text"] for page in paper["selected_pages"]],
            }
        )

    return final_results
