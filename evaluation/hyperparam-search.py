import statistics
from citegeist import Generator
from citegeist.utils.llm_clients import LLMClient, AzureClient
from citegeist.utils.citations import get_arxiv_abstract
import re
import os

BREADTH_VALUES = [5, 10, 15, 20]
DEPTH_VALUES = [1, 2, 3, 4, 5]
DIVERSITY_VALUES = [0.33, 0.66, 1.0]

DEFAULT_BREADTH = 10
DEFAULT_DEPTH = 2
DEFAULT_DIVERSITY = 0.0

OUTPUT_FILE = "/Users/carl/PycharmProjects/citegeist/evaluation/out/hyperparam-search-3.csv"
RESULTS = []

source_abstract = f"""\
The Transformer architecture is widely regarded as the most powerful tool for natural language processing, but due to the high number of complex operations, it inherently faces the issue of extensive energy consumption. To address this issue, we consider spiking neural networks (SNN), an energy-efficient alternative to common artificial neural networks (ANN) due to their naturally event-driven way of processing information. However, this inherently makes them difficult to train, which is why many SNN-related models circumvent this issue through the conversion of pre-trained ANN networks. More recently, attempts have been made to create directly trained SNN-based adaptions of the Transformer model structure. While the results showed great promise, their sole application field was computer vision and based on incorporating encoder blocks. In this paper, we propose SpikeDecoder, a fully spike-based low-power version of the Transformer decoder-only model, for application on the field of natural language processing. We further analyze the impact of exchanging different blocks of the ANN model with their spike-based alternatives to identify pain points and significant sources of performance loss. Similarly, we extend our investigation to the role of residual connections and the selection of spike-compatible normalization techniques. Besides the work on the model architecture, we formulate and compare different embedding methods to project text data into spike-range. Finally, it will be demonstrated that the spiking decoder block reduces the theoretical energy consumption by 87 to 93 percent compared to the power required for a regular encoder block.
"""

def generate_relevance_evaluation_prompt(source_abstract: str, target_abstract: str) -> str:
    prompt = f"""
        You are given two paper abstracts: the first is the source paper abstract, and the second is a related work paper abstract. Your task is to assess the relevance of the related work abstract to the source paper abstract on a scale of 0 to 10, where:

        - 0 means no relevance at all (completely unrelated).
        - 10 means the highest relevance (directly related and closely aligned with the source paper's topic and content).

        Consider factors such as:
        - Topic alignment: Does the related work paper address a similar research problem or area as the source paper?
        - Methodology: Does the related work discuss methods or techniques similar to those in the source paper?
        - Findings or contributions: Are the findings or contributions of the related work closely related to the source paper's content or conclusions?
        - The relationship between the two papers, such as whether the related work builds on, contrasts, or expands the source paper's work.

        Provide a score (0–10) and a brief explanation of your reasoning for the assigned score.

        Source Paper Abstract:
        {source_abstract}

        Related Work Paper Abstract:
        {target_abstract}

        Please provide only the score as your reply. Do not produce any other output, including things like formatting or markdown. Only the score.
    """
    return prompt

def evaluate_paper_relevance(client: LLMClient, source_abstract: str, target_abstract: str) -> float:
    prompt = generate_relevance_evaluation_prompt(source_abstract, target_abstract)
    response = client.get_completion(prompt)

    # Try to extract a numeric score from the response
    try:
        # Look for a number between 0 and 10, possibly with decimal point
        match = re.search(r'\b([0-9]|10)(\.\d+)?\b', response)
        if match:
            score = float(match.group(0))
            return score
        else:
            print(f"WARNING: Failed to extract score from client response: {response}")
            return 0.0
    except Exception as e:
        print(f"ERROR extracting score: {e}")
        return 0.0

def generate_related_work_score_prompt(source_abstract: str, related_work: str) -> str:
    return f"""
    Source Abstract:
    {source_abstract}

    Related Works Section:
    {related_work}

    Objective:
    Evaluate this related works section with regard to the source abstract provided.

    Consider factors such as comprehensiveness, clarity of writing, relevance, etc. when making your decision.
    If invalid citations occur, consider the information to be invalid (or even completely false).

    Exclusively respond with your choice of rating. For this purpose you can assign a score from 0-10 where 0 is worst and 10 is best.

    - **0**: Completely irrelevant, unclear, or inaccurate. 
     *Example*: The section does not address the Source Abstract's topics and contains multiple invalid citations.

    - **5**: Somewhat relevant but lacks comprehensiveness, clarity or relevance.
     *Example*: The section references a few relevant works but also includes irrelevant ones and has minor errors.

    - **10**: Exceptionally relevant, comprehensive, clear, and accurate.
      *Example*: The section thoroughly addresses all key topics, includes all relevant works, and is clearly written with no factual errors.

    Do not include anything else in your output.
    """

def evaluate_related_work_section(client: LLMClient, source_abstract: str, related_work: str) -> float:
    prompt = generate_related_work_score_prompt(source_abstract, related_work)

    # Use Gemini for evaluation
    response = client.get_completion(prompt)

    # Try to extract a numeric score from the response
    try:
        # Look for a number between 0 and 10, possibly with decimal point
        match = re.search(r'\b([0-9]|10)(\.\d+)?\b', response)
        if match:
            score = float(match.group(0))
            return score
        else:
            print(f"WARNING: Failed to extract score from client response: {response}")
            return 0.0
    except Exception as e:
        print(f"ERROR extracting score: {e}")
        return 0.0

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

client: LLMClient = AzureClient(
    api_key=os.getenv("AZURE_API_KEY"),
    endpoint="https://cai-project.openai.azure.com",
    deployment_id="gpt-4o",
    api_version="2024-10-21",
)

generator: Generator = Generator(
    llm_provider="azure",
    database_uri=os.environ.get("MILVUS_URI"),
    database_token=os.environ.get("MILVUS_TOKEN"),
    api_key=os.environ.get("AZURE_API_KEY"),
    endpoint="https://cai-project.openai.azure.com",
    deployment_id="gpt-4o",
    api_version="2024-10-21",
)

# Vary breadth
print("Varying breadth...")
for breadth in BREADTH_VALUES:
    result = generator.generate_related_work(abstract=source_abstract, breadth=breadth, depth=DEFAULT_DEPTH, diversity=DEFAULT_DIVERSITY)
    related_works: str = result["related_works"]
    citations: list[str] = result["citations"]

    # Calculate eval results
    quality_score = evaluate_related_work_section(client, source_abstract, result["related_works"])
    relevance_scores = []
    for citation in citations:
        try:
            match = re.search(r"(?<=arXiv:)\d+\.[\d|\w]+(?=\. )", citation)
            arxiv_id = match.group(0)
            paper_abstract = get_arxiv_abstract(arxiv_id)
            if not paper_abstract.startswith("No paper found"):
                relevance_score = evaluate_paper_relevance(client, source_abstract, paper_abstract)
                relevance_scores.append(relevance_score)
            else:
                print(f"ERROR finding abstract for {arxiv_id}: {e}")
                relevance_scores.append(0.0)
        except Exception as e:
            print(f"ERROR extracting arxiv id: {e}")
            relevance_scores.append(0.0)

    avg_relevance_score = statistics.mean(relevance_scores)
    std_relevance_score = statistics.stdev(relevance_scores) if len(relevance_scores) > 1 else 0.0
    relevance_score_string = "|".join([str(f) for f in relevance_scores])
    # Store eval results
    RESULTS.append({
        "breadth": breadth,
        "depth": DEFAULT_DEPTH,
        "diversity": DEFAULT_DIVERSITY,
        "quality_score": quality_score,
        "avg_relevance_score": avg_relevance_score,
        "std_relevance_score": std_relevance_score,
        "relevance_scores": relevance_score_string,
    })
    print(f"Breadth: {breadth}, Quality Score: {quality_score}, Average Relevance Score: {avg_relevance_score}, Standard Deviation: {std_relevance_score}, Relevance Scores: {relevance_score_string}")

print("Completed breadth experiments. Saving results...")
# intermediate save of results
with open(OUTPUT_FILE, "w") as f:
    f.write("breadth,depth,diversity,quality_score,avg_relevance_score,std_relevance_score,relevance_scores\n")
    for result in RESULTS:
        f.write(f"{result['breadth']},{result['depth']},{result['diversity']},{result['quality_score']},{result['avg_relevance_score']},{result['std_relevance_score']},{result['relevance_scores']}\n")
print("Intermediate results saved.")

# Vary depth
print("Varying depth...")
for depth in DEPTH_VALUES:
    result = generator.generate_related_work(abstract=source_abstract, breadth=DEFAULT_BREADTH, depth=depth, diversity=DEFAULT_DIVERSITY)
    related_works: str = result["related_works"]
    citations: list[str] = result["citations"]

    # Calculate eval results
    quality_score = evaluate_related_work_section(client, source_abstract, result["related_works"])
    relevance_scores = []
    for citation in citations:
        try:
            match = re.search(r"(?<=arXiv:)\d+\.[\d|\w]+(?=\. )", citation)
            arxiv_id = match.group(0)
            paper_abstract = get_arxiv_abstract(arxiv_id)
            if not paper_abstract.startswith("No paper found"):
                relevance_score = evaluate_paper_relevance(client, source_abstract, paper_abstract)
                relevance_scores.append(relevance_score)
            else:
                print(f"ERROR finding abstract for {arxiv_id}: {e}")
                relevance_scores.append(0.0)
        except Exception as e:
            print(f"ERROR extracting arxiv id: {e}")
            relevance_scores.append(0.0)

    avg_relevance_score = statistics.mean(relevance_scores)
    std_relevance_score = statistics.stdev(relevance_scores) if len(relevance_scores) > 1 else 0.0
    relevance_score_string = "|".join([str(f) for f in relevance_scores])
    # Store eval results
    RESULTS.append({
        "breadth": DEFAULT_BREADTH,
        "depth": depth,
        "diversity": DEFAULT_DIVERSITY,
        "quality_score": quality_score,
        "avg_relevance_score": avg_relevance_score,
        "std_relevance_score": std_relevance_score,
        "relevance_scores": relevance_score_string,
    })
    print(f"Depth: {depth}, Quality Score: {quality_score}, Average Relevance Score: {avg_relevance_score}, Standard Deviation: {std_relevance_score}, Relevance Scores: {relevance_score_string}")

print("Completed depth experiments. Saving results...")
with open(OUTPUT_FILE, "w") as f:
    f.write("breadth,depth,diversity,quality_score,avg_relevance_score,std_relevance_score,relevance_scores\n")
    for result in RESULTS:
        f.write(f"{result['breadth']},{result['depth']},{result['diversity']},{result['quality_score']},{result['avg_relevance_score']},{result['std_relevance_score']},{result['relevance_scores']}\n")

print("Intermediate results saved.")

# Vary diversity
print("Varying diversity...")
for diversity in DIVERSITY_VALUES:
    result = generator.generate_related_work(abstract=source_abstract, breadth=DEFAULT_BREADTH, depth=DEFAULT_DEPTH, diversity=diversity)
    related_works: str = result["related_works"]
    citations: list[str] = result["citations"]

    # Calculate eval results
    quality_score = evaluate_related_work_section(client, source_abstract, result["related_works"])
    relevance_scores = []
    for citation in citations:
        try:
            match = re.search(r"(?<=arXiv:)\d+\.[\d|\w]+(?=\. )", citation)
            arxiv_id = match.group(0)
            paper_abstract = get_arxiv_abstract(arxiv_id)
            if not paper_abstract.startswith("No paper found"):
                relevance_score = evaluate_paper_relevance(client, source_abstract, paper_abstract)
                relevance_scores.append(relevance_score)
            else:
                print(f"ERROR finding abstract for {arxiv_id}: {e}")
                relevance_scores.append(0.0)
        except Exception as e:
            print(f"ERROR extracting arxiv id: {e}")
            relevance_scores.append(0.0)

    avg_relevance_score = statistics.mean(relevance_scores)
    std_relevance_score = statistics.stdev(relevance_scores) if len(relevance_scores) > 1 else 0.0
    relevance_score_string = "|".join([str(f) for f in relevance_scores])
    # Store eval results
    RESULTS.append({
        "breadth": DEFAULT_BREADTH,
        "depth": DEFAULT_DEPTH,
        "diversity": diversity,
        "quality_score": quality_score,
        "avg_relevance_score": avg_relevance_score,
        "std_relevance_score": std_relevance_score,
        "relevance_scores": relevance_score_string,
    })
    print(f"Diversity: {diversity}, Quality Score: {quality_score}, Average Relevance Score: {avg_relevance_score}, Standard Deviation: {std_relevance_score}, Relevance Scores: {relevance_score_string}")

print("Completed diversity experiments. Saving results...")
with open(OUTPUT_FILE, "w") as f:
    f.write("breadth,depth,diversity,quality_score,avg_relevance_score,std_relevance_score,relevance_scores\n")
    for result in RESULTS:
        f.write(f"{result['breadth']},{result['depth']},{result['diversity']},{result['quality_score']},{result['avg_relevance_score']},{result['std_relevance_score']},{result['relevance_scores']}\n")

print("Final results saved.")



