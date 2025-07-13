import random
import time

from bertopic import BERTopic
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# from citegeist.utils.azure_client import AzureClient
from citegeist.utils.citations import get_arxiv_abstract, get_arxiv_citation
from citegeist.utils.helpers import load_api_key
from citegeist.utils.llm_clients import create_client
from citegeist.utils.prompts import (generate_related_work_prompt,
                                     generate_summary_prompt)
import json
import google.generativeai as genai

AZURE_ENDPOINT = "cai-project"
AZURE_PROMPTING_MODEL = "gpt-4o"
AZURE_PROMPTING_MODEL_VERSION = "2024-08-01-preview"
KEY_LOCATION = "carl_config.json"

topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
client = MilvusClient("./database.db")
# prompting_client = AzureClient(
#     endpoint=AZURE_ENDPOINT,
#     deployment_id=AZURE_PROMPTING_MODEL,
#     api_key=load_api_key(KEY_LOCATION),
# )
llm_client = create_client(
    provider="azure",
    endpoint=AZURE_ENDPOINT,
    deployment_id=AZURE_PROMPTING_MODEL_VERSION,
    api_key=load_api_key(KEY_LOCATION),
)

def load_gemini_key():
    with open("Gemini_Key.json", "r") as file:
        data = json.load(file)
        return data["secret_key"]


genai.configure(api_key=load_gemini_key())


def prompt_gemini_2(model_name: str, prompt: str, temperature: float = 0.0, max_tokens: int = 4096):
    """Prompt the Gemini model via Google Vertex AI."""
    model = genai.GenerativeModel(model_name)

    generation_config = {
        "temperature": temperature,
    }

    response = model.generate_content(prompt, generation_config=generation_config)

    return response.text


def exponential_backoff_retry(func, retries=10, backoff_factor=2, max_wait=120):
    """Retry with exponential backoff in case of rate limit error (429)."""
    wait = 1
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if "429" in str(e) or "529" in str(e):
                print(f"Rate limit exceeded. Attempt {attempt + 1} of {retries}. Retrying in {wait} seconds.")
                time.sleep(wait + random.uniform(0, 1))  # Adding jitter
                wait = min(wait * backoff_factor, max_wait)
            else:
                raise e
    print("Exceeded maximum retries due to rate limit.")
    raise Exception("Exceeded maximum retries due to rate limit.")


def prompt_gemini_with_backoff(model_name: str, prompt: str):
    """Wrap Gemini prompt with exponential backoff in case of rate limit."""

    def call_gemini():
        return prompt_gemini_2(model_name, prompt)

    return exponential_backoff_retry(call_gemini)


abstract = ("Large Language Models have shown impressive per- formance across a wide array of tasks involving both"
            " structured and unstructured textual data. More recently, adaptions of these models have drawn attention"
            " to their abilities to work with code across different programming languages. On this notion, different"
            " benchmarks for code generation, repair, or completion suggest that certain models have programming"
            " abilities comparable to or even surpass humans. In this work, we demonstrate that the performance on this"
            " benchmark does not translate to the innate ability of humans to appreciate the structural control flow of"
            " code. For this purpose, we extract code solutions from the Hu- manEval benchmark, which the relevant"
            " models perform very strongly on, and trace their execution path using function calls sampled from the"
            " respective test set. Using this dataset, we investigate the ability of 5 state-of-the-art LLMs to match"
            " the execution trace and find that, despite the model’s abilities to generate semantically identical code,"
            " they possess only limited ability to trace the execution path, especially for traces with increased"
            " length. We find that even the top-performing model, Gemini 1.5 Pro can only fully correctly generate"
            " the trace of 47% of HumanEval tasks. In addition, we introduce a specific subset for three key structures"
            " not, or only contained to a limited extent in Hu- manEval: Recursion, Parallel Processing, and Object"
            " Oriented Programming principles, including concepts like Inheritance and Polymorphism. Besides OOP,"
            " we show that none of the investigated models achieve an average accuracy of over 5% on the relevant"
            " traces. Aggregating these specialized parts with the ubiquitous HumanEval tasks, we present the Benchmark"
            " CoCoNUT: Code Control Flow for Navigation Understanding and Testing, which measures a models ability to"
            " trace the execu- tion of code upon relevant calls, including advanced structural components. We conclude"
            " that the current generation LLMs still need to significantly improve to enhance their code reasoning"
            " abilities. We hope our dataset can help researchers bridge this gap in the near future.")
embedded_abstract = embedding_model.encode(abstract)
topic = topic_model.transform(abstract)
topic_id = topic[0][0]

res = client.search(
    collection_name="abstracts",
    data=[embedded_abstract],
    limit=10,
    # filter = f'topic == {topic_id}',
    search_params={"metric_type": "COSINE", "params": {}},
    # output_fields = []
)
formatted_res = json.dumps(res, indent=4)
print(formatted_res)

for obj in res:
    arxiv_id = obj["id"]
    arxiv_abstract = get_arxiv_abstract(arxiv_id)
    response: str = prompting_client.get_completions(
        generate_summary_prompt(abstract, arxiv_abstract), AZURE_PROMPTING_MODEL_VERSION
    )
    obj["summary"] = response
    obj["citation"] = get_arxiv_citation(arxiv_id)
res = res[0]

for obj in res:
    arxiv_id = obj["id"]
    arxiv_abstract = get_arxiv_abstract(arxiv_id)
    response: str = prompting_client.get_completions(
        generate_summary_prompt(abstract, arxiv_abstract), AZURE_PROMPTING_MODEL_VERSION
    )
    obj["summary"] = response
    obj["citation"] = get_arxiv_citation(arxiv_id)

response: str = prompting_client.get_completions(
    generate_related_work_prompt(abstract, res), AZURE_PROMPTING_MODEL_VERSION
)
print(response)
