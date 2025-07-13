import logging
import random
import time

import google
import httpx
import vertexai
from google.auth.transport.requests import Request
# from google.cloud import aiplatform_v1beta1 as aiplatform
from google.oauth2 import service_account
from vertexai.preview.generative_models import (GenerationConfig,
                                                GenerativeModel)

# Google Cloud setup
scopes = ["https://www.googleapis.com/auth/cloud-platform"]
# Path to your service account key file
key_path = "/Users/carl/Downloads/stellar_depth_account_1.json"

# Create credentials using the service account key
credentials = service_account.Credentials.from_service_account_file(key_path, scopes=scopes)

# Initialize Google Cloud AI Platform
google.cloud.aiplatform.init(project="stellar-depth-441503-q5", location="us-central1", credentials=credentials)


def prompt_gemini(project_id: str, model_name: str, prompt: str, temperature: float = 0.0, max_tokens: int = 4096):
    """Prompt the Gemini model via Google Vertex AI."""
    vertexai.init(project=project_id, location="us-central1")
    model = GenerativeModel(model_name)

    response = model.generate_content(prompt, generation_config=GenerationConfig(temperature=temperature))

    return response.text


def prompt_gemini_with_backoff(project_id: str, model_name: str, prompt: str):
    """Wrap Gemini prompt with exponential backoff in case of rate limit."""

    def call_gemini():
        return prompt_gemini(project_id, model_name, prompt)

    return exponential_backoff_retry(call_gemini)


def exponential_backoff_retry(func, retries=10, backoff_factor=2, max_wait=120):
    """Retry with exponential backoff in case of rate limit error (429)."""
    wait = 1
    for attempt in range(retries):
        try:
            return func()  # Call the function
        except Exception as e:
            if "429" in str(e) or "529" in str(e):  # Rate limit errors
                logging.warning(f"Rate limit exceeded. Attempt {attempt + 1} of {retries}. Retrying in {wait} seconds.")
                time.sleep(wait + random.uniform(0, 1))  # Adding jitter to prevent thundering herd problem
                wait = min(wait * backoff_factor, max_wait)
            else:
                raise e
    logging.error("Exceeded maximum retries due to rate limit.")
    raise Exception("Exceeded maximum retries due to rate limit.")


def prompt_mistral_with_backoff(
    project_id: str,
    region: str,
    model_name: str,
    prompt: str,
    # credentials = credentials,
    retries: int = 10,
    backoff_factor: int = 2,
    max_wait: int = 120,
    temperature: float = 0.7,  # Default temperature
    streaming: bool = False,
    timeout: int = 60,
):
    """Prompt Mistral model with exponential backoff in case of rate limit error."""

    def call_mistral():
        return prompt_mistral_pre_authenticated(
            project_id,
            region,
            model_name,
            prompt,
            credentials,
            temperature=temperature,
            streaming=streaming,
            timeout=timeout,
        )

    return exponential_backoff_retry(call_mistral, retries, backoff_factor, max_wait)


def prompt_mistral_pre_authenticated(
    project_id: str,
    region: str,
    model_name: str,
    prompt: str,
    credentials,
    temperature: float = 0.7,  # Default temperature set to 0.7 (can be adjusted)
    streaming: bool = False,
    timeout: int = 60,
):
    """
    Prompt the Mistral Large model via Google Vertex AI API using pre-authenticated credentials.

    Args:
        project_id (str): Google Cloud project ID.
        region (str): Google Cloud region.
        model_name (str): Name of the Mistral model.
        prompt (str): Text prompt to send to the model.
        credentials: Pre-authenticated Google Cloud credentials object.
        temperature (float): Controls the randomness of the model's responses. Default is 0.7.
        streaming (bool): Whether to enable streaming responses. Default is False.
        timeout (int): HTTP request timeout in seconds. Default is 60.

    Returns:
        str: The response from the model.
    """
    # Ensure credentials are fresh
    request = Request()
    credentials.refresh(request)
    access_token = credentials.token

    # Build API URL
    url = build_endpoint_url(region=region, project_id=project_id, model_name=model_name, streaming=streaming)

    # Define request headers
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }

    # Define payload with temperature included
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": streaming,
        "temperature": temperature,  # Adding the temperature parameter here
    }

    # Make the HTTP POST request
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()["choices"][0]["message"]["content"] if not streaming else response.iter_lines()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")


def build_endpoint_url(region: str, project_id: str, model_name: str, streaming: bool = False):
    """
    Construct the endpoint URL for the Mistral Large model.

    Args:
        region (str): The region of the AI service.
        project_id (str): Google Cloud project ID.
        model_name (str): The name of the model (e.g., "mistral-large-2411").
        streaming (bool): Whether the request is streamed.

    Returns:
        str: Fully constructed URL for the API endpoint.
    """
    base_url = f"https://{region}-aiplatform.googleapis.com/v1/"
    project_fragment = f"projects/{project_id}"
    location_fragment = f"locations/{region}"
    specifier = "streamRawPredict" if streaming else "rawPredict"
    model_fragment = f"publishers/mistralai/models/{model_name}"
    return f"{base_url}{'/'.join([project_fragment, location_fragment, model_fragment])}:{specifier}"
