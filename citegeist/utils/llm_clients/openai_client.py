"""
OpenAI API client implementation.
"""

import requests

from .base_client import LLMClient, exponential_backoff_retry


class OpenAIClient(LLMClient):
    """Client for OpenAI API."""

    def __init__(self, api_key: str, model_name: str) -> None:
        """
        Initializes the OpenAI client.

        Args:
            api_key (str): API key.
            model_name (str): Name of the model.

        """
        self.api_key = api_key
        self.model_name = model_name

    def get_completion(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0, **kwargs) -> str:
        """
        Wraps prompt in a contents object, which is passed to get_chat_completions.

        Args:
            prompt (str): The prompt.
            max_tokens (int): The maximum number of tokens to return.
            temperature (float): The temperature to use.

        Returns:
            The completion.
        """
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        return self.get_chat_completion(messages, max_tokens=max_tokens, temperature=temperature, **kwargs)

    def get_chat_completion(
        self, messages: list[dict[str, str]], max_tokens: int = 4096, temperature: float = 0.0, **kwargs
    ) -> str:
        """
        Gets chat completions from OpenAI API based on the provided contents.

        Args:
            messages: The messages to get chat completions for.
            max_tokens (int): The maximum number of tokens to return.
            temperature (float): The temperature to use.

        Returns:
            The chat completion.
        """

        def call_model() -> str:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": max_tokens,
                "temperature": temperature,
            }

            request_url = "https://api.openai.com/v1/chat/completions"

            response = requests.post(url=request_url, headers=headers, json=payload)
            response.raise_for_status()
            reply = response.json()

            return reply["choices"][0]["message"]["content"]

        return exponential_backoff_retry(call_model)
