"""
Anthropic API client implementation.
"""

import requests

from .base_client import LLMClient, exponential_backoff_retry


class AnthropicClient(LLMClient):
    """Client for Anthropic API."""

    def __init__(self, api_key: str, model_name: str, api_version: str):
        """
        Initialize the OpenAI client.

        Args:
            api_key (str): API key.
            model_name (str): Name of the model.
            api_version (str): API version (equivalent to anthropic-version).

        """
        self.api_key = api_key
        self.model_name = model_name
        self.api_version = api_version

    def get_completion(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0, **kwargs) -> str:
        """
        Wraps prompt in a contents object, which is passed to get_chat_completions.

        Args:
            prompt (str): Prompt string.
            max_tokens (int): Maximum number of tokens to return.
            temperature (float): Temperature to use.

        Returns:
            The completion.
        """
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        return self.get_chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs)

    def get_chat_completion(
        self, messages: list[dict[str, str]], max_tokens: int = 4096, temperature: float = 0.0, **kwargs
    ) -> str:
        """
        Gets chat completions from Anthropic API based on the provided set of messages.

        Args:
            messages (list[dict[str, str]]): List of messages to be used for the completion.
            max_tokens (int): Maximum number of tokens to return.
            temperature (float): Temperature to use.

        Returns:
            The chat completion.
        """

        def call_model() -> str:
            headers = {
                "Content-Type": "application/json",
                "anthropic-version": self.api_version,
                "x-api-key": self.api_key,
            }

            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            }

            request_url = "https://api.anthropic.com/v1/messages"

            response = requests.post(url=request_url, headers=headers, json=payload)
            response.raise_for_status()
            reply = response.json()

            return reply["content"][0]["text"]

        return exponential_backoff_retry(call_model)
