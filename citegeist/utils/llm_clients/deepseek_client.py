"""
Gemini API client implementation.
"""

import requests
from openai import OpenAI
from .base_client import LLMClient, exponential_backoff_retry


class DeepSeekClient(LLMClient):
    """Client for DeepSeek API."""

    def __init__(self, api_key: str, model_name: str):
        """
        Initializes the DeepSeek client.

        Args:
            api_key (str): The API key.
            model_name (str): The model name.
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
        contents = [
            {"role":"user",
            "content":prompt}
            ]
        return self.get_chat_completion(contents, max_tokens, temperature, **kwargs)

    def get_chat_completion(
        self, contents: list[dict[str, list[dict]]], max_tokens: int = 4096, temperature: float = 0.0, **kwargs
    ) -> str:
        """
        Gets chat completions from Gemini API based on the provided contents.

        Args:
            contents: The contents to get chat completions from.
            max_tokens (int): The maximum number of tokens to return.
            temperature (float): The temperature to use.

        Returns:
            The chat completion.
        """

        def call_model() -> str:
            
            client = OpenAI(
                api_key=self.api_key,
                base_url = "https://api.deepseek.com/v1"
            )
            
            response = client.chat.completions.create(
                model = self.model_name,
                messages = contents,
                temperature = temperature,
                max_tokens = max_tokens
            )
            
            # headers = {"Content-Type": "application/json"}

            # payload = {
            #     "contents": contents,
            #     "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens, **kwargs},
            # }

            # request_url = (
            #     f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}"
            #     f":generateContent?key={self.api_key}"
            # )

            # response = requests.post(url=request_url, headers=headers, json=payload)
            # response.raise_for_status()
            # reply = response.json()

            return response.choices[0].message.content

        return exponential_backoff_retry(call_model)