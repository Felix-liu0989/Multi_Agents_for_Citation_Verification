"""
Base client for LLM APIs.
"""

import abc
from typing import Dict, List


class LLMClient(abc.ABC):
    """Abstract base class for LLM API clients."""

    @abc.abstractmethod
    def get_completion(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0, **kwargs) -> str:
        """
        Gets completions from the LLM based on the provided prompt.

        Args:
            prompt: The input prompt to send to the LLM
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            **kwargs: Additional model-specific parameters

        Returns:
            The completion text from the LLM
        """
        pass

    @abc.abstractmethod
    def get_chat_completion(
        self, messages: List[Dict[str, str]], max_tokens: int = 4096, temperature: float = 0.0, **kwargs
    ) -> str:
        """
        Gets chat completions based on a conversation history.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            **kwargs: Additional model-specific parameters

        Returns:
            The completion text from the LLM
        """
        pass


def exponential_backoff_retry(func, retries: int = 10, backoff_factor: int = 2, max_wait: int = 120):
    """
    Retries a function with exponential backoff on failure.

    Args:
        func: The function to retry
        retries: Maximum number of retries
        backoff_factor: Factor to increase wait time by after each failure
        max_wait: Maximum wait time between retries

    Returns:
        The result of the function

    Raises:
        Exception: If all retries fail
    """
    import logging
    import random
    import time

    wait = 1
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if "429" in str(e) or "529" in str(e) or "rate limit" in str(e).lower():
                logging.warning(f"Rate limit exceeded. Attempt {attempt + 1} of {retries}. Retrying in {wait} seconds.")
                # Time with additional jitter
                time.sleep(wait + random.uniform(0, 1))
                wait = min(wait * backoff_factor, max_wait)
            else:
                raise e

    print("Exceeded maximum retries due to rate limit.")
    raise Exception("Exceeded maximum retries due to rate limit.")
