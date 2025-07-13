"""
LLM API Client Module

This module provides a collection of clients for interacting with different LLM APIs.
"""

from .anthropic_client import AnthropicClient
from .azure_client import AzureClient
from .base_client import LLMClient
from .gemini_client import GeminiClient
from .mistral_client import MistralClient
from .openai_client import OpenAIClient


# Factory function to create appropriate client
def create_client(provider: str, **kwargs) -> LLMClient:
    """
    Factory function to create an LLM client based on the provider.

    Args:
        provider: The LLM provider name ('azure', 'openai', 'anthropic')
        **kwargs: Provider-specific configuration arguments

    Returns:
        An instance of the appropriate LLM client

    Raises:
        ValueError: If the provider is not supported
    """
    provider = provider.lower()

    if provider == "azure":
        return AzureClient(**kwargs)
    elif provider == "gemini":
        return GeminiClient(**kwargs)
    elif provider == "mistral":
        return MistralClient(**kwargs)
    elif provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider == "openai":
        return AnthropicClient(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
