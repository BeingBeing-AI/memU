import logging
from typing import cast
from urllib.parse import parse_qs, urlparse, urlunparse

from openai import AsyncAzureOpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)


def _normalize_openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    parsed_url = urlparse(normalized)
    normalized_path = parsed_url.path.rstrip("/")
    endpoint_suffixes = (
        "/chat/completions",
        "/responses",
        "/embeddings",
        "/audio/transcriptions",
    )
    for suffix in endpoint_suffixes:
        if normalized_path.endswith(suffix):
            normalized_path = normalized_path[: -len(suffix)] or "/"
            return urlunparse(parsed_url._replace(path=normalized_path))
    return normalized


class OpenAIEmbeddingSDKClient:
    """OpenAI embedding client that relies on the official Python SDK."""

    def __init__(self, *, base_url: str, api_key: str, embed_model: str):
        self.base_url = _normalize_openai_base_url(base_url)
        self.api_key = api_key or ""
        self.embed_model = embed_model

        parsed_url = urlparse(self.base_url)
        api_version = parse_qs(parsed_url.query).get("api-version", [None])[0]
        if api_version:
            azure_endpoint = f"{parsed_url.scheme}://{parsed_url.netloc}"
            self.client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )
        else:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def embed(self, inputs: list[str]) -> list[list[float]]:
        """
        Create text embeddings.

        Args:
            inputs: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        response = await self.client.embeddings.create(model=self.embed_model, input=inputs)
        return [cast(list[float], d.embedding) for d in response.data]
