import logging
from typing import cast
from urllib.parse import parse_qs, urlparse

from openai import AsyncAzureOpenAI

from ext.ext_config import VECTOR_DIMENSION
from memu.embedding import OpenAIEmbeddingSDKClient

logger = logging.getLogger(__file__)
MAX_EMBED_QUERY_CHARS = 8000


class ExtOpenAIEmbeddingSDKClient(OpenAIEmbeddingSDKClient):
    def __init__(self, *, base_url: str, api_key: str, embed_model: str):
        parsed_url = urlparse(base_url)
        api_version = parse_qs(parsed_url.query).get("api-version", [None])[0]

        # When api-version is present in the base_url, treat it as an Azure endpoint.
        if api_version:
            self.base_url = base_url.rstrip("/")
            self.api_key = api_key or ""
            self.embed_model = embed_model
            azure_endpoint = f"{parsed_url.scheme}://{parsed_url.netloc}"
            self.client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )
        else:
            super().__init__(base_url=base_url, api_key=api_key, embed_model=embed_model)

    async def embed(self, inputs: list[str], batch_size: int = 10) -> list[list[float]]:
        """
        Create text embeddings.

        Args:
            inputs: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            sanitized_batch = []
            for text in batch:
                trimmed = text.strip()
                if not trimmed:
                    msg = "Embedding input must not be empty after trimming"
                    logger.warning(msg)
                    raise ValueError(msg)
                if len(trimmed) > MAX_EMBED_QUERY_CHARS:
                    logger.info(
                        "Truncating embedding input: original_len=%d, max_len=%d",
                        len(trimmed),
                        MAX_EMBED_QUERY_CHARS,
                    )
                    trimmed = trimmed[:MAX_EMBED_QUERY_CHARS]
                sanitized_batch.append(trimmed)

            response = await self.client.embeddings.create(model=self.embed_model, input=sanitized_batch, dimensions=VECTOR_DIMENSION)
            emb = [cast(list[float], d.embedding) for d in response.data]
            all_embeddings.extend(emb)
        return all_embeddings
