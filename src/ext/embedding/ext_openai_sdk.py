from typing import cast

from memu.embedding import OpenAIEmbeddingSDKClient


class ExtOpenAIEmbeddingSDKClient(OpenAIEmbeddingSDKClient):
    def __init__(self, *, base_url: str, api_key: str, embed_model: str):
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
            response = await self.client.embeddings.create(model=self.embed_model, input=batch)
            emb = [cast(list[float], d.embedding) for d in response.data]
            all_embeddings.extend(emb)
        return all_embeddings
