from memu.embedding.openai_sdk import OpenAIEmbeddingSDKClient
from memu.llm import openai_sdk as llm_openai_sdk
from memu.llm.openai_sdk import OpenAISDKClient


class _FakeAsyncOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeAsyncAzureOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_llm_sdk_uses_azure_client_when_api_version_present(monkeypatch):
    monkeypatch.setattr(llm_openai_sdk, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr(llm_openai_sdk, "AsyncAzureOpenAI", _FakeAsyncAzureOpenAI)

    client = OpenAISDKClient(
        base_url="https://starfy-llm-jpe-resource.openai.azure.com/openai/v1/chat/completions?api-version=preview",
        api_key="test-key",
        chat_model="gpt-4.1",
    )

    assert isinstance(client.client, _FakeAsyncAzureOpenAI)
    assert client.client.kwargs == {
        "api_key": "test-key",
        "azure_endpoint": "https://starfy-llm-jpe-resource.openai.azure.com",
        "api_version": "preview",
    }


def test_llm_sdk_uses_standard_client_without_api_version(monkeypatch):
    monkeypatch.setattr(llm_openai_sdk, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr(llm_openai_sdk, "AsyncAzureOpenAI", _FakeAsyncAzureOpenAI)

    client = OpenAISDKClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        chat_model="gpt-4.1",
    )

    assert isinstance(client.client, _FakeAsyncOpenAI)
    assert client.client.kwargs == {
        "api_key": "test-key",
        "base_url": "https://api.openai.com/v1",
    }


def test_llm_sdk_normalizes_terminal_chat_completions_base_url(monkeypatch):
    monkeypatch.setattr(llm_openai_sdk, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr(llm_openai_sdk, "AsyncAzureOpenAI", _FakeAsyncAzureOpenAI)

    client = OpenAISDKClient(
        base_url="https://starfy-llm-jpe-resource.openai.azure.com/openai/v1/chat/completions",
        api_key="test-key",
        chat_model="gpt-5.1",
    )

    assert isinstance(client.client, _FakeAsyncOpenAI)
    assert client.client.kwargs == {
        "api_key": "test-key",
        "base_url": "https://starfy-llm-jpe-resource.openai.azure.com/openai/v1",
    }


def test_embedding_sdk_uses_azure_client_when_api_version_present(monkeypatch):
    from memu.embedding import openai_sdk as embedding_openai_sdk

    monkeypatch.setattr(embedding_openai_sdk, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr(embedding_openai_sdk, "AsyncAzureOpenAI", _FakeAsyncAzureOpenAI)

    client = OpenAIEmbeddingSDKClient(
        base_url="https://starfy-llm-jpe-resource.openai.azure.com/openai/v1/embeddings?api-version=2024-10-21",
        api_key="test-key",
        embed_model="text-embedding-3-large",
    )

    assert isinstance(client.client, _FakeAsyncAzureOpenAI)
    assert client.client.kwargs == {
        "api_key": "test-key",
        "azure_endpoint": "https://starfy-llm-jpe-resource.openai.azure.com",
        "api_version": "2024-10-21",
    }


def test_embedding_sdk_normalizes_terminal_embeddings_base_url(monkeypatch):
    from memu.embedding import openai_sdk as embedding_openai_sdk

    monkeypatch.setattr(embedding_openai_sdk, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr(embedding_openai_sdk, "AsyncAzureOpenAI", _FakeAsyncAzureOpenAI)

    client = OpenAIEmbeddingSDKClient(
        base_url="https://starfy-llm-jpe-resource.openai.azure.com/openai/v1/embeddings",
        api_key="test-key",
        embed_model="text-embedding-3-large",
    )

    assert isinstance(client.client, _FakeAsyncOpenAI)
    assert client.client.kwargs == {
        "api_key": "test-key",
        "base_url": "https://starfy-llm-jpe-resource.openai.azure.com/openai/v1",
    }
