import json
import os
import time

from dotenv import load_dotenv

from ext.prompts.summary_profile import PROMPT

load_dotenv()

from ext.llm.openai_azure_sdk import OpenAIAzureSDKClient
from ext.store.pg_repo import PgStore
from memu.app import MemoryService


def init_memory_service():
    memory_service = MemoryService(
        llm_config={
            "client_backend": "sdk",
            "base_url": "",
            "api_key": "",
            "chat_model": "",
        },
        embedding_config={
            "client_backend": "sdk",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.getenv("QWEN_API_KEY"),
            "embed_model": "text-embedding-v4",
        },
        # embedding_config={
        #     "client_backend": "sdk",
        #     "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        #     "api_key": os.getenv("ARK_API_KEY"),
        #     "embed_model": "doubao-embedding-text-240715",
        #     "provider": "doubao"
        # },
        memorize_config={
            "category_summary_target_length": 300
        },
        retrieve_config={"method": "rag"}
    )

    memory_service.llm_client = OpenAIAzureSDKClient(
        azure_endpoint="https://gpt-5-10.openai.azure.com",
        api_key=os.getenv("GPT_API_KEY"),
        api_version="2025-01-01-preview",
        chat_model="gpt-5.1",
    )

    memory_service.store = PgStore(connection_string="postgresql://root:dev123@localhost:5432/starfy")

    memory_service._categories_ready = False
    memory_service._ensure_categories_ready()

    return memory_service

memory_service = init_memory_service()

async def test_memorize():
    # Memorize
    for i in range(0, 24):
        file_path = os.path.abspath(f"../data/silvia/session_{i}.json")
        print(f"Memorizing {file_path}...")
        memory = await memory_service.memorize(resource_url=file_path, modality="conversation")
        for cat in memory.get('categories', []):
            print(f"  - {cat.get('name')}: {(cat.get('summary') or '')}")

    result = json.dumps(memory, indent=2, ensure_ascii=False)
    print(f"Final memory: \n {result}")


async def summary_categories():
    categories = memory_service.store.get_all_categories()
    formated = [
        {
            "name": cat.name,
            "summary": cat.summary,
        }
        for cat in categories
    ]
    response = await memory_service.llm_client.summarize(system_prompt=PROMPT, text=json.dumps(formated, indent=2))
    print(response)


async def test_retrieve():
    queries = [
        {"role": "user", "content": {"text": "工作地点在哪里"}},
    ]

    result_rag = await memory_service.retrieve(queries=queries)
    for item in result_rag.get('items', [])[:3]:
        print(f"  - [{item.get('memory_type')}] {item.get('summary', '')[:100]}...")

async def test_custom_retrieve():
    query = ""
    qvec = (await memory_service.embedding_client.embed([query]))[0]
    pg_store: PgStore = memory_service.store
    start_time = time.time()
    # results = pg_store.retrieve_memory_categories(qvec)
    # print(f"Time: {time.time() - start_time:.2f}s")
    # [print(f"{r.name}: {r.summary}\n") for r in results if r]

    # results = await  memory_service._rank_categories_by_summary(qvec, top_k=5)
    start_time = time.time()
    results = pg_store.retrieve_memory_items(qvec, min_similarity=0.3)
    print(f"Time: {time.time() - start_time:.2f}s")
    [print(f"{r.memory_type}: {r.summary}\n") for r in results if r]

    # results = pg_store.retrieve_memory_categories(qvec)
    # print([f"{r.name}: {r.summary}" for r in results if r])


async def main():
    # await test_memorize()
    # await test_custom_retrieve()
    await summary_categories()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
