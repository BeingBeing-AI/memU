import json
import logging
import os

from dotenv import load_dotenv

from ext.app.ext_service import ExtUserContext, ExtMemoryService
from ext.prompts.summary_profile import PROMPT

load_dotenv()

from ext.llm.openai_azure_sdk import OpenAIAzureSDKClient
from memu.app import DefaultUserModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)


def init_memory_service():
    memory_service = ExtMemoryService(
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

    return memory_service

memory_service = init_memory_service()

async def test_memorize(user_id):
    user = DefaultUserModel(user_id=user_id)
    memory_service._contexts[f"DefaultUserModel:{user.user_id}"] = ExtUserContext(user_id=user.user_id, categories_ready=False)
    # Memorize
    for i in range(0, 2):
        file_path = os.path.abspath(f"../data/{user_id}/session_{i}.json")
        print(f"Memorizing {file_path}...")
        memory = await memory_service.memorize(resource_url=file_path, modality="conversation", user=user)
        for cat in memory.get('categories', []):
            print(f"  - {cat.get('name')}: {(cat.get('summary') or '')}")

    result = json.dumps(memory, indent=2, ensure_ascii=False)
    print(f"Final memory: \n {result}")
    # await memory_service.summary_user_profile(user)


async def summary_categories(user_id):
    await memory_service.summary_user_profile(user=DefaultUserModel(user_id=user_id))


async def test_retrieve():
    queries = [
        {"role": "user", "content": {"text": "工作地点在哪里"}},
    ]

    result_rag = await memory_service.retrieve(queries=queries)
    for item in result_rag.get('items', [])[:3]:
        print(f"  - [{item.get('memory_type')}] {item.get('summary', '')[:100]}...")

async def test_custom_retrieve():
    query = "今天要去见新的投资人"
    # query = "把把胡今天生病了"
    result = await memory_service.retrieve_memory_items(user=DefaultUserModel(user_id="cobe"), query=query,
                                                        retrieve_type="light")
    for r in result:
        print(f"  - [{r.memory_type}] {r.summary}")

async def main():
    # await test_memorize("cobe")
    # await test_custom_retrieve()
    await summary_categories("cobe")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
