import json
import os
import traceback
import uuid
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv

from debug import memory_service
from ext.app.ext_service import ExtMemoryService, ExtUserContext
from ext.llm.openai_azure_sdk import OpenAIAzureSDKClient
from ext.store.pg_repo import PgStore
from memu.app import DefaultUserModel

load_dotenv()

from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Request models for API endpoints
class ConversationMessage(BaseModel):
    role: str
    content: str


class MemorizeRequest(BaseModel):
    user_id: str
    conversation: List[ConversationMessage]


class RetrieveRequest(BaseModel):
    user_id: str
    query: str


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


app = FastAPI()
memory_service = init_memory_service()

storage_dir = Path(os.getenv("MEMU_STORAGE_DIR", "./data"))
storage_dir.mkdir(parents=True, exist_ok=True)


@app.post("/api/v1/memory/memorize")
async def memorize(request: MemorizeRequest):
    try:
        file_path = storage_dir / f"conversation-{uuid.uuid4().hex}.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump([msg.model_dump() for msg in request.conversation], f, ensure_ascii=False)

        user = DefaultUserModel(user_id=request.user_id)
        memory_service._contexts[f"DefaultUserModel:{user.user_id}"] = ExtUserContext(user_id=user.user_id,
                                                                                      categories_ready=False)
        result = await memory_service.memorize(resource_url=str(file_path), modality="conversation", user=user)
        await memory_service.summary_user_profile(user=user)
        summaries = memory_service.get_all_category_summaries(user=user)
        return JSONResponse(content={"status": "success", "result": summaries})
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/retrieve")
async def retrieve(payload: Dict[str, Any]):
    if "query" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body")
    try:
        result = await memory_service.retrieve([payload["query"]])
        return JSONResponse(content={"status": "success", "result": result})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/retrieve-item")
async def retrieve_item(retrieve_request: RetrieveRequest):
    try:
        qvec = (await memory_service.embedding_client.embed([retrieve_request.query]))[0]
        pg_store: PgStore = memory_service.store
        results = pg_store.retrieve_memory_items(qvec)
        resp = [
            {
                "id": r.id,
                "memory_type": r.memory_type,
                "summary": r.summary,
            }
            for r in results
        ]
        return JSONResponse(content={"status": "success", "result": resp})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/")
async def root():
    return {"message": "Hello MemU user!"}
