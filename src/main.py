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
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel


# Request models for API endpoints
class ConversationMessage(BaseModel):
    role: str
    content: str


class MemorizeRequest(BaseModel):
    user_id: str
    external_id: str
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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # 记录异常日志
    traceback.print_exc()

    # 返回统一的错误响应
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error occurred",
            "detail": str(exc)
        }
    )


storage_dir = Path(os.getenv("MEMU_STORAGE_DIR", "./data"))
storage_dir.mkdir(parents=True, exist_ok=True)


@app.post("/api/v1/memory/memorize")
async def memorize(request: MemorizeRequest):
    user = DefaultUserModel(user_id=request.user_id)
    external_id = request.external_id
    file_path = storage_dir / external_id
    resource_url = str(file_path)
    # 幂等处理
    resource = memory_service.get_resource_by_url(user, resource_url)
    if resource:
        return JSONResponse(content={"status": resource.process_status})

    with file_path.open("w", encoding="utf-8") as f:
        json.dump([msg.model_dump() for msg in request.conversation], f, ensure_ascii=False)

    await memory_service.memorize(resource_url=resource_url, modality="conversation", user=user)
    memory_service.on_memorize_done(user, resource_url)
    await memory_service.summary_user_profile(user=user)
    summaries = memory_service.get_all_category_summaries(user=user)
    return JSONResponse(content={"status": "success", "result": summaries})


@app.post("/api/v1/memory/retrieve-category-summary")
async def retrieve_category_summary(request: RetrieveRequest):
    user = DefaultUserModel(user_id=request.user_id)
    return memory_service.get_category_summary(user=user, category_name=request.query)


@app.post("/retrieve")
async def retrieve(payload: Dict[str, Any]):
    if "query" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body")
    result = await memory_service.retrieve([payload["query"]])
    return JSONResponse(content={"status": "success", "result": result})


@app.post("/api/v1/memory/retrieve-item")
async def retrieve_item(retrieve_request: RetrieveRequest):
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


@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"})


@app.get("/")
async def root():
    return {"message": "Hello MemU user!"}
