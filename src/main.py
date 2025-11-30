import json
import os
from pathlib import Path
import traceback
from typing import Any, Dict
import uuid

from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from memu.app import MemoryService

load_dotenv()

def init_memory_service():
    memory_service = MemoryService(
        llm_config={
            "client_backend": "sdk",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "api_key": os.getenv("ARK_API_KEY"),
            "chat_model": "ep-20251011204137-4wknv",
        },
        embedding_config={
            "client_backend": "sdk",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "api_key": os.getenv("ARK_API_KEY"),
            "embed_model": "doubao-embedding-text-240715",
            "provider": "doubao"
        },
        memorize_config={
            "category_summary_target_length": 300
        },
        retrieve_config={"method": "rag"}
    )

    from ext.llm.openai_azure_sdk import OpenAIAzureSDKClient
    memory_service.llm_client = OpenAIAzureSDKClient(
        azure_endpoint="https://gpt-5-10.openai.azure.com",
        api_key=os.getenv("GPT_API_KEY"),
        api_version="2025-01-01-preview",
        chat_model="gpt-5.1",
    )

    from ext.store.pg_repo import PgStore
    memory_service.store = PgStore(connection_string=os.getenv("PG_URL"))

    return memory_service

app = FastAPI()
service = init_memory_service()

storage_dir = Path(os.getenv("MEMU_STORAGE_DIR", "./data"))
storage_dir.mkdir(parents=True, exist_ok=True)

@app.post("/memorize")
async def memorize(payload: Dict[str, Any]):
    try:
        file_path = storage_dir / f"conversation-{uuid.uuid4().hex}.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        result = await service.memorize(resource_url=str(file_path), modality="conversation")
        return JSONResponse(content={"status": "success", "result": result})
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/retrieve")
async def retrieve(payload: Dict[str, Any]):
    if "query" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body")
    try:
        result = await service.retrieve([payload["query"]])
        return JSONResponse(content={"status": "success", "result": result})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/")
async def root():
    return {"message": "Hello MemU user!"}
