import asyncio
import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

from debug import memory_service
from ext.app.ext_service import ExtMemoryService
from ext.ext_models import MemorizeRequest, RetrieveRequest, MultiRetrieveRequest, WeightedQuery
from ext.llm.openai_azure_sdk import OpenAIAzureSDKClient
from memu.app import DefaultUserModel

load_dotenv()

from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError


def configure_logging() -> None:
    """Configure logging to emit both to stdout and a rotating file for log collection."""
    log_dir = Path(os.getenv("MEMU_LOG_DIR", "/var/log/memu"))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / os.getenv("MEMU_LOG_FILENAME", "app.log")
    log_level_name = os.getenv("MEMU_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5),
    ]

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


configure_logging()
logger = logging.getLogger(__file__)


def init_memory_service():
    memory_service = ExtMemoryService(
        llm_config={
            "client_backend": "sdk",
            "base_url": os.getenv("OPENAI_BASE_URL"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "chat_model": os.getenv("OPENAI_MODEL_NAME"),
        },
        embedding_config={
            "client_backend": "sdk",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.getenv("QWEN_API_KEY"),
            "embed_model": "text-embedding-v4",
        },
        memorize_config={
            "category_summary_target_length": 300
        },
        retrieve_config={"method": "rag"}
    )

    return memory_service


app = FastAPI(strict_validation=False)
memory_service = init_memory_service()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log validation errors with the original payload for debugging
    try:
        body = await request.body()
    except Exception:  # pragma: no cover - defensive logging
        body = b"<unable to read body>"

    logger.warning(
        "Validation error for %s %s: %s | body=%s",
        request.method,
        request.url,
        exc.errors(),
        body.decode("utf-8", errors="replace"),
    )
    return JSONResponse(
        status_code=422,
        content={"status": "error", "message": "Request validation failed", "detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # 记录异常日志
    logger.exception(f"Failed to handle request {request.method} {request.url}")

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
    logger.info(f"memorize, request: {request}")
    user = DefaultUserModel(user_id=request.user_id)
    external_id = request.external_id
    file_path = storage_dir / external_id
    resource_url = str(file_path)
    # 幂等处理
    resource = memory_service.get_resource_by_url(user, resource_url)
    if resource:
        logger.info(f"Resource {resource.url} already exists, skipping memorization")
        return JSONResponse(content={"status": resource.process_status})

    with file_path.open("w", encoding="utf-8") as f:
        json.dump([msg.model_dump() for msg in request.conversation], f, ensure_ascii=False)

    await memory_service.memorize(resource_url=resource_url, modality="conversation", user=user)
    memory_service.on_memorize_done(user, resource_url)
    if request.summary_user_profile:
        await memory_service.summary_user_profile(user=user)
    summaries = memory_service.get_all_category_summaries(user=user)
    return JSONResponse(content={"status": "SUCCESS", "detail": summaries})


@app.post("/api/v1/memory/summary-user-profile")
async def summary_user_profile(user_id: str):
    logger.info(f"summary_user_profile, user_id: {user_id}")
    user = DefaultUserModel(user_id=user_id)
    resp = await memory_service.summary_user_profile(user=user)
    return JSONResponse(content={"status": "SUCCESS", "result": resp})


@app.post("/api/v1/memory/retrieve-category-summary")
async def retrieve_category_summary(request: RetrieveRequest):
    user = DefaultUserModel(user_id=request.user_id)
    return memory_service.get_category_summary(user=user, category_name=request.query)


@app.post("/retrieve")
async def retrieve(payload: Dict[str, Any]):
    logger.info(f"retrieve, payload: {payload}")
    if "query" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body")
    result = await memory_service.retrieve([payload["query"]])
    return JSONResponse(content={"status": "SUCCESS", "result": result})


@app.post("/api/v1/memory/retrieve/related-memory-items")
async def retrieve_items_by_queries(request: MultiRetrieveRequest):
    logger.info(f"retrieve_items_by_queries, request: {request}")
    if not request.queries:
        if not request.query:
            raise HTTPException(status_code=400, detail="`query` must not be empty")
        request.queries = [WeightedQuery(query=request.query, weight=1.0)]

    user = DefaultUserModel(user_id=request.user_id)

    async def fetch_results(weighted_query: WeightedQuery):
        items = await memory_service.retrieve_memory_items(
            user,
            weighted_query.query,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
            retrieve_type=request.retrieve_type,
        )
        return weighted_query, items

    query_results = await asyncio.gather(*(fetch_results(q) for q in request.queries))

    aggregated: Dict[str, Dict[str, Any]] = {}
    for weighted_query, items in query_results:
        for item in items:
            weighted_score = item.similarity_score * weighted_query.weight
            if item.id not in aggregated:
                aggregated[item.id] = {
                    "memory": {
                        "memory_id": item.id,
                        "memory_type": item.memory_type,
                        "content": item.summary,
                        "created_at": item.created_at,
                        "updated_at": item.updated_at,
                    },
                    "similarity_score": item.similarity_score,
                    "weighted_score": weighted_score,
                }
                continue

            aggregated_item = aggregated[item.id]
            aggregated_item["weighted_score"] += weighted_score
            if item.similarity_score > aggregated_item["similarity_score"]:
                aggregated_item["similarity_score"] = item.similarity_score

    related_memories = sorted(
        aggregated.values(),
        key=lambda entry: entry["weighted_score"],
        reverse=True,
    )

    resp = {
        "total_found": len(related_memories),
        "related_memories": related_memories,
    }
    return JSONResponse(content=resp)


@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"})


@app.get("/")
async def root():
    return {"message": "Hello MemU user!"}
