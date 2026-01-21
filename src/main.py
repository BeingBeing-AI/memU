import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

from memu.models import MemoryItem

load_dotenv()

from ext.app.ext_service import ExtMemoryService
from ext.ext_models import MemorizeRequest, RetrieveRequest, MultiRetrieveRequest, WeightedQuery, AddMemoryItemRequest
from ext.memory.cluster import cluster_memories
from ext.memory.condensation import condensation_memory_items, parse_condensation_result
from ext.store.activity_item_store import retrieve_activity_items_to_memory
from ext.store.memory_item_store import retrieve_memory_items, get_all_memory_items, update_condensation_items, \
    add_memory_items
from memu.app import DefaultUserModel
from memu.llm.openai_sdk import OpenAISDKClient


from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError

flash_llm_client = OpenAISDKClient(
    base_url="https://gemini-965808384446.asia-east1.run.app/v1beta/openai",
    api_key=os.getenv("NEBULA_API_KEY"),
    chat_model="gemini-3-flash-preview",
)

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
            "base_url": os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "api_key": os.getenv("EMBEDDING_API_KEY", os.getenv("QWEN_API_KEY")),
            "embed_model": os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
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


@app.post("/api/v1/memory/clustering")
async def clustering(user_id: str):
    logger.info(f"clustering, user_id: {user_id}")
    user = DefaultUserModel(user_id=user_id)
    resp = await memory_service.clustering(user=user)
    return JSONResponse(content={"status": "SUCCESS", "result": resp})

@app.post("/api/v1/memory/condensation")
async def condensation(user_id: str):
    logger.info(f"condensation, user_id: {user_id}")
    embedding_client = memory_service.embedding_client
    memory_items = get_all_memory_items(user_id, include_embedding=True)
    # TODO
    if len(memory_items) < 500:
        return JSONResponse(content={"status": "SKIP"})
    clusters = cluster_memories(memory_items)
    for label, c in clusters.items():
        logger.info(f"Cluster {label}: {len(c)} items")
        if label == -1:
            continue
        raw_items, result = await condensation_memory_items(flash_llm_client, c)
        new_items = parse_condensation_result(original_items=c, result=result)
        summaries = [i.summary for i in new_items]
        embeddings = await embedding_client.embed(summaries)

        # 将embeddings赋值给new_items
        for i, embedding in enumerate(embeddings):
            new_items[i].embedding = embedding

        # 获取需要删除的原数据ID
        old_ids = [item.id for item in c]

        # 在同一个事务中执行删除和插入操作
        try:
            delete_count, insert_count = update_condensation_items(user_id, old_ids, new_items)
            logger.info(f"condensation: {delete_count} old items deleted, {insert_count} new items inserted")
        except Exception as e:
            msg = f"Condensation error: {e}"
            logger.error(msg)
            return JSONResponse(content={"status": "ERROR", "message": msg})


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
async def retrieve_related_memory_items(request: MultiRetrieveRequest):
    start_time = time.time()
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
            min_similarity=request.min_similarity
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

    # 计算耗时
    elapsed_time = time.time() - start_time
    logger.info(f"retrieve_items_by_queries completed, elapsed_time: {elapsed_time:.4f}s, response: {resp}")

    return JSONResponse(content=resp)


@app.post("/api/v1/memory/retrieve/items")
async def retrieve_related_items(request: MultiRetrieveRequest):
    start_time = time.time()
    logger.info(f"retrieve_related_items, request: {request}")
    if not request.queries:
        if not request.query:
            raise HTTPException(status_code=400, detail="`query` must not be empty")
        request.queries = [WeightedQuery(query=request.query, weight=1.0)]

    async def fetch_results(qvec, query_source: str = "memory_item"):
        if query_source == "activity_item":
            items = await asyncio.to_thread(
                retrieve_activity_items_to_memory,
                int(request.user_id),
                qvec,
                request.top_k,
                request.min_similarity,
            )
        elif query_source == "memory_item":
            items = await asyncio.to_thread(
                retrieve_memory_items,
                request.user_id,
                qvec,
                request.top_k,
                request.min_similarity,
            )
        elif query_source == "global_memory":
            items = await asyncio.to_thread(
                retrieve_memory_items,
                # 默认使用 0 作为系统用户，存储 global memory
                '0',
                qvec,
                request.top_k,
                request.min_similarity,
            )
        else:
            items = []
        return query_source, items

    qvecs = await memory_service.embedding_client.embed([q.query for q in request.queries])
    embed_elapsed = time.time() - start_time

    async def fetch_with_meta(query_index: int, query_vector: list[float], source: str):
        query_source, items = await fetch_results(query_vector, source)
        return query_index, query_source, items

    tasks = [
        fetch_with_meta(query_index, qvec, source)
        for query_index, qvec in enumerate(qvecs)
        for source in request.query_sources
    ]
    db_start = time.time()
    query_results = await asyncio.gather(*tasks)
    db_elapsed = time.time() - db_start

    # 按照queries中的weight值对每个结果的相似度分数进行加权
    weighted_items_by_source: dict[str, list[dict[str, Any]]] = {
        query_source: [] for query_source in request.query_sources
    }

    # 将query_results与对应的weight关联
    total_found = 0
    for query_index, query_source, items in query_results:
        total_found += len(items)
        weight = request.queries[query_index].weight

        # 为每个item添加weight信息
        for item in items:
            item_dict = item.model_dump()
            item_dict["weighted_similarity"] = item.similarity_score * weight
            item_dict["query_weight"] = weight
            weighted_items_by_source[query_source].append(item_dict)

    # 按加权相似度排序并取top_k
    sources = []
    top_k = request.top_k
    for query_source, items in weighted_items_by_source.items():
        items.sort(key=lambda x: x["weighted_similarity"], reverse=True)
        sources.append({
            "query_source": query_source,
            "items": items[:top_k] if top_k < len(items) else items,
        })
    resp = {
        "total_found": total_found,
        "sources": sources,
    }

    # 计算耗时
    elapsed_time = time.time() - start_time
    logger.info(
        "retrieve_items_by_queries timings: total=%.4fs, embed=%.4fs, db=%.4fs, total_found=%d",
        elapsed_time,
        embed_elapsed,
        db_elapsed,
        total_found,
    )

    return JSONResponse(content=resp)

@app.post("/api/v1/memory/items/add")
async def add_memory_items_endpoint(request: AddMemoryItemRequest):
    logger.info(f"Add memory items: {request}")
    try:
        # 为每个摘要生成embedding
        embeddings = await memory_service.embedding_client.embed(request.summaries)

        # 创建MemoryItem对象列表
        memory_items = []
        for i, summary in enumerate(request.summaries):
            # 为每个摘要生成唯一的resource_id
            resource_id = request.resource_id if request.resource_id else f"manual-{uuid.uuid4()}"

            memory_item = MemoryItem(
                id=str(uuid.uuid4()),
                resource_id=resource_id,
                created_at=datetime.now(),
                memory_type="knowledge",
                summary=summary,
                embedding=embeddings[i] if i < len(embeddings) else None
            )
            memory_items.append(memory_item)

        # 调用批量插入方法
        saved_items = add_memory_items(memory_items, request.user_id)

        # 返回保存成功的项目数量和详情
        return JSONResponse(content={
            "status": "SUCCESS",
            "count": len(saved_items),
            "saved_items": [
                {
                    "id": item.id,
                    "summary": item.summary,
                    "memory_type": item.memory_type,
                    "created_at": item.created_at.isoformat() if item.created_at else None
                }
                for item in saved_items
            ]
        })

    except Exception as e:
        logger.error(f"Error adding memory items: {e}")
        return JSONResponse(
            content={"status": "ERROR", "message": str(e)},
            status_code=500
        )

@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"})


@app.get("/")
async def root():
    return {"message": "Hello MemU user!"}
