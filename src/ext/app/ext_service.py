import json
import logging
from collections.abc import Sequence
from typing import Any, List

from pydantic import BaseModel

from ext.ext_models import ExtMemoryItem, ConversationMessage
from ext.prompts.cluster import CLUSTER_PROMPT
from ext.prompts.summary_profile import PROMPT
from ext.store.pg_repo import PgStore, MemoryResourceModel
from memu.app import MemoryService
from memu.app.service import _UserContext
from memu.embedding import HTTPEmbeddingClient
from memu.models import MemoryType

logger = logging.getLogger(__file__)

USER_PROFILE_CAT_NAME = "user_profile"


class ExtUserContext(_UserContext):
    """Per-user in-memory state and category bookkeeping."""

    def __init__(self, user_id, *, categories_ready: bool) -> None:
        super().__init__(categories_ready=categories_ready)
        self.user_id = user_id
        self.store = PgStore(user_id=user_id)


from memu.llm.openai_sdk import OpenAISDKClient

flash_llm_client = OpenAISDKClient(
    base_url="http://llm.ai-nebula.com/v1",
    api_key="sk-i64nj7hr5z9rnpb00lyvhm",
    chat_model="gemini-3-flash-preview",
)


class ExtMemoryService(MemoryService):
    def _init_embedding_client(self) -> Any:
        """Initialize embedding client based on configuration."""
        backend = self.embedding_config.client_backend
        if backend == "sdk":
            from ext.embedding.ext_openai_sdk import ExtOpenAIEmbeddingSDKClient

            return ExtOpenAIEmbeddingSDKClient(
                base_url=self.embedding_config.base_url,
                api_key=self.embedding_config.api_key,
                embed_model=self.embedding_config.embed_model,
            )
        elif backend == "httpx":
            return HTTPEmbeddingClient(
                base_url=self.embedding_config.base_url,
                api_key=self.embedding_config.api_key,
                embed_model=self.embedding_config.embed_model,
                provider=self.embedding_config.provider,
                endpoint_overrides=self.embedding_config.endpoint_overrides,
            )
        else:
            msg = f"Unknown embedding_client_backend '{self.embedding_config.client_backend}'"
            raise ValueError(msg)

    def _get_user_context(self, user: BaseModel | None) -> _UserContext:
        key = self._context_key(user)
        ctx = self._contexts.get(key)
        if ctx:
            return ctx
        ctx = ExtUserContext(user_id=user.user_id, categories_ready=not bool(self.category_configs))
        self._contexts[key] = ctx
        self._start_category_initialization(ctx)
        return ctx

    def get_resource_by_url(self, user: BaseModel | None, resource_url: str) -> MemoryResourceModel | None:
        return self._get_user_context(user).store.get_resource_by_url(resource_url)

    def on_memorize_done(self, user: BaseModel | None, resource_url: str):
        self._get_user_context(user).store.update_resource_status(resource_url, "success")

    async def summary_user_profile(self, user: BaseModel | None) -> str | None:
        ctx = self._get_user_context(user)
        categories = ctx.store.get_all_categories()
        valid_categories = [cat for cat in categories if cat.summary and cat.name != USER_PROFILE_CAT_NAME]
        if not valid_categories:
            logger.info(f"No categories to summarize for user {user.user_id}")
            return None
        formated = [
            {
                "name": cat.name,
                "summary": cat.summary,
            }
            for cat in valid_categories
        ]
        response = await self.llm_client.summarize(system_prompt=PROMPT, text=json.dumps(formated, indent=2),
                                                   reasoning_effort="medium")
        logger.info(f"Summary user profile: {response}")
        cat = ctx.store.get_or_create_category(name="user_profile", description="summary of user profile",
                                               embedding=None)
        cat.summary = response
        return response

    async def clustering(self, user: BaseModel | None):
        ctx = self._get_user_context(user)
        exist_clusters = ctx.store.get_all_clusters()
        formated_cluster = [
            {
                "name": cat.name,
                "summary": cat.summary,
            }
            for cat in exist_clusters
        ]

        items = ctx.store.get_unclustered_items()
        filter_items = [item for item in items if item.memory_type != "knowledge"]
        formated_items = "\n".join([item.summary for item in filter_items])

        user_prompt = f"""现有记忆内容：{json.dumps(formated_cluster, indent=2)}
新记忆项目: {formated_items}
"""
        response = await flash_llm_client.summarize(system_prompt=CLUSTER_PROMPT, text=user_prompt)
        logger.info(f"Cluster response: {response}")
        if response and len(response) > 10:
            try:
                cluster_data = json.loads(response)
                if isinstance(cluster_data, list):
                    # 提取新聚类数据
                    new_clusters = []
                    for cluster in cluster_data:
                        if isinstance(cluster, dict) and "name" in cluster and "summary" in cluster:
                            new_clusters.append({
                                "name": cluster["name"],
                                "summary": cluster["summary"],
                                "embedding": None  # TODO: 这里需要生成embedding，暂时为空
                            })

                    # 提取未聚类的记忆项ID
                    item_ids = [item.id for item in filter_items]

                    # 调用新的事务方法更新聚类
                    if new_clusters and item_ids:
                        success = ctx.store.update_clusters_and_mark_items(new_clusters, item_ids)
                        if success:
                            logger.info(f"成功更新 {len(new_clusters)} 个聚类，标记 {len(item_ids)} 个记忆项为已聚类")
                        else:
                            logger.error("更新聚类失败")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse cluster response: {e}")
        return response

    def get_all_category_summaries(self, user: BaseModel | None):
        ctx = self._get_user_context(user)
        categories = ctx.store.get_all_categories()
        return [
            {
                "name": cat.name,
                "summary": cat.summary,
            }
            for cat in categories
        ]

    def get_category_summary(self, user: BaseModel | None, category_name: str):
        ctx = self._get_user_context(user)
        cat = ctx.store.get_category_by_name(category_name)
        return {
            "name": cat.name,
            "summary": cat.summary,
        } if cat else None

    async def retrieve_memory_items(self, user: BaseModel | None, query: str,
                                    context_messages: List[ConversationMessage] = None,
                                    retrieved_content: str | None = None,
                                    top_k: int = 10, min_similarity: float = 0.3,
                                    retrieve_type: str = "light", ) -> List[ExtMemoryItem]:
        ctx = self._get_user_context(user)
        if retrieve_type == "light":
            qvec = (await self.embedding_client.embed([query]))[0]
            return ctx.store.retrieve_memory_items(qvec, top_k=top_k, min_similarity=min_similarity)

        # 基于LLM+RAG的检索
        # Step 1: Decide if retrieval is needed
        needs_retrieval, rewritten_query = await self._decide_if_retrieval_needed(
            query, context_messages, retrieved_content=retrieved_content
        )
        logger.info(f"retrieve_memory_items, query: {query}, context_messages: {context_messages}, "
                    f"retrieved_content: {retrieved_content}, needs_retrieval: {needs_retrieval}, "
                    f"rewritten_query: {rewritten_query}")

        if not needs_retrieval:
            return []

        qvec = (await self.embedding_client.embed([rewritten_query]))[0]
        return ctx.store.retrieve_memory_items(qvec, top_k=top_k, min_similarity=min_similarity)

    def _parse_structured_entries(
            self, memory_types: list[MemoryType], responses: Sequence[str]
    ) -> list[tuple[MemoryType, str, list[str]]]:
        entries: list[tuple[MemoryType, str, list[str]]] = []
        for mtype, response in zip(memory_types, responses, strict=True):
            parsed = self._parse_memory_type_response(response)
            if not parsed:
                continue
                # fallback_entry = response.strip()
                # if fallback_entry:
                #     entries.append((mtype, fallback_entry, []))
            for entry in parsed:
                content = (entry.get("content") or "").strip()
                if not content:
                    continue
                cat_names = [c.strip() for c in entry.get("categories", []) if isinstance(c, str) and c.strip()]
                entries.append((mtype, content, cat_names))
        return entries
