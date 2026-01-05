import uuid
from typing import List

from memu.llm.openai_sdk import OpenAISDKClient
from memu.models import MemoryActivityItem, MemoryItem

CONDENSATION_PROMPT = """
你是一个记忆压缩专家，任务是将输入的多条用户与AI助手（名为Starfy）的对话记忆，压缩并合并。

## Rules
- 合并重复和类似内容，如："用户喜欢吃苹果。"和"用户喜欢吃香蕉。"可以合并为"用户喜欢吃苹果和香蕉。"
- 如果不同条目间存在冲突，则以时间更新的为准。例如："[2025-12-14] 用户放弃健身计划了。"和"[2025-12-01] 用户最近在减肥。"，则合并成"[2025-12-14] 用户之前在减肥，但最近放弃健身计划了。"
- 只总结陈述的具体事情、实质性活动、想法和情绪表达，忽略其他闲聊内容（如问好、致谢等）
- 保证总结后内容精炼完整，不丢失原始信息
- 最终应当输出一组互相之间无关联的记忆条目，条目之间分行输出

## Constrains
- 绝不使用"他"、"她"、"他们"、"它"等代词，必须使用具体姓名
- 绝不使用模糊称谓如"这本书"、"这个地方"，必须使用完整名称
- 绝不遗漏会话信息
- 每个记忆条目绝不超过200字
- ** 重要 ** 严格区分用户和Starfy说的话，绝不将Starfy说的话作为用户说的话

## OutputFormat
- 使用纯文本格式，不使用任何Markdown标记
- 只要输出最终合并结果，不要输出包含对话时间在内的任何其他内容

## InputFormat
输入内容为用户与Starfy的对话记忆, 格式为：
[对话时间] 具体记忆内容

## Input
{items}
"""


# - 不仅总结用户的内容，Starfy的内容也需要处理

async def condensation_memory_items(llm_client: OpenAISDKClient, items: List[MemoryItem]) -> tuple[str, str]:
    raw_items = ""
    for item in items:
        if not item.get_content():
            continue
        mentioned_at = item.created_at.strftime("%Y-%m-%d") if item.created_at else ""
        content = item.get_content() or ""
        raw_items += f"[{mentioned_at}] {content}\n"
    if not raw_items:
        return raw_items, ""

    sp = CONDENSATION_PROMPT.format(items=raw_items)
    return raw_items, await llm_client.summarize(text=sp)


async def condensation_activity_items(llm_client: OpenAISDKClient, items: List[MemoryActivityItem]) -> str:
    raw_items = ""
    for item in items:
        if not item.content:
            continue
        mentioned_at = item.mentioned_at.strftime("%Y-%m-%d") if item.mentioned_at else ""
        content = item.content or ""
        raw_items += f"[{mentioned_at}] {content}\n"
    if not raw_items:
        return ""

    sp = CONDENSATION_PROMPT.format(items=raw_items)
    return await llm_client.summarize(text=sp)


def parse_condensation_result(original_items: List[MemoryItem], result: str) -> List[MemoryItem]:
    if not result or not result.strip():
        return []
    sp = result.split("\n")
    max_created_at = max(item.created_at for item in original_items if item.created_at)
    new_items = []
    for r in sp:
        strip = r.strip()
        if not strip:
            continue
        new_item = MemoryItem(
            id = str(uuid.uuid4()),
            created_at=max_created_at,
            # TODO 合并后的memory_type等
            resource_id=original_items[0].resource_id,
            memory_type=original_items[0].memory_type,
            summary=strip,
            embedding=None
        )
        new_items.append(new_item)
    return new_items
