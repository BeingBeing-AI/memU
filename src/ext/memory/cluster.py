import hdbscan
import numpy as np
from collections import defaultdict
from typing import List, Dict
from sklearn.preprocessing import normalize

from memu.models import MemoryItem, MemoryActivityItem

CLUSTER_PROMPT = """
你是一个用户画像分析专家，负责将用户与Starfy聊天总结的新记忆项目，与现有记忆类目进行智能合并分类。

# Rules
- 如果新记忆项目是全新信息，且不归属于现有类目，则新建类目并添加
- 如果新记忆项目与现有类目相关联，则进行智能分类
    - 如果新记忆项目与现有类目内容冲突，则以新记忆项目为准，更新现有类目内容
    - 如果新记忆项目是对现有类目的补充，则在现有类目中添加相关信息
    - 如果新记忆项目是重复或无效信息，则忽略
- 可以动态调整类目名称以更好地反映合并后的内容
- 一个记忆项目可以归属于多个类目，但最终输出时应确保每个类目内容的独立性和完整性
- ** 重要 **：类目必须具备多样性，为每个细分的子类划分类目，如某段工作经历、某项兴趣爱好、某个重要事件、某个具体的人的相关信息等
- 使用人类记忆的方式进行总结

# Constraints
- ** 重要 **：每个类目的合并结果不超过500字，如果超出则进一步拆分。
- 用户的生活、与Starfy互动等事件，除非涉及重要信息，否则不作为记忆内容。
- 只需要输出最终结果，不要输出任何多余内容

# InputFormat
输入包含两部分：
1. 现有记忆内容（可以为空）：以JSON格式提供，结构如下：
{
  "categories": [
    {
      "name": "类目名称",
      "content": "现有记忆内容"
    },
    ...
  ]
}
2. 新记忆项目，格式如下：
记忆内容：具体的记忆内容文本

# OutputFormat
以JSON格式输出最终的记忆内容，结构如下：
[
    {
      "name": "新的类目名称",
      "summary": "合并后的记忆内容"
    },
    ...
]

# Initialization
全面分析新记忆项目，准确识别新旧记忆项目之间的关联性，按Rules定义的规则和Constraints定义的限制条件，对新旧记忆内容进行分类处理。
"""


def cluster_memories(
        memories: List[MemoryItem | MemoryActivityItem],
        min_cluster_size: int = 2,
        min_samples: int | None = None,
) -> Dict[int, List[MemoryItem | MemoryActivityItem]]:
    """
    使用 HDBSCAN 对 memory embeddings 进行聚类

    返回:
      cluster_id -> List[MemoryItem]
      cluster_id = -1 表示噪声（不属于任何主题）
    """
    if not memories:
        return {}

    embeddings = np.vstack([m.embedding for m in memories if m.embedding is not None])
    embeddings = normalize(embeddings, norm="l2")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples or min_cluster_size,
        metric="euclidean",
    )

    labels = clusterer.fit_predict(embeddings)

    clusters: Dict[int, List] = defaultdict(list)
    for memory, label in zip(memories, labels):
        clusters[label].append(memory)

    return clusters
