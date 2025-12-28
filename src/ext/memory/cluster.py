import hdbscan
import numpy as np
from collections import defaultdict
from typing import List, Dict
from sklearn.preprocessing import normalize

from memu.models import MemoryItem


def cluster_memories(
    memories: List[MemoryItem],
    min_cluster_size: int = 2,
    min_samples: int | None = None,
) -> Dict[int, List[MemoryItem]]:
    """
    使用 HDBSCAN 对 memory embeddings 进行聚类

    返回:
      cluster_id -> List[MemoryItem]
      cluster_id = -1 表示噪声（不属于任何主题）
    """
    if not memories:
        return {}

    embeddings = np.vstack([m.embedding for m in memories])
    embeddings = normalize(embeddings, norm="l2")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples or min_cluster_size,
        metric="euclidean",
    )

    labels = clusterer.fit_predict(embeddings)

    clusters: Dict[int, List[MemoryItem]] = defaultdict(list)
    for memory, label in zip(memories, labels):
        clusters[label].append(memory)

    return clusters
