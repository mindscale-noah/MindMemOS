"""Generic agglomerative clustering over fused two-axis text vectors.

Two parallel text columns are vectorized, L2-normalized, weighted-summed,
then clustered with cosine distance (average linkage). Takes plain lists
of strings; singletons are kept, never dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity


@dataclass
class TextCluster:
    """One cluster of text records.

    ``member_indices`` addresses rows of the caller's input list;
    ``representative_index`` is the most central member.
    """

    cluster_id: int
    member_indices: list[int] = field(default_factory=list)
    representative_index: int = 0

    @property
    def size(self) -> int:
        return len(self.member_indices)


def cluster_text_records(
    action_texts: list[str],
    summary_texts: list[str],
    *,
    action_weight: float,
    summary_weight: float,
    distance_threshold: float,
    vectorizer: Any = None,
) -> list[TextCluster]:
    """Cluster two parallel text columns by their fused vector.

    The vectorizer is fit on ``action_texts`` and reused for
    ``summary_texts``; pass ``vectorizer`` to inject a custom one.
    Empty input returns ``[]``.
    """
    if not action_texts:
        return []
    if len(action_texts) == 1:
        return [TextCluster(cluster_id=0, member_indices=[0], representative_index=0)]
    if len(action_texts) != len(summary_texts):
        raise ValueError(
            f"action_texts and summary_texts must be the same length: got {len(action_texts)} vs {len(summary_texts)}"
        )

    if vectorizer is None:
        from mindmemos_skill.algos.trace2skill.offline_evolution.impl.embedding import build_text_vectorizer

        vectorizer = build_text_vectorizer()

    action_matrix = vectorizer.fit_transform(action_texts)
    summary_matrix = vectorizer.transform(summary_texts)

    def _densify(matrix: Any) -> np.ndarray:
        if hasattr(matrix, "toarray"):
            return matrix.toarray()
        return np.asarray(matrix)

    a_dense = _densify(action_matrix)
    s_dense = _densify(summary_matrix)
    a_dense = a_dense / (np.linalg.norm(a_dense, axis=1, keepdims=True) + 1e-12)
    s_dense = s_dense / (np.linalg.norm(s_dense, axis=1, keepdims=True) + 1e-12)
    matrix = action_weight * a_dense + summary_weight * s_dense
    dense = _densify(matrix)
    distances = cosine_distances(dense)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(distances).tolist()

    clusters: dict[int, TextCluster] = {}
    for index, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = TextCluster(cluster_id=label, member_indices=[], representative_index=index)
        clusters[label].member_indices.append(index)

    # Representative = member with highest mean similarity to cluster mates.
    similarity = cosine_similarity(dense)
    finalized: list[TextCluster] = []
    for label in sorted(clusters):
        cluster = clusters[label]
        members = cluster.member_indices
        if len(members) == 1:
            cluster.representative_index = members[0]
        else:
            intra = similarity[np.ix_(members, members)]
            central = members[int(np.argmax(intra.mean(axis=1)))]
            cluster.representative_index = central
        finalized.append(cluster)

    # Re-number cluster_ids to a dense 0..N-1 label space.
    for new_id, cluster in enumerate(finalized):
        cluster.cluster_id = new_id
    return finalized


__all__ = ["TextCluster", "cluster_text_records"]
