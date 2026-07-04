from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .chunker import Chunk
from .config import CLUSTER_COUNT, LAMBDA_WIDTH, MAX_CONTEXT_CHUNKS, TOP_K
from .interval_distance import (
    IntervalDistance,
    IntervalRankResult,
    calculate_interval_from_vectors,
    interval_score,
    make_chunk_variants,
)


@dataclass
class ClusteredRankResult:
    rank: int
    chunk: Chunk
    interval: IntervalDistance
    cluster_id: int
    is_representative: bool = False


@dataclass
class ClusterSummary:
    cluster_id: int
    size: int
    representative: ClusteredRankResult
    members: list[ClusteredRankResult]


def calculate_chunk_interval_distance(
    chunk_a: Chunk,
    chunk_b: Chunk,
    vectorizer,
    lambda_width: float = LAMBDA_WIDTH,
) -> IntervalDistance:
    """
    チャンク同士の区間距離を計算する。

    chunk_a も chunk_b も複数表現に展開し、
    すべての組み合わせの距離から [low, high] を作る。
    """
    variants_a = make_chunk_variants(chunk_a)
    variants_b = make_chunk_variants(chunk_b)

    vectors_a = vectorizer.transform(variants_a)
    vectors_b = vectorizer.transform(variants_b)

    return calculate_interval_from_vectors(
        query_vectors=vectors_a,
        chunk_variant_vectors=vectors_b,
        lambda_width=lambda_width,
    )


def build_chunk_distance_matrix(
    ranked_results: list[IntervalRankResult],
    vectorizer,
    lambda_width: float = LAMBDA_WIDTH,
) -> np.ndarray:
    """
    候補チャンク同士の距離行列を作る。

    距離には、区間距離をスカラー化した score を使う。
    """
    n = len(ranked_results)
    matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            interval = calculate_chunk_interval_distance(
                ranked_results[i].chunk,
                ranked_results[j].chunk,
                vectorizer=vectorizer,
                lambda_width=lambda_width,
            )

            distance = interval.score

            matrix[i, j] = distance
            matrix[j, i] = distance

    return matrix


def remap_cluster_ids_by_rank(
    labels: np.ndarray,
    ranked_results: list[IntervalRankResult],
) -> dict[int, int]:
    """
    scipyのクラスタIDは順序が分かりにくいため、
    検索順位が良いクラスタから 1, 2, 3... に振り直す。
    """
    best_rank_by_label: dict[int, int] = {}

    for label, result in zip(labels, ranked_results):
        label_int = int(label)

        if label_int not in best_rank_by_label:
            best_rank_by_label[label_int] = result.rank
        else:
            best_rank_by_label[label_int] = min(
                best_rank_by_label[label_int],
                result.rank,
            )

    ordered_labels = sorted(
        best_rank_by_label.keys(),
        key=lambda label: best_rank_by_label[label],
    )

    return {
        old_label: new_id
        for new_id, old_label in enumerate(ordered_labels, start=1)
    }


def cluster_ranked_results(
    ranked_results: list[IntervalRankResult],
    vectorizer,
    candidate_count: int = TOP_K,
    cluster_count: int = CLUSTER_COUNT,
    lambda_width: float = LAMBDA_WIDTH,
    linkage_method: str = "average",
) -> list[ClusteredRankResult]:
    """
    区間距離で再ランキング済みの検索結果を、
    さらに階層的クラスタリングする。
    """
    candidates = ranked_results[:candidate_count]

    if not candidates:
        return []

    n = len(candidates)
    actual_cluster_count = min(cluster_count, n)

    if n == 1:
        return [
            ClusteredRankResult(
                rank=candidates[0].rank,
                chunk=candidates[0].chunk,
                interval=candidates[0].interval,
                cluster_id=1,
            )
        ]

    if actual_cluster_count >= n:
        labels = np.arange(1, n + 1)
    else:
        distance_matrix = build_chunk_distance_matrix(
            candidates,
            vectorizer=vectorizer,
            lambda_width=lambda_width,
        )

        condensed = squareform(distance_matrix, checks=False)

        z = linkage(condensed, method=linkage_method)

        labels = fcluster(
            z,
            t=actual_cluster_count,
            criterion="maxclust",
        )

    label_map = remap_cluster_ids_by_rank(labels, candidates)

    clustered_results: list[ClusteredRankResult] = []

    for label, result in zip(labels, candidates):
        cluster_id = label_map[int(label)]

        clustered_results.append(
            ClusteredRankResult(
                rank=result.rank,
                chunk=result.chunk,
                interval=result.interval,
                cluster_id=cluster_id,
            )
        )

    return clustered_results


def build_cluster_summaries(
    clustered_results: list[ClusteredRankResult],
    max_context_chunks: int = MAX_CONTEXT_CHUNKS,
) -> list[ClusterSummary]:
    """
    クラスタごとに代表チャンクを選ぶ。

    代表は、そのクラスタ内で interval.score が最も小さいチャンク。
    """
    groups: dict[int, list[ClusteredRankResult]] = defaultdict(list)

    for result in clustered_results:
        groups[result.cluster_id].append(result)

    summaries: list[ClusterSummary] = []

    for cluster_id, members in groups.items():
        members_sorted = sorted(
            members,
            key=lambda x: (x.interval.score, x.interval.width, x.rank),
        )

        representative = members_sorted[0]
        representative.is_representative = True

        summaries.append(
            ClusterSummary(
                cluster_id=cluster_id,
                size=len(members_sorted),
                representative=representative,
                members=members_sorted,
            )
        )

    summaries.sort(
        key=lambda summary: (
            summary.representative.interval.score,
            summary.representative.interval.width,
            summary.representative.rank,
        )
    )

    return summaries[:max_context_chunks]