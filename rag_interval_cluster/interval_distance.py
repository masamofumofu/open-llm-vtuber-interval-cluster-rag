from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import Chunk
from .config import LAMBDA_WIDTH


@dataclass
class IntervalDistance:
    low: float
    high: float
    center: float
    width: float
    score: float
    all_distances: list[float]


@dataclass
class IntervalRankResult:
    rank: int
    chunk: Chunk
    interval: IntervalDistance


def make_query_variants(query: str) -> list[str]:
    """
    1つの質問から複数の検索表現を作る。

    最初はLLMによる言い換え生成は使わず、
    固定ルールで軽量に作る。
    """
    query = query.strip()

    variants = [
        query,
        f"{query} 関連する設定 定義 パラメータ 項目",
        f"{query} ファイル名 項目名 設定値 内容",
    ]

    # 重複除去
    unique_variants: list[str] = []
    seen: set[str] = set()

    for variant in variants:
        if variant not in seen:
            unique_variants.append(variant)
            seen.add(variant)

    return unique_variants


def make_chunk_variants(chunk: Chunk) -> list[str]:
    """
    1つのチャンクから複数の表現を作る。

    d1: 本文のみ
    d2: ファイル名 + 本文
    d3: chunk_id + ファイル名 + 本文

    これにより、本文だけでなくファイル名やchunk_idも
    検索・距離計算に反映できる。
    """
    variants = [
        chunk.text,
        f"{chunk.file_name}\n{chunk.text}",
        f"{chunk.chunk_id}\n{chunk.file_name}\n{chunk.text}",
    ]

    return variants


def interval_score(
    low: float,
    high: float,
    lambda_width: float = LAMBDA_WIDTH,
) -> float:
    """
    区間距離をランキング用スコアに変換する。

    score = center + λ * width

    距離なので、小さいほど良い。
    """
    center = (low + high) / 2.0
    width = high - low
    return center + lambda_width * width


def calculate_interval_from_vectors(
    query_vectors,
    chunk_variant_vectors,
    lambda_width: float = LAMBDA_WIDTH,
) -> IntervalDistance:
    """
    質問ベクトル群 Q とチャンク表現ベクトル群 D の全組み合わせから
    距離区間 [low, high] を計算する。
    """
    similarities = cosine_similarity(query_vectors, chunk_variant_vectors)

    distances = 1.0 - similarities
    flat_distances = distances.reshape(-1).astype(float)

    low = float(np.min(flat_distances))
    high = float(np.max(flat_distances))
    center = float((low + high) / 2.0)
    width = float(high - low)
    score = float(interval_score(low, high, lambda_width=lambda_width))

    return IntervalDistance(
        low=low,
        high=high,
        center=center,
        width=width,
        score=score,
        all_distances=[float(x) for x in flat_distances],
    )


def rank_chunks_by_interval_distance(
    query: str,
    chunks: list[Chunk],
    vectorizer,
    lambda_width: float = LAMBDA_WIDTH,
) -> list[IntervalRankResult]:
    """
    各チャンクについて、質問との区間距離を計算して再ランキングする。
    """
    if not chunks:
        return []

    query_variants = make_query_variants(query)
    query_vectors = vectorizer.transform(query_variants)

    results: list[IntervalRankResult] = []

    for chunk in chunks:
        chunk_variants = make_chunk_variants(chunk)
        chunk_variant_vectors = vectorizer.transform(chunk_variants)

        interval = calculate_interval_from_vectors(
            query_vectors=query_vectors,
            chunk_variant_vectors=chunk_variant_vectors,
            lambda_width=lambda_width,
        )

        results.append(
            IntervalRankResult(
                rank=0,
                chunk=chunk,
                interval=interval,
            )
        )

    results.sort(key=lambda x: x.interval.score)

    for rank, result in enumerate(results, start=1):
        result.rank = rank

    return results