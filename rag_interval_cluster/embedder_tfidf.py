from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import Chunk


@dataclass
class SearchResult:
    rank: int
    chunk: Chunk
    similarity: float
    distance: float


class TfidfEmbedder:
    """
    RAG検索用の軽量TF-IDFベクトル化クラス。

    今回は日本語・SQL・YAML・BATなどをまとめて扱うため、
    単語分割ではなく文字N-gramを使う。
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (2, 5),
        max_features: int | None = 50000,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            lowercase=False,
            max_features=max_features,
        )
        self.chunk_vectors = None
        self.chunks: list[Chunk] = []

    def build_chunk_text(self, chunk: Chunk) -> str:
        """
        チャンク本文だけでなく、ファイル名も検索対象に含める。
        SQLファイル名や conf.yaml などの検索に効く。
        """
        return f"{chunk.file_name}\n{chunk.text}"

    def fit_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("chunks is empty")

        self.chunks = chunks
        texts = [self.build_chunk_text(chunk) for chunk in chunks]
        self.chunk_vectors = self.vectorizer.fit_transform(texts)

    def transform_queries(self, queries: list[str]):
        if self.chunk_vectors is None:
            raise RuntimeError("fit_chunks() must be called before transform_queries().")

        return self.vectorizer.transform(queries)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if self.chunk_vectors is None:
            raise RuntimeError("fit_chunks() must be called before search().")

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]

        # cosine distance = 1 - cosine similarity
        distances = 1.0 - similarities

        order = np.argsort(distances)[:top_k]

        results: list[SearchResult] = []

        for rank, index in enumerate(order, start=1):
            results.append(
                SearchResult(
                    rank=rank,
                    chunk=self.chunks[int(index)],
                    similarity=float(similarities[int(index)]),
                    distance=float(distances[int(index)]),
                )
            )

        return results