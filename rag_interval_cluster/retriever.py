from dataclasses import dataclass

from .chunker import Chunk, chunk_documents
from .config import CLUSTER_COUNT, MAX_CONTEXT_CHUNKS, TOP_K
from .document_loader import Document, load_documents
from .embedder_tfidf import SearchResult, TfidfEmbedder
from .hierarchical_cluster import (
    ClusterSummary,
    ClusteredRankResult,
    build_cluster_summaries,
    cluster_ranked_results,
)
from .interval_distance import IntervalRankResult, rank_chunks_by_interval_distance
from .prompt_builder import PromptBuildResult, build_rag_prompt


@dataclass
class RagIntervalResult:
    query: str
    documents: list[Document]
    chunks: list[Chunk]
    normal_results: list[SearchResult]
    interval_ranked_results: list[IntervalRankResult]
    clustered_results: list[ClusteredRankResult]
    cluster_summaries: list[ClusterSummary]
    prompt_result: PromptBuildResult


class IntervalClusterRetriever:
    """
    検索結果後処理型・階層的区間クラスタリングRAG。

    役割:
    1. docs配下のファイルを読む
    2. チャンク化する
    3. TF-IDFで通常検索する
    4. 区間距離で再ランキングする
    5. 階層的クラスタリングする
    6. クラスタ代表チャンクからRAGプロンプトを作る
    """

    def __init__(
        self,
        top_k: int = TOP_K,
        candidate_count: int = TOP_K,
        cluster_count: int = CLUSTER_COUNT,
        max_context_chunks: int = MAX_CONTEXT_CHUNKS,
        max_chars_per_cluster: int = 2500,
        neighbor_before: int = 1,
        neighbor_after: int = 6,
    ) -> None:
        self.top_k = top_k
        self.candidate_count = candidate_count
        self.cluster_count = cluster_count
        self.max_context_chunks = max_context_chunks
        self.max_chars_per_cluster = max_chars_per_cluster
        self.neighbor_before = neighbor_before
        self.neighbor_after = neighbor_after

        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.embedder = TfidfEmbedder()
        self._is_built = False

    def build(self) -> None:
        """
        docs配下の文書を読み込み、チャンク化してTF-IDF indexを作る。
        """
        self.documents = load_documents()
        self.chunks = chunk_documents(self.documents)

        if not self.chunks:
            raise RuntimeError("No chunks found. Please put documents in rag_interval_cluster/docs.")

        self.embedder.fit_chunks(self.chunks)
        self._is_built = True

    def retrieve(self, query: str) -> RagIntervalResult:
        """
        1つの質問に対して、RAG後処理結果を返す。
        """
        if not self._is_built:
            self.build()

        normal_results = self.embedder.search(
            query=query,
            top_k=self.top_k,
        )

        interval_ranked_results = rank_chunks_by_interval_distance(
            query=query,
            chunks=self.chunks,
            vectorizer=self.embedder.vectorizer,
        )

        clustered_results = cluster_ranked_results(
            ranked_results=interval_ranked_results,
            vectorizer=self.embedder.vectorizer,
            candidate_count=self.candidate_count,
            cluster_count=self.cluster_count,
        )

        cluster_summaries = build_cluster_summaries(
            clustered_results=clustered_results,
            max_context_chunks=self.max_context_chunks,
        )

        prompt_result = build_rag_prompt(
            query=query,
            summaries=cluster_summaries,
            all_chunks=self.chunks,
            max_chars_per_cluster=self.max_chars_per_cluster,
            neighbor_before=self.neighbor_before,
            neighbor_after=self.neighbor_after,
        )

        return RagIntervalResult(
            query=query,
            documents=self.documents,
            chunks=self.chunks,
            normal_results=normal_results,
            interval_ranked_results=interval_ranked_results,
            clustered_results=clustered_results,
            cluster_summaries=cluster_summaries,
            prompt_result=prompt_result,
        )