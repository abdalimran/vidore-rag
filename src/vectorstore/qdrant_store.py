"""Qdrant vector store implementation for Vidore RAG."""

from typing import Optional
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient, models

from ..utils.timer import Timer


class QdrantVectorStore:
    """Qdrant vector store for managing document embeddings."""

    def __init__(
        self, index_path: str = "./vector_store", collection_name: str = "rag"
    ) -> None:
        """Initialize the Qdrant vector store.

        Args:
            index_path: Path to store the index
            collection_name: Name of the collection
        """
        self.client = QdrantClient(path=index_path)
        self.collection_name = collection_name

        self.vectors_config = {
            "original": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                on_disk=True,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                hnsw_config=models.HnswConfigDiff(m=0),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                    )
                ),
            ),
            "mean_pooling_columns": models.VectorParams(
                size=128,
                on_disk=True,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                    )
                ),
            ),
            "mean_pooling_rows": models.VectorParams(
                size=128,
                on_disk=True,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                    )
                ),
            ),
        }

    def _collection_exists(self, collection_name: Optional[str] = None):
        """Check if a collection exists."""
        if not collection_name:
            collection_name = self.collection_name
        return self.client.collection_exists(collection_name=collection_name)

    @Timer("Create Collection")
    def create_collection(self, collection_name: str):
        """Create a new collection if it doesn't exist."""
        if not self._collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=self.vectors_config,
                on_disk_payload=True,
            )
            return True
        return False

    def delete_collection(self, collection_name: str):
        """Delete a collection if it exists."""
        if self._collection_exists(collection_name=collection_name):
            self.client.delete_collection(collection_name=collection_name)
            return True
        return False

    def _ensure_collection_exists(self):
        """Ensure the collection exists, creating it if necessary."""
        if not self._collection_exists():
            self.create_collection(self.collection_name)
            print(f"Collection {self.collection_name} created")

    @Timer("Upload Batch Vectors")
    def upload_batch_vectors(
        self,
        original_batch,
        pooled_by_rows_batch,
        pooled_by_columns_batch,
        payload_batch: list[dict],
    ):
        """Upload a batch of vectors to the collection."""
        self._ensure_collection_exists()
        try:
            self.client.upload_collection(
                collection_name=self.collection_name,
                vectors={
                    "original": original_batch,
                    "mean_pooling_rows": pooled_by_rows_batch,
                    "mean_pooling_columns": pooled_by_columns_batch,
                },
                payload=payload_batch,
                ids=[str(uuid4()) for _ in range(len(original_batch))],
            )
        except Exception as e:
            print(f"Error uploading upsert: {e}")

    @Timer("Vector Search")
    def search(
        self,
        queries: np.ndarray,
        search_limit: int = 10,
        prefetch_limit: int = 100,
        collection_name: Optional[str] = None,
    ):
        """Search for similar vectors in the collection."""
        if collection_name:
            if self._collection_exists(collection_name=collection_name):
                self.collection_name = collection_name

        self._ensure_collection_exists()

        search_queries = [
            models.QueryRequest(
                query=query,
                prefetch=[
                    models.Prefetch(
                        query=query,
                        limit=prefetch_limit,
                        using="mean_pooling_rows",
                    ),
                    models.Prefetch(
                        query=query,
                        limit=prefetch_limit,
                        using="mean_pooling_columns",
                    ),
                ],
                limit=search_limit,
                with_payload=True,
                with_vector=False,
                using="original",
            )
            for query in queries
        ]

        response = self.client.query_batch_points(
            requests=search_queries, collection_name=self.collection_name
        )
        return [result.points for result in response]

    def upsert_batch_vectors(
        self,
        ids: list[str],
        originals: list[list[float]],
        pooled_rows: list[list[float]],
        pooled_cols: list[list[float]],
        payloads: list[dict],
    ):
        """Upsert a list of points (multivector) to Qdrant using PointStruct.vectors."""
        self._ensure_collection_exists()
        points = []
        for _id, orig, prow, pcol, payload in zip(
            ids, originals, pooled_rows, pooled_cols, payloads
        ):
            pts = models.PointStruct(
                id=_id,
                vector={
                    "original": orig,
                    "mean_pooling_rows": prow,
                    "mean_pooling_columns": pcol,
                },
                payload=payload,
            )
            points.append(pts)

        try:
            # Upsert is incremental and suitable for streaming ingestion
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as e:
            print(f"[Qdrant] Upsert error: {e}")
