import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models


class QdrantDB:
    def __init__(
        self,
        collection: str,
        vector_size: int = 384,
        client: QdrantClient = None
    ):
        """
        QdrantDB wrapper that works for BOTH:
          • Qdrant Cloud (recommended)
          • Local Qdrant (fallback)

        Args:
            collection: Name of collection to use
            vector_size: Dimension of embedding vectors
            client: QdrantClient instance (Cloud or Local)
        """

        self.collection = collection
        self.vector_size = vector_size

        # Use provided cloud client
        if client is not None:
            self.client = client
        else:
            # Fallback local instance (in-memory)
            self.client = QdrantClient(location=":memory:")

        # Ensure the collection exists in Qdrant Cloud
        self.ensure_collection()


    # -------------------------------------------------------
    # Create collection if missing
    # -------------------------------------------------------
    def ensure_collection(self):
        """Ensures the collection exists in Qdrant."""
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if self.collection not in existing:
            print(f"[QdrantDB] Creating collection '{self.collection}'...")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
        else:
            # Optional diagnostic
            print(f"[QdrantDB] Collection '{self.collection}' already exists.")


    # -------------------------------------------------------
    # Reset collection completely
    # -------------------------------------------------------
    def reset_collection(self):
        """Deletes and recreates the collection."""
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if self.collection in existing:
            print(f"[QdrantDB] Resetting collection '{self.collection}'...")
            self.client.delete_collection(self.collection)

        self.ensure_collection()


    # -------------------------------------------------------
    # UPSERT
    # -------------------------------------------------------
    def upsert_points(self, points: List[Any]):
        """
        Upsert points into Qdrant.
        Points must be a list of models.PointStruct objects.
        """
        return self.client.upsert(
            collection_name=self.collection,
            points=points
        )


    # -------------------------------------------------------
    # SEARCH
    # -------------------------------------------------------
    def search(self, query_vector: list, top_k: int = 5):
        """
        Search the collection using cosine similarity.
        Returns list of ScoredPoint objects.
        """
        return self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )


    # -------------------------------------------------------
    # COUNT DOCS
    # -------------------------------------------------------
    def get_collection_size(self) -> int:
        """Return number of points in the collection."""
        info = self.client.get_collection(self.collection)
        return info.points_count
