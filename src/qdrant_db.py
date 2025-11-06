from qdrant_client import QdrantClient, models
from typing import List, Dict, Any

class QdrantDB:
    def __init__(self, collection: str, vector_size: int = 384, location=":memory:"):
        self.client = QdrantClient(location=location)
        self.collection = collection
        self.vector_size = vector_size

    def reset_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE
            ),
        )

    def upsert_points(self, points: List[Dict[str, Any]]):
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: list, top_k: int = 3):
        return self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
