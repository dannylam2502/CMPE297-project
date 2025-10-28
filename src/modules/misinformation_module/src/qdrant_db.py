from qdrant_client import QdrantClient, models
from typing import List, Dict, Any

class QdrantDB:
    def __init__(self, collection: str, vector_size: int = 384, location: str = None):
        if location is None:
            raise ValueError("location parameter is required")
        # Modern API: positional for :memory:, path= for local disk, url= for remote
        if location == ":memory:":
            self.client = QdrantClient(":memory:")
        elif location.startswith("http"):
            self.client = QdrantClient(url=location)
        else:
            self.client = QdrantClient(path=location)
        
        self.collection = collection
        self.vector_size = vector_size

    def reset_collection(self):
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]
        
        if self.collection in existing:
            self.client.delete_collection(self.collection)
        
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE
            )
        )

    def collection_exists(self) -> bool:
        """Check if collection exists"""
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]
        return self.collection in existing
    
    def get_collection_size(self) -> int:
        """Get number of points in collection"""
        if not self.collection_exists():
            return 0
        info = self.client.get_collection(self.collection)
        return info.points_count
    
    def ensure_collection(self):
        """Create collection only if it doesn't exist"""
        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
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