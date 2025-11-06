"""
Embed and upload NBA claims to local Qdrant vector DB.
"""

import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

def ingest_nba():
    nba_path = "data/nba.json"
    model_name = "intfloat/e5-small-v2"
    collection = "nba_claims"

    print(f"Loading NBA claims from {nba_path}")
    with open(nba_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    claims = [r["claim"] for r in records]
    ids = [r["id"] for r in records]

    print(f"Encoding {len(claims)} claims...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(claims, batch_size=64, show_progress_bar=True)

    print("Connecting to Qdrant...")
    client = QdrantClient(path="data/qdrant")

    print(f"Creating or replacing collection '{collection}'...")
    client.recreate_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=len(embeddings[0]), distance=models.Distance.COSINE)
    )

    print("Uploading vectors...")
    client.upload_collection(
        collection_name=collection,
        vectors=embeddings,
        payload=records,
        ids=ids
    )

    print(f"Uploaded {len(records)} NBA claims to Qdrant collection '{collection}'")

if __name__ == "__main__":
    ingest_nba()

