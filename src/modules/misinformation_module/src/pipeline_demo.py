import json
from pathlib import Path
from qdrant_client import models
from src.embedder import E5Embedder
from src.qdrant_db import QdrantDB


COLLECTION = "facts_demo"
EMBED_DIMS = 384

def load_mock_data(path: str):
    return json.loads(Path(path).read_text())

def build_points(rows, vectors):
    points = []
    for row, vec in zip(rows, vectors):
        points.append(
            models.PointStruct(
                id=row["id"],
                vector=vec,
                payload={
                    "claim": row["claim"],
                    "source": row["source"],
                    "confidence": row["confidence"],
                },
            )
        )
    return points

def main():
    print("Loading embedder…")
    embedder = E5Embedder("intfloat/e5-small-v2", normalize=True)

    print("Starting Qdrant (in-memory)…")
    db = QdrantDB(collection=COLLECTION, vector_size=EMBED_DIMS, location=":memory:")
    db.reset_collection()

    # Load data
    data = load_mock_data("data/mock.json")
    texts = [d["claim"] for d in data]

    # Embed + store
    print("Embedding passages…")
    vectors = embedder.embed_passages(texts)
    points = build_points(data, vectors)
    print("Upserting into Qdrant…")
    db.upsert_points(points)

    # Query
    user_query = "When are humans scheduled to return to the Moon?"
    print(f"\nQuery: {user_query}")
    qvec = embedder.embed_query(user_query)
    hits = db.search(qvec, top_k=2)

    # Results
    results = [{
        "id": h.id,
        "score": float(h.score),
        "claim": h.payload["claim"],
        "source": h.payload["source"],
        "confidence": h.payload["confidence"]
    } for h in hits]

    print("\nTop matches:")
    for r in results:
        print(f"- [score={r['score']:.4f}] {r['claim']} | source={r['source']} | conf={r['confidence']}")

    llm_handoff = {"query": user_query, "retrieved": results}
    print("\nLLM handoff payload:")
    print(llm_handoff)

if __name__ == "__main__":
    main()
