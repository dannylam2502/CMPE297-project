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


def keyword_overlap_score(query, text):
    """Compute simple keyword overlap ratio."""
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())
    if not q_words:
        return 0.0
    return len(q_words & t_words) / len(q_words)


def re_rank(results, query, semantic_weight=0.7, keyword_weight=0.3):
    """Combine semantic similarity with keyword overlap for re-ranking."""
    re_ranked = []
    for r in results:
        kw_score = keyword_overlap_score(query, r["claim"])
        final_score = semantic_weight * r["score"] + keyword_weight * kw_score
        re_ranked.append({**r, "keyword_score": kw_score, "final_score": final_score})
    return sorted(re_ranked, key=lambda x: x["final_score"], reverse=True)


def rule_based_label(claim, confidence):
    """Simple rule-based check for misinformation classification."""
    if confidence >= 0.8:
        label = "likely_true"
    elif confidence >= 0.5:
        label = "uncertain"
    else:
        label = "likely_false"

    return {"claim": claim, "confidence": confidence, "label": label}


def main():
    print("Loading embedder...")
    embedder = E5Embedder("intfloat/e5-small-v2", normalize=True)

    print("Starting Qdrant (in-memory)...")
    db = QdrantDB(collection=COLLECTION, vector_size=EMBED_DIMS, location=":memory:")
    db.reset_collection()

    # Load your mock data
    data = load_mock_data("data/mock.json")
    texts = [d["claim"] for d in data]

    print("Embedding passages...")
    vectors = embedder.embed_passages(texts)
    points = build_points(data, vectors)
    db.upsert_points(points)

    # Query
    user_query = "When are humans scheduled to return to the Moon?"
    print(f"\nQuery: {user_query}")
    qvec = embedder.embed_query(user_query)
    hits = db.search(qvec, top_k=5)

    # Retrieve top results
    results = [
        {
            "id": h.id,
            "score": float(h.score),
            "claim": h.payload["claim"],
            "source": h.payload["source"],
            "confidence": h.payload["confidence"],
        }
        for h in hits
    ]

    print("\nTop semantic matches (before re-ranking):")
    for r in results:
        print(
            f"- [sem_score={r['score']:.4f}] {r['claim']} | source={r['source']} | conf={r['confidence']}"
        )

    # --- Re-ranking ---
    reranked_results = re_rank(results, user_query)
    print("\nRe-ranked results (semantic + keyword):")
    for r in reranked_results:
        print(
            f"- [final={r['final_score']:.4f}] {r['claim']} "
            f"(sem={r['score']:.4f}, kw={r['keyword_score']:.2f})"
        )

    # --- Rule-based classification ---
    classified_results = [rule_based_label(r["claim"], r["confidence"]) for r in reranked_results]

    print("\nRule-based classification results:")
    for c in classified_results:
        print(f"- {c['claim']} => {c['label']} (conf={c['confidence']})")

    # --- LLM handoff ---
    llm_handoff = {
        "query": user_query,
        "retrieved": reranked_results,
        "rule_based_labels": classified_results,
    }

    print("\nLLM handoff payload:")
    print(json.dumps(llm_handoff, indent=2))


if __name__ == "__main__":
    main()
