import os, time, feedparser, requests
from dotenv import load_dotenv
load_dotenv()
from bs4 import BeautifulSoup
from datetime import datetime
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from typing import List

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("COLLECTION_NAME", "nba_news_claims")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer("intfloat/e5-small-v2")

FEEDS = [
    "https://www.espn.com/espn/rss/nba/news",           # ESPN (works)
    "https://sports.yahoo.com/nba/rss",                 # Yahoo Sports NBA (works)
    "https://www.cbssports.com/rss/headlines/nba/",     # CBS Sports NBA (works)
    "https://www.sbnation.com/rss/nba/index.xml",       # SB Nation NBA (works)
    "https://www.hoopsrumors.com/feed",                 # HoopsRumors NBA news (works)
]

# -----------------------------
#   FULL ARTICLE SCRAPING
# -----------------------------
def scrape_full_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=7)
        soup = BeautifulSoup(r.text, "html.parser")

        # ESPN / NBA.com / BleacherReport usually have <p> content
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(paragraphs)

        return text if len(text) > 200 else ""  # fallback to summary if too short
    except:
        return ""


# -----------------------------
#   TEXT CHUNKING FOR RAG
# -----------------------------
def chunk_text(text: str, chunk_size=500, overlap=100) -> List[str]:
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks


# -----------------------------
#   FETCH ARTICLES
# -----------------------------
def fetch_articles() -> List[dict]:
    items = []

    for url in FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            if "nba" not in (title + summary).lower():
                continue

            link = entry.get("link", "")
            published = entry.get("published", str(datetime.utcnow()))

            full_text = scrape_full_text(link)
            base_text = full_text if full_text else summary

            chunks = chunk_text(base_text)

            items.append({
                "id": entry.get("id", link),
                "title": title,
                "chunks": chunks,
                "link": link,
                "published": published
            })

    return items

def ensure_collection():
    collections = qdrant.get_collections().collections
    names = [c.name for c in collections]

    if COLLECTION not in names:
        print(f"Creating collection: {COLLECTION}")
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=384,      # e5-small-v2 embedding size
                distance="Cosine"
            )
        )
    else:
        print(f"Collection {COLLECTION} already exists.")

# -----------------------------
#   UPSERT TO QDRANT (CHUNKED)
# -----------------------------
def upsert_to_qdrant(items: List[dict]):
    all_points = []

    for item in items:
        for idx, chunk in enumerate(item["chunks"]):
            vec = embedder.encode(chunk, normalize_embeddings=True)

            point_id = abs(hash(f"{item['id']}_{idx}")) % (2**63)

            all_points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vec.tolist(),
                    payload={
                        "title": item["title"],
                        "content": chunk,
                        "source": item["link"],
                        "published_at": item["published"]
                    }
                )
            )

    if all_points:
        qdrant.upsert(collection_name=COLLECTION, points=all_points)
        print(f"Upserted {len(all_points)} chunked vectors to Qdrant.")
    else:
        print("No articles to insert.")


if __name__ == "__main__":
    ensure_collection()
    print(f"[{datetime.utcnow()}] Fetching NBA news articles...")
    articles = fetch_articles()
    upsert_to_qdrant(articles)

