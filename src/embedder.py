from sentence_transformers import SentenceTransformer
from typing import List

# E5 uses instruction prefixes:
#  - "passage: ..." for stored text
#  - "query: ..."   for user queries
class E5Embedder:
    def __init__(self, model_name: str = "intfloat/e5-small-v2", normalize: bool = True):
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def embed_passages(self, texts: List[str]) -> List[list]:
        texts = [f"passage: {t}" for t in texts]
        embs = self.model.encode(texts, normalize_embeddings=self.normalize)
        return [e.tolist() for e in embs]

    def embed_query(self, text: str) -> list:
        q = f"query: {text}"
        emb = self.model.encode([q], normalize_embeddings=self.normalize)[0]
        return emb.tolist()
