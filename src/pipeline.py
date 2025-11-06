"""
CMPE297 Fact-Checking System - Integration Pipeline (NBA Only)
"""

import os
from typing import Dict, Any, List
from datetime import datetime
from modules.claim_extraction.Fact_Validator import FactValidator
from modules.claim_extraction.NLIModel import NLI_LABELS, NLIModel
from modules.claim_extraction.training.Validator_Training_Data import get_training_data
from modules.llm.llm_ollama import llm_ollama
from modules.misinformation_module.src.qdrant_db import QdrantDB
from modules.misinformation_module.src.embedder import E5Embedder
from modules.claim_extraction.Fact_Validator_Data_models import SourcePassage, FactCheckResult
from modules.llm.llm_openai import llm_openai
from modules.llm.llm_reasoning import llm_reasoning
from modules.input_extraction.input_extractor import extract_claim_from_input


class FactCheckingPipeline:
    def __init__(self, qdrant_location: str, llm_provider: str = "openai"):
        self.qdrant_location = qdrant_location
        self.embedder = E5Embedder("intfloat/e5-small-v2", normalize=True)
        self.vector_db = QdrantDB(collection="nba_claims", vector_size=384, location=qdrant_location)
        self.vector_db.ensure_collection()

        # Choose LLM provider
        self.llm = llm_openai() if llm_provider.lower() == "openai" else llm_ollama()

        nli = NLIModel(
            emb_model_name="sentence-transformers/all-mpnet-base-v2",
            nli_model_name="roberta-large-mnli",
            nli_labels=NLI_LABELS,
        )
        self.fact_validator = FactValidator(self.llm, nli, get_training_data())
        self.reasoning_engine = llm_reasoning(self.llm)

        print("NBA Fact-Checking Pipeline Initialized")

    # -----------------------------
    # Evidence Retrieval
    # -----------------------------
    def retrieve_evidence(self, query: str, top_k: int = 10) -> List[SourcePassage]:
        """Retrieve evidence only from NBA collection."""
        query_vec = self.embedder.embed_query(query)
        hits = self.vector_db.search(query_vec, top_k=top_k, collection="nba_claims")

        passages = []
        for hit in hits:
            payload = hit.payload or {}
            title = payload.get("title") or payload.get("claim", "NBA Fact")
            source_url = payload.get("source") or "https://nba.com"
            passages.append(
                SourcePassage(
                    content=payload.get("claim", ""),
                    relevance_score=float(hit.score),
                    url=source_url,
                    domain=self._extract_domain(source_url),
                    title=title,
                    published_at=datetime.now(),
                )
            )
        return passages

    def _extract_domain(self, url: str) -> str:
        if "://" in url:
            return url.split("://")[1].split("/")[0]
        return "unknown"

    # -----------------------------
    # Core Query Processing
    # -----------------------------
    def process_query(self, user_input: str) -> Dict[str, Any]:
        """Main NBA-only fact-check pipeline."""
        try:
            claim_data = extract_claim_from_input(self.llm, user_input)
            claim_text = claim_data["claims"][0]["normalized"] if "claims" in claim_data else user_input
        except Exception:
            claim_text = user_input

        passages = self.retrieve_evidence(claim_text, top_k=10)
        if not passages:
            return {
                "claim": claim_text,
                "verdict": "Not enough evidence",
                "score": 0,
                "citations": [],
                "explanation": "No relevant NBA data found."
            }

        result: FactCheckResult = self.fact_validator.validate_claim(
            claim=claim_text, claim_type="factual", passages=passages
        )

        # Fix: Properly pull from Qdrant payloads
        citations = []
        for c in getattr(result, "citations", []) or passages:
            # ensure both attr and dict forms are checked
            c_data = getattr(c, "__dict__", {})
            title = getattr(c, "title", None) or c_data.get("title") or c_data.get("claim", "NBA Fact")
            url = getattr(c, "url", None) or getattr(c, "source", None) or c_data.get("url") or c_data.get("source", "https://nba.com")
            snippet = getattr(c, "content", None) or getattr(c, "claim", None) or c_data.get("content", "")
            snippet = snippet[:200]

            citations.append({
                "title": title.strip(),
                "url": url.strip(),
                "snippet": snippet.strip()
            })

        # Generate explanation prompt
        evidence_text = "\n".join([
            f"- {getattr(p, 'content', '')} (source: {getattr(p, 'url', '')})"
            for p in passages[:5]
        ])

        prompt = f"""
        You are verifying an NBA fact.
        Claim: "{claim_text}"

        Evidence:
        {evidence_text}

        Verdict: {result.verdict} (Score: {result.score}/100)

        Using only the factual evidence above, write a concise and accurate explanation.
        Include player names, correct stats, and clarify *why* the verdict was reached.
        Do not invent or guess statistics not in the evidence.
        """

        explanation = self.reasoning_engine.reasoning_agent(prompt)

        return {
            "claim": result.claim,
            "verdict": result.verdict,
            "score": result.score,
            "citations": citations,
            "explanation": explanation
        }

    # -----------------------------
    # UI Formatting
    # -----------------------------
    def format_for_ui(self, response: Dict[str, Any]) -> str:
        """Readable formatted UI output."""
        verdict = response.get("verdict", "Unknown")
        score = response.get("score", 0)
        claim = response.get("claim", "")
        out = f"\nVerdict: {verdict} (Score: {score}/100)\nClaim: {claim}\n\nSources:\n"

        for i, cite in enumerate(response.get("citations", []), 1):
            title = cite.get("title") or "NBA Fact"
            url = cite.get("url") or "https://nba.com"
            snippet = cite.get("snippet", "")[:150]
            out += f"{i}. {title}\n   {url}\n   {snippet}...\n\n"

        return out.strip()
