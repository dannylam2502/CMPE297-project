"""
CMPE297 Fact-Checking System - Integration Pipeline
Connects: Input Extraction → Vector DB → Fact Validation → LLM Response → Output
"""

import json
import os
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import asdict

# Module imports - adjust paths based on actual repo structure
from modules.input_extraction.input_extractor import extract_claim_from_input
from modules.misinformation_module.qdrant_db import QdrantDB
from modules.misinformation_module.embedder import E5Embedder
from modules.claim_extraction.fact_validator import FactValidator
from modules.claim_extraction.fact_validator_interface import SourcePassage, FactCheckResult
from modules.llm.llm_ollama import llm_ollama


class FactCheckingPipeline:
    """
    Main integration pipeline that orchestrates all modules.
    """
    
    def __init__(
        self,
        collection_name: str = "facts_collection",
        vector_size: int = 384,
        qdrant_location: str = ":memory:",
        embedding_model: str = "intfloat/e5-small-v2"
    ):
        # Initialize Vector DB
        self.embedder = E5Embedder(embedding_model, normalize=True)
        self.vector_db = QdrantDB(
            collection=collection_name,
            vector_size=vector_size,
            location=qdrant_location
        )
        self.vector_db.reset_collection()
        
        # Initialize LLM
        self.llm = llm_ollama()
        
        # Initialize Fact Validator
        self.fact_validator = FactValidator(self.llm)
        
        print(f"Pipeline initialized with collection '{collection_name}'")
    
    def load_knowledge_base(self, data_path: str) -> None:
        """
        Load and index knowledge base into vector DB.
        
        Args:
            data_path: Path to JSON file with format:
                [{"id": int, "claim": str, "source": str, "confidence": float}, ...]
        """
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        texts = [d["claim"] for d in data]
        vectors = self.embedder.embed_passages(texts)
        
        from qdrant_client import models
        points = []
        for row, vec in zip(data, vectors):
            points.append(
                models.PointStruct(
                    id=row["id"],
                    vector=vec,
                    payload={
                        "claim": row["claim"],
                        "source": row["source"],
                        "confidence": row.get("confidence", 1.0),
                    },
                )
            )
        
        self.vector_db.upsert_points(points)
        print(f"Loaded {len(points)} entries into knowledge base")
    
    def retrieve_evidence(self, query: str, top_k: int = 20) -> List[SourcePassage]:
        """
        Retrieve relevant passages from vector DB.
        
        Args:
            query: User's claim/query text
            top_k: Number of passages to retrieve
            
        Returns:
            List of SourcePassage objects for fact validation
        """
        query_vec = self.embedder.embed_query(query)
        hits = self.vector_db.search(query_vec, top_k=top_k)
        
        passages = []
        for hit in hits:
            passages.append(
                SourcePassage(
                    content=hit.payload["claim"],
                    relevance_score=float(hit.score),
                    url=hit.payload.get("source", "unknown"),
                    domain=self._extract_domain(hit.payload.get("source", "unknown")),
                    title=hit.payload["claim"][:100],  # Use first 100 chars as title
                    published_at=datetime.now()  # Stub - would need real dates
                )
            )
        
        return passages
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL, fallback to 'unknown'"""
        if "://" in url:
            return url.split("://")[1].split("/")[0]
        return "unknown"
    
    def process_query(self, user_input: str) -> Dict[str, Any]:
        """
        Main pipeline entry point.
        
        Pipeline steps:
        1. Extract claim from user input (Danny's module)
        2. Retrieve evidence from vector DB (Adam's module)
        3. Validate claim against evidence (Sam's module)
        4. Format response for LLM/UI
        
        Args:
            user_input: Raw user query text
            
        Returns:
            Dict with:
                - claim: Extracted claim text
                - verdict: "Supported" | "Refuted" | "Not enough evidence" | "Contested"
                - score: int (0-100)
                - citations: List of citation dicts
                - features: Feature scores dict
                - raw_result: Full FactCheckResult object
        """
        
        # Step 1: Extract claim
        try:
            claim_data = extract_claim_from_input(user_input)
            # Extract the normalized claim text from Danny's JSON structure
            if isinstance(claim_data, dict) and "claims" in claim_data:
                claims = claim_data["claims"]
                if not claims:
                    return {
                        "error": "No factual claims found in input",
                        "claim": user_input,
                        "verdict": "Not enough evidence",
                        "score": 0
                    }
                claim_text = claims[0]["normalized"]
                claim_type = claims[0].get("type", "unknown")
            else:
                # Fallback if structure is different
                claim_text = user_input
                claim_type = "unknown"
        except Exception as e:
            print(f"Claim extraction failed: {e}")
            claim_text = user_input
            claim_type = "unknown"
        
        # Step 2: Retrieve evidence
        passages = self.retrieve_evidence(claim_text, top_k=20)
        
        if not passages:
            return {
                "claim": claim_text,
                "verdict": "Not enough evidence",
                "score": 0,
                "citations": [],
                "features": {},
                "message": "No relevant evidence found in knowledge base"
            }
        
        # Step 3: Fact validation
        result: FactCheckResult = self.fact_validator.validate_claim(
            claim=claim_text,
            claim_type=claim_type,
            passages=passages
        )
        
        # Step 4: Format response
        response = {
            "claim": result.claim,
            "verdict": result.verdict,
            "score": result.score,
            "citations": [
                {
                    "url": c.url,
                    "title": c.title,
                    "published_at": c.published_at.isoformat() if isinstance(c.published_at, datetime) else str(c.published_at),
                    "snippet": c.snippet
                }
                for c in result.citations
            ],
            "features": {
                "entail_max": result.features.entail_max,
                "entail_mean3": result.features.entail_mean3,
                "contradict_max": result.features.contradict_max,
                "agree_domain_count": result.features.agree_domain_count,
                "reliability_avg": result.features.releliance_score_avg,
                "recency_max": result.features.recency_weight_max
            },
            "raw_result": result  # For debugging
        }
        
        return response
    
    def format_for_ui(self, response: Dict[str, Any]) -> str:
        """
        Format response for UI display.
        
        Args:
            response: Output from process_query
            
        Returns:
            Formatted string for display
        """
        verdict_emoji = {
            "Supported": "✓",
            "Refuted": "✗",
            "Contested": "~",
            "Not enough evidence": "?"
        }
        
        emoji = verdict_emoji.get(response["verdict"], "?")
        
        output = f"""
{emoji} {response['verdict'].upper()} (Score: {response['score']}/100)

Claim: "{response['claim']}"

Evidence Summary:
- Max Support: {response['features']['entail_max']:.2f}
- Max Contradiction: {response['features']['contradict_max']:.2f}
- Agreeing Sources: {response['features']['agree_domain_count']}

Citations:
"""
        
        for i, cite in enumerate(response['citations'], 1):
            output += f"{i}. {cite['title']}\n   {cite['url']}\n   {cite['snippet'][:150]}...\n\n"
        
        return output.strip()


def main():
    """
    Demo/test of the full pipeline.
    """
    # Initialize pipeline
    pipeline = FactCheckingPipeline()
    
    # Load knowledge base (assumes data/mock.json exists)
    knowledge_path = "data/mock.json"
    if os.path.exists(knowledge_path):
        pipeline.load_knowledge_base(knowledge_path)
    else:
        print(f"Warning: Knowledge base file not found at {knowledge_path}")
        print("Creating minimal test data...")
        test_data = [
            {"id": 1, "claim": "The Moon landing occurred in 1969", "source": "https://nasa.gov", "confidence": 1.0},
            {"id": 2, "claim": "Water boils at 100°C at sea level", "source": "https://physics.edu", "confidence": 1.0},
            {"id": 3, "claim": "The Earth is approximately 4.5 billion years old", "source": "https://science.org", "confidence": 1.0}
        ]
        # Would need to save and load in real scenario
    
    # Test queries
    test_queries = [
        "Did humans land on the Moon in 1969?",
        "The Moon landing was fake",
        "What temperature does water boil at?"
    ]
    
    print("\n" + "="*80)
    print("FACT-CHECKING PIPELINE DEMO")
    print("="*80 + "\n")
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        result = pipeline.process_query(query)
        formatted = pipeline.format_for_ui(result)
        
        print(formatted)
        print("\n" + "="*80)


if __name__ == "__main__":
    main()
