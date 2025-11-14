"""
CMPE297 Fact-Checking System - Integration Pipeline
Connects: Input Extraction â†’ Vector DB â†’ Fact Validation â†’ LLM Response â†’ Output
"""

import json
import os
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import asdict
from modules.llm.enhanced_llm_reasoning import NBA_Statistics_Reasoner
# Module imports - adjust paths based on actual repo structure
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
    """
    Main integration pipeline that orchestrates all modules.
    """
    
    def __init__(
        self,
        collection_name: str = "nba_claims",
        vector_size: int = 384,
        qdrant_location: str = None,
        embedding_model: str = None,
        use_reasoning: bool = True,
        llm_provider: str = None
    ):
        if llm_provider is None:
            raise ValueError("llm_provider must be specified")
        
        if qdrant_location is None:
            raise ValueError("qdrant_location must be specified")
        
        if embedding_model is None:
            embedding_model = os.environ.get('EMBEDDING_MODEL', 'intfloat/e5-small-v2')
        
        self.use_reasoning = use_reasoning
        self.qdrant_location = qdrant_location  # Store for metadata path
        self.embedder = E5Embedder(embedding_model, normalize=True)
        self.vector_db = QdrantDB(collection="nba_claims", vector_size=384, location=qdrant_location)

        self.vector_db.ensure_collection()
        
        if llm_provider.lower() == "ollama":
            self.llm = llm_ollama()
        elif llm_provider.lower() == "openai":
            self.llm = llm_openai()
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}. Use 'openai' or 'ollama'")
        self.current_llm_provider = llm_provider.lower()
        
        # Initialize Fact Validator
        nli = NLIModel(
            emb_model_name="sentence-transformers/all-mpnet-base-v2",
            nli_model_name="roberta-large-mnli",
            nli_labels=NLI_LABELS
        )
        self.fact_validator = FactValidator(self.llm, nli, training_data=None)  # Skip training
        # self.fact_validator = FactValidator(self.llm, nli)

        # Initialize Reasoning Engine (if enabled)
        if self.use_reasoning:
            self.reasoning_engine = NBA_Statistics_Reasoner(self.llm)
        
        print(f"Pipeline initialized:")
        print(f"  Collection: '{collection_name}'")
        print(f"  LLM: {llm_provider}")
        print(f"  Reasoning: {'enabled' if self.use_reasoning else 'disabled'}")

    # --- Runtime LLM Provider Switching ---
    def set_llm_provider(self, provider: str) -> str:
        """
        Reinitialize the pipeline's LLM at runtime.
        
        This helper is called by the Flask route `/set-llm` and keeps the startup
        behavior untouched. It validates the requested provider, rebuilds the LLM
        instance, refreshes the reasoning engine (if enabled), and returns the
        normalized provider string so callers can confirm which backend is active.
        """
        normalized = (provider or "").strip().lower()
        allowed = {"openai", "ollama"}
        if normalized not in allowed:
            raise ValueError(f"Invalid llm_provider '{provider}'. Allowed values: {sorted(allowed)}")
        
        if normalized == "ollama":
            new_llm = llm_ollama()
        else:
            new_llm = llm_openai()
        
        self.llm = new_llm
        self.current_llm_provider = normalized
        print(f"[DEBUG] Current LLM provider: {self.current_llm_provider}")
        
        # Keep the FactValidator and reasoning engine in sync with the refreshed LLM.
        if hasattr(self, "fact_validator") and self.fact_validator:
            self.fact_validator.llm = self.llm
        
        if self.use_reasoning:
            self.reasoning_engine = llm_reasoning(self.llm)
        
        return self.current_llm_provider
    
    def compute_source_hash(self, data_path: str) -> str:
        """Compute SHA256 hash of source file"""
        import hashlib
        with open(data_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def save_metadata(self, source_path: str, source_hash: str) -> None:
        """Save metadata about loaded knowledge base"""
        from pathlib import Path
        metadata = {
            "source_file": os.path.basename(source_path),
            "source_hash": source_hash,
            "embedding_model": os.environ.get('EMBEDDING_MODEL', 'intfloat/e5-small-v2'),
            "vector_size": self.vector_db.vector_size,
            "loaded_at": datetime.now().isoformat()
        }
        metadata_path = Path(self.qdrant_location) / 'metadata.json'
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load metadata about knowledge base"""
        from pathlib import Path
        metadata_path = Path(self.qdrant_location) / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def load_knowledge_base(self, data_path: str) -> None:
        """
        Load and index knowledge base into vector DB.
        
        Args:
            data_path: Path to JSON file with format:
                [{"id": int, "claim": str, "source": str, "confidence": float}, ...]
        """
        print(f"Loading source data from {os.path.basename(data_path)}...")
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        print(f"Processing {len(data)} claims...")
        
        # Batch embedding with progress
        texts = [d["claim"] for d in data]
        batch_size = 1000
        vectors = []
        
        try:
            from tqdm import tqdm
            with tqdm(total=len(texts), desc="Embedding claims", unit="claims") as pbar:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_vectors = self.embedder.embed_passages(batch)
                    vectors.extend(batch_vectors)
                    pbar.update(len(batch))
        except ImportError:
            # Fallback without tqdm
            print("Embedding claims (install tqdm for progress bar)...")
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_vectors = self.embedder.embed_passages(batch)
                vectors.extend(batch_vectors)
                if (i // batch_size) % 10 == 0:
                    print(f"  Progress: {i}/{len(texts)} claims embedded")
        
        print("Inserting into vector database...")
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
        
        # Batch insert with progress
        insert_batch_size = 100
        try:
            from tqdm import tqdm
            with tqdm(total=len(points), desc="Inserting vectors", unit="vectors") as pbar:
                for i in range(0, len(points), insert_batch_size):
                    batch = points[i:i + insert_batch_size]
                    self.vector_db.upsert_points(batch)
                    pbar.update(len(batch))
        except ImportError:
            for i in range(0, len(points), insert_batch_size):
                batch = points[i:i + insert_batch_size]
                self.vector_db.upsert_points(batch)
                if (i // insert_batch_size) % 100 == 0:
                    print(f"  Progress: {i}/{len(points)} vectors inserted")
        
        print(f"✓ Loaded {len(points)} entries into knowledge base")
        
        # Save metadata
        source_hash = self.compute_source_hash(data_path)
        self.save_metadata(data_path, source_hash)
    
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

        # Call the currently selected LLM with the raw user text so its response can
        # be returned alongside the fact-check verdict.
        llm_response = None
        try:
            llm_response = self.llm.message(user_input)
            preview = (llm_response or "None")[:100]
            print(f"[process_query] LLM response preview: {preview}")
        except Exception as llm_error:
            print(f"LLM call failed: {llm_error}")
        try:
            print("Extracting claim from user input...")
            claim_data = extract_claim_from_input(self.llm, user_input)
            print("Extracted claim data:", claim_data)
            if isinstance(claim_data, dict) and "claims" in claim_data:
                claims = claim_data["claims"]
                if not claims:
                    return {
                        "claim": user_input,
                        "verdict": "Not enough evidence",
                        "score": 0,
                        "citations": [],
                        "features": {},
                        "message": "No factual claims found in input"
                    }
                claim_text = claims[0]["normalized"]
                claim_type = claims[0].get("type", "unknown")
            else:
                claim_text = user_input
                claim_type = "unknown"
        except Exception as e:
            print(f"Claim extraction failed: {e}")
            claim_text = user_input
            claim_type = "unknown"
        
        # Step 2: Retrieve evidence
        print("Retrieving evidence from knowledge base...")
        passages = self.retrieve_evidence(claim_text, top_k=20)
        print(f"Retrieved {len(passages)} passages")
        
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
        print("Validating claim against evidence...")
        result: FactCheckResult = self.fact_validator.validate_claim(
            claim=claim_text,
            claim_type=claim_type,
            passages=passages
        )
        print(f"Validation result: Verdict={result.verdict}, Score={result.score}")
        
        # Step 4: Format response
        response = {
            "claim": result.claim,
            "verdict": result.verdict,
            "score": result.score,
            "citations": [
                {
                    "url": c.passage.url,
                    "title": c.passage.title,
                    "published_at": c.passage.published_at.isoformat() if isinstance(c.passage.published_at, datetime) else str(c.passage.published_at),
                    "snippet": c.passage.content[:200]
                }
                for c in result.citations
            ],
            "features": {
                "entail_max": result.features.entail_max,
                "entail_mean3": result.features.entail_mean3,
                "contradict_max": result.features.contradict_max,
                "agree_domain_count": result.features.agree_domain_count,
                "relevance_avg": result.features.relevance_score_avg,
                "recency_max": result.features.recency_weight_max
            },
            "raw_result": result,  # For debugging
            "explanation": self.generate_explanation(result),
            "llm_response": llm_response  # Surface direct model output for the UI if needed

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
            "Supported": "âœ“",
            "Refuted": "âœ—",
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

    def generate_explanation(self, result: FactCheckResult) -> str:
        """Generate explanation using reasoning with full citation context"""
        
        # Use all_evidence if available, fall back to citations
        evidence_to_analyze = result.all_evidence if result.all_evidence else result.citations
        
        print(f"\n[REASONING INPUT]")
        print(f"  Claim: {result.claim}")
        print(f"  Verdict: {result.verdict}")
        print(f"  Score: {result.score}")
        print(f"  Evidence passages: {len(evidence_to_analyze)}")
        
        if not self.use_reasoning:
            prompt = f"Explain this verdict: {result.claim} is {result.verdict} (score: {result.score}/100)"
            return self.llm.message(prompt)
        
        # Build citation context with NLI scores from ALL evidence
        citation_details = []
        for i, c in enumerate(evidence_to_analyze, 1):
            nli_info = f"[entail={c.entail_prob:.2f}, contradict={c.contradict_prob:.2f}]"
            content = c.passage.content[:300].strip()
            citation_details.append(f"{i}. {nli_info} {content}")
        
        citations_text = "\n".join(citation_details)
        
        question = f"""Analyze this fact-check result:

    Claim: {result.claim}
    Verdict: {result.verdict}
    Score: {result.score}/100

    Retrieved Evidence ({len(evidence_to_analyze)} passages with NLI scores):
    {citations_text}

    Explain why this verdict was reached, focusing on:
    1. Which passages support vs contradict the claim
    2. Any temporal or contextual conflicts in the evidence
    3. Why the score is {result.score}/100"""
        
        print(f"[REASONING PROMPT]:\n{question[:500]}...")
        
        explanation = self.reasoning_engine.reasoning_agent(question)
        
        print(f"[REASONING OUTPUT]: {explanation[:200]}...")
        
        return explanation


def main():
    """
    Demo/test of the full pipeline.
    """
    # Initialize pipeline with required parameters
    from pathlib import Path
    import os
    from dotenv import load_dotenv
    
    # Load environment
    load_dotenv()
    
    # Compute paths relative to project root
    project_root = Path(__file__).parent
    qdrant_path = str(project_root / "data" / "qdrant")
    
    llm_provider = os.environ.get('LLM_PROVIDER', 'openai')
    
    pipeline = FactCheckingPipeline(
        qdrant_location=qdrant_path,
        llm_provider=llm_provider
    )
    
    # Load knowledge base (assumes data/mock.json exists)
    knowledge_path = "data/mock.json"
    if os.path.exists(knowledge_path):
        pipeline.load_knowledge_base(knowledge_path)
    else:
        print(f"Warning: Knowledge base file not found at {knowledge_path}")
        print("Creating minimal test data...")
        test_data = [
            {"id": 1, "claim": "The Moon landing occurred in 1969", "source": "https://nasa.gov", "confidence": 1.0},
            {"id": 2, "claim": "Water boils at 100Â°C at sea level", "source": "https://physics.edu", "confidence": 1.0},
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
