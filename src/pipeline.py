"""
CMPE297 Fact-Checking System - Integration Pipeline
Connects: Input Extraction â†’ Vector DB â†’ Fact Validation â†’ LLM Response â†’ Output
"""

import json
import os
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import asdict
from qdrant_client import QdrantClient
from datetime import timezone
from modules.llm.enhanced_llm_reasoning import NBA_Statistics_Reasoner
# Module imports - adjust paths based on actual repo structure
from modules.claim_extraction.Fact_Validator import FactValidator
from modules.claim_extraction.NLIModel import NLI_LABELS, NLIModel
from modules.llm.llm_ollama import llm_ollama
from modules.misinformation_module.src.qdrant_db import QdrantDB
from modules.misinformation_module.src.embedder import E5Embedder
from modules.claim_extraction.Fact_Validator_Data_models import SourcePassage, FactCheckResult, print_fact_check_result
from modules.llm.llm_openai import llm_openai
from modules.llm.llm_reasoning import llm_reasoning 
from modules.input_extraction.input_extractor import CLAIM_TYPE, extract_claim_from_input
from modules.claim_extraction.training.Validator_Training_Data import get_training_data


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
        llm_provider: str = None,
        qdrant_url: str = None,          # <-- NEW
        qdrant_api_key: str = None       # <-- NEW
    ):
        if llm_provider is None:
            raise ValueError("llm_provider must be specified")

        if embedding_model is None:
            embedding_model = os.environ.get('EMBEDDING_MODEL', 'intfloat/e5-small-v2')

        self.use_reasoning = use_reasoning
        self.qdrant_location = qdrant_location
        self.embedder = E5Embedder(embedding_model, normalize=True)

        # ---------------------------------------------
        # USE PASSED-IN QDRANT CLOUD PARAMS
        # ---------------------------------------------
        if qdrant_url is None:
            qdrant_url = os.getenv("QDRANT_URL")

        if qdrant_api_key is None:
            qdrant_api_key = os.getenv("QDRANT_API_KEY")

        self.vector_db = QdrantDB(
            collection=os.getenv("COLLECTION_NAME", "nba_news_claims"),
            vector_size=vector_size,
            client=QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
        )

        # Choose LLM provider
        if llm_provider.lower() == "ollama":
            self.llm = llm_ollama()
        elif llm_provider.lower() == "openai":
            self.llm = llm_openai()
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}")

        self.current_llm_provider = llm_provider.lower()

        # Fact validator
        nli = NLIModel(
            emb_model_name="sentence-transformers/all-mpnet-base-v2",
            nli_model_name="roberta-large-mnli",
            nli_labels=NLI_LABELS
        )
        # self.fact_validator = FactValidator(self.llm, nli, training_data=get_training_data())
        self.fact_validator = FactValidator(self.llm, nli, training_data=None)

        # Reasoning
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
        
        print(f"Loaded {len(points)} entries into knowledge base")
        
        # Save metadata
        source_hash = self.compute_source_hash(data_path)
        self.save_metadata(data_path, source_hash)

    def _extract_domain(self, url: str) -> str:
        """Extract clean domain from a URL."""
        try:
            if "://" in url:
                return url.split("://")[1].split("/")[0]
            return url.split("/")[0]
        except Exception:
            return CLAIM_TYPE.UNKNOWN.name

    
    def retrieve_evidence(self, query: str, top_k: int = 20) -> List[SourcePassage]:
        """
        Retrieve relevant passages from vector DB.
        Supports NEWS payloads:
            • title
            • content
            • source
            • published_at
        And older CLAIM payloads:
            • claim
            • source
        """

        query_vec = self.embedder.embed_query(query)
        hits = self.vector_db.search(query_vec, top_k=top_k)

        passages = []

        for hit in hits:
            payload = hit.payload or {}

            # ----------------------------
            # Smart fallback content logic
            # ----------------------------
            content = (
                payload.get("content") or
                payload.get("summary") or
                payload.get("claim") or
                payload.get("title") or
                ""
            )

            # ----------------------------
            # Smart title logic
            # ----------------------------
            title = (
                payload.get("title") or
                payload.get("claim") or
                content[:80] or
                "Untitled"
            )

            # ----------------------------
            # Smart URL/source fallback
            # ----------------------------
            url = (
                payload.get("source") or
                payload.get("url") or
                CLAIM_TYPE.UNKNOWN.name
            )

            # ----------------------------
            # Published timestamp handling
            # ----------------------------
            published_raw = payload.get("published_at")

            if published_raw:
                try:
                    published_at = datetime.strptime(
                        published_raw, "%a, %d %b %Y %H:%M:%S %z"
                    )
                except Exception:
                    published_at = datetime.now(timezone.utc)
            else:
                published_at = datetime.now(timezone.utc)

            passages.append(
                SourcePassage(
                    content=content,
                    relevance_score=float(hit.score),
                    url=url,
                    domain=self._extract_domain(url),
                    title=title,
                    published_at=published_at
                )
            )

        return passages

    
    def process_query(self, user_input: str) -> Dict[str, Any]:
        """
        Main pipeline entry point with Logic Routing.
        """
        
        # 1. Extract claim/intent
        print("Extracting claim/intent...")
        claim_data = extract_claim_from_input(self.llm, user_input)
        
        # Get LLM's classification
        is_broad = claim_data.get("is_broad_query", False)
        sub_queries = claim_data.get("sub_search_queries", [])
        
        # ---------------------------------------------------------
        # [!] HEURISTIC OVERRIDE
        # If the input starts with Open-Ended Question words, FORCE the Broad/QA path.
        # This prevents "What team..." being treated as a True/False claim.
        # ---------------------------------------------------------
        open_ended_triggers = ("what", "which", "who", "where", "how", "list", "describe", "explain", "compare")
        if user_input.lower().strip().startswith(open_ended_triggers):
            print(f"[Pipeline] Heuristic Override: Detected open-ended question '{user_input.split()[0]}...'")
            is_broad = True
            # If LLM didn't generate sub-queries, use the user input itself
            if not sub_queries:
                sub_queries = [user_input]

        # Get the "normalized" text (or fallback to user input)
        if claim_data.get("claims"):
            claim_text = claim_data["claims"][0]["normalized"]
            claim_type = claim_data["claims"][0].get("type", "CLAIM")
        else:
            claim_text = user_input
            claim_type = "QUESTION"

        # --- BRANCH 1: BROAD QUESTION / QA MODE ---
        # We enter this if the LLM flagged it OR our Heuristic flagged it
        if is_broad:
            print(f"[Pipeline] Running QA Mode. Queries: {sub_queries}")
            
            all_passages = []
            seen_urls = set()
            
            # 1. Multi-hop Retrieval
            # If sub_queries is empty (rare case), default to claim_text
            queries_to_run = sub_queries if sub_queries else [claim_text]
            
            for query in queries_to_run:
                hits = self.retrieve_evidence(query, top_k=5)
                for hit in hits:
                    if hit.url not in seen_urls:
                        all_passages.append(hit)
                        seen_urls.add(hit.url)
            
            print(f"[Pipeline] Retrieved {len(all_passages)} unique passages for QA.")
            
            # 2. Generate Answer (Synthesis)
            # We skip the NLI Validator entirely here.
            
            context_text = "\n\n".join([f"Source ({p.domain}): {p.content}" for p in all_passages[:7]])
            
            prompt = f"""You are an intelligent assistant. Answer the user's question based ONLY on the evidence provided below.
            
            User Question: "{user_input}"
            
            Retrieved Evidence:
            {context_text}
            
            Instructions:
            1. Answer the question directly.
            2. Cite the specific sources (domains) used.
            3. If the evidence mentions conflicting info (e.g. different teams), explain the conflict.
            4. If the evidence is missing the answer, say "I couldn't find that specific information in the database."
            """
            
            summary = self.llm.message(prompt)
            
            # 3. Return "Informational" Verdict for UI
            return {
                "claim": user_input,
                "verdict": "Informational", # This triggers the 'i' icon in UI
                "score": 100, 
                "citations": [
                    {
                        "title": p.title,
                        "url": p.url,
                        "snippet": p.content[:150]
                    } for p in all_passages[:5]
                ],
                "features": {
                    "entail_max": 0, "contradict_max": 0, "agree_domain_count": 0,
                    "relevance_avg": 0, "recency_max": 0
                },
                "explanation": summary, # The actual answer
                "is_broad_answer": True
            }

        # --- BRANCH 2: SPECIFIC CLAIM VERIFICATION ---
        else:
            print("[Pipeline] Detected Specific Claim. Running Fact Verification.")
            
            # Step 2: Retrieve
            passages = self.retrieve_evidence(claim_text, top_k=20)
            
            # Step 3: Validate
            result = self.fact_validator.validate_claim(
                claim=claim_text,
                claim_type=claim_type,
                passages=passages
            )
            
            # Step 4: Format
            return {
                "verdict": result.verdict,
                "score": result.score,
                "citations": [
                     {
                        "url": c.passage.url,
                        "title": c.passage.title,
                        "snippet": c.passage.content[:200]
                    } for c in result.citations
                ],
                # Ensure features are serializable (dict)
                "features": asdict(result.features) if hasattr(result.features, "__dataclass_fields__") else result.features.__dict__,
                "explanation": self.generate_explanation(result, claim_type == "QUESTION", user_input)
            }
    
    def format_for_ui(self, response: Dict[str, Any]) -> str:
        verdict_emoji = {
            "Supported": "✓",
            "Refuted": "✗",
            "Contested": "~",
            "Not enough evidence": "?",
            "Informational": "ℹ️"  # <-- New
        }
        
        emoji = verdict_emoji.get(response["verdict"], "?")
        
        # If broad answer, show the explanation prominently
        if response.get("is_broad_answer"):
            output = f"""
{emoji} ANALYSIS
{response['explanation']}

Sources Used:
"""
        else:
            # Standard Fact Check Output
            output = f"""
{emoji} {response['verdict'].upper()} (Score: {response['score']}/100)
Evidence Summary:
- Max Support: {response['features']['entail_max']:.2f}
- Max Contradiction: {response['features']['contradict_max']:.2f}
- Agreeing Sources: {response['features']['agree_domain_count']}

Citations:
"""
        
        for i, cite in enumerate(response['citations'], 1):
            output += f"{i}. {cite['title']}\n   {cite['url']}\n   {cite['snippet'][:150]}...\n\n"
        
        return output.strip()

    def generate_explanation(self, result: FactCheckResult, is_question: bool, user_input: str) -> str:
        """Generate explanation using reasoning with full citation context"""
        print_fact_check_result(result)
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
    User Input: {user_input}
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
        
        explanation = self.reasoning_engine.reasoning_agent(question, is_question)
        
        print(f"[REASONING OUTPUT]: {explanation}...")
        
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
    qdrant_path = None
    
    llm_provider = os.environ.get('LLM_PROVIDER', 'openai')
    
    pipeline = FactCheckingPipeline(
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
