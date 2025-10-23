"""
CMPE297 Fact-Checking System Architecture
Integration and Testing Framework

Module Flow:
User Input → Claim Extraction → Vector DB → Fact Checking → LLM Reasoning → UI → User

Module Owners:
- Claim Extraction: Danny
- Vector DB: Adam
- Fact Checking: Sam
- LLM Engine: Akshay
- UI Interface: Yuxiao
- Integration: Stephen

Dependencies:
- Claim Extraction → Vector DB (claim object structure)
- Vector DB → Fact Checking (evidence passages with relevance scores)
- Fact Checking → LLM Engine (verdict, score, features, citations)
- LLM Engine → UI (natural language explanation)
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class Claim:
    """
    Output from Claim Extraction module.
    Defined by Danny's contract (NEEDS VALIDATION).
    """
    text: str                    # The factual claim statement
    category: Optional[str]      # Topic category
    subject: Optional[str]       # Main subject
    objects: Optional[List[str]] # Related entities
    confidence: float            # Extraction confidence (0-1)
    # TODO: Verify exact structure with Danny


@dataclass
class EvidencePassage:
    """
    Output from Vector DB retrieval.
    Based on Adam's contract and Sam's requirements.
    """
    id: int
    text: str                    # Passage content
    source_url: str              # Original source URL
    source_domain: str           # Domain name
    source_title: str            # Document title
    published_at: Optional[str]  # Publication date (ISO format)
    relevance_score: float       # Cosine similarity from Qdrant (0-1)
    reliability_score: Optional[float]  # Source reliability (0-1)


@dataclass
class FactCheckFeatures:
    """
    Intermediate features calculated by fact checker.
    As defined in Sam's module specification.
    """
    e_max: float        # Max entailment probability
    e_mean3: float      # Mean entailment of top-3 passages
    c_max: float        # Max contradiction probability
    agree_dom: int      # Number of agreeing unique domains
    rel_avg: float      # Average source reliability over top-3
    rec_max: float      # Max recency weight (0-1)


class Verdict(Enum):
    """Fact-checking verdict categories."""
    SUPPORTED = "Supported"
    REFUTED = "Refuted"
    NOT_ENOUGH_EVIDENCE = "Not enough evidence"
    CONTESTED = "Contested"  # Optional extension


@dataclass
class Citation:
    """Citation for fact-check response."""
    url: str
    title: str
    published_at: Optional[str]
    snippet: str


@dataclass
class FactCheckResult:
    """
    Output from Fact Checking module.
    Defined by Sam's module specification.
    Input to LLM Engine.
    """
    claim: str
    verdict: Verdict
    score: int  # 0-100
    citations: List[Citation]  # 2-3 citations
    features: FactCheckFeatures


@dataclass
class LLMResponse:
    """
    Output from LLM Engine.
    Based on Akshay's OpenAI API contract.
    """
    content: str              # Final answer from assistant
    trace: Optional[str]      # Reasoning steps (if trace=True in request)
    finish_reason: str        # "stop" | "length" | "error"
    usage: Dict[str, int]     # Token usage: prompt_tokens, completion_tokens, total_tokens


# ==============================================================================
# MODULE: Claim Extraction (Danny)
# ==============================================================================

class ClaimExtractor:
    """
    Parses user input and extracts factual claims.
    
    Owner: Danny
    
    Input: Raw user text (str)
    Output: List[Claim]
    
    Contract Status: MISSING - needs Danny's document
    """
    
    def extract_claims(self, user_input: str) -> List[Claim]:
        """
        Parse user text and extract factual statements.
        
        Args:
            user_input: Raw text from user
            
        Returns:
            List of structured Claim objects
            
        Notes:
            - May return multiple claims from single input
            - Filters out opinions (keeps only factual statements)
            - Assigns category, subject, objects metadata
        """
        raise NotImplementedError("Danny's implementation needed")


# ==============================================================================
# MODULE: Vector Database (Adam)
# ==============================================================================

class VectorDatabase:
    """
    Stores and retrieves evidence using Qdrant.
    
    Owner: Adam
    
    Input: Claim object (or query string)
    Output: List[EvidencePassage] with relevance scores
    
    Implementation:
    - Embedding model: intfloat/e5-small-v2 (Hugging Face)
    - Database: Qdrant (open source)
    - Scoring: Cosine similarity
    
    Mock data format:
    {
        "id": int,
        "claim": str,
        "source": str,
        "confidence": float
    }
    """
    
    def __init__(self, qdrant_url: str, embedding_model: str = "intfloat/e5-small-v2"):
        """
        Initialize connection to Qdrant and load embedding model.
        
        Args:
            qdrant_url: Qdrant server URL
            embedding_model: Hugging Face model identifier (default: intfloat/e5-small-v2)
        """
        self.qdrant_url = qdrant_url
        self.embedding_model = embedding_model
        # TODO: Initialize Qdrant client
        # TODO: Load embedding model from Hugging Face
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split long documents into meaningful chunks.
        
        Args:
            documents: List of raw documents
            
        Returns:
            List of chunked documents with metadata
            
        Notes:
            - Chunks by title/category for long documents
            - Preserves source metadata on each chunk
        """
        raise NotImplementedError("Adam's implementation needed")
    
    def store_embeddings(self, chunks: List[Dict]) -> None:
        """
        Generate embeddings and store in Qdrant.
        
        Args:
            chunks: Document chunks with metadata
        """
        raise NotImplementedError("Adam's implementation needed")
    
    def retrieve_evidence(
        self, 
        claim: Claim, 
        top_k: int = 20
    ) -> List[EvidencePassage]:
        """
        Query Qdrant for semantically similar passages.
        
        Args:
            claim: Claim object to find evidence for
            top_k: Number of passages to retrieve (default: 20 per Sam's spec)
            
        Returns:
            List of evidence passages with relevance scores
            
        Notes:
            - Returns cosine similarity as relevance_score
            - Sorted by relevance (highest first)
            - Sam's module expects k >= 20 passages
        """
        raise NotImplementedError("Adam's implementation needed")


# ==============================================================================
# MODULE: Fact Checking (Sam)
# ==============================================================================

class FactChecker:
    """
    Evaluates claim truthfulness against evidence.
    
    Owner: Sam
    
    Input: 
        - claim: Claim object
        - evidence: List[EvidencePassage] from Vector DB
        
    Output: FactCheckResult
    
    Algorithm: As specified in "Fact Checking Module by Samuel Tsao"
    """
    
    def __init__(self, nli_model: Optional[str] = None):
        """
        Initialize fact checker with NLI model.
        
        Args:
            nli_model: Natural Language Inference model for entailment/contradiction
        """
        self.nli_model = nli_model
        # TODO: Load NLI model
    
    def preprocess_passages(
        self, 
        passages: List[EvidencePassage]
    ) -> List[EvidencePassage]:
        """
        De-duplicate and filter passages.
        
        Steps:
        1. De-duplicate by URL/domain
        2. Drop very short/very long passages
        3. Compute recency weight from published_at
        
        Returns:
            Cleaned list of passages
        """
        raise NotImplementedError("Sam's implementation needed")
    
    def compute_nli_scores(
        self, 
        claim: str, 
        passages: List[EvidencePassage]
    ) -> List[Tuple[float, float, float]]:
        """
        Run NLI model on (claim, passage) pairs.
        
        Returns:
            List of (entail_prob, contradict_prob, neutral_prob) for each passage
        """
        raise NotImplementedError("Sam's implementation needed")
    
    def check_numeric_dates(
        self, 
        claim: str, 
        passage: str
    ) -> bool:
        """
        Parse and compare numbers/dates with tolerance.
        
        Tolerance: ±2% or ±1 unit
        
        Returns:
            True if numeric/date values match within tolerance
        """
        raise NotImplementedError("Sam's implementation needed")
    
    def calculate_features(
        self,
        nli_scores: List[Tuple[float, float, float]],
        passages: List[EvidencePassage],
        top_n: int = 5
    ) -> FactCheckFeatures:
        """
        Calculate features from top-N re-ranked passages.
        
        Features (as per Sam's spec):
        - e_max: max entailment probability
        - e_mean3: mean entailment of top-3
        - c_max: max contradiction probability
        - agree_dom: # agreeing unique domains (entail >= 0.6, contradict < 0.5)
        - rel_avg: average source reliability over top-3
        - rec_max: max recency weight
        
        Args:
            nli_scores: NLI probabilities for each passage
            passages: Evidence passages with metadata
            top_n: Number of top passages to use (default: 5-8)
            
        Returns:
            FactCheckFeatures object
        """
        raise NotImplementedError("Sam's implementation needed")
    
    def calculate_score(self, features: FactCheckFeatures) -> int:
        """
        Calculate final score (0-100) from features.
        
        Formula:
        raw = 0.40*e_max + 0.20*e_mean3 + 0.15*min(agree_dom/3, 1)
              + 0.15*rel_avg + 0.10*rec_max - 0.25*c_max
              
        score = round(100 * sigmoid((raw - 0.5) / 0.15))
        
        Returns:
            Score in range [0, 100]
        """
        raise NotImplementedError("Sam's implementation needed")
    
    def determine_verdict(self, features: FactCheckFeatures) -> Verdict:
        """
        Determine verdict from features.
        
        Rules:
        - Supported: e_max >= 0.70 AND agree_dom >= 2 AND c_max < 0.40
        - Refuted: c_max >= 0.70
        - Contested (optional): e_max >= 0.70 AND c_max >= 0.50
        - Not enough evidence: otherwise
        
        Returns:
            Verdict enum
        """
        raise NotImplementedError("Sam's implementation needed")
    
    def select_citations(
        self,
        passages: List[EvidencePassage],
        nli_scores: List[Tuple[float, float, float]]
    ) -> List[Citation]:
        """
        Select 2-3 best citations.
        
        Selection criteria:
        - Highest entailment score
        - Diverse domains
        - Most recent
        
        Returns:
            List of 2-3 Citation objects
        """
        raise NotImplementedError("Sam's implementation needed")
    
    def check_claim(
        self, 
        claim: Claim, 
        evidence_passages: List[EvidencePassage]
    ) -> FactCheckResult:
        """
        Main entry point: Check claim against evidence.
        
        Pipeline:
        1. Preprocess passages (dedupe, filter, recency weight)
        2. Run NLI checks (entailment/contradiction)
        3. Run numeric/date checks
        4. Calculate features
        5. Calculate score
        6. Determine verdict
        7. Select citations
        
        Args:
            claim: Claim object from extraction module
            evidence_passages: Retrieved passages from Vector DB (k >= 20)
            
        Returns:
            FactCheckResult with verdict, score, citations, features
        """
        raise NotImplementedError("Sam's implementation needed")


# ==============================================================================
# MODULE: LLM Reasoning Engine (Akshay)
# ==============================================================================

class LLMEngine:
    """
    Generates natural language explanation from verification data.
    
    Owner: Akshay
    
    Input: FactCheckResult
    Output: LLMResponse
    
    API: OpenAI Reasoning API (gpt-4-reasoner-v1)
    Endpoint: POST /v1/reasoning/completions
    
    Contract provided: API_Contact.txt
    """
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-4-reasoner-v1",
        use_trace: bool = True,
        temperature: float = 0.0
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model identifier (default: gpt-4-reasoner-v1)
            use_trace: Whether to request reasoning trace (default: True)
            temperature: Randomness control (default: 0.0 for deterministic)
        """
        self.api_key = api_key
        self.model = model
        self.use_trace = use_trace
        self.temperature = temperature
        self.endpoint = "https://api.openai.com/v1/reasoning/completions"
        # TODO: Initialize OpenAI client
    
    def construct_prompt(
        self, 
        claim: str, 
        fact_check_result: FactCheckResult
    ) -> List[Dict[str, str]]:
        """
        Build prompt messages with verification data.
        
        Prompt should include:
        - System message: Role and task description
        - User message: Original claim, verdict, score, features, citations
        
        Returns:
            List of message dicts in format:
            [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."}
            ]
        """
        system_msg = {
            "role": "system",
            "content": (
                "You are a fact-checking explanation assistant. "
                "Given a claim and verification results, provide a clear, "
                "concise explanation of the verdict with supporting evidence."
            )
        }
        
        user_content = f"""
Claim: {claim}

Verification Results:
- Verdict: {fact_check_result.verdict.value}
- Confidence Score: {fact_check_result.score}/100

Evidence Analysis:
- Max Entailment: {fact_check_result.features.e_max:.2f}
- Max Contradiction: {fact_check_result.features.c_max:.2f}
- Agreeing Domains: {fact_check_result.features.agree_dom}
- Average Source Reliability: {fact_check_result.features.rel_avg:.2f}

Citations:
"""
        for i, cit in enumerate(fact_check_result.citations, 1):
            user_content += f"{i}. {cit.title}\n   {cit.url}\n   {cit.snippet}\n\n"
        
        user_content += (
            "Provide a natural language explanation of this verdict, "
            "citing the key evidence. Be concise and factual."
        )
        
        user_msg = {"role": "user", "content": user_content}
        
        return [system_msg, user_msg]
    
    def call_api(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 500
    ) -> Dict:
        """
        Call OpenAI Reasoning API.
        
        Request format (per API_Contact.txt):
        {
            "model": "gpt-4-reasoner-v1",
            "prompt": [messages],
            "temperature": 0.0,
            "max_tokens": 500,
            "trace": true,
            "stream": false
        }
        
        Response format:
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "final answer",
                        "trace": "reasoning steps"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {...}
        }
        
        Args:
            messages: List of message dicts
            max_tokens: Maximum tokens in response
            
        Returns:
            API response dict
        """
        import requests
        
        payload = {
            "model": self.model,
            "prompt": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "trace": self.use_trace,
            "stream": False
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            self.endpoint,
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        
        return response.json()
    
    def generate_explanation(
        self, 
        claim: Claim, 
        fact_check_result: FactCheckResult
    ) -> LLMResponse:
        """
        Generate human-readable explanation.
        
        Steps:
        1. Construct prompt from fact-check result
        2. Call OpenAI Reasoning API
        3. Parse response
        4. Return formatted explanation
        
        Args:
            claim: Original claim
            fact_check_result: Output from fact checker
            
        Returns:
            LLMResponse with explanation and reasoning trace
        """
        # 1. Construct prompt
        messages = self.construct_prompt(claim.text, fact_check_result)
        
        # 2. Call API
        api_response = self.call_api(messages)
        
        # 3. Parse response
        choice = api_response["choices"][0]
        message = choice["message"]
        
        return LLMResponse(
            content=message["content"],
            trace=message.get("trace"),  # Optional, present if use_trace=True
            finish_reason=choice["finish_reason"],
            usage=api_response["usage"]
        )


# ==============================================================================
# MODULE: UI Interface (Yuxiao)
# ==============================================================================

class UIInterface:
    """
    Handles user interaction and response formatting.
    
    Owner: Yuxiao
    
    Responsibilities:
    - Receive user input
    - Display formatted responses
    - Manage conversation state
    - Format citations and scores
    """
    
    def process_user_input(self, query: str) -> str:
        """
        Receive and validate user input.
        
        Args:
            query: Raw user text
            
        Returns:
            Cleaned query string
        """
        raise NotImplementedError("Yuxiao's implementation needed")
    
    def format_response(
        self,
        llm_response: LLMResponse,
        fact_check_result: FactCheckResult
    ) -> str:
        """
        Format complete response for display.
        
        Should include:
        - LLM explanation (llm_response.content)
        - Reasoning trace if available (llm_response.trace)
        - Verdict badge/indicator
        - Score visualization
        - Citations with links
        
        Args:
            llm_response: LLM-generated explanation
            fact_check_result: Fact-check verdict and metadata
            
        Returns:
            Formatted response string (HTML/markdown/text)
        """
        raise NotImplementedError("Yuxiao's implementation needed")
    
    def display_response(self, formatted_response: str) -> None:
        """
        Display response to user.
        
        Args:
            formatted_response: Pre-formatted response string
        """
        raise NotImplementedError("Yuxiao's implementation needed")


# ==============================================================================
# INTEGRATION: Pipeline
# ==============================================================================

class FactCheckingPipeline:
    """
    Coordinates all modules in sequence.
    
    Owner: Stephen
    """
    
    def __init__(
        self,
        claim_extractor: ClaimExtractor,
        vector_db: VectorDatabase,
        fact_checker: FactChecker,
        llm_engine: LLMEngine,
        ui_interface: UIInterface
    ):
        """Initialize pipeline with all modules."""
        self.claim_extractor = claim_extractor
        self.vector_db = vector_db
        self.fact_checker = fact_checker
        self.llm_engine = llm_engine
        self.ui_interface = ui_interface
    
    def process_query(self, user_query: str) -> str:
        """
        Process user query through full pipeline.
        
        Steps:
        1. Extract claims from user input (Danny)
        2. Retrieve evidence from vector DB (Adam)
        3. Check claim against evidence (Sam)
        4. Generate explanation (Akshay)
        5. Format and display response (Yuxiao)
        
        Args:
            user_query: Raw user text
            
        Returns:
            Formatted response string
        """
        # 1. Extract claims
        claims = self.claim_extractor.extract_claims(user_query)
        
        # TODO: Handle multiple claims - for now take first
        if not claims:
            return "No factual claims found in input."
        
        claim = claims[0]
        
        # 2. Retrieve evidence
        evidence = self.vector_db.retrieve_evidence(claim, top_k=20)
        
        # 3. Check claim
        fact_check_result = self.fact_checker.check_claim(claim, evidence)
        
        # 4. Generate explanation
        llm_response = self.llm_engine.generate_explanation(
            claim, 
            fact_check_result
        )
        
        # 5. Format response
        formatted = self.ui_interface.format_response(
            llm_response,
            fact_check_result
        )
        
        return formatted


# ==============================================================================
# TESTING: Module Integration Tests
# ==============================================================================

class TestModuleIntegration:
    """
    Tests connections between modules.
    
    Owner: Stephen
    """
    
    def test_claim_to_vector_db(self):
        """
        Verify claim extraction output works with vector DB query.
        
        Test:
        1. Extract claim from test input
        2. Pass to vector DB retrieve_evidence
        3. Verify no type errors
        """
        raise NotImplementedError("Stephen's test implementation needed")
    
    def test_vector_db_to_fact_checker(self):
        """
        Verify vector DB output format matches fact checker input.
        
        Test:
        1. Retrieve evidence from vector DB
        2. Pass to fact_checker.check_claim
        3. Verify no type errors
        4. Verify all required fields present
        """
        raise NotImplementedError("Stephen's test implementation needed")
    
    def test_fact_checker_to_llm(self):
        """
        Verify fact checker output format matches LLM input.
        
        Test:
        1. Create mock FactCheckResult
        2. Pass to llm_engine.generate_explanation
        3. Verify no type errors
        """
        raise NotImplementedError("Stephen's test implementation needed")
    
    def test_llm_to_ui(self):
        """
        Verify LLM output can be displayed by UI.
        
        Test:
        1. Create mock LLMResponse
        2. Pass to ui_interface.format_response
        3. Verify formatted output is valid
        """
        raise NotImplementedError("Stephen's test implementation needed")


# ==============================================================================
# TESTING: End-to-End Pipeline Tests
# ==============================================================================

class TestEndToEnd:
    """
    Tests complete pipeline with validation data.
    
    Owner: Stephen
    
    Requirements:
    - Dataset must be selected and available
    - All modules must have working implementations
    """
    
    def test_pipeline_true_claim(self):
        """
        Test pipeline with known true claim from dataset.
        
        Expected:
        - Verdict: Supported
        - Score: >= 70
        """
        raise NotImplementedError("Stephen's test implementation needed")
    
    def test_pipeline_false_claim(self):
        """
        Test pipeline with known false claim from dataset.
        
        Expected:
        - Verdict: Refuted
        - Score: <= 30
        """
        raise NotImplementedError("Stephen's test implementation needed")
    
    def test_pipeline_no_evidence(self):
        """
        Test pipeline with claim lacking evidence in DB.
        
        Expected:
        - Verdict: Not enough evidence
        """
        raise NotImplementedError("Stephen's test implementation needed")
    
    def test_pipeline_latency(self):
        """
        Measure end-to-end response time.
        
        Target: < 5 seconds (TO BE DEFINED by team)
        """
        raise NotImplementedError("Stephen's test implementation needed")


# ==============================================================================
# TESTING: Module Unit Tests
# ==============================================================================

class TestClaimExtraction:
    """Tests for claim extraction module."""
    
    def test_extract_single_claim(self):
        """
        Input: Text with one factual claim
        Expected: List with one Claim object
        """
        raise NotImplementedError("Danny's test implementation needed")
    
    def test_extract_multiple_claims(self):
        """
        Input: Text with multiple claims
        Expected: List with all claims extracted
        """
        raise NotImplementedError("Danny's test implementation needed")
    
    def test_filter_opinions(self):
        """
        Input: Text with opinions and facts
        Expected: Only factual claims extracted
        """
        raise NotImplementedError("Danny's test implementation needed")


class TestVectorDatabase:
    """Tests for vector database module."""
    
    def test_store_and_retrieve(self):
        """
        Test basic store/retrieve cycle.
        
        Steps:
        1. Store test documents
        2. Query with similar text
        3. Verify retrieval works
        """
        raise NotImplementedError("Adam's test implementation needed")
    
    def test_relevance_scores(self):
        """
        Verify relevance scores are in valid range.
        
        Expected: All scores in [0, 1]
        """
        raise NotImplementedError("Adam's test implementation needed")
    
    def test_chunking(self):
        """
        Test document chunking logic.
        
        Input: Long document
        Expected: Multiple chunks with preserved metadata
        """
        raise NotImplementedError("Adam's test implementation needed")


class TestFactChecker:
    """Tests for fact-checking module."""
    
    def test_supported_claim(self):
        """
        Input: Claim + strong supporting evidence
        Expected: Verdict.SUPPORTED, high score
        """
        raise NotImplementedError("Sam's test implementation needed")
    
    def test_refuted_claim(self):
        """
        Input: Claim + strong contradicting evidence
        Expected: Verdict.REFUTED, low score
        """
        raise NotImplementedError("Sam's test implementation needed")
    
    def test_score_calculation(self):
        """
        Verify score calculation formula.
        
        Tests:
        - Score in [0, 100]
        - Matches verdict
        """
        raise NotImplementedError("Sam's test implementation needed")
    
    def test_numeric_check(self):
        """
        Test numeric/date comparison with tolerance.
        
        Input: Claims with numbers/dates
        Expected: Correct tolerance handling (±2% or ±1 unit)
        """
        raise NotImplementedError("Sam's test implementation needed")


class TestLLMEngine:
    """Tests for LLM reasoning engine."""
    
    def test_generate_explanation(self):
        """
        Input: Mock FactCheckResult
        Expected: Valid explanation generated
        """
        raise NotImplementedError("Akshay's test implementation needed")
    
    def test_api_error_handling(self):
        """
        Test graceful handling of API failures.
        
        Scenarios:
        - Network timeout
        - Invalid API key
        - Rate limit
        """
        raise NotImplementedError("Akshay's test implementation needed")
    
    def test_prompt_construction(self):
        """
        Verify prompt includes all necessary information.
        
        Expected in prompt:
        - Claim text
        - Verdict
        - Score
        - Citations
        """
        raise NotImplementedError("Akshay's test implementation needed")


class TestUIInterface:
    """Tests for UI interface module."""
    
    def test_format_response(self):
        """
        Input: LLMResponse + FactCheckResult
        Expected: Properly formatted output
        """
        raise NotImplementedError("Yuxiao's test implementation needed")
    
    def test_citation_formatting(self):
        """
        Verify citations are displayed correctly.
        
        Expected:
        - Clickable links
        - Source metadata visible
        """
        raise NotImplementedError("Yuxiao's test implementation needed")


# ==============================================================================
# CRITICAL BLOCKERS
# ==============================================================================

"""
BLOCKERS TO RESOLVE BEFORE INTEGRATION:

1. MISSING CONTRACTS:
   - Danny: Exact Claim object structure
   - Akshay: ✓ API contract provided (API_Contact.txt)
     * Using OpenAI Reasoning API (gpt-4-reasoner-v1)
     * Response includes content + optional trace

2. DATA STRUCTURE ALIGNMENT:
   - Adam's mock data uses: {"id", "claim", "source", "confidence"}
   - EvidencePassage expects: {"id", "text", "source_url", "source_domain", "source_title", "published_at", "relevance_score", "reliability_score"}
   - Need to align mock data format with actual dataset format
   - Dataset selection required to determine final structure

3. DATASET DECISION:
   - Which dataset from fact_checking_datasets.md
   - Who will download and prepare
   - Where will it be stored

4. INTERFACE VALIDATION:
   - Danny → Adam: Claim format
   - Adam → Sam: EvidencePassage format (verify relevance_score field)
   - Sam → Akshay: ✓ FactCheckResult format defined
   - Akshay → Yuxiao: ✓ LLMResponse format defined (content, trace, finish_reason, usage)

5. PERFORMANCE TARGETS:
   - Acceptable end-to-end latency
   - Per-module latency constraints
   - OpenAI API latency: 1-3 seconds

6. ERROR HANDLING:
   - Pipeline module failure handling
   - Fallback behaviors
   - OpenAI API errors: rate limits, timeouts, invalid keys
"""