from dataclasses import dataclass
from typing import List, Literal
from typing import Tuple
from typing import List, Tuple


VerdictType = Literal["Supported", "Refuted", "Not enough evidence", "Contested"]
class ModelInterface:
    def predict(self, inputs: List[Tuple[str, str]]) -> List[Tuple[float, float, float]]:
        """Stub for NLI model prediction."""
        pass

class SourcePassage:
    def __init__(self, content=None, domain=None, url=None, relevance_score=0, title=None, published_at=None):
        self.content = content
        self.domain = domain
        self.url = url
        self.relevance_score = relevance_score
        self.title = title
        self.published_at = published_at
@dataclass
class Citation:
    def __init__(self, passage):
        self.passage = passage
@dataclass
class CitationValidationScoring(Citation):
    passage: SourcePassage
    entail_prob: float = 0.0
    contradict_prob: float = 0.0
    neutral_prob: float = 0.0
    recency_weight: float = 0.0
    numeric_date_ok: bool = False
@dataclass
class FactCheckFeatures:
    entail_max: float
    entail_mean3: float
    contradict_max: float
    agree_domain_count: int
    relevance_score_avg: float
    recency_weight_max: float
    contest_score: float = 0.0
@dataclass
class FactCheckResult:
    claim: str
    verdict: VerdictType
    score: int
    citations: List[CitationValidationScoring]  # Top 3 for frontend
    features: FactCheckFeatures
    all_evidence: List[CitationValidationScoring] = None  # All valid results for reasoning


import textwrap
from dataclasses import dataclass
from typing import List, Literal, Tuple

# --- Your Provided Classes (Included for context) ---
VerdictType = Literal["Supported", "Refuted", "Not enough evidence", "Contested"]

class ModelInterface:
    def predict(self, inputs: List[Tuple[str, str]]) -> List[Tuple[float, float, float]]:
        pass

class SourcePassage:
    def __init__(self, content=None, domain=None, url=None, relevance_score=0, title=None, published_at=None):
        self.content = content
        self.domain = domain
        self.url = url
        self.relevance_score = relevance_score
        self.title = title
        self.published_at = published_at

@dataclass
class Citation:
    def __init__(self, passage):
        self.passage = passage

@dataclass
class CitationValidationScoring(Citation):
    passage: SourcePassage
    entail_prob: float = 0.0
    contradict_prob: float = 0.0
    neutral_prob: float = 0.0
    recency_weight: float = 0.0
    numeric_date_ok: bool = False

@dataclass
class FactCheckFeatures:
    entail_max: float
    entail_mean3: float
    contradict_max: float
    contradict_mean3: float  
    neutral_mean: float    
    agree_domain_count: int
    relevance_score_avg: float
    recency_weight_max: float
    contest_score: float = 0.0

@dataclass
class FactCheckResult:
    claim: str
    verdict: VerdictType
    score: int
    citations: List[CitationValidationScoring]
    features: FactCheckFeatures
    all_evidence: List[CitationValidationScoring] = None

# --- The Print Function ---

def print_fact_check_result(result: FactCheckResult):
    """
    Pretty-prints a FactCheckResult object in a structured, readable report format.
    """
    # ANSI Colors for terminal output
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    # Determine color based on verdict
    v_color = RESET
    if result.verdict == "Supported": v_color = GREEN
    elif result.verdict == "Refuted": v_color = RED
    elif result.verdict == "Contested": v_color = YELLOW
    elif result.verdict == "Not enough evidence": v_color = BLUE

    print("\n" + "="*60)
    print(f"{BOLD}FACT CHECK REPORT{RESET}")
    print("="*60)

    # 1. Claim and Verdict
    print(f"\n{BOLD}Claim:{RESET}")
    print(textwrap.fill(result.claim, width=80, initial_indent="  ", subsequent_indent="  "))
    
    print(f"\n{BOLD}Verdict:{RESET} {v_color}{result.verdict}{RESET}")
    print(f"{BOLD}Confidence Score:{RESET} {result.score}/100")

    # 2. Features (Why did we get this result?)
    f = result.features
    print(f"\n{BOLD}Decision Features:{RESET}")
    print(f"  • Max Entailment:   {f.entail_max:.4f}")
    print(f"  • Max Contradiction:{f.contradict_max:.4f}")
    print(f"  • Domain Agreement: {f.agree_domain_count} sources")
    print(f"  • Avg Relevance:    {f.relevance_score_avg:.4f}")
    print(f"  • Contest Score:    {f.contest_score:.4f}")

    # 3. Top Citations
    print(f"\n{BOLD}Top Evidence ({len(result.citations)}):{RESET}")
    
    for i, cite in enumerate(result.citations, 1):
        p = cite.passage
        
        # Header for the citation
        title = p.title if p.title else "No Title"
        domain = p.domain if p.domain else "Unknown Domain"
        date = p.published_at if p.published_at else "N/A"
        
        print(f"\n  {BOLD}{i}. [{domain}] {title}{RESET}")
        print(f"     Date: {date} | Relevance: {p.relevance_score:.2f}")
        print(f"     URL:  {p.url}")
        
        # Validation scores for this specific citation
        print(f"     {BOLD}NLI Analysis:{RESET} [Entail: {cite.entail_prob:.2f} | Contradict: {cite.contradict_prob:.2f} | Neutral: {cite.neutral_prob:.2f}]")
        
        # Content snippet
        content_snippet = textwrap.shorten(p.content, width=200, placeholder="...")
        print(f"     {BOLD}Excerpt:{RESET} \"{content_snippet}\"")

    # 4. Footer
    total_evidence = len(result.all_evidence) if result.all_evidence else 0
    print("-" * 60)
    print(f"Total evidence processed: {total_evidence} documents")
    print("=" * 60 + "\n")