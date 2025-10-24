from abc import ABC
from typing import List, Literal
from dataclasses import dataclass, field
from datetime import datetime
from modules.llm.llm_engine_interface import LLMInterface

# --- Unmodified Classes (default repr is fine) ---

@dataclass
class SourcePassage:
    """Represents a single retrieved passage from the RAG interface."""
    content: str
    relevance_score: float
    url: str
    domain: str
    title: str
    published_at: datetime

@dataclass
class ClaimCheckResult:
    """Represents the results of the claim checks (NLI, Numeric/Date, etc.)."""
    passage: SourcePassage
    entail_prob: float = 0.0
    contradict_prob: float = 0.0
    neutral_prob: float = 0.0
    recency_weight: float = 0.0
    numeric_date_ok: bool = False

VerdictType = Literal["Supported", "Refuted", "Not enough evidence", "Contested"]

@dataclass
class FactCheckFeatures:
    """Stores the computed features for scoring."""
    entail_max: float
    entail_mean3: float
    contradict_max: float
    agree_domain_count: int
    releliance_score_avg: float
    recency_weight_max: float


# --- Modified Classes (with custom printing) ---

@dataclass
class Citation:
    """Represents a single citation for the output."""
    url: str
    title: str
    published_at: datetime
    snippet: str

    def __repr__(self) -> str:
        """
        Provides a clean, one-line summary for when this object 
        is printed inside a list.
        """
        return f"<Citation: \"{self.title}\" ({self.url})>"

@dataclass
class FactCheckResult:
    """The final, enforced output structure (Fact Checking Module Output)."""
    claim: str
    verdict: VerdictType
    score: int  # 0-100
    citations: List[Citation]
    features: FactCheckFeatures

    def __repr__(self) -> str:
        """
        A concise, one-line representation for developers 
        (e.g., for logging).
        """
        # Create a short snippet of the claim
        claim_snippet = self.claim[:40] + "..." if len(self.claim) > 40 else self.claim
        return (
            f"<FactCheckResult: {self.verdict} (Score: {self.score}) "
            f"for claim \"{claim_snippet}\">"
        )

    def __str__(self) -> str:
        """
        A user-friendly, multi-line "pretty print" used by 
        the print() function.
        """
        divider = "-" * 20
        
        # Build the citations list. This will use the custom __repr__
        # we defined for the Citation class.
        if self.citations:
            citations_str = "\n  ".join(map(repr, self.citations))
        else:
            citations_str = "  None"
        
        # The default repr for FactCheckFeatures is good (key-value pairs)
        features_str = repr(self.features)

        # Assemble the final multi-line string
        return (
            f"[{self.verdict.upper()} (Score: {self.score}/100)]\n"
            f"Claim: \"{self.claim}\"\n"
            f"{divider}\n"
            f"Citations ({len(self.citations)}):\n{citations_str}\n"
            f"{divider}\n"
            f"Features: {features_str}"
        )


class FactValidatorInterface(ABC):
    """
    Abstract Base Class for the Fact Checking Module.
    Enforces the core validation interface and provides an extensible structure.
    """

    def __init__(self, llm: LLMInterface):
        # Store the LLM dependency
        self.llm = llm
    

    def validate_claim(self, claim: str, claim_type: str, passages: List[SourcePassage]) -> FactCheckResult:
        """
        The **Core Clean Interface** function.
        Returns the appropriate enforced interface type (FactCheckResult).
        """
        pass
