from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Literal, Any, Union
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
import math

@dataclass
class SourcePassage:
    """Represents a single retrieved passage from the RAG interface."""
    content: str
    relevance_score: float  # The ranking score (relevance, not truth)
    url: str
    domain: str
    title: str
    published_at: datetime

@dataclass
class ClaimCheckResult:
    """Represents the results of the claim checks (NLI, Numeric/Date, etc.)."""
    passage: SourcePassage
    # NLI Classification
    entail_prob: float = 0.0
    contradict_prob: float = 0.0
    neutral_prob: float = 0.0
    # Other checks
    recency_weight: float = 0.0  # Computed time decay
    numeric_date_ok: bool = False # 0/1 for future num_ok feature

@dataclass
class Citation:
    """Represents a single citation for the output."""
    url: str
    title: str
    published_at: datetime
    snippet: str

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

@dataclass
class FactCheckResult:
    """The final, enforced output structure (Fact Checking Module Output)."""
    claim: str
    verdict: VerdictType
    score: int  # 0-100
    citations: List[Citation]
    features: FactCheckFeatures


class FactValidatorInterface(ABC):
    """
    Abstract Base Class for the Fact Checking Module.
    Enforces the core validation interface and provides an extensible structure.
    """

    def __init__(self, llm_interface: LLMInterface):
        # Store the LLM dependency
        self.llm = llm_interface
    
    @abstractmethod
    def retrieve_passages(self, claim: str) -> List[SourcePassage]:
        """
        Retrieval & Preprocessing (Spec #2)
        Stub for the RAG interface call and initial filtering.
        """
        pass

    @abstractmethod
    def run_claim_checks(self, claim: str, passages: List[SourcePassage]) -> List[ClaimCheckResult]:
        """
        Claim Checks (Spec #3)
        Run NLI, Numeric/Date, and compute recency weight.
        """
        pass

    @abstractmethod
    def compute_features(self, checks: List[ClaimCheckResult]) -> FactCheckFeatures:
        """
        Features (Spec #4)
        Compute the six main features on the re-ranked top passages.
        """
        pass

    @abstractmethod
    def compute_score_and_verdict(self, raw_features: FactCheckFeatures) -> tuple[int, VerdictType]:
        """
        Scoring & Verdict (Spec #5)
        Apply raw score formula, sigmoid, and verdict policy.
        """
        pass

    @abstractmethod
    def select_citations(self, checks: List[ClaimCheckResult]) -> List[Citation]:
        """
        Output (Spec #6)
        Select 2-3 citations based on criteria.
        """
        pass

    def validate_claim(self, claim: str) -> FactCheckResult:
        """
        The **Core Clean Interface** function.
        Returns the appropriate enforced interface type (FactCheckResult).
        """
        
        # 2. Retrieval & Preprocessing
        passages = self.retrieve_passages(claim)
        
        # 3. Claim Checks
        checks = self.run_claim_checks(claim, passages)
        
        # Re-rank/Filter to top N for feature computation (e.g., top 8 by relevance/rank)
        # Assuming the run_claim_checks output maintains the desired order/top-N selection
        
        # 4. Features
        features = self.compute_features(checks)
        
        # 5. Scoring & Verdict
        score, verdict = self.compute_score_and_verdict(features)
        
        # 6. Output (Citations)
        citations = self.select_citations(checks)

        return FactCheckResult(
            claim=claim,
            verdict=verdict,
            score=score,
            citations=citations,
            features=features
        )
