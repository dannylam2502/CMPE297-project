from dataclasses import dataclass
from typing import List, Literal
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
class Citation:
    def __init__(self, passage):
        self.passage = passage
@dataclass
class ClaimCheckResult:
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
    releliance_score_avg: float
    recency_weight_max: float
    contest_score: float = 0.0
@dataclass
class FactCheckResult:
    claim: str
    verdict: VerdictType
    score: int
    citations: List[Citation]
    features: FactCheckFeatures
