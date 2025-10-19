from abc import ABC, abstractmethod

# New Abstract Class for the LLM Dependency
class LLMInterface(ABC):
    """
    Abstract interface for any underlying Language Model (e.g., OpenAI, Gemini, HuggingFace).
    Enforces the method needed by the Fact Checker: NLI classification.
    """
    @abstractmethod
    def classify_nli(self, claim: str, passage: str) -> tuple[float, float, float]:
        """
        Classifies (claim, passage) pair as (entail, contradict, neutral) probabilities.
        Returns: (entail_prob, contradict_prob, neutral_prob)
        """
        pass