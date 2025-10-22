from abc import ABC, abstractmethod

# New Abstract Class for the LLM Dependency
class LLMInterface(ABC):
    """
    Abstract interface for any underlying Language Model (e.g., OpenAI, Gemini, HuggingFace).
    """
    @abstractmethod
    def message(self, message:str) -> tuple[float, float, float]:
        pass