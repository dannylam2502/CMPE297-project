from abc import ABC, abstractmethod
from typing import List

# New Abstract Class for the LLM Dependency
class LLMInterface(ABC):
    """
    Abstract interface for any underlying Language Model (e.g., OpenAI, Gemini, HuggingFace).
    """
    @abstractmethod
    def message(self, message:str) -> str:
        pass
    
    @abstractmethod
    def raw_messages(self, message:List) -> any:
        pass

    def build() -> LLMInterface:
        pass

    def build(self) -> LLMInterface:
        pass