from abc import abstractmethod, ABC


# New Abstract Class for the LLM Dependency
class LLMReasoningInterface(ABC):
    """
    Abstract interface for any underlying Language Model (e.g., OpenAI, Gemini, HuggingFace).
    """
    @abstractmethod
    def reasoning_agent(self, message:str) :
        pass
