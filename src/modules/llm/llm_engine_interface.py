from abc import ABC, abstractmethod
from typing import List

class LLMInterface(ABC):
    """
    Abstract interface for LLM implementations (OpenAI, Ollama, etc).
    """
    @abstractmethod
    def message(self, message: str) -> str:
        """Send single message, return response text"""
        pass
    
    @abstractmethod
    def raw_messages(self, messages: List) -> str:
        """Send multi-message conversation, return response text"""
        pass

    @abstractmethod
    def build(self) -> 'LLMInterface':
        """Create new instance with current configuration"""
        pass