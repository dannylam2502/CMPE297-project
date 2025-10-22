from abc import ABC, abstractmethod
import ollama

from modules.llm.llm_engine_interface import *

class llm_ollama(LLMInterface):
    def __init__(self):
        self.role = "user"

    def message(self, message: str) -> str:
        return self.message_raw(message).message.content
    
    def message_raw(self, message: str) -> ollama.ChatResponse:
        return ollama.chat(model="llama3.1", messages=[{"role":self.role,"content":message}])
    
    def set_role(self, role):
        self.role = role
        return self
        