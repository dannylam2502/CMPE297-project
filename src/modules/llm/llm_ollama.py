from abc import ABC, abstractmethod
from typing import List
import ollama
from modules.llm.llm_engine_interface import LLMInterface

class llm_ollama(LLMInterface):
    # def __init__(self, role="user", model="llama3.1", temperature=0.3):
    def __init__(self, role="user", model="llama3.2:3b", temperature=0.3):
        self.role = role
        self.model = model
        self.temperature = temperature
        self._ensure_ollama_running()

    def _ensure_ollama_running(self):
        """Start Ollama if not running"""
        import subprocess
        try:
            ollama.list()
            return
        except:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import time
            time.sleep(3)

    def raw_messages(self, messages: List) -> str:
        response = ollama.chat(model=self.model, messages=messages)
        return response.message.content
    
    def message(self, message: str) -> str:
        response = ollama.chat(model=self.model, messages=[{"role": self.role, "content": message}])
        return response.message.content
    
    def set_role(self, role):
        self.role = role
        return self

    def build(self) -> LLMInterface:
        return llm_ollama(self.role, self.model, self.temperature)