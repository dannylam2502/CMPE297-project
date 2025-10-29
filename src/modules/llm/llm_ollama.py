from abc import ABC, abstractmethod
from typing import List
import ollama
from modules.llm.llm_engine_interface import *

class llm_ollama(LLMInterface):
    def __init__(self, role="user", model="llama3.1"):
        self.role = role
        self.model = model
        self._ensure_ollama_running()

    def _ensure_ollama_running(self):
        """Start Ollama if not running"""
        import subprocess
        try:
            ollama.list()
            return  # Already running
        except:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import time
            time.sleep(3)

    def raw_messages(self, messages: List) -> str:
        return ollama.chat(model=self.model, messages=messages)
    
    def message(self, message: str) -> str:
        return ollama.chat(model=self.model, messages=[{"role":self.role,"content":message}])
    
    def set_role(self, role):
        self.role = role
        return self

    def build():
        return llm_ollama()
    
    def build(self):
        return llm_ollama(self.role, self.model)