from abc import ABC, abstractmethod
import ollama
from modules.llm.llm_engine_interface import *

class llm_ollama(LLMInterface):
    def __init__(self):
        self.role = "user"
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

    def message(self, message: str) -> str:
        return self.message_raw(message).message.content
    
    def message_raw(self, message: str) -> ollama.ChatResponse:
        return ollama.chat(model="llama3.1", messages=[{"role":self.role,"content":message}])
    
    def set_role(self, role):
        self.role = role
        return self