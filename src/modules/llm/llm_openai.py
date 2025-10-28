from typing import List
from dotenv import load_dotenv
load_dotenv(override=True)
from modules.llm.llm_engine_interface import LLMInterface
from openai import OpenAI
import os

class llm_openai(LLMInterface):
    def __init__(self, role="user", temperature=0, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.role = role
        self.max_tokens = 1000
    
    def raw_messages(self, messages: List) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content
    
    def message(self, message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": self.role, "content": message}],
            temperature=self.temperature
        )
        return response.choices[0].message.content
    
    def build(self) -> LLMInterface:
        return llm_openai(self.role, self.temperature, self.model)