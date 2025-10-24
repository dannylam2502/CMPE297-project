from dotenv import load_dotenv
load_dotenv(override=True)
from modules.llm.llm_engine_interface import LLMInterface
from openai import OpenAI
import os

class llm_openai(LLMInterface):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"
    
    def message(self, message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}],
            temperature=0.0
        )
        return response.choices[0].message.content