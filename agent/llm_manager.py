import os
from langchain_google_genai import ChatGoogleGenerativeAI

class LLMManager:
    def __init__(self, model_name: str = "gemini-2.0-flash-lite"):
        self.model_name = model_name
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key
        )
    
    def get_llm(self):
        return self.llm
