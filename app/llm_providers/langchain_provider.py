from langchain_openai import ChatOpenAI
from .base import BaseLLMProvider

class LangChainProvider(BaseLLMProvider):
    def __init__(self, api_key):
        self.client = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)

    def request(self, prompt):
        response = self.client.invoke(prompt)
        return response.content, len(prompt.split())

    def get_token_count(self, response):
        return len(response.split())
