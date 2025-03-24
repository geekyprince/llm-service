from app.llm_providers.openai_provider import OpenAIProvider
from app.llm_providers.langchain_provider import LangChainProvider

class LLMFactory:
    @staticmethod
    def create_provider(provider_name, api_key):
        if provider_name == "openai":
            return OpenAIProvider(api_key)
        elif provider_name == "langchain":
            return LangChainProvider(api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
