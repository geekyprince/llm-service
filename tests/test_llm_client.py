import pytest
from app.services.llm_client import LLMClient

def test_llm_client_initialization():
    client = LLMClient(backend="openai", api_key="dummy_key")
    assert client.backend == "openai"