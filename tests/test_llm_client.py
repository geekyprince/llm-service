import pytest
from unittest.mock import MagicMock, patch
from app.services.llm_client import LLMClient

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI Client"""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Mocked OpenAI response"))],
        usage=MagicMock(total_tokens=10)
    )
    return mock_client

@pytest.fixture
def mock_langchain_client():
    """Mock LangChain Client"""
    mock_client = MagicMock()
    mock_client.invoke.return_value = MagicMock(content="Mocked LangChain response")
    return mock_client

def test_llm_client_initialization():
    """Test initialization of LLMClient"""
    client = LLMClient(backend="openai", api_key="dummy_key")
    assert client.backend == "openai"
    assert client.api_key == "dummy_key"

def test_openai_request(mock_openai_client):
    """Test OpenAI request with mocked API"""
    with patch("app.llm_providers.openai_provider.openai.OpenAI", return_value=mock_openai_client):
        client = LLMClient(backend="openai", api_key="dummy_key")
        response = client.request(agent_id="test_agent", prompt="Hello")
        assert response == "Mocked OpenAI response"

def test_langchain_request(mock_langchain_client):
    """Test LangChain request with mocked API"""
    with patch("app.llm_providers.langchain_provider.ChatOpenAI", return_value=mock_langchain_client):
        client = LLMClient(backend="langchain", api_key="dummy_key")
        response = client.request(agent_id="test_agent", prompt="Hello")
        assert response == "Mocked LangChain response"

def test_rate_limit():
    """Test rate limiting per agent"""
    client = LLMClient(backend="openai", api_key="dummy_key", rate_limit=2)  # Limit to 2 requests
    client.request(agent_id="test_agent", prompt="Test 1")
    client.request(agent_id="test_agent", prompt="Test 2")

    with pytest.raises(Exception, match="Rate limit exceeded for agent test_agent"):
        client.request(agent_id="test_agent", prompt="Test 3")
