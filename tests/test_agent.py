from app.services.agent import Agent
from app.services.llm_client import LLMClient

def test_agent_ask():
    llm_client = LLMClient(backend="openai", api_key="dummy_key")
    agent = Agent(name="test_agent", llm_client=llm_client)
    assert agent.name == "test_agent"