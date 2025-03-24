import pytest
import time
from app.services.llm_client import LLMClient


def test_rate_limit():
    client = LLMClient(rate_limit=2)  # Set a low limit for testing
    agent_id = "test_agent"

    # First two requests should pass
    client.request(agent_id, "Test prompt 1")
    client.request(agent_id, "Test prompt 2")

    # Third request should fail due to rate limit
    with pytest.raises(Exception, match="Rate limit exceeded for agent test_agent"):
        client.request(agent_id, "Test prompt 3")

    # Wait for rate limit window to reset
    time.sleep(60)

    # Request should now pass
    client.request(agent_id, "Test prompt 4")