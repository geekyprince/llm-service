import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_create_agent():
    response = client.post("/agents/", json={"agent_id": "test_agent"})
    assert response.status_code == 200
    assert response.json()["message"] == "Agent test_agent created successfully"

def test_create_duplicate_agent():
    client.post("/agents/", json={"agent_id": "test_agent"})
    response = client.post("/agents/", json={"agent_id": "test_agent"})
    assert response.status_code == 400

def test_list_agents():
    response = client.get("/agents/")
    assert response.status_code == 200
    assert "test_agent" in response.json()["agents"]

def test_ask_question():
    client.post("/agents/", json={"agent_id": "test_agent"})
    response = client.post("/ask/", json={"agent_id": "test_agent", "prompt": "Hello"})
    assert response.status_code == 200
    assert "response" in response.json()

def test_ask_nonexistent_agent():
    response = client.post("/ask/", json={"agent_id": "unknown_agent", "prompt": "Hello"})
    assert response.status_code == 404

def test_get_cost():
    client.post("/agents/", json={"agent_id": "test_agent"})
    response = client.get("/cost/test_agent")
    assert response.status_code == 200
    assert "total_cost" in response.json()

def test_get_prompts():
    client.post("/agents/", json={"agent_id": "test_agent"})
    client.post("/ask/", json={"agent_id": "test_agent", "prompt": "Hello"})
    response = client.get("/prompts/test_agent")
    assert response.status_code == 200
    assert "Hello" in response.json()["prompts"]