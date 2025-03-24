from pydantic import BaseModel

class AgentCreateRequest(BaseModel):
    agent_id: str

class PromptRequest(BaseModel):
    agent_id: str
    prompt: str