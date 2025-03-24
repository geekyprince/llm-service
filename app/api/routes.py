from fastapi import APIRouter, HTTPException
from app.models.request_models import AgentCreateRequest, PromptRequest
from app.services.agent import Agent
from app.services.llm_client import llm_client

router = APIRouter()
agents = {}

@router.post("/agents/")
def create_agent(request: AgentCreateRequest):
    if request.agent_id in agents:
        raise HTTPException(status_code=400, detail="Agent already exists")
    agents[request.agent_id] = Agent(name=request.agent_id, llm_client=llm_client)
    return {"message": f"Agent {request.agent_id} created successfully"}

@router.post("/ask/")
def ask_question(request: PromptRequest):
    if request.agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    try:
        response = agents[request.agent_id].ask(request.prompt)
        return {"agent_id": request.agent_id, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cost/{agent_id}")
def get_cost(agent_id: str):
    return {"agent_id": agent_id, "total_cost": llm_client.get_cost(agent_id)}

@router.get("/prompts/{agent_id}")
def get_all_prompts(agent_id: str):
    return {"agent_id": agent_id, "prompts": llm_client.get_all_prompts(agent_id)}

@router.get("/agents/")
def list_agents():
    return {"agents": list(agents.keys())}