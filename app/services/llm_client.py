import time
from collections import defaultdict, deque
from app.services.llm_factory import LLMFactory
from app.config import OPENAI_API_KEY

class LLMClient:
    cost_per_1000_tokens = {"openai": 0.002, "langchain": 0.002}

    def __init__(self, backend="openai", api_key=None, rate_limit=10):
        self.backend = backend
        self.api_key = api_key or OPENAI_API_KEY
        self.rate_limit = rate_limit
        self.requests_log = defaultdict(deque)
        self.prompts_log = defaultdict(list)
        self.total_costs = defaultdict(float)

        self.provider = LLMFactory.create_provider(backend, api_key)

    def enforce_rate_limit(self, agent_id):
        current_time = time.time()
        while self.requests_log[agent_id] and self.requests_log[agent_id][0] < current_time - 60:
            self.requests_log[agent_id].popleft()
        if len(self.requests_log[agent_id]) >= self.rate_limit:
            raise Exception(f"Rate limit exceeded for agent {agent_id}")
        self.requests_log[agent_id].append(current_time)

    def request(self, agent_id, prompt):
        self.enforce_rate_limit(agent_id)
        self.prompts_log[agent_id].append(prompt)
        response, tokens_used = self.provider.request(prompt)
        cost = (tokens_used / 1000) * self.cost_per_1000_tokens[self.backend]
        self.total_costs[agent_id] += cost
        return response

    def get_cost(self, agent_id):
        return self.total_costs[agent_id]

    def get_all_prompts(self, agent_id):
        return self.prompts_log[agent_id]
