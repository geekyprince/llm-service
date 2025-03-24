import openai
import time
from collections import deque, defaultdict
from langchain_openai import ChatOpenAI
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
        if backend == "openai":
            self.client = openai.OpenAI(api_key=self.api_key)
        elif backend == "langchain":
            self.client = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=self.api_key)

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
        response, tokens_used = (
            self._request_openai(prompt) if self.backend == "openai" else self._request_langchain(prompt))
        cost = (tokens_used / 1000) * self.cost_per_1000_tokens[self.backend]
        self.total_costs[agent_id] += cost
        return response

    def _request_openai(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            tokens_used = response.usage.total_tokens
            return response.choices[0].message.content, tokens_used
        except openai.RateLimitError:
            time.sleep(10)
            return self._request_openai(prompt)

    def _request_langchain(self, prompt):
        response = self.client.invoke(prompt)
        return response.content, len(prompt.split())

    def get_cost(self, agent_id):
        return self.total_costs[agent_id]

    def get_all_prompts(self, agent_id):
        return self.prompts_log[agent_id]


llm_client = LLMClient(backend="langchain", api_key=OPENAI_API_KEY, rate_limit=10)