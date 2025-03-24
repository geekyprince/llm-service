class Agent:
    def __init__(self, name, llm_client):
        self.name = name
        self.llm_client = llm_client

    def ask(self, prompt):
        return self.llm_client.request(self.name, prompt)