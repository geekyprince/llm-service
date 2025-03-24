from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    """Abstract Base Class for LLM Providers"""

    @abstractmethod
    def request(self, prompt):
        """Send a request to the LLM"""
        pass

    @abstractmethod
    def get_token_count(self, response):
        """Return the number of tokens used"""
        pass
