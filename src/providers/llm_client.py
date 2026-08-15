import openai

client = openai.OpenAI()

class LLMClient:
    def __init__(self, base_url: str, api_key:str, temperature: float):
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
