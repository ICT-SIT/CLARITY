import json
from abc import ABC, abstractmethod

class OpenAIClient(ABC):
    """
    Functions
      - __init__(api_key, model_type)
      - get_response(prompt) -> str
      - clean_response(resp) -> dict
      - process(input_text) -> dict
    Uses OpenAI's Chat Completions endpoint with JSON mode to force valid JSON.
    """
    def __init__(self, api_key, model_type="gpt-4o-mini"):
        print(f"Model used = {model_type}")
        self.api_key = api_key
        self.model = model_type
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def get_response(self, prompt: str) -> str:
        import requests

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        # JSON mode to force a JSON object response
        data = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a careful JSON-only generator."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 1,
        }

        resp = requests.post(self.api_url, headers=headers, json=data, timeout=60)
        if resp.status_code == 200:
            j = resp.json()
            return j["choices"][0]["message"]["content"]
        else:
            raise RuntimeError(f"API Error: {resp.status_code} {resp.text}")

    def clean_response(self, resp: str):
        s = resp.strip()
        if "```json" in s:
            s = s.split("```json", 1)[1]
        if "```" in s:
            s = s.split("```", 1)[0]
        s = s.strip()
        try:
            return json.loads(s)
        except Exception as e:
            raise ValueError(f"Response is not valid JSON: {e}\nRaw: {s[:500]}")

    def process(self, input_text):
        prompt = self.create_prompt(input_text)
        response = self.get_response(prompt)
        parsed = self.clean_response(response)
        self.validate_data(parsed)
        return parsed

    @abstractmethod
    def create_prompt(self, input_text):
        pass

    @abstractmethod
    def validate_data(self, structured_data):
        pass
