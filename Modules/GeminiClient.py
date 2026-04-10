import json
from abc import ABC, abstractmethod

class GeminiClient(ABC):
    def __init__(self, api_key, model_type):
        self.api_key = api_key
        self.api_url = f'https://generativelanguage.googleapis.com/v1beta/models/{model_type}:generateContent'

    def get_response(self, prompt):
        import requests
        headers = {'Content-Type': 'application/json'}
        params = {'key': self.api_key}
        data = {'contents': [{'parts': [{'text': prompt}]}]}
        response = requests.post(self.api_url, headers=headers, params=params, json=data)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            raise RuntimeError(f'API Error: {response.text}')

    def clean_response(self, resp):
        resp = resp.strip()
        # Extract everything only after ```json
        if "```json" in resp:
            resp = resp.split("```json", 1)[1]
        # Remove trailing ```
        if "```" in resp:
            resp = resp.split("```", 1)[0]
        resp = resp.strip()
        print(resp)
        try:
            return json.loads(resp)
        except Exception:
            raise ValueError("Response is not valid JSON")

    def process(self, input_text):
        prompt = self.create_prompt(input_text)
        response = self.get_response(prompt)
        parsed_response = self.clean_response(response)
        self.validate_data(parsed_response)
        return parsed_response

    @abstractmethod
    def create_prompt(self, input_text):
        pass

    @abstractmethod
    def validate_data(self, clean_text):
        pass