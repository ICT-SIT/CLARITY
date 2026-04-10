from Modules.OpenAIClient import OpenAIClient

class TextAdaptationModule(OpenAIClient):
    def __init__(self, api_key, model_type="gpt-4o-mini"):
        super().__init__(api_key, model_type)

    def create_prompt(self, input_text):
        structured_instructions, user_text = input_text
        prompt = f"""
You are a text localiser, and your goal is to produce localised, accented speech meant for a TTS LLM to use.

Speaker Information:
{structured_instructions}

User text to be spoken:
{user_text}

Instructions:
- DO NOT translate the text to any other language. Always keep it in the same language (e.g., English).
- Strictly use only ASCII characters in your adapted text.
- You may alter the sentence to fit how the local speaker would use the given language.
- You may add local expressions or discourse particles ONLY if it flows naturally in the local language.
- Ensure that expressions added are suitable with the tone and context of the sentence.

Return a JSON object with only the localized "text" field:
{{ "text": "<localized text here>." }}
"""
        return prompt

    def validate_data(self, structured_data):
        if "text" not in structured_data or structured_data["text"] is None:
            raise ValueError("Missing 'text' field in localized response.")
        return True
