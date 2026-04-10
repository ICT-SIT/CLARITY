from Modules.GeminiClient import GeminiClient

class InstructionAnalysisModule(GeminiClient):
    def __init__(self, api_key, model_type="gemini-2.5-flash-lite"):
        super().__init__(api_key, model_type)

    def create_prompt(self, input_text):
        user_instructions, user_text = input_text
        prompt = (
            f"""
You are given a natural language input from a user describing a desired speech generation request. Your job is to extract structured metadata suitable for retrieving a matching voice sample from a database.

User Instructions:
{user_instructions}

User Text:
{user_text}

The available accents in the dataset pool are: CAN, CHN, ESP, GBR, IND, JPN, KOR, PRT, RUS, USA, SG, MY.

Please extract the following fields from the input. If not explicitly stated, make a reasonable inference based on the speech to be spoken. Otherwise, you may mark it as \"unspecified\".

Format your output as a JSON object like this:
```json
{{
  "accent": "<Speaker accent. Choose from the above pool. If there is no exact match, choose the closest available option.>",
  "language": "<language in the phrase to be spoken, e.g. EN>",
  "age": "exact_age | [range_start, range_end]" — use exact age if given, e.g. 25; if approximate age only, infer a 10-year , range, e.g. [20, 30].
  "gender": "M | F",
  "tone": "<e.g. soft, angry, romantic, etc.'>",
  "emotion": "<optional emotion, e.g. love, sadness, happiness'>",
  "additional_context": "<any inferred intent or style, e.g. persuasive, affectionate, instructional>"
}}
```
"""
        )
        return prompt

    def validate_data(self, structured_data):
        required_fields = ['accent', 'language', 'age', 'gender']
        missing = [field for field in required_fields if field not in structured_data or not structured_data[field]]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")
        return True
