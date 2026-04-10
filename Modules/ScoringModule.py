from Modules.OpenAIClient import OpenAIClient

class ScoringModule(OpenAIClient):
    def __init__(self, api_key, model_type="gpt-5"):
        super().__init__(api_key, model_type)
        anon_map = {}
        reverse_map = {}

    def create_prompt(self, input_text):
        speaker_info_structured, original_text, samples = input_text

        # Step 1: Map to anonymised keys
        self.anon_map = {str(i+1): v for i, v in enumerate(samples.values())}
        self.reverse_map = {str(i+1): k for i, k in enumerate(samples.keys())}

        # Step 2: Create numbered list string
        samples_list_str = "\n".join(f"{k}. {v}" for k, v in self.anon_map.items())

        prompt = f"""You are an expert linguist and speech evaluator.
Your task is to score how well each given text sample is appropriate for a speaker from a specific accent or region.

Speaker info:
{speaker_info_structured}

Samples to evaluate:
{samples_list_str}

From the above, please derive the following:
- "score": an integer from 0 to 10, where 0 = completely inappropriate localisation for this speaker and 10 = appropriate and excellent regional adaptation with authentic local vocabulary, expressions, or language patterns.
- "reason": a short explanation of why the score was given, considering whether the language is natural for this speaker and noting any regional vocabulary or expressions.

IMPORTANT: Reward localisation only when it feels natural and contextually appropriate. The presence of a local particle or slang word does not automatically make a sentence better. If the expression feels forced, awkward, or out of place, the score should be reduced accordingly.

Format your output as a JSON object like this:
{{
    "score": <a list of scores in order of the samples>,
    "reason": "<a list of short reasonings in order of the samples>
}}
"""
        return prompt

    def validate_data(self, structured_data):
        if "score" not in structured_data or structured_data["score"] is None:
            raise ValueError("Missing 'score' field in response.")
        if "reason" not in structured_data or structured_data["reason"] is None:
            raise ValueError("Missing 'reason' field in response.")
        return True
    
    def map_scores_back_to_keys(self, parsed):
        # Map scores and reasonings back to original keys
        scores = parsed.get("score", [])
        reasonings = parsed.get("reason", [])
        if len(scores) != len(self.reverse_map) or len(reasonings) != len(self.reverse_map):
            print("Number of scores or reasonings does not match number of samples.")
            return False

        mapped_results = {
            self.reverse_map[str(i+1)]: {"score": scores[i], "reason": reasonings[i]}
            for i in range(len(scores))
        }
        return mapped_results

    def process(self, input_text):
        prompt = self.create_prompt(input_text)
        response = self.get_response(prompt)
        parsed = self.clean_response(response)
        self.validate_data(parsed)
        mapped_results = self.map_scores_back_to_keys(parsed)
        return mapped_results