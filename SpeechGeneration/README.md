# Speech Generation

This folder contains the speech generation pipeline, which runs after the data preprocessing step has completed. Given a set of scenarios and speaker adaptations, the pipeline:

1. **Analyses** the explicit instruction using the `InstructionAnalysisModule` (Gemini) to infer speaker attributes (accent, gender, age, tone, etc.)
2. **Adapts** the standard sentence to the target speaker profile using the `TextAdaptationModule` (OpenAI)
3. **Retrieves** the most relevant audio clip from the dataset using the `DataRetrieval` module

### Key Files

| File | Description |
|------|-------------|
| `generate_adapted_text.py` | Main pipeline script |
| `upstream_data/text-instruction-prompt-final.json` | Input JSON with scenarios and instructions |
| `prompts_with_results.json` | Output JSON with inferred speaker info, adapted text, and retrieved audio path per adaptation |
| `../DataPreprocessing/merged_selected_wmc.tsv` | TSV metadata of available audio clips (speaker, accent, gender, age, transcript, filepath) |

## Setup

Ensure the following environment variables are set in a `.env` file at the project root:

```
GEMINI_API_KEY=YOUR_GEMINI_KEY
OPENAI_API_KEY=YOUR_OPENAI_KEY
```

## Usage

Run the pipeline with default settings:

```bash
python ./SpeechGeneration/generate_adapted_text.py
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--metadata` | `./DataPreprocessing/merged_selected_wmc.tsv` | Path to speaker metadata TSV |
| `--input_json` | `./SpeechGeneration/upstream_data/text-instruction-prompt-final.json` | Input JSON with scenarios |
| `--output_json` | `./SpeechGeneration/prompts_with_results.json` | Output path for results |
| `--adaptation_model` | `gpt-4o-mini` | OpenAI model used for text adaptation |
| `--no_use_adapted` | *(flag)* | If set, retrieves audio using the **original** sentence instead of the adapted text |


## Notes

- The pipeline skips adaptations that already have a `results` field, making it safe to resume after interruptions.
- Audio retrieval filters candidates by gender, accent, and age (in priority order) before running TF-IDF similarity search on the transcript.
- If `--no_use_adapted` is passed, retrieval uses the original `standard_sentence` rather than the model-adapted text.