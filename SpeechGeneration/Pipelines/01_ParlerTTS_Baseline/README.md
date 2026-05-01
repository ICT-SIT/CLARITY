# 01 — ParlerTTS Baseline

## Description
Baseline system using **ParlerTTS** (`parler-tts/parler-tts-mini-v1`).
No accent pool or prompt audio is used. The model receives a standard sentence and a free-form natural language instruction describing the target speaker (accent, gender, age), and generates speech directly.

This corresponds to the **ParlerTTS** row in the paper's ablation table (Table I).

## Pipeline
- **TTS backbone**: `parler-tts/parler-tts-mini-v1`
- **Text input**: Standard (non-adapted) sentence
- **Instruction input**: Explicit speaker instruction (e.g. "Generate speech in a British accent, Male, 35 years old")
- **Accent prompt audio**: None
- **Text adaptation**: None

## Files
| File | Description |
|------|-------------|
| `pipeline.ipynb` | Full generation pipeline |
| `data/prompts_with_speaker_info.json` | Shared input — see [Pipelines README](../README.md#shared-input-prompts_with_speaker_infojson) |

## How to Run
1. Install dependencies: `pip install parler-tts transformers soundfile`
2. Open `pipeline.ipynb` and run all cells
3. Outputs are written to `ParlerTTS/` with a manifest TSV

## Notes
- The notebook reads `explicit_instruction` from each adaptation and passes it to ParlerTTS alongside the standard sentence
- No speaker prompt audio is used — ParlerTTS generates accent purely from the instruction text
