# 06 — Adapted_GPT_A (GPT-4o-mini Text Adaptation, Standard Retrieval)

## Description
CLARITY framework using **CosyVoice2** with **GPT-4o-mini text adaptation**.
The input sentence is localized to the target dialect by GPT-4o-mini. The adapted text is then spoken using a zero-shot accent prompt retrieved from the accent pool via text similarity (no confidence score filtering at retrieval stage — the adapted text JSON already contains the pre-selected audio reference).

This corresponds to the **Adapted_GPT_A** row in the paper's ablation table (Table I):
`Text Sim ✓ | Accent Score ✓ | Text Adaptation: GPT (standard selection)`

## Pipeline
- **TTS backbone**: CosyVoice2 (`iic/CosyVoice2-0.5B`)
- **Inference mode**: `inference_zero_shot`
- **Text input**: GPT-4o-mini adapted text (dialect-localized)
- **Accent prompt audio**: Pre-selected from the accent pool (stored in `adapted_text_results.json`)
- **Text adaptation**: GPT-4o-mini (`gpt-4o-mini`)
- **Text selection**: Standard — the single GPT candidate is used directly (not LLM-judge-selected)

## Files
| File | Description |
|------|-------------|
| `pipeline.ipynb` | Full generation pipeline |
| `data/adapted_text_results.json` | Pre-generated output from text adaptation step. Each adaptation contains `adapted_text.gpt-4o-mini` (the localized text) and `audio_path.gpt-4o-mini` (the selected accent pool reference). Original filename: `5gptA_2_prompts_with_results_gpt-4o-mini_standard.json` |

## How to Run
1. Activate the CosyVoice2 conda environment — see [Pipelines README](../README.md#dependencies)
2. Open `pipeline.ipynb` and run all cells
3. Outputs are written to `5_Adapted_GPT_A/` with manifest `5_Adapted_GPT_A.tsv`

## Notes
- Text adaptation was run separately (see `Retrieval/` or `SpeechGeneration/2_adapted_text_generation/`) before this pipeline
- The JSON includes both the adapted text and the audio reference path — no live retrieval is performed here
