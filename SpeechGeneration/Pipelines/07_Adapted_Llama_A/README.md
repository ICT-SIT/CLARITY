# 07 — Adapted_Llama_A (LLaMA-3.1-8B Text Adaptation, Standard Retrieval)

## Description
CLARITY framework using **CosyVoice2** with **LLaMA-3.1-8B text adaptation**.
Identical in structure to Adapted_GPT_A, but the text localization is done by LLaMA-3.1-8B instead of GPT-4o-mini.

This corresponds to the **Adapted_Llama_A** row in the paper's ablation table (Table I):
`Text Sim ✓ | Accent Score ✓ | Text Adaptation: LLaMA (standard selection)`

## Pipeline
- **TTS backbone**: CosyVoice2 (`iic/CosyVoice2-0.5B`)
- **Inference mode**: `inference_zero_shot`
- **Text input**: LLaMA-3.1-8B adapted text (dialect-localized)
- **Accent prompt audio**: Pre-selected from the accent pool (stored in `adapted_text_results.json`)
- **Text adaptation**: LLaMA-3.1-8B (`llama3.1:8b`)
- **Text selection**: Standard — the single LLaMA candidate is used directly

## Files
| File | Description |
|------|-------------|
| `pipeline.ipynb` | Full generation pipeline |
| `data/adapted_text_results.json` | Pre-generated output from text adaptation step. Each adaptation contains `adapted_text.llama3.1:8b` and `audio_path.llama3.1:8b`. Original filename: `6llamaA_2_prompts_with_results_llama3.1_8b_standard.json` |

## How to Run
1. Activate the CosyVoice2 conda environment — see [Pipelines README](../README.md#dependencies)
2. Open `pipeline.ipynb` and run all cells
3. Outputs are written to `6_Adapted_Llama_A/` with manifest `6_Adapted_Llama_A.tsv`

## Notes
- Text adaptation was run separately before this pipeline
- The JSON includes both the adapted text and the audio reference path — no live retrieval is performed here
