# 09 — Adapted_Llama_B (LLaMA-3.1-8B Text Adaptation, Confidence-Filtered Retrieval)

## Description
CLARITY framework using **CosyVoice2** with **LLaMA-3.1-8B text adaptation** and **RAAP** (Retrieval-Augmented Accent Prompting).
Identical in structure to Adapted_GPT_B, but using LLaMA-3.1-8B adapted text.

This corresponds to the **Adapted_Llama_B** row in the paper's ablation table (Table I):
`Text Sim ✓ | Accent Score ✓ | Text Adaptation: LLaMA (adapted)`

## Pipeline
- **TTS backbone**: CosyVoice2 (`iic/CosyVoice2-0.5B`)
- **Inference mode**: `inference_zero_shot`
- **Text input**: LLaMA-3.1-8B adapted text
- **Accent prompt audio**: Retrieved from confidence-filtered accent pool at runtime
- **Retrieval**: RAAP using `data/accent_pool.tsv` (`merged_selected_mos_and_confidence.tsv`)
- **Text adaptation**: LLaMA-3.1-8B

## Files
| File | Description |
|------|-------------|
| `pipeline.ipynb` | Generation pipeline (adapted from `07_Adapted_Llama_A`; see modification notes below) |
| `data/adapted_text_results.json` | Pre-generated LLaMA adaptations. Original filename: `2_prompts_with_results_llama3.1_8b.json` |
| `data/accent_pool.tsv` | Accent pool with ECAPA-TDNN confidence scores (`merged_selected_mos_and_confidence.tsv`) |

## How to Run
> **Note:** The `pipeline.ipynb` here is based on the Adapted_Llama_A notebook. Before running, update the following variables in the data loading cell:
> - `INPUT_JSON = Path("data/adapted_text_results.json")`
> - `ACCENT_POOL_TSV = Path("data/accent_pool.tsv")`

1. Activate the CosyVoice2 conda environment — see [Pipelines README](../README.md#dependencies)
2. Open `pipeline.ipynb`, apply the path changes above, and run all cells
3. Outputs are written to `8_Adapted_Llama_B/`

## Notes
- Key difference from Adapted_Llama_A: accent pool reference is retrieved live using RAAP rather than read from the pre-selected path in the JSON
