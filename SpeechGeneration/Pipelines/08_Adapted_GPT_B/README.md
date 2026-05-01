# 08 — Adapted_GPT_B (GPT-4o-mini Text Adaptation, Confidence-Filtered Retrieval)

## Description
CLARITY framework using **CosyVoice2** with **GPT-4o-mini text adaptation** and **RAAP** (Retrieval-Augmented Accent Prompting).
Similar to Adapted_GPT_A, but the accent pool reference is retrieved live using the confidence-filtered pool (`merged_selected_mos_and_confidence.tsv`) rather than using a pre-selected reference stored in the JSON.

This corresponds to the **Adapted_GPT_B** row in the paper's ablation table (Table I):
`Text Sim ✓ | Accent Score ✓ | Text Adaptation: GPT (adapted)`

## Pipeline
- **TTS backbone**: CosyVoice2 (`iic/CosyVoice2-0.5B`)
- **Inference mode**: `inference_zero_shot`
- **Text input**: GPT-4o-mini adapted text
- **Accent prompt audio**: Retrieved from confidence-filtered accent pool at runtime
- **Retrieval**: RAAP using `data/accent_pool.tsv` (`merged_selected_mos_and_confidence.tsv`)
- **Text adaptation**: GPT-4o-mini

## Files
| File | Description |
|------|-------------|
| `pipeline.ipynb` | Generation pipeline (adapted from `06_Adapted_GPT_A`; see modification notes below) |
| `data/adapted_text_results.json` | Pre-generated GPT adaptations. Original filename: `2_prompts_with_results_gpt-4o-mini.json` |
| `data/accent_pool.tsv` | Accent pool with ECAPA-TDNN confidence scores (`merged_selected_mos_and_confidence.tsv`) |

## How to Run
> **Note:** The `pipeline.ipynb` here is based on the Adapted_GPT_A notebook. Before running, update the following variables in the data loading cell:
> - `INPUT_JSON = Path("data/adapted_text_results.json")`
> - `ACCENT_POOL_TSV = Path("data/accent_pool.tsv")`

1. Activate the CosyVoice2 conda environment — see [Pipelines README](../README.md#dependencies)
2. Open `pipeline.ipynb`, apply the path changes above, and run all cells
3. Outputs are written to `7_Adapted_GPT_B/`

## Notes
- Key difference from Adapted_GPT_A: accent pool reference is retrieved live using RAAP rather than read from the pre-selected path in the JSON
