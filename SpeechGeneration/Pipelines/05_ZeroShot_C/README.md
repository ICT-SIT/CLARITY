# 05 — ZeroShot_C (Text Similarity + Accent Confidence Score)

## Description
CLARITY framework using **CosyVoice2** in zero-shot mode with **RAAP** (Retrieval-Augmented Accent Prompting).
The accent pool is filtered using **ECAPA-TDNN accent confidence scores** in addition to text similarity, selecting the prompt that best matches both the target accent and the input text.

This corresponds to the **ZeroShot_C** row in the paper's ablation table (Table I):
`Text Sim ✓ | Accent Score ✓ | Text Adaptation: Standard`

## Pipeline
- **TTS backbone**: CosyVoice2 (`iic/CosyVoice2-0.5B`)
- **Inference mode**: `inference_zero_shot`
- **Text input**: Standard (non-adapted) sentence
- **Accent prompt audio**: Retrieved via RAAP — ranked by `accent_confidence + text_similarity`
- **Retrieval**: `DataRetrieval` using `merged_selected_mos_and_confidence.tsv` which includes pre-computed ECAPA-TDNN confidence scores
- **Text adaptation**: None
- **Accent pool**: `data/accent_pool.tsv` (`merged_selected_mos_and_confidence.tsv`)

## Files
| File | Description |
|------|-------------|
| `pipeline.ipynb` | Full generation pipeline |
| `data/prompts_with_speaker_info.json` | Shared input — see [Pipelines README](../README.md#shared-input-prompts_with_speaker_infojson) |
| `data/accent_pool.tsv` | Accent pool — see [Pipelines README](../README.md#shared-input-accent-pools) |

## How to Run
1. Activate the CosyVoice2 conda environment — see [Pipelines README](../README.md#dependencies)
2. Open `pipeline.ipynb` and run all cells
3. Outputs are written to `4_Zeroshot_C/` with manifest `Zeroshot_C.tsv`

## Notes
- Key difference from ZeroShot_B: uses `merged_selected_mos_and_confidence.tsv` instead of `merged_selected_metadata_wer_mos.tsv`; `DataRetrieval` incorporates the confidence score into ranking
- The ECAPA-TDNN confidence scores were pre-computed using `Jzuluaga/accent-id-commonaccent_ecapa`
