# 03 — ZeroShot_A (Text Similarity Only)

## Description
CLARITY framework using **CosyVoice2** in zero-shot mode with accent pool retrieval.
The accent pool candidate is selected using **text similarity only** (TF-IDF cosine similarity between the pool transcript and the standard sentence). No accent confidence score is used.

This corresponds to the **ZeroShot_A** row in the paper's ablation table (Table I):
`Text Sim ✓ | Accent Score ✗ | Text Adaptation: Standard`

## Pipeline
- **TTS backbone**: CosyVoice2 (`iic/CosyVoice2-0.5B`)
- **Inference mode**: `inference_zero_shot`
- **Text input**: Standard (non-adapted) sentence
- **Accent prompt audio**: Retrieved from accent pool via text similarity
- **Retrieval**: Top-1 by TF-IDF cosine similarity (transcript vs. standard sentence), no accent confidence filtering
- **Text adaptation**: None
- **Accent pool**: `data/accent_pool.tsv` (`merged_selected_metadata_wer_mos.tsv`)

## Files
| File | Description |
|------|-------------|
| `pipeline.ipynb` | Full generation pipeline |
| `data/prompts_with_speaker_info.json` | Shared input — see [Pipelines README](../README.md#shared-input-prompts_with_speaker_infojson) |
| `data/accent_pool.tsv` | Accent pool — see [Pipelines README](../README.md#shared-input-accent-pools) |

## How to Run
1. Activate the CosyVoice2 conda environment — see [Pipelines README](../README.md#dependencies)
2. Open `pipeline.ipynb` and run all cells
3. Outputs are written to `2_Zeroshot_A/` with manifest `2_Zeroshot_A.tsv`

## Notes
- Retrieval uses `DataRetrieval.find_relevant(metadata, top_n=1)` with text similarity only — see [Retrieval Module](../README.md#retrieval-module)
