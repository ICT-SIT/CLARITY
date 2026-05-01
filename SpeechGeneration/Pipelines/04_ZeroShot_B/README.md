# 04 — ZeroShot_B (Text Similarity with User Sentence)

## Description
CLARITY framework using **CosyVoice2** in zero-shot mode with accent pool retrieval.
The retrieval query includes **both the speaker metadata and the user's input sentence**, giving more context to the text similarity scorer compared to ZeroShot_A.

This corresponds to the **ZeroShot_B** row in the paper's ablation table (Table I):
`Text Sim ✓ | Accent Score ✗ | Text Adaptation: Standard`

## Pipeline
- **TTS backbone**: CosyVoice2 (`iic/CosyVoice2-0.5B`)
- **Inference mode**: `inference_zero_shot`
- **Text input**: Standard (non-adapted) sentence
- **Accent prompt audio**: Retrieved from accent pool via text similarity (sentence-aware)
- **Retrieval**: Top-1 by TF-IDF cosine similarity; the user's sentence is included in the query dict passed to `DataRetrieval`
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
3. Outputs are written to `3_Zeroshot_B/` with manifest `Zeroshot_B.tsv`

## Notes
- Key difference from ZeroShot_A: `_pick_accent_ref(speaker_info, sentence)` — the sentence is passed into the retrieval query — see [Retrieval Module](../README.md#retrieval-module)
