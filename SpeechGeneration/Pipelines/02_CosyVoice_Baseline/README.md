# 02 — CosyVoice2 Baseline

## Description
Baseline system using **CosyVoice2** (`iic/CosyVoice2-0.5B`) in instruction-guided mode (`inference_instruct2`).
A near-silent tensor is provided as the prompt audio (no real accent reference), so the model relies entirely on the natural language instruction to determine accent and speaker characteristics.

This corresponds to the **CosyVoice2 Baseline** row in the paper's ablation table (Table I).

## Pipeline
- **TTS backbone**: CosyVoice2 (`iic/CosyVoice2-0.5B`)
- **Inference mode**: `inference_instruct2`
- **Text input**: Standard (non-adapted) sentence
- **Instruction input**: Explicit speaker instruction
- **Accent prompt audio**: Near-silence tensor (no real accent reference)
- **Text adaptation**: None
- **Accent pool retrieval**: None

## Files
| File | Description |
|------|-------------|
| `pipeline.ipynb` | Full generation pipeline |
| `data/prompts_with_speaker_info.json` | Shared input — see [Pipelines README](../README.md#shared-input-prompts_with_speaker_infojson) |

## How to Run
1. Activate the CosyVoice2 conda environment — see [Pipelines README](../README.md#dependencies)
2. Set up `.env` if needed — see [Pipelines README](../README.md#environment-variables-env)
3. Open `pipeline.ipynb` and run all cells
4. Outputs are written to `CosyVoice_Baseline/` with a manifest TSV

## Notes
- The near-silence prompt means the model cannot clone any specific speaker's accent acoustics
- All accent information comes from the instruction string alone
