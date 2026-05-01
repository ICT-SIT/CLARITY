# CLARITY — Audio Generation Pipelines

This folder contains the **nine audio generation pipelines** used in the CLARITY paper experiments. Each subfolder is self-contained with its pipeline notebook and input data files.

> **Paper:** *CLARITY: Contextual Linguistic Adaptation and Accent Retrieval for Dual-Bias Mitigation in Text-to-Speech Generation*

---

## Pipeline Overview

| # | Folder | System | Text Input | Accent Prompt | Text Adaptation |
|---|--------|--------|-----------|---------------|-----------------|
| 1 | [01_ParlerTTS_Baseline](01_ParlerTTS_Baseline/) | ParlerTTS | Standard | None | None |
| 2 | [02_CosyVoice_Baseline](02_CosyVoice_Baseline/) | CosyVoice2 (`instruct2`) | Standard | Near-silence | None |
| 3 | [03_ZeroShot_A](03_ZeroShot_A/) | CosyVoice2 (`zero-shot`) | Standard | Accent pool (text sim only) | None |
| 4 | [04_ZeroShot_B](04_ZeroShot_B/) | CosyVoice2 (`zero-shot`) | Standard | Accent pool (text sim + sentence) | None |
| 5 | [05_ZeroShot_C](05_ZeroShot_C/) | CosyVoice2 (`zero-shot`) | Standard | RAAP (text sim + confidence) | None |
| 6 | [06_Adapted_GPT_A](06_Adapted_GPT_A/) | CosyVoice2 (`zero-shot`) | GPT-4o-mini adapted | Pre-selected in JSON | GPT-4o-mini |
| 7 | [07_Adapted_Llama_A](07_Adapted_Llama_A/) | CosyVoice2 (`zero-shot`) | LLaMA-3.1-8B adapted | Pre-selected in JSON | LLaMA-3.1-8B |
| 8 | [08_Adapted_GPT_B](08_Adapted_GPT_B/) | CosyVoice2 (`zero-shot`) | GPT-4o-mini adapted | RAAP (text sim + confidence) | GPT-4o-mini |
| 9 | [09_Adapted_Llama_B](09_Adapted_Llama_B/) | CosyVoice2 (`zero-shot`) | LLaMA-3.1-8B adapted | RAAP (text sim + confidence) | LLaMA-3.1-8B |

**Rows 3–9 are CosyVoice2 + CLARITY components.** Each adds a layer:
- 3 → adds accent pool retrieval (text sim)
- 4 → adds sentence to retrieval query
- 5 → adds ECAPA-TDNN accent confidence score to retrieval (full RAAP)
- 6–9 → adds LLM text adaptation on top of 5

---

## Shared Input: `prompts_with_speaker_info.json`
Used by pipelines 01–05. Contains **30 scenarios × 120 adaptations = 3,600 rows**, each with:
- `standard_sentence`: The input text to speak
- `explicit_instruction`: Natural language speaker description (e.g. "Generate speech in a British accent, Male, 35 years old")
- `results.inferred_speaker_info`: Structured metadata (accent, gender, age) used for accent pool retrieval

---

## Shared Input: Accent Pools

| TSV | Used by | Description |
|-----|---------|-------------|
| `accent_pool.tsv` in 03, 04 | ZeroShot_A, B | `merged_selected_metadata_wer_mos.tsv` — pool filtered by WER and MOS |
| `accent_pool.tsv` in 05, 08, 09 | ZeroShot_C, Adapted_B variants | `merged_selected_mos_and_confidence.tsv` — pool additionally filtered by ECAPA-TDNN accent confidence ≥ 0.9 |

The accent pool is drawn from **AESRC** (10 accents: CA, CN, ES, GB, IN, JP, KR, PT, RU, US) and **SEAME** (MY, SG).

---

## Dependencies

### CosyVoice2 (pipelines 02–09)
```bash
# Follow CosyVoice2 setup instructions in the main repo
# https://github.com/FunAudioLLM/CosyVoice
conda activate cosyvoice
```

### ParlerTTS (pipeline 01)
```bash
pip install parler-tts transformers soundfile
```

### Environment variables (`.env`)
```
OPENAI_API_KEY=...   # for GPT-4o-mini text adaptation (pipelines 06, 08)
GEMINI_API_KEY=...   # for LLM-guided instruction parsing
```

### Output Sample Rates
- **ParlerTTS** (pipeline 01): 44,100 Hz
- **CosyVoice2** (pipelines 02–09): 24,000 Hz

---

## Retrieval Module
Pipelines 03–09 use `DataRetrieval` from `Retrieval/Classes/DataRetrieval.py` in the parent repo. Make sure the `../Retrieval` path is accessible when running the notebooks.

---

## Audio Pool Files
The accent pool TSVs reference audio files from the AESRC and SEAME datasets by relative path (e.g. `data/selected/AERSC2020/GBR/...`). Ensure the dataset is present at `../data/` relative to `SpeechGeneration/` before running.
