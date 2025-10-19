# Data Preprocessing
This folder contains the preprocessing steps for Seame and AERSC2020 dataset.

## /seame
This folder contains utilities to:
- parse SEAME transcripts and create a per-segment summary
- extract audio snippets for selected segments

### Steps
1. Download the original SEAME data and save it in `../data/seame` (relative to this folder).

2. Create a segment summary (parses transcripts and writes a summary file). This script walks transcript files and produces a per-segment summary (speaker id, start/end times, text, language tag). The summary is written to `all_segments_summary.tsv`.
```
python ./DataPreprocessing/seame/create_segment_summary.py --data data/seame/data
```

3. Extract audio segments. This script reads the segment summary and extracts corresponding audio snippets from the original recordings, with selection filters by language, duration, etc. The summary of all audio files extracted is written to `seame_selected_metadata.tsv`.

```
python ./DataPreprocessing/seame/extract_audio.py --language EN CS
```
## /AERSC2020
TODO

## Merging both data
Run the following command to use the default input/output paths:
```
python ./DataPreprocessing/merge_datasets.py
```
You can optionally specify your own file paths:
```
python ./DataPreprocessing/merge_datasets.py \
  --aersc path/to/aersc_metadata.tsv \
  --seame path/to/seame_metadata.tsv \
  --output path/to/merged_metadata.tsv
```

## Adding WERS, MOS, and Accent Confidence
### Steps
1. Perform WERS & MOS evaluation to obtain the corresponding scores.  
2. Perform accent recognition to obtain confidence scores. Filter the data to retain only entries with accurate predictions and confidence > 0.9.  
3. Store WER & MOS and accent recognition outputs in ``upstream_data``.
4. Combine the existing data with new metadata (WERS, MOS, Confidence) into a unified TSV file:
```
# Run with all defaults
python ./DataPreprocessing/merge_upstream_data.py
```
To specify custom input and output paths:
```
python ./DataPreprocessing/merge_upstream_data.py \
  --base path/to/base_metadata.tsv \
  --wm path/to/dataset1_wer_mos.csv \
       path/to/dataset2_wer_mos.csv \
  --conf path/to/high_confidence_results.csv \
  --output path/to/merged_output.tsv
```