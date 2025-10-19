import pandas as pd
import argparse
from pathlib import Path

def merge_wer_mos(df_base, df_source):
    """Merge WER and MOS columns from df_source into df_base."""
    return (
        df_base.merge(
            df_source[["filepath", "wer", "mos"]],
            on="filepath",
            how="left",
            suffixes=("", "_new")
        )
        .assign(
            wer=lambda d: d["wer"].fillna(d["wer_new"]),
            mos=lambda d: d["mos"].fillna(d["mos_new"])
        )
        .drop(columns=["wer_new", "mos_new"])
    )

def main():
    parser = argparse.ArgumentParser(
        description="Merge one or more WER/MOS metadata files into the merged metadata TSV."
    )
    parser.add_argument(
        "--base",
        default="data/selected/merged_selected_metadata.tsv",
        help="Path to base merged metadata TSV file."
    )
    parser.add_argument(
        "--wm",
        nargs="*",
        default=[
            "./DataPreprocessing/upstream_data/AERSC_selected_metadata_wer_mos.csv",
            "./DataPreprocessing/upstream_data/seame_selected_metadata_cs_wer_mos.csv",
            "./DataPreprocessing/upstream_data/seame_selected_metadata_en_wer_mos.csv",
        ],
        help="List of input CSV/TSV files containing filepath, wer, mos columns."
    )
    parser.add_argument(
        "--conf",
        default="./DataPreprocessing/upstream_data/high_confidence_correct.csv",
        help="Input path for filtered Accent Recognition Confidence"
    )
    parser.add_argument(
        "--output",
        default="./DataPreprocessing/merged_selected_wmc.tsv",
        help="Output path for merged TSV."
    )

    args = parser.parse_args()
    base_path = Path(args.base)
    input_files = [Path(f) for f in args.wm]
    conf_file = Path(args.conf)
    output_path = Path(args.output)

    print(f"Loading base metadata: {base_path}")
    df_merged = pd.read_csv(base_path, sep="\t")

    # Ensure columns exist
    for col in ["wer", "mos"]:
        if col not in df_merged.columns:
            df_merged[col] = pd.NA

    # Merge each source
    for src_file in input_files:
        if not src_file.exists():
            print(f"Skipping missing file: {src_file}")
            continue

        # Detect separator automatically (supports CSV or TSV)
        with open(src_file, "r", encoding="utf-8") as f:
            first_line = f.readline()
            sep = "\t" if "\t" in first_line else ","
        df_src = pd.read_csv(src_file, sep=sep)

        required_cols = {"filepath", "wer", "mos"}
        if not required_cols.issubset(df_src.columns):
            print(f"{src_file} missing required columns {required_cols}, skipping.")
            continue

        df_merged = merge_wer_mos(df_merged, df_src)

    # Report missing values
    missing_rows = df_merged[df_merged["wer"].isna() | df_merged["mos"].isna()]
    print(f"\nNumber of rows without MOS/WER: {missing_rows.shape[0]}")
    if not missing_rows.empty:
        print("\nRows missing MOS or WER:\n")
        print(missing_rows[["filepath", "wer", "mos"]].to_string(index=False))

    # Filter out missing
    df_filtered = df_merged.dropna(subset=["wer", "mos"])
    
    # Detect separator automatically (supports CSV or TSV)
    with open(conf_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
        sep = "\t" if "\t" in first_line else ","

    # Add and filter by accent recognition confidence
    conf_df = pd.read_csv(conf_file, sep=sep)

    prefix_to_remove = "C:\\Users\\Haoyu Song\\Desktop\\"
    conf_df['path'] = conf_df['path'].apply(lambda x: x.replace(prefix_to_remove, ""))
    conf_df['path'] = conf_df['path'].apply(lambda x: x.replace("\\", "/"))
    conf_df['filepath'] = conf_df['path']

    df_filtered = df_filtered.copy()
    df_filtered['filepath'] = df_filtered['filepath'].apply(lambda x: x.replace("data/", ""))
    merged_df = df_filtered.merge(conf_df[['filepath', 'confidence']], on='filepath', how='left')
    merged_df = merged_df.dropna(subset=['confidence'])
    merged_df['filepath'] = "data/" + merged_df['filepath']

    merged_df.to_csv(output_path, sep="\t")

    print(f"\nFiltered file saved as: {output_path}")
    print(f"Number of rows after filtering: {df_filtered.shape[0]}")

if __name__ == "__main__":
    main()