import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Merge and normalize AERSC2020 and SEAME metadata TSVs.")
    parser.add_argument(
        "--aersc", 
        default="data/selected/AERSC2020/aersc2020_selected_metadata.tsv",
        help="Path to AERSC2020 TSV file"
    )
    parser.add_argument(
        "--seame", 
        default="data/selected/seame/seame_selected_metadata.tsv",
        help="Path to SEAME TSV file"
    )
    parser.add_argument(
        "--output", 
        default="data/selected/merged_selected_metadata.tsv",
        help="Path for merged output TSV"
    )

    args = parser.parse_args()
    aersc_path = args.aersc
    seame_path = args.seame
    output_path = args.output

    df_aersc = process_file(aersc_path, 'AERSC2020')
    df_seame = process_file(seame_path, 'seame')
    merged_df = pd.concat([df_aersc, df_seame], ignore_index=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, sep='\t', index=False)
    print(f'Merged TSV written to {output_path}')


# Standardise naming conventions
def normalize_filepath(filepath, source):
    if source == 'AERSC2020':
        # Remove leading ../../ and ensure starts with data/selected
        fp = filepath.replace('../../../', '')
        if not fp.startswith('data/selected'):
            fp = 'data/selected/' + fp.lstrip('/')
        return fp
    elif source == 'seame':
        # Add data/ if not present
        fp = filepath.replace('selected/seame/', 'data/selected/seame/')
        if not fp.startswith('data/selected'):
            fp = 'data/selected/' + fp.lstrip('/')
        return fp
    return filepath

def normalize_gender(gender):
    if str(gender).lower().startswith('f'):
        return 'F'
    elif str(gender).lower().startswith('m'):
        return 'M'
    return gender

def normalize_language(language):
    if str(language).lower() == 'english':
        return 'EN'
    return language

def process_file(path, source):
    df = pd.read_csv(path, sep='\t')
    if 'filepath' in df.columns:
        df['filepath'] = df['filepath'].apply(lambda x: normalize_filepath(x, source))
    if 'gender' in df.columns:
        df['gender'] = df['gender'].apply(normalize_gender)
    if 'language' in df.columns:
        df['language'] = df['language'].apply(normalize_language)
    return df

# aersc_path = 'data/selected/AERSC2020/aersc2020_selected_metadata.tsv'
# seame_path = 'data/selected/seame/seame_selected_metadata.tsv'

# # Process both files
# df_aersc = process_file(aersc_path, 'AERSC2020')
# df_seame = process_file(seame_path, 'seame')

# # Merge and output
# merged_df = pd.concat([df_aersc, df_seame], ignore_index=True)
# merged_df.to_csv('data/selected/merged_selected_metadata.tsv', sep='\t', index=False)
# print('Merged TSV written to merged_selected_metadata.tsv')

if __name__ == "__main__":
    main()
