"""
Audio Extraction

USAGE EXAMPLES (from main folder):
1. Basic usage with language-specific defaults:
   python ./DataPreprocessing/seame/extract_audio.py --language EN

2. Multiple languages with defaults:
   python ./DataPreprocessing/seame/extract_audio.py --language EN CS

3. Custom paths:
   python ./DataPreprocessing/seame/extract_audio.py --language CS --segments custom/segments.tsv --audio-dir custom/audio --output custom/output
"""

import os
import pandas as pd
from typing import List, Dict, Optional
import argparse
from tqdm import tqdm
import librosa
import soundfile as sf


LANGUAGE_THRESHOLDS = {
    'EN': {
        'min_utterances': 1,
        'min_words': 5,
        'ideal_utterances': 100
    },
    'ZH': {
        'min_utterances': 1,
        'min_words': 5,
        'ideal_utterances': 100
    },
    'CS': {
        'min_utterances': 1,
        'min_words': 5,
        'ideal_utterances': 100
    }
}

def main():
    
    parser = argparse.ArgumentParser(
        description="Extract audio files by criteria from SEAME dataset using language-specific thresholds",
        epilog="Examples: python %(prog)s --language EN | python %(prog)s --language EN ZH CS"
    )
    
    # Required arguments
    parser.add_argument("--language", "-l", nargs="+", required=True,
                       choices=["EN", "ZH", "CS"],
                       help="Target language codes: EN=English, ZH=Chinese, CS=Code-Switch (required). Can specify multiple: --language EN ZH")
    
    # Optional arguments with defaults
    parser.add_argument("--segments", "-s", 
                       default="./DataPreprocessing/seame/analysis_results/all/all_segments_summary.tsv",
                       help="Path to segment summary TSV file (default: analysis_results/all/all_segments_summary.tsv)")
    parser.add_argument("--audio-dir", "-a", 
                       default="./data/seame/data",
                       help="Directory containing source audio files (default: ./data/seame/data)")
    parser.add_argument("--output", "-o", 
                       default="./data/selected/seame",
                       help="Output directory for selected audio files (default: ./data/selected/seame)")
    parser.add_argument("--audio-types", nargs="+", 
                       choices=["conversation", "interview"],
                       default=None,
                       help="Audio types to include. If not specified, includes both conversation and interview")


    args = parser.parse_args()
    print_configuration(args)

    results = extract_audio(
        segment_summary_path=args.segments,
        audio_source_dir=args.audio_dir,
        output_dir=args.output,
        target_languages=args.language,
        audio_types=args.audio_types,
    )

    generate_extraction_report(results, output_dir=args.output)
    print_statistics(results)



def extract_audio(
    segment_summary_path: str,
    audio_source_dir: str,
    output_dir: str,
    target_languages: List[str] = ["EN"],
    audio_types: Optional[List[str]] = None,
) -> Dict[str, any]:
    """
    Extract audio files based on specified criteria with language-specific thresholds.
    
    Parameters:
    -----------
    segment_summary_path : str
        Path to the segment summary TSV file
    audio_source_dir : str
        Directory containing the original audio files
    output_dir : str
        Directory where selected audio files will be copied
    target_languages : List[str], default=["EN"]
        Target language codes (EN, ZH, CS)
    audio_types : List[str], optional
        List of audio types to include (e.g., ['conversation', 'interview'])
        If None, includes all types

    Returns dict containing:
        - selected_segments: DataFrame of segments that meet criteria
        - selected_speakers: List of speaker IDs that meet criteria
        - stats: Dictionary with extraction statistics
        - file_mapping: Dictionary mapping original to new file paths
    """
    
    print(f"[INFO] Loading segment summary from: {segment_summary_path}")
    
    # Load segment summary
    try:
        segments_df = pd.read_csv(segment_summary_path, sep='\t')
    except Exception as e:
        raise FileNotFoundError(f"Could not load segment summary: {e}")
    
    print(f"[INFO] Loaded {len(segments_df)} total segments")
    
    # Filter by language
    if target_languages:
        segments_df = segments_df[segments_df['language'].isin(target_languages)]
        print(f"[INFO] After language filter ({', '.join(target_languages)}): {len(segments_df)} segments")
    
    # Filter by audio type if specified
    if audio_types:
        segments_df = segments_df[segments_df['audio_type'].isin(audio_types)]
        print(f"[INFO] After audio type filter {audio_types}: {len(segments_df)} segments")
    
    # Add word count column
    segments_df['word_count'] = segments_df['text'].apply(count_words)
    
    # Apply language-specific filtering
    print(f"[INFO] Applying language-specific thresholds:")
    for lang in target_languages:
        min_words = LANGUAGE_THRESHOLDS[lang]['min_words']
        ideal_utterances = LANGUAGE_THRESHOLDS[lang]['ideal_utterances']
        print(f"  {lang}: ≥{min_words} words per utterance, max {ideal_utterances} utterances per speaker")
    
    # Filter by language-specific word count thresholds
    filtered_segments = []
    for lang in target_languages:
        lang_segments = segments_df[segments_df['language'] == lang].copy()
        min_words = LANGUAGE_THRESHOLDS[lang]['min_words']
        lang_segments = lang_segments[lang_segments['word_count'] >= min_words]
        filtered_segments.append(lang_segments)
        print(f"[INFO] {lang} after min words filter (≥{min_words}): {len(lang_segments)} segments")
    
    segments_df = pd.concat(filtered_segments, ignore_index=True) if filtered_segments else pd.DataFrame()
    print(f"[INFO] Total after word filtering: {len(segments_df)} segments")
    
    # Apply language-specific utterance count filtering and prioritization
    qualifying_speakers = []
    selected_segments_list = []
    
    print(f"[INFO] Applying language-specific utterance thresholds and prioritization:")
    for lang in target_languages:
        min_utterances = LANGUAGE_THRESHOLDS[lang]['min_utterances']
        ideal_utterances = LANGUAGE_THRESHOLDS[lang]['ideal_utterances']
        print(f"  {lang}: ≥{min_utterances} utterances per speaker")
    
    # Group by speaker-language and count utterances
    speaker_lang_counts = segments_df.groupby(['speaker_id', 'language']).size().reset_index(name='utterance_count')
    
    # Filter by language-specific utterance thresholds and select best utterances
    for lang in target_languages:
        min_utterances = LANGUAGE_THRESHOLDS[lang]['min_utterances']
        ideal_utterances = LANGUAGE_THRESHOLDS[lang]['ideal_utterances']
        
        lang_speaker_counts = speaker_lang_counts[speaker_lang_counts['language'] == lang]
        qualifying_lang_speakers = lang_speaker_counts[lang_speaker_counts['utterance_count'] >= min_utterances]['speaker_id'].tolist()
        qualifying_speakers.extend(qualifying_lang_speakers)
        
        # For each qualifying speaker in this language, select the best utterances
        lang_segments = segments_df[segments_df['language'] == lang]
        for speaker_id in qualifying_lang_speakers:
            speaker_segments = lang_segments[lang_segments['speaker_id'] == speaker_id].copy()
            
            # Sort by word count (descending) to prioritize higher word count utterances
            speaker_segments = speaker_segments.sort_values('word_count', ascending=False)
            
            # Cap at ideal_utterances
            selected_speaker_segments = speaker_segments.head(ideal_utterances)
            selected_segments_list.append(selected_speaker_segments)
            
            print(f"[INFO] {lang} speaker {speaker_id}: selected {len(selected_speaker_segments)} utterances (from {len(speaker_segments)} available)")
        
        print(f"[INFO] {lang} speakers meeting min utterance criteria (≥{min_utterances}): {len(qualifying_lang_speakers)}")
    
    # Combine all selected segments
    if selected_segments_list:
        selected_segments = pd.concat(selected_segments_list, ignore_index=True)
    else:
        selected_segments = pd.DataFrame()
    
    # Remove duplicates from qualifying speakers list
    qualifying_speakers = list(set(qualifying_speakers))
    print(f"[INFO] Total unique qualifying speakers: {len(qualifying_speakers)}")
    print(f"[INFO] Final selected segments (after prioritization and capping): {len(selected_segments)}")
    
    # File mapping for copying
    file_mapping = {}
    
    # Extract selected audio segments
    print(f"[INFO] Extracting audio segments to: {output_dir}")
    extracted_count = 0
    successful_segments = []
    failed_segments = []

    # Create progress bar for audio extraction
    with tqdm(total=len(selected_segments), desc="Extracting audio", unit="segment") as pbar:
        for _, segment in selected_segments.iterrows():
            # Use the original_audio_file directly
            original_audio_id = segment['original_audio_id']
            original_audio_file = segment['original_audio_file']
            audio_type = segment['audio_type']
            
            # Construct source path to the full audio file
            source_path = create_audio_path(audio_source_dir, original_audio_file, audio_type)
            
            # Create output filename with speaker, segment info, and timing to ensure uniqueness
            segment_language = segment['language']
            start_ms = int(segment['start_ms'])
            end_ms = int(segment['end_ms'])
            segment_id = f"{segment['speaker_id']}_{segment_language}_{original_audio_file.replace('.flac', '')}_{start_ms}_{end_ms}"
            output_dir_with_country = output_dir + f"/{segment['nationality']}/{segment['nationality']}{segment['speaker_id']}"
            dest_path = os.path.join(output_dir_with_country, f"{segment_id}.wav")

            # Check if file already exists
            if os.path.exists(dest_path):
                file_mapping[source_path] = dest_path
                extracted_count += 1
                successful_segments.append(segment)
                pbar.set_postfix({"Extracted": extracted_count, "Speaker": segment['speaker_id'], "Status": "skipped"})
            elif source_path and os.path.exists(source_path):
                # Extract the specific segment (directory will be created only if extraction succeeds)
                if extract_audio_segment(source_path, dest_path, segment['start_ms'], segment['end_ms']):
                    file_mapping[source_path] = dest_path
                    extracted_count += 1
                    successful_segments.append(segment)
                    pbar.set_postfix({"Extracted": extracted_count, "Speaker": segment['speaker_id'], "Status": "extracted"})
                else:
                    failed_segments.append(segment)
                    pbar.set_postfix({"Extracted": extracted_count, "Speaker": segment['speaker_id'], "Status": "failed"})
            else:
                failed_segments.append(segment)
                pbar.write(f"[WARNING] Audio file not found: {original_audio_file}")
                pbar.set_postfix({"Extracted": extracted_count, "Speaker": segment['speaker_id'], "Status": "missing"})
            # Update progress bar
            pbar.update(1)

    print(f"[INFO] Successfully extracted or found {extracted_count} audio segments")

    # Create custom TSV with specific columns for successful segments only
    if successful_segments:
        custom_tsv_data = []
        for _, segment in pd.DataFrame(successful_segments).iterrows():
            segment_language = segment['language']
            start_ms = int(segment['start_ms'])
            end_ms = int(segment['end_ms'])
            original_audio_file = segment['original_audio_file']
            segment_id = f"{segment['speaker_id']}_{segment_language}_{original_audio_file.replace('.flac', '')}_{start_ms}_{end_ms}"
            filename = f"{segment_id}.wav"
            nationality = segment['nationality']
            speaker_id = segment['speaker_id']
            filepath = f"selected/seame/{nationality}/{nationality}{speaker_id}/{filename}"
            custom_row = {
                'filename': filename,
                'filepath': filepath,
                'speaker': nationality + speaker_id,
                'duration': segment['duration_sec'],
                'accent': nationality,
                'language': segment['language'],
                'age': segment['age'],
                'gender': segment['gender'],
                'transcript': segment['text']
            }
            custom_tsv_data.append(custom_row)
        output_df = pd.DataFrame(custom_tsv_data)
    else:
        output_df = pd.DataFrame(columns=['filename', 'filepath', 'duration', 'accent', 'language', 'age', 'gender', 'transcript'])

    # Save custom segment summary with specific columns (only successful)
    summary_output_path = os.path.join(output_dir, "seame_selected_metadata.tsv")
    output_df.to_csv(summary_output_path, sep='\t', index=False)
    print(f"[INFO] Saved segment summary to: {summary_output_path}")

    # Save failed/omitted segments to a separate TSV
    if failed_segments:
        failed_df = pd.DataFrame(failed_segments)
        failed_output_path = os.path.join(output_dir, "_failed_or_omitted_segments.tsv")
        failed_df.to_csv(failed_output_path, sep='\t', index=False)
        print(f"[INFO] Saved failed/omitted segment summary to: {failed_output_path}")

    # Generate statistics (once for both saving and return)
    statistics = generate_extraction_statistics(
        segment_summary_path,
        segments_df,
        selected_segments,
        qualifying_speakers,
        target_languages,
        audio_types
    )
    
    # Save speaker statistics
    speaker_stats_df = pd.DataFrame(statistics['speaker_statistics'])
    speaker_stats_path = os.path.join(output_dir, "speaker_statistics.tsv")
    speaker_stats_df.to_csv(speaker_stats_path, sep='\t', index=False)
    print(f"[INFO] Saved speaker statistics to: {speaker_stats_path}")
    
    return {
        'selected_segments': selected_segments,
        'selected_speakers': qualifying_speakers,
        'speaker_statistics': statistics['speaker_statistics'],
        'stats': statistics['stats'],
        'file_mapping': file_mapping
    }


def count_words(text: str) -> int:
    if pd.isna(text) or not isinstance(text, str):
        return 0
    
    # split by whitespace and filter empty strings
    words = [word.strip() for word in text.split() if word.strip()]
    return len(words)


def extract_audio_segment(input_audio_path, output_audio_path, start_ms, end_ms):
    """Extract audio segment using librosa"""

    start_time = float(start_ms) / 1000.0
    end_time = float(end_ms) / 1000.0
    sampling_rate = 16000
    
    try:
        # Load audio file
        y, sr = librosa.load(input_audio_path)
        
        # Extract the specific time segment
        audio = y[int(start_time * sr): int(end_time * sr)]
        
        # Resample if necessary
        if sr != sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
        
        # Create output directory when file is ready
        output_dir = os.path.dirname(output_audio_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Write the output file
        sf.write(output_audio_path, audio, sampling_rate)
        return True
        
    except Exception as e:
        print(f"[WARNING] Error extracting audio segment: {e}")
        return False


def create_audio_path(audio_source_dir: str, filename: str, audio_type: str) -> Optional[str]:
    # Create file path based on seame dataset file structure
    path = os.path.join(audio_source_dir, audio_type, "audio", filename)
    if os.path.exists(path):
        return path
    
    return None


# ┌────────────────────────────────────────────────────────────────────────────┐
# │      PRINTING AND STATISTICS GENERATION (Not main functionality            │
# └────────────────────────────────────────────────────────────────────────────┘

def print_configuration(args):
    print("=" * 60)
    print("AUDIO EXTRACTION CONFIGURATION")
    print("=" * 60)
    print(f"Segment summary file: {args.segments}")
    print(f"Audio source directory: {args.audio_dir}")
    print(f"Output directory: {args.output}")
    print(f"Target languages: {', '.join(args.language)}")
    print(f"Audio types: {args.audio_types if args.audio_types else 'All (conversation + interview)'}")
    
    print(f"\nLanguage-specific thresholds:")
    for lang in args.language:
        min_utterances = LANGUAGE_THRESHOLDS[lang]['min_utterances']
        min_words = LANGUAGE_THRESHOLDS[lang]['min_words']
        ideal_utterances = LANGUAGE_THRESHOLDS[lang]['ideal_utterances']
        print(f"  {lang}: ≥{min_utterances} utterances (min), ≥{min_words} words, max {ideal_utterances} utterances (ideal)")
    
    print("=" * 60)


def generate_extraction_statistics(
    segment_summary_path: str,
    segments_df: pd.DataFrame,
    selected_segments: pd.DataFrame,
    qualifying_speakers: List[str],
    target_languages: List[str],
    audio_types: Optional[List[str]]
) -> Dict[str, any]:
    
    # Prepare language-specific criteria for display
    criteria_info = {}
    for lang in target_languages:
        min_utterances = LANGUAGE_THRESHOLDS[lang]['min_utterances']
        min_words = LANGUAGE_THRESHOLDS[lang]['min_words']
        ideal_utterances = LANGUAGE_THRESHOLDS[lang]['ideal_utterances']
        criteria_info[lang] = {
            'min_utterances': min_utterances,
            'min_words': min_words,
            'ideal_utterances': ideal_utterances
        }
    
    # Prepare statistics
    stats = {
        'total_original_segments': len(pd.read_csv(segment_summary_path, sep='\t')),
        'segments_after_language_filter': len(segments_df),
        'segments_after_word_filter': len(segments_df),  # Already filtered in main function
        'qualifying_speakers': len(qualifying_speakers),
        'final_selected_segments': len(selected_segments),
        'total_duration_hours': selected_segments['duration_sec'].sum() / 3600,
        'criteria': {
            'language_specific_thresholds': criteria_info,
            'target_languages': target_languages,
            'audio_types': audio_types
        }
    }
    
    # Generate speaker statistics by speaker-language combination
    speaker_stats = []
    for speaker_id in qualifying_speakers:
        speaker_segments = selected_segments[selected_segments['speaker_id'] == speaker_id]
        
        # Get unique languages for this speaker
        speaker_languages = speaker_segments['language'].unique()
        
        for language in speaker_languages:
            # Filter segments for this speaker-language combination
            speaker_lang_segments = speaker_segments[speaker_segments['language'] == language]
            
            speaker_info = {
                'speaker_id': speaker_id,
                'language': language,
                'utterance_count': len(speaker_lang_segments),
                'total_duration_sec': speaker_lang_segments['duration_sec'].sum(),
                'avg_words_per_utterance': speaker_lang_segments['word_count'].mean(),
                'gender': speaker_lang_segments['gender'].iloc[0] if len(speaker_lang_segments) > 0 else None,
                'nationality': speaker_lang_segments['nationality'].iloc[0] if len(speaker_lang_segments) > 0 else None,
                'age': speaker_lang_segments['age'].iloc[0] if len(speaker_lang_segments) > 0 else None,
            }
            speaker_stats.append(speaker_info)
    
    return {
        'stats': stats,
        'speaker_statistics': speaker_stats
    }

def print_statistics(results):
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    stats = results['stats']
    print(f"Selected segments: {stats['final_selected_segments']:,}")
    print(f"Qualifying speakers: {stats['qualifying_speakers']:,}")
    print(f"Total duration: {stats['total_duration_hours']:.2f} hours")
    print(f"Avg segments per speaker: {stats['final_selected_segments'] / stats['qualifying_speakers']:.1f}")
    
    # Show top speakers
    speaker_stats = results['speaker_statistics']
    if speaker_stats:
        print(f"\nTop 5 speaker-language combinations by utterance count:")
        sorted_speakers = sorted(speaker_stats, key=lambda x: x['utterance_count'], reverse=True)[:5]
        for i, speaker in enumerate(sorted_speakers, 1):
            print(f"  {i}. {speaker['speaker_id']} ({speaker['language']}): {speaker['utterance_count']} utterances, "
                  f"{speaker['total_duration_sec']:.1f}s, {speaker['avg_words_per_utterance']:.1f} avg words")
            

def generate_extraction_report(results: Dict, output_dir: str) -> None:
    report_path = os.path.join(output_dir, "extraction_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Audio Extraction Report\n")
        f.write("=" * 50 + "\n\n")
        
        stats = results['stats']
        criteria = stats['criteria']
        
        f.write("Extraction Criteria:\n")
        f.write(f"  - Target Languages: {', '.join(criteria['target_languages'])}\n")
        f.write(f"  - Audio Types: {criteria['audio_types']}\n")
        f.write(f"  - Language-specific thresholds:\n")
        for lang, thresholds in criteria['language_specific_thresholds'].items():
            f.write(f"    {lang}: ≥{thresholds['min_utterances']} utterances (min), ≥{thresholds['min_words']} words, max {thresholds['ideal_utterances']} utterances (ideal)\n")
        f.write("\n")
        
        f.write("Extraction Statistics:\n")
        f.write(f"  - Original Total Segments: {stats['total_original_segments']:,}\n")
        f.write(f"  - After Language Filter: {stats['segments_after_language_filter']:,}\n")
        f.write(f"  - After Word Count Filter: {stats['segments_after_word_filter']:,}\n")
        f.write(f"  - Qualifying Speakers: {stats['qualifying_speakers']:,}\n")
        f.write(f"  - Final Selected Segments: {stats['final_selected_segments']:,}\n")
        f.write(f"  - Total Duration: {stats['total_duration_hours']:.2f} hours\n\n")
        
        f.write("Speaker Breakdown (by Speaker-Language):\n")
        speaker_stats = results['speaker_statistics']
        for speaker in speaker_stats[:20]:  # Show first 20 speaker-language combinations
            f.write(f"  - {speaker['speaker_id']} ({speaker['language']}): {speaker['utterance_count']} utterances, "
                   f"{speaker['total_duration_sec']:.1f}s, "
                   f"{speaker['avg_words_per_utterance']:.1f} avg words\n")
        
        if len(speaker_stats) > 20:
            f.write(f"  ... and {len(speaker_stats) - 20} more speaker-language combinations\n")
    
    print(f"[INFO] Extraction report saved to: {report_path}")



if __name__ == "__main__":
    main()
