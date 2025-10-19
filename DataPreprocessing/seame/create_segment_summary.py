# HOW TO USE (from main dir):

# Create segment summary from original SEAME dataset
# python ./data_preprocessing/seame/create_segment_summary.py --data data/seame/data

# To add custom output dir
# python ./data_preprocessing/seame/create_segment_summary.py --data data/seame/data --output custom_output_dir

"""
Processes all transcript files and creates a comprehensive TSV with every segment
using the language classification provided in the transcript files (ZH=Chinese, CS=CodeSwitch, EN=English).
"""

import argparse
import os
import csv
from collections import defaultdict
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Create comprehensive segment summary from SEAME corpus")
    
    parser.add_argument("--data", "-d", type=str, required=True,
                       help="Path to original SEAME data directory")
    parser.add_argument("--output", "-o", type=str, default="./analysis_results/all",
                       help="Output directory for summary files (default: segment_summary relative to script location)")
    parser.add_argument("--speaker_info", type=str, default="./data/seame/docs/speaker-info.xls",
                       help="Path to speaker info Excel file (default: ./data/seame/docs/speaker-info.xls)")
    
    args = parser.parse_args()
    print(args)
    # Make output path relative to script location, not current working directory
    if not os.path.isabs(args.output):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(script_dir, args.output)
    
    print(f"[INFO] Processing age data")
    age_mapping = load_unified_speaker_ages(args.speaker_info)

    print(f"[INFO] Processing SEAME corpus: {args.data}")
    
    # Process all segments
    all_segments = process_all_segments(args.data)
    
    if not all_segments:
        print("[ERROR] No segments found")
        return
    
    print(f"[INFO] Processed {len(all_segments)} total segments")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate summary files
    create_segment_summary_tsv(all_segments, age_mapping, args.output)
    create_language_statistics_report(all_segments, args.output)
    
    print(f"\n[INFO] Segment summary completed. Results saved to: {os.path.abspath(args.output)}")


def process_all_segments(data_dir):
    """Process all transcript files and extract every segment with language classification"""
    all_segments = []
    audio_types = ["conversation", "interview"]
    
    for audio_type in audio_types:
        print(f"[INFO] Processing {audio_type} data...")
        
        transcript_dir = os.path.join(data_dir, audio_type, "transcript", "phaseII")
        
        if not os.path.exists(transcript_dir):
            print(f"[WARNING] Transcript directory not found: {transcript_dir}")
            continue
        
        # Process all transcript files
        transcript_files = [f for f in os.listdir(transcript_dir) if f.endswith('.txt')]
        print(f"[INFO] Found {len(transcript_files)} transcript files in {audio_type}")
        
        for transcript_file in transcript_files:
            transcript_path = os.path.join(transcript_dir, transcript_file)
            segments = process_transcript_file(transcript_path, audio_type)
            all_segments.extend(segments)
            
        print(f"[INFO] Processed {len([s for s in all_segments if s['audio_type'] == audio_type])} segments from {audio_type}")
    
    return all_segments


def process_transcript_file(transcript_path, audio_type):
    """Process a single transcript file and extract all segments with language classification"""
    segments = []
    
    with open(transcript_path, "r", encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse transcript line (phaseII format)
                parts = line.split("\t")
                if len(parts) != 5:
                    print(f"[WARNING] Unexpected format in {os.path.basename(transcript_path)} line {line_num}: {len(parts)} parts")
                    continue
                
                idx, start_ms, end_ms, lang, text = parts
                
                # Calculate duration
                try:
                    duration_sec = (float(end_ms) - float(start_ms)) / 1000.0
                except ValueError:
                    print(f"[WARNING] Invalid timestamps in {os.path.basename(transcript_path)} line {line_num}")
                    continue
                
                # Use language code from transcript (ZH=Chinese, CS=CodeSwitch, EN=English)
                language_type = lang.strip()
                

                audio_file = idx + ".flac"

                # Parse information from audio id
                audio_id_info = parse_audio_segment_id(idx)
                
                # Create segment record
                segment = {
                    'speaker_id': audio_id_info.get('speaker_id'),
                    'audio_type': audio_type,
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'duration_sec': duration_sec,
                    'original_audio_file': audio_file,
                    'original_transcript_file': os.path.basename(transcript_path),
                    'line_number': line_num,
                    'original_audio_id': idx,
                    'text': text.strip(),
                    'language': language_type,
                    'gender': audio_id_info.get('gender'),
                    'nationality': audio_id_info.get('nationality')
                }
                
                segments.append(segment)
                        
            except Exception as e:
                print(f"[WARNING] Error processing line {line_num} in {os.path.basename(transcript_path)}: {e}")
                continue
    
    return segments

def parse_audio_segment_id(audio_segment_id):
    """
    Parse SEAME audio segment ID into its components.
    Returns a dictionary with:
        - group_id (for conversation style, 2 digits)
        - location (N=NTU, U=USM)
        - style (C=conversational, I=interview)
        - speaker_id (2 digits)
        - gender (F/M)
        - nationality (A=Malaysian, B=Singaporean)
        - channel (P/Q/X/Y/Z)
        - session (2 digits)
        - part (2 digits)
    Handles both naming conventions (e.g., 08NC16FBQ_0101 and UI06MAZ_0105).
    """
    base_id = audio_segment_id.split('.')[0]
    if '_' in base_id:
        main, rest = base_id.split('_', 1)
    else:
        main, rest = base_id, None

    # If first char is digit, it's the conversation style (08NC16FBQ_0101)
    if main and main[0].isdigit():
        # 08NC16FBQ_0101
        group_id = main[0:2]
        location = main[2]
        style = main[3]
        speaker_id = main[4:6]
        gender = main[6]
        nationality = main[7]
        channel = main[8]
        session = rest[0:2] if rest else None
        part = rest[2:4] if rest else None
    else:
        # UI06MAZ_0105
        location = main[0]
        style = main[1]
        speaker_id = main[2:4]
        gender = main[4]
        nationality = main[5]
        channel = main[6]
        session = rest[0:2] if rest else None
        part = rest[2:4] if rest else None
        group_id = None  # Not present in this format

    return {
        "group_id": group_id,
        "location": location,
        "style": style,
        "speaker_id": style + location + speaker_id,  # Include style to ensure uniqueness
        "gender": gender,
        "nationality": "MY" if nationality=='A' else "SG" if nationality=='B' else nationality,
        "channel": channel,
        "session": session,
        "part": part
    }


def create_segment_summary_tsv(all_segments, age_mapping, output_dir):
    """Create the main TSV file with all segments"""
    
    tsv_file = os.path.join(output_dir, "all_segments_summary.tsv")
    
    # Define column headers - moved file info and text to end
    headers = [
        'audio_type', 'speaker_id', 'start_ms', 'end_ms', 'duration_sec',
        'language', 'gender', 'nationality', 'age', 'original_audio_id',
        'original_audio_file', 'original_transcript_file', 'line_number', 'text'
    ]
    
    print(f"[INFO] Writing {len(all_segments)} segments to TSV file...")
    
    with open(tsv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Write header
        writer.writerow(headers)
        
        # Write all segments - reordered to match new header
        for segment in all_segments:
            row = [
                segment['audio_type'],
                segment['speaker_id'],
                segment['start_ms'],
                segment['end_ms'],
                f"{segment['duration_sec']:.3f}",
                segment['language'],
                segment['gender'],
                segment['nationality'],
                age_mapping[segment['speaker_id']],
                segment['original_audio_id'],
                segment['original_audio_file'],
                segment['original_transcript_file'],
                segment['line_number'],
                segment['text']
            ]
            writer.writerow(row)
    
    print(f"[INFO] Segment summary TSV saved to: {os.path.abspath(tsv_file)}")


def load_unified_speaker_ages(excel_path):
    tab_mapping = {
        'Interview-Malaysia': 'IM',
        'Interview-Singapore': 'IS', 
        'Conversation': 'C'
    }
    
    if not os.path.exists(excel_path):
        print(f"[WARNING] Speaker info file not found: {excel_path}")
        return {}
    
    speaker_ages = {}
    
    try:
        print(f"[INFO] Loading speaker age information from: {excel_path}")
        
        for tab_name, tab_label in tab_mapping.items():
            try:
                df = pd.read_excel(excel_path, sheet_name=tab_name)
                
                # Find speaker ID and age columns
                speaker_col = None
                age_col = None
                
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'speaker' in col_lower and 'id' in col_lower:
                        speaker_col = col
                    elif 'age' in col_lower:
                        age_col = col
                
                if speaker_col is None or age_col is None:
                    print(f"[WARNING] Could not find speaker/age columns in {tab_name}")
                    continue
                
                # Process each row
                count = 0
                for _, row in df.iterrows():
                    style = "C" if tab_label == 'C' else "I"  # Conversation = C, Interview = I
                    location = 'U' if tab_label == 'IM' else 'N'  # Only Interview-Malaysia in USM. 'Conversations' style are recorded in NTU.
                    raw_speaker_num = str(row[speaker_col]).strip()
                    if raw_speaker_num.isdigit() and len(raw_speaker_num) == 1:
                        raw_speaker_num = raw_speaker_num.zfill(2)
                    speaker_id = style + location + raw_speaker_num

                    age_value = row[age_col]
                    
                    # Handle "Not Given" and convert to None
                    if pd.isna(age_value) or str(age_value).strip() == "Not Given":
                        age = None
                    else:
                        age = age_value
                    

                    speaker_ages[speaker_id] = age
                    count += 1
                
                print(f"[INFO] Loaded {count} speakers from {tab_name}")
                
            except Exception as e:
                print(f"[WARNING] Error processing {tab_name}: {e}")
                continue
        
        print(f"[INFO] Total speakers loaded: {len(speaker_ages)}")
        return speaker_ages
        
    except Exception as e:
        print(f"[ERROR] Failed to load speaker information: {e}")
        return {}
    


def create_language_statistics_report(all_segments, output_dir):
    """Create a detailed statistics report in txt file"""
    
    report_file = os.path.join(output_dir, "language_statistics_report.txt")
    
    # Calculate statistics
    total_segments = len(all_segments)
    
    # Count by language type
    language_counts = defaultdict(int)
    for segment in all_segments:
        language_counts[segment['language']] += 1
    
    # Count by speaker
    speaker_counts = defaultdict(int)
    speaker_language = defaultdict(lambda: defaultdict(int))
    for segment in all_segments:
        speaker_counts[segment['speaker_id']] += 1
        speaker_language[segment['speaker_id']][segment['language']] += 1
    
    # Count by audio type
    audio_type_counts = defaultdict(int)
    audio_type_language = defaultdict(lambda: defaultdict(int))
    for segment in all_segments:
        audio_type_counts[segment['audio_type']] += 1
        audio_type_language[segment['audio_type']][segment['language']] += 1
    
    # Duration statistics
    total_duration = sum(segment['duration_sec'] for segment in all_segments)
    duration_by_language = defaultdict(float)
    for segment in all_segments:
        duration_by_language[segment['language']] += segment['duration_sec']
    
    # Write report
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("SEAME Corpus Complete Segment Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Overall Statistics:\n")
        f.write(f"  Total segments: {total_segments:,}\n")
        f.write(f"  Total duration: {total_duration/3600:.2f} hours\n\n")
        
        f.write("Language Distribution:\n")
        for lang_type, count in sorted(language_counts.items()):
            percentage = (count / total_segments) * 100
            duration_hours = duration_by_language[lang_type] / 3600
            f.write(f"  {lang_type}: {count:,} segments ({percentage:.1f}%) - {duration_hours:.2f} hours\n")
        
        f.write("\nAudio Type Breakdown:\n")
        for audio_type, count in sorted(audio_type_counts.items()):
            percentage = (count / total_segments) * 100
            f.write(f"  {audio_type}: {count:,} segments ({percentage:.1f}%)\n")
            
            # Language breakdown within audio type
            for lang_type, lang_count in sorted(audio_type_language[audio_type].items()):
                lang_percentage = (lang_count / count) * 100
                f.write(f"    {lang_type}: {lang_count:,} ({lang_percentage:.1f}%)\n")
            f.write("\n")
        
        f.write("Duration Statistics by Language:\n")
        for lang_type, duration in sorted(duration_by_language.items()):
            hours = duration / 3600
            percentage = (duration / total_duration) * 100
            avg_duration = duration / language_counts[lang_type] if language_counts[lang_type] > 0 else 0
            f.write(f"  {lang_type}: {hours:.2f} hours ({percentage:.1f}%) - avg: {avg_duration:.2f}s per segment\n")
        
        f.write(f"\nSpeaker Statistics:\n")
        f.write(f"  Total unique speakers: {len(speaker_counts)}\n")
        f.write(f"  Segments per speaker (top 10):\n")
        for speaker_id, count in sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / total_segments) * 100
            f.write(f"    {speaker_id}: {count:,} segments ({percentage:.1f}%)\n")
    
    print(f"[INFO] Statistics report saved to: {os.path.abspath(report_file)}")
    
    # Also print summary to console
    print(f"\n" + "=" * 50)
    print("SEGMENT SUMMARY")
    print("=" * 50)
    print(f"Total segments processed: {total_segments:,}")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"\nLanguage breakdown:")
    for lang_type, count in sorted(language_counts.items()):
        percentage = (count / total_segments) * 100
        print(f"  {lang_type}: {count:,} ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
