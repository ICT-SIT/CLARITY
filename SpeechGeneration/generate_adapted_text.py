import os
import sys
import json
import argparse
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Retrieval')))

from Modules.DataRetrieval import DataRetrieval
from Modules.TextAdaptationModule import TextAdaptationModule
from Modules.InstructionAnalysisModule import InstructionAnalysisModule


def main():
    parser = argparse.ArgumentParser(
        description="Run instruction analysis and text adaptation pipeline."
    )
    parser.add_argument(
        "--metadata",
        default="./DataPreprocessing/merged_selected_wmc.tsv",
        help="Path to metadata TSV file (default: ./DataPreprocessing/merged_selected_wmc.tsv)"
    )
    parser.add_argument(
        "--input_json",
        default="./SpeechGeneration/upstream_data/text-instruction-prompt-final.json",
        help="Path to input JSON file containing scenarios and adaptations."
    )
    parser.add_argument(
        "--output_json",
        default="./SpeechGeneration/prompts_with_results.json",
        help="Path to save output JSON with results."
    )
    parser.add_argument(
        "--adaptation_model",
        default="gpt-4o-mini",
        help="Model used for text adaptation (default: gpt-4o-mini)."
    )
    parser.add_argument(
        "--no_use_adapted",
        dest="use_adapted",
        action="store_false",
        help="If set, disables using adapted text for retrieval (default: use adapted text)."
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in .env")

    # Initialise modules and load data
    instruction_module = InstructionAnalysisModule(api_key=GEMINI_API_KEY)
    text_module = TextAdaptationModule(api_key=OPENAI_API_KEY)
    retriever = DataRetrieval(metadata_path=args.metadata)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} scenarios.")
    print(f"Adaptations per scenario: {len(data[0].get('adaptations', []))}")

    # Process each scenario
    for i, scenario in enumerate(data):
        print(f"\n====== Scenario {i}: {scenario['scenario']}")
        user_text = scenario['standard_sentence']
        print(user_text)

        for j, adaptation in enumerate(scenario['adaptations']):
            adaptation = process_adaptation(adaptation)
            results = adaptation.get('results', {})

            if results:
                continue  # Skip if already processed

            print(f"=== Adaptation {j}")
            user_instructions = adaptation.get("explicit_instruction", "")
            module1_input = (user_instructions, user_text)
            info = instruction_module.process(module1_input)

            # --- Text adaptation ---
            module2_input = (info, user_text)
            localised = text_module.process(module2_input)
            adapted_text = localised.get("text", "")

            # --- Audio retrieval ---
            retrieval_text = adapted_text if args.use_adapted else user_text
            top_row = retriever.find_relevant({**info, "text": retrieval_text}).head(1)
            file_path = top_row.iloc[0]['filepath']
            transcript = top_row.iloc[0]['transcript']

            # --- Save results into structure ---
            adaptation["results"] = {
                "inferred_speaker_info": info,
                "adapted_text": {args.adaptation_model: adapted_text},
                "audio_path": {args.adaptation_model: [file_path, transcript]}
            }

            # Save incrementally
            with open(args.output_json, "w", encoding="utf-8") as out_f:
                json.dump(data, out_f, indent=2, ensure_ascii=False)

    print(f"\nProcessing complete. Results saved to: {args.output_json}")


# Process adaptation for correct format
def process_adaptation(adaptation):
    adaptation.pop("implicit_instruction", None)
    if "results" in adaptation:
        adaptation["results"].pop("implicit", None)

    # Rename field for clarity
    adaptation["initial_gt_adapted_text"] = adaptation.pop("adapted_text", None)

    # Extract metadata into prefixed fields
    metadata = adaptation.pop("metadata", {})
    adaptation["gt_accent"] = metadata.get("accent")
    adaptation["gt_gender"] = metadata.get("gender")
    adaptation["gt_age"] = metadata.get("age")
    adaptation["gt_language"] = metadata.get("language")

    return adaptation


if __name__ == "__main__":
    main()
