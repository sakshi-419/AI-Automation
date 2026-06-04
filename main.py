import sys
import os
from pathlib import Path

from ocr_module import extract_ocr
from transcription_module import transcribe_audio
from planner_module import generate_annotation_plan
from renderer_module import render_video
from validation_module import validate_plan


def main():
    print("Automated Annotation System")
    print("===========================")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Enter Gemini API Key: ").strip()

    if not api_key:
        print("API Key is required.")
        sys.exit(1)

    model_name = os.getenv("GEMINI_MODEL")
    if not model_name:
        model_name = input("Enter Gemini Model: ").strip()

    if not model_name:
        print("Model Name is required.")
        sys.exit(1)

    root = Path.cwd()

    image_path = root / "input" / "background.png"
    audio_path = root / "input" / "audio.mp3"

    if not image_path.exists():
        print(f"Error: Could not find {image_path}")
        sys.exit(1)

    if not audio_path.exists():
        print(f"Error: Could not find {audio_path}")
        sys.exit(1)

    output_dir = root / "output"
    output_dir.mkdir(exist_ok=True)

    ocr_path = output_dir / "ocr.json"
    transcript_path = output_dir / "transcript.json"
    annotation_path = output_dir / "annotation.json"
    video_path = output_dir / "final_video.mp4"

    try:
        print("Starting OCR...")
        extract_ocr(image_path, ocr_path)

        print("Starting Transcription...")
        transcribe_audio(audio_path, transcript_path)

        print("Starting Annotation Planning...")
        generate_annotation_plan(
            ocr_path,
            transcript_path,
            annotation_path,
            api_key,
            model_name
        )

        print("Starting Validation...")
        validate_plan(
            image_path,
            annotation_path,
            ocr_path,
            transcript_path
        )

        print("Starting Rendering...")
        render_video(
            image_path,
            audio_path,
            annotation_path,
            video_path
        )

        print("Export Complete.")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()