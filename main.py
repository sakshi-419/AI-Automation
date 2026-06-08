import os
from pathlib import Path
from google.genai import Client

# Import your sub-modules explicitly
import ocr_module
import transcription_module
import planner_module
import renderer_module


def main():
    # 1. Pipeline Environment Configuration Setup
    root_dir = Path("C:/AI-Automation-main")
    print(f"Project Working Directory Root: {root_dir}")

    # Set up input asset paths
    video_source_path = root_dir / "input" / "raw_video.mp4"
    image_frame_path = root_dir / "input" / "background.png"
    # FIXED: Changed from output/extracted_audio.mp3 to match your actual file location on disk
    audio_track_path = root_dir / "input" / "audio.mp3"

    # Set up generated artifact paths
    ocr_json_path = root_dir / "output" / "ocr.json"
    transcript_json_path = root_dir / "output" / "transcript.json"
    annotation_json_path = root_dir / "output" / "annotations.json"
    final_output_path = root_dir / "output" / "final_whiteboard_video.mp4"

    # Verify your API key configuration before running
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "\n❌ CRITICAL ERROR: GEMINI_API_KEY system environment variable not found!"
        )
        print(
            "Please set it in PowerShell using: $env:GEMINI_API_KEY='your_key_here' and try again.\n"
        )
        return

    # Initialize the modern Google GenAI Client
    client = Client(api_key=api_key)

    # Shifted to 1.5-flash to completely bypass the 2.5-flash 20-request daily ceiling
    model_name = "gemini-1.5-flash"

    # --- Step 1: Processing OCR Extraction ---
    print("\n--- Step 1: Processing OCR ---")
    print("-> dynamically executing: ocr_module.extract_ocr()")
    ocr_module.extract_ocr(image_path=image_frame_path, output_path=ocr_json_path)

    # --- Step 2: Processing Audio Transcript Timeline ---
    print("\n--- Step 2: Processing Transcript ---")
    print("-> dynamically executing: transcription_module.transcribe_audio()")
    transcription_module.transcribe_audio(
        audio_path=audio_track_path, output_path=transcript_json_path
    )

    # --- Step 3: Generating Instructor Annotation Plan with Highlights ---
    print("\n--- Step 3: Generating Instructor Annotation Plan with Highlights ---")
    planner_module.plan_annotations(
        ocr_path=ocr_json_path,
        transcript_path=transcript_json_path,
        annotation_path=annotation_json_path,
        client=client,
        model_name=model_name,
    )

    # --- Step 4: Compiling Final Rendered Video Output ---
    print("\n--- Step 4: Compiling Video Layers ---")
    print("-> dynamically executing: renderer_module.render_video()")
    renderer_module.render_video(
        image_path=image_frame_path,
        audio_path=audio_track_path,
        annotation_path=annotation_json_path,
        output_path=final_output_path,
    )


if __name__ == "__main__":
    main()
