import json
from pathlib import Path

def transcribe_audio(audio_path: Path, output_path: Path) -> dict:
    print("Processing Transcript...")
    try:
        import whisper
    except ImportError:
        raise RuntimeError("openai-whisper is not installed.")

    # Load base model for fast local transcription
    model = whisper.load_model("base")
    
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        verbose=False,
        fp16=False,
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    return result
