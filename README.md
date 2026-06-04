# Automated Annotation System

A fully modular, AI-driven video annotation generator using EasyOCR, Whisper, and Gemini.

## Setup

```bash
pip install -r requirements.txt
```

Whisper also requires `ffmpeg` to be installed and available on PATH.

## Usage

Place your files in:
- `images/background.png`
- `audio/audio.mpeg`

Run the pipeline:
```bash
python main.py
```

The system will prompt you for your Gemini API key and model name, and will output the results to the `output/` directory:
- `ocr.json`
- `transcript.json`
- `annotation.json`
- `sync_report.json`
- `final_video.mp4`

## Modules
- `ocr_module.py`: Handles OCR extraction via EasyOCR. Outputs explicit text bounding boxes.
- `transcription_module.py`: Handles audio transcription via Whisper, extracting word-level timestamps.
- `planner_module.py`: Interfaces with the Gemini API to orchestrate notes. Strict prompt constraints prevent arbitrary coordinate generation.
- `renderer_module.py`: Renders the final video. Features a dedicated Solution Workspace layout engine, precise word-level synchronization, and beautiful Matplotlib-based LaTeX rendering for formulas.
- `validation_module.py`: Performs pre-render checks to ensure no missing timestamps, invalid OCR references, or duplicated formulas.
