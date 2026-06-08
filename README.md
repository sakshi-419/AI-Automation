# AI Educational Video Annotation System

This repository contains a complete end‑to‑end pipeline that transforms a question image and a teacher narration audio into a polished educational video with:

- Word‑level text highlighting synchronized to speech
- Handwritten step‑by‑step solution appearing on a virtual whiteboard
- Hand image overlay that follows the writing cursor
- Persistent annotations (highlights, circles) that never disappear
- Automatic answer detection and circling

## Folder Structure
```
AI-Automation-main/
├─ main.py                 # orchestrates the whole pipeline
├─ ocr_module.py          # EasyOCR wrapper
├─ transcription_module.py# Whisper wrapper with word timestamps
├─ planner_module.py      # Gemini‑based annotation planner (fallback simple heuristic)
├─ renderer_module.py     # Video rendering with OpenCV / MoviePy
├─ validation_module.py   # sanity checks for generated artefacts
├─ requirements.txt        # Python dependencies
├─ README.md               # this file
├─ input/                  # place your background.png and audio.mp3 here
└─ output/                 # generated artefacts and final video
```

## Setup
```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your Gemini API key as an environment variable before running:
```bash
export GEMINI_API_KEY=YOUR_KEY   # Windows: set GEMINI_API_KEY=YOUR_KEY
```

## Run the pipeline
```bash
python main.py
```
The script will:
1. Run OCR on `input/background.png` → `output/ocr.json`
2. Transcribe `input/audio.mp3` with Whisper → `output/transcript.json`
3. Generate annotation plan via Gemini (or simple heuristic) → `output/annotation.json`
4. Render the final video → `output/final_video.mp4`

## Dependencies
- `opencv-python`
- `moviepy`
- `easyocr`
- `whisper`
- `google-generativeai`
- `numpy`
- `pillow`
- `matplotlib`
- `scikit-learn`

## Notes
- The hand image used for the writing animation is bundled automatically and referenced from the internal assets folder.
- The current planner uses a simple heuristic for answer detection; you can replace `planner_module.generate_annotation` with a richer Gemini prompt if desired.
- Adjust `resolution` and `fps` in `renderer_module.render_video` to suit your needs.

---
Happy coding! 🎓🚀
