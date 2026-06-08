import json
from pathlib import Path
import ast
from groq import Groq

def plan_annotations(ocr_path: Path, transcript_path: Path, annotation_path: Path, client=None, model_name=None) -> None:
    """
    Constructs a granular visual-audio event map, synchronizing text positions 
    and handwritten math steps with word-level transcript timestamps.
    """
    with ocr_path.open("r", encoding="utf-8") as f:
        ocr_data = json.load(f)
        
    with transcript_path.open("r", encoding="utf-8") as f:
        full_transcript = json.load(f)
        
    segments = full_transcript.get("segments", full_transcript.get("chunks", []))
    simple_transcript = [
        {
            "text": seg.get("text", "").strip(),
            "start": seg.get("start", seg.get("timestamp", [0])[0]),
            "end": seg.get("end", seg.get("timestamp", [0, 1])[1]),
            "words": seg.get("words", [])
        } for seg in segments
    ]
    
    video_end_time = simple_transcript[-1]["end"] if simple_transcript else 70.0

    prompt = f"""
You are an expert audio-visual synchronization architect mapping out a handwritten math video.
Your job is to match visual actions exactly with spoken timestamps.

SYNCHRONIZATION TARGET TIMELINES:
1. "progressive_text_highlight": Highlights the question as it's read. Cover the whole question box tracking left-to-right.
2. "marker_highlight": Draw a crisp yellow marker box around specific parameters (like coordinates or choices) ONLY during their spoken word window.
3. "board_write": Slowly write out lines from left-to-right. Set start_time exactly when the instructor begins explaining that line, and end_time when they finish speaking it.

OCR BLOCK SPACE MANIFEST:
{json.dumps(ocr_data)}

GRANULAR AUDIO TIMELINE:
{json.dumps(simple_transcript)}

Output ONLY a valid JSON object matching this schema layout precisely:
{{
  "annotations": [
    {{
      "type": "progressive_text_highlight",
      "target_coords": [12, 95, 680, 40],
      "start_time": 0.0,
      "end_time": 4.8
    }},
    {{
      "type": "marker_highlight",
      "target_text": "(1, 2)",
      "target_coords": [490, 95, 65, 35],
      "start_time": 3.2,
      "end_time": 4.8
    }},
    {{
      "type": "board_write",
      "content": "d = √((x₂ - x₁)² + (y₂ - y₁)²)",
      "x": 520,
      "y": 100,
      "start_time": 5.2,
      "end_time": 11.5
    }},
    {{
      "type": "board_write",
      "content": "d = √((4 - 1)² + (6 - 2)²)",
      "x": 520,
      "y": 170,
      "start_time": 12.0,
      "end_time": 21.4
    }},
    {{
      "type": "board_write",
      "content": "d = √(3² + 4²)",
      "x": 520,
      "y": 240,
      "start_time": 22.0,
      "end_time": 29.1
    }},
    {{
      "type": "board_write",
      "content": "d = √(9 + 16)",
      "x": 520,
      "y": 310,
      "start_time": 30.0,
      "end_time": 36.8
    }},
    {{
      "type": "board_write",
      "content": "d = √25",
      "x": 520,
      "y": 380,
      "start_time": 37.5,
      "end_time": 42.1
    }},
    {{
      "type": "board_write",
      "content": "d = 5 units",
      "x": 520,
      "y": 450,
      "start_time": 43.0,
      "end_time": 48.5
    }},
    {{
      "type": "marker_highlight",
      "target_text": "(C) 5 units",
      "target_coords": [10, 330, 155, 45],
      "start_time": 49.0,
      "end_time": {video_end_time}
    }}
  ]
}}
"""

    print("Connecting to Groq Engine to synchronize visual audio progression matrices...")
    groq_client = Groq()
    
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        response_format={"type": "json_object"}
    )

    raw_text = chat_completion.choices[0].message.content.strip()
    try:
        structured_plan = json.loads(raw_text)
    except Exception:
        structured_plan = ast.literal_eval(raw_text)

    with annotation_path.open("w", encoding="utf-8") as f:
        json.dump(structured_plan, f, indent=2, ensure_ascii=False)
    print("Success: Time-synchronized annotation blueprint populated successfully.")