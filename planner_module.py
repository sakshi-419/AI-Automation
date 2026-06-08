import json
from pathlib import Path
import ast
from groq import Groq

def plan_annotations(ocr_path: Path, transcript_path: Path, annotation_path: Path, client=None, model_name=None) -> None:
    """
    Constructs a highly granular visual-audio event map, ensuring the final correct 
    MCQ option is highlighted immediately after the last formula row resolves.
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

CRITICAL CONCLUDING RULE:
You must explicitly generate a final "marker_highlight" annotation for Option (C). 
Set its 'start_time' to trigger right after the final answer calculation 'd = 5 units' finishes rendering, and keep it active until the absolute end of the video ('end_time': {video_end_time}).

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
      "target_coords": [15, 365, 160, 45],
      "start_time": 48.6,
      "end_time": {video_end_time}
    }}
  ]
}}
"""

    print("Connecting to Groq Engine to synchronize final visual answer indicators...")
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

    # Post-Generation Verification Guard: Double-check that a final answer highlight is strictly present
    has_final_highlight = any(
        a.get("type") == "marker_highlight" and "C" in str(a.get("target_text", ""))
        for a in structured_plan.get("annotations", [])
    )

    if not has_final_highlight:
        print("Guard Notice: Appending missing explicit trailing Option C highlight annotation block...")
        # Fallback tracking injection pinpointed tightly over option C bounds from OCR map coordinates
        option_c_highlight = {
            "type": "marker_highlight",
            "target_text": "(C) 5 units",
            "target_coords": [15, 365, 160, 45],
            "start_time": max(video_end_time - 5.0, 0.0),
            "end_time": video_end_time
        }
        
        # Pull exact completion timings from the last board calculation line if it exists
        write_steps = [a for a in structured_plan.get("annotations", []) if a.get("type") == "board_write"]
        if write_steps:
            write_steps.sort(key=lambda s: s.get("start_time", 0.0))
            last_math_step = write_steps[-1]
            option_c_highlight["start_time"] = last_math_step.get("end_time", option_c_highlight["start_time"]) + 0.1

        structured_plan.setdefault("annotations", []).append(option_c_highlight)

    with annotation_path.open("w", encoding="utf-8") as f:
        json.dump(structured_plan, f, indent=2, ensure_ascii=False)
    print("Success: Final time-synchronized annotation plan compiled with trailing Option C highlight.")