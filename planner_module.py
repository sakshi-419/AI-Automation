import json
from pathlib import Path
from google import genai
from google.genai import types

def generate_annotation_plan(ocr_path: Path, transcript_path: Path, output_path: Path, api_key: str, model_name: str) -> dict:
    print("Generating Annotation Plan...")
    client = genai.Client(api_key=api_key)
    
    with ocr_path.open("r", encoding="utf-8") as f:
        ocr_data = json.load(f)
        
    with transcript_path.open("r", encoding="utf-8") as f:
        transcript_data = json.load(f)
        
    # Simplify transcript to save token count and focus on segments
    simple_transcript = []
    for seg in transcript_data.get("segments", []):
        simple_transcript.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"]
        })

    prompt = f"""
You are an expert teacher creating a board-writing animation.

Generate annotations that mimic a human solving the problem.
I am providing you with the OCR output of a question image, and the audio transcript of a teacher solving it.

Your goal is to generate an annotation plan that syncs with the teacher's narration.

Use these annotation types:
- highlight
- circle
- underline
- arrow
- answer_box
- formula_box
- text

Rules:
1. NEVER INVENT COORDINATES. Do not output "position" tags. The rendering engine will automatically place text, answer_box, and formula_box into a dedicated "Solution Workspace" on the right side of the screen.
2. For highlighting, circling, or underlining existing text, use the "ocr_reference" field pointing to exactly matching OCR text (e.g. "ocr_reference": "(C) 5 units"). Do not create OCR references for substrings that are not separate OCR entries.
3. Formula boxes must reuse the same screen region. Do not stack formulas vertically. Each new formula replaces the previous formula in time.
4. Never create more than 3 simultaneous annotations.
5. Keep all annotations inside image boundaries (The video resolution is approx 1000x575).
6. Create arrows pointing from formulas to substituted values.
7. Circle important numbers. Underline intermediate results.
8. Final answer must use answer_box.
9. Each annotation should represent one visual action.
10. Minimize floating text cards. Prefer highlighting actual question elements over creating new text.
11. Keep text annotations concise (e.g. "Substitute" instead of "Now substitute values").
12. Output purely valid JSON in the requested format, nothing else. No markdown wrappers.

FORMAT EXAMPLE:
{{
  "annotations": [
    {{
      "type": "circle",
      "ocr_reference": "4, 6",
      "start_time": 2.1,
      "end_time": 5.4
    }},
    {{
      "type": "highlight",
      "ocr_reference": "(C) 5 units",
      "start_time": 6.2,
      "end_time": 8.0
    }},
    {{
      "type": "formula_box",
      "content": "d = \\\\sqrt{{(x2-x1)^2 + (y2-y1)^2}}",
      "start_time": 8.1,
      "end_time": 10.0
    }},
    {{
      "type": "answer_box",
      "content": "d = 5 units",
      "start_time": 20.0,
      "end_time": 30.0
    }}
  ]
}}

OCR DATA:
{json.dumps(ocr_data, indent=2)}

TRANSCRIPT:
{json.dumps(simple_transcript, indent=2)}
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    
    try:
        plan = json.loads(response.text)
    except Exception as e:
        print("Failed to parse Gemini output. Raw text:", response.text)
        raise e
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
        
    return plan
