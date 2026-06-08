import json
from pathlib import Path

def validate_outputs(
    image_path: Path,
    annotation_path: Path,
    ocr_path: Path,
    transcript_path: Path
):
    print("Validating Educational Whiteboard Generation Plan...")

    with annotation_path.open("r", encoding="utf-8") as f:
        plan = json.load(f)

    if not isinstance(plan, dict) or "annotations" not in plan:
        raise ValueError("Invalid annotation JSON structure: Root must be an object containing an 'annotations' array.")

    annotations = plan.get("annotations", [])
    
    for i, ann in enumerate(annotations):
        atype = ann.get("type")
        if atype not in ("ocr_highlight", "whiteboard_write", "gesture_point"):
            raise ValueError(f"Annotation index {i} has invalid type: {atype}")

        if "start_time" not in ann or "end_time" not in ann:
            raise ValueError(f"Annotation index {i} is missing lifecycle timestamps.")

        if ann["start_time"] > ann["end_time"]:
            raise ValueError(f"Annotation index {i} presents time-inversion anomalies: start_time > end_time.")

        if atype == "whiteboard_write":
            if "content" not in ann or not str(ann["content"]).strip():
                raise ValueError(f"Whiteboard write index {i} lacks structural string content.")
            if "step_sequence" not in ann:
                raise ValueError(f"Whiteboard step index {i} lacks a valid 'step_sequence' layout integer.")

    print("✅ Validation complete: Pipeline strategy parameters verified.")
    return True