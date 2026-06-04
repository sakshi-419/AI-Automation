import json
from pathlib import Path

def validate_plan(image_path: Path, annotation_path: Path, ocr_path: Path, transcript_path: Path):
    print("Validating Annotation Plan...")
    
    with annotation_path.open("r", encoding="utf-8") as f:
        plan = json.load(f)
    
    ocr_data = {}
    if ocr_path.exists():
        with ocr_path.open("r", encoding="utf-8") as f:
            ocr_data = json.load(f)
            
    if isinstance(plan, list):
        annotations = plan
    else:
        annotations = plan.get("annotations", [])
    
    seen_formulas = set()
    
    # Simple validation checks
    for i, ann in enumerate(annotations):
        # 1. Missing timestamps
        if "start_time" not in ann or "end_time" not in ann:
            raise ValueError(f"Annotation {i} missing start_time or end_time: {ann}")
            
        # 2. Invalid OCR References
        ocr_ref = ann.get("ocr_reference")
        if ocr_ref:
            ref_text = str(ocr_ref).lower()
            found = False
            for item in ocr_data.get("detected_text", []):
                t = item["text"].lower()
                if ref_text in t or t in ref_text:
                    found = True
                    break
            if not found:
                raise ValueError(f"Annotation {i} has invalid OCR reference: '{ocr_ref}' not found in OCR results.")
                
        # 3. Duplicated formulas
        if ann.get("type") == "formula_box":
            content = ann.get("content", "").strip()
            if content in seen_formulas:
                raise ValueError(f"Annotation {i} contains a duplicated formula: '{content}'")
            seen_formulas.add(content)
            
    print("Validation passed. No missing timestamps, valid OCR references, and no duplicate formulas.")
    return True
