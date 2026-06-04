import easyocr
import json
from pathlib import Path
import cv2

def extract_ocr(image_path: Path, output_path: Path) -> dict:
    print("Processing OCR...")
    # Initialize EasyOCR reader (using CPU by default for broader compatibility)
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Read image dimensions
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w, _ = image.shape

    # Extract text
    results = reader.readtext(str(image_path))
    
    detected_text = []
    full_text = []
    
    for bbox, text, prob in results:
        # bbox is returned as [top-left, top-right, bottom-right, bottom-left]
        tl, tr, br, bl = bbox
        x1, y1 = int(tl[0]), int(tl[1])
        x2, y2 = int(br[0]), int(br[1])
        
        detected_text.append({
            "text": text,
            "bbox": [x1, y1, x2, y2]
        })
        full_text.append(text)
        
    ocr_data = {
        "image_width": w,
        "image_height": h,
        "question_text": " ".join(full_text),
        "detected_text": detected_text
    }
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(ocr_data, f, indent=2, ensure_ascii=False)
        
    return ocr_data
