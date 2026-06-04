import cv2
import json
import math
import numpy as np
from pathlib import Path
from moviepy.editor import AudioFileClip, VideoClip
import matplotlib.pyplot as plt
import io
from PIL import Image

FPS = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_PADDING = 10

class BoundingBox:
    def __init__(self, x, y, w, h):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)

def measure_text(text, img_w):
    font_scale = max(0.6, img_w / 1500)
    thickness = 2
    size, _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    w = size[0] + BOX_PADDING * 2
    h = size[1] + BOX_PADDING * 2
    return w, h, font_scale, thickness

def render_latex_to_image(formula, font_size=24):
    if not formula.strip().startswith('$'):
        formula = f"${formula.strip()}$"
    try:
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, formula, fontsize=font_size, color=(24/255, 38/255, 56/255))
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True, dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        pil_img = Image.open(buf).convert("RGBA")
        return np.array(pil_img)
    except Exception as e:
        print(f"Latex rendering failed: {e}")
        return None

def overlay_transparent(bg_img, img_to_overlay_t, x, y, progress=1.0):
    bg_img = bg_img.copy()
    
    w = img_to_overlay_t.shape[1]
    h = img_to_overlay_t.shape[0]
    
    vis_w = max(1, int(w * progress))
    if vis_w == 0: return bg_img
    
    img_to_overlay = img_to_overlay_t[:, :vis_w]
    
    if x >= bg_img.shape[1] or y >= bg_img.shape[0]:
        return bg_img
        
    h, w, _ = img_to_overlay.shape
    if x + w > bg_img.shape[1]: w = bg_img.shape[1] - x
    if y + h > bg_img.shape[0]: h = bg_img.shape[0] - y
    
    img_to_overlay = img_to_overlay[:h, :w]
    
    b,g,r,a = cv2.split(img_to_overlay)
    overlay_color = cv2.merge((b,g,r))
    
    mask = cv2.medianBlur(a, 1)
    
    roi = bg_img[y:y+h, x:x+w]
    img1_bg = cv2.bitwise_and(roi, roi, mask = cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask = mask)
    
    bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)
    return bg_img

def get_sync_progress(t, start_t, end_t, transcript_words):
    if not transcript_words:
        return min(1.0, max(0.0, (t - start_t) / max(0.1, end_t - start_t)))
        
    words_in_span = [w for w in transcript_words if w.get("start",0) >= start_t - 0.5 and w.get("end",0) <= end_t + 0.5]
    if not words_in_span:
        return min(1.0, max(0.0, (t - start_t) / max(0.1, end_t - start_t)))
        
    spoken = sum(1 for w in words_in_span if t >= w.get("end",0))
    for w in words_in_span:
        ws = w.get("start",0)
        we = w.get("end",0)
        if ws <= t < we:
            spoken += (t - ws) / max(0.01, we - ws)
            
    return min(1.0, spoken / len(words_in_span))

def draw_annotation(frame, ann, t, img_w, transcript_words):
    atype = ann.get("type", "text")
    start_t = ann.get("start_time", 0)
    end_t = ann.get("end_time", 0)
    
    progress = get_sync_progress(t, start_t, end_t, transcript_words)
    color = (42, 91, 220)
    
    if atype == "arrow":
        start_pt = ann.get("from", [0, 0])
        end_pt = ann.get("to", [0, 0])
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        cur_end = (int(start_pt[0] + dx * progress), int(start_pt[1] + dy * progress))
        cv2.arrowedLine(frame, tuple(start_pt), cur_end, color, 3, tipLength=0.1, line_type=cv2.LINE_AA)
        return frame
        
    bbox = ann.get("_bbox")
    if not bbox: return frame
        
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    text = ann.get("content", "")
    
    if atype == "highlight":
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (88, 242, 255), -1)
        cv2.addWeighted(overlay, progress * 0.6, frame, 1.0 - (progress * 0.6), 0, frame)
        
    elif atype == "circle":
        center = (x + w//2, y + h//2)
        axes = (int(w//2 * progress), int(h//2 * progress))
        if axes[0] > 0 and axes[1] > 0:
            cv2.ellipse(frame, center, axes, 0, 0, 360, color, 3, cv2.LINE_AA)
            
    elif atype == "underline":
        end_x = x + int(w * progress)
        cv2.line(frame, (x, y+h), (end_x, y+h), color, 3, cv2.LINE_AA)
        
    elif atype in ("answer_box", "formula_box"):
        if progress < 0.5:
            box_prog = progress * 2
            text_prog = 0
        else:
            box_prog = 1.0
            text_prog = (progress - 0.5) * 2
            
        bw, bh = int(w * box_prog), int(h * box_prog)
        if bw > 0 and bh > 0:
            bg_color = (240, 248, 255) if atype == "answer_box" else (255, 250, 240)
            line_color = color if atype == "answer_box" else (40, 160, 40)
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), bg_color, -1)
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), line_color, 3, cv2.LINE_AA)
            
        latex_img = ann.get("_rendered_img")
        if latex_img is not None:
            px = x + BOX_PADDING
            py = y + max(0, (h - latex_img.shape[0]) // 2)
            frame = overlay_transparent(frame, latex_img, px, py, text_prog)
        else:
            _, _, font_scale, thickness = measure_text(text, img_w)
            chars = max(1, int(math.ceil(len(text) * text_prog)))
            visible = text[:chars]
            if visible:
                cv2.putText(frame, visible, (x + BOX_PADDING, y + h - BOX_PADDING), FONT, font_scale, (24, 38, 56), thickness, cv2.LINE_AA)
                
    else: 
        latex_img = ann.get("_rendered_img")
        if latex_img is not None:
            frame = overlay_transparent(frame, latex_img, x, y, progress)
        else:
            _, _, font_scale, thickness = measure_text(text, img_w)
            chars = max(1, int(math.ceil(len(text) * progress)))
            visible = text[:chars]
            if visible:
                cv2.putText(frame, visible, (x + BOX_PADDING, y + h - BOX_PADDING), FONT, font_scale, (24, 38, 56), thickness, cv2.LINE_AA)

    return frame

def render_video(image_path: Path, audio_path: Path, annotation_path: Path, output_path: Path) -> None:
    print("Rendering Video...")
    image = cv2.imread(str(image_path))
    rgb_bg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = image.shape
    
    # Enforce target resolution (≈1000x575) while preserving aspect ratio
    target_w, target_h = 1000, 575
    if img_w != target_w or img_h != target_h:
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgb_bg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = image.shape
    
    solution_x = int(img_w * 0.55)
    
    with annotation_path.open("r", encoding="utf-8") as f:
        plan = json.load(f)
        
    ocr_path = annotation_path.parent / "ocr.json"
    ocr_data = {}
    if ocr_path.exists():
        with ocr_path.open("r", encoding="utf-8") as f:
            ocr_data = json.load(f)
            
    transcript_path = annotation_path.parent / "transcript.json"
    transcript_words = []
    if transcript_path.exists():
        with transcript_path.open("r", encoding="utf-8") as f:
            tr_data = json.load(f)
            for seg in tr_data.get("segments", []):
                for w in seg.get("words", []):
                    transcript_words.append(w)
            
    def get_ocr_bbox(ref_text):
        if not ref_text: return None
        ref_text = str(ref_text).lower()
        for item in ocr_data.get("detected_text", []):
            text = item["text"].lower()
            if ref_text in text or text in ref_text:
                x1, y1, x2, y2 = item["bbox"]
                return BoundingBox(x1, y1, x2 - x1, y2 - y1)
        return None

    if isinstance(plan, list):
        annotations = plan
    else:
        annotations = plan.get("annotations", [])
    
    current_sol_y = 50
    
    for ann in annotations:
        if ann.get("type") == "arrow":
            from_ocr = ann.get("from_ocr")
            if from_ocr:
                from_box = get_ocr_bbox(from_ocr)
                if from_box: ann["from"] = [from_box.x + from_box.w//2, from_box.y + from_box.h//2]
            to_ocr = ann.get("to_ocr")
            if to_ocr:
                to_box = get_ocr_bbox(to_ocr)
                if to_box: ann["to"] = [to_box.x + to_box.w//2, to_box.y + to_box.h//2]
            continue
            
        # Process annotation positions and bounding boxes
        # Try to get bounding box from OCR reference if provided
        ocr_ref = ann.get("ocr_reference")
        box = None
        if ocr_ref:
            box = get_ocr_bbox(ocr_ref)
        
        # If no OCR reference or OCR not found, handle based on annotation type
        ann_type = ann.get("type")
        if box is None:
            if ann_type in ("formula_box", "answer_box"):
                # Use a fixed position within the Solution Workspace for formulas/answers
                default_x = solution_x + 20
                # Choose a consistent Y coordinate for formula evolution (e.g., 80 pixels from top of workspace)
                default_y = 80
                box = BoundingBox(default_x, default_y, 100, 30)  # placeholder size; will be updated after rendering
                ann["_bbox"] = box
            else:
                # For other annotation types (highlight, circle, arrow, underline) we require a valid OCR reference.
                # If missing, skip this annotation (it will be ignored during rendering).
                continue
        else:
            ann["_bbox"] = box
        
        # At this point, we have a bounding box (either from OCR or default) and can process content.
        content = ann.get("content", "")
        if content:
            if ann_type in ("formula_box", "answer_box", "text"):
                latex_img = render_latex_to_image(content)
                if latex_img is not None:
                    ann["_rendered_img"] = latex_img
                    bw = latex_img.shape[1] + BOX_PADDING * 2
                    bh = latex_img.shape[0] + BOX_PADDING * 2
                else:
                    bw, bh, _, _ = measure_text(content, img_w)
            else:
                bw, bh, _, _ = measure_text(content, img_w)
            # Update bounding box dimensions based on rendered size
            if box:
                box.w = bw
                box.h = bh
                ann["_bbox"] = box
            else:
                ann["_bbox"] = BoundingBox(solution_x + 20, current_sol_y, bw, bh)
        
        # Formula evolution: do not stack vertically; formulas reuse same region.
        # No change to current_sol_y for formula types.

    def make_frame(t):
        frame = rgb_bg.copy()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Draw solution workspace divider
        cv2.line(bgr, (solution_x - 10, 0), (solution_x - 10, img_h), (200, 200, 200), 2, cv2.LINE_AA)
        
        # Determine active annotations, enforce max 3 simultaneous
        active_anns = [ann for ann in annotations if ann.get("start_time", 0) <= t <= ann.get("end_time", 0)]
        if len(active_anns) > 3:
            active_anns = active_anns[:3]
        
        for ann in active_anns:
            # Clamp bounding boxes to image boundaries
            bbox = ann.get("_bbox")
            if bbox:
                x = max(0, min(bbox.x, img_w - bbox.w))
                y = max(0, min(bbox.y, img_h - bbox.h))
                bbox.x, bbox.y = x, y
                ann["_bbox"] = bbox
            bgr = draw_annotation(bgr, ann, t, img_w, transcript_words)
        
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
    with AudioFileClip(str(audio_path)) as audio_clip:
        duration = audio_clip.duration
        video_clip = VideoClip(make_frame, duration=duration).set_fps(FPS)
        final_clip = video_clip.set_audio(audio_clip)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_clip.write_videofile(
            str(output_path),
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            verbose=False,
            logger=None
        )
        final_clip.close()
    video_clip.close()
