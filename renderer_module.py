import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoClip, AudioFileClip

def render_video(image_path: Path, audio_path: Path, annotation_path: Path, output_path: Path) -> None:
    """
    Renders video using precise, tight bounding-box highlights on the left 
    and progressive handwritten math steps in the empty space on the right.
    """
    base_img = cv2.imread(str(image_path))
    if base_img is None:
        raise FileNotFoundError(f"Source background image not accessible: {image_path}")
        
    img_h, img_w, _ = base_img.shape
    
    with annotation_path.open("r", encoding="utf-8") as f:
        plan_data = json.load(f)
    annotations = plan_data.get("annotations", [])

    audio_clip = AudioFileClip(str(audio_path))
    duration = audio_clip.duration

    # Initialize font assets securely 
    try:
        font_path = "C:\\Windows\\Fonts\\arial.ttf"
        main_font = ImageFont.truetype(font_path, 26)
        sub_font = ImageFont.truetype(font_path, 16)  # Smaller font specifically for subscripts
    except IOError:
        main_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    def draw_custom_math_text(draw_obj, text, base_x, base_y, font, s_font, fill_color):
        """
        Renders subscripts by lowering the offset of numerical index characters, 
        eliminating hollow square box glyph artifacts.
        """
        # Standarize variable strings to safe parsable elements
        text = text.replace("x₂", "x_2").replace("x₁", "x_1")
        text = text.replace("y₂", "y_2").replace("y₁", "y_1")
        
        curr_x = base_x
        i = 0
        while i < len(text):
            if text[i] == '_' and i + 1 < len(text):
                subscript_char = text[i+1]
                # Lower the text y-position by 10 pixels to draw an accurate subscript
                draw_obj.text((curr_x, base_y + 10), subscript_char, font=s_font, fill=fill_color)
                curr_x += draw_obj.textlength(subscript_char, font=s_font) + 2
                i += 2
            else:
                char = text[i]
                draw_obj.text((curr_x, base_y), char, font=font, fill=fill_color)
                curr_x += draw_obj.textlength(char, font=font)
                i += 1

    def make_frame(t):
        """
        Generates individual frames by layering tight text highlights and sequential math steps.
        """
        canvas = base_img.copy()
        overlay = base_img.copy()
        has_highlights = False

        # Phase 1: Tight, Targeted Highlights Only
        for step in annotations:
            start = step.get("start_time", 0.0)
            end = step.get("end_time", duration)
            step_type = step.get("type", "")
            
            if t >= start:
                coords = step.get("target_coords", [0, 0, 0, 0])
                if len(coords) == 4 and sum(coords) > 0:
                    x, y, w, h = [int(v) for v in coords]
                    
                    # RESTRICTION: Ensure highlights cannot overflow onto the solution workspace area
                    if x > 500: 
                        continue
                        
                    if step_type == "progressive_text_highlight":
                        # Smooth left-to-right highlight crawl across the text path
                        factor = min((t - start) / (end - start), 1.0) if end > start else 1.0
                        current_w = int(w * factor)
                        if current_w > 0:
                            # Tightly draw only over the text line's actual bounding box
                            cv2.rectangle(overlay, (x, y), (x + current_w, y + h), (0, 242, 255), -1)
                            has_highlights = True
                            
                    elif step_type == "marker_highlight":
                        # Strict target highlight box for coordinates or choice (C)
                        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 242, 255), -1)
                        has_highlights = True

        # Blend highlights with 35% alpha transparency to keep background text perfectly readable
        if has_highlights:
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

        # Phase 2: Convert to PIL Context for Math Text Rendering
        pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # Gather and sort handwriting steps
        write_steps = [s for s in annotations if s.get("type") == "board_write"]
        write_steps.sort(key=lambda s: s.get("start_time", 0.0))

        previous_step_finished = True

        for step in write_steps:
            start = step.get("start_time", 0.0)
            end = step.get("end_time", duration)
            content = step.get("content", "")
            
            # FORCE PLACEMENT: Enforce right-side rendering area (x >= 520) to keep the question fully visible
            x = int(step.get("x", 520))
            if x < 520:
                x = 520
            y = int(step.get("y", 100))

            if not previous_step_finished:
                # Keep subsequent steps hidden until the preceding animation finishes completely
                continue

            if t >= start:
                if t >= end:
                    # Current step is finished; render full static line text
                    draw_custom_math_text(draw, content, x, y, main_font, sub_font, (35, 35, 35))
                    previous_step_finished = True
                else:
                    # Animate characters typing out from left-to-right based on narration timing
                    char_factor = (t - start) / (end - start) if end > start else 1.0
                    char_count = int(len(content) * char_factor)
                    visible_text = content[:char_count]
                    
                    if visible_text:
                        draw_custom_math_text(draw, visible_text, x, y, main_font, sub_font, (35, 35, 35))
                    
                    # Lock timeline sequence to prevent overlapping steps
                    previous_step_finished = False
            else:
                previous_step_finished = False

        return np.array(pil_image)

    print("Rendering synchronized layers onto background canvas...")
    video_clip = VideoClip(make_frame, duration=duration)
    video_clip = video_clip.set_audio(audio_clip)

    video_clip.write_videofile(
        str(output_path),
        fps=15,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=4,
        logger=None
    )
    
    audio_clip.close()
    video_clip.close()
    print(f"Success! Video exported successfully to: {output_path}")