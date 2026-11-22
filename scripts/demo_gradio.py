"""
===============================================================================
UrbanRoadDamage — Gradio Inference Demo with Severity & Priority (PRD §20.2)
===============================================================================

PURPOSE
-------
Interactive Gradio web application that loads a YOLOv8 model from `models/best.pt`
(supplied from Colab) and assesses road damage in uploaded photographs:
  1. Detects damage using RDD2022 classes (D00, D10, D20, D40).
  2. Computes box area ratio (a = box_area / image_area).
  3. Evaluates rule-based severity (Low, Medium, High) and priority action
     (Monitor, Scheduled, Urgent) per PRD Section 20.2.
  4. Visualizes bounding boxes color-coded by severity (Green=Low, Orange=Medium, Red=High).
  5. Displays a rich summary table and overall maintenance urgency badge.

HONESTY NOTICE
--------------
Do not train the detector on this laptop. models/best.pt is supplied from Google Colab.
All test metrics must come from Colab test evaluation (Cell 9).

USAGE
-----
    python scripts/demo_gradio.py
===============================================================================
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image

# Add scripts directory to path to import severity
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from severity import analyse_image, load_config, CLASS_ID_TO_CODE

MODEL_PATH = _REPO_ROOT / "models" / "best.pt"

CLASS_FULL_NAMES = {
    0: "D00 — Longitudinal Crack",
    1: "D10 — Transverse Crack",
    2: "D20 — Alligator Crack",
    3: "D40 — Pothole",
}

# Color coding by SEVERITY level (RGB):
# High -> Red, Medium -> Orange/Yellow, Low -> Green
SEVERITY_COLORS_RGB = {
    "High": (220, 20, 60),      # Crimson Red
    "Medium": (255, 140, 0),    # Dark Orange
    "Low": (34, 139, 34),       # Forest Green
    "None": (128, 128, 128),    # Gray
}

DEFAULT_CONF = 0.25


def check_and_load_model():
    """Load model if best.pt exists and is not the placeholder."""
    if not MODEL_PATH.exists():
        return None, "File models/best.pt does not exist."
    if MODEL_PATH.stat().st_size < 1024:
        return None, "models/best.pt is still the placeholder. Copy trained best.pt from Colab."
    try:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        return model, "Model loaded successfully."
    except Exception as e:
        return None, f"Error loading model: {e}"


def run_pipeline(
    input_image: Optional[Image.Image],
    conf_threshold: float = DEFAULT_CONF,
) -> Tuple[Optional[np.ndarray], str, pd.DataFrame]:
    """
    Run detection + rule-based severity and priority assessment.
    """
    if input_image is None:
        return None, "Please upload an image.", pd.DataFrame()

    model, status_msg = check_and_load_model()
    img_rgb = np.array(input_image)
    img_h, img_w = img_rgb.shape[:2]

    # If model is not supplied yet, provide simulated mock detection for viva UI testing if requested
    if model is None:
        status_banner = (
            f"⚠️ **Detector weights not yet loaded:** {status_msg}\n\n"
            "*(Supplying placeholder detection to preview Gradio severity UI. "
            "Real detections will run once `models/best.pt` is copied from Colab.)*"
        )
        # Synthetic sample detections for demonstration
        raw_detections = [
            {"box": [int(img_w * 0.2), int(img_h * 0.4), int(img_w * 0.4), int(img_h * 0.6)], "class_id": 3, "confidence": 0.88},
            {"box": [int(img_w * 0.5), int(img_h * 0.2), int(img_w * 0.7), int(img_h * 0.35)], "class_id": 1, "confidence": 0.76},
        ]
    else:
        results = model.predict(source=img_rgb, conf=conf_threshold, verbose=False)
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        raw_detections = []
        for b, c, cl in zip(boxes, confs, classes):
            raw_detections.append({
                "box": [float(x) for x in b],
                "class_id": int(cl),
                "confidence": float(c),
            })
        status_banner = f"✅ **Inference completed with {len(raw_detections)} detections.**"

    # Evaluate Rule-based Severity & Priority per PRD §20.2
    assessment = analyse_image(raw_detections, (img_h, img_w))
    overall_sev = assessment["overall_severity"]
    prio_action = assessment["priority_label"]
    reason = assessment.get("reason", "")

    # Draw annotated boxes
    annotated = img_rgb.copy()
    for det in assessment.get("per_detection", []):
        x1, y1, x2, y2 = map(int, det["box"])
        sev = det.get("severity", "Low")
        color = SEVERITY_COLORS_RGB.get(sev, (0, 255, 0))
        c_name = CLASS_FULL_NAMES.get(det["class_id"], det.get("class_code", "Defect"))

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # Label text
        label = f"{c_name} | {det['confidence']:.2f} | Sev: {sev}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Summary table
    table_rows = []
    for d in assessment.get("ranked_detections", []):
        table_rows.append({
            "Class": CLASS_FULL_NAMES.get(d["class_id"], d.get("class_code")),
            "Confidence": f"{d['confidence']:.3f}",
            "Area Ratio": f"{d['area_ratio']:.2%}",
            "Severity": d["severity"],
            "Priority Action": d["priority_action"],
            "Box [x1, y1, x2, y2]": f"[{int(d['box'][0])}, {int(d['box'][1])}, {int(d['box'][2])}, {int(d['box'][3])}]",
        })

    summary_df = pd.DataFrame(table_rows) if table_rows else pd.DataFrame([{
        "Class": "No defects detected",
        "Confidence": "—",
        "Area Ratio": "0.0%",
        "Severity": "None",
        "Priority Action": "No action",
        "Box [x1, y1, x2, y2]": "—",
    }])

    # Build markdown result card
    badge_colors = {"High": "red", "Medium": "orange", "Low": "green", "None": "gray"}
    sev_color = badge_colors.get(overall_sev, "gray")

    markdown_card = f"""
### Assessment Summary
- **Overall Severity**: <span style="color:{sev_color}; font-weight:bold; font-size:1.2em;">{overall_sev}</span>
- **Recommended Maintenance Action**: <span style="color:{sev_color}; font-weight:bold; font-size:1.2em;">{prio_action}</span>
- **Rule Triggered**: *{reason}*
- **Status**: {status_banner}
"""

    return annotated, markdown_card, summary_df


def build_app() -> gr.Blocks:
    with gr.Blocks(title="UrbanRoadDamage — Road Damage Assessment Demo") as app:
        gr.Markdown(
            """
            # 🛣️ Urban Pothole & Road Damage Assessment System
            ### Project 23 | Deep Learning - Semester V | Municipal Infrastructure
            **Primary Pipeline**: YOLOv8s Baseline / RetinaNet (ONNX) + Rule-Based Severity & Priority (§20.2)
            """
        )

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Upload Road Image")
                conf_slider = gr.Slider(minimum=0.05, maximum=1.0, value=DEFAULT_CONF, step=0.05, label="Confidence Threshold")
                detect_btn = gr.Button("🔍 Run Assessment", variant="primary")
            with gr.Column():
                output_img = gr.Image(type="numpy", label="Annotated Road Defects (Color-coded by Severity)")
                summary_md = gr.Markdown("### Assessment Summary\n*Upload an image and run assessment.*")

        with gr.Row():
            summary_table = gr.Dataframe(
                label="Detailed Detections & Priority Ranking",
                interactive=False,
            )

        detect_btn.click(
            fn=run_pipeline,
            inputs=[input_img, conf_slider],
            outputs=[output_img, summary_md, summary_table],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
