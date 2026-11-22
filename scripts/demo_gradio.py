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

from severity import analyse_image, load_config, CLASS_ID_TO_CODE, filter_road_surface_roi

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

DEFAULT_CONF = 0.40


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
    filter_horizon: bool = True,
) -> Tuple[Optional[np.ndarray], str, pd.DataFrame]:
    """
    Run detection + rule-based severity and priority assessment.
    Includes PRD §20.2 Rule 2 Road Surface Horizon Prior to eliminate false positives
    on tree canopies, sky, and non-road regions.
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

        # Apply PRD §20.2 Rule 2 Road Surface Horizon Filter
        discarded_count = 0
        if filter_horizon:
            raw_detections, discarded = filter_road_surface_roi(raw_detections, (img_h, img_w), horizon_ratio=0.38)
            discarded_count = len(discarded)

        if raw_detections:
            status_banner = f"✅ **Inference completed: {len(raw_detections)} road defect(s) detected.**"
            if discarded_count > 0:
                status_banner += f" *(Suppressed {discarded_count} non-road horizon false positive(s) like trees/sky)*"
        else:
            status_banner = "✅ **Road surface clear: No defects detected above confidence threshold.**"
            if discarded_count > 0:
                status_banner += f" *(Filtered out {discarded_count} non-road false positive(s) on trees/sky/structures)*"

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

    # Build modern summary card
    badge_bg = {
        "High": "#ef4444",
        "Medium": "#f97316",
        "Low": "#10b981",
        "None": "#64748b",
    }
    action_bg = {
        "Urgent": "#dc2626",
        "Scheduled": "#ea580c",
        "Monitor": "#059669",
        "No action": "#475569",
    }
    sev_bg = badge_bg.get(overall_sev, "#64748b")
    prio_bg = action_bg.get(prio_action, "#475569")

    markdown_card = f"""
<div style="background:#0f172a; border:1px solid #334155; border-radius:12px; padding:16px; margin-bottom:12px;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
    <div>
      <span style="color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">Overall Severity</span>
      <div style="margin-top:4px;">
        <span style="background:{sev_bg}; color:#ffffff; padding:4px 14px; border-radius:6px; font-size:1.05rem; font-weight:700;">
          {overall_sev.upper()}
        </span>
      </div>
    </div>
    <div>
      <span style="color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">Maintenance Action</span>
      <div style="margin-top:4px;">
        <span style="background:{prio_bg}; color:#ffffff; padding:4px 14px; border-radius:6px; font-size:1.05rem; font-weight:700;">
          {prio_action.upper()}
        </span>
      </div>
    </div>
  </div>

  <div style="border-top:1px solid #1e293b; padding-top:10px; margin-top:8px;">
    <div style="color:#cbd5e1; font-size:0.9rem; margin-bottom:4px;">
      <strong style="color:#f8fafc;">Rule Triggered:</strong> <span style="color:#e2e8f0;">{reason}</span>
    </div>
    <div style="color:#94a3b8; font-size:0.85rem;">
      {status_banner}
    </div>
  </div>
</div>
"""

    return annotated, markdown_card, summary_df


def build_app() -> gr.Blocks:
    custom_css = """
    .gradio-container { max-width: 1380px !important; margin: auto !important; }
    #detect_btn { font-size: 1.05rem !important; font-weight: 600 !important; }
    """

    with gr.Blocks(title="Road Damage Assessment System", css=custom_css, theme=gr.themes.Soft()) as app:
        gr.HTML(
            """
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; border-bottom:2px solid #334155; padding-bottom:14px; margin-bottom:18px;">
              <div>
                <h1 style="margin:0; font-size:1.85rem; font-weight:700; color:#f8fafc; display:flex; align-items:center; gap:10px;">
                  <span>🛣️</span> Urban Pothole & Road Damage Assessment System
                </h1>
                <p style="margin:6px 0 0 0; color:#94a3b8; font-size:0.95rem;">
                  Automated Pavement Distress Detection & Explainable Municipal Priority Dispatch
                </p>
              </div>
              <div style="display:flex; gap:8px; margin-top:8px; flex-wrap:wrap;">
                <span style="background:#1e293b; color:#38bdf8; border:1px solid #0284c7; padding:4px 10px; border-radius:8px; font-size:0.75rem; font-weight:600;">YOLOv8 Edge Vision</span>
                <span style="background:#1e293b; color:#4ade80; border:1px solid #16a34a; padding:4px 10px; border-radius:8px; font-size:0.75rem; font-weight:600;">RDD2022 Standards</span>
                <span style="background:#1e293b; color:#f59e0b; border:1px solid #d97706; padding:4px 10px; border-radius:8px; font-size:0.75rem; font-weight:600;">PRD §20.2 Rule Engine</span>
              </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(type="pil", label="Upload Road Image")
                conf_slider = gr.Slider(minimum=0.05, maximum=1.0, value=DEFAULT_CONF, step=0.05, label="Confidence Threshold")
                horizon_filter = gr.Checkbox(value=True, label="🛣️ Suppress Non-Road False Positives (PRD §20.2 Horizon Prior: ignores trees/sky)")
                detect_btn = gr.Button("🔍 Run AI Road Assessment", variant="primary", size="lg", elem_id="detect_btn")

                # Sample test images (dynamic discovery)
                samples_dir = _REPO_ROOT / "sample_inputs"
                sample_files = sorted(
                    [str(p) for p in samples_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
                )
                existing_samples = [[f] for f in sample_files]
                if existing_samples:
                    gr.Examples(
                        examples=existing_samples,
                        inputs=input_img,
                        label="📷 Click a sample image to test",
                    )
            with gr.Column(scale=1):
                output_img = gr.Image(type="numpy", label="Annotated Road Defects (Color-coded by Severity)")
                summary_md = gr.Markdown(
                    """
                    <div style="background:#0f172a; border:1px solid #334155; border-radius:12px; padding:16px; color:#94a3b8;">
                      <h4 style="margin:0 0 6px 0; color:#f8fafc;">Assessment Summary</h4>
                      <em>Upload a road image or click any sample thumbnail on the left, then click <strong>Run AI Road Assessment</strong>.</em>
                    </div>
                    """
                )

        with gr.Row():
            summary_table = gr.Dataframe(
                label="Detailed Detections & Priority Ranking",
                interactive=False,
            )

        detect_btn.click(
            fn=run_pipeline,
            inputs=[input_img, conf_slider, horizon_filter],
            outputs=[output_img, summary_md, summary_table],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
