"""
===============================================================================
UrbanRoadDamage — Gradio Inference Demo
===============================================================================

PURPOSE
-------
This script launches an interactive **Gradio web application** that lets a user
upload a road-surface photograph and instantly see which damage types (cracks,
potholes) the trained YOLOv8s detector finds.

HOW IT WORKS (high-level)
-------------------------
1. We load a trained YOLOv8s model from  ``models/best.pt``.
   That file must be copied from Google Colab after training — the repo ships
   only a text placeholder.
2. The user uploads an image through the browser.
3. We run ``model.predict()`` with a configurable **confidence threshold**
   (default 0.25).  YOLOv8 internally:
   - resizes the image to 640×640,
   - runs the CNN forward pass,
   - applies Non-Maximum Suppression (NMS) to filter overlapping boxes.
4. We draw bounding boxes + class labels on the image using OpenCV.
5. We build a summary table (class, confidence, box coordinates) with pandas.
6. Both the annotated image and the table are returned to the Gradio UI.

CLASS MAPPING  (from the RDD-2022 dataset)
------------------------------------------
  D00 → Longitudinal Crack   (runs along the direction of travel)
  D10 → Transverse Crack     (runs across the road)
  D20 → Alligator Crack      (interconnected pattern, like alligator skin)
  D40 → Pothole              (bowl-shaped hole in the pavement)

VIVA NOTES
----------
*  **Why YOLOv8s?**  It balances accuracy and speed — the "s" (small) variant
   has ~11 M parameters, fast enough for real-time inference on a laptop GPU,
   yet accurate enough for road-damage detection benchmarks.
*  **Why Gradio?**  Gradio gives us a shareable web UI with zero front-end
   code.  Examiners can try the model live during the viva.
*  **Confidence threshold:**  Detections below this score are discarded.
   Raising it reduces false positives; lowering it catches more damage but
   may introduce noise.  0.25 is the YOLO default.

USAGE
-----
    cd UrbanRoadDamage
    python scripts/demo_gradio.py          # opens http://127.0.0.1:7860

METRICS
-------
TODO: fill with real metrics from Colab Cell 9

STATUS: PARTIAL — demo code is DONE; model weights are PLANNED (need Colab).
===============================================================================
"""

# ── Standard-library imports ────────────────────────────────────────────────
import os          # filesystem path operations
import sys         # system-level utilities (e.g. exit on error)
from pathlib import Path  # object-oriented file paths

# ── Third-party imports ─────────────────────────────────────────────────────
import cv2                    # OpenCV — drawing boxes, colour conversions
import numpy as np            # numerical arrays
import pandas as pd           # tabular summary of detections
import gradio as gr           # web-UI framework
from PIL import Image         # Pillow — load user-uploaded images
from ultralytics import YOLO  # YOLOv8 model loading & inference


# ============================================================================
# 1.  CONSTANTS & CONFIGURATION
# ============================================================================

# Path to the trained YOLOv8s weights.
# Resolve relative to *this script's* location so the demo works regardless
# of the caller's working directory.
#   scripts/demo_gradio.py  →  .. (repo root)  →  models/best.pt
SCRIPT_DIR = Path(__file__).resolve().parent          # .../scripts/
REPO_ROOT  = SCRIPT_DIR.parent                        # .../UrbanRoadDamage/
MODEL_PATH = REPO_ROOT / "models" / "best.pt"         # trained weights

# Human-readable class names keyed by the integer class-id that YOLO returns.
# The order matches the data.yaml used during training (alphabetical by code):
#   0 → D00,  1 → D10,  2 → D20,  3 → D40
CLASS_NAMES = {
    0: "D00 — Longitudinal Crack",
    1: "D10 — Transverse Crack",
    2: "D20 — Alligator Crack",
    3: "D40 — Pothole",
}

# Distinct colours for each class (BGR format for OpenCV).
# Chosen for strong visual contrast on asphalt-grey backgrounds.
#   Green  → longitudinal cracks (subtle, common)
#   Blue   → transverse cracks
#   Orange → alligator cracks (severe)
#   Red    → potholes (most dangerous)
CLASS_COLORS_BGR = {
    0: (0, 200, 0),       # green  — D00
    1: (255, 128, 0),     # blue   — D10  (note: BGR, so this is orange in RGB)
    2: (0, 165, 255),     # orange — D20  (BGR → RGB orange)
    3: (0, 0, 255),       # red    — D40
}

# Default confidence threshold.
# YOLOv8 discards any detection whose objectness × class-probability product
# falls below this value.  Formula:  score = P(object) × P(class | object).
DEFAULT_CONF = 0.25


# ============================================================================
# 2.  MODEL LOADING
# ============================================================================

def load_model(model_path: Path) -> YOLO:
    """
    Load the trained YOLOv8 model from disk.

    Parameters
    ----------
    model_path : Path
        Absolute path to ``best.pt`` (the checkpoint saved after training).

    Returns
    -------
    YOLO
        An ultralytics YOLO object ready for ``.predict()``.

    Raises
    ------
    FileNotFoundError
        If ``best.pt`` does not exist or is the placeholder text file.

    Design note (viva)
    ------------------
    We load the model **once** at startup (module level) rather than on every
    request.  This avoids the ~2-second overhead of re-loading weights into
    GPU/CPU memory for each image.
    """
    # ── Guard: file must exist ──────────────────────────────────────────
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at:\n  {model_path}\n\n"
            "👉  Train the model in Google Colab, download best.pt, and copy "
            "it into the  models/  folder."
        )

    # ── Guard: file must NOT be the placeholder ─────────────────────────
    # The placeholder is a small text file (<200 bytes).  A real .pt file
    # is at least several megabytes, so a size check is a quick sanity test.
    if model_path.stat().st_size < 1024:
        raise FileNotFoundError(
            f"The file at  {model_path}  is still the placeholder.\n\n"
            "👉  Replace it with the real best.pt from Colab."
        )

    # ── Load with ultralytics ───────────────────────────────────────────
    # YOLO() automatically detects the architecture (YOLOv8s) from the
    # checkpoint metadata and moves weights to the best available device
    # (CUDA GPU if present, else CPU).
    model = YOLO(str(model_path))
    return model


# ============================================================================
# 3.  INFERENCE + VISUALISATION
# ============================================================================

def detect_damage(
    input_image: Image.Image,
    conf_threshold: float = DEFAULT_CONF,
) -> tuple:
    """
    Run YOLOv8 inference on a single image and return annotated results.

    Parameters
    ----------
    input_image : PIL.Image.Image
        The road-surface photo uploaded by the user via Gradio.
    conf_threshold : float
        Minimum detection confidence (0.0–1.0).  Boxes below this score are
        discarded by YOLOv8's post-processing (NMS stage).

    Returns
    -------
    annotated_image : np.ndarray
        The input image with bounding boxes, class labels, and confidence
        scores drawn on top (RGB, uint8).
    summary_df : pd.DataFrame
        One row per detection: class name, confidence, x_min, y_min,
        x_max, y_max.

    How NMS works (viva explanation)
    --------------------------------
    After the network proposes thousands of candidate boxes, **Non-Maximum
    Suppression** keeps only the best ones:
      1. Sort all boxes by confidence (high → low).
      2. Take the top box; mark it as "kept".
      3. Compute IoU (Intersection over Union) of that box with every
         remaining box.  Remove any box whose IoU > 0.45 (they overlap too
         much and probably detect the same object).
      4. Repeat from step 2 with the remaining boxes.
    This ensures one box per object, not dozens of overlapping rectangles.
    """
    # ── 3-a.  Check that the model is loaded ────────────────────────────
    # If the user hasn't replaced the placeholder yet, `model` will be None.
    if model is None:
        raise gr.Error(
            "⚠️  Model not loaded.  Replace models/best.pt with the real "
            "weights from Colab and restart the demo."
        )

    # ── 3-b.  Convert PIL → numpy (RGB) ────────────────────────────────
    # Gradio hands us a PIL Image.  Ultralytics accepts PIL directly, but
    # we also need a numpy copy to draw on with OpenCV.
    img_rgb = np.array(input_image)                 # shape: (H, W, 3), uint8

    # ── 3-c.  Run YOLOv8 inference ──────────────────────────────────────
    # model.predict() performs:
    #   • letterbox resize to 640×640 (adds grey padding to keep aspect ratio)
    #   • forward pass through the YOLOv8s backbone + head
    #   • NMS to filter overlapping / low-confidence boxes
    # verbose=False silences per-image logs in the terminal.
    results = model.predict(
        source=img_rgb,
        conf=conf_threshold,  # minimum confidence to keep a detection
        verbose=False,        # suppress console spam
    )

    # results is a list (one entry per image).  We sent a single image,
    # so we take results[0].
    result = results[0]

    # ── 3-d.  Extract boxes, classes, confidences ───────────────────────
    # result.boxes is an ultralytics Boxes object with these tensors:
    #   .xyxy   — (N, 4) tensor of [x_min, y_min, x_max, y_max] in pixels
    #   .conf   — (N,) tensor of confidence scores
    #   .cls    — (N,) tensor of integer class indices
    boxes      = result.boxes.xyxy.cpu().numpy()   # (N, 4)  float32
    confidences = result.boxes.conf.cpu().numpy()  # (N,)    float32
    class_ids   = result.boxes.cls.cpu().numpy().astype(int)  # (N,) int

    # ── 3-e.  Draw bounding boxes on the image ─────────────────────────
    # We work on a copy so the original stays clean.
    annotated = img_rgb.copy()

    # OpenCV needs BGR for drawing, but Gradio expects RGB output.
    # Strategy: draw in RGB directly — cv2.rectangle / putText don't care
    # about channel order, only about consistent colour tuples.  We just
    # need to pass RGB colours instead of the default BGR ones.

    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        # Unpack box coordinates and round to nearest pixel.
        x_min, y_min, x_max, y_max = map(int, box)

        # Look up the human-readable class name.
        # .get() with a fallback handles unexpected class ids gracefully.
        label_text = CLASS_NAMES.get(cls_id, f"Class {cls_id}")

        # Compose the annotation string:  "D40 — Pothole 0.87"
        annotation = f"{label_text} {conf:.2f}"

        # Pick the class colour.  Convert BGR → RGB for our RGB image.
        bgr = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))
        rgb_color = (bgr[2], bgr[1], bgr[0])  # reverse channels

        # Draw the bounding-box rectangle (2-pixel thick).
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max),
                       color=rgb_color, thickness=2)

        # ── Draw a filled background strip behind the text so it's
        # readable even on bright pavement.
        # We measure the text size first, then draw a filled rectangle,
        # then overlay the white text.
        (text_w, text_h), baseline = cv2.getTextSize(
            annotation,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            thickness=1,
        )
        # Background rectangle: top-left sits just above the box edge.
        cv2.rectangle(
            annotated,
            (x_min, y_min - text_h - baseline - 4),
            (x_min + text_w + 4, y_min),
            color=rgb_color,
            thickness=-1,  # -1 = filled
        )
        # White text on top of the coloured strip.
        cv2.putText(
            annotated,
            annotation,
            org=(x_min + 2, y_min - baseline - 2),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),  # white
            thickness=1,
            lineType=cv2.LINE_AA,   # anti-aliased for smooth edges
        )

    # ── 3-f.  Build a summary table ─────────────────────────────────────
    # One row per detection; columns: Class, Confidence, x_min … y_max.
    # This helps the examiner inspect exact numbers without reading pixels.
    if len(boxes) > 0:
        summary_df = pd.DataFrame({
            "Class":      [CLASS_NAMES.get(c, f"Class {c}") for c in class_ids],
            "Confidence":  [f"{c:.4f}" for c in confidences],
            "x_min":       [int(b[0]) for b in boxes],
            "y_min":       [int(b[1]) for b in boxes],
            "x_max":       [int(b[2]) for b in boxes],
            "y_max":       [int(b[3]) for b in boxes],
        })
    else:
        # No detections → show a friendly empty table.
        summary_df = pd.DataFrame({
            "Class": ["No damage detected"],
            "Confidence": ["—"],
            "x_min": ["—"], "y_min": ["—"],
            "x_max": ["—"], "y_max": ["—"],
        })

    return annotated, summary_df


# ============================================================================
# 4.  GRADIO USER INTERFACE
# ============================================================================

def build_gradio_app() -> gr.Blocks:
    """
    Construct and return the Gradio Blocks application.

    Layout
    ------
    ┌──────────────────────────────────────────┐
    │  Title + description                     │
    ├────────────────────┬─────────────────────┤
    │  Upload image      │  Annotated output   │
    ├────────────────────┴─────────────────────┤
    │  Confidence slider                       │
    ├──────────────────────────────────────────┤
    │  Detection summary table                 │
    └──────────────────────────────────────────┘

    Why gr.Blocks instead of gr.Interface?
    --------------------------------------
    ``gr.Blocks`` gives us full control over the layout (rows, columns,
    sliders below the images, a dataframe beneath everything).
    ``gr.Interface`` would work for a minimal demo but doesn't support the
    two-column + table layout we want.
    """
    with gr.Blocks(
        title="UrbanRoadDamage — Road Damage Detector",
        # A custom CSS tweak: slightly larger heading.
        css=".gradio-container h1 { font-size: 2rem; }",
    ) as app:

        # ── Title & description ─────────────────────────────────────────
        gr.Markdown(
            """
            # 🛣️ UrbanRoadDamage — Road Damage Detection Demo

            Upload a road-surface image and the YOLOv8s model will detect
            **four types of pavement damage**:

            | Code | Damage Type | Description |
            |------|-------------|-------------|
            | D00  | Longitudinal Crack | Crack running *along* the road |
            | D10  | Transverse Crack   | Crack running *across* the road |
            | D20  | Alligator Crack    | Interconnected crack pattern |
            | D40  | Pothole            | Bowl-shaped surface hole |

            Adjust the **confidence slider** to control sensitivity.

            **Model performance:** TODO: fill with real metrics from Colab Cell 9
            """
        )

        # ── Image row: input on the left, output on the right ──────────
        with gr.Row():
            # Input: the user drags & drops or clicks to upload a photo.
            input_image = gr.Image(
                label="📤 Upload a road image",
                type="pil",        # Gradio converts the upload to PIL.Image
                height=400,
            )
            # Output: the same image with bounding boxes drawn on it.
            output_image = gr.Image(
                label="🔍 Detected damage",
                type="numpy",      # we return a numpy array from detect_damage
                height=400,
            )

        # ── Confidence threshold slider ─────────────────────────────────
        # Minimum = 0.05 (don't allow 0 — too many false positives).
        # Maximum = 1.0.
        # Step of 0.05 gives 20 discrete positions — fine-grained enough.
        conf_slider = gr.Slider(
            minimum=0.05,
            maximum=1.0,
            value=DEFAULT_CONF,
            step=0.05,
            label="Confidence Threshold",
            info=(
                "Only show detections above this confidence score. "
                "Lower = more boxes (possibly noisy). "
                "Higher = fewer but more certain boxes."
            ),
        )

        # ── Run button ──────────────────────────────────────────────────
        run_btn = gr.Button("🚀  Detect Damage", variant="primary")

        # ── Detection summary table ─────────────────────────────────────
        # gr.Dataframe renders a pandas DataFrame as an interactive table.
        output_table = gr.Dataframe(
            label="📊 Detection Summary",
            headers=["Class", "Confidence", "x_min", "y_min", "x_max", "y_max"],
            interactive=False,  # read-only
        )

        # ── Wire up the button click ───────────────────────────────────
        # When the user clicks "Detect Damage":
        #   inputs  → (uploaded image, slider value)
        #   outputs → (annotated image, summary table)
        run_btn.click(
            fn=detect_damage,
            inputs=[input_image, conf_slider],
            outputs=[output_image, output_table],
        )

        # ── Example images (optional) ──────────────────────────────────
        # If the user puts sample images in  data/samples/, they appear as
        # clickable thumbnails.  We only add this component if the folder
        # exists and is non-empty.
        samples_dir = REPO_ROOT / "data" / "samples"
        if samples_dir.exists():
            sample_files = sorted(samples_dir.glob("*.jpg")) + \
                           sorted(samples_dir.glob("*.png"))
            if sample_files:
                gr.Examples(
                    examples=[[str(f)] for f in sample_files[:6]],
                    inputs=input_image,
                    label="📷 Try a sample image",
                )

    return app


# ============================================================================
# 5.  ENTRY POINT
# ============================================================================

# ── Attempt to load the model at import time ────────────────────────────────
# If the placeholder is still present, we set model = None and let
# detect_damage() raise a user-friendly Gradio error at request time (rather
# than crashing the whole server at startup).
model = None
try:
    model = load_model(MODEL_PATH)
    print(f"✅  Model loaded from {MODEL_PATH}")
except FileNotFoundError as exc:
    # Print the warning but don't crash — the Gradio UI will still launch
    # so the user can see the interface and read the error message.
    print(f"⚠️  {exc}")
    print("   The demo will start, but inference will fail until you provide "
          "the real best.pt.\n")


if __name__ == "__main__":
    # Build and launch the Gradio app.
    # share=False  → local only (set True to get a public Gradio link).
    # server_name="0.0.0.0" → accept connections from other machines on the
    #   local network (useful during the viva if the examiner's laptop is
    #   different from the one running the demo).
    app = build_gradio_app()
    app.launch(
        server_name="0.0.0.0",   # accessible on the local network
        server_port=7860,        # Gradio's default port
        share=False,             # set True for a public tunnel
    )
