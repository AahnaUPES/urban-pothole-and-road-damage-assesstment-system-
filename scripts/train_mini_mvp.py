"""
===============================================================================
UrbanRoadDamage — Local MVP Training Script
Script: scripts/train_mini_mvp.py
===============================================================================

PURPOSE:
Trains a YOLOv8 nano detector locally on CPU using the mini-RDD dataset (40 images)
for 5 epochs as an MVP prototype for immediate evaluation and demonstration.
After training, it automatically updates models/best.pt and exports models/best.onnx.
===============================================================================
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = REPO_ROOT / "dataset" / "mini_rdd" / "mini_rdd.yaml"
OUTPUT_DIR = REPO_ROOT / "runs" / "mini_mvp"
MODELS_DIR = REPO_ROOT / "models"


def train_mini_mvp():
    print("=" * 70)
    print("URBAN ROAD DAMAGE — LOCAL MVP TRAINING (CPU)")
    print(f"Dataset config : {YAML_PATH}")
    print("Architecture   : YOLOv8n (Transfer Learning from COCO)")
    print("Epochs         : 5 (Fast local convergence)")
    print("=" * 70)

    # 1. Initialize YOLOv8n (downloads ~6MB weights if not cached)
    model = YOLO("yolov8n.pt")

    # 2. Train on CPU (imgsz=416 for ultra-fast local CPU training, batch=8, 5 epochs)
    results = model.train(
        data=str(YAML_PATH),
        epochs=5,
        imgsz=416,
        batch=8,
        device="cpu",
        project=str(OUTPUT_DIR),
        name="train_run",
        exist_ok=True,
        verbose=True,
        seed=42,
    )

    # 3. Copy trained best.pt into models/
    trained_best = OUTPUT_DIR / "train_run" / "weights" / "best.pt"
    if not trained_best.exists():
        # Fallback to last.pt if best.pt is not created
        trained_best = OUTPUT_DIR / "train_run" / "weights" / "last.pt"

    dest_best = MODELS_DIR / "best.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(trained_best, dest_best)
    print(f"\n✅ Training complete! Saved trained weights to: {dest_best}")
    print(f"   Model file size: {dest_best.stat().st_size / (1024*1024):.2f} MB")

    # 4. Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = model.val(data=str(YAML_PATH), imgsz=640, split="val")
    print(f"Validation mAP@0.5      : {val_metrics.box.map50:.4f}")
    print(f"Validation mAP@0.5:0.95 : {val_metrics.box.map:.4f}")
    print(f"Validation Precision    : {val_metrics.box.mp:.4f}")
    print(f"Validation Recall       : {val_metrics.box.mr:.4f}")

    # 5. Export to ONNX
    print("\nExporting trained model to ONNX...")
    exported_onnx = model.export(format="onnx", imgsz=640, opset=12, simplify=True)
    dest_onnx = MODELS_DIR / "best.onnx"
    shutil.copy(exported_onnx, dest_onnx)
    print(f"✅ ONNX model ready at: {dest_onnx}")


if __name__ == "__main__":
    train_mini_mvp()
