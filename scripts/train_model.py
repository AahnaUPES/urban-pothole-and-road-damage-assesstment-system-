"""
===============================================================================
UrbanRoadDamage — Model Training Pipeline (Fine-Tuning on Expanded Dataset)
Script: scripts/train_model.py
===============================================================================

PURPOSE:
Fine-tunes YOLOv8n detector on the expanded dataset (55 images: 44 train, 11 val)
incorporating real-world user road and pothole imagery alongside RDD2022 samples.
Saves best model weights to models/best.pt and logs real performance metrics.
===============================================================================
"""

import json
import shutil
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = REPO_ROOT / "dataset" / "mini_rdd" / "mini_rdd.yaml"
OUTPUT_DIR = REPO_ROOT / "runs" / "rdd_enhanced"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"


def run_training():
    print("=" * 70)
    print("URBAN ROAD DAMAGE — TRAINING YOLOv8 ON ENHANCED DATASET")
    print(f"Dataset config   : {YAML_PATH}")
    print("Base weights     : yolov8n.pt")
    print("Total images     : 55 (44 train, 11 val)")
    print("Classes          : 0: D00, 1: D10, 2: D20, 3: D40 (Pothole)")
    print("=" * 70)

    # 1. Initialize YOLOv8n
    model = YOLO("yolov8n.pt")

    # 2. Train on local CPU for 8 epochs
    # imgsz=416 gives fast CPU inference and clean feature maps
    results = model.train(
        data=str(YAML_PATH),
        epochs=8,
        imgsz=416,
        batch=8,
        device="cpu",
        project=str(OUTPUT_DIR),
        name="train_run",
        exist_ok=True,
        verbose=True,
        seed=42,
        augment=True,
        flipud=0.0,
        fliplr=0.5,
    )

    # 3. Locate best weights
    weights_dir = OUTPUT_DIR / "train_run" / "weights"
    best_pt = weights_dir / "best.pt"
    if not best_pt.exists():
        best_pt = weights_dir / "last.pt"

    dest_pt = MODELS_DIR / "best.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_pt, dest_pt)
    print(f"\n✅ Training complete! Saved trained model weights to:\n   {dest_pt}")
    print(f"   Model file size: {dest_pt.stat().st_size / (1024*1024):.2f} MB")

    # 4. Rigorous validation on val split (real metrics only)
    print("\n" + "=" * 70)
    print("VALIDATION EVALUATION (REAL METRICS)")
    print("=" * 70)
    val_model = YOLO(str(dest_pt))
    val_metrics = val_model.val(data=str(YAML_PATH), imgsz=416, split="val")

    map50 = float(val_metrics.box.map50)
    map_all = float(val_metrics.box.map)
    precision = float(val_metrics.box.mp)
    recall = float(val_metrics.box.mr)

    metrics_summary = {
        "dataset_size": 55,
        "train_size": 44,
        "val_size": 11,
        "epochs": 8,
        "imgsz": 416,
        "mAP50": round(map50, 4),
        "mAP50_95": round(map_all, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = RESULTS_DIR / "metrics_summary.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\n📊 Evaluated Metrics Summary:")
    print(f"   mAP@0.5      : {map50:.4f}")
    print(f"   mAP@0.5:0.95 : {map_all:.4f}")
    print(f"   Precision    : {precision:.4f}")
    print(f"   Recall       : {recall:.4f}")
    print(f"   Saved to     : {metrics_file}")

    # 5. Export to ONNX if supported
    try:
        print("\nAttempting ONNX export...")
        exported = val_model.export(format="onnx", imgsz=416, opset=12)
        dest_onnx = MODELS_DIR / "best.onnx"
        shutil.copy(exported, dest_onnx)
        print(f"✅ ONNX model successfully saved to: {dest_onnx}")
    except Exception as e:
        print(f"ℹ️ ONNX export skipped ({e}). PyTorch model best.pt is fully functional.")

    return metrics_summary


if __name__ == "__main__":
    run_training()
