"""
===============================================================================
UrbanRoadDamage — ONNX Export Module
Script: scripts/export_onnx.py
===============================================================================

PURPOSE & VIVA EXPLANATION
--------------------------
1. What is ONNX?
   ONNX (Open Neural Network Exchange) is an open, cross-platform standard
   format for representing deep learning models. It defines a common set of
   operators and a standard file format (.onnx).

2. Why export to ONNX for C++ Deployment? (PRD Section 15 & 20.1 D2)
   - PyTorch in Python is heavyweight, memory-intensive, and introduces Python
     Global Interpreter Lock (GIL) overhead.
   - In production municipal road assessment systems (e.g. edge cameras or mobile
     survey vehicles), inference must run efficiently in native C++ using
     ONNX Runtime (or OpenCV DNN).
   - Exporting the trained weights (models/best.pt) to ONNX decouples training
     (Python/Colab) from deployment (C++ CLI on edge CPU/GPU).

3. Export Settings:
   - Input shape: [1, 3, 640, 640] (Batch, Channels, Height, Width)
   - Dynamic axes: Batch dimension is dynamic (allows processing single images or batches)
   - Opset version: 12 (standard compatibility across ONNX Runtime C++ releases)

HONESTY NOTICE:
--------------
Do not run training here. Copy the trained `best.pt` from Google Colab into `models/`.
===============================================================================
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PT_PATH = REPO_ROOT / "models" / "best.pt"
MODEL_ONNX_PATH = REPO_ROOT / "models" / "best.onnx"


def export_model_to_onnx():
    print("=" * 70)
    print("URBAN ROAD DAMAGE — ONNX EXPORT UTILITY (scripts/export_onnx.py)")
    print("Target: models/best.pt -> models/best.onnx")
    print("=" * 70)

    # 1. Guard check: File exists
    if not MODEL_PT_PATH.exists():
        print(f"❌ Error: Model weights not found at: {MODEL_PT_PATH}")
        print("👉 Instructions:")
        print("   1. Run Google Colab notebook (Appendix A of PRD).")
        print("   2. Download best.pt from Drive.")
        print("   3. Place it in UrbanRoadDamage/models/best.pt")
        print("   4. Re-run: python scripts/export_onnx.py")
        sys.exit(1)

    # 2. Guard check: Not placeholder
    if MODEL_PT_PATH.stat().st_size < 1024:
        print(f"❌ Error: '{MODEL_PT_PATH}' is a placeholder file.")
        with open(MODEL_PT_PATH, "r", encoding="utf-8") as f:
            print(f"   Contents: {f.read().strip()}")
        print("\n👉 Replace this placeholder with the actual weights from Colab.")
        sys.exit(1)

    # 3. Perform Export using Ultralytics YOLO
    print(f"Found model weights ({MODEL_PT_PATH.stat().st_size / (1024*1024):.2f} MB).")
    print("Loading model and initiating ONNX export...")

    try:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PT_PATH))

        # Export with imgsz=640, opset=12, dynamic=True
        exported_path = model.export(
            format="onnx",
            imgsz=640,
            opset=12,
            dynamic=False,  # Fixed 1x3x640x640 input for predictable C++ tensor allocation
            simplify=True,
        )
        print(f"✅ Export successful!")
        print(f"   ONNX model generated at: {exported_path}")
        print("   Ready for C++ CLI inference via ONNX Runtime.")

    except ImportError:
        print("⚠️ Ultralytics is not installed locally. Run: pip install ultralytics")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    export_model_to_onnx()
