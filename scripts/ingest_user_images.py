"""
===============================================================================
UrbanRoadDamage — Ingest and Annotate User Uploaded Road Images
Script: scripts/ingest_user_images.py
===============================================================================

PURPOSE:
Ingests 15 real-world road and pothole images uploaded by the user:
- 4 severe pothole images annotated with RDD2022 Class 3 (D40: Pothole)
- 11 real-world Indian road perspective images (negative clean road backgrounds
  and surface distress) to train the model to suppress false positives on vehicles,
  trees, road shoulders, and clean asphalt.
===============================================================================
"""

import shutil
import cv2
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UPLOADED_DIR = Path(r"C:\Users\AAHNA\.gemini\antigravity\brain\f59f8550-8410-48cf-bd69-f66ebecdbda4\.user_uploaded")
DATASET_DIR = REPO_ROOT / "dataset" / "mini_rdd"
SAMPLE_INPUTS_DIR = REPO_ROOT / "sample_inputs"

TRAIN_IMG = DATASET_DIR / "images" / "train"
TRAIN_LBL = DATASET_DIR / "labels" / "train"
VAL_IMG = DATASET_DIR / "images" / "val"
VAL_LBL = DATASET_DIR / "labels" / "val"

# Define metadata and annotations for user images
USER_IMAGES = {
    # 4 Pothole images (D40 = class 3)
    "user_pothole_01.jpg": {
        "src": "media_1795383874327_1ce7b209.jpg",
        "split": "train",
        "sample": True,
        "labels": [
            "3 0.227 0.686 0.411 0.395",  # Large water-filled pothole bottom-left
            "3 0.911 0.714 0.177 0.188",  # Pothole puddle middle-right
            "3 0.480 0.301 0.137 0.075",  # Center pothole
        ]
    },
    "user_pothole_02.png": {
        "src": "media_1795383894186_fe3566e9.png",
        "split": "train",
        "sample": True,
        "labels": [
            "3 0.415 0.652 0.489 0.364",  # Severe deep pothole
            "3 0.319 0.409 0.255 0.136",  # Surface depression
        ]
    },
    "user_pothole_03.png": {
        "src": "media_1795383907538_71741348.png",
        "split": "val",  # Put in validation split to evaluate real pothole detection!
        "sample": True,
        "labels": [
            "3 0.596 0.577 0.426 0.410",  # Large pothole by curb
        ]
    },
    "user_pothole_04.png": {
        "src": "media_1795383922004_82a68e16.png",
        "split": "train",
        "sample": True,
        "labels": [
            "3 0.511 0.643 0.298 0.434",  # Puddle/pothole depression
        ]
    },
    # 11 Indian road dashcam images (highways, arterial roads, clean asphalt, surface distress)
    "user_road_01.jpg": {
        "src": "media_1795383959036_14d77865.jpg",
        "split": "train",
        "sample": False,
        "labels": []  # Clean highway road (negative sample)
    },
    "user_road_02.jpg": {
        "src": "media_1795383959042_97a22e74.jpg",
        "split": "train",
        "sample": False,
        "labels": []  # Urban road with traffic (negative sample)
    },
    "user_road_03.jpg": {
        "src": "media_1795383959044_9d09dbf3.jpg",
        "split": "val",
        "sample": False,
        "labels": []  # Curved road with divider (negative val sample)
    },
    "user_road_04.jpg": {
        "src": "media_1795383959062_864cee39.jpg",
        "split": "train",
        "sample": False,
        "labels": []  # Industrial road (negative sample)
    },
    "user_road_05.jpg": {
        "src": "media_1795383959063_a3879763.jpg",
        "split": "train",
        "sample": False,
        "labels": [
            "0 0.500 0.650 0.250 0.500"  # Longitudinal distress (D00 = 0)
        ]
    },
    "user_road_06.jpg": {
        "src": "media_1795383959065_18f3150a.jpg",
        "split": "train",
        "sample": True,  # Good clean asphalt sample for demo
        "labels": []  # Smooth paved asphalt (negative sample)
    },
    "user_road_07.jpg": {
        "src": "media_1795383959068_f9b32e5e.jpg",
        "split": "train",
        "sample": False,
        "labels": []  # Commercial roadside (negative sample)
    },
    "user_road_08.jpg": {
        "src": "media_1795383959069_f303cb2e.jpg",
        "split": "train",
        "sample": False,
        "labels": []  # Highway with scooters (negative sample)
    },
    "user_road_09.jpg": {
        "src": "media_1795383959071_a5bd0672.jpg",
        "split": "val",
        "sample": False,
        "labels": []  # Curved urban road with billboard (negative val sample)
    },
    "user_road_10.jpg": {
        "src": "media_1795383959072_9765c39f.jpg",
        "split": "train",
        "sample": False,
        "labels": []  # Wide arterial road (negative sample)
    },
    "user_road_11.jpg": {
        "src": "media_1795383959073_bb4090b0.jpg",
        "split": "train",
        "sample": False,
        "labels": []  # Urban road with power lines (negative sample)
    },
}


def ingest_images():
    print("=" * 70)
    print("INGESTING AND ANNOTATING USER ROAD IMAGES")
    print("=" * 70)

    SAMPLE_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    debug_dir = REPO_ROOT / "runs" / "annotation_check"
    debug_dir.mkdir(parents=True, exist_ok=True)

    ingested_count = 0
    for dst_name, meta in USER_IMAGES.items():
        src_path = UPLOADED_DIR / meta["src"]
        if not src_path.exists():
            print(f"⚠️ Source missing: {src_path}")
            continue

        split = meta["split"]
        img_dest = (TRAIN_IMG if split == "train" else VAL_IMG) / dst_name
        lbl_dest = (TRAIN_LBL if split == "train" else VAL_LBL) / (Path(dst_name).stem + ".txt")

        # Copy image
        shutil.copy(src_path, img_dest)

        # Write label file
        label_content = "\n".join(meta["labels"]) + ("\n" if meta["labels"] else "")
        lbl_dest.write_text(label_content, encoding="utf-8")

        # Copy to sample inputs if marked
        if meta.get("sample", False):
            shutil.copy(src_path, SAMPLE_INPUTS_DIR / dst_name)

        # Generate visual check for annotated boxes
        if meta["labels"]:
            img_bgr = cv2.imread(str(src_path))
            h, w = img_bgr.shape[:2]
            for line in meta["labels"]:
                cls_id, cx, cy, bw, bh = [float(x) for x in line.split()]
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cls_name = "D40 Pothole" if int(cls_id) == 3 else "D00 Crack"
                cv2.putText(img_bgr, cls_name, (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(str(debug_dir / f"check_{dst_name}.jpg"), img_bgr)

        ingested_count += 1
        print(f"✓ Ingested [{split.upper()}]: {dst_name} ({len(meta['labels'])} labels)")

    print(f"\n✅ Successfully ingested {ingested_count} user images into dataset!")
    print(f"Train images: {len(list(TRAIN_IMG.glob('*.*')))}")
    print(f"Val images  : {len(list(VAL_IMG.glob('*.*')))}")


if __name__ == "__main__":
    ingest_images()
