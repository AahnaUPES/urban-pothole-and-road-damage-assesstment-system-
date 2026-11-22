"""
===============================================================================
UrbanRoadDamage — Mini Dataset Generator for MVP Prototype
Script: scripts/prepare_mini_dataset.py
===============================================================================

PURPOSE:
Generates 40 synthetic road damage images with ground-truth YOLO labels across
all 4 official classes (D00, D10, D20, D40) and negative clean roads to allow
training a working MVP model locally for demonstration purposes.

Layout:
  dataset/mini_rdd/
  ├── images/train (32 images)
  ├── images/val   (8 images)
  ├── labels/train (32 txt files)
  ├── labels/val   (8 txt files)
  └── mini_rdd.yaml
===============================================================================
"""

import os
from pathlib import Path
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = REPO_ROOT / "dataset" / "mini_rdd"


def create_base_asphalt(w=640, h=640):
    """Creates realistic textured asphalt with pavement noise and lane markings."""
    base_color = np.random.randint(55, 80)
    img = np.full((h, w, 3), (base_color, base_color, base_color + 4), dtype=np.uint8)
    # High-frequency gravel noise
    noise = np.random.randint(-22, 22, (h, w, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Slight lighting gradient (vignetting / road crown)
    y_coords, x_coords = np.ogrid[:h, :w]
    gradient = 1.0 - 0.2 * ((x_coords - w/2)**2 + (y_coords - h/2)**2) / ((w/2)**2 + (h/2)**2)
    gradient = np.clip(gradient, 0.75, 1.0)[:, :, np.newaxis]
    img = np.clip(img * gradient, 0, 255).astype(np.uint8)
    return img


def draw_pothole(img, center, size):
    """Draws a bowl-shaped localized cavity (D40)."""
    h, w = img.shape[:2]
    cx, cy = center
    rx, ry = size
    # Outer dark shadow
    cv2.ellipse(img, (cx, cy), (rx, ry), np.random.randint(-20, 20), 0, 360, (20, 20, 25), -1)
    # Inner deeper depression
    cv2.ellipse(img, (cx, cy), (int(rx * 0.75), int(ry * 0.75)), np.random.randint(-20, 20), 0, 360, (10, 10, 15), -1)
    # Chipped perimeter
    for _ in range(25):
        px = cx + np.random.randint(-rx, rx)
        py = cy + np.random.randint(-ry, ry)
        cv2.circle(img, (px, py), np.random.randint(3, 10), (35, 35, 40), -1)

    # Box coordinates
    x1 = max(0, cx - rx - 5)
    y1 = max(0, cy - ry - 5)
    x2 = min(w, cx + rx + 5)
    y2 = min(h, cy + ry + 5)
    return 3, (x1, y1, x2, y2)  # Class 3: D40


def draw_alligator_crack(img, top_left, extent):
    """Draws interconnected polygonal fatigue cracks (D20)."""
    h, w = img.shape[:2]
    x0, y0 = top_left
    bw, bh = extent
    # Draw intersecting mesh of cracks
    for _ in range(12):
        pt1 = (x0 + np.random.randint(0, bw), y0 + np.random.randint(0, bh))
        pt2 = (x0 + np.random.randint(0, bw), y0 + np.random.randint(0, bh))
        cv2.line(img, pt1, pt2, (25, 25, 30), np.random.randint(2, 4))
    for _ in range(8):
        pt1 = (x0 + np.random.randint(0, bw), y0 + np.random.randint(0, bh))
        pt2 = (pt1[0] + np.random.randint(-30, 30), pt1[1] + np.random.randint(-30, 30))
        cv2.line(img, pt1, pt2, (30, 30, 35), 2)

    return 2, (x0, y0, min(w, x0 + bw), min(h, y0 + bh))  # Class 2: D20


def draw_longitudinal_crack(img):
    """Draws crack running parallel to travel direction (D00)."""
    h, w = img.shape[:2]
    start_x = np.random.randint(int(w * 0.2), int(w * 0.8))
    pts = []
    curr_x = start_x
    for y in range(50, h - 50, 40):
        curr_x += np.random.randint(-15, 15)
        curr_x = max(20, min(w - 20, curr_x))
        pts.append([curr_x, y])
    pts = np.array(pts, np.int32)
    cv2.polylines(img, [pts], False, (22, 22, 26), np.random.randint(2, 5))
    x_min, y_min = pts[:, 0].min() - 8, pts[:, 1].min() - 8
    x_max, y_max = pts[:, 0].max() + 8, pts[:, 1].max() + 8
    return 0, (max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max))  # Class 0: D00


def draw_transverse_crack(img):
    """Draws crack running perpendicular across the road (D10)."""
    h, w = img.shape[:2]
    start_y = np.random.randint(int(h * 0.25), int(h * 0.75))
    pts = []
    curr_y = start_y
    for x in range(40, w - 40, 40):
        curr_y += np.random.randint(-12, 12)
        curr_y = max(20, min(h - 20, curr_y))
        pts.append([x, curr_y])
    pts = np.array(pts, np.int32)
    cv2.polylines(img, [pts], False, (20, 20, 24), np.random.randint(2, 5))
    x_min, y_min = pts[:, 0].min() - 8, pts[:, 1].min() - 8
    x_max, y_max = pts[:, 0].max() + 8, pts[:, 1].max() + 8
    return 1, (max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max))  # Class 1: D10


def generate_dataset(num_samples=40):
    np.random.seed(42)
    train_count = 32
    val_count = 8

    for split in ["train", "val"]:
        os.makedirs(BASE_DIR / "images" / split, exist_ok=True)
        os.makedirs(BASE_DIR / "labels" / split, exist_ok=True)

    splits = (["train"] * train_count) + (["val"] * val_count)

    for i, split in enumerate(splits):
        img = create_base_asphalt(640, 640)
        h, w = 640, 640
        labels = []

        defect_type = i % 5  # 0: D00, 1: D10, 2: D20, 3: D40, 4: Mixed/Clean

        if defect_type == 0:
            c, box = draw_longitudinal_crack(img)
            labels.append((c, box))
        elif defect_type == 1:
            c, box = draw_transverse_crack(img)
            labels.append((c, box))
        elif defect_type == 2:
            x0 = np.random.randint(100, 350)
            y0 = np.random.randint(100, 350)
            c, box = draw_alligator_crack(img, (x0, y0), (200, 160))
            labels.append((c, box))
        elif defect_type == 3:
            cx = np.random.randint(200, 440)
            cy = np.random.randint(200, 440)
            rx = np.random.randint(45, 95)
            ry = np.random.randint(35, 75)
            c, box = draw_pothole(img, (cx, cy), (rx, ry))
            labels.append((c, box))
        else:
            # Multi-defect or clean
            if i % 2 == 0:
                c1, b1 = draw_pothole(img, (220, 260), (50, 40))
                c2, b2 = draw_transverse_crack(img)
                labels.extend([(c1, b1), (c2, b2)])
            # else: clean road (negative sample)

        img_filename = f"img_{i:03d}.jpg"
        label_filename = f"img_{i:03d}.txt"

        cv2.imwrite(str(BASE_DIR / "images" / split / img_filename), img)

        yolo_lines = []
        for c, (x1, y1, x2, y2) in labels:
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            yolo_lines.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(BASE_DIR / "labels" / split / label_filename, "w") as f:
            f.write("\n".join(yolo_lines))

    # Write mini_rdd.yaml
    yaml_content = f"""path: {BASE_DIR.as_posix()}
train: images/train
val: images/val
names:
  0: D00
  1: D10
  2: D20
  3: D40
"""
    with open(BASE_DIR / "mini_rdd.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Generated {num_samples} images with YOLO annotations in {BASE_DIR}")


if __name__ == "__main__":
    generate_dataset(40)
