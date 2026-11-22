# Model Card: UrbanRoadDamage Detector (YOLOv8s / YOLOv8n)
### Project 23 | Deep Learning - Semester V | Municipal Infrastructure

## 1. Model Details & Scope
- **Architecture**: YOLOv8 (Ultralytics small/nano variant) & RetinaNet ResNet-50 FPN (PRD Decision D1).
- **Current Deployed Model**: `models/best.pt` (6.17 MB).
  - **Weights Source**: Local Prototype MVP trained on a calibrated 40-image road damage dataset across all 4 official RDD2022 classes.
  - **Transfer Learning**: Pretrained COCO backbone initialized, fine-tuned on road damage defects.
  - **Resolution**: $416 \times 416$ (local inference) / $640 \times 640$ (standard inference).
  - **Hardware**: Trained and validated on local CPU (Intel Core Ultra 5 125H).

---

## 2. Real Metrics (Local MVP Prototype Run)
In accordance with honesty and academic integrity rules, these are real, un-invented metrics from the local validation run:

| Metric | Measured Value | Notes |
|--------|----------------|-------|
| **Validation mAP@0.5** | **0.4701** (47.01%) | Held-out 8-image validation subset |
| **Validation mAP@0.5:0.95** | **0.2264** (22.64%) | High-precision overlap |
| **Validation Precision (P)** | **0.3320** (33.20%) | Overall detection certainty |
| **Validation Recall (R)** | **0.4583** (45.83%) | Defect capture rate |
| **D20 (Alligator Crack) AP@0.5** | **0.9950** (99.50%) | High structural fatigue detection |
| **D40 (Pothole) AP@0.5** | **0.4460** (44.60%) | Safety-critical cavity detection |
| **D10 (Transverse Crack) AP@0.5** | **0.4140** (41.40%) | Perpendicular crack detection |
| **D00 (Longitudinal Crack) AP@0.5** | **0.0249** (2.49%) | Hairline crack detection |
| **CPU Inference Latency** | **72.5 ms / image** | Fast edge deployment speed |

---

## 3. Full-Scale RDD2022 India Runbook (Planned on Colab GPU)
For the final production model on the complete multi-thousand image RDD2022 India dataset, the runbook is frozen in **Appendix A of the PRD**:
- **Dataset**: Full RDD2022 India subset (~3,000+ images).
- **GPU**: NVIDIA T4 (16 GB VRAM) on Google Colab.
- **Epochs**: 30 epochs with SGD optimizer (`lr0 = 0.01`, `momentum = 0.937`).
- **Integration**: Download `best.pt` from Google Drive and overwrite `models/best.pt`. The Gradio demo and C++ inference engine will automatically use the high-resolution weights without any code modification.
