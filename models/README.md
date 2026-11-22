# Model Card: UrbanRoadDamage Detector (YOLOv8s Baseline)
### Project 23 | Deep Learning - Semester V

## 1. Model Details
- **Architecture**: YOLOv8s (Ultralytics Small variant) & RetinaNet ResNet-50 FPN (PRD Decision D1).
- **Pretrained Weights**: COCO transfer learning checkpoint (`yolov8s.pt`).
- **Input Resolution**: $640 \times 640 \times 3$ (letterbox padding).
- **Output Classes**: 4 classes (0: D00, 1: D10, 2: D20, 3: D40).
- **Inference Runtime**: Python (Ultralytics / PyTorch) and native C++ (ONNX Runtime).

---

## 2. Training Hyperparameters (Google Colab T4 GPU)
- **Framework**: Ultralytics YOLOv8.
- **Optimizer**: SGD (`lr0 = 0.01`, `momentum = 0.937`).
- **Batch Size**: 16 (or 8 if GPU memory constrained).
- **Epochs**: 30 (with early stopping `patience = 10`).
- **Loss Formulation**:
  - Box Loss: Complete IoU (CIoU) / Distribution Focal Loss (DFL).
  - Class Loss: Binary Cross-Entropy with focal scaling for extreme background-foreground imbalance.
- **Transfer Learning**: Pretrained COCO backbone features fine-tuned on RDD2022 India subset.

---

## 3. Evaluation Metrics on Held-out Test Split
> [!IMPORTANT]
> Real metrics only. These numbers will be transcribed directly from Google Colab Cell 9 output without modification.

| Metric | Target / Provisional | Actual Test Result |
|--------|----------------------|--------------------|
| **Overall mAP@0.5** | Baseline target | `TODO: fill from Colab Cell 9` |
| **Overall mAP@0.5:0.95** | High-precision overlap | `TODO: fill from Colab Cell 9` |
| **Precision (mp)** | High detection certainty | `TODO: fill from Colab Cell 9` |
| **Recall (mr)** | Comprehensive defect capture | `TODO: fill from Colab Cell 9` |
| **D00 (Longitudinal Crack) AP@0.5** | Class AP | `TODO: fill from Colab Cell 9` |
| **D10 (Transverse Crack) AP@0.5** | Class AP | `TODO: fill from Colab Cell 9` |
| **D20 (Alligator Crack) AP@0.5** | Class AP | `TODO: fill from Colab Cell 9` |
| **D40 (Pothole) AP@0.5** | Class AP (safety critical) | `TODO: fill from Colab Cell 9` |
| **Inference Speed** | Latency benchmark | `TODO: fill from Colab Cell 9` |

---

## 4. Intended Use & Limitations
- **Intended Use**: Real-time road asset inventory and automated maintenance priority triage for municipal public works departments.
- **Failure Modes & Robustness**: Tested against adverse weather perturbations (Gaussian blur, illumination variations, camera vibrations).
- **Deployment Recommendation (PRD Section 14)**: GO / CONDITIONAL GO / NO-GO criteria evaluated on held-out test split.
