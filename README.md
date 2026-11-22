# Urban Pothole and Road Damage Assessment System
### Project 23 | Deep Learning - Semester V | Municipal Infrastructure

[![Status: In-Progress](https://img.shields.io/badge/Status-Core%20Complete%20%7C%20Awaiting%20Colab%20Weights-orange)](#implementation-status)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Evaluation-blue)](#)

> [!IMPORTANT]
> **COMPUTE & TRAINING POLICY**:
> Detector training is strictly executed in **Google Colab (GPU)** using the reproducible notebook in PRD Appendix A.
> **Do not train the detector on this laptop.** `models/best.pt` is supplied from Colab; all local development builds the inference, severity rules, fundamentals, and evaluation harness around it.

---

## 1. Problem Statement & Core Operational Question
Traditional municipal road inspection is manual, labor-intensive, hazardous, and difficult to scale across urban road networks. This system addresses the core municipal infrastructure question:

> **"Where is the road damaged, what type of damage is present, how severe is it, and how urgently should it be repaired?"**

---

## 2. Repository Layout (PRD Section 17)

```
UrbanRoadDamage/
|-- CMakeLists.txt              # CMake configuration for native C++ CLI
|-- README.md                   # Project overview, setup, and viva guide
|-- requirements.txt            # Pinned Python package dependencies
|-- .gitignore                  # Git tracking exclusion rules
|-- configs/
|   |-- config.yaml             # Main runtime thresholds (IoU, confidence, paths)
|   `-- severity_config.yaml    # PRD §20.2 rule-based severity & priority thresholds
|-- scripts/
|   |-- demo_gradio.py          # Interactive web UI (loads models/best.pt)
|   |-- severity.py             # Rule-based severity and maintenance priority module
|   |-- test_severity.py        # Automated unit tests (10/10 passed)
|   `-- export_onnx.py          # Script exporting PyTorch weights to ONNX
|-- src/
|   |-- main.cpp                # Native C++ CLI entry point (--image, --dir, --video)
|   |-- detection/              # ONNX Runtime C++ detector, letterbox, NMS
|   |-- severity/               # C++ severity & priority rule evaluator
|   `-- export/                 # Per-image JSON and run-level CSV exporters
|-- fundamentals/
|   |-- backprop/               # Numerical gradient checking vs analytical gradients
|   |-- mlp/                    # Scratch NumPy MLP on non-linear 2D data
|   |-- optimizers/             # Batch GD, SGD, Momentum, Nesterov comparison
|   |-- tensor/                 # Tensor math, broadcasting, SVD low-rank approximation
|   `-- neurons/                # McCulloch-Pitts, Perceptron, Sigmoid/Tanh/ReLU
|-- learning_rules/
|   `-- learning_rules_demo.py  # Hebbian, Oja's PCA, Competitive, Delta, Energy rules
|-- models/
|   |-- README.md               # Model card (YOLOv8s / RetinaNet specifications)
|   `-- best.pt                 # Trained checkpoint (supplied from Colab)
|-- dataset/
|   `-- README.md               # Dataset card (RDD2022 India subset specifications)
|-- docs/
|   `-- viva_prep.md            # Viva examination defense notes & mathematical derivations
|-- tests/
|   `-- test_severity.cpp       # C++ unit test suite
`-- results/                    # Saved convergence curves, decision boundaries, plots
    |-- gradient_check.png
    |-- mlp_decision_boundary.png
    |-- optimizer_comparison.png
    |-- learning_rules.png
    |-- svd_approximation.png
    `-- activation_functions.png
```

---

## 3. Implementation Status (PRD Section 20 Scope)

| Component | Scope Tier | Status | Notes |
|-----------|------------|--------|-------|
| **RDD2022 India Dataset Preparation** | MVP | **PLANNED (Colab)** | Runbook defined in Colab Appendix A; split manifests generated there |
| **Detector Training (best.pt)** | MVP | **PLANNED (Colab)** | Ultralytics YOLOv8s fine-tuning with SGD on T4 GPU |
| **Gradio Web Demo** | MVP | **DONE** | Complete in `scripts/demo_gradio.py`; loads `models/best.pt` |
| **Rule-Based Severity & Priority (§20.2)** | MVP | **DONE** | Complete in `scripts/severity.py` & `src/severity/`; 10/10 tests pass |
| **Numerical Gradient Checking** | Fundamentals | **DONE** | Complete in `fundamentals/backprop/`; max error < 3.08e-09 |
| **Scratch MLP on Non-Linear Data** | Fundamentals | **DONE** | Complete in `fundamentals/mlp/`; 100% accuracy on concentric circles |
| **Optimizer Comparison (BGD/SGD/Mom/NAG)** | Fundamentals | **DONE** | Complete in `fundamentals/optimizers/`; convergence curves plotted |
| **Classical Learning Rules (5 rules)** | Academic | **DONE** | Complete in `learning_rules/`; Hebbian, Oja, WTA, Delta, Hopfield |
| **Tensor Math & SVD Approximation** | Fundamentals | **DONE** | Complete in `fundamentals/tensor/`; Eckart-Young rank reduction |
| **Neuron & Activation Dynamics** | Fundamentals | **DONE** | Complete in `fundamentals/neurons/`; step, sigmoid, tanh, relu |
| **ONNX Model Export Utility** | MVP | **DONE** | Complete in `scripts/export_onnx.py` |
| **C++ CLI Inference Application** | MVP | **DONE** | Complete in `src/main.cpp` with CMake configuration |
| **Segmentation (IoU)** | Tier 2 | **PLANNED** | Dependent on decision gate M3 |
| **GPS Geotag Error Measurement** | Tier 2 | **PLANNED** | Requires self-collected ground-truth GPS |
| **LSTM Road Deterioration** | Tier 3 | **PLANNED** | Conceptual simulation |

---

## 4. Honesty Rules & Real Metrics Policy

> [!WARNING]
> In accordance with academic integrity rules, **no performance metrics are fabricated or placeholder-guessed**.
> Test metrics must be filled directly from the held-out test evaluation in Colab (Cell 9).

### Official Metrics Table
| Metric | Value | Source |
|--------|-------|--------|
| **mAP@0.5 (Overall)** | `TODO: fill from Colab Cell 9` | Colab test evaluation run |
| **mAP@0.5:0.95 (Overall)** | `TODO: fill from Colab Cell 9` | Colab test evaluation run |
| **Precision** | `TODO: fill from Colab Cell 9` | Colab test evaluation run |
| **Recall** | `TODO: fill from Colab Cell 9` | Colab test evaluation run |
| **D00 AP@0.5 (Longitudinal Crack)** | `TODO: fill from Colab Cell 9` | Colab test evaluation run |
| **D10 AP@0.5 (Transverse Crack)** | `TODO: fill from Colab Cell 9` | Colab test evaluation run |
| **D20 AP@0.5 (Alligator Crack)** | `TODO: fill from Colab Cell 9` | Colab test evaluation run |
| **D40 AP@0.5 (Pothole)** | `TODO: fill from Colab Cell 9` | Colab test evaluation run |
| **Inference Speed (ms/img)** | `TODO: fill from Colab Cell 9` | Colab test evaluation run |
| **Gradient Check Max Relative Error** | **3.08e-09** (W1) | Laptop run (`fundamentals/backprop/`) |
| **Scratch MLP Final Accuracy** | **100.00%** | Laptop run (`fundamentals/mlp/`) |
| **Severity Unit Tests** | **10 / 10 Passed** | Laptop run (`scripts/test_severity.py`) |

---

## 5. How to Run

### A. Environment Setup
```powershell
# 1. Clone repository and navigate
cd UrbanRoadDamage

# 2. Install Python dependencies
python -m pip install -r requirements.txt
```

### B. Run Interactive Gradio Web Demo
```powershell
python scripts/demo_gradio.py
# Opens browser interface at http://127.0.0.1:7860
```

### C. Run Severity Unit Tests
```powershell
python -m pytest scripts/test_severity.py -v
```

### D. Run Neural Network Fundamentals Demos
```powershell
# 1. Numerical Gradient Checking vs Analytical Backpropagation
python fundamentals/backprop/gradient_check.py

# 2. Scratch Multilayer Perceptron on 2D concentric circles
python fundamentals/mlp/mlp_demo.py

# 3. Optimizer Comparison (BGD vs SGD vs Momentum vs Nesterov)
python fundamentals/optimizers/optimizer_comparison.py

# 4. Classical Learning Rules (Hebbian, Oja, WTA, Delta, Energy)
python learning_rules/learning_rules_demo.py

# 5. Tensor Mathematics & SVD Low-Rank Approximation
python fundamentals/tensor/tensor_math.py

# 6. Artificial Neurons & Activation Dynamics
python fundamentals/neurons/mcculloch_pitts.py
```

### E. Export ONNX Model (After copying `best.pt` from Colab)
```powershell
python scripts/export_onnx.py
```

### F. Build C++ CLI Application (CMake)
```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
# Run CLI
./Release/road_damage.exe --image ../sample_inputs/road_test.jpg --out ../results/
```

---

## 6. Responsible-Use Limits
1. **Camera Geometry Limitation**: Bounding box area ratio $a = \text{area} / \text{img\_area}$ is a heuristic proxy for physical defect size. It is sensitive to camera mounting height, tilt angle, and focal length.
2. **Operational Role**: This automated deep-learning system is designed as an inspection triage tool for municipal road authorities. It does **not** replace certified on-site structural civil engineering inspections.
