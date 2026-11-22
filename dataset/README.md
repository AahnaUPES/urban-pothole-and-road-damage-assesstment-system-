# Dataset Card: RDD2022 (India Subset)
### Road Damage Dataset 2022 — Project 23

## 1. Dataset Overview & Provenance
- **Source**: RDD2022 (Crowdsensing-based Road Damage Detection Challenge 2022).
- **Subset**: India subset (`RDD2022_India`).
- **Context**: Captured via vehicle-mounted smartphone cameras on Indian urban and arterial roads.
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).

---

## 2. Damage Classes
The official four RDD2022 damage classes are used consistently across data processing, detector training, inference, and evaluation:

| Class Code | Class Name | Description | Municipal Risk Profile |
|------------|------------|-------------|------------------------|
| **D00** | Longitudinal Crack | Cracks running parallel to the direction of vehicle travel | Low risk initially; monitor for water penetration |
| **D10** | Transverse Crack | Cracks running perpendicular to the roadway centerline | Medium risk; can induce joint faulting |
| **D20** | Alligator Crack | Interconnected polygonal fatigue cracking (crocodile skin pattern) | High risk; indicates base/sub-base structural failure |
| **D40** | Pothole | Bowl-shaped localized depression or surface disintegration | Highest risk; severe hazard to traffic safety and tire/suspension |

---

## 3. Split Strategy & Leakage Prevention (PRD Section 7 & 20.2)
- **Challenge Test-Set Limitation**: The official challenge test-set annotations are unreleased. Therefore, our test set is carved from the labelled training data.
- **Near-Duplicate Grouping (Perceptual Hashing)**: Sequential video frames often have high visual redundancy. Splitting randomly by frame causes severe **data leakage** (identical road patches in train and test).
  - Perceptual hash algorithm: `imagehash.phash(img)` (64-bit).
  - Grouping threshold: Hamming distance $\le 4$.
  - Group-aware partitioning: Whole near-duplicate clusters are assigned to either Train, Val, or Test together.
- **Split Ratio**: Approximately 70% Train, 15% Validation, 15% Test.
- **Random Seed**: Fixed at `SEED = 0`.
- **Negative Samples**: Images with zero annotations are preserved in the validation/test sets to evaluate false positives per image on undamaged roads.

---

## 4. Dataset Counts & Distribution
> [!NOTE]
> Per honesty rules, exact counts are filled directly from Colab Cell 4 and Cell 5 after execution.

- **Total Usable Images**: `TODO: fill from Colab Cell 4`
- **Total Skipped Images**: `TODO: fill from Colab Cell 4`
- **Negative (Undamaged) Images**: `TODO: fill from Colab Cell 4`
- **Per-Class Instance Counts**:
  - D00 (Longitudinal): `TODO: fill from Colab Cell 4`
  - D10 (Transverse): `TODO: fill from Colab Cell 4`
  - D20 (Alligator): `TODO: fill from Colab Cell 4`
  - D40 (Pothole): `TODO: fill from Colab Cell 4`
- **Split Distribution**:
  - Train: `TODO: fill from Colab Cell 5` (approx 70%)
  - Validation: `TODO: fill from Colab Cell 5` (approx 15%)
  - Test: `TODO: fill from Colab Cell 5` (approx 15%)
