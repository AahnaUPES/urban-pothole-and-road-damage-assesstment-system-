"""
===============================================================================
UrbanRoadDamage — Severity & Priority Module (PRD Section 20.2)
===============================================================================

PURPOSE
-------
Implements the rule-based severity and priority assessment system strictly
defined in Section 20.2 of the Urban Pothole and Road Damage Assessment System PRD.

MATHEMATICAL DEFINITIONS & RULES (PRD Section 20.2)
---------------------------------------------------
Let a be a detection's box area divided by image area:
    a = (box_width * box_height) / (image_width * image_height)

Overall image severity and maintenance priority are determined as follows:

| Severity | Rule (any condition)                                                    | Priority Action |
|----------|-------------------------------------------------------------------------|-----------------|
| High     | D40 pothole with a >= 2%; or D20 alligator crack with a >= 5%;          | Urgent          |
|          | or 3 or more D40 detections in one image                                |                 |
| Medium   | D40 with a < 2%; or D20 with a < 5%; or D00/D10 with a >= 3%             | Scheduled       |
| Low      | D00/D10 with a < 3%, or only low-confidence detections                  | Monitor         |
| None     | No detection above the confidence threshold                             | No action       |

VIVA EXPLANATION & THEORETICAL BASIS
------------------------------------
1. Why rule-based instead of an end-to-end learned severity model?
   - RDD2022 provides ground-truth bounding boxes and damage categories (D00,
     D10, D20, D40), but NO municipal severity labels (Low/Medium/High).
   - Training a learned classifier without ground-truth labels would invent
     supervision. Rule-based evaluation provides deterministic, explainable,
     and auditable domain heuristics.
2. What are the key limitations?
   - Box area depends on camera distance, tilt angle, and focal length. A 20cm
     pothole close to the bumper has a larger 'a' than a 50cm pothole 10m ahead.
   - PRD §20.2 notes this stated limitation. In production, camera calibration
     or depth sensors would be required.
3. How is overall priority decided?
   - If any condition for High severity is met, the image is marked High (Urgent).
   - Else if any condition for Medium is met, it is marked Medium (Scheduled).
   - Else if any valid detection is present, it is Low (Monitor).
   - Otherwise, None (No action).

STATUS: DONE (Implementation complete; initial thresholds per PRD §20.2)
===============================================================================
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

# Resolve config path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "severity_config.yaml"

CLASS_ID_TO_CODE = {
    0: "D00",  # Longitudinal Crack
    1: "D10",  # Transverse Crack
    2: "D20",  # Alligator Crack
    3: "D40",  # Pothole
}

CLASS_CODE_TO_ID = {v: k for k, v in CLASS_ID_TO_CODE.items()}


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load severity configuration from YAML."""
    path = config_path if config_path is not None else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_box_area_ratio(box: List[float], image_shape: Tuple[int, int]) -> float:
    """
    Compute area ratio: a = box_area / image_area.

    Parameters
    ----------
    box : List[float]
        [x_min, y_min, x_max, y_max] in pixel coordinates.
    image_shape : Tuple[int, int]
        (height, width) of the image.

    Returns
    -------
    float
        Box area divided by image area in [0.0, 1.0].
    """
    x1, y1, x2, y2 = box
    w = max(0.0, float(x2 - x1))
    h = max(0.0, float(y2 - y1))
    img_h, img_w = image_shape
    total_area = float(img_h * img_w)
    if total_area <= 0.0:
        return 0.0
    return (w * h) / total_area


def classify_detection_severity(
    class_code: str, area_ratio: float, config: Optional[Dict[str, Any]] = None
) -> Tuple[str, str]:
    """
    Classify a single detection according to PRD Section 20.2 rules.

    Returns
    -------
    Tuple[str, str]
        (severity_level, priority_action)
        e.g. ("High", "Urgent")
    """
    if config is None:
        config = load_config()

    thresh = config.get("thresholds", {})
    d40_high_area = thresh.get("D40_high_area", 0.02)
    d20_high_area = thresh.get("D20_high_area", 0.05)
    crack_med_area = thresh.get("D00_D10_medium_area", 0.03)

    if class_code == "D40":
        if area_ratio >= d40_high_area:
            return "High", "Urgent"
        else:
            return "Medium", "Scheduled"
    elif class_code == "D20":
        if area_ratio >= d20_high_area:
            return "High", "Urgent"
        else:
            return "Medium", "Scheduled"
    elif class_code in ("D00", "D10"):
        if area_ratio >= crack_med_area:
            return "Medium", "Scheduled"
        else:
            return "Low", "Monitor"
    else:
        # Default fallback for unknown defect
        return "Low", "Monitor"


def compute_severity(
    detections: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate overall image severity and priority according to PRD Section 20.2.

    Parameters
    ----------
    detections : List[Dict[str, Any]]
        List of detection dictionaries with keys:
          - 'box': [x_min, y_min, x_max, y_max]
          - 'class_id': int (0=D00, 1=D10, 2=D20, 3=D40) or 'class_code': str
          - 'confidence': float (0.0 to 1.0)
    image_shape : Tuple[int, int]
        (height, width)
    config : Optional[Dict[str, Any]]
        Configuration dictionary

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
          - 'overall_severity': 'High' | 'Medium' | 'Low' | 'None'
          - 'priority_action': 'Urgent' | 'Scheduled' | 'Monitor' | 'No action'
          - 'per_detection': list of enriched detections
          - 'counts': per-class detection counts
          - 'reason': textual explanation of the rule triggered for viva defense
    """
    if config is None:
        config = load_config()

    conf_thresh = config.get("confidence_threshold", 0.25)
    thresh = config.get("thresholds", {})
    d40_high_area = thresh.get("D40_high_area", 0.02)
    d40_high_count = thresh.get("D40_high_count", 3)
    d20_high_area = thresh.get("D20_high_area", 0.05)
    crack_med_area = thresh.get("D00_D10_medium_area", 0.03)

    # 1. Filter detections by confidence threshold
    kept = [d for d in detections if d.get("confidence", 1.0) >= conf_thresh]

    if not kept:
        return {
            "overall_severity": "None",
            "priority_action": "No action",
            "per_detection": [],
            "counts": {"D00": 0, "D10": 0, "D20": 0, "D40": 0},
            "num_detections": 0,
            "reason": "No detections above confidence threshold.",
        }

    # 2. Enrich detections with area ratio, class code, and individual severity
    enriched = []
    counts = {"D00": 0, "D10": 0, "D20": 0, "D40": 0}

    for d in kept:
        c_id = d.get("class_id")
        c_code = d.get("class_code", CLASS_ID_TO_CODE.get(c_id, "D00"))
        if c_code in counts:
            counts[c_code] += 1

        ratio = compute_box_area_ratio(d["box"], image_shape)
        sev, act = classify_detection_severity(c_code, ratio, config)

        item = dict(d)
        item["class_code"] = c_code
        item["area_ratio"] = ratio
        item["severity"] = sev
        item["priority_action"] = act
        enriched.append(item)

    # 3. Apply PRD §20.2 Image-level Rules
    # Rule HIGH:
    #   - D40 pothole with a >= 2%
    #   - OR D20 alligator crack with a >= 5%
    #   - OR 3 or more D40 detections in one image
    high_reasons = []
    if counts["D40"] >= d40_high_count:
        high_reasons.append(f"Contains {counts['D40']} D40 potholes (>= {d40_high_count})")
    for d in enriched:
        if d["class_code"] == "D40" and d["area_ratio"] >= d40_high_area:
            high_reasons.append(f"D40 pothole area ratio {d['area_ratio']:.2%} >= {d40_high_area:.0%}")
        elif d["class_code"] == "D20" and d["area_ratio"] >= d20_high_area:
            high_reasons.append(f"D20 alligator crack area ratio {d['area_ratio']:.2%} >= {d20_high_area:.0%}")

    if high_reasons:
        return {
            "overall_severity": "High",
            "priority_action": "Urgent",
            "per_detection": enriched,
            "counts": counts,
            "num_detections": len(enriched),
            "reason": "; ".join(high_reasons),
        }

    # Rule MEDIUM:
    #   - D40 with a < 2%
    #   - OR D20 with a < 5%
    #   - OR D00/D10 with a >= 3%
    med_reasons = []
    for d in enriched:
        if d["class_code"] == "D40" and d["area_ratio"] < d40_high_area:
            med_reasons.append(f"D40 pothole area ratio {d['area_ratio']:.2%} < {d40_high_area:.0%}")
        elif d["class_code"] == "D20" and d["area_ratio"] < d20_high_area:
            med_reasons.append(f"D20 alligator crack area ratio {d['area_ratio']:.2%} < {d20_high_area:.0%}")
        elif d["class_code"] in ("D00", "D10") and d["area_ratio"] >= crack_med_area:
            med_reasons.append(f"{d['class_code']} crack area ratio {d['area_ratio']:.2%} >= {crack_med_area:.0%}")

    if med_reasons:
        return {
            "overall_severity": "Medium",
            "priority_action": "Scheduled",
            "per_detection": enriched,
            "counts": counts,
            "num_detections": len(enriched),
            "reason": "; ".join(med_reasons),
        }

    # Rule LOW:
    #   - D00/D10 with a < 3%
    return {
        "overall_severity": "Low",
        "priority_action": "Monitor",
        "per_detection": enriched,
        "counts": counts,
        "num_detections": len(enriched),
        "reason": f"Only minor cracks (D00/D10 < {crack_med_area:.0%}).",
    }


def compute_priority(
    severity_result: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper that adds ranked detections and priority ranks."""
    if config is None:
        config = load_config()

    priority_ranks = config.get(
        "priority_ranks",
        {"Urgent": 3, "Scheduled": 2, "Monitor": 1, "No action": 0},
    )

    ranked = sorted(
        severity_result.get("per_detection", []),
        key=lambda d: (
            priority_ranks.get(d.get("priority_action", "Monitor"), 0),
            d.get("area_ratio", 0.0),
        ),
        reverse=True,
    )

    return {
        "ranked_detections": ranked,
        "overall_priority_score": priority_ranks.get(
            severity_result.get("priority_action", "No action"), 0
        ),
        "priority_label": severity_result.get("priority_action", "No action"),
    }


def analyse_image(
    detections: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """High-level entry point combining severity and priority ranking."""
    if config is None:
        config = load_config()

    sev = compute_severity(detections, image_shape, config)
    prio = compute_priority(sev, config)

    result = dict(sev)
    result.update(prio)
    return result


if __name__ == "__main__":
    test_dets = [
        {"box": [50, 50, 100, 100], "class_id": 0, "confidence": 0.85},   # small crack
        {"box": [200, 200, 350, 350], "class_id": 3, "confidence": 0.92}, # pothole (a > 2%)
    ]
    res = analyse_image(test_dets, image_shape=(640, 640))
    print(f"Severity: {res['overall_severity']} | Priority: {res['priority_label']}")
    print(f"Reason:   {res['reason']}")
    for d in res["ranked_detections"]:
        print(f" - {d['class_code']}: a={d['area_ratio']:.3%}, sev={d['severity']}, act={d['priority_action']}")
