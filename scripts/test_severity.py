"""
===============================================================================
UrbanRoadDamage — Unit Tests for Severity & Priority Module (PRD Section 20.2)
===============================================================================

PURPOSE
-------
Verifies the exact rule table from PRD Section 20.2 using synthetic data:

| Severity | Rule (any condition)                                                    | Priority Action |
|----------|-------------------------------------------------------------------------|-----------------|
| High     | D40 pothole with a >= 2%; or D20 alligator crack with a >= 5%;          | Urgent          |
|          | or 3 or more D40 detections in one image                                |                 |
| Medium   | D40 with a < 2%; or D20 with a < 5%; or D00/D10 with a >= 3%             | Scheduled       |
| Low      | D00/D10 with a < 3%, or only low-confidence detections                  | Monitor         |
| None     | No detection above the confidence threshold                             | No action       |

All tests are synthetic unit checks; no model weights or GPUs are required.
===============================================================================
"""

import sys
from pathlib import Path
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from severity import (
    compute_box_area_ratio,
    classify_detection_severity,
    compute_severity,
    compute_priority,
    analyse_image,
    load_config,
)


@pytest.fixture
def image_640():
    return (640, 640)


@pytest.fixture
def config():
    return load_config()


def test_box_area_ratio_computation(image_640):
    """Test box area ratio a = box_area / image_area."""
    # Box of 64x64 on 640x640 is 4096 / 409600 = 0.01 (1.00%)
    ratio = compute_box_area_ratio([0, 0, 64, 64], image_640)
    assert pytest.approx(ratio, rel=1e-4) == 0.01


def test_no_detections_returns_none(image_640, config):
    """No detections should return overall severity None and No action."""
    res = compute_severity([], image_640, config)
    assert res["overall_severity"] == "None"
    assert res["priority_action"] == "No action"
    assert res["num_detections"] == 0


def test_low_confidence_filtered(image_640, config):
    """Low confidence detections (< 0.25) must be discarded."""
    dets = [{"box": [0, 0, 300, 300], "class_id": 3, "confidence": 0.15}]
    res = compute_severity(dets, image_640, config)
    assert res["overall_severity"] == "None"
    assert res["priority_action"] == "No action"


def test_small_crack_is_low_severity(image_640, config):
    """D00 or D10 with a < 3% must be Low severity, Monitor priority."""
    # Box 50x50 on 640x640: a = 2500 / 409600 = 0.0061 < 0.03
    dets = [{"box": [10, 10, 60, 60], "class_id": 0, "confidence": 0.85}]
    res = compute_severity(dets, image_640, config)
    assert res["overall_severity"] == "Low"
    assert res["priority_action"] == "Monitor"


def test_large_crack_is_medium_severity(image_640, config):
    """D00 or D10 with a >= 3% must be Medium severity, Scheduled priority."""
    # Box 120x120 on 640x640: a = 14400 / 409600 = 0.03515 >= 0.03
    dets = [{"box": [10, 10, 130, 130], "class_id": 1, "confidence": 0.80}]
    res = compute_severity(dets, image_640, config)
    assert res["overall_severity"] == "Medium"
    assert res["priority_action"] == "Scheduled"


def test_small_pothole_is_medium_severity(image_640, config):
    """D40 with a < 2% must be Medium severity, Scheduled priority."""
    # Box 80x80 on 640x640: a = 6400 / 409600 = 0.0156 < 0.02
    dets = [{"box": [100, 100, 180, 180], "class_id": 3, "confidence": 0.90}]
    res = compute_severity(dets, image_640, config)
    assert res["overall_severity"] == "Medium"
    assert res["priority_action"] == "Scheduled"


def test_large_pothole_is_high_severity(image_640, config):
    """D40 with a >= 2% must be High severity, Urgent priority."""
    # Box 100x100 on 640x640: a = 10000 / 409600 = 0.0244 >= 0.02
    dets = [{"box": [100, 100, 200, 200], "class_id": 3, "confidence": 0.95}]
    res = compute_severity(dets, image_640, config)
    assert res["overall_severity"] == "High"
    assert res["priority_action"] == "Urgent"


def test_three_or_more_potholes_triggers_high_severity(image_640, config):
    """>= 3 D40 pothole detections triggers High severity even if each is small."""
    # Three small potholes, each 40x40 (a = 0.0039 < 0.02)
    dets = [
        {"box": [10, 10, 50, 50], "class_id": 3, "confidence": 0.8},
        {"box": [60, 60, 100, 100], "class_id": 3, "confidence": 0.8},
        {"box": [110, 110, 150, 150], "class_id": 3, "confidence": 0.8},
    ]
    res = compute_severity(dets, image_640, config)
    assert res["overall_severity"] == "High"
    assert res["priority_action"] == "Urgent"
    assert "3 D40 potholes" in res["reason"]


def test_large_alligator_crack_is_high_severity(image_640, config):
    """D20 alligator crack with a >= 5% must be High severity, Urgent priority."""
    # Box 160x160 on 640x640: a = 25600 / 409600 = 0.0625 >= 0.05
    dets = [{"box": [100, 100, 260, 260], "class_id": 2, "confidence": 0.88}]
    res = compute_severity(dets, image_640, config)
    assert res["overall_severity"] == "High"
    assert res["priority_action"] == "Urgent"


def test_analyse_image_complete_pipeline(image_640, config):
    """End-to-end test on analyse_image returning ranked detections."""
    dets = [
        {"box": [10, 10, 50, 50], "class_id": 0, "confidence": 0.85},     # small D00 (Low)
        {"box": [100, 100, 220, 220], "class_id": 3, "confidence": 0.92}, # large D40 (High)
    ]
    res = analyse_image(dets, image_640, config)
    assert res["overall_severity"] == "High"
    assert res["priority_label"] == "Urgent"
    assert res["ranked_detections"][0]["class_code"] == "D40"
    assert res["ranked_detections"][0]["severity"] == "High"
