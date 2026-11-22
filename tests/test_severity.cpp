/**
 * ============================================================================
 * UrbanRoadDamage — C++ Severity Evaluator Unit Tests (PRD Section 15 & 20.2)
 * File: tests/test_severity.cpp
 * ============================================================================
 */

#include <iostream>
#include <cassert>
#include "../src/severity/severity.h"

using namespace road_damage;

void test_empty_detections() {
    SeverityEvaluator evaluator(0.02f, 3, 0.05f, 0.03f);
    ImageAssessment res = evaluator.assess({}, 640, 640);
    assert(res.overall_severity == "None");
    assert(res.priority_action == "No action");
    assert(res.total_detections == 0);
    std::cout << "[PASS] test_empty_detections" << std::endl;
}

void test_high_severity_pothole() {
    SeverityEvaluator evaluator(0.02f, 3, 0.05f, 0.03f);
    Detection d;
    d.class_name = "D40";
    d.box = cv::Rect2f(100, 100, 120, 120); // area = 14400 / 409600 = 0.035 > 0.02
    d.confidence = 0.90f;

    ImageAssessment res = evaluator.assess({d}, 640, 640);
    assert(res.overall_severity == "High");
    assert(res.priority_action == "Urgent");
    std::cout << "[PASS] test_high_severity_pothole" << std::endl;
}

void test_medium_severity_crack() {
    SeverityEvaluator evaluator(0.02f, 3, 0.05f, 0.03f);
    Detection d;
    d.class_name = "D10";
    d.box = cv::Rect2f(10, 10, 150, 150); // area = 22500 / 409600 = 0.0549 >= 0.03
    d.confidence = 0.85f;

    ImageAssessment res = evaluator.assess({d}, 640, 640);
    assert(res.overall_severity == "Medium");
    assert(res.priority_action == "Scheduled");
    std::cout << "[PASS] test_medium_severity_crack" << std::endl;
}

void test_three_potholes_rule() {
    SeverityEvaluator evaluator(0.02f, 3, 0.05f, 0.03f);
    Detection d1, d2, d3;
    d1.class_name = "D40"; d1.box = cv::Rect2f(0, 0, 40, 40); d1.confidence = 0.8f;
    d2.class_name = "D40"; d2.box = cv::Rect2f(50, 50, 40, 40); d2.confidence = 0.8f;
    d3.class_name = "D40"; d3.box = cv::Rect2f(100, 100, 40, 40); d3.confidence = 0.8f;

    ImageAssessment res = evaluator.assess({d1, d2, d3}, 640, 640);
    assert(res.overall_severity == "High");
    assert(res.priority_action == "Urgent");
    std::cout << "[PASS] test_three_potholes_rule" << std::endl;
}

int main() {
    std::cout << "Running C++ Severity Evaluator Unit Tests..." << std::endl;
    test_empty_detections();
    test_high_severity_pothole();
    test_medium_severity_crack();
    test_three_potholes_rule();
    std::cout << "All C++ Severity Unit Tests PASSED!" << std::endl;
    return 0;
}
