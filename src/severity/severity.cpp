/**
 * ============================================================================
 * UrbanRoadDamage — C++ Rule-Based Severity & Maintenance Priority Implementation
 * File: src/severity/severity.cpp (PRD Section 20.2)
 * ============================================================================
 */

#include "severity.h"
#include <sstream>

namespace road_damage {

SeverityEvaluator::SeverityEvaluator(float d40_high_area,
                                     int d40_high_count,
                                     float d20_high_area,
                                     float crack_med_area)
    : d40_high_area_(d40_high_area),
      d40_high_count_(d40_high_count),
      d20_high_area_(d20_high_area),
      crack_med_area_(crack_med_area)
{
}

ImageAssessment SeverityEvaluator::assess(const std::vector<Detection>& detections,
                                          int img_width,
                                          int img_height)
{
    ImageAssessment result;
    result.total_detections = static_cast<int>(detections.size());

    if (detections.empty() || img_width <= 0 || img_height <= 0) {
        result.overall_severity = "None";
        result.priority_action = "No action";
        result.rule_reason = "No defects detected above confidence threshold.";
        return result;
    }

    float total_image_area = static_cast<float>(img_width * img_height);
    std::vector<std::string> high_reasons;
    std::vector<std::string> med_reasons;

    // 1. Process each detection and classify individual severity
    for (const auto& det : detections) {
        Detection d = det;
        float box_area = d.box.width * d.box.height;
        d.area_ratio = box_area / total_image_area;

        if (d.class_name == "D00") result.d00_count++;
        else if (d.class_name == "D10") result.d10_count++;
        else if (d.class_name == "D20") result.d20_count++;
        else if (d.class_name == "D40") result.d40_count++;

        // Classify per-detection
        if (d.class_name == "D40") {
            if (d.area_ratio >= d40_high_area_) {
                d.severity = "High";
                d.priority = "Urgent";
                high_reasons.push_back("D40 pothole area >= 2%");
            } else {
                d.severity = "Medium";
                d.priority = "Scheduled";
                med_reasons.push_back("D40 pothole area < 2%");
            }
        } else if (d.class_name == "D20") {
            if (d.area_ratio >= d20_high_area_) {
                d.severity = "High";
                d.priority = "Urgent";
                high_reasons.push_back("D20 alligator crack area >= 5%");
            } else {
                d.severity = "Medium";
                d.priority = "Scheduled";
                med_reasons.push_back("D20 alligator crack area < 5%");
            }
        } else if (d.class_name == "D00" || d.class_name == "D10") {
            if (d.area_ratio >= crack_med_area_) {
                d.severity = "Medium";
                d.priority = "Scheduled";
                med_reasons.push_back(d.class_name + " crack area >= 3%");
            } else {
                d.severity = "Low";
                d.priority = "Monitor";
            }
        } else {
            d.severity = "Low";
            d.priority = "Monitor";
        }

        result.enriched_detections.push_back(d);
    }

    // 2. Check 3 or more D40 detections rule
    if (result.d40_count >= d40_high_count_) {
        high_reasons.push_back("3 or more D40 potholes in one image");
    }

    // 3. Resolve Overall Image Severity
    if (!high_reasons.empty()) {
        result.overall_severity = "High";
        result.priority_action = "Urgent";
        std::ostringstream ss;
        for (size_t i = 0; i < high_reasons.size(); ++i) {
            if (i > 0) ss << "; ";
            ss << high_reasons[i];
        }
        result.rule_reason = ss.str();
    } else if (!med_reasons.empty()) {
        result.overall_severity = "Medium";
        result.priority_action = "Scheduled";
        std::ostringstream ss;
        for (size_t i = 0; i < med_reasons.size(); ++i) {
            if (i > 0) ss << "; ";
            ss << med_reasons[i];
        }
        result.rule_reason = ss.str();
    } else {
        result.overall_severity = "Low";
        result.priority_action = "Monitor";
        result.rule_reason = "Minor hairline cracks (D00/D10 < 3%).";
    }

    return result;
}

} // namespace road_damage
