#pragma once

/**
 * ============================================================================
 * UrbanRoadDamage — C++ Rule-Based Severity & Maintenance Priority Module
 * File: src/severity/severity.h (PRD Section 20.2)
 * ============================================================================
 *
 * SPECIFICATION (PRD Section 20.2):
 * ---------------------------------
 * Let a = detection's box area divided by image area.
 *
 * High Severity | Priority: Urgent
 *   - D40 pothole with a >= 2% (0.02)
 *   - OR D20 alligator crack with a >= 5% (0.05)
 *   - OR 3 or more D40 detections in one image
 *
 * Medium Severity | Priority: Scheduled
 *   - D40 pothole with a < 2% (0.02)
 *   - OR D20 alligator crack with a < 5% (0.05)
 *   - OR D00/D10 crack with a >= 3% (0.03)
 *
 * Low Severity | Priority: Monitor
 *   - D00/D10 crack with a < 3% (0.03)
 *   - Or only low-confidence detections
 *
 * None | Priority: No action
 *   - No detection above the confidence threshold
 * ============================================================================
 */

#include <string>
#include <vector>
#include "../detection/inference.h"

namespace road_damage {

struct ImageAssessment {
    std::string overall_severity; // "High", "Medium", "Low", "None"
    std::string priority_action;  // "Urgent", "Scheduled", "Monitor", "No action"
    int d00_count{0};
    int d10_count{0};
    int d20_count{0};
    int d40_count{0};
    int total_detections{0};
    std::string rule_reason;
    std::vector<Detection> enriched_detections;
};

class SeverityEvaluator {
public:
    SeverityEvaluator(float d40_high_area = 0.02f,
                      int d40_high_count = 3,
                      float d20_high_area = 0.05f,
                      float crack_med_area = 0.03f);

    ImageAssessment assess(const std::vector<Detection>& detections, int img_width, int img_height);

private:
    float d40_high_area_;
    int d40_high_count_;
    float d20_high_area_;
    float crack_med_area_;
};

} // namespace road_damage
