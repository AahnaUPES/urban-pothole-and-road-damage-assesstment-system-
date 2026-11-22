#pragma once

/**
 * ============================================================================
 * UrbanRoadDamage — JSON Output Writer Header
 * File: src/export/json_writer.h (PRD Section 5 FR-08)
 * ============================================================================
 */

#include <string>
#include "../severity/severity.h"

namespace road_damage {

class JsonWriter {
public:
    static bool writeAssessmentJson(const std::string& output_path,
                                    const std::string& image_name,
                                    const ImageAssessment& assessment,
                                    const std::string& gps_status,
                                    double latency_ms);
};

} // namespace road_damage
