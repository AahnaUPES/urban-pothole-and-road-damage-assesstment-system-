#pragma once

/**
 * ============================================================================
 * UrbanRoadDamage — CSV Summary Output Writer Header
 * File: src/export/csv_writer.h (PRD Section 5 FR-08)
 * ============================================================================
 */

#include <string>
#include "../severity/severity.h"

namespace road_damage {

class CsvWriter {
public:
    static bool appendAssessmentRow(const std::string& csv_path,
                                   const std::string& image_name,
                                   const ImageAssessment& assessment,
                                   const std::string& gps_status,
                                   double latency_ms);
};

} // namespace road_damage
