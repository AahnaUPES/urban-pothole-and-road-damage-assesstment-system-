/**
 * ============================================================================
 * UrbanRoadDamage — CSV Summary Output Writer Implementation
 * File: src/export/csv_writer.cpp (PRD Section 5 FR-08)
 * ============================================================================
 */

#include "csv_writer.h"
#include <fstream>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

namespace road_damage {

bool CsvWriter::appendAssessmentRow(const std::string& csv_path,
                                   const std::string& image_name,
                                   const ImageAssessment& assessment,
                                   const std::string& gps_status,
                                   double latency_ms)
{
    bool file_exists = fs::exists(csv_path);
    std::ofstream out(csv_path, std::ios::app);
    if (!out.is_open()) return false;

    // Write header if creating new file
    if (!file_exists) {
        out << "filename,overall_severity,priority_action,d00_count,d10_count,d20_count,d40_count,total_defects,latency_ms,gps_status,rule_reason\n";
    }

    out << "\"" << image_name << "\","
        << "\"" << assessment.overall_severity << "\","
        << "\"" << assessment.priority_action << "\","
        << assessment.d00_count << ","
        << assessment.d10_count << ","
        << assessment.d20_count << ","
        << assessment.d40_count << ","
        << assessment.total_detections << ","
        << std::fixed << std::setprecision(2) << latency_ms << ","
        << "\"" << gps_status << "\","
        << "\"" << assessment.rule_reason << "\"\n";

    return true;
}

} // namespace road_damage
