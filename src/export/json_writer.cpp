/**
 * ============================================================================
 * UrbanRoadDamage — JSON Output Writer Implementation
 * File: src/export/json_writer.cpp (PRD Section 5 FR-08)
 * ============================================================================
 */

#include "json_writer.h"
#include <fstream>
#include <iomanip>

namespace road_damage {

bool JsonWriter::writeAssessmentJson(const std::string& output_path,
                                    const std::string& image_name,
                                    const ImageAssessment& assessment,
                                    const std::string& gps_status,
                                    double latency_ms)
{
    std::ofstream out(output_path);
    if (!out.is_open()) return false;

    out << "{\n";
    out << "  \"image_file\": \"" << image_name << "\",\n";
    out << "  \"overall_severity\": \"" << assessment.overall_severity << "\",\n";
    out << "  \"priority_action\": \"" << assessment.priority_action << "\",\n";
    out << "  \"gps_status\": \"" << gps_status << "\",\n";
    out << "  \"inference_latency_ms\": " << std::fixed << std::setprecision(2) << latency_ms << ",\n";
    out << "  \"rule_reason\": \"" << assessment.rule_reason << "\",\n";
    out << "  \"counts\": {\n";
    out << "    \"D00\": " << assessment.d00_count << ",\n";
    out << "    \"D10\": " << assessment.d10_count << ",\n";
    out << "    \"D20\": " << assessment.d20_count << ",\n";
    out << "    \"D40\": " << assessment.d40_count << "\n";
    out << "  },\n";
    out << "  \"detections\": [\n";

    for (size_t i = 0; i < assessment.enriched_detections.size(); ++i) {
        const auto& d = assessment.enriched_detections[i];
        out << "    {\n";
        out << "      \"class\": \"" << d.class_name << "\",\n";
        out << "      \"confidence\": " << std::setprecision(4) << d.confidence << ",\n";
        out << "      \"box\": [" << d.box.x << ", " << d.box.y << ", " << d.box.width << ", " << d.box.height << "],\n";
        out << "      \"area_ratio\": " << std::setprecision(5) << d.area_ratio << ",\n";
        out << "      \"severity\": \"" << d.severity << "\",\n";
        out << "      \"priority\": \"" << d.priority << "\"\n";
        out << "    }" << (i + 1 < assessment.enriched_detections.size() ? "," : "") << "\n";
    }

    out << "  ]\n";
    out << "}\n";
    return true;
}

} // namespace road_damage
