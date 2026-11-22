/**
 * ============================================================================
 * UrbanRoadDamage — C++ Command-Line Assessment Application
 * File: src/main.cpp (PRD Section 15 & FR-01 through FR-08)
 * ============================================================================
 *
 * COMMAND-LINE INTERFACE:
 *   road_damage --image <file> | --dir <folder> | --video <file>
 *               [--gps lat,lon] [--config config.yaml] [--out results/]
 *
 * WORKFLOW:
 *   1. Parse command-line flags.
 *   2. Load configuration (thresholds, model path).
 *   3. Initialize OnnxDetector and load ONNX weights.
 *   4. For each image / video frame:
 *      a. Run ONNX inference (resize, normalize, forward pass, NMS).
 *      b. Apply rule-based severity and priority assessment (PRD §20.2).
 *      c. Annotate bounding boxes on image color-coded by severity.
 *      d. Write per-image JSON record and append to run-level CSV.
 *      e. Print execution summary with latency to console.
 * ============================================================================
 */

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "detection/inference.h"
#include "severity/severity.h"
#include "export/json_writer.h"
#include "export/csv_writer.h"

namespace fs = std::filesystem;
using namespace road_damage;

void printUsage() {
    std::cout << "============================================================================\n"
              << "UrbanRoadDamage — Road Damage Assessment C++ CLI Application\n"
              << "============================================================================\n"
              << "Usage:\n"
              << "  road_damage --image <path>  [--gps lat,lon] [--config <path>] [--out <dir>]\n"
              << "  road_damage --dir   <path>  [--config <path>] [--out <dir>]\n"
              << "  road_damage --video <path>  [--config <path>] [--out <dir>]\n\n"
              << "Options:\n"
              << "  --image   Single image file (JPG/PNG)\n"
              << "  --dir     Directory containing images\n"
              << "  --video   Video file for frame-by-frame analysis\n"
              << "  --gps     Optional GPS coordinates (e.g. 19.0760,72.8777)\n"
              << "  --config  Path to YAML config (default: configs/config.yaml)\n"
              << "  --out     Output directory (default: results/)\n"
              << "============================================================================\n";
}

int main(int argc, char* argv[]) {
    std::string image_path = "";
    std::string dir_path = "";
    std::string video_path = "";
    std::string gps_info = "GPS: not available";
    std::string config_path = "configs/config.yaml";
    std::string out_dir = "results/";

    // 1. Parse Command-Line Arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--image" && i + 1 < argc) image_path = argv[++i];
        else if (arg == "--dir" && i + 1 < argc) dir_path = argv[++i];
        else if (arg == "--video" && i + 1 < argc) video_path = argv[++i];
        else if (arg == "--gps" && i + 1 < argc) gps_info = argv[++i];
        else if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        }
    }

    if (image_path.empty() && dir_path.empty() && video_path.empty()) {
        std::cerr << "[ERROR] Please specify either --image, --dir, or --video.\n\n";
        printUsage();
        return 1;
    }

    fs::create_directories(out_dir);
    std::string run_csv_path = (fs::path(out_dir) / "assessment_summary.csv").string();

    std::cout << "[INFO] Initializing UrbanRoadDamage C++ CLI\n"
              << "[INFO] Output directory: " << out_dir << "\n"
              << "[INFO] GPS status: " << gps_info << "\n";

    // 2. Initialize Detector & Severity Evaluator
    std::string model_onnx = "models/best.onnx";
    OnnxDetector detector(model_onnx, 0.25f, 0.45f, 640, 640);
    detector.loadModel();

    SeverityEvaluator evaluator(0.02f, 3, 0.05f, 0.03f);

    // Collect image files to process
    std::vector<std::string> target_files;
    if (!image_path.empty()) {
        target_files.push_back(image_path);
    } else if (!dir_path.empty()) {
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                target_files.push_back(entry.path().string());
            }
        }
    }

    // 3. Process Images
    if (!target_files.empty()) {
        std::cout << "[INFO] Processing " << target_files.size() << " image(s)...\n";
        for (const auto& file : target_files) {
            cv::Mat img = cv::imread(file);
            if (img.empty()) {
                std::cerr << "[WARNING] Could not read image: " << file << std::endl;
                continue;
            }

            auto start = std::chrono::high_resolution_clock::now();
            std::vector<Detection> detections = detector.detect(img);
            ImageAssessment assessment = evaluator.assess(detections, img.cols, img.rows);
            auto end = std::chrono::high_resolution_clock::now();
            double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

            // Print Console Summary (PRD Section 5 FR-07)
            std::cout << "--------------------------------------------------------\n"
                      << "Image: " << fs::path(file).filename().string() << "\n"
                      << "  Detections: " << assessment.total_detections << "\n"
                      << "  Severity:   " << assessment.overall_severity << "\n"
                      << "  Priority:   " << assessment.priority_action << "\n"
                      << "  Reason:     " << assessment.rule_reason << "\n"
                      << "  Latency:    " << latency_ms << " ms\n";

            // Annotate image
            cv::Mat annotated = img.clone();
            for (const auto& d : assessment.enriched_detections) {
                cv::Scalar color = (d.severity == "High") ? cv::Scalar(0, 0, 220) :
                                   (d.severity == "Medium") ? cv::Scalar(0, 140, 255) :
                                   cv::Scalar(0, 200, 0);
                cv::rectangle(annotated, d.box, color, 2);
                std::string label = d.class_name + " " + std::to_string(int(d.confidence * 100)) + "%";
                cv::putText(annotated, label, cv::Point(d.box.x, std::max(20.0f, d.box.y - 5)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            }

            std::string stem = fs::path(file).stem().string();
            std::string out_img_path = (fs::path(out_dir) / (stem + "_annotated.jpg")).string();
            std::string out_json_path = (fs::path(out_dir) / (stem + "_assessment.json")).string();

            cv::imwrite(out_img_path, annotated);
            JsonWriter::writeAssessmentJson(out_json_path, fs::path(file).filename().string(), assessment, gps_info, latency_ms);
            CsvWriter::appendAssessmentRow(run_csv_path, fs::path(file).filename().string(), assessment, gps_info, latency_ms);
        }
    }

    std::cout << "========================================================\n"
              << "[SUCCESS] All runs completed. Results saved in: " << out_dir << "\n"
              << "========================================================\n";

    return 0;
}
