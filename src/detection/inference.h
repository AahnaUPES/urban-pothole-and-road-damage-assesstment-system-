#pragma once

/**
 * ============================================================================
 * UrbanRoadDamage — ONNX Runtime Object Detector Header
 * File: src/detection/inference.h (PRD Section 15)
 * ============================================================================
 *
 * PURPOSE & VIVA EXPLANATION:
 * 1. Why ONNX Runtime in C++?
 *    Provides hardware-accelerated deep learning inference with zero Python
 *    dependency, minimal memory footprint, and low latency for edge deployment.
 *
 * 2. Key Algorithms:
 *    - Letterbox Preprocessing: Resizes input image preserving aspect ratio
 *      with grey padding (114, 114, 114) to 640x640.
 *    - Tensor Conversion: Converts HWC BGR uint8 -> CHW RGB float32 [0.0, 1.0].
 *    - Non-Maximum Suppression (NMS): Filters redundant candidate boxes by
 *      sorting by confidence and suppressing overlapping boxes with IoU > nms_thresh.
 * ============================================================================
 */

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// Forward declarations if ONNX Runtime headers are conditionally available
#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace road_damage {

struct Detection {
    cv::Rect2f box;         // [x, y, width, height] in original image coordinates
    int class_id;           // 0=D00, 1=D10, 2=D20, 3=D40
    std::string class_name; // e.g. "D40"
    float confidence;       // [0.0, 1.0]
    float area_ratio;       // box_area / image_area
    std::string severity;   // "Low", "Medium", "High"
    std::string priority;   // "Monitor", "Scheduled", "Urgent"
};

class OnnxDetector {
public:
    OnnxDetector(const std::string& model_path,
                 float conf_thresh = 0.25f,
                 float nms_thresh = 0.45f,
                 int input_w = 640,
                 int input_h = 640);
    ~OnnxDetector();

    bool loadModel();
    std::vector<Detection> detect(const cv::Mat& image);

    static float computeIoU(const cv::Rect2f& boxA, const cv::Rect2f& boxB);
    static std::vector<Detection> applyNMS(const std::vector<Detection>& candidates, float nms_thresh);

private:
    std::string model_path_;
    float conf_threshold_;
    float nms_threshold_;
    int input_width_;
    int input_height_;
    bool is_loaded_;

    cv::Mat letterbox(const cv::Mat& src, float& scale, int& pad_x, int& pad_y);

#ifdef HAVE_ONNXRUNTIME
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
#endif
};

} // namespace road_damage
