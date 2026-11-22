/**
 * ============================================================================
 * UrbanRoadDamage — ONNX Runtime Object Detector Implementation
 * File: src/detection/inference.cpp (PRD Section 15)
 * ============================================================================
 */

#include "inference.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace road_damage {

static const std::vector<std::string> RDD_CLASSES = {
    "D00", // Longitudinal Crack
    "D10", // Transverse Crack
    "D20", // Alligator Crack
    "D40"  // Pothole
};

OnnxDetector::OnnxDetector(const std::string& model_path,
                           float conf_thresh,
                           float nms_thresh,
                           int input_w,
                           int input_h)
    : model_path_(model_path),
      conf_threshold_(conf_thresh),
      nms_threshold_(nms_thresh),
      input_width_(input_w),
      input_height_(input_h),
      is_loaded_(false)
#ifdef HAVE_ONNXRUNTIME
      , env_(ORT_LOGGING_LEVEL_WARNING, "UrbanRoadDamage")
#endif
{
}

OnnxDetector::~OnnxDetector() = default;

bool OnnxDetector::loadModel() {
#ifdef HAVE_ONNXRUNTIME
    try {
        session_options_.SetIntraOpNumThreads(2);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        std::wstring wide_path(model_path_.begin(), model_path_.end());
        session_ = std::make_unique<Ort::Session>(env_, wide_path.c_str(), session_options_);
#else
        session_ = std::make_unique<Ort::Session>(env_, model_path_.c_str(), session_options_);
#endif
        is_loaded_ = true;
        std::cout << "[INFO] Loaded ONNX model from: " << model_path_ << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to load ONNX model: " << e.what() << std::endl;
        is_loaded_ = false;
        return false;
    }
#else
    std::cout << "[INFO] ONNX Detector created in mock/stub mode (HAVE_ONNXRUNTIME not defined)." << std::endl;
    is_loaded_ = true;
    return true;
#endif
}

cv::Mat OnnxDetector::letterbox(const cv::Mat& src, float& scale, int& pad_x, int& pad_y) {
    float r = std::min((float)input_width_ / src.cols, (float)input_height_ / src.rows);
    scale = r;

    int new_unpad_w = std::round(src.cols * r);
    int new_unpad_h = std::round(src.rows * r);

    pad_x = (input_width_ - new_unpad_w) / 2;
    pad_y = (input_height_ - new_unpad_h) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat out(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(pad_x, pad_y, new_unpad_w, new_unpad_h)));

    return out;
}

float OnnxDetector::computeIoU(const cv::Rect2f& boxA, const cv::Rect2f& boxB) {
    float x1 = std::max(boxA.x, boxB.x);
    float y1 = std::max(boxA.y, boxB.y);
    float x2 = std::min(boxA.x + boxA.width, boxB.x + boxB.width);
    float y2 = std::min(boxA.y + boxA.height, boxB.y + boxB.height);

    float inter_w = std::max(0.0f, x2 - x1);
    float inter_h = std::max(0.0f, y2 - y1);
    float inter_area = inter_w * inter_h;

    float areaA = boxA.width * boxA.height;
    float areaB = boxB.width * boxB.height;
    float union_area = areaA + areaB - inter_area;

    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

std::vector<Detection> OnnxDetector::applyNMS(const std::vector<Detection>& candidates, float nms_thresh) {
    std::vector<Detection> sorted = candidates;
    std::sort(sorted.begin(), sorted.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });

    std::vector<Detection> kept;
    std::vector<bool> suppressed(sorted.size(), false);

    for (size_t i = 0; i < sorted.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(sorted[i]);

        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (suppressed[j]) continue;
            // Class-specific NMS
            if (sorted[i].class_id == sorted[j].class_id) {
                float iou = computeIoU(sorted[i].box, sorted[j].box);
                if (iou > nms_thresh) {
                    suppressed[j] = true;
                }
            }
        }
    }
    return kept;
}

std::vector<Detection> OnnxDetector::detect(const cv::Mat& image) {
    if (image.empty()) return {};

    float scale = 1.0f;
    int pad_x = 0, pad_y = 0;
    cv::Mat boxed = letterbox(image, scale, pad_x, pad_y);

#ifdef HAVE_ONNXRUNTIME
    if (!is_loaded_ || !session_) {
        std::cerr << "[ERROR] Model session is not loaded." << std::endl;
        return {};
    }

    // Convert BGR -> RGB and normalize [0, 1]
    cv::Mat rgb;
    cv::cvtColor(boxed, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    // HWC to CHW tensor layout
    std::vector<float> input_tensor_values(1 * 3 * input_height_ * input_width_);
    int channel_stride = input_height_ * input_width_;
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_height_; ++h) {
            for (int w = 0; w < input_width_; ++w) {
                input_tensor_values[c * channel_stride + h * input_width_ + w] =
                    rgb.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    std::array<int64_t, 4> input_shape{1, 3, input_height_, input_width_};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    // Typically shape is [1, 84, 8400] for YOLOv8 (4 bbox coords + 80 or 4 classes)

    std::vector<Detection> candidates;
    int num_classes = 4;
    int num_anchors = output_shape[2];

    for (int i = 0; i < num_anchors; ++i) {
        // Find best class score
        float max_score = 0.0f;
        int best_cls = -1;
        for (int c = 0; c < num_classes; ++c) {
            float score = output_data[(4 + c) * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                best_cls = c;
            }
        }

        if (max_score >= conf_threshold_) {
            float cx = output_data[0 * num_anchors + i];
            float cy = output_data[1 * num_anchors + i];
            float w  = output_data[2 * num_anchors + i];
            float h  = output_data[3 * num_anchors + i];

            // Transform back from letterbox coords to original image
            float x1 = (cx - w / 2.0f - pad_x) / scale;
            float y1 = (cy - h / 2.0f - pad_y) / scale;
            float bw = w / scale;
            float bh = h / scale;

            x1 = std::max(0.0f, std::min(x1, (float)image.cols));
            y1 = std::max(0.0f, std::min(y1, (float)image.rows));
            bw = std::min(bw, (float)image.cols - x1);
            bh = std::min(bh, (float)image.rows - y1);

            Detection det;
            det.box = cv::Rect2f(x1, y1, bw, bh);
            det.class_id = best_cls;
            det.class_name = (best_cls >= 0 && best_cls < 4) ? RDD_CLASSES[best_cls] : "Defect";
            det.confidence = max_score;
            det.area_ratio = (bw * bh) / (float)(image.cols * image.rows);
            candidates.push_back(det);
        }
    }

    return applyNMS(candidates, nms_threshold_);

#else
    // Fallback stub detection when compiled without ONNX runtime binary
    return {};
#endif
}

} // namespace road_damage
