#pragma once

#include <array>
#include <variant>

#include "Infer.h"

namespace vision_simple {

enum class YOLOTask : uint8_t { kSegmentation, kPose, kOBB };

struct YOLOSegmentationResult {
  int32_t class_id;
  float confidence;
  cv::Rect bbox;
  // Owning CV_8UC1 binary mask (0/255), cropped to bbox in original pixels.
  cv::Mat mask;
};
struct YOLOSegmentationFrame {
  std::vector<YOLOSegmentationResult> results;
};
struct YOLOKeypoint {
  float x, y, confidence;
};
struct YOLOPoseResult {
  int32_t class_id;
  float confidence;
  cv::Rect bbox;
  std::vector<YOLOKeypoint> keypoints;
};
struct YOLOPoseFrame {
  std::vector<YOLOPoseResult> results;
};
struct YOLOOBBResult {
  int32_t class_id;
  float confidence;
  // Ordered corners in original-image pixels, not an axis-aligned envelope.
  // Corners may lie outside the image; clipping would destroy the
  // quadrilateral.
  std::array<cv::Point2f, 4> corners;
  // Radians: orientation of the edge from corners[0] to corners[1].
  float angle;
};
struct YOLOOBBFrame {
  std::vector<YOLOOBBResult> results;
};
using YOLOTaskFrameResult =
    std::variant<YOLOSegmentationFrame, YOLOPoseFrame, YOLOOBBFrame>;

// YOLO11 non-detection exports. Existing InferYOLO/YOLOResult remain unchanged.
class VISION_SIMPLE_API InferYOLOTask {
 public:
  using CreateResult = VSResult<std::unique_ptr<InferYOLOTask>>;
  using RunResult = VSResult<YOLOTaskFrameResult>;
  InferYOLOTask() = default;
  virtual ~InferYOLOTask() = default;
  InferYOLOTask(const InferYOLOTask&) = delete;
  InferYOLOTask& operator=(const InferYOLOTask&) = delete;
  InferYOLOTask(InferYOLOTask&&) noexcept = default;
  InferYOLOTask& operator=(InferYOLOTask&&) noexcept = default;
  virtual YOLOTask task() const noexcept = 0;
  virtual const std::vector<std::string>& class_names() const noexcept = 0;
  virtual RunResult Run(const cv::Mat& image,
                        float confidence_threshold) noexcept = 0;
  static CreateResult Create(InferContext& context, std::span<uint8_t> data,
                             YOLOTask task, size_t device_id = 0) noexcept;
  static CreateResult Create(InferContext& context, const std::string& path,
                             YOLOTask task, size_t device_id = 0) noexcept;
};

}  // namespace vision_simple
