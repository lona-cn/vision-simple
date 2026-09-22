#pragma once

#include <variant>

#include "InferPipeline.h"

namespace vision_simple::detail {

enum class PipelineLane { kPreprocess, kInference, kPostprocess };
using FrameResult =
    std::variant<YOLOFrameResult, OCRFrameResult, YOLOTaskFrameResult>;

// A task exclusively owns reusable backend workspace. Advance performs exactly
// one native stage. nullopt completes the frame; otherwise return its next
// lane. No worker may block waiting for another pipeline task.
class FrameTask {
 public:
  virtual ~FrameTask() = default;
  virtual VSResult<std::optional<PipelineLane>> Advance() noexcept = 0;
  virtual FrameResult TakeResult() noexcept = 0;
};

VSResult<std::unique_ptr<FrameTask>> MakeFrameTask(
    InferYOLO& model, const cv::Mat& image,
    float confidence_threshold) noexcept;
VSResult<std::unique_ptr<FrameTask>> MakeFrameTask(
    InferOCR& model, const cv::Mat& image, float confidence_threshold) noexcept;
VSResult<std::unique_ptr<FrameTask>> MakeFrameTask(
    InferYOLOTask& model, const cv::Mat& image,
    float confidence_threshold) noexcept;

}  // namespace vision_simple::detail
