#pragma once

#include <chrono>
#include <optional>
#include <stop_token>

#include "Infer.h"
#include "InferYOLOTask.h"

namespace vision_simple {

struct PipelineOptions {
  size_t capacity = 4;
  size_t max_batches = 4;
  size_t max_batch_images = 128;
};

struct PipelineControl {
  std::stop_token stop;
  std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::time_point::max();
};

enum class PipelineFailureKind {
  kInference,
  kInvalidRequest,
  kBusy,
  kCancelled,
  kTimedOut,
  kClosed
};

struct PipelineFailure {
  PipelineFailureKind kind;
  std::optional<size_t> image_index;
  VisionSimpleError cause;
};

template <typename T>
using PipelineResult = std::expected<std::vector<T>, PipelineFailure>;

// Owns bounded stage queues and workers. Input pixels and models must remain
// alive and immutable until Run returns. Cancellation/deadlines are
// cooperative: Run drains executing native stages before releasing their
// workspaces.
class VISION_SIMPLE_API InferPipeline {
 public:
  using CreateResult = VSResult<std::unique_ptr<InferPipeline>>;
  static CreateResult Create(PipelineOptions options = {}) noexcept;
  ~InferPipeline();
  InferPipeline(const InferPipeline&) = delete;
  InferPipeline& operator=(const InferPipeline&) = delete;

  PipelineResult<YOLOFrameResult> Run(InferYOLO& model,
                                      std::span<const cv::Mat> images,
                                      float confidence_threshold,
                                      PipelineControl control = {}) noexcept;
  PipelineResult<OCRFrameResult> Run(InferOCR& model,
                                     std::span<const cv::Mat> images,
                                     float confidence_threshold,
                                     PipelineControl control = {}) noexcept;
  PipelineResult<YOLOTaskFrameResult> Run(
      InferYOLOTask& model, std::span<const cv::Mat> images,
      float confidence_threshold, PipelineControl control = {}) noexcept;

  // Reject new batches and cooperatively cancel admitted work; thread-safe.
  // The owner must join callers before destroying this object.
  void Close() noexcept;

 private:
  struct Impl;
  explicit InferPipeline(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vision_simple
