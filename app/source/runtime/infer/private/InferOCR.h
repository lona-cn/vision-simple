#pragma once
#include <array>

#include "Infer.h"
#include "InferTask.h"

namespace vision_simple {
class InferContextORT;

class InferOCROrtPaddleImpl final : public InferOCR {
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend VSResult<std::unique_ptr<detail::FrameTask>> detail::MakeFrameTask(
      InferOCR&, const cv::Mat&, float) noexcept;

 public:
  explicit InferOCROrtPaddleImpl(InferContextORT& ort_ctx,
                                 OCRModelType model_type,
                                 std::map<int, std::string> char_dict,
                                 std::unique_ptr<Ort::Session> det,
                                 std::unique_ptr<Ort::Session> rec,
                                 size_t batch_size,
                                 std::array<int64_t, 4> recognition_shape);
  ~InferOCROrtPaddleImpl() override;

  OCRModelType model_type() const noexcept override;
  RunResult Run(const cv::Mat& image,
                float confidence_threshold) noexcept override;
};
}  // namespace vision_simple
