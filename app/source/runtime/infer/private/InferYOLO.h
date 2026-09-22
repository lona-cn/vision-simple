#pragma once
#include <magic_enum.hpp>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <array>

#include "../Infer.h"
#include "InferORT.h"
#include "InferTask.h"
#include "VisionHelper.hpp"

namespace vision_simple {
class YOLOFilter {
  YOLOVersion version_;
  std::vector<std::string> class_names_;
  std::vector<int64_t> shapes_;

  static std::vector<YOLOResult> ApplyNMS(
      const std::vector<YOLOResult>& detections, float iou_threshold);

 public:
  using FilterResult = InferResult<YOLOFrameResult>;

  explicit YOLOFilter(YOLOVersion version, std::vector<std::string> class_names,
                      std::vector<int64_t> shapes);

  YOLOVersion version() const noexcept;

  FilterResult v11(std::span<const float> infer_output,
                   float confidence_threshold,
                   const LetterboxTransform& transform) const;

  FilterResult v10(std::span<const float> infer_output,
                   float confidence_threshold,
                   const LetterboxTransform& transform) const;

  FilterResult operator()(std::span<const float> infer_output,
                          float confidence_threshold,
                          const LetterboxTransform& transform) const;
};

class InferYOLOOrtImpl : public InferYOLO {
  std::unique_ptr<Ort::Session> session_;
  YOLOVersion version_;
  YOLOFilter filter_;
  Ort::Allocator allocator_;
  ONNXTensorElementDataType input_value_type_, output_value_type_;
  std::string input_name_, output_name_;
  cv::Size2i input_size_;
  std::vector<int64_t> input_shape_, output_shape_;
  std::vector<std::string> class_names_;

  struct Workspace;
  class Task;
  std::unique_ptr<Workspace> legacy_workspace_;
  std::array<std::unique_ptr<Workspace>, 2> idle_workspaces_;
  std::mutex run_mutex_;
  std::mutex session_mutex_;
  std::mutex pool_mutex_;

  std::unique_ptr<Workspace> AcquireWorkspace();
  void ReleaseWorkspace(std::unique_ptr<Workspace> workspace) noexcept;
  VSResult<void> PreProcess(Workspace& workspace, const cv::Mat& image,
                            float confidence_threshold);
  void Execute(Workspace& workspace);
  RunResult PostProcess(Workspace& workspace, float confidence_threshold);

  friend VSResult<std::unique_ptr<detail::FrameTask>> detail::MakeFrameTask(
      InferYOLO& model, const cv::Mat& image,
      float confidence_threshold) noexcept;

 public:
  InferYOLOOrtImpl(InferContextORT& ort_ctx,
                   std::unique_ptr<Ort::Session>&& session,
                   Ort::Allocator&& allocator, YOLOVersion version,
                   std::vector<std::string> class_names);
  ~InferYOLOOrtImpl() override;

  YOLOVersion version() const noexcept override;

  const std::vector<std::string>& class_names() const noexcept override;

  RunResult Run(const cv::Mat& image,
                float confidence_threshold) noexcept override;
};
}  // namespace vision_simple
