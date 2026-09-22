#pragma once

#include <memory>
#include <span>
#include <string_view>
#include <variant>

#include "InferenceService.h"
#include "VisionSimpleConfig.h"

namespace vision_simple {
using RegisteredModel =
    std::variant<std::unique_ptr<InferYOLO>, std::unique_ptr<InferOCR>,
                 std::unique_ptr<InferYOLOTask>>;
using InferencePayload = decltype(InferenceResponse::payload);

struct TaskDescriptor {
  InferenceKind kind;
  std::string_view id;
  std::string_view description;
  VSResult<RegisteredModel> (*load)(InferContext&, const ModelDefinition&,
                                    size_t device);
};

std::span<const TaskDescriptor> RegisteredTasks() noexcept;
const TaskDescriptor* FindTask(InferenceKind kind) noexcept;
const TaskDescriptor* FindTask(std::string_view id) noexcept;
ServiceResult<InferencePayload> RunRegisteredTask(
    RegisteredModel& model, InferPipeline& pipeline,
    std::span<const cv::Mat> images, PipelineControl control) noexcept;
}  // namespace vision_simple
