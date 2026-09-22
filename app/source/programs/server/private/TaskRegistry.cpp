#include "TaskRegistry.h"

#include <turbobase64/turbob64.h>

#include <array>
#include <exception>
#include <format>
#include <functional>
#include <magic_enum.hpp>
#include <utility>

#include "IOUtil.h"
#include "LogFacade.h"

namespace vision_simple {
namespace {
constexpr std::string_view kYOLOTaskId = "yolo";
constexpr std::string_view kOCRTaskId = "ocr";

VSResult<std::reference_wrapper<const std::string>> RequiredFile(
    const ModelDefinition& definition, const char* key) {
  const auto found = definition.files.find(key);
  if (found == definition.files.end() || found->second.empty())
    return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                      std::format("missing {} resource for {} model:{}", key,
                                  definition.task, definition.name));
  return std::cref(found->second);
}

VSResult<RegisteredModel> LoadYOLO(InferContext& context,
                                   const ModelDefinition& definition,
                                   size_t device) {
  if (definition.task != kYOLOTaskId)
    return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                      "task does not match yolo loader");
  const auto version = magic_enum::enum_cast<YOLOVersion>(definition.version);
  if (!version)
    return MK_VSERROR(
        VisionSimpleErrorCode::kModelError,
        std::format("unknown yolo version: {}", definition.version));
  auto path = RequiredFile(definition, "model");
  if (!path) return std::unexpected(std::move(path.error()));
  auto data = ReadAll(path->get());
  if (!data) return std::unexpected(std::move(data.error()));
  auto loaded = InferYOLO::Create(context, data->span(), *version, device);
  if (!loaded)
    return MK_VSERROR(
        VisionSimpleErrorCode::kModelError,
        std::format("unable to create infer yolo model:{},message:{} ",
                    definition.name, loaded.error().message));
  if (!*loaded)
    return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                      "yolo factory returned no model");
  return RegisteredModel{std::move(*loaded)};
}

VSResult<RegisteredModel> LoadOCR(InferContext& context,
                                  const ModelDefinition& definition,
                                  size_t device) {
  if (definition.task != kOCRTaskId)
    return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                      "task does not match ocr loader");
  const auto version = magic_enum::enum_cast<OCRModelType>(definition.version);
  if (!version)
    return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                      std::format("unknown version:{}", definition.version));
  auto det = RequiredFile(definition, "det");
  if (!det) return std::unexpected(std::move(det.error()));
  auto rec = RequiredFile(definition, "rec");
  if (!rec) return std::unexpected(std::move(rec.error()));
  auto dictionary = RequiredFile(definition, "dictionary");
  if (!dictionary) return std::unexpected(std::move(dictionary.error()));
  auto loaded = InferOCR::Create(context, dictionary->get(), det->get(),
                                 rec->get(), *version, device);
  if (!loaded)
    return MK_VSERROR(
        VisionSimpleErrorCode::kModelError,
        std::format("unable to create infer ocr model:{},message:{} ",
                    definition.name, loaded.error().message));
  if (!*loaded)
    return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                      "ocr factory returned no model");
  return RegisteredModel{std::move(*loaded)};
}

template <YOLOTask Task>
VSResult<RegisteredModel> LoadYOLOTask(InferContext& context,
                                       const ModelDefinition& definition,
                                       size_t device) {
  if (definition.version != "kV11")
    return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                      "YOLO segmentation, pose and OBB require kV11 exports");
  auto path = RequiredFile(definition, "model");
  if (!path) return std::unexpected(std::move(path.error()));
  auto loaded = InferYOLOTask::Create(context, path->get(), Task, device);
  if (!loaded) return std::unexpected(std::move(loaded.error()));
  if (!*loaded)
    return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                      "YOLO task factory returned no model");
  return RegisteredModel{std::move(*loaded)};
}

constexpr std::array kTasks{
    TaskDescriptor{
        InferenceKind::kYOLO, kYOLOTaskId,
        "Detect objects in base64 images. Returns class_names and "
        "per-image class_id, confidence and bbox [x,y,width,height].",
        LoadYOLO},
    TaskDescriptor{
        InferenceKind::kOCR, kOCRTaskId,
        "Recognize text in base64 images. Returns per-image text "
        "lines, confidence and bbox [x,y,width,height]; no free-form "
        "language generation.",
        LoadOCR},
    TaskDescriptor{InferenceKind::kSegmentation, "seg",
                   "Segment objects in base64 images. Returns class_names and "
                   "per-image class_id, confidence, bbox [x,y,width,height], "
                   "and mask_png_base64: a binary PNG cropped to bbox.",
                   LoadYOLOTask<YOLOTask::kSegmentation>},
    TaskDescriptor{InferenceKind::kPose, "pose",
                   "Estimate poses in base64 images. Returns class_names and "
                   "per-image class_id, confidence, bbox [x,y,width,height], "
                   "and keypoints with original-pixel x, y and confidence.",
                   LoadYOLOTask<YOLOTask::kPose>},
    TaskDescriptor{
        InferenceKind::kOBB, "obb",
        "Detect oriented objects in base64 images. Returns "
        "class_names and per-image class_id, confidence, four ordered "
        "original-pixel corners and first-edge angle in radians.",
        LoadYOLOTask<YOLOTask::kOBB>}};

ServiceError PipelineError(const PipelineFailure& failure) noexcept {
  LogFacade::Error("inference", failure.cause.message);
  switch (failure.kind) {
    case PipelineFailureKind::kInference:
      return {ServiceFailure::kInference, failure.image_index};
    case PipelineFailureKind::kInvalidRequest:
      return {ServiceFailure::kInvalidRequest, {}};
    case PipelineFailureKind::kBusy:
      return {ServiceFailure::kBusy, {}};
    case PipelineFailureKind::kCancelled:
      return {ServiceFailure::kCancelled, {}};
    case PipelineFailureKind::kTimedOut:
      return {ServiceFailure::kTimedOut, {}};
    case PipelineFailureKind::kClosed:
      return {ServiceFailure::kClosed, {}};
  }
  return {ServiceFailure::kInternal, {}};
}

InferYOLOResponse PackResponse(const InferYOLO& model,
                               const std::vector<YOLOFrameResult>& frames) {
  InferYOLOResponse body;
  body.class_names.assign(model.class_names().cbegin(),
                          model.class_names().cend());
  body.results.reserve(frames.size());
  for (const auto& frame : frames) {
    std::vector<YOLODetectedObject> objects;
    objects.reserve(frame.results.size());
    for (const auto& result : frame.results) {
      const auto& box = result.bbox;
      objects.emplace_back(
          YOLODetectedObject{result.class_id,
                             result.confidence,
                             {box.x, box.y, box.width, box.height}});
    }
    body.results.emplace_back(std::move(objects));
  }
  return body;
}

InferOCRResponse PackResponse(const InferOCR&,
                              std::vector<OCRFrameResult>& frames) {
  InferOCRResponse body;
  body.results.reserve(frames.size());
  for (auto& frame : frames) {
    std::vector<OCRLine> lines;
    lines.reserve(frame.results.size());
    for (auto& result : frame.results) {
      const auto& box = result.rect;
      lines.emplace_back(OCRLine{std::move(result.line),
                                 result.confidence,
                                 {box.x, box.y, box.width, box.height}});
    }
    body.results.emplace_back(std::move(lines));
  }
  return body;
}

ServiceResult<InferencePayload> PackResponse(
    const InferYOLOTask& model, std::vector<YOLOTaskFrameResult>& frames) {
  if (model.task() == YOLOTask::kSegmentation) {
    InferSegmentationResponse body{model.class_names(), {}};
    body.results.reserve(frames.size());
    std::vector<uint8_t> png;
    for (const auto& frame : frames) {
      const auto& results = std::get<YOLOSegmentationFrame>(frame).results;
      auto& objects = body.results.emplace_back();
      objects.reserve(results.size());
      for (const auto& result : results) {
        if (!cv::imencode(".png", result.mask, png) || png.empty())
          return std::unexpected(ServiceError{ServiceFailure::kInternal, {}});
        std::string encoded(tb64enclen(png.size()), '\0');
        if (tb64enc(png.data(), png.size(),
                    reinterpret_cast<unsigned char*>(encoded.data())) !=
            encoded.size())
          return std::unexpected(ServiceError{ServiceFailure::kInternal, {}});
        const auto& box = result.bbox;
        objects.push_back({result.class_id,
                           result.confidence,
                           {box.x, box.y, box.width, box.height},
                           std::move(encoded)});
      }
    }
    return InferencePayload{std::move(body)};
  }
  if (model.task() == YOLOTask::kPose) {
    InferPoseResponse body{model.class_names(), {}};
    body.results.reserve(frames.size());
    for (const auto& frame : frames) {
      const auto& results = std::get<YOLOPoseFrame>(frame).results;
      auto& objects = body.results.emplace_back();
      objects.reserve(results.size());
      for (const auto& result : results) {
        const auto& box = result.bbox;
        auto& object = objects.emplace_back(
            PoseObject{result.class_id,
                       result.confidence,
                       {box.x, box.y, box.width, box.height},
                       {}});
        object.keypoints.reserve(result.keypoints.size());
        for (const auto& point : result.keypoints)
          object.keypoints.push_back({point.x, point.y, point.confidence});
      }
    }
    return InferencePayload{std::move(body)};
  }
  if (model.task() == YOLOTask::kOBB) {
    InferOBBResponse body{model.class_names(), {}};
    body.results.reserve(frames.size());
    for (const auto& frame : frames) {
      const auto& results = std::get<YOLOOBBFrame>(frame).results;
      auto& objects = body.results.emplace_back();
      objects.reserve(results.size());
      for (const auto& result : results) {
        auto& object = objects.emplace_back(RotatedObject{
            result.class_id, result.confidence, {}, result.angle});
        for (size_t i = 0; i < result.corners.size(); ++i) {
          object.corners[i][0] = result.corners[i].x;
          object.corners[i][1] = result.corners[i].y;
        }
      }
    }
    return InferencePayload{std::move(body)};
  }
  return std::unexpected(ServiceError{ServiceFailure::kInternal, {}});
}
}  // namespace

std::span<const TaskDescriptor> RegisteredTasks() noexcept { return kTasks; }

const TaskDescriptor* FindTask(InferenceKind kind) noexcept {
  for (const auto& task : kTasks)
    if (task.kind == kind) return &task;
  return nullptr;
}

const TaskDescriptor* FindTask(std::string_view id) noexcept {
  for (const auto& task : kTasks)
    if (task.id == id) return &task;
  return nullptr;
}

ServiceResult<InferencePayload> RunRegisteredTask(
    RegisteredModel& model, InferPipeline& pipeline,
    std::span<const cv::Mat> images, PipelineControl control) noexcept {
  try {
    return std::visit(
        [&](auto& typed_model) -> ServiceResult<InferencePayload> {
          if (!typed_model) {
            LogFacade::Error("inference", "registered model is null");
            return std::unexpected(ServiceError{ServiceFailure::kInternal, {}});
          }
          auto batch = pipeline.Run(*typed_model, images, 0.125f, control);
          if (!batch) return std::unexpected(PipelineError(batch.error()));
          if constexpr (std::is_same_v<
                            std::remove_reference_t<decltype(*typed_model)>,
                            InferYOLOTask>) {
            return PackResponse(*typed_model, *batch);
          } else {
            return InferencePayload{PackResponse(*typed_model, *batch)};
          }
        },
        model);
  } catch (const std::exception& error) {
    LogFacade::Error("inference", error.what());
  } catch (...) {
    LogFacade::Error("inference", "Unknown inference failure");
  }
  return std::unexpected(ServiceError{ServiceFailure::kInternal, {}});
}
}  // namespace vision_simple
