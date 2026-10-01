#include "InferYOLO.h"

#include <algorithm>
#include <climits>
#include <limits>

#include "InferValidation.hpp"
#include "LogContext.h"
#include "LogFacade.h"
#include "YOLOMetadata.hpp"

using namespace std;
using namespace cv;
using namespace vision_simple;

namespace {
bool SupportedFloatType(ONNXTensorElementDataType type) noexcept {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

bool ValidOutputShape(YOLODetectionLayout layout, std::span<const int64_t> shape,
                      size_t classes) noexcept {
  if (shape.size() != 3 || shape[0] != 1 || classes == 0 || classes > INT_MAX)
    return false;
  if (shape[1] < 0 || shape[2] < 0 ||
      (shape[2] != 0 &&
       static_cast<uint64_t>(shape[1]) >
           std::numeric_limits<size_t>::max() / sizeof(float) /
               static_cast<uint64_t>(shape[2])))
    return false;
  if (layout == YOLODetectionLayout::kEndToEnd)
    return shape[1] >= 0 && shape[1] <= INT_MAX && shape[2] == 6;
  if (layout == YOLODetectionLayout::kRaw)
    return shape[1] == static_cast<int64_t>(classes) + 4 && shape[2] >= 0 &&
           shape[2] <= INT_MAX;
  return false;
}

bool ValidOutputLength(std::span<const int64_t> shape, size_t length) noexcept {
  const auto rows = static_cast<size_t>(shape[1]);
  const auto cols = static_cast<size_t>(shape[2]);
  return (cols == 0 || rows <= std::numeric_limits<size_t>::max() / cols) &&
         rows * cols == length;
}

using InferYOLOFactory = std::function<InferYOLO::CreateResult(
    InferContext&, std::span<uint8_t>, YOLOVersion, size_t)>;

std::map<InferFramework, InferYOLOFactory> infer_yolo_factories{std::make_pair(
    InferFramework::kONNXRUNTIME,
    [](InferContext& context, std::span<uint8_t> data, YOLOVersion version,
       size_t device_id) -> InferYOLO::CreateResult {
      auto& ort_ctx = dynamic_cast<InferContextORT&>(context);
      auto session_opt = ort_ctx.CreateSession(data, device_id);
      if (!session_opt) return std::unexpected{std::move(session_opt.error())};
      auto& session = **session_opt;
      if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
        return std::unexpected(
            VisionSimpleError{VisionSimpleErrorCode::kModelError,
                              "YOLO requires one input and one output"});
      auto input_info = session.GetInputTypeInfo(0);
      auto output_info = session.GetOutputTypeInfo(0);
      if (input_info.GetONNXType() != ONNX_TYPE_TENSOR ||
          output_info.GetONNXType() != ONNX_TYPE_TENSOR)
        return std::unexpected(
            VisionSimpleError{VisionSimpleErrorCode::kModelError,
                              "YOLO requires tensor input and output"});
      auto input_tensor = input_info.GetTensorTypeAndShapeInfo();
      auto output_tensor = output_info.GetTensorTypeAndShapeInfo();
      auto input_shape = input_tensor.GetShape();
      if (input_shape.size() != 4 || input_shape[0] != 1 ||
          input_shape[1] != 3 || input_shape[2] <= 0 ||
          input_shape[2] > INT_MAX || input_shape[3] <= 0 ||
          input_shape[3] > INT_MAX ||
          static_cast<uint64_t>(input_shape[2]) >
              std::numeric_limits<size_t>::max() / sizeof(float) / 3 /
                  static_cast<uint64_t>(input_shape[3]) ||
          !SupportedFloatType(input_tensor.GetElementType()) ||
          !SupportedFloatType(output_tensor.GetElementType()))
        return std::unexpected(
            VisionSimpleError{VisionSimpleErrorCode::kModelError,
                              "Unsupported YOLO input shape or tensor type"});
      Ort::Allocator allocator{session, ort_ctx.env_memory_info()};
      std::vector<std::string> class_names;
      YOLODetectionLayout layout =
          version == YOLOVersion::kV10 ? YOLODetectionLayout::kEndToEnd
          : version == YOLOVersion::kV11 ? YOLODetectionLayout::kRaw
                                        : YOLODetectionLayout::kUnspecified;
      // Release borrowed metadata before construction can take the allocator.
      {
        auto names = session.GetModelMetadata().LookupCustomMetadataMapAllocated(
            "names", allocator);
        if (!names || !*names.get())
          return std::unexpected(
              VisionSimpleError{VisionSimpleErrorCode::kModelError,
                                "Missing YOLO class names metadata"});
        if (!vision_simple::detail::ParseYOLOClassNames(names.get(), class_names))
          return std::unexpected(
              VisionSimpleError{VisionSimpleErrorCode::kModelError,
                                "Invalid YOLO class names metadata"});
        if (version == YOLOVersion::kV26) {
          auto metadata = session.GetModelMetadata();
          auto task = metadata.LookupCustomMetadataMapAllocated("task", allocator);
          auto args = metadata.LookupCustomMetadataMapAllocated("args", allocator);
          auto end2end =
              metadata.LookupCustomMetadataMapAllocated("end2end", allocator);
          auto nms = metadata.LookupCustomMetadataMapAllocated("nms", allocator);
          bool end_to_end = false;
          if (!task || std::string_view(task.get()) != "detect" ||
              !vision_simple::detail::ParseYOLO26Export(args ? args.get() : "",
                                        end2end ? end2end.get() : "",
                                        nms ? nms.get() : "", end_to_end))
            return std::unexpected(VisionSimpleError{
                VisionSimpleErrorCode::kModelError,
                "YOLO26 requires detect task and explicit supported export metadata"});
          layout = end_to_end ? YOLODetectionLayout::kEndToEnd
                              : YOLODetectionLayout::kRaw;
        }
        if (!ValidOutputShape(layout, output_tensor.GetShape(),
                              class_names.size()))
          return std::unexpected(
              VisionSimpleError{VisionSimpleErrorCode::kModelError,
                                "Unsupported YOLO output shape or class names"});
      }
      return std::make_unique<InferYOLOOrtImpl>(
          ort_ctx, std::move(*session_opt), std::move(allocator), version,
          std::move(class_names), layout);
    })};
}  // namespace

InferYOLO::CreateResult InferYOLO::Create(InferContext& context,
                                          std::span<uint8_t> data,
                                          YOLOVersion version,
                                          size_t device_id) noexcept {
  try {
    const auto factory = infer_yolo_factories.find(context.framework());
    if (factory == infer_yolo_factories.end())
      return std::unexpected(
          VisionSimpleError{VisionSimpleErrorCode::kParameterError,
                            "Unsupported YOLO framework"});
    return factory->second(context, data, version, device_id);
  } catch (const cv::Exception& e) {
    return std::unexpected(
        VisionSimpleError{VisionSimpleErrorCode::kRuntimeError, e.what()});
  } catch (const Ort::Exception& e) {
    return std::unexpected(
        VisionSimpleError{VisionSimpleErrorCode::kModelError, e.what()});
  } catch (const std::exception& e) {
    return std::unexpected(
        VisionSimpleError{VisionSimpleErrorCode::kRuntimeError, e.what()});
  }
}

YOLOFilter::YOLOFilter(YOLOVersion version,
                       std::vector<std::string> class_names,
                       std::vector<int64_t> shapes, YOLODetectionLayout layout)
    : version_(version),
      layout_(version == YOLOVersion::kV10 ? YOLODetectionLayout::kEndToEnd
              : version == YOLOVersion::kV11 ? YOLODetectionLayout::kRaw
                                             : layout),
      class_names_(std::move(class_names)),
      shapes_(std::move(shapes)) {}

YOLOVersion YOLOFilter::version() const noexcept { return version_; }

YOLOFilter::FilterResult YOLOFilter::DecodeRaw(
    std::span<const float> infer_output, float confidence_threshold, float nms_iou,
    const LetterboxTransform& transform) const {
  if (!ValidOutputShape(YOLODetectionLayout::kRaw, shapes_, class_names_.size()) ||
      !ValidOutputLength(shapes_, infer_output.size()))
    return std::unexpected(
        VisionSimpleError{VisionSimpleErrorCode::kModelError,
                          "Invalid YOLO raw output shape or length"});
  const size_t num_detections = static_cast<size_t>(shapes_[2]);
  struct Candidate {
    cv::Vec4f xyxy;
    double area;
    float confidence;
    int class_id;
    size_t index;
    bool suppressed = false;
  };
  std::vector<Candidate> candidates;
  for (size_t d = 0; d < num_detections; ++d) {
    const float cx = infer_output[d], cy = infer_output[num_detections + d];
    const float width = infer_output[2 * num_detections + d];
    const float height = infer_output[3 * num_detections + d];
    if (!IsFinite(cx) || !IsFinite(cy) || !IsFinite(width) || !IsFinite(height))
      return std::unexpected(VisionSimpleError{
          VisionSimpleErrorCode::kModelError, "Non-finite YOLO coordinates"});
    const cv::Vec4f xyxy{cx - width * 0.5f, cy - height * 0.5f,
                         cx + width * 0.5f, cy + height * 0.5f};
    if (!IsFinite(xyxy[0]) || !IsFinite(xyxy[1]) ||
        !IsFinite(xyxy[2]) || !IsFinite(xyxy[3]))
      return std::unexpected(VisionSimpleError{
          VisionSimpleErrorCode::kModelError, "Overflowing YOLO coordinates"});
    int class_id = 0;
    float confidence = infer_output[4 * num_detections + d];
    for (size_t c = 0; c < class_names_.size(); ++c) {
      const float score = infer_output[(4 + c) * num_detections + d];
      if (!IsFinite(score))
        return std::unexpected(VisionSimpleError{
            VisionSimpleErrorCode::kModelError, "Non-finite YOLO class score"});
      if (score > confidence) {
        confidence = score;
        class_id = static_cast<int>(c);
      }
    }
    if (confidence > confidence_threshold && xyxy[2] > xyxy[0] &&
        xyxy[3] > xyxy[1]) {
      const double area = (static_cast<double>(xyxy[2]) - xyxy[0]) *
                          (static_cast<double>(xyxy[3]) - xyxy[1]);
      candidates.push_back({xyxy, area, confidence, class_id, d});
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              return a.confidence != b.confidence
                         ? a.confidence > b.confidence
                         : a.index < b.index;
            });
  std::vector<YOLOResult> detections;
  for (size_t i = 0; i < candidates.size(); ++i) {
    const auto& candidate = candidates[i];
    if (candidate.suppressed) continue;
    // Compare unclipped floating model-space boxes only within the class.
    // Explicit class isolation also avoids offsets failing for negative boxes.
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      auto& other = candidates[j];
      if (other.suppressed || candidate.class_id != other.class_id) continue;
      const double width = std::max(
          0.0, static_cast<double>(std::min(candidate.xyxy[2], other.xyxy[2])) -
                   std::max(candidate.xyxy[0], other.xyxy[0]));
      const double height = std::max(
          0.0, static_cast<double>(std::min(candidate.xyxy[3], other.xyxy[3])) -
                   std::max(candidate.xyxy[1], other.xyxy[1]));
      const double intersection = width * height;
      const double iou = intersection /
                         (candidate.area + other.area - intersection);
      if (iou > nms_iou) other.suppressed = true;
    }
    const auto box = VisionHelper::ScaleCoords(transform, candidate.xyxy);
    if (box.width > 0 && box.height > 0)
      detections.emplace_back(candidate.class_id, box, candidate.confidence,
                              class_names_[candidate.class_id]);
  }
  return YOLOFrameResult{std::move(detections)};
}

YOLOFilter::FilterResult YOLOFilter::DecodeEndToEnd(
    std::span<const float> infer_output, float confidence_threshold,
    const LetterboxTransform& transform) const {
  if (!ValidOutputShape(YOLODetectionLayout::kEndToEnd, shapes_, class_names_.size()) ||
      !ValidOutputLength(shapes_, infer_output.size()))
    return std::unexpected(
        VisionSimpleError{VisionSimpleErrorCode::kModelError,
                          "YOLO requires end-to-end [1,N,6] output"});
  std::vector<YOLOResult> detections;
  for (size_t i = 0; i < infer_output.size(); i += 6) {
    for (size_t j = 0; j < 6; ++j)
      if (!IsFinite(infer_output[i + j]))
        return std::unexpected(VisionSimpleError{
            VisionSimpleErrorCode::kModelError, "Non-finite YOLO detection"});
    const float class_value = infer_output[i + 5];
    if (class_value < 0 ||
        static_cast<double>(class_value) >= class_names_.size() ||
        std::trunc(class_value) != class_value)
      return std::unexpected(VisionSimpleError{
          VisionSimpleErrorCode::kModelError, "Invalid YOLO class index"});
    const int class_id = static_cast<int>(class_value);
    const float confidence = infer_output[i + 4];
    if (confidence >= confidence_threshold) {
      const auto box = VisionHelper::ScaleCoords(
          transform, {infer_output[i], infer_output[i + 1], infer_output[i + 2],
                      infer_output[i + 3]});
      if (box.width > 0 && box.height > 0)
        detections.emplace_back(class_id, box, confidence,
                                class_names_[class_id]);
    }
  }
  return YOLOFrameResult{std::move(detections)};
}

YOLOFilter::FilterResult YOLOFilter::operator()(
    std::span<const float> infer_output, YOLOInferenceOptions options,
    const LetterboxTransform& transform) const {
  if (!options.IsValid())
    return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                      "Detector controls must be finite and within [0,1]");
  const float confidence_threshold = options.confidence.value_or(.125f);
  if (version_ != YOLOVersion::kV10 && version_ != YOLOVersion::kV11 &&
      version_ != YOLOVersion::kV26)
    return std::unexpected(VisionSimpleError{
        VisionSimpleErrorCode::kParameterError, "Unsupported YOLO version"});
  if (layout_ == YOLODetectionLayout::kEndToEnd)
    return DecodeEndToEnd(infer_output, confidence_threshold, transform);
  if (layout_ == YOLODetectionLayout::kRaw)
    return DecodeRaw(infer_output, confidence_threshold,
                     options.nms_iou.value_or(.3f), transform);
  return std::unexpected(VisionSimpleError{
      VisionSimpleErrorCode::kModelError, "Missing YOLO detection layout"});
}

struct InferYOLOOrtImpl::Workspace {
  Ort::Value input{nullptr}, output{nullptr};
  Ort::IoBinding binding;
  VisionHelper helper;
  cv::Mat preprocessed;
  LetterboxTransform transform;
  std::vector<float> output_fp32;

  explicit Workspace(InferYOLOOrtImpl& model) : binding(*model.session_) {
    input = Ort::Value::CreateTensor(
        model.allocator_, model.input_shape_.data(), model.input_shape_.size(),
        model.input_value_type_);
    output = Ort::Value::CreateTensor(
        model.allocator_, model.output_shape_.data(),
        model.output_shape_.size(), model.output_value_type_);
    binding.BindOutput(model.output_name_.c_str(), output);
  }
};

class InferYOLOOrtImpl::Task final : public detail::FrameTask {
  InferYOLOOrtImpl& model_;
  const cv::Mat& image_;
  YOLOInferenceOptions options_;
  std::unique_ptr<Workspace> workspace_;
  detail::PipelineLane lane_ = detail::PipelineLane::kPreprocess;
  YOLOFrameResult result_;

 public:
  Task(InferYOLOOrtImpl& model, const cv::Mat& image, YOLOInferenceOptions options,
       std::unique_ptr<Workspace> workspace)
      : model_(model),
        image_(image),
        options_(options),
        workspace_(std::move(workspace)) {}

  ~Task() override { model_.ReleaseWorkspace(std::move(workspace_)); }

  VSResult<std::optional<detail::PipelineLane>> Advance() noexcept override {
    try {
      if (lane_ == detail::PipelineLane::kPreprocess) {
        auto prepared = model_.PreProcess(*workspace_, image_, options_);
        if (!prepared) return std::unexpected(std::move(prepared.error()));
        lane_ = detail::PipelineLane::kInference;
        return lane_;
      }
      if (lane_ == detail::PipelineLane::kInference) {
        model_.Execute(*workspace_);
        lane_ = detail::PipelineLane::kPostprocess;
        return lane_;
      }
      auto result = model_.PostProcess(*workspace_, options_);
      if (!result) return std::unexpected(std::move(result.error()));
      result_ = std::move(*result);
      return std::nullopt;
    } catch (const cv::Exception& e) {
      return std::unexpected(
          VisionSimpleError{VisionSimpleErrorCode::kRuntimeError, e.what()});
    } catch (const Ort::Exception& e) {
      return std::unexpected(
          VisionSimpleError{VisionSimpleErrorCode::kRuntimeError, e.what()});
    } catch (const std::exception& e) {
      return std::unexpected(
          VisionSimpleError{VisionSimpleErrorCode::kRuntimeError, e.what()});
    }
  }

  detail::FrameResult TakeResult() noexcept override {
    return std::move(result_);
  }
};

InferYOLOOrtImpl::InferYOLOOrtImpl(InferContextORT& ort_ctx,
                                   std::unique_ptr<Ort::Session>&& session,
                                   Ort::Allocator&& allocator,
                                   YOLOVersion version,
                                   std::vector<std::string> class_names,
                                   YOLODetectionLayout layout)
    : session_(std::move(session)),
      version_(version),
      filter_(version, class_names,
              session_->GetOutputTypeInfo(0)
                  .GetTensorTypeAndShapeInfo()
                  .GetShape(), layout),
      allocator_(std::move(allocator)),
      class_names_(std::move(class_names)) {
  auto input_info = session_->GetInputTypeInfo(0);
  auto tensor_info = input_info.GetTensorTypeAndShapeInfo();
  input_shape_ = tensor_info.GetShape();
  input_size_ = {static_cast<int>(input_shape_[3]),
                 static_cast<int>(input_shape_[2])};
  input_name_ = session_->GetInputNameAllocated(0, allocator_).get();
  output_name_ = session_->GetOutputNameAllocated(0, allocator_).get();
  input_value_type_ = tensor_info.GetElementType();
  auto output_info = session_->GetOutputTypeInfo(0);
  auto output_tensor = output_info.GetTensorTypeAndShapeInfo();
  output_shape_ = output_tensor.GetShape();
  output_value_type_ = output_tensor.GetElementType();
  legacy_workspace_ = std::make_unique<Workspace>(*this);
}

InferYOLOOrtImpl::~InferYOLOOrtImpl() = default;

std::unique_ptr<InferYOLOOrtImpl::Workspace>
InferYOLOOrtImpl::AcquireWorkspace() {
  {
    const std::lock_guard lock(pool_mutex_);
    for (auto& idle : idle_workspaces_)
      if (idle) return std::move(idle);
  }
  const std::lock_guard lock(session_mutex_);
  return std::make_unique<Workspace>(*this);
}

void InferYOLOOrtImpl::ReleaseWorkspace(
    std::unique_ptr<Workspace> workspace) noexcept {
  const std::lock_guard lock(pool_mutex_);
  for (auto& idle : idle_workspaces_) {
    if (!idle) {
      idle = std::move(workspace);
      return;
    }
  }
}

VSResult<std::unique_ptr<vision_simple::detail::FrameTask>>
vision_simple::detail::MakeFrameTask(InferYOLO& model, const cv::Mat& image,
                                     YOLOInferenceOptions options) noexcept {
  try {
    auto* ort = dynamic_cast<InferYOLOOrtImpl*>(&model);
    if (!ort)
      return std::unexpected(
          VisionSimpleError{VisionSimpleErrorCode::kUnimplementedError,
                            "Unsupported YOLO backend for staged inference"});
    return std::make_unique<InferYOLOOrtImpl::Task>(
        *ort, image, options, ort->AcquireWorkspace());
  } catch (const cv::Exception& e) {
    return std::unexpected(
        VisionSimpleError{VisionSimpleErrorCode::kRuntimeError, e.what()});
  } catch (const Ort::Exception& e) {
    return std::unexpected(
        VisionSimpleError{VisionSimpleErrorCode::kRuntimeError, e.what()});
  } catch (const std::exception& e) {
    return std::unexpected(
        VisionSimpleError{VisionSimpleErrorCode::kRuntimeError, e.what()});
  }
}

YOLOVersion InferYOLOOrtImpl::version() const noexcept { return version_; }

const std::vector<std::string>& InferYOLOOrtImpl::class_names() const noexcept {
  return class_names_;
}

VSResult<void> InferYOLOOrtImpl::PreProcess(Workspace& workspace,
                                            const cv::Mat& image,
                                            YOLOInferenceOptions options) {
  if (auto valid = ValidateInferInput(image, options); !valid)
    return std::unexpected(std::move(valid.error()));
  auto timer = LogContext::ScopedTimer("YOLO::preprocess", nullptr);
  auto& chw =
      workspace.helper.Letterbox(image, input_size_, workspace.transform);
  if (chw.empty())
    return std::unexpected(
        VisionSimpleError{VisionSimpleErrorCode::kParameterError,
                          "Image dimensions cannot be letterboxed"});
  workspace.helper.HWC2CHW_BGR2RGB<uint8_t>(chw, chw);
  const size_t elements = chw.total() * chw.channels();
  if (input_value_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    chw.convertTo(workspace.preprocessed, CV_32F, 1.0 / 255);
    Cvt::cvt(
        std::span<const float>(workspace.preprocessed.ptr<float>(), elements),
        workspace.input.GetTensorMutableData<Ort::Float16_t>());
  } else {
    cv::Mat input_image(chw.size(), CV_32FC3,
                        workspace.input.GetTensorMutableData<float>());
    chw.convertTo(input_image, CV_32F, 1.0 / 255);
  }
  LogFacade::Timing("yolo", "YOLO::preprocess", timer.elapsed_ms());
  return {};
}

void InferYOLOOrtImpl::Execute(Workspace& workspace) {
  const std::lock_guard lock(session_mutex_);
  auto timer = LogContext::ScopedTimer("YOLO::infer", nullptr);
  // Rebind after writing: device providers may copy CPU inputs at bind time.
  // Binding and execution share the gate with the legacy direct Run path.
  workspace.binding.BindInput(input_name_.c_str(), workspace.input);
  Ort::RunOptions run_options;
  session_->Run(run_options, workspace.binding);
  LogFacade::Timing("yolo", "YOLO::infer", timer.elapsed_ms());
}

InferYOLO::RunResult InferYOLOOrtImpl::PostProcess(Workspace& workspace,
                                                   YOLOInferenceOptions options) {
  auto timer = LogContext::ScopedTimer("YOLO::postprocess", nullptr);
  auto outputs = workspace.binding.GetOutputValues();
  if (outputs.size() != 1 || !outputs[0].IsTensor())
    return std::unexpected(VisionSimpleError{VisionSimpleErrorCode::kModelError,
                                             "Invalid YOLO output tensor"});
  auto& output = outputs[0];
  auto info = output.GetTensorTypeAndShapeInfo();
  if (info.GetShape() != output_shape_ ||
      info.GetElementType() != output_value_type_)
    return std::unexpected(
        VisionSimpleError{VisionSimpleErrorCode::kModelError,
                          "YOLO output does not match model metadata"});
  const size_t count = info.GetElementCount();
  const float* output_data;
  if (output_value_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    workspace.output_fp32.resize(count);
    Cvt::cvt(std::span<const Ort::Float16_t>(
                 output.GetTensorData<Ort::Float16_t>(), count),
             workspace.output_fp32.data());
    output_data = workspace.output_fp32.data();
  } else {
    output_data = output.GetTensorData<float>();
  }
  auto result = filter_(std::span<const float>(output_data, count),
                        options, workspace.transform);
  LogFacade::Timing("yolo", "YOLO::postprocess", timer.elapsed_ms());
  return result;
}

InferYOLO::RunResult InferYOLOOrtImpl::Run(
    const cv::Mat& image, YOLOInferenceOptions options) noexcept {
  try {
    const std::lock_guard lock(run_mutex_);
    auto total_timer = LogContext::ScopedTimer(
        "YOLO::Run::total", LogFacade::TimerCallback("yolo"));
    LogFacade::Info("yolo", "YOLO inference started");
    auto prepared = PreProcess(*legacy_workspace_, image, options);
    if (!prepared) return std::unexpected(std::move(prepared.error()));
    Execute(*legacy_workspace_);
    return PostProcess(*legacy_workspace_, options);
  } catch (const cv::Exception& e) {
    return std::unexpected(
        VisionSimpleError{VisionSimpleErrorCode::kRuntimeError, e.what()});
  } catch (const Ort::Exception& e) {
    return std::unexpected(
        VisionSimpleError{VisionSimpleErrorCode::kRuntimeError, e.what()});
  } catch (const std::exception& e) {
    return std::unexpected(
        VisionSimpleError{VisionSimpleErrorCode::kRuntimeError, e.what()});
  }
}
