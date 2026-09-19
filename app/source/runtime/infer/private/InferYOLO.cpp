#include "InferYOLO.h"

#include <climits>
#include <limits>
#include <regex>

#include "InferValidation.hpp"
#include "LogContext.h"
#include "LogFacade.h"

using namespace std;
using namespace cv;
using namespace vision_simple;

namespace {
bool SupportedFloatType(ONNXTensorElementDataType type) noexcept {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

bool ValidOutputShape(YOLOVersion version, std::span<const int64_t> shape,
                      size_t classes) noexcept {
  if (shape.size() != 3 || shape[0] != 1 || classes == 0 || classes > INT_MAX)
    return false;
  if (version == YOLOVersion::kV10)
    return shape[1] >= 0 && shape[1] <= INT_MAX && shape[2] == 6;
  if (version == YOLOVersion::kV11)
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
          !SupportedFloatType(input_tensor.GetElementType()) ||
          !SupportedFloatType(output_tensor.GetElementType()))
        return std::unexpected(
            VisionSimpleError{VisionSimpleErrorCode::kModelError,
                              "Unsupported YOLO input shape or tensor type"});
      Ort::Allocator allocator{session, ort_ctx.env_memory_info()};
      auto names = session.GetModelMetadata().LookupCustomMetadataMapAllocated(
          "names", allocator);
      if (!names || !*names.get())
        return std::unexpected(
            VisionSimpleError{VisionSimpleErrorCode::kModelError,
                              "Missing YOLO class names metadata"});
      const std::string names_str(names.get());
      const std::regex reg{R"('([^']+)')"};
      std::vector<std::string> class_names;
      for (auto i =
               std::sregex_iterator(names_str.begin(), names_str.end(), reg);
           i != std::sregex_iterator(); ++i)
        class_names.emplace_back((*i)[1].str());
      if (!ValidOutputShape(version, output_tensor.GetShape(),
                            class_names.size()))
        return std::unexpected(
            VisionSimpleError{VisionSimpleErrorCode::kModelError,
                              "Unsupported YOLO output shape or class names"});
      return std::make_unique<InferYOLOOrtImpl>(
          ort_ctx, std::move(*session_opt), std::move(allocator), version,
          std::move(class_names));
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
                       std::vector<int64_t> shapes)
    : version_(version),
      class_names_(std::move(class_names)),
      shapes_(std::move(shapes)) {}

YOLOVersion YOLOFilter::version() const noexcept { return version_; }

std::vector<YOLOResult> YOLOFilter::ApplyNMS(
    const std::vector<YOLOResult>& detections, float iou_threshold) {
  std::vector<int> indices, class_ids;
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  boxes.reserve(detections.size());
  scores.reserve(detections.size());
  class_ids.reserve(detections.size());
  for (const auto& detection : detections) {
    boxes.push_back(detection.bbox);
    scores.push_back(detection.confidence);
    class_ids.push_back(detection.class_id);
  }
  cv::dnn::NMSBoxesBatched(boxes, scores, class_ids, 0.0f, iou_threshold,
                           indices);
  std::vector<YOLOResult> result;
  result.reserve(indices.size());
  for (int idx : indices) result.push_back(detections[idx]);
  return result;
}

YOLOFilter::FilterResult YOLOFilter::v11(
    std::span<const float> infer_output, float confidence_threshold,
    const LetterboxTransform& transform) const {
  if (!ValidOutputShape(YOLOVersion::kV11, shapes_, class_names_.size()) ||
      !ValidOutputLength(shapes_, infer_output.size()))
    return std::unexpected(
        VisionSimpleError{VisionSimpleErrorCode::kModelError,
                          "Invalid YOLO v11 output shape or length"});
  const size_t num_detections = static_cast<size_t>(shapes_[2]);
  std::vector<YOLOResult> detections;
  for (size_t d = 0; d < num_detections; ++d) {
    const float cx = infer_output[d], cy = infer_output[num_detections + d];
    const float width = infer_output[2 * num_detections + d];
    const float height = infer_output[3 * num_detections + d];
    if (!IsFinite(cx) || !IsFinite(cy) || !IsFinite(width) || !IsFinite(height))
      return std::unexpected(VisionSimpleError{
          VisionSimpleErrorCode::kModelError, "Non-finite YOLO coordinates"});
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
    if (confidence > confidence_threshold) {
      const auto box = VisionHelper::ScaleCoords(
          transform, {cx - width * 0.5f, cy - height * 0.5f, cx + width * 0.5f,
                      cy + height * 0.5f});
      if (box.width > 0 && box.height > 0)
        detections.emplace_back(class_id, box, confidence,
                                class_names_[class_id]);
    }
  }
  return YOLOFrameResult{ApplyNMS(detections, 0.3f)};
}

YOLOFilter::FilterResult YOLOFilter::v10(
    std::span<const float> infer_output, float confidence_threshold,
    const LetterboxTransform& transform) const {
  if (!ValidOutputShape(YOLOVersion::kV10, shapes_, class_names_.size()) ||
      !ValidOutputLength(shapes_, infer_output.size()))
    return std::unexpected(
        VisionSimpleError{VisionSimpleErrorCode::kModelError,
                          "YOLO v10 requires end-to-end [1,N,6] output"});
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
    std::span<const float> infer_output, float confidence_threshold,
    const LetterboxTransform& transform) const {
  if (version_ == YOLOVersion::kV10)
    return v10(infer_output, confidence_threshold, transform);
  if (version_ == YOLOVersion::kV11)
    return v11(infer_output, confidence_threshold, transform);
  return std::unexpected(VisionSimpleError{
      VisionSimpleErrorCode::kParameterError, "Unsupported YOLO version"});
}

cv::Mat& InferYOLOOrtImpl::PreProcess(const cv::Mat& image) {
  auto& dst_image = vision_helper_.Letterbox(image, input_size_, transform_);
  if (!dst_image.empty())
    vision_helper_.HWC2CHW_BGR2RGB<uint8_t>(dst_image, dst_image);
  return dst_image;
}

InferYOLOOrtImpl::InferYOLOOrtImpl(InferContextORT& ort_ctx,
                                   std::unique_ptr<Ort::Session>&& session,
                                   Ort::Allocator&& allocator,
                                   YOLOVersion version,
                                   std::vector<std::string> class_names)
    : session_(std::move(session)),
      version_(version),
      filter_(version, class_names,
              session_->GetOutputTypeInfo(0)
                  .GetTensorTypeAndShapeInfo()
                  .GetShape()),
      allocator_(std::move(allocator)),
      input_value_(nullptr),
      output_memory_info_(Ort::MemoryInfo::CreateCpu(
          ort_ctx.env_memory_info().GetAllocatorType(),
          ort_ctx.env_memory_info().GetMemoryType())),
      class_names_(std::move(class_names)) {
  auto input_info = session_->GetInputTypeInfo(0);
  auto tensor_info = input_info.GetTensorTypeAndShapeInfo();
  auto shape = tensor_info.GetShape();
  input_size_ = {static_cast<int>(shape[3]), static_cast<int>(shape[2])};
  input_name_ = session_->GetInputNameAllocated(0, allocator_).get();
  output_name_ = session_->GetOutputNameAllocated(0, allocator_).get();
  input_value_type_ = tensor_info.GetElementType();
  input_value_ = Ort::Value::CreateTensor(allocator_, shape.data(),
                                          shape.size(), input_value_type_);
  auto output_info = session_->GetOutputTypeInfo(0);
  auto output_tensor = output_info.GetTensorTypeAndShapeInfo();
  output_shape_ = output_tensor.GetShape();
  output_value_type_ = output_tensor.GetElementType();
}

YOLOVersion InferYOLOOrtImpl::version() const noexcept { return version_; }

const std::vector<std::string>& InferYOLOOrtImpl::class_names() const noexcept {
  return class_names_;
}

InferYOLO::RunResult InferYOLOOrtImpl::Run(
    const cv::Mat& image, float confidence_threshold) noexcept {
  try {
    if (auto valid = ValidateInferInput(image, confidence_threshold); !valid)
      return std::unexpected(std::move(valid.error()));
    auto total_timer = LogContext::ScopedTimer(
        "YOLO::Run::total", LogFacade::TimerCallback("yolo"));
    LogFacade::Info("yolo", "YOLO inference started");
    auto preprocess_timer =
        LogContext::ScopedTimer("YOLO::preprocess", nullptr);
    cv::Mat& chw = PreProcess(image);
    if (chw.empty())
      return std::unexpected(
          VisionSimpleError{VisionSimpleErrorCode::kParameterError,
                            "Image dimensions cannot be letterboxed"});
    const size_t elements = chw.total() * chw.channels();
    if (input_value_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      chw.convertTo(preprocessed_image_, CV_32F, 1.0 / 255);
      Cvt::cvt(
          std::span<const float>(preprocessed_image_.ptr<float>(), elements),
          input_value_.GetTensorMutableData<Ort::Float16_t>());
    } else {
      chw.convertTo(preprocessed_image_, CV_32F, 1.0 / 255);
      std::memcpy(input_value_.GetTensorMutableData<float>(),
                  preprocessed_image_.ptr<float>(), elements * sizeof(float));
    }
    Ort::IoBinding binding(*session_);
    binding.BindInput(input_name_.c_str(), input_value_);
    binding.BindOutput(output_name_.c_str(), output_memory_info_);
    LogFacade::Timing("yolo", "YOLO::preprocess",
                      preprocess_timer.elapsed_ms());
    auto infer_timer = LogContext::ScopedTimer("YOLO::infer", nullptr);
    Ort::RunOptions run_options;
    session_->Run(run_options, binding);
    LogFacade::Timing("yolo", "YOLO::infer", infer_timer.elapsed_ms());
    auto postprocess_timer =
        LogContext::ScopedTimer("YOLO::postprocess", nullptr);
    auto outputs = binding.GetOutputValues();
    if (outputs.size() != 1 || !outputs[0].IsTensor())
      return std::unexpected(VisionSimpleError{
          VisionSimpleErrorCode::kModelError, "Invalid YOLO output tensor"});
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
      output_fp32_cache_.resize(count);
      Cvt::cvt(std::span<const Ort::Float16_t>(
                   output.GetTensorData<Ort::Float16_t>(), count),
               output_fp32_cache_.data());
      output_data = output_fp32_cache_.data();
    } else {
      output_data = output.GetTensorData<float>();
    }
    auto result = filter_(std::span<const float>(output_data, count),
                          confidence_threshold, transform_);
    LogFacade::Timing("yolo", "YOLO::postprocess",
                      postprocess_timer.elapsed_ms());
    return result;
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
