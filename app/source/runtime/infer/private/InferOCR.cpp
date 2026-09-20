#include "InferOCR.h"

#include <algorithm>
#include <limits>

#include "InferORT.h"
#include "InferValidation.hpp"
#include "LogContext.h"
#include "LogFacade.h"
#include "OCRCTC.hpp"
#include "VisionHelper.hpp"

vision_simple::InferOCR::CreateResult vision_simple::InferOCR::Create(
    InferContext& context, std::map<int, std::string> char_dict,
    std::span<uint8_t> det_data, std::span<uint8_t> rec_data,
    OCRModelType model_type, size_t device_id) noexcept {
  try {
    auto* ort_ctx = dynamic_cast<InferContextORT*>(&context);
    if (!ort_ctx) {
      return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                        "OCR requires an ONNX Runtime context");
    }
    auto det = ort_ctx->CreateSession(det_data, device_id);
    if (!det) return std::unexpected(std::move(det.error()));
    auto rec = ort_ctx->CreateSession(rec_data, device_id);
    if (!rec) return std::unexpected(std::move(rec.error()));
    if ((*det)->GetInputCount() != 1 || (*det)->GetOutputCount() != 1 ||
        (*rec)->GetInputCount() != 1 || (*rec)->GetOutputCount() != 1) {
      return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                        "OCR requires one input and one output per model");
    }
    return std::make_unique<InferOCROrtPaddleImpl>(
        *ort_ctx, model_type, std::move(char_dict), std::move(*det),
        std::move(*rec));
  } catch (const cv::Exception& error) {
    return MK_VSERROR(VisionSimpleErrorCode::kModelError, error.what());
  } catch (const Ort::Exception& error) {
    return MK_VSERROR(VisionSimpleErrorCode::kModelError, error.what());
  } catch (const std::exception& error) {
    return MK_VSERROR(VisionSimpleErrorCode::kModelError, error.what());
  }
}

struct vision_simple::InferOCROrtPaddleImpl::Impl {
  VisionHelper vision_helper;
  OCRModelType model_type;
  const std::map<int, std::string> char_dict;
  std::unique_ptr<Ort::Session> det, rec;
  Ort::Allocator det_allocator, rec_allocator;
  std::string det_input_name, det_output_name, rec_input_name, rec_output_name;
  Ort::MemoryInfo det_memory_info, rec_memory_info;
  cv::Mat chwrgb_image, preprocessed_image;

  explicit Impl(InferContextORT& ort_ctx, OCRModelType model_type,
                std::map<int, std::string> char_dict,
                std::unique_ptr<Ort::Session> det,
                std::unique_ptr<Ort::Session> rec)
      : model_type(model_type),
        char_dict(std::move(char_dict)),
        det(std::move(det)),
        rec(std::move(rec)),
        det_allocator(*this->det, ort_ctx.env_memory_info()),
        rec_allocator(*this->rec, ort_ctx.env_memory_info()),
        det_input_name(std::string(
            this->det->GetInputNameAllocated(0, det_allocator).get())),
        det_output_name(std::string(
            this->det->GetOutputNameAllocated(0, det_allocator).get())),
        rec_input_name(std::string(
            this->rec->GetInputNameAllocated(0, rec_allocator).get())),
        rec_output_name(std::string(
            this->rec->GetOutputNameAllocated(0, rec_allocator).get())),
        det_memory_info(Ort::MemoryInfo::CreateCpu(
            ort_ctx.env_memory_info().GetAllocatorType(),
            ort_ctx.env_memory_info().GetMemoryType())),
        rec_memory_info(Ort::MemoryInfo::CreateCpu(
            ort_ctx.env_memory_info().GetAllocatorType(),
            ort_ctx.env_memory_info().GetMemoryType())) {}

  static int PadLength(int length, int pad = 32) noexcept {
    if (length <= 0) return 0;
    const int remainder = length % pad;
    const int padding = remainder ? pad - remainder : 0;
    if (length > std::numeric_limits<int>::max() - padding) return 0;
    return length + padding;
  }

  cv::Mat& DetPreProcess(const cv::Mat& image, LetterboxTransform& transform) {
    const auto target_size =
        cv::Size{PadLength(image.cols), PadLength(image.rows)};
    auto& padded_img = vision_helper.Letterbox(image, target_size, transform);
    if (padded_img.empty()) {
      preprocessed_image.release();
      return preprocessed_image;
    }
    if (chwrgb_image.rows != target_size.height ||
        chwrgb_image.cols != target_size.width)
      chwrgb_image = cv::Mat::zeros(target_size, CV_8UC3);
    vision_helper.HWC2CHW_BGR2RGB<uint8_t>(padded_img, chwrgb_image);
    if (preprocessed_image.rows != target_size.height ||
        preprocessed_image.cols != target_size.width)
      preprocessed_image = cv::Mat::zeros(target_size, CV_32FC3);
    chwrgb_image.convertTo(preprocessed_image, CV_32F, 1.f / 255.f);
    return preprocessed_image;
  }

  /**
   *
   * @param output_tensor session.Run()后通过IOBinding获得的张量
   * @param transform 检测预处理的实际缩放与填充
   * @param iou_threshold 矩形重合区域IOU阈值，大于该阈值的将会被去重
   * @param contours_min_area 轮廓查找最小区域阈值
   * @param rect_min_area 矩形最小区域阈值
   * @param kernel_size 膨胀操作的kernel_size
   * @return 找到的所有矩形
   */
  InferResult<std::vector<cv::Rect>> DetPostProcess(
      const Ort::Value& output_tensor, const LetterboxTransform& transform,
      double iou_threshold = 0.3f, double contours_min_area = 12. * 12.,
      double rect_min_area = 8 * 8, int kernel_size = 2) {
    if (!output_tensor.IsTensor()) {
      return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                        "OCR detection output must be a tensor");
    }
    const auto info = output_tensor.GetTensorTypeAndShapeInfo();
    const auto output_shape = info.GetShape();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_shape.size() != 4 || output_shape[0] != 1 ||
        output_shape[1] != 1 ||
        output_shape[2] != transform.target_size.height ||
        output_shape[3] != transform.target_size.width ||
        output_shape[2] <= 0 || output_shape[3] <= 0 ||
        static_cast<size_t>(output_shape[2]) >
            std::numeric_limits<size_t>::max() /
                static_cast<size_t>(output_shape[3]) ||
        info.GetElementCount() != static_cast<size_t>(output_shape[2]) *
                                      static_cast<size_t>(output_shape[3])) {
      return MK_VSERROR(
          VisionSimpleErrorCode::kModelError,
          "OCR detection requires float [1,1,H,W] matching its input");
    }
    const auto* output_ptr = output_tensor.GetTensorData<float>();
    auto img = cv::Mat{static_cast<int>(output_shape[2]),
                       static_cast<int>(output_shape[3]), CV_32FC1,
                       const_cast<float*>(output_ptr)};
    cv::Mat gray{img.rows, img.cols, CV_8UC1};
    img.convertTo(gray, CV_8UC1);
    cv::Mat dilated;
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::dilate(gray, dilated, kernel);
    for (auto i = 0; i < 2; i++) {
      cv::dilate(dilated, dilated, kernel);
    }
    std::vector<std::vector<cv::Point>> contours, filtered_contours;
#if (CV_MAJOR_VERSION >= 4) && (CV_MINOR_VERSION >= 10)
    cv::findContoursLinkRuns(dilated, contours);
#else
    std::vector<std::vector<cv::Point>> hierarchy;
    cv::findContours(gray, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE);
#endif
    for (const auto& contour : contours) {
      if (contourArea(contour) > contours_min_area) {
        filtered_contours.push_back(contour);  // 添加满足条件的轮廓
      }
    }
    std::vector<cv::Rect> rects;
    for (const auto& contour : contours) {
      // 使用 boundingRect 拟合矩形
      if (auto rect{boundingRect(contour)}; rect.area() > rect_min_area) {
        rects.emplace_back(rect);
      }
    }
    auto filtered_boxes = vision_helper.FilterByIOU(rects, iou_threshold);
    for (auto& filtered_box : filtered_boxes) {
      const cv::Vec4f endpoints{
          static_cast<float>(filtered_box.x),
          static_cast<float>(filtered_box.y),
          static_cast<float>(filtered_box.x) + filtered_box.width,
          static_cast<float>(filtered_box.y) + filtered_box.height};
      filtered_box = VisionHelper::ScaleCoords(transform, endpoints);
    }
    std::erase_if(filtered_boxes, [](const cv::Rect& box) {
      return box.width <= 0 || box.height <= 0;
    });
    return filtered_boxes;
  }

  InferResult<Ort::Value> RecPreProcess(const cv::Mat& image,
                                        const cv::Rect& box,
                                        int fixed_height = 48) {
    const auto crop = box & cv::Rect{0, 0, image.cols, image.rows};
    if (crop.width <= 0 || crop.height <= 0) {
      return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                        "OCR recognition crop has no image pixels");
    }
    const double scaled_width =
        static_cast<double>(fixed_height) * crop.width / crop.height;
    if (scaled_width > std::numeric_limits<int>::max()) {
      return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                        "OCR recognition crop is too wide");
    }
    const int width =
        PadLength(std::max(1, static_cast<int>(scaled_width)), fixed_height);
    if (width == 0) {
      return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                        "OCR recognition crop width cannot be padded");
    }
    cv::Size output_size{width, fixed_height};
    const auto output_image_size_bytes =
        static_cast<size_t>(output_size.width) * output_size.height * 3 *
        sizeof(float);
    int64_t tensor_shape[4] = {1, 3, output_size.height, output_size.width};
    auto tensor =
        Ort::Value::CreateTensor<float>(rec_allocator, tensor_shape, 4);
    auto tensor_base_ptr = tensor.GetTensorMutableData<float>();
    cv::Mat resized_image;
    cv::resize(image(crop), resized_image, output_size);
    VisionHelper vision_helper_tmp;
    vision_helper_tmp.HWC2CHW_BGR2RGB<uint8_t>(resized_image, resized_image);
    cv::Mat output_image{output_size, CV_32FC3};
    resized_image.convertTo(output_image, CV_32F, 1.f / 255.f, -0.5f);
    output_image /= 0.5f;
    std::memcpy(tensor_base_ptr, output_image.ptr<float>(),
                output_image_size_bytes);
    return tensor;
  }

  InferResult<std::pair<std::string, float>> RecPostProcess(
      const Ort::Value& output_tensor, float confidence_threshold) {
    if (!output_tensor.IsTensor()) {
      return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                        "OCR recognition output must be a tensor");
    }
    const auto info = output_tensor.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        shape.size() != 3 || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0 ||
        static_cast<uint64_t>(shape[1]) > std::numeric_limits<size_t>::max() ||
        static_cast<uint64_t>(shape[2]) > std::numeric_limits<size_t>::max()) {
      return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                        "OCR recognition requires float [1,T,C]");
    }
    return DecodeOCRCTC(
        std::span<const float>{output_tensor.GetTensorData<float>(),
                               info.GetElementCount()},
        static_cast<size_t>(shape[1]), static_cast<size_t>(shape[2]), char_dict,
        confidence_threshold);
  }

  RunResult Run(const cv::Mat& image, float confidence_threshold) {
    auto total_timer = LogContext::ScopedTimer("OCR::Run::total",
                                               LogFacade::TimerCallback("ocr"));
    LogFacade::Info("ocr", "OCR inference started");
    // det stage
    auto det_timer = LogContext::ScopedTimer("OCR::det", nullptr);
    LetterboxTransform transform;
    auto& input_image = DetPreProcess(image, transform);
    if (input_image.empty()) {
      return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                        "OCR image dimensions cannot be letterboxed");
    }
    int64_t input_image_shape[4] = {1, 3, input_image.rows, input_image.cols};
    auto input_size_bytes =
        static_cast<unsigned long long>(input_image.channels()) *
        input_image.rows * input_image.cols * sizeof(float);
    auto det_input_tensor = Ort::Value::CreateTensor(
        det_allocator, input_image_shape,
        sizeof(input_image_shape) / sizeof(decltype(input_image_shape[0])),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memcpy(det_input_tensor.GetTensorMutableData<float>(),
                input_image.ptr<float>(), input_size_bytes);
    Ort::IoBinding det_io_binding(*det);
    det_io_binding.BindInput(det_input_name.c_str(), det_input_tensor);
    det_io_binding.BindOutput(det_output_name.c_str(), det_memory_info);
    Ort::RunOptions run_options;
    det->Run(run_options, det_io_binding);
    const auto ovalues = det_io_binding.GetOutputValues();
    if (ovalues.size() != 1) {
      return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                        "OCR detection returned an unexpected output count");
    }
    auto maybe_boxes = DetPostProcess(ovalues[0], transform);
    if (!maybe_boxes) return std::unexpected(std::move(maybe_boxes.error()));
    const auto& boxes = *maybe_boxes;
    LogFacade::Timing("ocr", "OCR::det", det_timer.elapsed_ms());
    LogFacade::Info(
        "ocr", std::format("OCR rec processed {} text boxes", boxes.size()));
    OCRFrameResult frame_result;
    // rec stage
    auto rec_timer = LogContext::ScopedTimer("OCR::rec", nullptr);
    // Recognize each crop separately; batching changes recognition behavior.
    for (const auto& box : boxes) {
      auto tensor = RecPreProcess(image, box);
      if (!tensor) return std::unexpected(std::move(tensor.error()));
      Ort::IoBinding rec_io_binding(*rec);
      rec_io_binding.BindInput(rec_input_name.c_str(), *tensor);
      rec_io_binding.BindOutput(rec_output_name.c_str(), rec_memory_info);
      Ort::RunOptions rec_run_options{};
      rec->Run(rec_run_options, rec_io_binding);
      const auto rec_outputs = rec_io_binding.GetOutputValues();
      if (rec_outputs.size() != 1) {
        return MK_VSERROR(
            VisionSimpleErrorCode::kModelError,
            "OCR recognition returned an unexpected output count");
      }
      auto line = RecPostProcess(rec_outputs[0], confidence_threshold);
      if (!line) return std::unexpected(std::move(line.error()));
      frame_result.results.emplace_back(box, line->second,
                                        std::move(line->first));
    }
    LogFacade::Timing("ocr", "OCR::rec", rec_timer.elapsed_ms());
    return frame_result;
  }
};

vision_simple::InferOCROrtPaddleImpl::InferOCROrtPaddleImpl(
    InferContextORT& ort_ctx, OCRModelType model_type,
    std::map<int, std::string> char_dict, std::unique_ptr<Ort::Session> det,
    std::unique_ptr<Ort::Session> rec)
    : impl_(std::make_unique<Impl>(ort_ctx, model_type, std::move(char_dict),
                                   std::move(det), std::move(rec))) {}

vision_simple::OCRModelType vision_simple::InferOCROrtPaddleImpl::model_type()
    const noexcept {
  return this->impl_->model_type;
}

vision_simple::InferOCR::RunResult vision_simple::InferOCROrtPaddleImpl::Run(
    const cv::Mat& image, float confidence_threshold) noexcept {
  try {
    auto valid = ValidateInferInput(image, confidence_threshold);
    if (!valid) return std::unexpected(std::move(valid.error()));
    return this->impl_->Run(image, confidence_threshold);
  } catch (const cv::Exception& error) {
    return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError, error.what());
  } catch (const Ort::Exception& error) {
    return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError, error.what());
  } catch (const std::exception& error) {
    return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError, error.what());
  }
}
