#include "InferOCR.h"

#include <codecvt>
#include <magic_enum.hpp>
#include <memory_resource>
#include <numeric>

#include "InferORT.h"
#include "VisionHelper.hpp"

namespace {
// std::pmr::monotonic_buffer_resource mr{114514};
}

vision_simple::InferOCR::CreateResult vision_simple::InferOCR::Create(
    InferContext& context, std::map<int, std::string> char_dict,
    std::span<uint8_t> det_data, std::span<uint8_t> rec_data,
    OCRModelType model_type, size_t device_id) noexcept {
  auto& ort_ctx = dynamic_cast<InferContextORT&>(context);
  auto det = ort_ctx.CreateSession(det_data, device_id);
  if (!det) return std::unexpected(std::move(det.error()));
  auto rec = ort_ctx.CreateSession(rec_data, device_id);
  if (!rec) return std::unexpected(std::move(rec.error()));
  return std::make_unique<InferOCROrtPaddleImpl>(
      ort_ctx, model_type, std::move(char_dict), std::move(*det),
      std::move(*rec));
}

struct vision_simple::InferOCROrtPaddleImpl::Impl {
  VisionHelper vision_helper;
  OCRModelType model_type;
  std::map<int, std::string> char_dict;
  std::unique_ptr<Ort::Session> det, rec;
  Ort::Allocator det_allocator, rec_allocator;
  Ort::IoBinding det_io_binding, rec_io_binding;
  Ort::Value det_input_tensor, rec_input_tensor;
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
        det_io_binding(*this->det),
        rec_io_binding(*this->rec),
        det_input_tensor{nullptr},
        rec_input_tensor{nullptr},
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
            ort_ctx.env_memory_info().GetMemoryType())) {
    det_io_binding.BindOutput(det_output_name.c_str(), det_memory_info);
    rec_io_binding.BindOutput(rec_output_name.c_str(), rec_memory_info);
  }

  template <typename T>
    requires std::is_arithmetic_v<T>
  static T PadLength(T length, T pad = static_cast<T>(32)) {
    auto remainder = length % pad;
    auto pad_length = remainder ? pad - remainder : 0;
    return length + pad_length;
  }

  static size_t GetTensorSize(const Ort::Value& value) noexcept {
    auto shapes = value.GetTensorTypeAndShapeInfo().GetShape();
    return std::accumulate(shapes.begin(), shapes.end(), 1llu,
                           std::multiplies());
  }

  cv::Mat& DetPreProcess(const cv::Mat& image) noexcept {
    const auto target_size =
        cv::Size{PadLength(image.cols), PadLength(image.rows)};
    auto& padded_img = vision_helper.Letterbox(image, target_size);
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
   * @param input_image_size 输入张量图片的尺寸
   * @param original_image_size 原始图片尺寸
   * @param iou_threshold 矩形重合区域IOU阈值，大于该阈值的将会被去重
   * @param contours_min_area 轮廓查找最小区域阈值
   * @param rect_min_area 矩形最小区域阈值
   * @param kernel_size 膨胀操作的kernel_size
   * @return 找到的所有矩形
   */
  std::vector<cv::Rect> DetPostProcess(const Ort::Value& output_tensor,
                                       const cv::Size input_image_size,
                                       const cv::Size original_image_size,
                                       double iou_threshold = 0.3f,
                                       double contours_min_area = 12. * 12.,
                                       double rect_min_area = 8 * 8,
                                       int kernel_size = 6) noexcept {
    auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    auto output_ptr = output_tensor.GetConst().GetTensorData<float>();
    auto img = cv::Mat{static_cast<int>(output_shape[2]),
                       static_cast<int>(output_shape[3]), CV_32FC1,
                       (void*)(output_ptr)};
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
    // 重设到原始图片大小
    for (auto& filtered_box : filtered_boxes) {
      filtered_box = VisionHelper::ScaleCoords(input_image_size, filtered_box,
                                               original_image_size, true);
    }
    return filtered_boxes;
  }

  Ort::Value RecPreProcess(const cv::Mat& image, const cv::Rect& box,
                           int fixed_height = 48) {
    auto scale = static_cast<float>(fixed_height) / box.height;
    auto width = PadLength(static_cast<int>(scale * box.width), fixed_height);
    cv::Size output_size{width, fixed_height};
    const auto output_image_size_bytes =
        static_cast<size_t>(output_size.width) * output_size.height * 3 *
        sizeof(float);
    int64_t tensor_shape[4] = {1, 3, output_size.height, output_size.width};
    auto tensor =
        Ort::Value::CreateTensor<float>(rec_allocator, tensor_shape, 4);
    auto tensor_base_ptr = tensor.GetTensorMutableData<float>();
    auto box_image = image(box).clone();
    resize(box_image, box_image, output_size);
    cv::Mat resized_image = cv::Mat::zeros(output_size, CV_8UC3);
    box_image.copyTo(
        resized_image(cv::Rect(0, 0, box_image.cols, box_image.rows)));
    VisionHelper vision_helper_tmp;
    vision_helper_tmp.HWC2CHW_BGR2RGB<uint8_t>(resized_image, resized_image);
    cv::Mat output_image{output_size, CV_32FC3};
    resized_image.convertTo(output_image, CV_32F, 1.f / 255.f, -0.5f);
    output_image /= 0.5f;
    std::memcpy(tensor_base_ptr, output_image.ptr<float>(),
                output_image_size_bytes);
    return tensor;
  }

  static auto FindMaxValueIndex(const std::span<const float> vec)
      -> std::pair<float, size_t> {
    // 使用std::max_element找到最大值的迭代器
    auto max_it = std::ranges::max_element(vec);
    // 如果vector为空，返回一个无效的值
    if (max_it == vec.cend()) {
      return {std::numeric_limits<float>::lowest(), static_cast<size_t>(-1)};
    }
    // 获取最大值及其索引
    float max_value = *max_it;
    size_t index = std::distance(vec.begin(), max_it);

    return {max_value, index};
  }

  /**
   *
   * @param output 张量输出
   * @param output_shape 张量shape
   * @param confidence_threshold 每个字符的置信度阈值
   * @return
   */
  std::vector<std::pair<std::string, float>> RecPostProcess(
      const std::span<const float> output,
      const std::vector<int64_t>& output_shape,
      float confidence_threshold = 0.5f) {
    const auto stride = output_shape[1] * output_shape[2];
    auto output_base_ptr = output.data();
    std::vector<std::pair<std::string, float>> results;
    auto num_features = output_shape[2];
    for (auto i{0}; i < output_shape[0]; ++i) {
      auto output_ptr = output_base_ptr + i * stride;
      std::vector<std::pair<int, double>> idx_scores;
      idx_scores.reserve(output_shape[1]);
      for (auto j{0}; j < output_shape[1]; ++j) {
        auto [max_score, max_idx] = FindMaxValueIndex(
            std::span(output_ptr + j * num_features, num_features));
        idx_scores.emplace_back(max_idx, max_score);
      }
      std::stringstream ss;
      std::vector<float> scores;
      scores.reserve(idx_scores.size());
      for (auto [idx, score] : idx_scores) {
        if (idx == 0) continue;
        if (score <= confidence_threshold) continue;
        auto& c = char_dict[idx - 1];
        ss << c;
        scores.emplace_back(score);
      }
      auto confidence = !scores.empty() ? std::accumulate(scores.cbegin(),
                                                          scores.cend(), 0.0f) /
                                              static_cast<float>(scores.size())
                                        : 0.0f;
      results.emplace_back(ss.str(), confidence);
    }
    return results;
  }

  RunResult Run(const cv::Mat& image, float confidence_threshold) noexcept {
    auto& input_image = DetPreProcess(image);
    const cv::Size input_image_size{input_image.cols, input_image.rows},
        original_image_size{image.cols, image.rows};
    int64_t input_image_shape[4] = {1, 3, input_image.rows, input_image.cols};
    auto input_size_bytes =
        static_cast<unsigned long long>(input_image.channels()) *
        input_image.rows * input_image.cols * sizeof(float);
    det_input_tensor = Ort::Value::CreateTensor(
        det_allocator, input_image_shape,
        sizeof(input_image_shape) / sizeof(decltype(input_image_shape[0])),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memcpy(det_input_tensor.GetTensorMutableData<float>(),
                input_image.ptr<float>(), input_size_bytes);
    det_io_binding.BindInput(det_input_name.c_str(), det_input_tensor);
    det_io_binding.BindOutput(det_output_name.c_str(), det_memory_info);
    Ort::RunOptions run_options;
    det->Run(run_options, det_io_binding);
    const auto& ovalues = det_io_binding.GetOutputValues();
    auto& output_tensor = ovalues[0];
    auto boxes =
        DetPostProcess(output_tensor, input_image_size, original_image_size);
    OCRFrameResult frame_result;
    // Q:为什么不批处理呢？
    // A:因为效果不好
    std::vector<Ort::Value> rec_input_tensors;
    for (int64_t i = 0; i < boxes.size(); ++i) {
      auto tensor = RecPreProcess(image, boxes[i]);
      rec_input_tensors.emplace_back(std::move(tensor));
    }
    for (size_t i = 0; i < boxes.size(); ++i) {
      auto& box = boxes[i];
      rec_input_tensor = std::move(rec_input_tensors[i]);
      rec_io_binding.BindInput(rec_input_name.c_str(), rec_input_tensor);
      rec_io_binding.BindOutput(rec_output_name.c_str(), rec_memory_info);
      // predict string
      Ort::RunOptions rec_run_options{};
      rec->Run(rec_run_options, rec_io_binding);
      const auto& rec_outputs = rec_io_binding.GetOutputValues();
      auto& rec_output_tensor = rec_outputs[0];
      auto rec_output_shape =
          rec_output_tensor.GetTensorTypeAndShapeInfo().GetShape();
      const auto rec_output_tensor_size =
          std::accumulate(rec_output_shape.begin(), rec_output_shape.end(),
                          1llu, std::multiplies());
      auto span = std::span(rec_output_tensor.GetTensorData<float>(),
                            rec_output_tensor_size);
      auto lines = RecPostProcess(span, rec_output_shape, confidence_threshold);
      if (!lines.empty())
        frame_result.results.emplace_back(box, lines[0].second,
                                          std::move(lines[0].first));
    }
    return frame_result;
  }
};

vision_simple::InferOCROrtPaddleImpl::InferOCROrtPaddleImpl(
    InferContextORT& ort_ctx, OCRModelType model_type,
    std::map<int, std::string> char_dict, std::unique_ptr<Ort::Session> det,
    std::unique_ptr<Ort::Session> rec)
    : impl_(std::make_unique<Impl>(ort_ctx, model_type, char_dict,
                                   std::move(det), std::move(rec))) {}

vision_simple::OCRModelType vision_simple::InferOCROrtPaddleImpl::model_type()
    const noexcept {
  return this->impl_->model_type;
}

vision_simple::InferOCR::RunResult vision_simple::InferOCROrtPaddleImpl::Run(
    const cv::Mat& image, float confidence_threshold) noexcept {
  return this->impl_->Run(image, confidence_threshold);
}
