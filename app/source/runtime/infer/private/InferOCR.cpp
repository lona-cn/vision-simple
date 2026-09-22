#include "InferOCR.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <limits>
#include <mutex>
#include <numeric>

#include "InferORT.h"
#include "InferValidation.hpp"
#include "LogContext.h"
#include "LogFacade.h"
#include "OCRPostProcess.hpp"
#include "VisionHelper.hpp"

vision_simple::InferOCR::CreateResult vision_simple::InferOCR::Create(
    InferContext& context, std::map<int, std::string> char_dict,
    std::span<uint8_t> det_data, std::span<uint8_t> rec_data,
    OCRModelType model_type, size_t device_id) noexcept {
  try {
    const auto* postprocessor = FindOCRPostProcessor(model_type);
    if (!postprocessor) {
      return MK_VSERROR(VisionSimpleErrorCode::kUnimplementedError,
                        "Unsupported OCR model type");
    }
    size_t batch_size = 1;
    if (const auto it = context.args().find("ocr_rec_batch_size");
        it != context.args().end()) {
      const auto& text = it->second;
      const auto parsed =
          std::from_chars(text.data(), text.data() + text.size(), batch_size);
      if (text.empty() || parsed.ec != std::errc{} ||
          parsed.ptr != text.data() + text.size() || batch_size < 1 ||
          batch_size > 64) {
        return MK_VSERROR(
            VisionSimpleErrorCode::kParameterError,
            "ocr_rec_batch_size must be a decimal integer in 1..64");
      }
    }
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
    const auto input_type = (*rec)->GetInputTypeInfo(0);
    const auto output_type = (*rec)->GetOutputTypeInfo(0);
    const auto input_info = input_type.GetTensorTypeAndShapeInfo();
    const auto output_info = output_type.GetTensorTypeAndShapeInfo();
    const auto input_shape = input_info.GetShape();
    const auto output_shape = output_info.GetShape();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        input_shape.size() != 4 || input_shape[1] != 3 ||
        (input_shape[0] != -1 && (input_shape[0] < 1 || input_shape[0] > 64)) ||
        (input_shape[2] != -1 &&
         (input_shape[2] < 1 ||
          input_shape[2] > std::numeric_limits<int>::max())) ||
        (input_shape[3] != -1 &&
         (input_shape[3] < 1 ||
          input_shape[3] > std::numeric_limits<int>::max()))) {
      return MK_VSERROR(
          VisionSimpleErrorCode::kModelError,
          "OCR recognition requires float [N,3,H,W], N in 1..64 or dynamic");
    }
    const size_t max_elements =
        std::numeric_limits<size_t>::max() / sizeof(float);
    const size_t allocation_batch =
        input_shape[0] > 0 ? static_cast<size_t>(input_shape[0]) : batch_size;
    const size_t allocation_height =
        input_shape[2] > 0 ? static_cast<size_t>(input_shape[2]) : 48;
    if (input_shape[3] > 0 &&
        allocation_height > max_elements / 3 / allocation_batch /
                                static_cast<size_t>(input_shape[3])) {
      return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                        "OCR recognition input dimensions overflow");
    }
    const size_t classes = char_dict.size() + postprocessor->extra_classes;
    if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_shape.size() != 3 ||
        (output_shape[0] != -1 &&
         (output_shape[0] < 1 || output_shape[0] > 64 ||
          (input_shape[0] > 0 && output_shape[0] != input_shape[0]))) ||
        (input_shape[0] == -1 && output_shape[0] != -1) ||
        (output_shape[1] != -1 && output_shape[1] <= 0) ||
        (output_shape[2] != -1 &&
         (output_shape[2] <= 0 ||
          static_cast<uint64_t>(output_shape[2]) != classes))) {
      return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                        "OCR recognition requires float [N,T,C] matching batch "
                        "and dictionary");
    }
    std::array<int64_t, 4> recognition_shape;
    std::copy(input_shape.begin(), input_shape.end(),
              recognition_shape.begin());
    return std::make_unique<InferOCROrtPaddleImpl>(
        *ort_ctx, model_type, std::move(char_dict), std::move(*det),
        std::move(*rec), batch_size, recognition_shape);
  } catch (const cv::Exception& error) {
    return MK_VSERROR(VisionSimpleErrorCode::kModelError, error.what());
  } catch (const Ort::Exception& error) {
    return MK_VSERROR(VisionSimpleErrorCode::kModelError, error.what());
  } catch (const std::exception& error) {
    return MK_VSERROR(VisionSimpleErrorCode::kModelError, error.what());
  }
}

struct vision_simple::InferOCROrtPaddleImpl::Impl {
  struct Workspace {
    std::array<int64_t, 4> shape{};
    Ort::Value input{nullptr};
    Ort::IoBinding binding;

    explicit Workspace(Ort::Session& session) : binding(session) {}

    cv::Mat InputImage(Ort::Allocator& allocator, cv::Size size,
                       size_t batch_size = 1) {
      const std::array<int64_t, 4> next{static_cast<int64_t>(batch_size), 3,
                                        size.height, size.width};
      if (!input || shape != next) {
        // Drop the old binding before replacing its tensor. Commit the shape
        // only after allocation succeeds so a failed resize can be retried.
        binding.ClearBoundInputs();
        input = Ort::Value::CreateTensor<float>(allocator, next.data(),
                                                next.size());
        shape = next;
      }
      return cv::Mat(size, CV_32FC3, input.GetTensorMutableData<float>());
    }

    void Bind(const std::string& input_name, const std::string& output_name,
              const Ort::MemoryInfo& memory_info) {
      // Rebind CPU input after preprocessing for providers that copy on bind.
      binding.BindInput(input_name.c_str(), input);
      // Outputs can depend on input data as well as shape. Never retain a
      // previous dynamic OrtValue, including one left by an unsuccessful Run.
      binding.ClearBoundOutputs();
      binding.BindOutput(output_name.c_str(), memory_info);
    }
  };

  struct FrameWorkspace {
    Workspace det_workspace, rec_workspace;
    VisionHelper vision_helper, rec_vision_helper;
    cv::Mat chwrgb_image, preprocessed_image, rec_resized_image;
    std::vector<Ort::Value> outputs;
    LetterboxTransform transform;

    FrameWorkspace(Ort::Session& det, Ort::Session& rec)
        : det_workspace(det), rec_workspace(rec) {}
  };

  OCRModelType model_type;
  const std::map<int, std::string> char_dict;
  const OCRPostProcessor& postprocessor;
  const size_t batch_size;
  const std::array<int64_t, 4> recognition_shape;
  std::unique_ptr<Ort::Session> det, rec;
  Ort::Allocator det_allocator, rec_allocator;
  std::string det_input_name, det_output_name, rec_input_name, rec_output_name;
  Ort::MemoryInfo det_memory_info, rec_memory_info;
  FrameWorkspace legacy_workspace;
  std::mutex run_mutex, session_mutex, pool_mutex;
  std::array<std::unique_ptr<FrameWorkspace>, 2> idle_workspaces;

  std::unique_ptr<FrameWorkspace> AcquireWorkspace() {
    {
      const std::lock_guard lock(pool_mutex);
      for (auto& idle : idle_workspaces) {
        if (idle) return std::move(idle);
      }
    }
    const std::lock_guard lock(session_mutex);
    return std::make_unique<FrameWorkspace>(*det, *rec);
  }

  void ReleaseWorkspace(std::unique_ptr<FrameWorkspace> workspace) noexcept {
    try {
      const std::lock_guard lock(pool_mutex);
      for (auto& idle : idle_workspaces) {
        if (!idle) {
          idle = std::move(workspace);
          return;
        }
      }
    } catch (...) {
      // Returning scratch must never turn task destruction into an exception.
    }
  }

  explicit Impl(InferContextORT& ort_ctx, OCRModelType model_type,
                std::map<int, std::string> char_dict,
                std::unique_ptr<Ort::Session> det,
                std::unique_ptr<Ort::Session> rec, size_t batch_size,
                std::array<int64_t, 4> recognition_shape)
      : model_type(model_type),
        char_dict(std::move(char_dict)),
        postprocessor(*FindOCRPostProcessor(model_type)),
        batch_size(batch_size),
        recognition_shape(recognition_shape),
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
            ort_ctx.env_memory_info().GetMemoryType())),
        legacy_workspace(*this->det, *this->rec) {}

  static int PadLength(int length, int pad = 32) noexcept {
    if (length <= 0) return 0;
    const int remainder = length % pad;
    const int padding = remainder ? pad - remainder : 0;
    if (length > std::numeric_limits<int>::max() - padding) return 0;
    return length + padding;
  }

  cv::Mat& DetPreProcess(FrameWorkspace& workspace, const cv::Mat& image) {
    auto& vision_helper = workspace.vision_helper;
    auto& preprocessed_image = workspace.preprocessed_image;
    auto& chwrgb_image = workspace.chwrgb_image;
    auto& transform = workspace.transform;
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
    {
      const std::lock_guard lock(session_mutex);
      preprocessed_image =
          workspace.det_workspace.InputImage(det_allocator, target_size);
    }
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
      FrameWorkspace& workspace, const Ort::Value& output_tensor,
      const LetterboxTransform& transform, double iou_threshold = 0.3f,
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
    std::vector<std::vector<cv::Point>> contours;
#if (CV_MAJOR_VERSION >= 4) && (CV_MINOR_VERSION >= 10)
    cv::findContoursLinkRuns(dilated, contours);
#else
    std::vector<std::vector<cv::Point>> hierarchy;
    cv::findContours(gray, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE);
#endif
    std::vector<cv::Rect> rects;
    for (const auto& contour : contours) {
      // DBNet predicts shrunken text regions. Restore their extent before
      // recognition; the AABB of a round polygon offset expands by distance
      // on every axis. Paddle's DB unclip ratio is 1.5.
      if (auto rect{boundingRect(contour)}; rect.area() > rect_min_area) {
        const auto box = cv::minAreaRect(contour);
        const double perimeter = 2.0 * (box.size.width + box.size.height);
        const auto padding =
            perimeter > 0 ? static_cast<int64_t>(
                                std::ceil(static_cast<double>(box.size.width) *
                                          box.size.height * 1.5 / perimeter))
                          : int64_t{0};
        const int left = static_cast<int>(
            std::max<int64_t>(0, static_cast<int64_t>(rect.x) - padding));
        const int top = static_cast<int>(
            std::max<int64_t>(0, static_cast<int64_t>(rect.y) - padding));
        const int right = static_cast<int>(std::min<int64_t>(
            img.cols, static_cast<int64_t>(rect.x) + rect.width + padding));
        const int bottom = static_cast<int>(std::min<int64_t>(
            img.rows, static_cast<int64_t>(rect.y) + rect.height + padding));
        rects.emplace_back(left, top, right - left, bottom - top);
      }
    }
    auto filtered_boxes =
        workspace.vision_helper.FilterByIOU(rects, iou_threshold);
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

  InferResult<int> RecWidth(const cv::Rect& crop) const {
    if (crop.width <= 0 || crop.height <= 0) {
      return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                        "OCR recognition crop has no image pixels");
    }
    if (recognition_shape[3] > 0) return static_cast<int>(recognition_shape[3]);
    const int height =
        recognition_shape[2] > 0 ? static_cast<int>(recognition_shape[2]) : 48;
    const double scaled_width =
        static_cast<double>(height) * crop.width / crop.height;
    if (scaled_width > std::numeric_limits<int>::max()) {
      return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                        "OCR recognition crop is too wide");
    }
    const int width =
        PadLength(std::max(1, static_cast<int>(scaled_width)), height);
    if (width == 0) {
      return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                        "OCR recognition crop width cannot be padded");
    }
    return width;
  }

  InferResult<void> RecPreProcess(FrameWorkspace& workspace,
                                  const cv::Mat& image,
                                  const std::vector<cv::Rect>& boxes,
                                  std::span<const size_t> indices, int width,
                                  size_t tensor_batch) {
    const int height =
        recognition_shape[2] > 0 ? static_cast<int>(recognition_shape[2]) : 48;
    const size_t max_elements =
        std::numeric_limits<size_t>::max() / sizeof(float);
    if (static_cast<size_t>(height) >
        max_elements / 3 / tensor_batch / static_cast<size_t>(width)) {
      return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                        "OCR recognition input dimensions overflow");
    }
    const size_t sample_elements = size_t{3} * height * width;
    {
      const std::lock_guard lock(session_mutex);
      workspace.rec_workspace.InputImage(rec_allocator, {width, height},
                                         tensor_batch);
    }
    auto* input = workspace.rec_workspace.input.GetTensorMutableData<float>();
    auto& resized = workspace.rec_resized_image;
    for (size_t sample = 0; sample < indices.size(); ++sample) {
      const auto crop =
          boxes[indices[sample]] & cv::Rect{0, 0, image.cols, image.rows};
      // Dynamic widths retain the original height-multiple resize exactly.
      // Fixed width models resize the whole crop to their declared H/W: no
      // invented valid-length or proportional output truncation is applied.
      cv::resize(image(crop), resized, {width, height});
      workspace.rec_vision_helper.HWC2CHW_BGR2RGB<uint8_t>(resized, resized);
      cv::Mat output(cv::Size{width, height}, CV_32FC3,
                     input + sample * sample_elements);
      resized.convertTo(output, CV_32F, 1.f / 255.f, -0.5f);
      output /= 0.5f;
    }
    // Fixed-N exports require a complete tensor even for the final minibatch.
    // Zero is a finite neutral normalized image; dummy outputs are ignored.
    std::fill(input + indices.size() * sample_elements,
              input + tensor_batch * sample_elements, 0.f);
    return {};
  }

  InferResult<void> RecPostProcess(const Ort::Value& output_tensor,
                                   float confidence_threshold,
                                   size_t tensor_batch,
                                   std::span<const size_t> indices,
                                   const std::vector<cv::Rect>& boxes,
                                   OCRFrameResult& result) {
    if (!output_tensor.IsTensor()) {
      return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                        "OCR recognition output must be a tensor");
    }
    const auto info = output_tensor.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        shape.size() != 3 || shape[0] != static_cast<int64_t>(tensor_batch) ||
        shape[1] <= 0 || shape[2] <= 0 ||
        static_cast<uint64_t>(shape[1]) > std::numeric_limits<size_t>::max() ||
        static_cast<uint64_t>(shape[2]) !=
            char_dict.size() + postprocessor.extra_classes) {
      return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                        "OCR recognition requires matching float [N,T,C]");
    }
    const auto timesteps = static_cast<size_t>(shape[1]);
    const auto classes = static_cast<size_t>(shape[2]);
    if (timesteps >
            std::numeric_limits<size_t>::max() / tensor_batch / classes ||
        info.GetElementCount() != tensor_batch * timesteps * classes) {
      return MK_VSERROR(
          VisionSimpleErrorCode::kModelError,
          "OCR recognition output dimensions overflow or mismatch");
    }
    const size_t sample_elements = timesteps * classes;
    const auto* data = output_tensor.GetTensorData<float>();
    for (size_t sample = 0; sample < indices.size(); ++sample) {
      auto line = postprocessor.decode(
          {data + sample * sample_elements, sample_elements}, timesteps,
          classes, char_dict, confidence_threshold);
      if (!line) return std::unexpected(std::move(line.error()));
      const auto index = indices[sample];
      result.results[index] = {boxes[index], line->second,
                               std::move(line->first)};
    }
    return {};
  }

  InferResult<void> Execute(FrameWorkspace& workspace, bool detection) {
    const std::lock_guard lock(session_mutex);
    auto& stage = detection ? workspace.det_workspace : workspace.rec_workspace;
    workspace.outputs.clear();
    stage.Bind(detection ? det_input_name : rec_input_name,
               detection ? det_output_name : rec_output_name,
               detection ? det_memory_info : rec_memory_info);
    Ort::RunOptions options;
    (detection ? det : rec)->Run(options, stage.binding);
    workspace.outputs = stage.binding.GetOutputValues();
    if (workspace.outputs.size() != 1) {
      return MK_VSERROR(
          VisionSimpleErrorCode::kModelError,
          detection ? "OCR detection returned an unexpected output count"
                    : "OCR recognition returned an unexpected output count");
    }
    return {};
  }

  struct Task final : detail::FrameTask {
    enum class Stage {
      kDetPre,
      kDetRun,
      kDetPost,
      kRecPre,
      kRecRun,
      kRecPost,
      kDone
    };
    Impl& owner;
    std::unique_ptr<FrameWorkspace> owned_workspace;
    FrameWorkspace& workspace;
    const cv::Mat& image;
    float threshold;
    Stage stage = Stage::kDetPre;
    std::vector<cv::Rect> boxes;
    size_t crop_index = 0;
    std::vector<size_t> crop_order;
    std::vector<int> crop_widths;
    size_t minibatch_count = 0, tensor_batch = 0;
    OCRFrameResult result;

    Task(Impl& owner, const cv::Mat& image, float threshold,
         std::unique_ptr<FrameWorkspace> scratch)
        : owner(owner),
          owned_workspace(std::move(scratch)),
          workspace(*owned_workspace),
          image(image),
          threshold(threshold) {}

    Task(Impl& owner, const cv::Mat& image, float threshold)
        : owner(owner),
          workspace(owner.legacy_workspace),
          image(image),
          threshold(threshold) {}

    ~Task() override {
      if (owned_workspace) owner.ReleaseWorkspace(std::move(owned_workspace));
    }

    VSResult<std::optional<detail::PipelineLane>> Advance() noexcept override {
      try {
        using Lane = detail::PipelineLane;
        switch (stage) {
          case Stage::kDetPre:
            if (owner.DetPreProcess(workspace, image).empty()) {
              return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                                "OCR image dimensions cannot be letterboxed");
            }
            stage = Stage::kDetRun;
            return Lane::kInference;
          case Stage::kDetRun: {
            auto executed = owner.Execute(workspace, true);
            if (!executed) return std::unexpected(std::move(executed.error()));
            stage = Stage::kDetPost;
            return Lane::kPostprocess;
          }
          case Stage::kDetPost: {
            auto detected = owner.DetPostProcess(
                workspace, workspace.outputs[0], workspace.transform);
            if (!detected) return std::unexpected(std::move(detected.error()));
            boxes = std::move(*detected);
            if (boxes.empty()) {
              stage = Stage::kDone;
              return std::nullopt;
            }
            result.results.resize(boxes.size());
            crop_order.resize(boxes.size());
            crop_widths.reserve(boxes.size());
            std::iota(crop_order.begin(), crop_order.end(), size_t{0});
            for (const auto& box : boxes) {
              auto width = owner.RecWidth(box);
              if (!width) return std::unexpected(std::move(width.error()));
              crop_widths.push_back(*width);
            }
            std::stable_sort(crop_order.begin(), crop_order.end(),
                             [this](size_t left, size_t right) {
                               return crop_widths[left] < crop_widths[right];
                             });
            stage = Stage::kRecPre;
            return Lane::kPreprocess;
          }
          case Stage::kRecPre: {
            const int width = crop_widths[crop_order[crop_index]];
            const size_t limit =
                owner.recognition_shape[0] > 0
                    ? static_cast<size_t>(owner.recognition_shape[0])
                    : owner.batch_size;
            minibatch_count = 1;
            while (minibatch_count < limit &&
                   crop_index + minibatch_count < crop_order.size() &&
                   crop_widths[crop_order[crop_index + minibatch_count]] ==
                       width)
              ++minibatch_count;
            tensor_batch =
                owner.recognition_shape[0] > 0 ? limit : minibatch_count;
            auto prepared =
                owner.RecPreProcess(workspace, image, boxes,
                                    std::span<const size_t>{crop_order}.subspan(
                                        crop_index, minibatch_count),
                                    width, tensor_batch);
            if (!prepared) return std::unexpected(std::move(prepared.error()));
            stage = Stage::kRecRun;
            return Lane::kInference;
          }
          case Stage::kRecRun: {
            auto executed = owner.Execute(workspace, false);
            if (!executed) return std::unexpected(std::move(executed.error()));
            stage = Stage::kRecPost;
            return Lane::kPostprocess;
          }
          case Stage::kRecPost: {
            auto decoded = owner.RecPostProcess(
                workspace.outputs[0], threshold, tensor_batch,
                std::span<const size_t>{crop_order}.subspan(crop_index,
                                                            minibatch_count),
                boxes, result);
            if (!decoded) return std::unexpected(std::move(decoded.error()));
            crop_index += minibatch_count;
            if (crop_index == boxes.size()) {
              stage = Stage::kDone;
              return std::nullopt;
            }
            stage = Stage::kRecPre;
            return Lane::kPreprocess;
          }
          case Stage::kDone:
            return std::nullopt;
        }
        return std::nullopt;
      } catch (const cv::Exception& error) {
        return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError, error.what());
      } catch (const Ort::Exception& error) {
        return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError, error.what());
      } catch (const std::exception& error) {
        return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError, error.what());
      }
    }

    detail::FrameResult TakeResult() noexcept override {
      return std::move(result);
    }
  };

  RunResult Run(const cv::Mat& image, float confidence_threshold) {
    auto total_timer = LogContext::ScopedTimer("OCR::Run::total",
                                               LogFacade::TimerCallback("ocr"));
    LogFacade::Info("ocr", "OCR inference started");
    Task task(*this, image, confidence_threshold);
    auto det_timer = LogContext::ScopedTimer("OCR::det", nullptr);
    for (int i = 0; i < 3; ++i) {
      auto next = task.Advance();
      if (!next) return std::unexpected(std::move(next.error()));
    }
    LogFacade::Timing("ocr", "OCR::det", det_timer.elapsed_ms());
    LogFacade::Info("ocr", std::format("OCR rec processed {} text boxes",
                                       task.boxes.size()));
    auto rec_timer = LogContext::ScopedTimer("OCR::rec", nullptr);
    while (task.stage != Task::Stage::kDone) {
      auto next = task.Advance();
      if (!next) return std::unexpected(std::move(next.error()));
    }
    LogFacade::Timing("ocr", "OCR::rec", rec_timer.elapsed_ms());
    return std::move(task.result);
  }
};
vision_simple::InferOCROrtPaddleImpl::~InferOCROrtPaddleImpl() = default;

vision_simple::InferOCROrtPaddleImpl::InferOCROrtPaddleImpl(
    InferContextORT& ort_ctx, OCRModelType model_type,
    std::map<int, std::string> char_dict, std::unique_ptr<Ort::Session> det,
    std::unique_ptr<Ort::Session> rec, size_t batch_size,
    std::array<int64_t, 4> recognition_shape)
    : impl_(std::make_unique<Impl>(ort_ctx, model_type, std::move(char_dict),
                                   std::move(det), std::move(rec), batch_size,
                                   recognition_shape)) {}

vision_simple::OCRModelType vision_simple::InferOCROrtPaddleImpl::model_type()
    const noexcept {
  return this->impl_->model_type;
}

vision_simple::InferOCR::RunResult vision_simple::InferOCROrtPaddleImpl::Run(
    const cv::Mat& image, float confidence_threshold) noexcept {
  try {
    const std::lock_guard lock(impl_->run_mutex);
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

vision_simple::VSResult<std::unique_ptr<vision_simple::detail::FrameTask>>
vision_simple::detail::MakeFrameTask(InferOCR& model, const cv::Mat& image,
                                     float confidence_threshold) noexcept {
  try {
    auto* backend = dynamic_cast<InferOCROrtPaddleImpl*>(&model);
    if (!backend) {
      return MK_VSERROR(VisionSimpleErrorCode::kUnimplementedError,
                        "OCR model does not support staged inference");
    }
    auto valid = ValidateInferInput(image, confidence_threshold);
    if (!valid) return std::unexpected(std::move(valid.error()));
    auto workspace = backend->impl_->AcquireWorkspace();
    return std::make_unique<InferOCROrtPaddleImpl::Impl::Task>(
        *backend->impl_, image, confidence_threshold, std::move(workspace));
  } catch (const cv::Exception& error) {
    return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError, error.what());
  } catch (const Ort::Exception& error) {
    return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError, error.what());
  } catch (const std::exception& error) {
    return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError, error.what());
  }
}
