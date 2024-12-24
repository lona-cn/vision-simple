#include "Infer.h"
#include <map>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_float16.h>
#include <onnxruntime_session_options_config_keys.h>
#include <onnxruntime_run_options_config_keys.h>
#ifdef VISION_SIMPLE_WITH_DML
#include <dml_provider_factory.h>
#endif
#include <magic_enum.hpp>
#include <memory>
#include <numeric>
#include <regex>
#define INFER_CTX_LOG_ID "vision-simple"
#define INFER_CTX_LOG_LEVEL OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING
using namespace std;
using namespace cv;
using namespace vision_simple;

namespace vision_simple
{
    class InferContextONNXRuntime : public InferContext
    {
        std::unique_ptr<Ort::Env> env_;
        InferEP ep_;
        Ort::MemoryInfo env_memory_info_;

    public:
        InferContextONNXRuntime(const InferEP ep): env_(std::make_unique<Ort::Env>(
                                                       INFER_CTX_LOG_LEVEL, INFER_CTX_LOG_ID)),
                                                   ep_(ep),
                                                   env_memory_info_(
                                                       Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
        {
            env_->CreateAndRegisterAllocator(env_memory_info_, nullptr);
        }

        InferContextONNXRuntime(const InferContextONNXRuntime& other) = delete;
        InferContextONNXRuntime(InferContextONNXRuntime&& other) noexcept = default;
        InferContextONNXRuntime& operator=(const InferContextONNXRuntime& other) = delete;
        InferContextONNXRuntime& operator=(InferContextONNXRuntime&& other) noexcept = default;
        ~InferContextONNXRuntime() override = default;

        InferFramework framework() const noexcept override { return InferFramework::kONNXRUNTIME; }

        InferEP execution_provider() const noexcept override { return ep_; }

        Ort::Env& env() const noexcept { return *env_; }

        [[nodiscard]] Ort::MemoryInfo& env_memory_info()
        {
            return env_memory_info_;
        }
    };

    InferContext::CreateResult InferContext::Create(InferFramework framework, InferEP ep) noexcept
    {
        if (framework == InferFramework::kONNXRUNTIME && (ep == InferEP::kCPU || ep == InferEP::kDML))
            return std::make_unique<InferContextONNXRuntime>(ep);
        return std::unexpected{
            InferError{
                InferErrorCode::kParameterError, std::format("unsupported framework({}) or ep({})",
                                                             magic_enum::enum_name(framework),
                                                             magic_enum::enum_name(ep))
            }
        };
    }

    class YOLOFilter
    {
        YOLOVersion version_;
        std::vector<std::string> class_names_;
        std::vector<long long> shapes_;

    public:
        using FilterResult = InferResult<YOLOFrameResult>;

        explicit YOLOFilter(YOLOVersion version, std::vector<std::string> class_names,
                            std::vector<long long> shapes): version_(version),
                                                            class_names_(std::move(class_names)),
                                                            shapes_(std::move(shapes))
        {
        }

        YOLOVersion version() const noexcept { return version_; }

        static cv::Rect ScaleCoords(const cv::Size& imageShape, cv::Rect coords,
                                    const cv::Size& imageOriginalShape, bool p_Clip)
        {
            cv::Rect result;
            float gain = std::min(
                static_cast<float>(imageShape.height) / static_cast<float>(
                    imageOriginalShape.height),
                static_cast<float>(imageShape.width) / static_cast<float>(
                    imageOriginalShape.width));

            int padX = static_cast<int>(std::round(
                (imageShape.width - imageOriginalShape.width * gain) / 2.0f));
            int padY = static_cast<int>(std::round(
                (imageShape.height - imageOriginalShape.height * gain) / 2.0f));

            result.x = static_cast<int>(std::round((coords.x - padX) / gain));
            result.y = static_cast<int>(std::round((coords.y - padY) / gain));
            result.width = static_cast<int>(std::round(coords.width / gain));
            result.height = static_cast<int>(std::round(coords.height / gain));

            if (p_Clip)
            {
                result.x = std::clamp(result.x, 0, imageOriginalShape.width);
                result.y = std::clamp(result.y, 0, imageOriginalShape.height);
                result.width = std::clamp(result.width, 0,
                                          imageOriginalShape.width - result.x);
                result.height = std::clamp(result.height, 0,
                                           imageOriginalShape.height - result.y);
            }
            return result;
        }

        static std::vector<YOLOResult> ApplyNMS(
            const std::vector<YOLOResult>& detections, float iou_threshold)
        {
            std::vector<int> indices;
            std::vector<cv::Rect> boxes;
            std::vector<float> scores;

            // 提取所有的框和置信度
            for (const auto& detection : detections)
            {
                boxes.push_back(detection.bbox);
                scores.push_back(detection.confidence);
            }

            // 使用 OpenCV 的 NMS 函数
            cv::dnn::NMSBoxes(boxes, scores, 0.0f, iou_threshold, indices);

            // 过滤出应用NMS后剩余的检测框
            std::vector<YOLOResult> result;
            for (int idx : indices)
            {
                result.push_back(detections[idx]);
            }

            return result;
        }

        YOLOFrameResult v11(
            std::span<const float> infer_output,
            float confidence_threshold,
            int img_width, int img_height,
            int orig_width, int orig_height) const
        {
            std::vector<YOLOResult> detections{0};
            const size_t num_features = shapes_[1];
            const size_t num_detections = shapes_[2];
            const int num_classes = static_cast<int>(num_features) - 4;
            for (size_t d = 0; d < num_detections; ++d)
            {
                const float cx = infer_output[0 * num_detections + d];
                const float cy = infer_output[1 * num_detections + d];
                const float ow = infer_output[2 * num_detections + d];
                const float oh = infer_output[3 * num_detections + d];
                int object_class_id = 0;
                float object_confidence = infer_output[4 * num_detections + d];
                // Find the class with the highest confidence
                for (int class_id = 1; class_id < num_classes; ++class_id)
                {
                    float class_confidence = infer_output[(4 + class_id) * num_detections + d];
                    if (class_confidence > object_confidence)
                    {
                        object_confidence = class_confidence;
                        object_class_id = class_id;
                    }
                }
                if (object_confidence > confidence_threshold)
                {
                    auto& class_name = this->class_names_[object_class_id];
                    // const int x = static_cast<int>(cx - ow * 0.5f);
                    // const int y = static_cast<int>(cy - oh * 0.5f);
                    // const int width = static_cast<int>(ow);
                    // const int height = static_cast<int>(oh);
                    const auto x = static_cast<float>(cx - ow * 0.5f);
                    const auto y = static_cast<float>(cy - oh * 0.5f);
                    const auto width = static_cast<float>(ow);
                    const auto height = static_cast<float>(oh);
                    auto scaled_rect = cv::Rect2f(x, y, width, height);
                    auto origin_rect = ScaleCoords(cv::Size2f(img_width, img_height),
                                                   scaled_rect,
                                                   cv::Size2f(orig_width, orig_height),
                                                   true);
                    detections.emplace_back(object_class_id, origin_rect, object_confidence,
                                            class_name);
                }
            }
            auto nmsed_detections = ApplyNMS(detections, 0.3f);
            return YOLOFrameResult{std::move(nmsed_detections)};
        }

        YOLOFrameResult v10(
            std::span<const float> infer_output,
            float confidence_threshold,
            int img_width, int img_height,
            int orig_width, int orig_height) const
        {
            std::vector<YOLOResult> detections;
            const int num_detections = infer_output.size() / 6;

            // Calculate scale and padding factors
            float width_scale = img_width / (float)orig_width;
            float height_scale = img_height / (float)orig_height;
            int new_width = static_cast<int>(orig_width * width_scale);
            int new_height = static_cast<int>(orig_height * height_scale);
            int pad_x = (img_width - new_width) / 2;
            int pad_y = (img_height - new_height) / 2;

            detections.reserve(num_detections);
            for (int i = 0; i < num_detections; ++i)
            {
                float left = infer_output[i * 6 + 0];
                float top = infer_output[i * 6 + 1];
                float right = infer_output[i * 6 + 2];
                float bottom = infer_output[i * 6 + 3];
                float confidence = infer_output[i * 6 + 4];
                int class_id = static_cast<int>(infer_output[i * 6 + 5]);

                if (confidence >= confidence_threshold)
                {
                    // Remove padding and rescale to original image dimensions
                    left = (left - pad_x) / width_scale;
                    top = (top - pad_y) / height_scale;
                    right = (right - pad_x) / width_scale;
                    bottom = (bottom - pad_y) / height_scale;

                    int x = static_cast<int>(left);
                    int y = static_cast<int>(top);
                    int width = static_cast<int>(right - left);
                    int height = static_cast<int>(bottom - top);
                    detections.emplace_back(class_id, cv::Rect(x, y, width, height), confidence,
                                            this->class_names_[class_id]);
                }
            }
            auto nmsed_detections = ApplyNMS(detections, 0.3f);
            return YOLOFrameResult{nmsed_detections};
        }

        FilterResult operator ()(
            std::span<const float> infer_output,
            float confidence_threshold,
            int img_width, int img_height,
            int orig_width, int orig_height) const
        {
            if (version_ == YOLOVersion::kV10)
            {
                return v10(infer_output, confidence_threshold, img_width, img_height, orig_width, orig_height);
            }
            if (version_ == YOLOVersion::kV11)
            {
                return v11(infer_output, confidence_threshold, img_width, img_height, orig_width, orig_height);
            }
            return std::unexpected(InferError{
                InferErrorCode::kParameterError,
                std::format("unsupported version: {}", magic_enum::enum_name(version_))
            });
        }
    };

    class InferYOLOONNXImpl : public InferYOLO
    {
        std::unique_ptr<Ort::Session> session_;
        YOLOVersion version_;
        YOLOFilter filter_;
        Ort::Allocator allocator_;
        Ort::IoBinding io_binding_;
        ONNXTensorElementDataType input_value_type_, output_value_type_;
        std::string input_name_, output_name_;
        cv::Size2i input_size_;
        Ort::Value input_value_;
        std::vector<std::string> class_names_;

        Mat letterbox_resized_image, letterbox_dst_image, preprocess_image;
        std::vector<Mat> channels{3};
        std::vector<float> output_fp32_cache;

    protected:
        Mat& Letterbox(const Mat& src,
                       const Size& target_size,
                       const Scalar& color = cv::Scalar(0, 0, 0))
        {
            float scale = std::min(
                static_cast<float>(target_size.width) / static_cast<float>(src.cols),
                static_cast<float>(target_size.height) / static_cast<float>(src.rows));
            int new_width = static_cast<int>(static_cast<float>(src.cols) * scale);
            int new_height = static_cast<int>(static_cast<float>(src.rows) * scale);
            resize(src, letterbox_resized_image,
                   cv::Size(new_width, new_height));
            if (letterbox_dst_image.rows != target_size.height ||
                letterbox_dst_image.cols != target_size.width)
            {
                letterbox_dst_image = cv::Mat::zeros(target_size.height,
                                                     target_size.width,
                                                     src.type());
            }
            letterbox_dst_image.setTo(color);
            int top = (target_size.height - new_height) / 2;
            int left = (target_size.width - new_width) / 2;
            letterbox_resized_image.copyTo(
                letterbox_dst_image(cv::Rect(left, top,
                                             letterbox_resized_image.cols,
                                             letterbox_resized_image.
                                             rows)));

            return letterbox_dst_image;
        }

        Mat& PreProcess(const Mat& image) noexcept
        {
            if (preprocess_image.cols != image.rows || preprocess_image.rows != image.cols)
            {
                preprocess_image = Mat::zeros(image.rows, image.cols,CV_32F);
            }
            auto& dst_image = Letterbox(image, input_size_);
            dst_image.convertTo(preprocess_image, CV_32F, 1.0 / 255);
            cvtColor(preprocess_image, preprocess_image, cv::COLOR_BGR2RGB);
            //chw2hwc
            split(preprocess_image, channels);
            const size_t num_pixels = input_size_.width * input_size_.height;
            auto preprocess_image_ptr = preprocess_image.ptr<float>();
#pragma omp parallel for num_threads(3)
            for (int c = 0; c < 3; ++c)
            {
                const auto src = channels[c].data;
                auto dst = preprocess_image_ptr + num_pixels * c;
                std::memcpy(dst, src, num_pixels * sizeof(float));
            }
            return preprocess_image;
        }

    public:
        InferYOLOONNXImpl(InferContextONNXRuntime& ort_ctx,
                          std::unique_ptr<Ort::Session>&& session,
                          Ort::Allocator&& allocator,
                          YOLOVersion version,
                          std::vector<std::string> class_names):
            session_(std::move(session)),
            version_(version),
            filter_(version, class_names, session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()),
            allocator_(std::move(allocator)),
            io_binding_(*session_),
            input_value_(nullptr),
            class_names_(std::move(class_names))
        {
            auto input_info = session_->GetInputTypeInfo(0);
            auto type_and_shape_info = input_info.GetTensorTypeAndShapeInfo();
            auto ele_type = type_and_shape_info.GetElementType();
            auto shape = type_and_shape_info.GetShape();
            input_size_.width = static_cast<int>(shape[3]);
            input_size_.height = static_cast<int>(shape[2]);
            auto input_name_ptr = session_->GetInputNameAllocated(0, allocator_);
            input_name_ = std::string(input_name_ptr.get());
            auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator_);
            output_name_ = std::string(output_name_ptr.get());
            // read shapes from model metadata
            input_value_ = Ort::Value::CreateTensor(allocator_, shape.data(), shape.size(),
                                                    ele_type);
            io_binding_.BindOutput(output_name_ptr.get(), ort_ctx.env_memory_info());
            input_value_type_ = input_value_.GetTensorTypeAndShapeInfo().GetElementType();
            output_value_type_ = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
        }

        YOLOVersion version() const noexcept override
        {
            return version_;
        }

        const std::vector<std::string>& class_names() const noexcept override
        {
            return class_names_;
        }

        RunResult Run(const Mat& image, float confidence_threshold) noexcept override
        {
            //PreProcess
            Mat& hwc = PreProcess(image);
            auto hwc_ptr = hwc.ptr<float>();
            auto hwc_size = hwc.channels() * hwc.cols * hwc.rows;
            auto hwc_size_bytes = hwc_size * sizeof(float);
            if (input_value_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
            {
                Cvt::cvt(std::span(hwc_ptr, hwc_size), input_value_.GetTensorMutableData<Ort::Float16_t>());
            }
            else if (input_value_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                std::memcpy(input_value_.GetTensorMutableData<float>(), hwc_ptr, hwc_size_bytes);
            }
            else
            {
                return std::unexpected(InferError{
                    InferErrorCode::kParameterError,
                    std::format("unsupported input value type:{}", magic_enum::enum_name(input_value_type_))
                });
            }
            io_binding_.BindInput(input_name_.data(), input_value_);
            Ort::RunOptions run_options;
            session_->Run(run_options, io_binding_);
            auto output_values = io_binding_.GetOutputValues();
            auto& output_value = output_values[0];
            //fp16 fp32
            auto output_shape = output_value.GetTensorTypeAndShapeInfo().GetShape();
            auto output_size =
                std::accumulate(output_shape.begin(), output_shape.end(), 1llu, std::multiplies());
            // auto output_size_bytes = output_size * sizeof(float);
            const float* output_data = nullptr;
            if (output_value_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
            {
                if (output_fp32_cache.size() != output_size)output_fp32_cache.resize(output_size);
                auto output_data_mut = output_fp32_cache.data();
                output_data = output_data_mut;
                Cvt::cvt(std::span(output_value.GetConst().GetTensorData<Ort::Float16_t>(), output_size),
                         output_data_mut);
            }
            else if (output_value_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                output_data = output_value.GetConst().GetTensorData<float>();
            }
            else
            {
                return std::unexpected(InferError{
                    InferErrorCode::kParameterError,
                    std::format("unsupported output value type:{}", magic_enum::enum_name(output_value_type_))
                });
            }
            auto result = filter_(std::span(output_data, output_size), confidence_threshold,
                                  this->input_size_.width, this->input_size_.height,
                                  image.cols, image.rows);
            return result;
        }
    };

    namespace
    {
        using InferYOLOFactory = std::function<InferYOLO::CreateResult(
            InferContext& context, std::span<uint8_t> data, YOLOVersion version, size_t device_id)>;

        std::map<InferFramework, InferYOLOFactory> infer_yolo_factories{
            std::make_pair(InferFramework::kONNXRUNTIME,
                           [](InferContext& context, std::span<uint8_t> data,
                              YOLOVersion version, size_t device_id)-> InferYOLO::CreateResult
                           {
                               auto& ort_ctx = dynamic_cast<InferContextONNXRuntime&>(context);
                               auto& env = ort_ctx.env();
                               Ort::SessionOptions session_options;
                               session_options.SetGraphOptimizationLevel(
                                   GraphOptimizationLevel::ORT_ENABLE_ALL);
                               session_options.DisableProfiling();
                               session_options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators,
                                                              "1");
                               session_options.AddConfigEntry(kOrtSessionOptionsConfigAllowInterOpSpinning,
                                                              "0");
                               session_options.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning,
                                                              "0");
                               if (context.execution_provider() == InferEP::kDML)
                               {
#ifndef VISION_SIMPLE_WITH_DML
                                   return std::unexpected{
                                       InferError{
                                           InferErrorCode::kRuntimeError,
                                           std::format("unsupported execution_provider:{}",
                                                       magic_enum::enum_name(context.execution_provider()))
                                       }
                                   };
#else
                                   session_options.SetInterOpNumThreads(1);
                                   session_options.SetIntraOpNumThreads(1);
                                   // session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);
                                   session_options.SetExecutionMode(ORT_SEQUENTIAL);
                                   session_options.DisableMemPattern();
                                   session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
                                   const OrtDmlApi* dml_api = nullptr;
                                   Ort::GetApi().GetExecutionProviderApi("DML",ORT_API_VERSION,
                                                                         reinterpret_cast<const void**>(&
                                                                             dml_api));
                                   dml_api->SessionOptionsAppendExecutionProvider_DML(
                                       session_options, static_cast<int>(device_id));
#endif
                               }
                               try
                               {
                                   auto session = std::make_unique<Ort::Session>(
                                       env, data.data(), data.size_bytes(), session_options);
                                   Ort::Allocator allocator{*session, ort_ctx.env_memory_info()};
                                   // read class_names_
                                   auto GetClassNames = [&](
                                       const Ort::ModelMetadata& metadata)-> std::optional<std::vector<std::string>>
                                   {
                                       auto names_str = std::string(
                                           metadata.LookupCustomMetadataMapAllocated("names", allocator).get());
                                       std::regex reg{R"('([^']+)')"};
                                       auto begin =
                                           std::sregex_iterator(names_str.begin(), names_str.end(), reg);
                                       auto end = std::sregex_iterator();
                                       std::vector<std::string> class_names;
                                       for (auto i = begin; i != end; ++i)
                                       {
                                           auto result = i->operator[](1).str();
                                           class_names.emplace_back(result);
                                       }
                                       return std::move(class_names);
                                   };
                                   auto class_names_opt = GetClassNames(session->GetModelMetadata());
                                   if (!class_names_opt)
                                   {
                                       return std::unexpected{
                                           InferError{
                                               InferErrorCode::kModelError,
                                               std::format("unable to find class names from model metadata")
                                           }
                                       };
                                   }
                                   return std::make_unique<InferYOLOONNXImpl>(
                                       ort_ctx, std::move(session),
                                       std::move(allocator), version,
                                       std::move(*class_names_opt));
                               }
                               catch (std::exception& e)
                               {
                                   return std::unexpected{
                                       InferError{
                                           InferErrorCode::kRuntimeError,
                                           std::format("unable to create ONNXRuntime Session:{}", e.what())
                                       }
                                   };
                               }
                           })
        };
    }

    InferYOLO::CreateResult InferYOLO::Create(InferContext& context, std::span<uint8_t> data,
                                              YOLOVersion version, size_t device_id) noexcept
    {
        try
        {
            return infer_yolo_factories.at(context.framework())(context, data, version, device_id);
        }
        catch (std::exception& _)
        {
            return std::unexpected{
                InferError{
                    InferErrorCode::kParameterError, std::format("unsupported context:framework({}),ep({})",
                                                                 magic_enum::enum_name(context.framework()),
                                                                 magic_enum::enum_name(context.execution_provider()))
                }
            };
        }
        // return std::unexpected{InferError{InferErrorCode::kUnknownError, "InferYOLO::Create"}};
    }
}
