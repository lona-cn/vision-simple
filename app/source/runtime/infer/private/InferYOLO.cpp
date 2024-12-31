#include "InferYOLO.h"

#ifdef VISION_SIMPLE_WITH_DML
#include <dml_provider_factory.h>
#endif

// #ifdef VISION_SIMPLE_WITH_CUDA
// #include <cuda_provider_factory.h>
// #endif
#include <magic_enum.hpp>
#include <numeric>
#include <onnxruntime_float16.h>
#include <onnxruntime_session_options_config_keys.h>
#include <onnxruntime_run_options_config_keys.h>
#include <regex>

using namespace std;
using namespace cv;
using namespace vision_simple;

namespace
{
    using InferYOLOFactory = std::function<InferYOLO::CreateResult(
        InferContext& context, std::span<uint8_t> data, YOLOVersion version, size_t device_id)>;

    std::map<InferFramework, InferYOLOFactory> infer_yolo_factories{
        std::make_pair(InferFramework::kONNXRUNTIME,
                       [](InferContext& context, std::span<uint8_t> data,
                          YOLOVersion version, size_t device_id)-> InferYOLO::CreateResult
                       {
                           auto& ort_ctx = dynamic_cast<InferContextORT&>(context);
                           try
                           {
                               auto session_opt = ort_ctx.CreateSession(data, device_id);
                               if (!session_opt)return std::unexpected{std::move(session_opt.error())};
                               Ort::Allocator allocator{**session_opt, ort_ctx.env_memory_info()};
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
                               auto class_names_opt = GetClassNames((*session_opt)->GetModelMetadata());
                               if (!class_names_opt)
                               {
                                   return std::unexpected{
                                       InferError{
                                           InferErrorCode::kModelError,
                                           std::format("unable to find class names from model metadata")
                                       }
                                   };
                               }
                               return std::make_unique<InferYOLOOrtImpl>(
                                   ort_ctx, std::move(*session_opt),
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
    catch (std::exception& e)
    {
        return std::unexpected{
            InferError{
                InferErrorCode::kParameterError, std::format(
                    "unsupported context:framework({}),ep({}),with exception:{}",
                    magic_enum::enum_name(context.framework()),
                    magic_enum::enum_name(context.execution_provider()),
                    e.what())
            }
        };
    }
    // return std::unexpected{InferError{InferErrorCode::kUnknownError, "InferYOLO::Create"}};
}

YOLOFilter::YOLOFilter(YOLOVersion version, std::vector<std::string> class_names, std::vector<int64_t> shapes):
    version_(version),
    class_names_(std::move(class_names)),
    shapes_(std::move(shapes))
{
}

YOLOVersion YOLOFilter::version() const noexcept
{
    return version_;
}

std::vector<YOLOResult> YOLOFilter::ApplyNMS(const std::vector<YOLOResult>& detections, float iou_threshold) noexcept
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

YOLOFrameResult YOLOFilter::v11(std::span<const float> infer_output, float confidence_threshold, int img_width,
                                int img_height, int orig_width, int orig_height) const noexcept
{
    const float* infer_output_ptr = infer_output.data();
    std::vector<YOLOResult> detections{};
    // TODO:自适应预分配
    detections.reserve(256);
    const size_t num_features = shapes_[1];
    const size_t num_detections = shapes_[2];
    const int num_classes = static_cast<int>(num_features) - 4;
    for (int64_t d = 0; d < num_detections; ++d)
    {
        const float cx = infer_output_ptr[0 * num_detections + d];
        const float cy = infer_output_ptr[1 * num_detections + d];
        const float ow = infer_output_ptr[2 * num_detections + d];
        const float oh = infer_output_ptr[3 * num_detections + d];
        int object_class_id = 0;
        float object_confidence = infer_output_ptr[4 * num_detections + d];
        // Find the class with the highest confidence
        for (int class_id = 1; class_id < num_classes; ++class_id)
        {
            float class_confidence = infer_output_ptr[(4 + class_id) * num_detections + d];
            if (class_confidence > object_confidence)
            {
                object_confidence = class_confidence;
                object_class_id = class_id;
            }
        }
        if (object_confidence > confidence_threshold)
        {
            const float x = cx - ow * 0.5f;
            const float y = cy - oh * 0.5f;
            const float width = ow;
            const float height = oh;
            auto scaled_rect = cv::Rect2f(x, y, width, height);
            auto origin_rect = VisionHelper::ScaleCoords(cv::Size2f(img_width, img_height),
                                           scaled_rect,
                                           cv::Size2f(orig_width, orig_height),
                                           true);
            detections.emplace_back(object_class_id, origin_rect, object_confidence,
                                    this->class_names_[object_class_id]);
        }
    }
    auto nmsed_detections = ApplyNMS(detections, 0.3f);
    return YOLOFrameResult{std::move(nmsed_detections)};
}

YOLOFrameResult YOLOFilter::v10(std::span<const float> infer_output, float confidence_threshold, int img_width,
                                int img_height, int orig_width, int orig_height) const noexcept
{
    std::vector<YOLOResult> detections{};
    detections.reserve(256);
    const int num_detections = infer_output.size() / 6;

    // Calculate scale and padding factors
    float width_scale = img_width / (float)orig_width;
    float height_scale = img_height / (float)orig_height;
    int new_width = static_cast<int>(orig_width * width_scale);
    int new_height = static_cast<int>(orig_height * height_scale);
    int pad_x = (img_width - new_width) / 2;
    int pad_y = (img_height - new_height) / 2;
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
            detections.emplace_back(class_id, cv::Rect{x, y, width, height}, confidence,
                                    this->class_names_[class_id]);
        }
    }
    auto nmsed_detections = ApplyNMS(detections, 0.3f);
    return YOLOFrameResult{nmsed_detections};
}

YOLOFilter::FilterResult YOLOFilter::operator()(std::span<const float> infer_output, float confidence_threshold,
                                                int img_width, int img_height, int orig_width,
                                                int orig_height) const noexcept
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

cv::Mat& InferYOLOOrtImpl::PreProcess(const cv::Mat& image) noexcept
{
    auto& dst_image = vision_helper_.Letterbox(image, input_size_);
    //TODO: 根据模型输入，自动调整type
    vision_helper_.HWC2CHW_BGR2RGB<uint8_t>(dst_image, dst_image);
    return dst_image;
    // dst_image.convertTo(preprocessed_image_, CV_32F, 1.0 / 255);
    // return preprocessed_image_;
}

InferYOLOOrtImpl::InferYOLOOrtImpl(InferContextORT& ort_ctx, std::unique_ptr<Ort::Session>&& session,
                                   Ort::Allocator&& allocator, YOLOVersion version,
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

YOLOVersion InferYOLOOrtImpl::version() const noexcept
{
    return version_;
}

const std::vector<std::string>& InferYOLOOrtImpl::class_names() const noexcept
{
    return class_names_;
}

InferYOLO::RunResult InferYOLOOrtImpl::Run(const cv::Mat& image, float confidence_threshold) noexcept
{
    //PreProcess
    if (image.rows == 0 || image.cols == 0)
        return std::unexpected(InferError{
            InferErrorCode::kParameterError, "image is empty"
        });
    cv::Mat& chw = PreProcess(image);
    // auto hwc_ptr = hwc.ptr<float>();
    // auto hwc_size = hwc.channels() * hwc.cols * hwc.rows;
    // auto hwc_size_bytes = hwc_size * sizeof(float);
    if (input_value_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
        chw.convertTo(preprocessed_image_, CV_16F, 1.0 / 255);
        std::memcpy(input_value_.GetTensorMutableData<Ort::Float16_t>(),
                    preprocessed_image_.ptr<uint8_t>(),
                    chw.channels() * chw.rows * chw.cols * sizeof(Ort::Float16_t));
        // Cvt::cvt(std::span(hwc_ptr, hwc_size), input_value_.GetTensorMutableData<Ort::Float16_t>());
    }
    else if (input_value_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        chw.convertTo(preprocessed_image_, CV_32F, 1.0 / 255);
        std::memcpy(input_value_.GetTensorMutableData<Ort::Float16_t>(),
                    preprocessed_image_.ptr<float>(),
                    chw.channels() * chw.rows * chw.cols * sizeof(float));
        // std::memcpy(input_value_.GetTensorMutableData<float>(), hwc_ptr, hwc_size_bytes);
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
        if (output_fp32_cache_.size() != output_size)output_fp32_cache_.resize(output_size);
        auto output_data_mut = output_fp32_cache_.data();
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
