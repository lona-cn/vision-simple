#include "InferOCR.h"

#include <codecvt>
#include <magic_enum.hpp>
#include <memory_resource>
#include <numeric>

#include "InferORT.h"
#include "VisionHelper.hpp"

namespace
{
    // std::pmr::monotonic_buffer_resource mr{114514};
    double computeIOU(const cv::Rect& rect1, const cv::Rect& rect2)
    {
        // 计算交集
        cv::Rect intersection = rect1 & rect2;

        // 计算并集
        cv::Rect union_rect = rect1 | rect2;

        // 交集面积
        double intersectionArea = intersection.area();

        // 并集面积
        double unionArea = union_rect.area();

        // 计算IOU
        return intersectionArea / unionArea;
    }

    // 根据IOU阈值进行过滤
    std::vector<cv::Rect> filterByIOU(std::vector<cv::Rect>& boxes, double iou_threshold)
    {
        std::vector<cv::Rect> filteredBoxes;

        for (size_t i = 0; i < boxes.size(); i++)
        {
            bool keep = true;
            for (size_t j = 0; j < filteredBoxes.size(); j++)
            {
                // 如果当前框和已过滤框的IOU大于阈值，丢弃当前框
                if (computeIOU(boxes[i], filteredBoxes[j]) > iou_threshold)
                {
                    keep = false;
                    break;
                }
            }
            if (keep)
            {
                filteredBoxes.push_back(boxes[i]);
            }
        }

        // 将过滤后的框替换回原始框列表
        return filteredBoxes;
    }
}

vision_simple::InferOCR::CreateResult vision_simple::InferOCR::Create(InferContext& context,
                                                                      std::map<int, std::string> char_dict,
                                                                      std::span<uint8_t> det_data,
                                                                      std::span<uint8_t> rec_data,
                                                                      OCRModelType model_type,
                                                                      size_t device_id) noexcept
{
    auto& ort_ctx = dynamic_cast<InferContextORT&>(context);
    auto det = ort_ctx.CreateSession(det_data, device_id);
    if (!det)return std::unexpected(std::move(det.error()));
    auto rec = ort_ctx.CreateSession(rec_data, device_id);
    if (!rec)return std::unexpected(std::move(rec.error()));
    return std::make_unique<
        InferOCROrtPaddleImpl>(ort_ctx, model_type, std::move(char_dict), std::move(*det), std::move(*rec));
}

struct vision_simple::InferOCROrtPaddleImpl::Impl
{
    VisionHelper vision_helper;
    OCRModelType model_type;
    std::map<int, std::string> char_dict;
    std::unique_ptr<Ort::Session> det, rec;
    // Ort::MemoryInfo& memory_info;
    Ort::Allocator det_allocator, rec_allocator;
    Ort::IoBinding det_io_binding, rec_io_binding;
    Ort::Value det_input_tensor, rec_input_tensor;
    std::string det_input_name, det_output_name,
                rec_input_name, rec_output_name;
    Ort::MemoryInfo rec_memory_info;
    cv::Mat preprocessed_image;

    explicit Impl(InferContextORT& ort_ctx,
                  OCRModelType model_type, std::map<int, std::string> char_dict, std::unique_ptr<Ort::Session> det,
                  std::unique_ptr<Ort::Session> rec)
        : model_type(model_type), char_dict(std::move(char_dict)), det(std::move(det)), rec(std::move(rec)),
          det_allocator(*this->det, ort_ctx.env_memory_info()),
          rec_allocator(*this->rec, ort_ctx.env_memory_info()),
          det_io_binding(*this->det), rec_io_binding(*this->rec),
          det_input_tensor{nullptr}, rec_input_tensor{nullptr},
          det_input_name(std::string(this->det->GetInputNameAllocated(0, det_allocator).get())),
          det_output_name(std::string(this->det->GetOutputNameAllocated(0, det_allocator).get())),
          rec_input_name(std::string(this->rec->GetInputNameAllocated(0, rec_allocator).get())),
          rec_output_name(std::string(this->rec->GetOutputNameAllocated(0, rec_allocator).get())),
          rec_memory_info(Ort::MemoryInfo::CreateCpu(ort_ctx.env_memory_info().GetAllocatorType(),
                                                     ort_ctx.env_memory_info().GetMemoryType()))
    {
        det_io_binding.BindOutput(det_output_name.c_str(), ort_ctx.env_memory_info());
        rec_io_binding.BindOutput(rec_output_name.c_str(), rec_memory_info);
    }

    template <typename T>
        requires std::is_arithmetic_v<T>
    static T PadLength(T length, T pad = static_cast<T>(32))
    {
        auto remainder = length % pad;
        auto pad_length = remainder ? pad - remainder : 0;
        return length + pad_length;
    }

    cv::Mat& DetPreProcess(const cv::Mat& image) noexcept
    {
        const auto target_size = cv::Size{PadLength(image.cols), PadLength(image.rows)};
        auto& padded_img = vision_helper.Letterbox(image, target_size);
        preprocessed_image = cv::Mat::zeros(target_size, CV_8UC3);
        vision_helper.HWC2CHW_BGR2RGB<uint8_t>(padded_img, preprocessed_image);
        return preprocessed_image;
    }

    template <typename T>
    void DetPostProcess(std::span<const T> infer_output, float confidence_threshold) noexcept
    {
        //filter
    }

    cv::Mat RecPreProcess(const cv::Mat& image)
    {
        // resize to height with 48
        auto scale = 48.0f / image.rows;
        auto width = scale * image.cols;
        cv::Size output_size{static_cast<int>(width), 48};
        cv::Mat resized_image = vision_helper.Letterbox(image, output_size);
        // cv::imshow("Output", resized_image);
        // cv::waitKey(0);
        // cv::imwrite("R:/aaaaa.png",resized_image);
        // cv::resize(image, resized_image, output_size);
        cv::Mat output_image{resized_image.rows, resized_image.cols,CV_32FC3};
        resized_image.convertTo(output_image,CV_32F, 1.f / 255.f);
        output_image = (output_image - 0.5) / 0.5;
        vision_helper.HWC2CHW_BGR2RGB<float>(output_image, output_image);
        return output_image.clone();
    }

    static size_t GetTensorSize(const Ort::Value& value) noexcept
    {
        auto shapes = value.GetTensorTypeAndShapeInfo().GetShape();
        return std::accumulate(shapes.begin(), shapes.end(), 1llu, std::multiplies());
    }

    RunResult Run(const cv::Mat& image,
                  float confidence_threshold) noexcept
    {
        std::vector<std::vector<cv::Point>> contours, filtered_contours;
        cv::Size input_image_size;
        {
            auto& preprocessed_image = DetPreProcess(image);
            cv::Mat input_image{preprocessed_image.rows, preprocessed_image.cols,CV_32FC3};
            preprocessed_image.convertTo(input_image,CV_32F, 1.f / 255.f);
            for (int row = 0; row < input_image_size.height; ++row)
                for (int col = 0; col < input_image_size.width; ++col)
                {
                    auto stride = input_image_size.width * input_image_size.height;
                    auto ptr = input_image.ptr<float>() + row * col;
                    *ptr = (*ptr - 0.485) / 0.229;
                    *(ptr + stride) = (*(ptr + stride) - 0.456) / 0.224;
                    *(ptr + stride * 2) = (*(ptr + stride * 2) - 0.406) / 0.225;
                }
            int64_t input_image_shape[4] = {1, 3, input_image.rows, input_image.cols};
            //uint8_t
            auto input_size = preprocessed_image.channels() * preprocessed_image.rows * preprocessed_image.cols;
            auto input_size_bytes = preprocessed_image.channels() * preprocessed_image.rows * preprocessed_image.cols *
                sizeof(float);
            det_input_tensor = Ort::Value::CreateTensor(det_allocator, input_image_shape,
                                                        sizeof(input_image_shape) / sizeof(decltype(input_image_shape[0]
                                                        )),
                                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            auto input_tensor_size = GetTensorSize(det_input_tensor), input_tensor_size_bytes = input_tensor_size *
                     sizeof(float);
            std::memcpy(det_input_tensor.GetTensorMutableData<float>(), input_image.ptr<float>(), input_size_bytes);
            Ort::RunOptions run_options;
            det_io_binding.BindInput(det_input_name.c_str(), det_input_tensor);
            det->Run(run_options, det_io_binding);
            const auto& onames = det_io_binding.GetOutputNames();
            const auto& ovalues = det_io_binding.GetOutputValues();
            auto& output_tensor = ovalues[0];
            auto output_type = output_tensor.GetTensorTypeAndShapeInfo().GetElementType();
            auto output_type_str = magic_enum::enum_name(output_type);
            auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
            auto output_tensor_size = GetTensorSize(output_tensor);
            auto output_ptr = output_tensor.GetConst().GetTensorData<float>();
            auto img = cv::Mat{(int)output_shape[2], (int)output_shape[3],CV_32FC1, (void*)output_ptr};
            cv::Mat gray{img.rows, img.cols,CV_8UC1};
            img.convertTo(gray, CV_8UC1);
            cv::Mat dilated;
            int kernel_size = 6;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
            cv::dilate(gray, dilated, kernel);
            for (auto i = 0; i < 2; ++i)
            {
                cv::dilate(dilated, dilated, kernel);
            }
            // cv::imshow("Dilated", dilated);
            cv::findContoursLinkRuns(dilated, contours);
            // cv::findContours(gray, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
            double minArea = 12. * 12.; // 设置最小面积阈值
            for (const auto& contour : contours)
            {
                if (contourArea(contour) > minArea)
                {
                    filtered_contours.push_back(contour); // 添加满足条件的轮廓
                }
            }

            input_image_size = cv::Size2f{
                static_cast<float>(input_image.cols),
                static_cast<float>(input_image.rows)
            };
        }
        std::vector<cv::Rect> rects;
        for (size_t i = 0; i < contours.size(); i++)
        {
            // 使用 boundingRect 拟合矩形
            auto rect{boundingRect(contours[i])};
            if (rect.area() > 8 * 8)
            {
                rects.emplace_back(rect);
            }
            // 绘制矩形
            // rectangle(img, rect, Scalar(0, 255, 0), 2);
        }

        // cv::Mat result(img.rows, img.cols, CV_8UC3);
        // cv::cvtColor(img, result, cv::COLOR_GRAY2BGR);
        cv::Mat result = image.clone();
        auto filtered_boxes = filterByIOU(rects, 0.3);
        auto origin_image_size = cv::Size2f{
            static_cast<float>(image.cols),
            static_cast<float>(image.rows)
        };
        for (auto& filtered_box : filtered_boxes)
        {
            filtered_box = VisionHelper::ScaleCoords(input_image_size, filtered_box,
                                                     origin_image_size, true);
            auto i = 0;
        }
        // for (size_t i = 0; i < filtered_contours.size(); i++)
        // {
        // cv::Scalar color = cv::Scalar(0, 255, 0); // 绿色绘制轮廓
        // drawContours(result, filtered_contours, (int)i, color, 2, cv::LINE_8, hierarchy, 0);
        // }
        for (const auto& rect : filtered_boxes)
        {
            cv::rectangle(result, rect, cv::Scalar(255, 0, 0), 2);
        }
        // cv::imshow("Output", result);
        // cv::waitKey(0);
        // rec stage
        for (const auto& rect : filtered_boxes)
        {
            // preprocess
            // cv::Mat pure_img = cv::imread("R:/padding_im_t.png");
            // auto pp_rec_image = RecPreProcess(pure_img);
            // cv::imwrite("R:/bbb.png", pp_rec_image);
            auto pp_rec_image = RecPreProcess(image(rect).clone());
            auto r = pp_rec_image.rows, cols = pp_rec_image.cols,
                 channels = pp_rec_image.channels();
            int64_t input_image_shape[4] = {1, 3, pp_rec_image.rows, pp_rec_image.cols};
            int64_t rec_input_shape[4] = {1, 3, pp_rec_image.rows, pp_rec_image.cols};
            rec_input_tensor = Ort::Value::CreateTensor(rec_allocator, rec_input_shape, 4,
                                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            auto input_tensor_size = GetTensorSize(rec_input_tensor);
            size_t input_image_size_bytes = pp_rec_image.cols * pp_rec_image.rows * pp_rec_image.channels() * sizeof(
                float);
            std::memcpy(rec_input_tensor.GetTensorMutableData<float>(), pp_rec_image.ptr<float>(),
                        input_image_size_bytes);
            rec_io_binding.BindInput(rec_input_name.c_str(), rec_input_tensor);
            rec_io_binding.BindOutput(rec_output_name.c_str(), rec_memory_info);
            // predict string
            Ort::RunOptions run_options{};
            rec->Run(run_options, rec_io_binding);
            const auto& outputs = rec_io_binding.GetOutputValues();
            auto& output_tensor = outputs[0];
            auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
            auto output_size = GetTensorSize(output_tensor);
            auto output_ptr = output_tensor.GetTensorData<float>();
            std::vector<std::pair<int, double>> idx_scores;
            for (auto i{0}; i < output_shape[1]; ++i)
            {
                auto find_max_value_and_index = [](const std::span<const float> vec)-> std::pair<float, size_t>
                {
                    // 使用std::max_element找到最大值的迭代器
                    auto max_it = std::max_element(vec.cbegin(), vec.cend());
                    // 如果vector为空，返回一个无效的值
                    if (max_it == vec.cend())
                    {
                        return {std::numeric_limits<float>::lowest(), static_cast<size_t>(-1)};
                    }
                    // 获取最大值及其索引
                    float max_value = *max_it;
                    size_t index = std::distance(vec.begin(), max_it);

                    return {max_value, index};
                };
                auto [max_score,max_idx] = find_max_value_and_index(
                    std::span(output_ptr + i * output_shape[2], output_shape[2]));
                idx_scores.emplace_back(max_idx, max_score);
            }
            std::vector<std::string> line;
            for (auto& idx : idx_scores | std::views::keys)
            {
                if (idx == 0)continue;
                auto& c = char_dict[idx - 1];
                // std::cout << c;
                printf("%s", c.c_str());
                line.emplace_back(c);
            }
            puts("");
            // std::cout << ss.str() << std::endl;
            // ReadAll("F:/workspace/cpp/vision-simple/app/source/programs/demo/assets/ppocr_keys_v1.txt");
            // auto keys = std::move(keys_data.value());
            // cv::imshow("rec", pp_rec_image);
            // cv::waitKey(0);
            auto i = 0;
        }


        return std::unexpected{InferError::Unimplemented()};
    }
};

vision_simple::InferOCROrtPaddleImpl::InferOCROrtPaddleImpl(InferContextORT& ort_ctx, OCRModelType model_type,
                                                            std::map<int, std::string> char_dict,
                                                            std::unique_ptr<Ort::Session> det,
                                                            std::unique_ptr<Ort::Session> rec):
    impl_(std::make_unique<Impl>(ort_ctx, model_type, char_dict, std::move(det), std::move(rec)))
{
}

vision_simple::OCRModelType vision_simple::InferOCROrtPaddleImpl::model_type() const noexcept
{
    return this->impl_->model_type;
}

vision_simple::InferOCR::RunResult vision_simple::InferOCROrtPaddleImpl::Run(const cv::Mat& image,
                                                                             float confidence_threshold) noexcept
{
    return this->impl_->Run(image, confidence_threshold);
}
