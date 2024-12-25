#pragma once
#include "../Infer.h"
#include "InferORT.h"
#include <magic_enum.hpp>
#include <opencv2/opencv.hpp>

namespace vision_simple
{
    class YOLOFilter
    {
        YOLOVersion version_;
        std::vector<std::string> class_names_;
        std::vector<long long> shapes_;

    public:
        using FilterResult = InferResult<YOLOFrameResult>;

        explicit YOLOFilter(YOLOVersion version, std::vector<std::string> class_names,
                            std::vector<long long> shapes);

        YOLOVersion version() const noexcept;

        static cv::Rect ScaleCoords(const cv::Size& imageShape, cv::Rect coords,
                                    const cv::Size& imageOriginalShape, bool p_Clip);

        static std::vector<YOLOResult> ApplyNMS(
            const std::vector<YOLOResult>& detections, float iou_threshold);

        YOLOFrameResult v11(
            std::span<const float> infer_output,
            float confidence_threshold,
            int img_width, int img_height,
            int orig_width, int orig_height) const;

        YOLOFrameResult v10(
            std::span<const float> infer_output,
            float confidence_threshold,
            int img_width, int img_height,
            int orig_width, int orig_height) const;

        FilterResult operator ()(
            std::span<const float> infer_output,
            float confidence_threshold,
            int img_width, int img_height,
            int orig_width, int orig_height) const;
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

        cv::Mat letterbox_resized_image_, letterbox_dst_image_, preprocess_image_;
        std::vector<cv::Mat> channels_{3};
        std::vector<float> output_fp32_cache_;

    protected:
        cv::Mat& Letterbox(const cv::Mat& src,
                           const cv::Size& target_size,
                           const cv::Scalar& color = cv::Scalar(0, 0, 0));

        cv::Mat& PreProcess(const cv::Mat& image) noexcept;

    public:
        InferYOLOONNXImpl(InferContextONNXRuntime& ort_ctx,
                          std::unique_ptr<Ort::Session>&& session,
                          Ort::Allocator&& allocator,
                          YOLOVersion version,
                          std::vector<std::string> class_names);

        YOLOVersion version() const noexcept override;

        const std::vector<std::string>& class_names() const noexcept override;

        RunResult Run(const cv::Mat& image, float confidence_threshold) noexcept override;
    };
}
