#pragma once
#include "../Infer.h"
#include "InferORT.h"
#include <magic_enum.hpp>
#include <opencv2/opencv.hpp>

#include "VisionHelper.hpp"

namespace vision_simple
{
    class YOLOFilter
    {
        YOLOVersion version_;
        std::vector<std::string> class_names_;
        std::vector<int64_t> shapes_;

        static std::vector<YOLOResult> ApplyNMS(
            const std::vector<YOLOResult>& detections, float iou_threshold) noexcept;

    public:
        using FilterResult = InferResult<YOLOFrameResult>;

        explicit YOLOFilter(YOLOVersion version, std::vector<std::string> class_names,
                            std::vector<int64_t> shapes);

        YOLOVersion version() const noexcept;

        YOLOFrameResult v11(
            std::span<const float> infer_output,
            float confidence_threshold,
            int img_width, int img_height,
            int orig_width, int orig_height) const noexcept;

        YOLOFrameResult v10(
            std::span<const float> infer_output,
            float confidence_threshold,
            int img_width, int img_height,
            int orig_width, int orig_height) const noexcept;

        FilterResult operator ()(
            std::span<const float> infer_output,
            float confidence_threshold,
            int img_width, int img_height,
            int orig_width, int orig_height) const noexcept;
    };


    class InferYOLOOrtImpl : public InferYOLO
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

        VisionHelper vision_helper_;
        cv::Mat preprocessed_image_;
        std::vector<float> output_fp32_cache_;

    protected:

        cv::Mat& PreProcess(const cv::Mat& image) noexcept;

    public:
        InferYOLOOrtImpl(InferContextORT& ort_ctx,
                         std::unique_ptr<Ort::Session>&& session,
                         Ort::Allocator&& allocator,
                         YOLOVersion version,
                         std::vector<std::string> class_names);

        YOLOVersion version() const noexcept override;

        const std::vector<std::string>& class_names() const noexcept override;

        RunResult Run(const cv::Mat& image, float confidence_threshold) noexcept override;
    };
}
