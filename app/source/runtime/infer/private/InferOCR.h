#pragma once
#include "Infer.h"

namespace vision_simple
{
    class InferContextORT;

    class InferOCROrtPaddleImpl final : public InferOCR
    {
        struct Impl;
        std::unique_ptr<Impl> impl_;

    public:
        explicit InferOCROrtPaddleImpl(InferContextORT& ort_ctx, OCRModelType model_type,
                                       std::map<int, std::string> char_dict,
                                       std::unique_ptr<Ort::Session> det,
                                       std::unique_ptr<Ort::Session> rec);

        OCRModelType model_type() const noexcept override;
        RunResult Run(const cv::Mat& image, float confidence_threshold) noexcept override;
    };
}
