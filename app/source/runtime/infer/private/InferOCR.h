#pragma once
#include "Infer.h"

namespace vision_simple
{
    class InferOCROrtPaddleImpl final : public InferOCR
    {
        struct Impl;
        std::unique_ptr<Impl> impl_;

    public:
        explicit InferOCROrtPaddleImpl(OCRModelType model_type);

        OCRModelType model_type() const noexcept override;
        RunResult Run(const cv::Mat& image, float confidence_threshold) noexcept override;
    };
}
