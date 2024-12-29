#include "InferOCR.h"
#include <memory_resource>

namespace
{
    std::pmr::monotonic_buffer_resource mr{114514};
}

struct vision_simple::InferOCROrtPaddleImpl::Impl
{
    OCRModelType model_type;
    cv::Mat padding_img;

    explicit Impl(OCRModelType model_type)
        : model_type(model_type)
    {
    }

    template <typename T>
        requires std::is_arithmetic_v<T>
    static T PadLength(T length, T pad = static_cast<T>(32))
    {
        return length / pad + (length % pad == 0 ? 0 : pad);
    }

    cv::Mat& PreProcess(const cv::Mat& image) noexcept
    {
        //padding
        const auto pad_width = PadLength(image.cols),
                   pad_height = PadLength(image.rows);
        if (padding_img.cols != pad_width || padding_img.rows != pad_height)
        {
            padding_img = cv::Mat{pad_height, pad_width, CV_32F, mr.allocate(pad_width * pad_height * 3)};
        }
        //TODO: letterbox
    }

    template <typename T>
    void PostProcess(std::span<const T> infer_output, float confidence_threshold) noexcept
    {
        //filter
    }

    RunResult Run(const cv::Mat& image,
                  float confidence_threshold) noexcept
    {
        return std::unexpected{InferError::Unimplemented()};
    }
};

vision_simple::InferOCROrtPaddleImpl::InferOCROrtPaddleImpl(OCRModelType model_type): impl_(
    std::make_unique<Impl>(model_type))
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
