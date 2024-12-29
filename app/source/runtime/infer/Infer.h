#pragma once
#include <optional>
#include <expected>
#include <onnxruntime_cxx_api.h>
#include <span>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <immintrin.h>
#ifdef _WIN32
#include "DXInfo.hpp"
#endif
#include "InferError.hpp"


namespace vision_simple
{
    template <typename T>
    using InferResult = std::expected<T, InferError>;

    enum class InferFramework:uint8_t
    {
        kCUSTOM_FRAMEWORK = 0,
        kONNXRUNTIME = 1,
        kTVM
    };

    enum class InferEP:uint8_t
    {
        kCUSTOM_EP = 0,
        kCPU = 1,
        kDML,
        kCUDA,
        kTensorRT,
        kVulkan,
        kOpenGL,
        kOpenCL
    };

    class InferContext
    {
    public:
        using CreateResult = InferResult<std::unique_ptr<InferContext>>;
        InferContext() = default;
        virtual ~InferContext() = default;
        InferContext(const InferContext&) = delete;
        InferContext(InferContext&&) = default;
        InferContext& operator=(const InferContext&) = delete;
        InferContext& operator=(InferContext&&) = default;
        virtual InferFramework framework() const noexcept =0;
        virtual InferEP execution_provider() const noexcept =0;
        static CreateResult Create(InferFramework framework, InferEP ep) noexcept;
    };


    //--------YOLO--------
    enum class YOLOVersion:uint8_t
    {
        kVCustom = 0,
        kV10 = 10,
        kV11,
    };

    struct YOLOResult
    {
        int32_t class_id;
        cv::Rect bbox;
        float confidence;
        const std::string_view class_name;
    };

    struct YOLOFrameResult
    {
        std::vector<YOLOResult> results;
    };

    class InferYOLO
    {
    public:
        using CreateResult = InferResult<std::unique_ptr<InferYOLO>>;
        using RunResult = InferResult<YOLOFrameResult>;
        InferYOLO() = default;
        virtual ~InferYOLO() = default;
        InferYOLO(const InferYOLO&) = delete;
        InferYOLO(InferYOLO&&) = default;
        InferYOLO& operator=(const InferYOLO&) = delete;
        InferYOLO& operator=(InferYOLO&&) = default;
        virtual YOLOVersion version() const noexcept = 0;
        virtual const std::vector<std::string>& class_names() const noexcept =0;
        virtual RunResult Run(const cv::Mat& image, float confidence_threshold) noexcept = 0;
        static CreateResult Create(InferContext& context, std::span<uint8_t> data, YOLOVersion version,
                                   size_t device_id = 0) noexcept;

        template <typename T>
            requires std::is_arithmetic_v<T>
        static CreateResult Create(InferContext& context, std::span<T> data, YOLOVersion version,
                                   size_t device_id = 0) noexcept
        {
            return Create(context,
                          std::span(reinterpret_cast<uint8_t*>(data.data()), data.size_bytes()),
                          version, device_id);
        }
    };

    //--------OCR--------
    enum class OCRModelType:uint8_t
    {
        kPPOCRv3 = 0,
        kPPOCRv4,
        kEasyOCR
    };

    struct OCRResult
    {
        cv::Rect2i rect;
        float confidence;
        std::string line;
    };

    struct OCRFrameResult
    {
        std::vector<OCRResult> results;
    };

    class InferOCR
    {
    public:
        using CreateResult = InferResult<std::unique_ptr<InferOCR>>;
        using RunResult = InferResult<OCRFrameResult>;
        InferOCR() = default;
        virtual ~InferOCR() = default;
        InferOCR(const InferOCR&) = delete;
        InferOCR(InferOCR&&) = default;
        InferOCR& operator=(const InferOCR&) = delete;
        InferOCR& operator=(InferOCR&&) = default;
        virtual OCRModelType model_type() const noexcept =0;
        virtual RunResult Run(const cv::Mat& image, float confidence_threshold) noexcept = 0;
        static CreateResult Create(InferContext& context, std::span<uint8_t> data, OCRModelType model_type,
                                   size_t device_id = 0) noexcept;

        template <typename T>
            requires std::is_arithmetic_v<T>
        static CreateResult Create(InferContext& context, std::span<T> data, OCRModelType model_type,
                                   size_t device_id = 0) noexcept
        {
            return Create(context,
                          std::span(reinterpret_cast<uint8_t*>(data.data()), data.size_bytes()),
                          model_type, device_id);
        }
    };
}
