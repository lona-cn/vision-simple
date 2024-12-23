#pragma once
#include <optional>
#include <expected>
#include <onnxruntime_cxx_api.h>
#include <span>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace vision_simple
{
    enum class InferErrorCode:uint8_t
    {
        kOK = 0,
        kIOError,
        kDeviceError,
        kModelError,
        kParameterError,
        kRangeError,
        kRuntimeError,
        kCustomError,
        kUnknownError,
    };

    class InferError
    {
    public:
        InferErrorCode code;
        std::string message;
        std::unique_ptr<void*> user_data;

        InferError(InferErrorCode code, std::string message, std::unique_ptr<void*> user_data = nullptr)
            : code(code),
              message(std::move(message)),
              user_data(std::move(user_data))
        {
        }

        static InferError ok(std::string msg = "ok") noexcept
        {
            return InferError{InferErrorCode::kOK, std::move(msg)};
        }
    };

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
        kDML
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
        std::string_view class_name;
    };

    struct YOLOFrameResult
    {
        std::vector<YOLOResult> results;
    };

    class Cvt
    {
        Cvt() = delete;

    public:
        //fp32->fp16
        static void cvt(std::span<const float> from, Ort::Float16_t* output)
        {
            //TODO optimize
            for (auto i : std::views::iota(from.size()))
            {
                output[i] = Ort::Float16_t(from[i]);
            }
        }

        //fp16->fp32
        static void cvt(std::span<const Ort::Float16_t> from, float* output)
        {
            //TODO optimize
            for (auto i : std::views::iota(from.size()))
            {
                output[i] = from[i].ToFloat();
            }
        }
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
        static CreateResult Create(InferContext& context, std::span<uint8_t> data, YOLOVersion version) noexcept;

        template <typename T>
            requires std::is_arithmetic_v<T>
        static CreateResult Create(InferContext& context, std::span<T> data, YOLOVersion version) noexcept
        {
            return Create(context,
                          std::span(reinterpret_cast<uint8_t*>(data.data()), data.size_bytes()),
                          version);
        }
    };
}
