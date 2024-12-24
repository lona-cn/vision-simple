#pragma once
#include <optional>
#include <expected>
#include <onnxruntime_cxx_api.h>
#include <span>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <immintrin.h>


#ifndef VISION_NUM_CVT_THREADS
// 数据类型转换线程数，多线程同时会调用F16C提高速度
// 相较于软件转换，调用F16C是软件转换速度的十几倍
// 经测试，多加几个线程用处不大
#define VISION_NUM_CVT_THREADS 1
#endif

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
            constexpr int step = 8; // 每批次处理8个元素
            const auto size = from.size();

            // 主体并行处理部分
#pragma omp parallel for num_threads(VISION_NUM_CVT_THREADS)
            for (int i = 0; i < static_cast<int>(size / step) * step; i += step)
            {
                __m256 float32_vec = _mm256_loadu_ps(&from[i]); // 加载8个float
                __m128i float16_vec = _mm256_cvtps_ph(float32_vec,
                                                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(&output[i]), float16_vec); // 存储8个Float16
            }

            // 边界处理部分，单线程完成，避免多线程争用资源
            const int remaining = size % step;
            if (remaining > 0)
            {
                const int offset = size - remaining;

                alignas(32) float temp_input[step] = {};
                alignas(16) uint16_t temp_output[step] = {};

                // 拷贝剩余元素
                std::copy_n(from.begin() + offset, remaining, temp_input);

                // 转换剩余元素
                __m256 float32_vec = _mm256_loadu_ps(temp_input);
                __m128i float16_vec = _mm256_cvtps_ph(float32_vec,
                                                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(temp_output), float16_vec);

                // 拷贝结果
                std::copy_n(temp_output, remaining, reinterpret_cast<uint16_t*>(&output[offset]));
            }
        }

        //fp16->fp32
        static void cvt(std::span<const Ort::Float16_t> from, float* output)
        {
            constexpr int step = 8; // 每批次处理8个元素
            const auto size = from.size();

            // 主体并行处理部分
#pragma omp parallel for num_threads(VISION_NUM_CVT_THREADS)
            for (int i = 0; i < static_cast<int>(size / step) * step; i += step)
            {
                __m128i float16_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&from[i])); // 加载8个Float16
                __m256 float32_vec = _mm256_cvtph_ps(float16_vec); // 转换为Float32
                _mm256_storeu_ps(&output[i], float32_vec); // 存储结果
            }

            // 边界处理部分，单线程完成，避免多线程争用资源
            const int remaining = size % step;
            if (remaining > 0)
            {
                const int offset = size - remaining;

                alignas(16) uint16_t temp_input[step] = {};
                alignas(32) float temp_output[step] = {};

                // 拷贝剩余元素
                std::copy_n(reinterpret_cast<const uint16_t*>(&from[offset]), remaining, temp_input);

                // 转换剩余元素
                __m128i float16_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(temp_input));
                __m256 float32_vec = _mm256_cvtph_ps(float16_vec);
                _mm256_storeu_ps(temp_output, float32_vec);

                // 拷贝结果
                std::copy_n(temp_output, remaining, &output[offset]);
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
}
