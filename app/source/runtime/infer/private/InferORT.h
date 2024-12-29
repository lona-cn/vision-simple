#pragma once
#include <memory>
#include <onnxruntime_cxx_api.h>

#include "../Infer.h"

namespace vision_simple
{
    class InferContextORT : public InferContext
    {
        std::unique_ptr<Ort::Env> env_;
        InferEP ep_;
        Ort::MemoryInfo env_memory_info_;

    public:
        using CreateResult = InferResult<std::unique_ptr<Ort::Session>>;
        InferContextORT(InferEP ep);
        InferContextORT(const InferContextORT& other) = delete;
        InferContextORT(InferContextORT&& other) noexcept = default;
        InferContextORT& operator=(const InferContextORT& other) = delete;
        InferContextORT& operator=(InferContextORT&& other) noexcept = default;
        ~InferContextORT() override = default;

        InferFramework framework() const noexcept override;

        InferEP execution_provider() const noexcept override;

        Ort::Env& env() const noexcept;

        [[nodiscard]] Ort::MemoryInfo& env_memory_info();
        CreateResult CreateSession(std::span<uint8_t> data, size_t device_id);
    };
}
