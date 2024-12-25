#pragma once
#include <memory>
#include <onnxruntime_cxx_api.h>

#include "../Infer.h"

namespace vision_simple
{
    class InferContextONNXRuntime : public InferContext
    {
        std::unique_ptr<Ort::Env> env_;
        InferEP ep_;
        Ort::MemoryInfo env_memory_info_;

    public:
        InferContextONNXRuntime(InferEP ep);

        InferContextONNXRuntime(const InferContextONNXRuntime& other) = delete;
        InferContextONNXRuntime(InferContextONNXRuntime&& other) noexcept = default;
        InferContextONNXRuntime& operator=(const InferContextONNXRuntime& other) = delete;
        InferContextONNXRuntime& operator=(InferContextONNXRuntime&& other) noexcept = default;
        ~InferContextONNXRuntime() override = default;

        InferFramework framework() const noexcept override;

        InferEP execution_provider() const noexcept override;

        Ort::Env& env() const noexcept;

        [[nodiscard]] Ort::MemoryInfo& env_memory_info();
    };
}
