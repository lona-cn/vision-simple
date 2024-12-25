#include "InferORT.h"


#define INFER_CTX_LOG_ID "vision-simple"
#ifdef VISION_SIMPLE_DEBUG
#define INFER_CTX_LOG_LEVEL OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE
#else
#define INFER_CTX_LOG_LEVEL OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL
#endif

vision_simple::InferContextONNXRuntime::InferContextONNXRuntime(const InferEP ep): env_(std::make_unique<Ort::Env>(
        INFER_CTX_LOG_LEVEL, INFER_CTX_LOG_ID)),
    ep_(ep),
    env_memory_info_(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    env_->CreateAndRegisterAllocator(env_memory_info_, nullptr);
}

vision_simple::InferFramework vision_simple::InferContextONNXRuntime::framework() const noexcept
{
    return InferFramework::kONNXRUNTIME;
}

vision_simple::InferEP vision_simple::InferContextONNXRuntime::execution_provider() const noexcept
{
    return ep_;
}

Ort::Env& vision_simple::InferContextONNXRuntime::env() const noexcept
{
    return *env_;
}

Ort::MemoryInfo& vision_simple::InferContextONNXRuntime::env_memory_info()
{
    return env_memory_info_;
}
