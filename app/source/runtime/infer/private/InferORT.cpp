#include "InferORT.h"
#ifdef VISION_SIMPLE_WITH_DML
#include <dml_provider_factory.h>
#endif
#ifdef VISION_SIMPLE_WITH_RKNPU
#include <rknpu_provider_factory.h>
#endif
#include <onnxruntime_session_options_config_keys.h>

#include <magic_enum.hpp>

#define INFER_CTX_LOG_ID "vision-simple"
#ifdef VISION_SIMPLE_DEBUG
#define INFER_CTX_LOG_LEVEL OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING
#else
#define INFER_CTX_LOG_LEVEL OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL
#endif

#define UNSUPPORTED_EP(ep)  std::unexpected{ \
      VisionSimpleError{VisionSimpleErrorCode::kRuntimeError, \
                        std::format("unsupported execution_provider:{}", \
                                    magic_enum::enum_name((ep)))}}

vision_simple::InferContextORT::InferContextORT(const InferEP ep,
                                                InferArgs args)
  : InferContext(InferFramework::kONNXRUNTIME, ep, std::move(args)),
    env_(std::make_unique<Ort::Env>(INFER_CTX_LOG_LEVEL, INFER_CTX_LOG_ID)),
    env_memory_info_(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  env_->CreateAndRegisterAllocator(env_memory_info_, nullptr);
}

Ort::Env& vision_simple::InferContextORT::env() const noexcept { return *env_; }

Ort::MemoryInfo& vision_simple::InferContextORT::env_memory_info() {
  return env_memory_info_;
}

vision_simple::InferContextORT::CreateResult
vision_simple::InferContextORT::CreateSession(std::span<uint8_t> data,
                                              size_t device_id) const {
  Ort::SessionOptions session_options;
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options.DisableProfiling();
  session_options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1");
  session_options.AddConfigEntry(kOrtSessionOptionsConfigAllowInterOpSpinning,
                                 "0");
  session_options.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning,
                                 "0");
  session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "0");
  session_options.SetLogSeverityLevel(INFER_CTX_LOG_LEVEL);
  if (ep_ == InferEP::kDML) {
#ifndef VISION_SIMPLE_WITH_DML
    return UNSUPPORTED_EP(ep_);
#else
    session_options.SetInterOpNumThreads(1);
    session_options.SetIntraOpNumThreads(1);
    session_options.SetExecutionMode(ORT_SEQUENTIAL);
    session_options.DisableMemPattern();

    try {
      const OrtDmlApi* dml_api = nullptr;
      Ort::GetApi().GetExecutionProviderApi(
          "DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dml_api));
      dml_api->SessionOptionsAppendExecutionProvider_DML(
          session_options, static_cast<int>(device_id));
    } catch (const std::exception& e) {
      return std::unexpected{VisionSimpleError{
          VisionSimpleErrorCode::kRuntimeError,
          std::format("unable to create ONNXRuntime Session:{}", e.what())}};
    }
#endif
  } else if (ep_ == InferEP::kCUDA) {
#ifndef VISION_SIMPLE_WITH_CUDA
    return UNSUPPORTED_EP(ep_);
#else
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = static_cast<int>(device_id);
    try {
      Ort::ThrowOnError(
          OrtSessionOptionsAppendExecutionProvider_CUDA(
              session_options, device_id));
    } catch (const std::exception& e) {
      return std::unexpected{VisionSimpleError{
        VisionSimpleErrorCode::kRuntimeError,
        std::format("unable to create ONNXRuntime Session:{}", e.what())}};
    }
#endif
  } else if (ep_ == InferEP::kTensorRT) {
#ifndef VISION_SIMPLE_WITH_TENSORRT
    return UNSUPPORTED_EP(ep_);
#else
#ifndef VISION_SIMPLE_WITH_CUDA
#error "TensorRT need CUDA Execution Provider!"
#endif
    // TODO fixit
    OrtTensorRTProviderOptions trt_options{};
    trt_options.device_id = static_cast<int>(device_id);
    try {
    //session_options.AppendExecutionProvider_TensorRT(trt_options);
      OrtSessionOptionsAppendExecutionProvider_Tensorrt(
          session_options, device_id);
      OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device_id)
    } catch (const std::exception& e) {
      return std::unexpected{VisionSimpleError{
        VisionSimpleErrorCode::kRuntimeError,
        std::format("unable to create ONNXRuntime Session:{}", e.what())}};
    }
#endif
  } else if (ep_ == InferEP::kRKNPU) {
    if (device_id != 0) {
      return std::unexpected{VisionSimpleError{
          VisionSimpleErrorCode::kRuntimeError,
          std::format("unsupported device_id:{}", device_id)}};
    }
#ifndef VISION_SIMPLE_WITH_RKNPU
    return UNSUPPORTED_EP(ep_);
#else
    try{
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Rknpu(session_options));
    }catch (const std::exception& e) {
      return std::unexpected{VisionSimpleError{
        VisionSimpleErrorCode::kRuntimeError,
        std::format("unable to create ONNXRuntime Session:{}", e.what())}};
    }
#endif
  }
  try {
    return std::make_unique<Ort::Session>(*env_, data.data(), data.size_bytes(),
                                          session_options);
  } catch (std::exception& e) {
    return std::unexpected{VisionSimpleError{
        VisionSimpleErrorCode::kRuntimeError,
        std::format("unable to create ONNXRuntime Session:{}", e.what())}};
  }
}
