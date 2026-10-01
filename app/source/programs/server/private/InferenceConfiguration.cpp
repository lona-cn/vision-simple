#include "InferenceConfiguration.h"
#include "HTTPServer.h"
#include <charconv>
#include <format>
#include <magic_enum.hpp>

namespace vision_simple {
namespace {
template <typename T>
bool ParseInteger(std::string_view text, T& value) {
  if (text.empty()) return false;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}
}
VSResult<InferenceServiceOptions> ParseInferenceServiceOptions(HTTPServerOptions& options) noexcept try {
  const auto& device_text = options.OptionOrPut(
      HTTPSERVER_OPT_KEY_INFER_DEVICE, HTTPSERVER_OPT_DEFVAL_INFER_DEVICE);
  const auto& idle_text =
      options.OptionOrPut(HTTPSERVER_OPT_KEY_INFER_IDLE_TIMEOUT_MS,
                          HTTPSERVER_OPT_DEFVAL_INFER_IDLE_TIMEOUT_MS);
  const auto& sweep_text =
      options.OptionOrPut(HTTPSERVER_OPT_KEY_INFER_SWEEP_INTERVAL_MS,
                          HTTPSERVER_OPT_DEFVAL_INFER_SWEEP_INTERVAL_MS);
  int device_id = 0;
  uint64_t idle_ms = 0, sweep_ms = 0;
  // Bound conversions and steady-clock arithmetic, including wait deadlines.
  const auto max_ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::duration::max())
          .count() /
      2);
  if (!ParseInteger(device_text, device_id) || device_id < 0 ||
      !ParseInteger(idle_text, idle_ms) || idle_ms > max_ms ||
      !ParseInteger(sweep_text, sweep_ms) || sweep_ms == 0 ||
      sweep_ms > max_ms) {
    return MK_VSERROR(
        VisionSimpleErrorCode::kParameterError,
        "infer_device and infer_idle_timeout_ms must be nonnegative integers; "
        "infer_sweep_interval_ms must be a positive integer within clock "
        "range");
  }
  PipelineOptions pipeline_options;
  uint64_t timeout_ms = 0;
  size_t ocr_rec_batch_size = 1;
  size_t max_image_pixels = 0, max_batch_decoded_bytes = 0,
         max_inflight_decoded_bytes = 0;
  if (!ParseInteger(
          options.OptionOrPut(HTTPSERVER_OPT_KEY_PIPELINE_CAPACITY,
                              HTTPSERVER_OPT_DEFVAL_PIPELINE_CAPACITY),
          pipeline_options.capacity) ||
      !ParseInteger(options.OptionOrPut(HTTPSERVER_OPT_KEY_PIPELINE_BATCHES,
                                        HTTPSERVER_OPT_DEFVAL_PIPELINE_BATCHES),
                    pipeline_options.max_batches) ||
      !ParseInteger(options.OptionOrPut(HTTPSERVER_OPT_KEY_MAX_BATCH_IMAGES,
                                        HTTPSERVER_OPT_DEFVAL_MAX_BATCH_IMAGES),
                    pipeline_options.max_batch_images) ||
      !ParseInteger(options.OptionOrPut(HTTPSERVER_OPT_KEY_TIMEOUT_MS,
                                        HTTPSERVER_OPT_DEFVAL_TIMEOUT_MS),
                    timeout_ms) ||
      !ParseInteger(
          options.OptionOrPut(HTTPSERVER_OPT_KEY_OCR_REC_BATCH_SIZE,
                              HTTPSERVER_OPT_DEFVAL_OCR_REC_BATCH_SIZE),
          ocr_rec_batch_size) ||
      ocr_rec_batch_size == 0 || ocr_rec_batch_size > 64 || timeout_ms == 0 ||
      timeout_ms > 300000) {
    return MK_VSERROR(
        VisionSimpleErrorCode::kParameterError,
        "pipeline limits must be integers and infer_timeout_ms "
        "must be 1 to 300000; ocr_rec_batch_size must be 1 to 64");
  }
  if (!ParseInteger(
          options.OptionOrPut(HTTPSERVER_OPT_KEY_MAX_IMAGE_PIXELS,
                              HTTPSERVER_OPT_DEFVAL_MAX_IMAGE_PIXELS),
          max_image_pixels) ||
      !ParseInteger(
          options.OptionOrPut(HTTPSERVER_OPT_KEY_MAX_BATCH_DECODED_BYTES,
                              HTTPSERVER_OPT_DEFVAL_MAX_BATCH_DECODED_BYTES),
          max_batch_decoded_bytes) ||
      !ParseInteger(
          options.OptionOrPut(HTTPSERVER_OPT_KEY_MAX_INFLIGHT_DECODED_BYTES,
                              HTTPSERVER_OPT_DEFVAL_MAX_INFLIGHT_DECODED_BYTES),
          max_inflight_decoded_bytes) ||
      max_image_pixels == 0 || max_batch_decoded_bytes == 0 ||
      max_inflight_decoded_bytes == 0) {
    return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                      "decoded image budgets must be positive integers "
                      "within size_t range");
  }
  auto infer_fw_str =
      options.OptionOrPut(HTTPSERVER_OPT_KEY_INFER_FRAMEWORK,
                          HTTPSERVER_OPT_DEFVAL_INFER_FRAMEWORK);
  auto infer_ep_str = options.OptionOrPut(HTTPSERVER_OPT_KEY_INFER_EP,
                                          HTTPSERVER_OPT_DEFVAL_INFER_EP);
  auto infer_fw = magic_enum::enum_cast<InferFramework>(infer_fw_str);
  auto infer_ep = magic_enum::enum_cast<InferEP>(infer_ep_str);
  if (!infer_fw || !infer_ep)
    return std::unexpected(VisionSimpleError{
        VisionSimpleErrorCode::kParameterError,
        std::format("unsupported infer_framework:{} or infer_ep:{}",
                    infer_fw_str, infer_ep_str)});

  return InferenceServiceOptions{
      .framework = *infer_fw,
      .ep = *infer_ep,
      .device_id = device_id,
      .idle_timeout = std::chrono::milliseconds(idle_ms),
      .sweep_interval = std::chrono::milliseconds(sweep_ms),
      .pipeline = pipeline_options,
      .request_timeout = std::chrono::milliseconds(timeout_ms),
      .ocr_rec_batch_size = ocr_rec_batch_size,
      .max_image_pixels = max_image_pixels,
      .max_batch_decoded_bytes = max_batch_decoded_bytes,
      .max_inflight_decoded_bytes = max_inflight_decoded_bytes};

} catch (...) {
  return MK_VSERROR(VisionSimpleErrorCode::kParameterError, "Inference configuration cannot be parsed");
}
}
