#pragma once

#include <bit>
#include <cstdint>
#include <expected>
#include <nlohmann/json.hpp>
#include <string_view>

#include "InferenceService.h"

namespace vision_simple {
inline nlohmann::json ParseInferenceJson(std::string_view body) {
  using Json = nlohmann::json;
  bool nonfinite = false;
  auto result = Json::parse(
      body,
      [&nonfinite](int, Json::parse_event_t event, Json& value) {
        if (event == Json::parse_event_t::value && value.is_number_float()) {
          const auto word = std::bit_cast<uint64_t>(value.get<double>());
          nonfinite |= (word & UINT64_C(0x7ff0000000000000)) ==
                       UINT64_C(0x7ff0000000000000);
        }
        // Keep every value, including duplicate keys. Reject the entire
        // message afterward instead of silently dropping an invalid field.
        return true;
      },
      false);
  // nlohmann uses isfinite for overflow, which fast-math can optimize away.
  if (nonfinite) return Json(Json::value_t::discarded);
  return result;
}

struct InferenceOptionError {
  const char* parameter;
  const char* message;
};

inline std::expected<YOLOInferenceOptions, InferenceOptionError>
ParseInferenceOptions(const nlohmann::json& request, InferenceKind kind) {
  YOLOInferenceOptions options;
  for (const char* parameter : {"confidence", "nms_iou"}) {
    const auto value = request.find(parameter);
    if (value == request.end()) continue;
    if (kind == InferenceKind::kOCR)
      return std::unexpected(InferenceOptionError{
          parameter, "Detector controls are not supported for OCR"});
    if (!value->is_number())
      return std::unexpected(InferenceOptionError{
          parameter, "Detector control must be a number from 0 to 1"});
    // Check the original JSON double before narrowing. Bit checks also remain
    // correct under fast-math, rejecting nonfinite and negative tiny values.
    const auto word = std::bit_cast<uint64_t>(value->get<double>());
    if (word > UINT64_C(0x3ff0000000000000) &&
        word != UINT64_C(0x8000000000000000))
      return std::unexpected(InferenceOptionError{
          parameter, "Detector control must be a number from 0 to 1"});
    const auto threshold = value->get<float>();
    if (parameter[0] == 'c')
      options.confidence = threshold;
    else
      options.nms_iou = threshold;
  }
  return options;
}
}  // namespace vision_simple
