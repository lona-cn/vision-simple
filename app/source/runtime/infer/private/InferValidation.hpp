#pragma once

#include <bit>
#include <cstdint>

#include "Infer.h"

namespace vision_simple {
// Bit classification remains valid under the project's release fast-math mode.
inline bool IsFinite(float value) noexcept {
  return (std::bit_cast<uint32_t>(value) & 0x7f800000u) != 0x7f800000u;
}

inline VSResult<void> ValidateInferInput(const cv::Mat& image,
                                         float confidence) noexcept {
  if (image.empty() || image.dims != 2 || image.rows <= 0 || image.cols <= 0 ||
      image.type() != CV_8UC3) {
    return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                      "image must be a nonempty two-dimensional CV_8UC3 Mat");
  }
  if (!IsFinite(confidence) || confidence < 0.0f || confidence > 1.0f) {
    return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                      "confidence must be finite and within [0,1]");
  }
  return {};
}
}  // namespace vision_simple
