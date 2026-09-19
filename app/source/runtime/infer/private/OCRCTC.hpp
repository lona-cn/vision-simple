#pragma once

#include <algorithm>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <utility>

#include "VisionSimpleCommon.h"

namespace vision_simple {

// Decode one [T,C] sequence. Class zero is blank; dictionary keys are class
// - 1.
inline VSResult<std::pair<std::string, float>> DecodeOCRCTC(
    std::span<const float> logits, size_t timesteps, size_t classes,
    const std::map<int, std::string>& dictionary, float confidence_threshold) {
  if (timesteps == 0 || classes == 0 ||
      timesteps > std::numeric_limits<size_t>::max() / classes ||
      logits.size() != timesteps * classes) {
    return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                      "Invalid OCR recognition tensor dimensions or length");
  }
  if (classes - 1 > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      dictionary.size() != classes - 1) {
    return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                      "OCR recognition classes do not match the dictionary");
  }

  std::string text;
  double score_sum = 0.0;
  size_t emitted = 0;
  size_t previous = 0;
  for (size_t t = 0; t < timesteps; ++t) {
    const auto row = logits.subspan(t * classes, classes);
    const auto maximum = std::max_element(row.begin(), row.end());
    const size_t index = static_cast<size_t>(maximum - row.begin());
    const bool eligible = index != 0 && index != previous;
    previous = index;  // Blank and low-confidence timesteps still fold repeats.
    if (index == 0) continue;
    const auto character = dictionary.find(static_cast<int>(index - 1));
    if (character == dictionary.end()) {
      return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                        "OCR recognition class is missing from the dictionary");
    }
    if (!eligible || !(*maximum > confidence_threshold)) continue;
    text += character->second;
    score_sum += *maximum;
    ++emitted;
  }
  return std::pair{std::move(text),
                   emitted ? static_cast<float>(score_sum / emitted) : 0.0f};
}

}  // namespace vision_simple
