#pragma once

#include <exception>

#include "Infer.h"
#include "OCRCTC.hpp"

namespace vision_simple {

// PaddleOCR SAR prediction tokens: ordinary characters, UKN, BOS/EOS, PAD.
// Unlike CTC, adjacent repeated predictions are independent characters.
inline VSResult<std::pair<std::string, float>> DecodeOCRSAR(
    std::span<const float> logits, size_t timesteps, size_t classes,
    const std::map<int, std::string>& dictionary, float confidence_threshold) {
  if (timesteps == 0 || classes < 3 ||
      timesteps > std::numeric_limits<size_t>::max() / classes ||
      logits.size() != timesteps * classes) {
    return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                      "Invalid SAR recognition tensor dimensions or length");
  }
  if (classes - 3 > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      dictionary.size() != classes - 3) {
    return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                      "SAR recognition classes do not match the dictionary");
  }
  size_t expected_index = 0;
  for (const auto& [index, character] : dictionary) {
    if (index < 0 || static_cast<size_t>(index) != expected_index++) {
      return MK_VSERROR(
          VisionSimpleErrorCode::kModelError,
          "SAR dictionary keys must be contiguous and zero-based");
    }
  }

  try {
    const size_t unknown = classes - 3;
    const size_t end = classes - 2;
    const size_t padding = classes - 1;
    std::string text;
    double score_sum = 0.0;
    size_t emitted = 0;
    bool ended = false;
    for (size_t t = 0; t < timesteps; ++t) {
      const auto row = logits.subspan(t * classes, classes);
      size_t index = 0;
      float maximum = row.front();
      for (size_t c = 0; c < classes; ++c) {
        // Validate the complete tensor, including predictions after EOS.
        if (!IsFinite(row[c])) {
          return MK_VSERROR(
              VisionSimpleErrorCode::kModelError,
              "SAR recognition output contains non-finite values");
        }
        if (!ended && row[c] > maximum) {
          maximum = row[c];
          index = c;
        }
      }
      if (ended) continue;
      if (index == end) {
        ended = true;
        continue;
      }
      if (index == padding || !(maximum > confidence_threshold)) continue;
      if (index == unknown) {
        text += "<UKN>";
      } else {
        text += dictionary.find(static_cast<int>(index))->second;
      }
      score_sum += maximum;
      ++emitted;
    }
    return std::pair{std::move(text),
                     emitted ? static_cast<float>(score_sum / emitted) : 0.0f};
  } catch (const std::exception& error) {
    return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError, error.what());
  }
}

struct OCRPostProcessor {
  OCRModelType model_type;
  size_t extra_classes;
  bool append_dictionary_space;
  VSResult<std::pair<std::string, float>> (*decode)(
      std::span<const float>, size_t timesteps, size_t classes,
      const std::map<int, std::string>&, float threshold);
};

inline const OCRPostProcessor* FindOCRPostProcessor(
    OCRModelType model_type) noexcept {
  static constexpr OCRPostProcessor processors[]{
      {OCRModelType::kPPOCRv3, 1, true, DecodeOCRCTC},
      {OCRModelType::kPPOCRv4, 1, true, DecodeOCRCTC},
      {OCRModelType::kPaddleSAR, 3, false, DecodeOCRSAR},
  };
  for (const auto& processor : processors) {
    if (processor.model_type == model_type) return &processor;
  }
  return nullptr;
}

}  // namespace vision_simple
