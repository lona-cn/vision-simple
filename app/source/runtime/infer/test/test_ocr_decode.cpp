#include <array>
#include <iostream>
#include <limits>
#include <map>
#include <string>

#include "../private/OCRPostProcess.hpp"
#include "Util.hpp"

using namespace vision_simple;

namespace {
const std::map<int, std::string> dictionary{{0, "A"}, {1, "B"}};

int test_repeats_and_blanks() {
  // blank,A,A,blank,B,B: only the first score in each run contributes.
  const std::array<float, 18> logits{.9f, .1f,  .0f, .1f, .8f, .0f,
                                     .1f, .95f, .0f, .9f, .1f, .0f,
                                     .0f, .1f,  .6f, .0f, .1f, .9f};
  auto result = DecodeOCRCTC(logits, 6, 3, dictionary, .5f);
  TEST_ASSERT(result, "valid repeated sequence decodes");
  TEST_ASSERT_EQ(result->first, "AB", "adjacent repeated classes fold");
  TEST_ASSERT_FLOAT_EQ(result->second, .7f, 1e-6f,
                       "confidence averages emitted characters only");

  const std::array<float, 9> separated{.1f, .9f, .0f, .9f, .1f,
                                       .0f, .1f, .7f, .0f};
  result = DecodeOCRCTC(separated, 3, 3, dictionary, .5f);
  TEST_ASSERT(result, "blank-separated sequence decodes");
  TEST_ASSERT_EQ(result->first, "AA", "blank separates equal characters");
  TEST_ASSERT_FLOAT_EQ(result->second, .8f, 1e-6f,
                       "separated characters both contribute confidence");
  TEST_PASS("CTC repeat folding and blank separation");
  return 0;
}

int test_low_confidence_timesteps() {
  const std::array<float, 6> low_then_high{.1f, .4f, .0f, .1f, .9f, .0f};
  auto result = DecodeOCRCTC(low_then_high, 2, 3, dictionary, .5f);
  TEST_ASSERT(result, "low-confidence sequence decodes");
  TEST_ASSERT_EQ(result->first, "", "filtered A still suppresses adjacent A");
  TEST_ASSERT_EQ(result->second, 0.0f,
                 "no emitted characters has zero confidence");

  const std::array<float, 9> low_other{.1f, .9f, .0f, .1f, .0f,
                                       .4f, .1f, .7f, .0f};
  result = DecodeOCRCTC(low_other, 3, 3, dictionary, .5f);
  TEST_ASSERT(result, "low-confidence different class decodes");
  TEST_ASSERT_EQ(result->first, "AA", "filtered B separates two A runs");

  const std::array<float, 3> equal_threshold{.1f, .5f, .0f};
  result = DecodeOCRCTC(equal_threshold, 1, 3, dictionary, .5f);
  TEST_ASSERT(result, "threshold boundary decodes");
  TEST_ASSERT_EQ(result->first, "", "emission requires strictly greater score");
  TEST_ASSERT_EQ(result->second, 0.0f,
                 "threshold rejection keeps zero confidence");
  TEST_PASS("CTC confidence filtering preserves raw timestep adjacency");
  return 0;
}

int test_empty_line() {
  const std::array<float, 6> logits{.9f, .1f, .0f, .8f, .1f, .0f};
  const auto result = DecodeOCRCTC(logits, 2, 3, dictionary, .5f);
  TEST_ASSERT(result, "all-blank sequence is a successful empty line");
  TEST_ASSERT_EQ(result->first, "", "all blanks emit no text");
  TEST_ASSERT_EQ(result->second, 0.0f, "all blanks have zero line confidence");
  TEST_PASS("CTC empty-line semantics");
  return 0;
}

int test_invalid_output() {
  const std::array<float, 3> logits{.1f, .2f, .9f};
  const auto short_tensor = DecodeOCRCTC(logits, 2, 3, dictionary, .5f);
  TEST_ASSERT(!short_tensor, "short tensor is rejected before row access");
  TEST_ASSERT(short_tensor.error().code == VisionSimpleErrorCode::kModelError,
              "tensor mismatch is a model error");
  TEST_ASSERT(!DecodeOCRCTC(logits, 0, 3, dictionary, .5f),
              "zero timesteps rejected");
  TEST_ASSERT(!DecodeOCRCTC(logits, 1, 0, dictionary, .5f),
              "zero classes rejected");
  TEST_ASSERT(!DecodeOCRCTC(logits, std::numeric_limits<size_t>::max(), 3,
                            dictionary, .5f),
              "overflowing tensor dimensions rejected");
  const std::map<int, std::string> too_short{{0, "A"}};
  TEST_ASSERT(!DecodeOCRCTC(logits, 1, 3, too_short, .5f),
              "class count must match dictionary");
  const std::map<int, std::string> missing_index{{0, "A"}, {2, "C"}};
  TEST_ASSERT(!DecodeOCRCTC(logits, 1, 3, missing_index, .5f),
              "unmapped argmax rejected rather than inserting empty text");
  TEST_PASS("CTC tensor and dictionary boundaries");
  return 0;
}

int test_nonfinite_output() {
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  const auto infinity = std::numeric_limits<float>::infinity();
  const std::array<std::array<float, 3>, 3> invalid{{
      {.1f, nan, .9f},
      {.1f, infinity, .0f},
      {.1f, -infinity, .9f},
  }};
  for (const auto& logits : invalid) {
    const auto result = DecodeOCRCTC(logits, 1, 3, dictionary, .5f);
    TEST_ASSERT(!result, "non-finite OCR logits are rejected");
    TEST_ASSERT(result.error().code == VisionSimpleErrorCode::kModelError,
                "non-finite OCR logits are a model error");
  }
  TEST_PASS("CTC rejects NaN and infinities");
  return 0;
}

int test_sar_tokens() {
  // A,A,PAD,UKN,EOS,B: repeats survive, PAD is ignored, EOS stops output.
  const std::array<float, 30> logits{.8f, .0f, .0f,  .0f, .0f, .6f, .0f,  .0f,
                                     .0f, .0f, .0f,  .0f, .0f, .0f, .99f, .0f,
                                     .0f, .7f, .0f,  .0f, .0f, .0f, .0f,  .9f,
                                     .0f, .0f, .95f, .0f, .0f, .0f};
  const auto result = DecodeOCRSAR(logits, 6, 5, dictionary, .5f);
  TEST_ASSERT(result, "SAR token sequence decodes");
  TEST_ASSERT_EQ(result->first, "AA<UKN>",
                 "SAR preserves repeats and emits unknown before EOS");
  TEST_ASSERT_FLOAT_EQ(result->second, .7f, 1e-6f,
                       "SAR averages emitted tokens, excluding PAD and EOS");
  TEST_PASS("SAR repeated characters and special tokens");
  return 0;
}

int test_sar_threshold_and_empty() {
  const std::array<float, 15> filtered{.5f, .0f, .0f, .0f, .0f, .9f, .0f, .0f,
                                       .0f, .0f, .0f, .0f, .0f, .0f, .8f};
  auto result = DecodeOCRSAR(filtered, 3, 5, dictionary, .5f);
  TEST_ASSERT(result, "SAR threshold boundary decodes");
  TEST_ASSERT_EQ(result->first, "A",
                 "strict threshold rejects equality without folding repeats");
  TEST_ASSERT_FLOAT_EQ(result->second, .9f, 1e-6f,
                       "filtered tokens do not contribute confidence");

  const std::array<float, 10> immediate_end{.0f, .0f, .0f, .4f, .0f,
                                            .9f, .0f, .0f, .0f, .0f};
  result = DecodeOCRSAR(immediate_end, 2, 5, dictionary, .5f);
  TEST_ASSERT(result, "initial BOS/EOS prediction decodes");
  TEST_ASSERT_EQ(result->first, "", "even low-confidence EOS ends predictions");
  TEST_ASSERT_EQ(result->second, 0.0f,
                 "immediate EOS has finite zero confidence");

  const std::array<float, 10> no_emission{.0f, .0f, .0f, .0f, .9f,
                                          .5f, .0f, .0f, .0f, .0f};
  result = DecodeOCRSAR(no_emission, 2, 5, dictionary, .5f);
  TEST_ASSERT(result, "SAR can finish without an EOS prediction");
  TEST_ASSERT_EQ(result->first, "", "PAD and rejected tokens emit nothing");
  TEST_ASSERT_EQ(result->second, 0.0f, "empty SAR output has zero confidence");
  TEST_PASS("SAR strict threshold and empty output");
  return 0;
}

int test_sar_invalid_output() {
  const std::array<float, 5> logits{.9f, .0f, .0f, .0f, .0f};
  TEST_ASSERT(!DecodeOCRSAR(logits, 2, 5, dictionary, .5f),
              "SAR rejects short tensors");
  TEST_ASSERT(!DecodeOCRSAR(logits, 0, 5, dictionary, .5f),
              "SAR rejects zero timesteps");
  TEST_ASSERT(!DecodeOCRSAR(logits, 1, 0, dictionary, .5f),
              "SAR rejects zero classes");
  TEST_ASSERT(!DecodeOCRSAR(logits, 1, 2, dictionary, .5f),
              "SAR needs all three special classes");
  TEST_ASSERT(!DecodeOCRSAR(logits, std::numeric_limits<size_t>::max(), 5,
                            dictionary, .5f),
              "SAR rejects overflowing tensor products");
  const std::map<int, std::string> too_short{{0, "A"}};
  TEST_ASSERT(!DecodeOCRSAR(logits, 1, 5, too_short, .5f),
              "SAR rejects dictionary/class count mismatch");
  const std::map<int, std::string> sparse{{0, "A"}, {2, "B"}};
  TEST_ASSERT(!DecodeOCRSAR(logits, 1, 5, sparse, .5f),
              "SAR rejects gaps even when the predicted character exists");
  const std::map<int, std::string> negative{{-1, "A"}, {0, "B"}};
  TEST_ASSERT(!DecodeOCRSAR(logits, 1, 5, negative, .5f),
              "SAR rejects negative dictionary keys");
  TEST_PASS("SAR tensor and dictionary boundaries");
  return 0;
}

int test_sar_nonfinite_after_eos() {
  const std::array<float, 3> invalid{std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::infinity(),
                                     -std::numeric_limits<float>::infinity()};
  for (const auto value : invalid) {
    const std::array<float, 10> logits{.0f, .0f, .0f,   .9f, .0f,
                                       .9f, .0f, value, .0f, .0f};
    const auto result = DecodeOCRSAR(logits, 2, 5, dictionary, .5f);
    TEST_ASSERT(!result, "SAR validates non-winning values even after EOS");
    TEST_ASSERT(result.error().code == VisionSimpleErrorCode::kModelError,
                "non-finite SAR output is a model error");
  }
  TEST_PASS("SAR validates the complete tensor after EOS");
  return 0;
}

int test_model_registry() {
  const std::array<float, 6> ctc_logits{.0f, .9f, .0f, .0f, .8f, .0f};
  for (const auto model : {OCRModelType::kPPOCRv3, OCRModelType::kPPOCRv4}) {
    const auto* processor = FindOCRPostProcessor(model);
    TEST_ASSERT(processor, "Paddle CTC models have a postprocessor");
    const auto result = processor->decode(ctc_logits, 2, 3, dictionary, .5f);
    TEST_ASSERT(result, "registered CTC decoder accepts its class layout");
    TEST_ASSERT_EQ(result->first, "A", "CTC registry entry folds repeats");
  }
  const auto* processor = FindOCRPostProcessor(OCRModelType::kPaddleSAR);
  TEST_ASSERT(processor, "Paddle SAR has a postprocessor");
  const std::array<float, 10> sar_logits{.9f, .0f, .0f, .0f, .0f,
                                         .8f, .0f, .0f, .0f, .0f};
  const auto result = processor->decode(sar_logits, 2, 5, dictionary, .5f);
  TEST_ASSERT(result, "registered SAR decoder accepts its class layout");
  TEST_ASSERT_EQ(result->first, "AA", "SAR registry entry preserves repeats");
  TEST_ASSERT(!FindOCRPostProcessor(OCRModelType::kEasyOCR),
              "EasyOCR remains unsupported");
  TEST_ASSERT(!FindOCRPostProcessor(static_cast<OCRModelType>(255)),
              "unknown model types remain unsupported");
  TEST_PASS("Model registry selects CTC or SAR and rejects unsupported types");
  return 0;
}
}  // namespace

int main() {
  int failures = 0;
  std::cout << "=== OCR Decoder Tests ===" << std::endl;
  failures += test_repeats_and_blanks();
  failures += test_low_confidence_timesteps();
  failures += test_empty_line();
  failures += test_invalid_output();
  failures += test_nonfinite_output();
  failures += test_sar_tokens();
  failures += test_sar_threshold_and_empty();
  failures += test_sar_invalid_output();
  failures += test_sar_nonfinite_after_eos();
  failures += test_model_registry();
  std::cout << (failures ? "\n*** FAILED ***" : "\n*** ALL PASSED ***")
            << std::endl;
  return failures ? 1 : 0;
}
