#include <array>
#include <iostream>
#include <limits>
#include <map>
#include <string>

#include "../private/OCRCTC.hpp"
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
}  // namespace

int main() {
  int failures = 0;
  std::cout << "=== OCR CTC Decoder Tests ===" << std::endl;
  failures += test_repeats_and_blanks();
  failures += test_low_confidence_timesteps();
  failures += test_empty_line();
  failures += test_invalid_output();
  failures += test_nonfinite_output();
  std::cout << (failures ? "\n*** FAILED ***" : "\n*** ALL PASSED ***")
            << std::endl;
  return failures ? 1 : 0;
}
