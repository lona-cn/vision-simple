#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

#include "Util.hpp"
#include "VisionHelper.hpp"

using namespace vision_simple;

int test_known_binary16_patterns() {
  const std::array<uint16_t, 9> bits{0x0000, 0x8000, 0x3c00, 0xc000, 0x0001,
                                     0x7bff, 0x7c00, 0xfc00, 0x7e01};
  const std::array<uint32_t, 8> expected{0x00000000, 0x80000000, 0x3f800000,
                                         0xc0000000, 0x33800000, 0x477fe000,
                                         0x7f800000, 0xff800000};
  std::array<Ort::Float16_t, bits.size()> input{};
  for (size_t i = 0; i < bits.size(); ++i) input[i].val = bits[i];
  std::array<float, bits.size()> output{};
  Cvt::cvt(input, output.data());
  for (size_t i = 0; i < expected.size(); ++i)
    TEST_ASSERT_EQ(std::bit_cast<uint32_t>(output[i]), expected[i],
                   "binary16 known value");
  const auto nan_bits = std::bit_cast<uint32_t>(output.back());
  TEST_ASSERT(
      (nan_bits & 0x7f800000u) == 0x7f800000u && (nan_bits & 0x007fffffu) != 0,
      "binary16 NaN stays NaN");
  TEST_PASS(
      "direct binary16 patterns including signed zero, subnormal and specials");
  return 0;
}

int test_fp32_rounding_and_specials() {
  const std::array<uint32_t, 13> bits{
      0x00000000, 0x80000000, 0x3f800000, 0xc0000000, 0x33800000,
      0x477fe000, 0x7f800000, 0xff800000, 0x7fc12345, 0x3f801000,
      0x3f803000, 0x33000000, 0x477ff000};
  const std::array<uint16_t, 13> expected{
      0x0000, 0x8000, 0x3c00, 0xc000, 0x0001, 0x7bff, 0x7c00,
      0xfc00, 0x7e00, 0x3c00, 0x3c02, 0x0000, 0x7c00};
  std::array<float, bits.size()> input{};
  for (size_t i = 0; i < bits.size(); ++i)
    input[i] = std::bit_cast<float>(bits[i]);
  std::array<Ort::Float16_t, bits.size()> output{};
  Cvt::cvt(input, output.data());
  for (size_t i = 0; i < expected.size(); ++i) {
    if (i == 8) {
      TEST_ASSERT(
          (output[i].val & 0x7c00) == 0x7c00 && (output[i].val & 0x03ff) != 0,
          "fp32 NaN stays NaN");
    } else {
      TEST_ASSERT_EQ(output[i].val, expected[i], "fp32 IEEE binary16 rounding");
    }
  }
  TEST_PASS("fp32 known patterns and round-to-nearest-even boundaries");
  return 0;
}

int test_conversion_tails() {
  for (size_t count :
       {size_t{0}, size_t{1}, size_t{7}, size_t{15}, size_t{16}, size_t{17},
        size_t{31}, size_t{32}, size_t{33}, size_t{65}}) {
    std::vector<float> input(count, -2.0f);
    std::vector<Ort::Float16_t> halves(count + 2);
    for (auto& half : halves) half.val = 0x3555;
    Cvt::cvt(input, halves.data() + 1);
    TEST_ASSERT_EQ(halves.front().val, 0x3555, "fp32 conversion leading guard");
    TEST_ASSERT_EQ(halves.back().val, 0x3555, "fp32 conversion trailing guard");
    for (size_t i = 1; i <= count; ++i)
      TEST_ASSERT_EQ(halves[i].val, 0xc000,
                     "fp32 conversion writes every tail element");
    std::vector<float> output(count + 2, 123.0f);
    Cvt::cvt(std::span<const Ort::Float16_t>(halves.data() + 1, count),
             output.data() + 1);
    TEST_ASSERT_EQ(output.front(), 123.0f, "fp16 conversion leading guard");
    TEST_ASSERT_EQ(output.back(), 123.0f, "fp16 conversion trailing guard");
    for (size_t i = 1; i <= count; ++i)
      TEST_ASSERT_EQ(output[i], -2.0f,
                     "fp16 conversion writes every tail element");
  }
  TEST_PASS("empty, partial and full block lengths preserve guards");
  return 0;
}

int main() {
  int failures = 0;
  failures += test_known_binary16_patterns();
  failures += test_fp32_rounding_and_specials();
  failures += test_conversion_tails();
  return failures ? 1 : 0;
}
