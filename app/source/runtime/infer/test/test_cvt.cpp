#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

#include <onnxruntime_float16.h>

#include "Util.hpp"
#include "VisionHelper.hpp"

using namespace vision_simple;

int test_fp32tofp16_roundtrip() {
  std::vector<float> input(256);
  for (int i = 0; i < 256; ++i) input[i] = static_cast<float>(i) / 255.0f;

  std::vector<Ort::Float16_t> fp16_buf(256);
  Cvt::cvt(std::span<const float>{input.data(), 256}, fp16_buf.data());

  std::vector<float> output(256);
  Cvt::cvt(std::span<const Ort::Float16_t>{fp16_buf.data(), 256}, output.data());

  for (int i = 0; i < 256; ++i) {
    TEST_ASSERT_FLOAT_EQ(input[i], output[i], 0.01f, "fp32-fp16 roundtrip");
  }
  TEST_PASS("fp32tofp16 roundtrip (256 values)");
  return 0;
}

int test_fp16tofp32_roundtrip() {
  std::vector<float> input(64);
  for (int i = 0; i < 64; ++i) input[i] = -1.0f + static_cast<float>(i) * 2.0f / 63.0f;

  std::vector<Ort::Float16_t> fp16_buf(64);
  Cvt::cvt(std::span<const float>{input}, fp16_buf.data());
  std::vector<float> output(64);
  Cvt::cvt(std::span<const Ort::Float16_t>{fp16_buf.data(), 64}, output.data());

  for (int i = 0; i < 64; ++i) {
    TEST_ASSERT_FLOAT_EQ(input[i], output[i], 0.01f, "fp16-fp32 roundtrip");
  }
  TEST_PASS("fp16tofp32 roundtrip (64 values)");
  return 0;
}

int test_u8tofp32_normalized() {
  std::vector<uint8_t> zeros(32, 0);
  std::vector<float> out_zeros(32);
  Cvt::cvt(std::span<const uint8_t>{zeros}, out_zeros.data());
  for (size_t i = 0; i < 32; ++i) {
    TEST_ASSERT_FLOAT_EQ(0.0f, out_zeros[i], 0.001f, "u8tofp32 zero");
  }

  std::vector<uint8_t> maxes(32, 255);
  std::vector<float> out_max(32);
  Cvt::cvt(std::span<const uint8_t>{maxes}, out_max.data());
  for (size_t i = 0; i < 32; ++i) {
    TEST_ASSERT_FLOAT_EQ(1.0f, out_max[i], 0.001f, "u8tofp32 max");
  }
  TEST_PASS("u8tofp32_normalized (zero and max, 32 each)");
  return 0;
}

int main() {
  int failures = 0;
  std::cout << "=== Cvt Unit Tests ===" << std::endl;
  failures += test_fp32tofp16_roundtrip();
  failures += test_fp16tofp32_roundtrip();
  failures += test_u8tofp32_normalized();
  std::cout << (failures ? "\n*** FAILED ***" : "\n*** ALL PASSED ***") << std::endl;
  return failures ? 1 : 0;
}
