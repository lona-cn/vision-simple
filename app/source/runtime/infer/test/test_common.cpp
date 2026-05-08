#include <iostream>
#include <string>

#include <magic_enum.hpp>

#include "Util.hpp"
#include "VisionSimpleConfig.h"
#include "VisionSimpleError.h"

using namespace vision_simple;

int test_error_ok() {
  auto err = VisionSimpleError::Ok();
  TEST_ASSERT(static_cast<bool>(err), "Ok is truthy");
  TEST_PASS("VisionSimpleError::Ok bool=true");
  return 0;
}

int test_error_code() {
  VisionSimpleError err(VisionSimpleErrorCode::kIOError, "file not found");
  TEST_ASSERT(!static_cast<bool>(err), "IOError is falsy");
  TEST_ASSERT_EQ(static_cast<int>(err.code), static_cast<int>(VisionSimpleErrorCode::kIOError), "error code");
  TEST_PASS("VisionSimpleError code/bool");
  return 0;
}

int test_config() {
  auto maybe_config = Config::Instance();
  if (!maybe_config) {
    std::cout << "  SKIP: config not available" << std::endl;
    return 0;
  }
  auto& config = maybe_config->get();
  auto& model_cfg = config.model_config();
  // 至少验证 model_config 结构可被访问
  std::cout << "  config: yolo models=" << model_cfg.yolo.size()
            << " ocr models=" << model_cfg.ocr.size() << std::endl;
  TEST_PASS("config load and access");
  return 0;
}

int main() {
  int failures = 0;
  std::cout << "=== Common Module Tests ===" << std::endl;
  failures += test_error_ok();
  failures += test_error_code();
  failures += test_config();
  std::cout << (failures ? "\n*** FAILED ***" : "\n*** ALL PASSED ***") << std::endl;
  return failures ? 1 : 0;
}
