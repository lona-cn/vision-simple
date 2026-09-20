#include <iostream>
#include <magic_enum.hpp>
#include <string>

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
  TEST_ASSERT_EQ(static_cast<int>(err.code),
                 static_cast<int>(VisionSimpleErrorCode::kIOError),
                 "error code");
  TEST_PASS("VisionSimpleError code/bool");
  return 0;
}

int main() {
  int failures = 0;
  std::cout << "=== Common Module Tests ===" << std::endl;
  failures += test_error_ok();
  failures += test_error_code();
  std::cout << (failures ? "\n*** FAILED ***" : "\n*** ALL PASSED ***")
            << std::endl;
  return failures ? 1 : 0;
}
