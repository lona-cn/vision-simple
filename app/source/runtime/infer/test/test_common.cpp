#include <iostream>
#include <magic_enum.hpp>
#include <string>
#include <array>
#include <barrier>
#include <thread>

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

int test_concurrent_errors() {
  std::array<std::vector<VisionSimpleError>, 4> errors;
  std::barrier ready(4);
  std::vector<std::jthread> workers;
  for (size_t worker = 0; worker < errors.size(); ++worker) {
    workers.emplace_back([&, worker] {
      ready.arrive_and_wait();
      for (int index = 0; index < 1000; ++index) {
        errors[worker].emplace_back(
            VisionSimpleErrorCode::kRuntimeError,
            std::format("worker {} inference {}: message survives the producing thread", worker, index));
      }
    });
  }
  workers.clear();
  for (size_t worker = 0; worker < errors.size(); ++worker) {
    for (size_t index = 0; index < errors[worker].size(); ++index) {
      TEST_ASSERT(
          std::string_view(errors[worker][index].message) ==
              std::format("worker {} inference {}: message survives the producing thread", worker, index),
          "concurrent errors retain independent messages after thread exit");
    }
  }
  TEST_PASS("Concurrent error ownership");
  return 0;
}

int main() {
  int failures = 0;
  std::cout << "=== Common Module Tests ===" << std::endl;
  failures += test_error_ok();
  failures += test_error_code();
  failures += test_concurrent_errors();
  std::cout << (failures ? "\n*** FAILED ***" : "\n*** ALL PASSED ***")
            << std::endl;
  return failures ? 1 : 0;
}
