#include <array>
#include <barrier>
#include <iostream>
#include <string_view>
#include <thread>

#include "LogContext.h"

using namespace vision_simple;

namespace {
int failures = 0;

bool Require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "trace ID test failed: " << message << '\n';
    ++failures;
  }
  return condition;
}

bool HasUuidV4Format(std::string_view id) {
  if (id.size() != 36) return false;
  for (size_t i = 0; i < id.size(); ++i) {
    if (i == 8 || i == 13 || i == 18 || i == 23) {
      if (id[i] != '-') return false;
    } else if (!((id[i] >= '0' && id[i] <= '9') ||
                 (id[i] >= 'a' && id[i] <= 'f'))) {
      return false;
    }
  }
  return id[14] == '4' &&
         (id[19] == '8' || id[19] == '9' || id[19] == 'a' || id[19] == 'b');
}

void TestScalarFormat() {
  const auto id = LogContext::GenerateTraceId();
  Require(HasUuidV4Format(id),
          "generated ID must be 36 lowercase ASCII characters in 8-4-4-4-12 "
          "UUID v4 format, with version 4 and variant 8, 9, a, or b");
  Require(id.find('\0') == std::string::npos,
          "generated ID must not contain an embedded NUL");
}

bool GeneratesValidBatch(size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!HasUuidV4Format(LogContext::GenerateTraceId())) return false;
  }
  return true;
}

void TestBatchFormat() {
  Require(GeneratesValidBatch(4096),
          "every ID in a sequential batch must retain UUID v4 format");
}

void TestConcurrentFormat() {
  constexpr size_t thread_count = 4;
  std::barrier start{static_cast<std::ptrdiff_t>(thread_count)};
  std::array<bool, thread_count> valid{};
  std::array<std::jthread, thread_count> workers;
  for (size_t i = 0; i < thread_count; ++i) {
    workers[i] = std::jthread([&, i] {
      start.arrive_and_wait();
      valid[i] = GeneratesValidBatch(1024);
    });
  }
  for (auto& worker : workers) worker.join();
  for (bool result : valid) {
    Require(result,
            "every ID generated concurrently must retain UUID v4 format");
  }
}
}  // namespace

int main() {
  TestScalarFormat();
  TestBatchFormat();
  TestConcurrentFormat();
  return failures == 0 ? 0 : 1;
}
