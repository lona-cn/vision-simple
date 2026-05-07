#pragma once
#include <chrono>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <string_view>

namespace vision_simple {

/// 日志上下文: 线程局部 traceId + RAII 计时器
class LogContext {
 public:
  LogContext() = delete;

  /// 生成 UUID v4 格式的 traceId (8-4-4-4-12)
  static std::string GenerateTraceId() noexcept {
    static thread_local std::mt19937 gen{std::random_device{}()};
    static thread_local std::uniform_int_distribution<uint32_t> dist4{0, 0xFFFFU};
    static thread_local std::uniform_int_distribution<uint32_t> dist8{0, 0xFFFFFFFFU};
    char buf[37]{};
    char* p = buf;
    auto hex8 = [&](uint32_t v) {
      for (int i = 7; i >= 0; --i) *p++ = "0123456789abcdef"[(v >> (i * 4)) & 0xF];
    };
    auto hex4 = [&](uint32_t v) {
      for (int i = 3; i >= 0; --i) *p++ = "0123456789abcdef"[(v >> (i * 4)) & 0xF];
    };
    hex8(dist8(gen));
    *p++ = '-';
    hex4(dist4(gen));
    *p++ = '-';
    *p++ = '4';
    hex4(dist4(gen) & 0x0FFFU | 0x4000U);
    *p++ = '-';
    *p++ = '8' | static_cast<char>(dist4(gen) & 0x03U);
    hex4(dist4(gen) & 0x3FFFU | 0x8000U);
    *p++ = '-';
    hex8(dist8(gen));
    return std::string{buf, 36};
  }

  static void SetTraceId(std::string_view id) noexcept {
    current_trace_id_ = id;
  }

  static std::string_view CurrentTraceId() noexcept {
    return current_trace_id_;
  }

  /// RAII 计时器: 构造时开始, 析构时通过回调输出耗时
  class ScopedTimer {
   public:
    using Callback = std::function<void(std::string_view phase, int64_t ms)>;

    ScopedTimer(std::string_view phase, Callback on_destroy = nullptr)
        : phase_(phase), on_destroy_(on_destroy), start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
      if (on_destroy_) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_);
        on_destroy_(phase_, elapsed.count());
      }
    }

    [[nodiscard]] int64_t elapsed_ms() const noexcept {
      return std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_).count();
    }

   private:
    std::string_view phase_;
    Callback on_destroy_;
    std::chrono::steady_clock::time_point start_;
  };

 private:
  static thread_local std::string current_trace_id_;
};

}  // namespace vision_simple