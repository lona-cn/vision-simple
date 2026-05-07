#pragma once
#include <format>
#include <string_view>

#include "LogSink.h"
#include "LogContext.h"

namespace vision_simple {

/// 日志门面 — 所有模块的唯一切入点
/// 使用前需调用 RegisterSink() 注入具体的 LogSink 实现
class LogFacade {
 public:
  LogFacade() = delete;

  static void RegisterSink(LogSink* sink) noexcept {
    sink_ = sink;
  }

  static void UnregisterSink() noexcept {
    sink_ = nullptr;
  }

  static void Debug(std::string_view domain, std::string_view msg) noexcept {
    Write(domain, LogLevel::Debug, msg);
  }

  static void Info(std::string_view domain, std::string_view msg) noexcept {
    Write(domain, LogLevel::Info, msg);
  }

  static void Warn(std::string_view domain, std::string_view msg) noexcept {
    Write(domain, LogLevel::Warn, msg);
  }

  static void Error(std::string_view domain, std::string_view msg) noexcept {
    Write(domain, LogLevel::Error, msg);
  }

  static void Fatal(std::string_view domain, std::string_view msg) noexcept {
    Write(domain, LogLevel::Fatal, msg);
  }

  /// 带 traceId 的日志
  static void Trace(std::string_view domain, std::string_view msg,
                    std::string_view trace_id) noexcept {
    Write(domain, LogLevel::Info,
          std::format("[{}] {}", trace_id, msg));
  }

  /// 耗时日志
  static void Timing(std::string_view domain, std::string_view phase,
                     int64_t ms) noexcept {
    Write(domain, LogLevel::Info,
          std::format("{} took {}ms", phase, ms));
  }

  /// 获取 ScopedTimer 的析构回调
  static LogContext::ScopedTimer::Callback TimerCallback(
      std::string_view domain) noexcept {
    return [domain](std::string_view phase, int64_t ms) {
      Timing(domain, phase, ms);
    };
  }

 private:
  static void Write(std::string_view domain, LogLevel level,
                    std::string_view msg) noexcept {
    if (sink_) {
      sink_->Write(domain, level, msg);
    }
  }

  static LogSink* sink_;
};

}  // namespace vision_simple