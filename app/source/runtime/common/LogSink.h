#pragma once
#include <string_view>

namespace vision_simple {

/// 日志级别 — 与现有 LogLevel 保持一致
enum class LogLevel : uint8_t {
  Debug,
  Info,
  Warn,
  Error,
  Fatal
};

/// 日志接收器抽象接口
/// server 模块注入 log4cplus 实现, infer 模块通过 LogFacade 间接使用
class LogSink {
 public:
  virtual ~LogSink() = default;

  /// 写入一条日志
  virtual void Write(std::string_view domain, LogLevel level,
                     std::string_view message) noexcept = 0;
};

}  // namespace vision_simple