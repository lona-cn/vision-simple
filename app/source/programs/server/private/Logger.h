#pragma once
#include <memory>
#include <string>

#include "LogSink.h"
#include "VisionSimpleCommon.h"

namespace vision_simple {

class Logger : public LogSink {
  struct Impl;
  std::unique_ptr<Impl> impl_;

  Logger(const std::string& config_path);

 public:
  static VSResult<std::reference_wrapper<Logger>> Instance() noexcept;

  // 实现 LogSink 接口
  void Write(std::string_view domain, LogLevel level,
             std::string_view message) noexcept override;

  void Debug(std::string_view domain, std::string_view message) const noexcept {
    const_cast<Logger*>(this)->Write(domain, LogLevel::Debug, message);
  }

  void Info(std::string_view domain, std::string_view message) const noexcept {
    const_cast<Logger*>(this)->Write(domain, LogLevel::Info, message);
  }

  void Warn(std::string_view domain, std::string_view message) const noexcept {
    const_cast<Logger*>(this)->Write(domain, LogLevel::Warn, message);
  }

  void Error(std::string_view domain, std::string_view message) const noexcept {
    const_cast<Logger*>(this)->Write(domain, LogLevel::Error, message);
  }

  void Fatal(std::string_view domain, std::string_view message) const noexcept {
    const_cast<Logger*>(this)->Write(domain, LogLevel::Fatal, message);
  }
};
}
