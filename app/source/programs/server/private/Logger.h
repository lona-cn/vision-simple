#pragma once
#include <memory>
#include <string>

#include "VisionSimpleCommon.h"

namespace vision_simple {
enum class LogLevel:uint8_t {
  Debug,
  Info,
  Warn,
  Error,
  Fatal
};

class Logger {
  struct Impl;
  std::unique_ptr<Impl> impl_;

  Logger(const std::string& config_path);

public:
  static VSResult<std::reference_wrapper<Logger>> Instance() noexcept;

  void Log(std::string_view domain, std::string_view message,
           LogLevel log_level = LogLevel::Info) const noexcept;

  void Debug(std::string_view domain, std::string_view message) const noexcept {
    Log(domain, message, LogLevel::Debug);
  }

  void Info(std::string_view domain, std::string_view message) const noexcept {
    Log(domain, message, LogLevel::Info);
  }

  void Warn(std::string_view domain, std::string_view message) const noexcept {
    Log(domain, message, LogLevel::Warn);
  }

  void Error(std::string_view domain, std::string_view message) const noexcept {
    Log(domain, message, LogLevel::Error);
  }

  void Fatal(std::string_view domain, std::string_view message) const noexcept {
    Log(domain, message, LogLevel::Fatal);
  }
};
}
