#include "Logger.h"
#include <atomic>
#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>
#include <log4cplus/configurator.h>
#include <log4cplus/initializer.h>
#include <algorithm>
#include <filesystem>
#include <iostream>

#include "VisionSimpleCommon.h"

namespace {
std::unique_ptr<log4cplus::Initializer> log4cplus_init{nullptr};
std::unique_ptr<vision_simple::Logger> instance = nullptr;
std::mutex instance_mutex{};
std::string_view CONFIG_PATH = "config/log.properties";
auto LogSystemInitialize = [mutex = std::make_shared<std::mutex>(),
      should_run = std::make_shared<std::atomic<bool>>(true)] {
  if (should_run->load()) {
    std::lock_guard lock(*mutex);
    if (should_run->load()) {
      should_run->store(false);
      std::cout << "    initialize log system" << std::endl;
      log4cplus_init = std::make_unique<log4cplus::Initializer>();
    }
  }
};
}

struct StringHash {
  using is_transparent = void;

  [[nodiscard]] size_t operator()(const char* txt) const {
    return std::hash<std::string_view>{}(txt);
  }

  [[nodiscard]] size_t operator()(std::string_view txt) const {
    return std::hash<std::string_view>{}(txt);
  }

  [[nodiscard]] size_t operator()(const std::string& txt) const {
    return std::hash<std::string>{}(txt);
  }
};

struct vision_simple::Logger::Impl {
  std::unordered_map<std::string, log4cplus::Logger, StringHash,
                     std::equal_to<>> loggers{};

  log4cplus::Logger& GetLogger(std::string_view logger_name) noexcept {
    if (auto it = loggers.find(logger_name); it != loggers.end()) {
      return it->second;
    }
    auto logger = log4cplus::Logger::getInstance(
        LOG4CPLUS_STRING_TO_TSTRING(std::string(logger_name)));
    loggers.emplace(logger_name, std::move(logger));
    return loggers.find(logger_name)->second;
  }
};

vision_simple::Logger::Logger(const std::string& config_path)
  : impl_(std::make_unique<Impl>()) {
  LogSystemInitialize();
  log4cplus::PropertyConfigurator::doConfigure(
      LOG4CPLUS_STRING_TO_TSTRING(config_path));
}

vision_simple::VSResult<std::reference_wrapper<vision_simple::Logger>>
vision_simple::Logger::Instance() noexcept {
  if (!instance) {
    auto lock = std::lock_guard{instance_mutex};
    if (!instance) {
      if (!std::filesystem::exists(CONFIG_PATH)) {
        return std::unexpected{
            VisionSimpleError{VisionSimpleErrorCode::kIOError,
                              std::format("Configuration file not found:{}",
                                          CONFIG_PATH)
            }
        };
      }
      instance = std::unique_ptr<Logger>(new Logger{std::string(CONFIG_PATH)});
    }
  }
  return *instance;
}

void vision_simple::Logger::Log(std::string_view domain,
                                std::string_view message,
                                LogLevel log_level) const noexcept {
  auto logger = impl_->GetLogger(domain);
  switch (log_level) {
    case LogLevel::Debug:
      LOG4CPLUS_DEBUG(logger, message.data());
      break;
    case LogLevel::Info:
      LOG4CPLUS_INFO(logger, message.data());
      break;
    case LogLevel::Warn:
      LOG4CPLUS_WARN(logger, message.data());
      break;
    case LogLevel::Error:
      LOG4CPLUS_ERROR(logger, message.data());
      break;
    case LogLevel::Fatal:
      LOG4CPLUS_FATAL(logger, message.data());
      break;
  }
}
