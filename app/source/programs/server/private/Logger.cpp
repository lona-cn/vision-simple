#include "Logger.h"

#include <log4cplus/configurator.h>
#include <log4cplus/initializer.h>
#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>

#include <filesystem>
#include <iostream>
#include <mutex>
#include <unordered_map>

#include "LogFacade.h"
#include "VisionSimpleCommon.h"

namespace {
constexpr std::string_view CONFIG_PATH = "config/log.properties";
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
  // Construct the runtime before cached loggers and destroy it after them.
  log4cplus::Initializer initializer;
  std::unordered_map<std::string, log4cplus::Logger, StringHash,
                     std::equal_to<>>
      loggers{};
  std::mutex loggers_mutex;

  log4cplus::Logger& GetLogger(std::string_view logger_name) noexcept {
    std::lock_guard lock{loggers_mutex};
    if (auto it = loggers.find(logger_name); it != loggers.end()) {
      return it->second;
    }
    auto logger = log4cplus::Logger::getInstance(
        LOG4CPLUS_STRING_TO_TSTRING(std::string(logger_name)));
    loggers.emplace(logger_name, std::move(logger));
    return loggers.find(logger_name)->second;
  }
};
vision_simple::Logger::~Logger() { LogFacade::RegisterSink(nullptr); }

vision_simple::Logger::Logger(const std::string& config_path)
    : impl_(std::make_unique<Impl>()) {
  std::cout << "    initialize log system" << std::endl;
  log4cplus::PropertyConfigurator::doConfigure(
      LOG4CPLUS_STRING_TO_TSTRING(config_path));
}

vision_simple::VSResult<std::reference_wrapper<vision_simple::Logger>>
vision_simple::Logger::Instance() noexcept {
  static std::mutex instance_mutex;
  static Logger* instance = nullptr;
  std::lock_guard lock{instance_mutex};
  if (!instance) {
    if (!std::filesystem::exists(CONFIG_PATH)) {
      return std::unexpected{VisionSimpleError{
          VisionSimpleErrorCode::kIOError,
          std::format("Configuration file not found:{}", CONFIG_PATH)}};
    }
    // Function-local construction registers destruction after log4cplus's
    // own lazy context. A global unique_ptr registers too early.
    static Logger singleton{std::string(CONFIG_PATH)};
    instance = &singleton;
    LogFacade::RegisterSink(instance);
  }
  return *instance;
}

void vision_simple::Logger::Write(std::string_view domain, LogLevel level,
                                  std::string_view message) noexcept {
  auto& logger = impl_->GetLogger(domain);
  switch (level) {
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
