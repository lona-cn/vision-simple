#include "Logger.h"

#include <log4cplus/configurator.h>
#include <log4cplus/initializer.h>
#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>

#include <codecvt>
#include <fstream>
#include <iostream>
#include <locale>
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

vision_simple::Logger::Logger(log4cplus::tistream& properties)
    : impl_(std::make_unique<Impl>()) {
  std::cout << "    initialize log system" << std::endl;
  log4cplus::PropertyConfigurator{properties}.configure();
}

vision_simple::VSResult<std::reference_wrapper<vision_simple::Logger>>
vision_simple::Logger::Instance() noexcept {
  static std::mutex instance_mutex;
  static Logger* instance = nullptr;
  std::lock_guard lock{instance_mutex};
  if (!instance) {
    // Decode UTF-8 independently of the process locale and log4cplus's
    // optional encoding flags, which are not exposed by every package build.
    std::basic_ifstream<log4cplus::tchar> properties;
#if defined(UNICODE)
    properties.imbue(std::locale(
        properties.getloc(),
        new std::codecvt_utf8<wchar_t, 0x10FFFF, std::consume_header>));
#endif
    properties.open(std::string(CONFIG_PATH), std::ios::binary);
    if (!properties) {
      return std::unexpected{VisionSimpleError{
          VisionSimpleErrorCode::kIOError,
          std::format("Unable to open logging configuration:{}", CONFIG_PATH)}};
    }
    // Function-local construction registers destruction after log4cplus's
    // own lazy context. A global unique_ptr registers too early.
    static Logger singleton{properties};
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
