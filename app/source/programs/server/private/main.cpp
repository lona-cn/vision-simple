#include <hv/hasync.h>
#include <hv/hv.h>
#include <ylt/struct_yaml/yaml_reader.h>

#include <csignal>
#include <iostream>
#include <magic_enum.hpp>
#include <thread>

#include "HTTPServer.h"
#include "IOUtil.h"
#include "Logger.h"

#define LOG_DOMAIN_NAME "main"

namespace {
constexpr std::string_view SERVER_YAML_PATH = "config/server.yaml";
std::function<void()> CleanUp = nullptr;

[[noreturn]] void signal_handler(int signal) {
  if (CleanUp) CleanUp();
  std::exit(signal);
}

void RegisterSignals() {
  std::signal(SIGINT, signal_handler);
  // std::signal(SIGILL, signal_handler);
  // std::signal(SIGFPE, signal_handler);
  // std::signal(SIGSEGV, signal_handler);
  std::signal(SIGTERM, signal_handler);
  std::signal(SIGABRT, signal_handler);
#if defined(_WIN32)
  std::signal(SIGBREAK, signal_handler);
  std::signal(SIGABRT_COMPAT, signal_handler);
#endif
}
}  // namespace

int main(int argc, char* argv[]) try {
  RegisterSignals();
#if defined(_WIN32)
  SetConsoleOutputCP(CP_UTF8);
#endif
  vision_simple::HTTPServerOptions options{
      .host = "", .port = 11451, .options = {}};
  auto server_yaml_str_result =
      vision_simple::ReadAllString(std::string{SERVER_YAML_PATH});
  if (!server_yaml_str_result) {
    const auto& error = server_yaml_str_result.error();
    std::cerr << "Unable to read server configuration: " << error.message
              << '\n';
    return 1;
  }
  struct_yaml::from_yaml(options, *server_yaml_str_result);
  auto server_result = vision_simple::HTTPServer::Create(std::move(options));
  if (!server_result) {
    std::cerr << "Unable to create server: " << server_result.error().message
              << '\n';
    return 1;
  }
  auto start_result = (*server_result)->StartAsync();
  if (!start_result) {
    std::cerr << "Unable to start server: " << start_result.error().message
              << '\n';
    return 1;
  }
  CleanUp = [&, mutex = std::make_shared<std::mutex>(),
             cleaned = std::make_shared<std::atomic<bool>>(false)] {
    if (!cleaned->load()) {
      auto lock = std::lock_guard{*mutex};
      if (!cleaned->load()) {
        cleaned->store(true);
        (*server_result)->Stop();
        hv::async::cleanup();
      }
    }
  };
  auto logger_result = vision_simple::Logger::Instance();
  logger_result->get().Info(
      LOG_DOMAIN_NAME, std::format("current workdir: {}",
                                   std::filesystem::current_path().string()));
  while (getchar() != '\n') {
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
  }
  if (CleanUp) CleanUp();
  return 0;
} catch (const std::exception& error) {
  std::cerr << "Server failed: " << error.what() << '\n';
  return 1;
}
