#include <hv/hasync.h>
#include <hv/hv.h>
#include <ylt/struct_yaml/yaml_reader.h>

#include <csignal>
#include <iostream>
#include <chrono>
#include <thread>
#if !defined(_WIN32)
#include <poll.h>
#include <unistd.h>
#endif

#include "HTTPServer.h"
#include "Diagnostics.h"
#include "IOUtil.h"
#include "Logger.h"

#define LOG_DOMAIN_NAME "main"

namespace {
constexpr std::string_view SERVER_YAML_PATH = "config/server.yaml";
volatile std::sig_atomic_t received_signal = 0;

void signal_handler(int signal) { received_signal = signal; }

void RegisterSignals() {
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  std::signal(SIGABRT, signal_handler);
#if defined(_WIN32)
  std::signal(SIGBREAK, signal_handler);
  std::signal(SIGABRT_COMPAT, signal_handler);
#endif
}

// Do not block in getchar: stdin may remain open without any input while
// shutdown signals arrive. EOF disables input, but keeps waiting for signals.
int WaitForShutdown() {
  bool input_open = true;
#if defined(_WIN32)
  const HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
  DWORD console_mode = 0;
  const bool console = input != INVALID_HANDLE_VALUE && input != nullptr &&
                       GetConsoleMode(input, &console_mode);
  const DWORD input_type = GetFileType(input);
#endif
  while (received_signal == 0) {
    if (input_open) {
#if defined(_WIN32)
      if (console) {
        DWORD count = 0;
        if (!GetNumberOfConsoleInputEvents(input, &count)) {
          input_open = false;
        } else if (count != 0) {
          INPUT_RECORD event{};
          DWORD read = 0;
          if (!ReadConsoleInputW(input, &event, 1, &read)) {
            input_open = false;
          } else if (read && event.EventType == KEY_EVENT &&
                     event.Event.KeyEvent.bKeyDown &&
                     event.Event.KeyEvent.uChar.UnicodeChar == L'\r') {
            return received_signal;
          }
        }
      } else {
        DWORD available = 0;
        bool readable = input_type == FILE_TYPE_DISK;
        if (input_type == FILE_TYPE_PIPE) {
          if (!PeekNamedPipe(input, nullptr, 0, nullptr, &available, nullptr))
            input_open = false;
          else
            readable = available != 0;
        } else if (!readable) {
          input_open = false;
        }
        if (input_open && readable) {
          char bytes[256];
          DWORD read = 0;
          const DWORD capacity = input_type == FILE_TYPE_PIPE
              ? (available < sizeof(bytes) ? available : sizeof(bytes))
              : sizeof(bytes);
          if (!ReadFile(input, bytes, capacity, &read, nullptr) || read == 0) {
            input_open = false;
          } else {
            for (DWORD i = 0; i < read; ++i)
              if (bytes[i] == '\n') return received_signal;
          }
        }
      }
#else
      pollfd input{STDIN_FILENO, POLLIN, 0};
      if (::poll(&input, 1, 100) > 0) {
        if (input.revents & (POLLIN | POLLHUP)) {
          char bytes[256];
          const auto count = ::read(STDIN_FILENO, bytes, sizeof(bytes));
          if (count <= 0) {
            input_open = false;
          } else {
            for (ssize_t i = 0; i < count; ++i)
              if (bytes[i] == '\n') return received_signal;
          }
        } else if (input.revents & (POLLERR | POLLNVAL)) {
          input_open = false;
        }
      }
      if (input_open) continue;
#endif
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
  }
  return received_signal;
}
}  // namespace

int main(int argc, char* argv[]) try {
  if (argc > 1) return vision_simple::RunDiagnosticsCLI(argc, argv);
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
  auto logger_result = vision_simple::Logger::Instance();
  logger_result->get().Info(
      LOG_DOMAIN_NAME, std::format("current workdir: {}",
                                   std::filesystem::current_path().string()));
  const int exit_code = WaitForShutdown();
  (*server_result)->Stop();
  hv::async::cleanup();
  return exit_code;
} catch (const std::exception& error) {
  std::cerr << "Server failed: " << error.what() << '\n';
  return 1;
}
