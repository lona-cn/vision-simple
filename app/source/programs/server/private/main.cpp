#include <hv/hasync.h>
#include <hv/hv.h>
#include <ylt/struct_yaml/yaml_reader.h>

#include <iostream>
#include <magic_enum.hpp>
#include <thread>

#include "HTTPServer.h"
#include "IOUtil.h"

#define CHECK_RESULT(result)                                      \
  do {                                                            \
    if (!(result)) {                                              \
      auto&(err) = (result).error();                              \
      std::cout << std::format("failed code:{} message:{}",       \
                               magic_enum::enum_name((err).code), \
                               (err).message)                     \
                << std::endl;                                     \
      return -1;                                                  \
    }                                                             \
  } while (0)

namespace {
constexpr std::string_view SERVER_YAML_PATH = "config/server.yaml";
}

int main(int argc, char* argv[]) {
  vision_simple::HTTPServerOptions options{
      .host = "localhost", .port = 11451, .options = {}};
  auto server_yaml_str_result =
      vision_simple::ReadAllString(SERVER_YAML_PATH.data());
  if (!server_yaml_str_result) {
    std::cout << std::format("failed to Read file:{} with error:{}",
                             SERVER_YAML_PATH,
                             server_yaml_str_result.error().message)
              << std::endl;
    return -1;
  }
  std::error_code error_code;
  struct_yaml::from_yaml(options, *server_yaml_str_result, error_code);
  if (error_code) {
    std::cout << std::format("failed to parse yaml:{},error:{}",
                             SERVER_YAML_PATH, error_code.message())
              << std::endl;
    return -1;
  }
  auto server_result = vision_simple::HTTPServer::Create(
      vision_simple::HTTPServerOptions{options});
  CHECK_RESULT(server_result);
  (*server_result)->StartAsync();
  while (getchar() != '\n') {
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
  }
  (*server_result)->Stop();
  hv::async::cleanup();
  return 0;
};
