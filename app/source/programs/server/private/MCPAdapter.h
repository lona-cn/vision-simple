#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "VisionSimpleCommon.h"

namespace hv {
struct HttpService;
class HttpServer;
}  // namespace hv
namespace vision_simple {
class InferenceService;
// Embedded legacy HTTP+SSE transport. Stop before stopping the HTTP IO loops.
class MCPAdapter {
 public:
  static VSResult<std::unique_ptr<MCPAdapter>> Create(
      std::shared_ptr<InferenceService> service, std::string host,
      uint16_t port) noexcept;
  ~MCPAdapter();
  MCPAdapter(const MCPAdapter&) = delete;
  MCPAdapter& operator=(const MCPAdapter&) = delete;
  void Mount(hv::HttpService& service, hv::HttpServer& server);
  void Stop() noexcept;

 private:
  struct State;
  explicit MCPAdapter(std::shared_ptr<State> state);
  std::shared_ptr<State> state_;
};
}  // namespace vision_simple
