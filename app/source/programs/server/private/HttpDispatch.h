#pragma once

#include <hv/HttpContext.h>

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <stop_token>

#include "VisionSimpleCommon.h"

namespace vision_simple {
// Bounds HTTP handler residency, not inference image credit or decoded bytes.
class HttpDispatch {
 public:
  using Clock = std::chrono::steady_clock;
  using Reply = std::function<void(const HttpContextPtr&)>;
  using Work = std::function<Reply(std::stop_token, Clock::time_point)>;
  enum class Lane { kData, kControl };
  struct Options {
    size_t data_workers = 4;
    size_t data_queue_capacity = 4;
    size_t control_workers = 1;
    size_t control_queue_capacity = 4;
  };

  static VSResult<std::unique_ptr<HttpDispatch>> Create(Options options) noexcept;
  ~HttpDispatch();
  HttpDispatch(const HttpDispatch&) = delete;
  HttpDispatch& operator=(const HttpDispatch&) = delete;

  // Called on the context's IO thread. Streaming uploads may submit before
  // the body is complete by passing false and owning a bounded chunk stream.
  // Work must own its inputs and must not read/write the context or writer.
  // false leaves the context untouched; the caller sends its overload envelope.
  bool Submit(const HttpContextPtr& ctx, Lane lane, Work work,
              bool body_complete = true) noexcept;
  bool Accepting() const noexcept;
  void BeginStop() noexcept;
  // Call outside an IO thread, with every listener IO loop still running.
  void Stop() noexcept;

 private:
  struct Impl;
  explicit HttpDispatch(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};
}  // namespace vision_simple
