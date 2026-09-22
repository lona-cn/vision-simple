#pragma once
#include <memory>

#include "VisionSimpleCommon.h"
namespace hv {
struct HttpService;
}
namespace vision_simple {
class TrackingAdapter {
 public:
  static VSResult<std::unique_ptr<TrackingAdapter>> Create() noexcept;
  ~TrackingAdapter();
  TrackingAdapter(const TrackingAdapter&) = delete;
  TrackingAdapter& operator=(const TrackingAdapter&) = delete;
  void Mount(hv::HttpService& service);
  void Stop() noexcept;

 private:
  struct State;
  explicit TrackingAdapter(std::shared_ptr<State> state);
  std::shared_ptr<State> state_;
};
}  // namespace vision_simple
