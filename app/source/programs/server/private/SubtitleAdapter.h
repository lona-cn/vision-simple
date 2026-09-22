#pragma once

#include <memory>

#include "VisionSimpleCommon.h"

namespace hv {
struct HttpService;
}
namespace vision_simple {
class InferenceService;
class SubtitleAdapter {
 public:
  static VSResult<std::unique_ptr<SubtitleAdapter>> Create(
      std::shared_ptr<InferenceService> inference) noexcept;
  ~SubtitleAdapter();
  SubtitleAdapter(const SubtitleAdapter&) = delete;
  SubtitleAdapter& operator=(const SubtitleAdapter&) = delete;
  void Mount(hv::HttpService& service);
  void Stop() noexcept;

 private:
  struct State;
  explicit SubtitleAdapter(std::shared_ptr<State> state);
  std::shared_ptr<State> state_;
};
}  // namespace vision_simple
