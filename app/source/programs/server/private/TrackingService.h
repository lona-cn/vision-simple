#pragma once

#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "Tracker.h"

namespace vision_simple {
enum class TrackingFailure {
  kInvalid,
  kInvalidImage,
  kMissing,
  kBusy,
  kOrder,
  kCapacity,
  kClosed,
  kFailed
};
template <class T>
using TrackingResult = std::expected<T, TrackingFailure>;
struct TrackingSessionInfo {
  std::string id;
  TrackerAlgorithm algorithm;
  TrackingStatus status;
};
struct TrackingSessionPage {
  std::vector<std::string> ids;
  std::string next_cursor;
};
// Transport-independent stream ownership. Stop closes admission and waits for
// all admitted native operations; callers must still drain transport handlers.
class TrackingService {
 public:
  TrackingService();
  ~TrackingService();
  TrackingResult<void> Add(std::string id, TrackerOptions options);
  TrackingResult<TrackingSessionInfo> Get(const std::string& id);
  TrackingResult<TrackingSessionPage> List(size_t limit,
                                           const std::string& after);
  TrackingResult<TrackingFrameResult> Step(
      const std::string& id, uint64_t index, double timestamp,
      std::span<const TrackingDetection> detections,
      std::optional<std::string_view> encoded_image);
  TrackingResult<void> Reset(const std::string& id);
  TrackingResult<void> Delete(const std::string& id);
  void Stop() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace vision_simple
