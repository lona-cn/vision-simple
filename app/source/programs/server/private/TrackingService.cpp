#include "TrackingService.h"

#include <bit>
#include <chrono>
#include <cmath>
#include <map>
#include <mutex>
#include <utility>

#include "ImageCodec.h"

namespace vision_simple {
struct TrackingService::Impl {
  using Clock = std::chrono::steady_clock;
  struct Session {
    std::mutex mutex;
    std::unique_ptr<Tracker> tracker;
    Clock::time_point touched = Clock::now();
  };
  struct Lease {
    std::shared_ptr<Session> session;
    std::unique_lock<std::mutex> lock;
  };
  std::mutex mutex;
  std::map<std::string, std::shared_ptr<Session>> sessions;
  bool closed = false;
  std::once_flag stopped;

  void Sweep() {
    const auto now = Clock::now();
    for (auto it = sessions.begin(); it != sessions.end();) {
      auto session = it->second;
      std::unique_lock lock(session->mutex, std::try_to_lock);
      if (lock && now - session->touched >= std::chrono::seconds(300))
        it = sessions.erase(it);
      else
        ++it;
    }
  }
  TrackingResult<Lease> Acquire(const std::string& id) {
    std::lock_guard guard(mutex);
    if (closed) return std::unexpected(TrackingFailure::kClosed);
    Sweep();
    auto it = sessions.find(id);
    if (it == sessions.end()) return std::unexpected(TrackingFailure::kMissing);
    std::unique_lock lock(it->second->mutex, std::try_to_lock);
    if (!lock) return std::unexpected(TrackingFailure::kBusy);
    return Lease{it->second, std::move(lock)};
  }
};
TrackingService::TrackingService() : impl_(std::make_unique<Impl>()) {}
TrackingService::~TrackingService() { Stop(); }
TrackingResult<void> TrackingService::Add(std::string id,
                                          TrackerOptions options) {
  if (options.max_tracks > 256 || options.max_detections > 256)
    return std::unexpected(TrackingFailure::kInvalid);
  auto tracker = Tracker::Create(options);
  if (!tracker)
    return std::unexpected(
        tracker.error().code == VisionSimpleErrorCode::kParameterError ||
                tracker.error().code == VisionSimpleErrorCode::kRangeError
            ? TrackingFailure::kInvalid
            : TrackingFailure::kFailed);
  auto session = std::make_shared<Impl::Session>();
  session->tracker = std::move(*tracker);
  std::lock_guard lock(impl_->mutex);
  if (impl_->closed) return std::unexpected(TrackingFailure::kClosed);
  impl_->Sweep();
  if (impl_->sessions.size() >= 32)
    return std::unexpected(TrackingFailure::kCapacity);
  if (!impl_->sessions.emplace(std::move(id), std::move(session)).second)
    return std::unexpected(TrackingFailure::kFailed);
  return {};
}
TrackingResult<TrackingSessionInfo> TrackingService::Get(
    const std::string& id) {
  auto lease = impl_->Acquire(id);
  if (!lease) return std::unexpected(lease.error());
  return TrackingSessionInfo{id, lease->session->tracker->options().algorithm,
                             lease->session->tracker->Status()};
}
TrackingResult<TrackingSessionPage> TrackingService::List(
    size_t limit, const std::string& after) {
  if (!limit || limit > 100) return std::unexpected(TrackingFailure::kInvalid);
  std::lock_guard lock(impl_->mutex);
  if (impl_->closed) return std::unexpected(TrackingFailure::kClosed);
  impl_->Sweep();
  TrackingSessionPage page;
  auto it = impl_->sessions.upper_bound(after);
  for (; it != impl_->sessions.end() && page.ids.size() < limit; ++it)
    page.ids.push_back(it->first);
  if (it != impl_->sessions.end()) page.next_cursor = page.ids.back();
  return page;
}
TrackingResult<TrackingFrameResult> TrackingService::Step(
    const std::string& id, uint64_t index, double timestamp,
    std::span<const TrackingDetection> detections,
    std::optional<std::string_view> encoded_image) {
  if (index > 9007199254740991ULL ||
      (std::bit_cast<uint64_t>(timestamp) & 0x7ff0000000000000ULL) ==
          0x7ff0000000000000ULL ||
      timestamp < 0 || detections.size() > 256)
    return std::unexpected(TrackingFailure::kInvalid);
  for (const auto& detection : detections)
    if (detection.embedding.size() > 512)
      return std::unexpected(TrackingFailure::kInvalid);
  auto lease = impl_->Acquire(id);
  if (!lease) return std::unexpected(lease.error());
  auto& session = *lease->session;
  const auto status = session.tracker->Status();
  if ((status.last_frame_index && index <= *status.last_frame_index) ||
      (status.last_timestamp && timestamp <= *status.last_timestamp))
    return std::unexpected(TrackingFailure::kOrder);
  cv::Mat image;
  if (encoded_image) {
    try {
      auto decoded = DecodeEncodedImage(*encoded_image, 16777216);
      if (!decoded) return std::unexpected(TrackingFailure::kInvalidImage);
      image = std::move(*decoded);
    } catch (const cv::Exception&) {
      return std::unexpected(TrackingFailure::kInvalidImage);
    }
  }
  auto result = session.tracker->Step(index, timestamp, detections, image);
  if (!result)
    return std::unexpected(
        result.error().code == VisionSimpleErrorCode::kParameterError ||
                result.error().code == VisionSimpleErrorCode::kRangeError
            ? TrackingFailure::kInvalid
            : TrackingFailure::kFailed);
  session.touched = Impl::Clock::now();
  return std::move(*result);
}
TrackingResult<void> TrackingService::Reset(const std::string& id) {
  auto lease = impl_->Acquire(id);
  if (!lease) return std::unexpected(lease.error());
  lease->session->tracker->Reset();
  lease->session->touched = Impl::Clock::now();
  return {};
}
TrackingResult<void> TrackingService::Delete(const std::string& id) {
  std::lock_guard guard(impl_->mutex);
  if (impl_->closed) return std::unexpected(TrackingFailure::kClosed);
  impl_->Sweep();
  auto it = impl_->sessions.find(id);
  if (it == impl_->sessions.end())
    return std::unexpected(TrackingFailure::kMissing);
  auto session = it->second;
  std::unique_lock lock(session->mutex, std::try_to_lock);
  if (!lock) return std::unexpected(TrackingFailure::kBusy);
  impl_->sessions.erase(it);
  return {};
}
void TrackingService::Stop() noexcept {
  std::call_once(impl_->stopped, [&] {
    std::map<std::string, std::shared_ptr<Impl::Session>> closing;
    {
      std::lock_guard lock(impl_->mutex);
      impl_->closed = true;
      closing.swap(impl_->sessions);
    }
    for (auto& [id, session] : closing) {
      std::lock_guard drain(session->mutex);
    }
  });
}
}  // namespace vision_simple
