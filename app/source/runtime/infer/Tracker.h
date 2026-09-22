#pragma once

#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <optional>
#include <span>
#include <vector>

#include "VisionSimpleCommon.h"

namespace vision_simple {

enum class TrackerAlgorithm : uint8_t { kByteTrack, kBoTSORT };
struct TrackerOptions {
  TrackerAlgorithm algorithm = TrackerAlgorithm::kByteTrack;
  float high_threshold = 0.5f;
  float low_threshold = 0.1f;
  float new_track_threshold = 0.6f;
  // Maximum assignment costs (1 - similarity), not minimum IoU values.
  float match_threshold = 0.8f;
  uint32_t max_lost_frames = 30;
  uint32_t min_hits = 2;
  size_t max_tracks = 256;
  size_t max_detections = 256;
  // BoT-SORT only. GMC requires a same-sized CV_8UC3 image on every Step.
  bool camera_motion = true;
  // Caller supplies finite, nonzero appearance embeddings; no implicit model.
  bool appearance = false;
  float proximity_threshold = 0.5f;
  float appearance_threshold = 0.25f;
};
struct TrackingDetection {
  int32_t class_id;
  float confidence;
  cv::Rect2f bbox;
  std::vector<float> embedding;
};
struct TrackedObject {
  uint64_t track_id;
  int32_t class_id;
  float confidence;
  cv::Rect2f bbox;
};
struct TrackingFrameResult {
  uint64_t frame_index;
  double timestamp;
  // Confirmed tracks observed in this frame; lost predictions are not emitted.
  std::vector<TrackedObject> tracks;
};
struct TrackingStatus {
  std::optional<uint64_t> last_frame_index;
  std::optional<double> last_timestamp;
  size_t active_tracks = 0;
  size_t lost_tracks = 0;
};

// One instance is one independent stream. Calls are serialized per instance.
// Indices and timestamps must increase strictly; rejected frames do not advance
// state. Index gaps count towards expiration; timestamps are seconds.
class VISION_SIMPLE_API Tracker {
 public:
  using CreateResult = VSResult<std::unique_ptr<Tracker>>;
  static CreateResult Create(TrackerOptions options = {}) noexcept;
  ~Tracker();
  Tracker(const Tracker&) = delete;
  Tracker& operator=(const Tracker&) = delete;
  VSResult<TrackingFrameResult> Step(
      uint64_t frame_index, double timestamp,
      std::span<const TrackingDetection> detections,
      const cv::Mat& image = {}) noexcept;
  // Starts a fresh sequence, including IDs, timing and motion/appearance state.
  void Reset() noexcept;
  TrackingStatus Status() const noexcept;
  const TrackerOptions& options() const noexcept;

 private:
  struct Impl;
  explicit Tracker(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vision_simple
