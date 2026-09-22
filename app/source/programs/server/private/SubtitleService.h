#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vision_simple {
class InferenceService;
enum class SubtitleFailure {
  kInvalid,
  kMissing,
  kUnknownModel,
  kBusy,
  kCapacity,
  kClosed,
  kTooLarge,
  kInvalidVideo,
  kNotReady,
  kFailed
};
template <class T>
using SubtitleResult = std::expected<T, SubtitleFailure>;
struct SubtitleOptions {
  std::string model;
  unsigned sample_interval_ms = 200;
  std::array<double, 4> roi{0.0, 0.5, 1.0, 0.5};
  double min_confidence = 0.5;
  unsigned stable_samples = 2;
  unsigned gap_samples = 2;
};
enum class SubtitleJobState {
  kCreated,
  kUploading,
  kQueued,
  kRunning,
  kCancelling,
  kCancelled,
  kCompleted,
  kFailed
};
struct SubtitleJobInfo {
  std::string id;
  SubtitleJobState state = SubtitleJobState::kCreated;
  uint64_t uploaded_bytes = 0;
  uint64_t decoded_frames = 0;
  uint64_t sampled_frames = 0;
  int64_t position_ms = 0;
  std::optional<int64_t> duration_ms;
  size_t cue_count = 0;
  // Stable, path-free asynchronous failure code; absent unless failed.
  std::optional<std::string> error_code;
};
struct SubtitleJobPage {
  std::vector<SubtitleJobInfo> jobs;
  std::string next_cursor;
};

// Eight jobs including pending uploads and retained terminal results; one
// worker. Upload <=64 MiB, duration <=1800s, <=1000000 decoded frames,
// <=16777216 pixels. Created/uploading idle jobs expire after 60s; terminal
// jobs after 300s. Uploaded bytes never enter an unbounded HTTP body. No URLs
// or user paths.
class SubtitleService {
 public:
  static SubtitleResult<std::shared_ptr<SubtitleService>> Create(
      std::shared_ptr<InferenceService> inference) noexcept;
  ~SubtitleService();
  SubtitleService(const SubtitleService&) = delete;
  SubtitleService& operator=(const SubtitleService&) = delete;
  SubtitleResult<SubtitleJobInfo> Add(SubtitleOptions options) noexcept;
  SubtitleResult<SubtitleJobInfo> Get(const std::string& id) noexcept;
  SubtitleResult<SubtitleJobPage> List(size_t limit,
                                       const std::string& after) noexcept;
  SubtitleResult<void> BeginUpload(const std::string& id) noexcept;
  SubtitleResult<void> AppendUpload(const std::string& id,
                                    std::span<const char> bytes) noexcept;
  SubtitleResult<void> FinishUpload(const std::string& id) noexcept;
  void AbortUpload(const std::string& id) noexcept;
  SubtitleResult<void> Cancel(const std::string& id) noexcept;
  // Created or terminal jobs only; cancel active work and poll before deleting.
  SubtitleResult<void> Delete(const std::string& id) noexcept;
  SubtitleResult<std::string> Download(const std::string& id,
                                       bool webvtt) noexcept;
  void Stop() noexcept;

 private:
  struct Impl;
  explicit SubtitleService(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
}  // namespace vision_simple
