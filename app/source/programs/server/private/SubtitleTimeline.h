#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vision_simple {
struct SubtitleCue {
  int64_t start_ms = 0;
  int64_t end_ms = 0;
  std::string text;
};

// Consecutive observations confirm changes; short errors/gaps do not split
// cues. UTF-8 text is bounded to 4096 bytes per observation, 10000 cues / 2 MiB
// total.
class SubtitleTimeline {
 public:
  explicit SubtitleTimeline(unsigned stable_samples = 2,
                            unsigned gap_samples = 2);
  bool Observe(int64_t timestamp_ms, std::string text) noexcept;
  bool Finish(int64_t end_ms) noexcept;
  const std::vector<SubtitleCue>& cues() const noexcept;

 private:
  bool Close(int64_t end_ms);
  unsigned stable_samples_, gap_samples_;
  int64_t last_ms_ = -1, active_start_ = 0, candidate_start_ = 0,
          gap_start_ = 0;
  unsigned candidate_count_ = 0, gap_count_ = 0;
  size_t text_bytes_ = 0;
  bool finished_ = false;
  std::string active_, candidate_;
  std::vector<SubtitleCue> cues_;
};

// UTF-8, millisecond timestamps, escaped markup, no empty cue injection.
std::string EncodeSubtitles(const std::vector<SubtitleCue>& cues, bool webvtt);
}  // namespace vision_simple
