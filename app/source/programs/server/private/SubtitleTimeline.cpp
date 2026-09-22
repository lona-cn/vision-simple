#include "SubtitleTimeline.h"

#include <string_view>
#include <utility>

namespace vision_simple {
namespace {
constexpr size_t kMaxObservation = 4096;
constexpr size_t kMaxCues = 10000;
constexpr size_t kMaxText = 2 * 1024 * 1024;

// Validate before normalizing: malformed UTF-8 must not silently change
// captions.
bool Normalize(std::string_view input, std::string& output) {
  if (input.size() > kMaxObservation) return false;
  output.clear();
  output.reserve(input.size());
  bool space = false, newline = false;
  for (size_t i = 0; i < input.size();) {
    const size_t begin = i;
    const auto first = static_cast<unsigned char>(input[i++]);
    uint32_t code = first;
    unsigned count = 0;
    if (first >= 0xc2 && first <= 0xdf) {
      code = first & 0x1f;
      count = 1;
    } else if (first >= 0xe0 && first <= 0xef) {
      code = first & 0x0f;
      count = 2;
    } else if (first >= 0xf0 && first <= 0xf4) {
      code = first & 0x07;
      count = 3;
    } else if (first >= 0x80) {
      return false;
    }
    if (input.size() - i < count) return false;
    for (unsigned j = 0; j < count; ++j) {
      const auto next = static_cast<unsigned char>(input[i++]);
      if ((next & 0xc0) != 0x80) return false;
      code = (code << 6) | (next & 0x3f);
    }
    if ((count == 1 && code < 0x80) || (count == 2 && code < 0x800) ||
        (count == 3 && code < 0x10000) || code > 0x10ffff ||
        (code >= 0xd800 && code <= 0xdfff))
      return false;
    if (code == '\r' || code == '\n' || code == 0x2028 || code == 0x2029) {
      newline = !output.empty();
      space = false;
      continue;
    }
    if (code <= 0x20 || (code >= 0x7f && code <= 0x9f) || code == 0xa0) {
      space = !output.empty();
      continue;
    }
    if (code == 0xfeff) continue;
    if (newline)
      output.push_back('\n');
    else if (space)
      output.push_back(' ');
    newline = space = false;
    output.append(input.substr(begin, i - begin));
  }
  return true;
}

void AppendTime(std::string& output, int64_t milliseconds, char separator) {
  const auto hours = milliseconds / 3600000;
  const auto minutes = milliseconds / 60000 % 60;
  const auto seconds = milliseconds / 1000 % 60;
  const auto fraction = milliseconds % 1000;
  const auto append = [&output](int64_t value, size_t width) {
    const auto digits = std::to_string(value);
    if (digits.size() < width) output.append(width - digits.size(), '0');
    output += digits;
  };
  append(hours, 2);
  output += ':';
  append(minutes, 2);
  output += ':';
  append(seconds, 2);
  output += separator;
  append(fraction, 3);
}
}  // namespace

SubtitleTimeline::SubtitleTimeline(unsigned stable_samples,
                                   unsigned gap_samples)
    : stable_samples_(stable_samples), gap_samples_(gap_samples) {}

bool SubtitleTimeline::Close(int64_t end_ms) {
  if (active_.empty()) return true;
  if (end_ms < active_start_) return false;
  if (end_ms > active_start_) {
    cues_.push_back({active_start_, end_ms, active_});
  } else {
    text_bytes_ -= active_.size();
  }
  active_.clear();
  return true;
}

bool SubtitleTimeline::Observe(int64_t timestamp_ms,
                               std::string text) noexcept {
  try {
    if (finished_ || stable_samples_ == 0 || gap_samples_ == 0 ||
        timestamp_ms < 0 || timestamp_ms <= last_ms_)
      return false;
    std::string normalized;
    if (!Normalize(text, normalized)) return false;
    last_ms_ = timestamp_ms;
    if (normalized.empty()) {
      candidate_.clear();
      candidate_count_ = 0;
      if (active_.empty()) return true;
      if (gap_count_ == 0) gap_start_ = timestamp_ms;
      if (++gap_count_ == gap_samples_) {
        if (!Close(gap_start_)) return false;
        gap_count_ = 0;
      }
      return true;
    }
    gap_count_ = 0;
    if (normalized == active_) {
      candidate_.clear();
      candidate_count_ = 0;
      return true;
    }
    if (normalized != candidate_) {
      candidate_ = std::move(normalized);
      candidate_start_ = timestamp_ms;
      candidate_count_ = 1;
    } else {
      ++candidate_count_;
    }
    if (candidate_count_ < stable_samples_) return true;
    // Count the active cue too; resource failure cannot leave an unbounded
    // tail.
    if (cues_.size() + (active_.empty() ? 0 : 1) >= kMaxCues ||
        candidate_.size() > kMaxText - text_bytes_) {
      finished_ = true;
      return false;
    }
    if (!Close(candidate_start_)) return false;
    active_ = std::move(candidate_);
    candidate_.clear();
    active_start_ = candidate_start_;
    text_bytes_ += active_.size();
    candidate_count_ = 0;
    return true;
  } catch (...) {
    finished_ = true;
    return false;
  }
}

bool SubtitleTimeline::Finish(int64_t end_ms) noexcept {
  try {
    if (finished_ || stable_samples_ == 0 || gap_samples_ == 0 || end_ms < 0 ||
        end_ms < last_ms_)
      return false;
    if (!Close(gap_count_ ? gap_start_ : end_ms)) return false;
    candidate_.clear();
    candidate_count_ = gap_count_ = 0;
    finished_ = true;
    return true;
  } catch (...) {
    finished_ = true;
    return false;
  }
}

const std::vector<SubtitleCue>& SubtitleTimeline::cues() const noexcept {
  return cues_;
}

std::string EncodeSubtitles(const std::vector<SubtitleCue>& cues, bool webvtt) {
  if (cues.size() > kMaxCues) return {};
  std::string output = webvtt ? "WEBVTT\n\n" : "";
  int64_t previous_end = 0;
  size_t bytes = 0, number = 0;
  for (const auto& cue : cues) {
    std::string normalized;
    if (cue.start_ms < previous_end || cue.end_ms <= cue.start_ms ||
        !Normalize(cue.text, normalized) || normalized.empty() ||
        normalized.size() > kMaxText - bytes)
      return {};
    bytes += normalized.size();
    previous_end = cue.end_ms;
    if (!webvtt) output += std::to_string(++number) + '\n';
    AppendTime(output, cue.start_ms, webvtt ? '.' : ',');
    output += " --> ";
    AppendTime(output, cue.end_ms, webvtt ? '.' : ',');
    output += '\n';
    for (char c : normalized) {
      switch (c) {
        case '&':
          output += "&amp;";
          break;
        case '<':
          output += "&lt;";
          break;
        case '>':
          output += "&gt;";
          break;
        default:
          output += c;
          break;
      }
    }
    output += "\n\n";
  }
  return output;
}
}  // namespace vision_simple
