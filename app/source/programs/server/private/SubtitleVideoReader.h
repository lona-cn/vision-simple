#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <opencv2/core.hpp>
#include <optional>
#include <span>
#include <stop_token>

namespace vision_simple {
// Only service-generated private upload paths may be passed to Open.
class SubtitleVideoReader {
 public:
  enum class Container { kInvalid, kAvi, kMp4, kMkv, kAsf };
  // Bounded signature/size classification, not full demux or codec validation.
  static Container ProbeContainer(std::span<const uint8_t> prefix,
                                  uint64_t total_bytes) noexcept;
  // Compile-time reader availability; codec/startup failures remain asynchronous.
  static bool SupportsContainer(Container container) noexcept;
  // Only supported containers have a private upload extension.
  static const char* ContainerExtension(Container container) noexcept;
  enum class Result { kFrame, kEof, kError, kCancelled };
  struct Frame {
    // Decoder-owned pixels are reused by the next Read; consume before then.
    cv::Mat image;
    int64_t start_ms = 0;
    int64_t end_ms = 0;
  };
  SubtitleVideoReader();
  ~SubtitleVideoReader();
  const char* Open(const std::filesystem::path& path, std::stop_token cancel);
  Result Read(Frame& frame);
  const char* error() const;
  std::optional<int64_t> duration_hint() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace vision_simple
