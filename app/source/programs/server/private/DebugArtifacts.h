#pragma once
#include "InferenceService.h"
#include "OCRDetectionOptions.h"
#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace vision_simple {
// Explicit local diagnostic output only; construction performs no filesystem work.
class DebugArtifacts {
 public:
  DebugArtifacts();
  ~DebugArtifacts();
  DebugArtifacts(const DebugArtifacts&) = delete;
  DebugArtifacts& operator=(const DebugArtifacts&) = delete;
  static bool ValidName(std::string_view name) noexcept;
  std::expected<void, const char*> Init(std::string_view name, size_t max_bytes, size_t max_files) noexcept;
  std::expected<void, const char*> Export(std::span<const std::string> images,
      const InferOCRResponse& response, OCRDetectionOptions options, size_t max_pixels) noexcept;
  std::expected<void, const char*> Rollback() noexcept;
  void Retain() noexcept;
  size_t Files() const noexcept;
  size_t Bytes() const noexcept;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace vision_simple
