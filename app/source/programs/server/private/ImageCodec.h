#pragma once

#include <cstddef>
#include <cstdint>
#include <opencv2/core.hpp>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace vision_simple {
struct PreparedImage {
  std::vector<uint8_t> bytes;
  size_t pixels;
};
// Decode canonical base64 once and inspect dimensions without invoking a codec.
// Reserve decoded storage before DecodeImageBytes and verify its pixel count;
// preflight validates headers, not the compressed image payload.
std::optional<PreparedImage> PrepareEncodedImage(std::string_view text);
// Codec/allocation exceptions belong to the caller; malformed data returns nullopt.
// A pixel limit enables PNG/JPEG-only header preflight before codec allocation.
std::optional<cv::Mat> DecodeEncodedImage(
    std::string_view text, std::optional<size_t> max_pixels = std::nullopt);
std::optional<cv::Mat> DecodeImageBytes(
    std::span<const uint8_t> bytes,
    std::optional<size_t> max_pixels = std::nullopt,
    cv::Mat* storage = nullptr);
}  // namespace vision_simple
