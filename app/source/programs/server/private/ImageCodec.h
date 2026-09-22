#pragma once

#include <cstddef>
#include <cstdint>
#include <opencv2/core.hpp>
#include <optional>
#include <span>
#include <string_view>

namespace vision_simple {
// Canonical raw base64 only. Codec/allocation exceptions belong to the caller's
// request error boundary; malformed data returns nullopt.
// A pixel limit enables PNG/JPEG-only header preflight before codec allocation.
std::optional<cv::Mat> DecodeEncodedImage(
    std::string_view text, std::optional<size_t> max_pixels = std::nullopt);
std::optional<cv::Mat> DecodeImageBytes(
    std::span<const uint8_t> bytes,
    std::optional<size_t> max_pixels = std::nullopt,
    cv::Mat* storage = nullptr);
}  // namespace vision_simple
