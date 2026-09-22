#include "ImageCodec.h"

#include <turbobase64/turbob64.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <vector>

namespace vision_simple {
namespace {
int Base64Digit(unsigned char value) noexcept {
  if (value >= 'A' && value <= 'Z') return value - 'A';
  if (value >= 'a' && value <= 'z') return value - 'a' + 26;
  if (value >= '0' && value <= '9') return value - '0' + 52;
  if (value == '+') return 62;
  if (value == '/') return 63;
  return -1;
}
bool IsBase64(std::string_view text) noexcept {
  if (text.empty() || text.size() % 4 != 0) return false;
  const size_t padding =
      text.back() == '=' ? (text[text.size() - 2] == '=' ? 2 : 1) : 0;
  const size_t digits = text.size() - padding;
  for (size_t i = 0; i < digits; ++i) {
    if (Base64Digit(static_cast<unsigned char>(text[i])) < 0) return false;
  }
  const int last = Base64Digit(static_cast<unsigned char>(text[digits - 1]));
  return (padding != 2 || (last & 15) == 0) &&
         (padding != 1 || (last & 3) == 0);
}
struct Dimensions {
  uint32_t width;
  uint32_t height;
};
std::optional<Dimensions> ImageDimensions(std::span<const uint8_t> bytes) {
  const auto be32 = [&](size_t at) {
    return (uint32_t(bytes[at]) << 24) | (uint32_t(bytes[at + 1]) << 16) |
           (uint32_t(bytes[at + 2]) << 8) | uint32_t(bytes[at + 3]);
  };
  constexpr uint8_t png[] = {137, 80, 78, 71, 13, 10, 26, 10};
  if (bytes.size() >= 33 &&
      std::equal(std::begin(png), std::end(png), bytes.begin())) {
    if (be32(8) != 13 || bytes[12] != 'I' || bytes[13] != 'H' ||
        bytes[14] != 'D' || bytes[15] != 'R')
      return std::nullopt;
    return Dimensions{be32(16), be32(20)};
  }
  if (bytes.size() < 4 || bytes[0] != 0xff || bytes[1] != 0xd8)
    return std::nullopt;
  std::optional<Dimensions> dimensions;
  size_t at = 2;
  while (at < bytes.size()) {
    if (bytes[at++] != 0xff) return std::nullopt;
    while (at < bytes.size() && bytes[at] == 0xff) ++at;
    if (at == bytes.size()) return std::nullopt;
    const uint8_t marker = bytes[at++];
    // Standalone/restart markers and DNL are invalid in the header stream.
    if (marker == 0 || marker == 1 || (marker >= 0xd0 && marker <= 0xd9) ||
        marker == 0xdc || bytes.size() - at < 2)
      return std::nullopt;
    const size_t length = (size_t(bytes[at]) << 8) | bytes[at + 1];
    if (length < 2 || length > bytes.size() - at) return std::nullopt;
    const bool sof = marker >= 0xc0 && marker <= 0xcf && marker != 0xc4 &&
                     marker != 0xc8 && marker != 0xcc;
    if (sof) {
      if (dimensions || length < 8 || bytes[at + 7] == 0 ||
          length != size_t(8) + size_t(3) * bytes[at + 7])
        return std::nullopt;
      dimensions = Dimensions{(uint32_t(bytes[at + 5]) << 8) | bytes[at + 6],
                              (uint32_t(bytes[at + 3]) << 8) | bytes[at + 4]};
    }
    if (marker == 0xda) return dimensions;
    at += length;
  }
  return std::nullopt;
}
}  // namespace

std::optional<cv::Mat> DecodeEncodedImage(std::string_view text,
                                          std::optional<size_t> max_pixels) {
  if (!IsBase64(text)) return std::nullopt;
  const auto* data = reinterpret_cast<const unsigned char*>(text.data());
  const size_t length = tb64declen(data, text.size());
  if (length == 0) return std::nullopt;
  std::vector<uint8_t> bytes(length);
  if (tb64dec(data, text.size(), bytes.data()) != length) return std::nullopt;
  return DecodeImageBytes(bytes, max_pixels);
}

std::optional<cv::Mat> DecodeImageBytes(std::span<const uint8_t> bytes,
                                        std::optional<size_t> max_pixels,
                                        cv::Mat* storage) {
  if (bytes.empty() || bytes.size() > size_t(std::numeric_limits<int>::max()))
    return std::nullopt;
  std::optional<Dimensions> dimensions;
  if (max_pixels) {
    dimensions = ImageDimensions(bytes);
    if (!dimensions || !dimensions->width || !dimensions->height ||
        dimensions->width > uint32_t(std::numeric_limits<int>::max()) ||
        dimensions->height > uint32_t(std::numeric_limits<int>::max()) ||
        dimensions->width > *max_pixels / dimensions->height)
      return std::nullopt;
  }
  auto image =
      cv::imdecode(cv::Mat(1, static_cast<int>(bytes.size()), CV_8UC1,
                           const_cast<uint8_t*>(bytes.data())),
                   max_pixels ? cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION
                              : cv::IMREAD_COLOR,
                   storage);
  if (image.empty() || image.dims != 2 || image.rows <= 0 || image.cols <= 0 ||
      image.type() != CV_8UC3)
    return std::nullopt;
  if (dimensions && (uint32_t(image.cols) != dimensions->width ||
                     uint32_t(image.rows) != dimensions->height))
    return std::nullopt;
  return image;
}
}  // namespace vision_simple
