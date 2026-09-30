#include "ImageCodec.h"

#include <turbobase64/turbob64.h>

#include <algorithm>
#include <charconv>
#include <climits>
#include <cmath>
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
// All reads are bounded by the compressed input; textual headers use views.
bool Space(char c) noexcept {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f';
}
std::string_view Trim(std::string_view s) noexcept {
  while (!s.empty() && Space(s.front())) s.remove_prefix(1);
  while (!s.empty() && Space(s.back())) s.remove_suffix(1);
  return s;
}
std::optional<uint32_t> Positive(std::string_view s) {
  uint32_t n = 0;
  const auto result = std::from_chars(s.data(), s.data() + s.size(), n);
  if (result.ec != std::errc{} || result.ptr != s.data() + s.size() ||
      n == 0 || n > uint32_t(std::numeric_limits<int>::max()))
    return std::nullopt;
  return n;
}
struct HeaderText {
  std::string_view rest;
  std::optional<std::string_view> Line(bool accept_cr = false) {
    const auto end = accept_cr ? rest.find_first_of("\r\n") : rest.find('\n');
    if (end == std::string_view::npos) return std::nullopt;
    auto line = rest.substr(0, end);
    rest.remove_prefix(end + 1);
    return line;
  }
  std::optional<std::string_view> Token() {
    for (;;) {
      while (!rest.empty() && Space(rest.front())) rest.remove_prefix(1);
      if (rest.empty()) return std::nullopt;
      if (rest.front() != '#') break;
      if (!Line(true)) return std::nullopt;
    }
    size_t end = 0;
    while (end < rest.size() && !Space(rest[end])) ++end;
    // PNM ReadNumber consumes its first non-digit. A number-adjacent # must
    // not become a comment here: its following digits can be the next codec
    // dimension. Keep it in the token so checked numeric parsing rejects it.
    if (end == rest.size() || end == 0) return std::nullopt;
    auto token = rest.substr(0, end);
    rest.remove_prefix(end);
    return token;
  }
};
std::optional<Dimensions> PreparedDimensions(std::span<const uint8_t> bytes) {
  if (auto dimensions = ImageDimensions(bytes)) return dimensions;
  const auto le16 = [&](size_t at) {
    return uint32_t(bytes[at]) | (uint32_t(bytes[at + 1]) << 8);
  };
  const auto le32 = [&](size_t at) {
    return le16(at) | (le16(at + 2) << 16);
  };
  const auto be32 = [&](size_t at) {
    return (uint32_t(bytes[at]) << 24) | (uint32_t(bytes[at + 1]) << 16) |
           (uint32_t(bytes[at + 2]) << 8) | uint32_t(bytes[at + 3]);
  };
  if (bytes.size() >= 26 && bytes[0] == 'B' && bytes[1] == 'M') {
    const uint32_t header = le32(14);
    if (header == 12) return Dimensions{le16(18), le16(20)};
    if (header < 36 || header > bytes.size() - 14) return std::nullopt;
    const uint32_t width = le32(18), raw_height = le32(22);
    if (width > uint32_t(INT_MAX) || raw_height == 0x80000000u)
      return std::nullopt;
    return Dimensions{width, raw_height > uint32_t(INT_MAX)
                                 ? uint32_t(0) - raw_height : raw_height};
  }
  if (bytes.size() >= 32 && be32(0) == 0x59a66a95u)
    return Dimensions{be32(4), be32(8)};
  HeaderText text{std::string_view(reinterpret_cast<const char*>(bytes.data()),
                                  bytes.size())};
  if (bytes.size() >= 3 && bytes[0] == 'P' && Space(char(bytes[2]))) {
    const char kind = char(bytes[1]);
    text.rest.remove_prefix(3);
    if (kind >= '1' && kind <= '6') {
      auto w = text.Token(), h = text.Token();
      if (!w || !h) return std::nullopt;
      auto width = Positive(*w), height = Positive(*h);
      if (!width || !height) return std::nullopt;
      if (kind != '1' && kind != '4') {
        auto max = text.Token();
        auto value = max ? Positive(*max) : std::nullopt;
        if (!value || *value > 65535) return std::nullopt;
      }
      return Dimensions{*width, *height};
    }
    if (kind == 'F' || kind == 'f') {
      auto w = text.Token(), h = text.Token(), scale = text.Token();
      if (!w || !h || !scale) return std::nullopt;
      auto width = Positive(*w), height = Positive(*h);
      float value = 0;
      auto parsed = std::from_chars(scale->data(), scale->data() + scale->size(), value);
      if (!width || !height || parsed.ec != std::errc{} ||
          parsed.ptr != scale->data() + scale->size() || !std::isfinite(value) || value == 0)
        return std::nullopt;
      return Dimensions{*width, *height};
    }
    if (kind == '7') {
      std::optional<uint32_t> width, height, depth, max;
      while (auto line = text.Line(true)) {
        auto s = Trim(*line);
        if (s.empty() || s.front() == '#') continue;
        if (s == "ENDHDR") {
          if (!width || !height || !depth || !max || *depth > 4 || *max > 65535)
            return std::nullopt;
          return Dimensions{*width, *height};
        }
        const auto split = s.find_first_of(" \t");
        if (split == std::string_view::npos) return std::nullopt;
        auto key = s.substr(0, split), value = Trim(s.substr(split));
        if (key == "TUPLTYPE") {
          if (value.empty()) return std::nullopt;
          continue;
        }
        auto* field = key == "WIDTH" ? &width : key == "HEIGHT" ? &height
                    : key == "DEPTH" ? &depth : key == "MAXVAL" ? &max : nullptr;
        if (!field || *field) return std::nullopt;
        *field = Positive(value);
        if (!*field) return std::nullopt;
      }
    }
    return std::nullopt;
  }
  if (text.rest.starts_with("#?RADIANCE\n") || text.rest.starts_with("#?RGBE\n")) {
    if (!text.Line()) return std::nullopt;
    bool format = false, separator = false;
    while (auto line = text.Line()) {
      if (line->size() >= 128) return std::nullopt;
      if (line->empty()) { separator = true; break; }
      if (*line == "FORMAT=32-bit_rle_rgbe") format = true;
    }
    auto line = text.Line();
    if (!format || !separator || !line || line->size() >= 128 ||
        !line->starts_with("-Y"))
      return std::nullopt;
    auto resolution = line->substr(2);
    const auto number = [&]() -> std::optional<uint32_t> {
      while (!resolution.empty() && Space(resolution.front()))
        resolution.remove_prefix(1);
      if (!resolution.empty() && resolution.front() == '+')
        resolution.remove_prefix(1);
      uint32_t value = 0;
      const auto parsed = std::from_chars(resolution.data(),
          resolution.data() + resolution.size(), value);
      if (parsed.ec != std::errc{} || parsed.ptr == resolution.data() ||
          value == 0 || value > uint32_t(INT_MAX))
        return std::nullopt;
      resolution.remove_prefix(size_t(parsed.ptr - resolution.data()));
      return value;
    };
    // RGBE uses "-Y %d +X %d": whitespace can be tabs or absent, and each
    // positive dimension may have one leading '+'. Keep conversion checked.
    auto height = number();
    resolution = Trim(resolution);
    if (!height || !resolution.starts_with("+X")) return std::nullopt;
    resolution.remove_prefix(2);
    auto width = number();
    if (width && Trim(resolution).empty()) return Dimensions{*width, *height};
  }
  return std::nullopt;
}
std::optional<std::vector<uint8_t>> DecodeBase64(std::string_view text) {
  if (!IsBase64(text)) return std::nullopt;
  const auto* data = reinterpret_cast<const unsigned char*>(text.data());
  const size_t length = tb64declen(data, text.size());
  if (length == 0 || length > size_t(INT_MAX)) return std::nullopt;
  std::vector<uint8_t> bytes(length);
  if (tb64dec(data, text.size(), bytes.data()) != length) return std::nullopt;
  return bytes;
}
}  // namespace

std::optional<PreparedImage> PrepareEncodedImage(std::string_view text) {
  auto bytes = DecodeBase64(text);
  if (!bytes) return std::nullopt;
  auto dimensions = PreparedDimensions(*bytes);
  if (!dimensions || dimensions->width == 0 || dimensions->height == 0 ||
      dimensions->width > uint32_t(INT_MAX) || dimensions->height > uint32_t(INT_MAX) ||
      size_t(dimensions->width) > std::numeric_limits<size_t>::max() / dimensions->height)
    return std::nullopt;
  const size_t pixels = size_t(dimensions->width) * dimensions->height;
  return PreparedImage{std::move(*bytes), pixels};
}

std::optional<cv::Mat> DecodeEncodedImage(std::string_view text,
                                          std::optional<size_t> max_pixels) {
  auto bytes = DecodeBase64(text);
  if (!bytes) return std::nullopt;
  return DecodeImageBytes(*bytes, max_pixels);
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
