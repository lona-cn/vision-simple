#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Test-only corpus v1. No fonts, random inputs, model output, or external images
// participate in labeling. Each rendered word is one region; ignore policy: none.
namespace ocr_text_corpus {
inline constexpr size_t kCaseCount = 8;
inline constexpr size_t kMaxRegions = 4096;
inline constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;
inline constexpr int kLineType = cv::LINE_8;
struct Frame {
  std::string id;
  std::string cohort;
  cv::Mat image;
  std::vector<cv::Rect> words;
};

inline Frame Render(size_t index) {
  if (index >= kCaseCount) throw std::out_of_range("corpus case");
  Frame frame{"case-" + std::to_string(index),
              index < 3 ? "ordinary" : index < 6 ? "dense" : "negative",
              cv::Mat(384, 640, CV_8UC3, cv::Scalar(255, 255, 255)), {}};
  if (index == 6) return frame;  // Truly blank negative.
  if (index == 7) {
    // Non-text background: a large low-contrast gradient and geometric shapes.
    for (int y = 0; y < frame.image.rows; ++y)
      frame.image.row(y).setTo(cv::Scalar(220 + y / 16, 220 + y / 16, 220 + y / 16));
    cv::circle(frame.image, {150, 160}, 80, {180, 180, 180}, -1, kLineType);
    cv::rectangle(frame.image, {360, 90, 180, 200}, {195, 195, 195}, -1, kLineType);
    return frame;
  }
  constexpr std::array<const char*, 6> words{
      "ALPHA", "BRAVO", "DELTA", "RIVER", "STONE", "NORTH"};
  const bool dense = index >= 3;
  const int columns = dense ? 4 : 2;
  const int rows = dense ? 6 : 3;
  const double scale = dense ? .48 + .07 * (index - 3) : .85 + .15 * index;
  const int thickness = dense ? 1 : 2;
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      const auto word = words[(row * columns + column + index) % words.size()];
      const cv::Point origin{20 + column * (dense ? 155 : 310),
                             (dense ? 45 : 75) + row * (dense ? 55 : 110)};
      // Label from the actual isolated rasterized ink, before copying it onto
      // the scene. This is independent of the detector and its unclip geometry.
      cv::Mat ink(frame.image.size(), CV_8UC1, cv::Scalar(0));
      cv::putText(ink, word, origin, kFont, scale, cv::Scalar(255), thickness, kLineType);
      const cv::Rect bounds = cv::boundingRect(ink);
      if (bounds.empty() || bounds.x <= 0 || bounds.y <= 0 ||
          bounds.br().x >= frame.image.cols || bounds.br().y >= frame.image.rows)
        throw std::logic_error("corpus ink must be nonempty and unclipped");
      for (const auto& existing : frame.words)
        if (!(existing & bounds).empty()) throw std::logic_error("overlapping labels");
      frame.words.push_back(bounds);
      frame.image.setTo(cv::Scalar(0, 0, 0), ink);
    }
  }
  return frame;
}

inline double IoU(const cv::Rect& a, const cv::Rect& b) {
  if (a.width <= 0 || a.height <= 0 || b.width <= 0 || b.height <= 0) return 0;
  const int64_t left = std::max<int64_t>(a.x, b.x);
  const int64_t top = std::max<int64_t>(a.y, b.y);
  const int64_t right = std::min<int64_t>(int64_t(a.x) + a.width, int64_t(b.x) + b.width);
  const int64_t bottom = std::min<int64_t>(int64_t(a.y) + a.height, int64_t(b.y) + b.height);
  const int64_t intersection = std::max<int64_t>(0, right - left) * std::max<int64_t>(0, bottom - top);
  const int64_t total = int64_t(a.width) * a.height + int64_t(b.width) * b.height - intersection;
  return double(intersection) / double(total);
}

struct Counts {
  size_t images = 0, gt = 0, tp = 0, fn = 0, fp = 0, empty_text = 0;
  void Add(const Counts& other) {
    images += other.images; gt += other.gt; tp += other.tp;
    fn += other.fn; fp += other.fp; empty_text += other.empty_text;
  }
};

inline Counts Score(const std::vector<cv::Rect>& gt, const std::vector<cv::Rect>& predictions) {
  if (gt.size() > kMaxRegions || predictions.size() > kMaxRegions)
    throw std::length_error("bounded matching graph");
  std::vector<std::vector<size_t>> edges(predictions.size());
  for (size_t p = 0; p < predictions.size(); ++p)
    for (size_t g = 0; g < gt.size(); ++g)
      if (IoU(predictions[p], gt[g]) >= .5) edges[p].push_back(g);
  // Augmenting paths, not greedy highest-IoU: maximize one-to-one cardinality.
  std::vector<size_t> owner(gt.size(), predictions.size());
  std::vector<bool> seen(gt.size());
  const auto augment = [&](auto&& self, size_t p) -> bool {
    for (const size_t g : edges[p]) {
      if (seen[g]) continue;
      seen[g] = true;
      if (owner[g] == predictions.size() || self(self, owner[g])) {
        owner[g] = p;
        return true;
      }
    }
    return false;
  };
  size_t matched = 0;
  for (size_t p = 0; p < predictions.size(); ++p) {
    std::fill(seen.begin(), seen.end(), false);
    if (augment(augment, p)) ++matched;
  }
  return {1, gt.size(), matched, gt.size() - matched, predictions.size() - matched, 0};
}

inline std::string Metrics(const Counts& counts) {
  std::ostringstream out;
  out << "{\"images\":" << counts.images << ",\"gt\":" << counts.gt
      << ",\"tp\":" << counts.tp << ",\"fn\":" << counts.fn << ",\"fp\":" << counts.fp
      << ",\"empty_text\":" << counts.empty_text << ",\"recall\":";
  if (counts.gt) out << double(counts.tp) / counts.gt; else out << "null";
  out << ",\"precision\":";
  if (counts.tp + counts.fp) out << double(counts.tp) / (counts.tp + counts.fp); else out << "null";
  out << ",\"fp_per_image\":";
  if (counts.images) out << double(counts.fp) / counts.images; else out << "null";
  return out.str() + "}";
}

// Works against both immutable #61 SDK headers and the new SDK. The hex UTF-8
// representation preserves every text byte, including controls and empty text;
// confidence_bits preserves the exact float rather than a rounded decimal.
template <class Result>
std::string SerializeNormalizedResult(const Result& result) {
  std::ostringstream out;
  out << '[';
  bool first = true;
  for (const auto& item : result.results) {
    if (!first) out << ',';
    first = false;
    out << "{\"box\":[" << item.rect.x << ',' << item.rect.y << ','
        << item.rect.width << ',' << item.rect.height << "],\"confidence_bits\":"
        << std::bit_cast<uint32_t>(item.confidence) << ",\"text_utf8_hex\":\"";
    for (const unsigned char byte : item.line)
      out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    out << std::dec << "\"}";
  }
  return out.str() + "]";
}
}  // namespace ocr_text_corpus
