#pragma once
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <span>

namespace vision_simple {
class Cvt {
  Cvt() = delete;

 public:
  static void cvt(std::span<const float> from,
                  Ort::Float16_t* output) noexcept {
    for (size_t i = 0; i < from.size(); ++i)
      output[i] = Ort::Float16_t(from[i]);
  }

  static void cvt(std::span<const Ort::Float16_t> from,
                  float* output) noexcept {
    for (size_t i = 0; i < from.size(); ++i) output[i] = from[i].ToFloat();
  }
};

struct LetterboxTransform {
  cv::Size original_size{}, target_size{}, resized_size{};
  double gain_x = 0, gain_y = 0;
  int left = 0, top = 0;
};

class VisionHelper {
  cv::Mat letterbox_resized_image_, letterbox_dst_image_;
  std::vector<cv::Mat> channels_{3};

 public:
  VisionHelper() = default;

  cv::Mat& Letterbox(const cv::Mat& src, const cv::Size& target_size,
                     LetterboxTransform& transform,
                     const cv::Scalar& color = cv::Scalar(0, 0, 0)) {
    transform = {};
    if (src.empty() || src.dims != 2 || src.cols <= 0 || src.rows <= 0 ||
        target_size.width <= 0 || target_size.height <= 0) {
      letterbox_dst_image_.release();
      return letterbox_dst_image_;
    }
    const double scale =
        std::min(static_cast<double>(target_size.width) / src.cols,
                 static_cast<double>(target_size.height) / src.rows);
    const int new_width = static_cast<int>(src.cols * scale);
    const int new_height = static_cast<int>(src.rows * scale);
    if (new_width <= 0 || new_height <= 0) {
      letterbox_dst_image_.release();
      return letterbox_dst_image_;
    }
    resize(src, letterbox_resized_image_, {new_width, new_height});
    letterbox_dst_image_.create(target_size, src.type());
    letterbox_dst_image_.setTo(color);
    const int top = (target_size.height - new_height) / 2;
    const int left = (target_size.width - new_width) / 2;
    letterbox_resized_image_.copyTo(
        letterbox_dst_image_(cv::Rect(left, top, new_width, new_height)));
    transform = {src.size(),
                 target_size,
                 {new_width, new_height},
                 static_cast<double>(new_width) / src.cols,
                 static_cast<double>(new_height) / src.rows,
                 left,
                 top};
    return letterbox_dst_image_;
  }

  template <typename T>
  void HWC2CHW_BGR2RGB(const cv::Mat& from, cv::Mat& to) {
    constexpr int channel_mapper[3] = {2, 1, 0};
    size_t width = from.cols, height = from.rows;
    split(from, channels_);
    const size_t num_pixels = width * height;
    auto dst_base_ptr = to.ptr<T>();
    for (int c = 0; c < 3; ++c) {
      const auto src = channels_[channel_mapper[c]].data;
      auto dst = dst_base_ptr + num_pixels * c;
      std::memcpy(dst, src, num_pixels * sizeof(T));
    }
  }

  static cv::Rect ScaleCoords(const LetterboxTransform& transform,
                              const cv::Vec4f& xyxy) noexcept {
    // Bit checks remain reliable under the release fast floating-point mode.
    const auto finite = [](double value) {
      return (std::bit_cast<uint64_t>(value) & UINT64_C(0x7ff0000000000000)) !=
             UINT64_C(0x7ff0000000000000);
    };
    if (transform.original_size.width <= 0 ||
        transform.original_size.height <= 0 || !finite(transform.gain_x) ||
        !finite(transform.gain_y) || transform.gain_x <= 0 ||
        transform.gain_y <= 0)
      return {};
    for (float value : xyxy.val)
      if ((std::bit_cast<uint32_t>(value) & 0x7f800000u) == 0x7f800000u)
        return {};
    if (xyxy[2] <= xyxy[0] || xyxy[3] <= xyxy[1]) return {};
    const auto restore = [](float value, int pad, double gain, int limit) {
      return static_cast<int>(
          std::round(std::clamp((static_cast<double>(value) - pad) / gain, 0.0,
                                static_cast<double>(limit))));
    };
    const int left = restore(xyxy[0], transform.left, transform.gain_x,
                             transform.original_size.width);
    const int top = restore(xyxy[1], transform.top, transform.gain_y,
                            transform.original_size.height);
    const int right = restore(xyxy[2], transform.left, transform.gain_x,
                              transform.original_size.width);
    const int bottom = restore(xyxy[3], transform.top, transform.gain_y,
                               transform.original_size.height);
    if (right <= left || bottom <= top) return {};
    return {left, top, right - left, bottom - top};
  }

  double ComputeIOU(const cv::Rect& rect1, const cv::Rect& rect2) noexcept {
    if (rect1.width <= 0 || rect1.height <= 0 || rect2.width <= 0 ||
        rect2.height <= 0)
      return 0;
    const int64_t left = std::max<int64_t>(rect1.x, rect2.x);
    const int64_t top = std::max<int64_t>(rect1.y, rect2.y);
    const int64_t right = std::min(int64_t{rect1.x} + rect1.width,
                                   int64_t{rect2.x} + rect2.width);
    const int64_t bottom = std::min(int64_t{rect1.y} + rect1.height,
                                    int64_t{rect2.y} + rect2.height);
    const double intersection =
        static_cast<double>(std::max<int64_t>(0, right - left)) *
        static_cast<double>(std::max<int64_t>(0, bottom - top));
    const double area1 = static_cast<double>(rect1.width) * rect1.height;
    const double area2 = static_cast<double>(rect2.width) * rect2.height;
    return intersection / (area1 + area2 - intersection);
  }

  // 根据IOU阈值进行过滤
  std::vector<cv::Rect> FilterByIOU(std::vector<cv::Rect>& boxes,
                                    double iou_threshold) {
    std::vector<cv::Rect> filteredBoxes;

    for (size_t i = 0; i < boxes.size(); i++) {
      bool keep = true;
      for (size_t j = 0; j < filteredBoxes.size(); j++) {
        // 如果当前框和已过滤框的IOU大于阈值，丢弃当前框
        if (ComputeIOU(boxes[i], filteredBoxes[j]) > iou_threshold) {
          keep = false;
          break;
        }
      }
      if (keep) {
        filteredBoxes.push_back(boxes[i]);
      }
    }

    // 将过滤后的框替换回原始框列表
    return filteredBoxes;
  }

  static cv::Rect ScaleRect(const cv::Rect& rect, double scale_width,
                            double scale_height) noexcept {
    // 计算新的宽度和高度
    int new_width = static_cast<int>(rect.width * scale_width);
    int new_height = static_cast<int>(rect.height * scale_height);

    // 计算原矩形的中心
    int center_x = rect.x + rect.width / 2;
    int center_y = rect.y + rect.height / 2;

    // 计算新的x和y坐标，使得缩放后的矩形中心不变
    int new_x = center_x - new_width / 2;
    int new_y = center_y - new_height / 2;

    // 返回新的矩形
    return {new_x, new_y, new_width, new_height};
  }
};
}  // namespace vision_simple
