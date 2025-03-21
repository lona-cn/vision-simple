﻿#pragma once
#include <span>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>

namespace vision_simple {
template <typename From, typename To>
  requires std::is_default_constructible_v<From> &&
           std::is_default_constructible_v<To>
class DataConverter {
  uint8_t step_;
  std::function<void(const From* from_ptr, To* to_ptr)> block_cvt_;
  std::vector<std::remove_const_t<From>> temp_input_;
  std::vector<std::remove_const_t<To>> temp_output_;

 public:
  using CVT_FUNC = decltype(block_cvt_);

  DataConverter(uint8_t step_, CVT_FUNC batch_cvt)
      : step_(step_),
        block_cvt_(std::move(batch_cvt)),
        temp_input_(step_),
        temp_output_(step_) {}

  void operator()(std::span<const From> from, To* output) noexcept {
    const auto size = from.size();
    auto* from_ptr = from.data();
    // 主体并行处理部分
    // #pragma omp parallel for schedule(guided)
    for (int64_t i = 0; i < static_cast<int64_t>(size / step_) * step_;
         i += step_) {
      block_cvt_(from_ptr + i, output + i);
    }
    // 边界处理部分，单线程完成，避免多线程争用资源
    const int remaining = size % step_;
    if (remaining > 0) {
      const int64_t offset = size - remaining;
      // 拷贝剩余元素
      // std::copy_n(from_ptr + offset, remaining, temp_input_.data());
      std::memcpy(temp_input_.data(), from_ptr + offset,
                  remaining * sizeof(From));
      // 转换剩余元素
      block_cvt_(temp_input_.data(), temp_output_.data());
      // 拷贝结果
      // std::copy_n(temp_output_.data(), remaining,
      // reinterpret_cast<To*>(output + offset));
      std::memcpy(output + offset, temp_output_.data(), remaining * sizeof(To));
    }
  }
};

class Cvt {
  Cvt() = delete;
  static inline DataConverter<float, Ort::Float16_t> fp32tofp16{
      16, [](const float* from, Ort::Float16_t* to) {
#if defined(__x86_64__)
        _mm_prefetch(reinterpret_cast<char const*>(from) + 64, _MM_HINT_T0);
        for (uint8_t i = 0, offset = 0; i < 2; ++i, offset = i * 8) {
          __m256 float32_vec = _mm256_loadu_ps(from + offset);  // 加载8个float
          __m128i float16_vec = _mm256_cvtps_ph(
              float32_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(to + offset),
                           float16_vec);  // 存储8个Float16
        }
#endif
      }};

  static inline DataConverter<Ort::Float16_t, float> fp16tofp32{
      32, [](const Ort::Float16_t* from, float* to) {
#if defined(__x86_64__)
        _mm_prefetch(reinterpret_cast<char const*>(from) + 64, _MM_HINT_T0);
        for (uint8_t i = 0, offset = 0; i < 4; ++i, offset = i * 8) {
          // 加载8个Float16
          __m128i float16_vec =
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(from + offset));
          __m256 float32_vec = _mm256_cvtph_ps(float16_vec);  // 转换为Float32
          _mm256_storeu_ps(to + offset, float32_vec);         // 存储结果
        }
#endif
      }};

  static inline DataConverter<uint8_t, Ort::Float16_t> u8tofp16_normalized{
      16, [scale_factor = 1.0f / 255.0f](const uint8_t* from,
                                                  Ort::Float16_t* to) {
#if defined(__x86_64__)
        auto scale = _mm256_set1_ps(scale_factor);
        // TODO: fixit
        //  加载16个uint8
        __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(from));
        // 扩展为uint16
        __m256i v0_lo = _mm256_cvtepu8_epi16(v0);
        // 转换为浮点数
        __m256 f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_unpacklo_epi16(
                                      v0_lo, _mm256_setzero_si256())),
                                  scale);
        __m256 f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_unpackhi_epi16(
                                      v0_lo, _mm256_setzero_si256())),
                                  scale);
        // 转换为fp16 (F16C指令)
        __m128i h0 = _mm256_cvtps_ph(f0, 0);
        __m128i h1 = _mm256_cvtps_ph(f1, 0);
        // 存储结果
        _mm_storeu_si128(reinterpret_cast<__m128i*>(to), h0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(to + 8), h1);
#endif
      }};
  static inline DataConverter<uint8_t, float> u8tofp32_normalized{
      32,
      [scale_factor = 1.0f / 255.0f](const uint8_t* from, float* to) {
#if defined(__x86_64__)
        auto scale = _mm256_set1_ps(1.0f / 255.0f);
        // TODO: fixit
        //  加载32字节数据
        __m256i v0 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(from));  // 低16字节
        __m256i v1 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(from + 16));  // 高16字节

        // 将uint8扩展为uint16
        __m256i v0_lo = _mm256_unpacklo_epi8(v0, _mm256_setzero_si256());
        __m256i v0_hi = _mm256_unpackhi_epi8(v0, _mm256_setzero_si256());
        __m256i v1_lo = _mm256_unpacklo_epi8(v1, _mm256_setzero_si256());
        __m256i v1_hi = _mm256_unpackhi_epi8(v1, _mm256_setzero_si256());

        // 转换为浮点数
        __m256 f0_lo = _mm256_cvtepi32_ps(
            _mm256_unpacklo_epi16(v0_lo, _mm256_setzero_si256()));
        __m256 f0_hi = _mm256_cvtepi32_ps(
            _mm256_unpackhi_epi16(v0_hi, _mm256_setzero_si256()));
        __m256 f1_lo = _mm256_cvtepi32_ps(
            _mm256_unpacklo_epi16(v1_lo, _mm256_setzero_si256()));
        __m256 f1_hi = _mm256_cvtepi32_ps(
            _mm256_unpackhi_epi16(v1_hi, _mm256_setzero_si256()));

        // 缩放到[0.0, 1.0]
        f0_lo = _mm256_mul_ps(f0_lo, scale);
        f0_hi = _mm256_mul_ps(f0_hi, scale);
        f1_lo = _mm256_mul_ps(f1_lo, scale);
        f1_hi = _mm256_mul_ps(f1_hi, scale);

        // 存储结果
        _mm256_storeu_ps(to, f0_lo);
        _mm256_storeu_ps(to + 8, f0_hi);
        _mm256_storeu_ps(to + 16, f1_lo);
        _mm256_storeu_ps(to + 24, f1_hi);
#endif
      }};

 public:
  // uint8_t->fp16
  static void cvt(std::span<const uint8_t> from,
                  Ort::Float16_t* output) noexcept {
    // TODO
  }

  static void cvt(std::span<const uint8_t> from, float* output) noexcept {
    // TODO
  }

  // fp32->fp16
  static void cvt(std::span<const float> from,
                  Ort::Float16_t* output) noexcept {
    return fp32tofp16(from, output);
  }

  // fp16->fp32
  static void cvt(std::span<const Ort::Float16_t> from,
                  float* output) noexcept {
    return fp16tofp32(from, output);
  }
};

class VisionHelper {
  cv::Mat letterbox_resized_image_, letterbox_dst_image_;
  std::vector<cv::Mat> channels_{3};

 public:
  VisionHelper() = default;

  cv::Mat& Letterbox(const cv::Mat& src, const cv::Size& target_size,
                     const cv::Scalar& color = cv::Scalar(0, 0, 0)) noexcept {
    const float scale = std::min(
        static_cast<float>(target_size.width) / static_cast<float>(src.cols),
        static_cast<float>(target_size.height) / static_cast<float>(src.rows));
    const int new_width =
        static_cast<int>(static_cast<float>(src.cols) * scale);
    const int new_height =
        static_cast<int>(static_cast<float>(src.rows) * scale);
    resize(src, letterbox_resized_image_, {new_width, new_height});
    if (letterbox_dst_image_.rows != target_size.height ||
        letterbox_dst_image_.cols != target_size.width) {
      letterbox_dst_image_ =
          cv::Mat::zeros(target_size.height, target_size.width, src.type());
    }
    letterbox_dst_image_.setTo(color);
    int top = (target_size.height - new_height) / 2;
    int left = (target_size.width - new_width) / 2;
    letterbox_resized_image_.copyTo(
        letterbox_dst_image_(cv::Rect(left, top, letterbox_resized_image_.cols,
                                      letterbox_resized_image_.rows)));

    return letterbox_dst_image_;
  }

  template <typename T>
  void HWC2CHW_BGR2RGB(const cv::Mat& from, cv::Mat& to) noexcept {
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

  static cv::Rect ScaleCoords(const cv::Size& image_shape, cv::Rect coords,
                              const cv::Size& image_original_shape,
                              bool clip) noexcept {
    cv::Rect result;
    float gain = std::min(static_cast<float>(image_shape.height) /
                              static_cast<float>(image_original_shape.height),
                          static_cast<float>(image_shape.width) /
                              static_cast<float>(image_original_shape.width));

    int padX = static_cast<int>(std::round(
        (image_shape.width - image_original_shape.width * gain) / 2.0f));
    int padY = static_cast<int>(std::round(
        (image_shape.height - image_original_shape.height * gain) / 2.0f));

    result.x = static_cast<int>(std::round((coords.x - padX) / gain));
    result.y = static_cast<int>(std::round((coords.y - padY) / gain));
    result.width = static_cast<int>(std::round(coords.width / gain));
    result.height = static_cast<int>(std::round(coords.height / gain));

    if (clip) {
      result.x = std::clamp(result.x, 0, image_original_shape.width);
      result.y = std::clamp(result.y, 0, image_original_shape.height);
      result.width =
          std::clamp(result.width, 0, image_original_shape.width - result.x);
      result.height =
          std::clamp(result.height, 0, image_original_shape.height - result.y);
    }
    return result;
  }

  double ComputeIOU(const cv::Rect& rect1, const cv::Rect& rect2) {
    // 计算交集
    cv::Rect intersection = rect1 & rect2;

    // 计算并集
    cv::Rect union_rect = rect1 | rect2;

    // 交集面积
    double intersectionArea = intersection.area();

    // 并集面积
    double unionArea = union_rect.area();

    // 计算IOU
    return intersectionArea / unionArea;
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
