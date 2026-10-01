#pragma once

namespace vision_simple {
// Model construction only. The area filter uses the pre-unclip bounding box
// and rejects boxes whose area is less than or equal to min_box_area.
struct OCRDetectionOptions {
  int kernel_size = 2;
  int dilation_iterations = 3;
  int min_box_area = 64;

  constexpr bool IsValid() const noexcept {
    return kernel_size >= 1 && kernel_size <= 32 &&
           dilation_iterations >= 0 && dilation_iterations <= 8 &&
           min_box_area >= 0 && min_box_area <= 1048576;
  }
  constexpr bool operator==(const OCRDetectionOptions&) const noexcept = default;
};
}  // namespace vision_simple
