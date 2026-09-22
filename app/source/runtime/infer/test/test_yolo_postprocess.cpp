#include <array>
#include <bit>
#include <climits>
#include <iostream>
#include <limits>

#include "../private/InferYOLO.h"
#include "../private/YOLOMetadata.hpp"
#include "Util.hpp"

using namespace vision_simple;

int test_v11_class_aware_nms() {
  VisionHelper helper;
  LetterboxTransform transform;
  helper.Letterbox(cv::Mat(640, 640, CV_8UC3), {640, 640}, transform);
  YOLOFilter filter(YOLOVersion::kV11, {"first", "second"}, {1, 6, 4});
  const std::array<float, 24> output{
      100, 100, 100, 400, 100,  100,  100,   400,  80,    80,    80,    80,
      80,  80,  80,  80,  0.9f, 0.8f, 0.05f, 0.1f, 0.05f, 0.05f, 0.85f, 0.05f};
  auto result = filter(output, 0.5f, transform);
  TEST_ASSERT(result.has_value(), "v11 valid output accepted");
  TEST_ASSERT_EQ(
      result->results.size(), size_t{2},
      "cross-class boxes survive, same-class and low score suppressed");
  bool first = false, second = false;
  for (const auto& detection : result->results) {
    TEST_ASSERT_EQ(detection.bbox, cv::Rect(60, 60, 80, 80),
                   "NMS keeps original geometry");
    if (detection.class_id == 0) {
      first = true;
      TEST_ASSERT_EQ(detection.confidence, 0.9f,
                     "same-class highest confidence survives");
      TEST_ASSERT_EQ(detection.class_name, "first", "first class mapping");
    } else if (detection.class_id == 1) {
      second = true;
      TEST_ASSERT_EQ(detection.confidence, 0.85f,
                     "second class survives independently");
      TEST_ASSERT_EQ(detection.class_name, "second", "second class mapping");
    }
  }
  TEST_ASSERT(first && second, "both classes retained");
  auto boundary = filter(output, 0.9f, transform);
  TEST_ASSERT(boundary && boundary->results.empty(),
              "v11 preserves strict confidence threshold");
  TEST_PASS("v11 class-aware NMS, confidence and class mapping");
  return 0;
}

int test_v10_coordinates_without_second_nms() {
  VisionHelper helper;
  LetterboxTransform transform;
  helper.Letterbox(cv::Mat(720, 1280, CV_8UC3), {640, 640}, transform);
  TEST_ASSERT_EQ(transform.top, 140,
                 "720-pixel source resizes to 360 with 140 top padding");
  YOLOFilter filter(YOLOVersion::kV10, {"target"}, {1, 5, 6});
  const std::array<float, 30> output{
      100,  200, 300,  300, 0.9f, 0,   100,  200, 300,   300,
      0.5f, 0,   -20,  130, 50,   170, 0.8f, 0,   100,   200,
      100,  300, 0.9f, 0,   400,  200, 500,  300, 0.49f, 0};
  auto result = filter(output, 0.5f, transform);
  TEST_ASSERT(result.has_value(), "v10 end-to-end output accepted");
  TEST_ASSERT_EQ(result->results.size(), size_t{3},
                 "v10 retains overlapping detections without NMS");
  TEST_ASSERT_EQ(result->results[0].bbox, cv::Rect(200, 120, 400, 200),
                 "v10 actual-padding inverse geometry");
  TEST_ASSERT_EQ(result->results[1].bbox, cv::Rect(200, 120, 400, 200),
                 "confidence equality retained");
  TEST_ASSERT_EQ(result->results[2].bbox, cv::Rect(0, 0, 100, 60),
                 "v10 endpoint clipping");
  TEST_PASS("v10 inverse geometry and already-postprocessed NMS policy");
  return 0;
}

int test_malformed_shapes_and_lengths() {
  const LetterboxTransform transform{{640, 640}, {640, 640}, {640, 640}, 1,
                                     1,          0,          0};
  const std::array<float, 6> detection{10, 10, 20, 20, 0.9f, 0};
  for (const auto& shape : std::vector<std::vector<int64_t>>{
           {},
           {1, 6},
           {2, 1, 6},
           {1, 6, 1},
           {1, -1, 6},
           {1, std::numeric_limits<int64_t>::max(), 6}}) {
    YOLOFilter filter(YOLOVersion::kV10, {"target"}, shape);
    auto result = filter(detection, 0.5f, transform);
    TEST_ASSERT(!result, "v10 rejects unsupported shape");
    TEST_ASSERT(result.error().code == VisionSimpleErrorCode::kModelError,
                "v10 shape is model error");
  }
  YOLOFilter v10(YOLOVersion::kV10, {"target"}, {1, 1, 6});
  TEST_ASSERT(!v10(std::span<const float>(detection).first(5), 0.5f, transform),
              "v10 rejects truncated span");
  std::array<float, 7> extra{};
  TEST_ASSERT(!v10(extra, 0.5f, transform), "v10 rejects trailing data");
  for (const auto& shape : std::vector<std::vector<int64_t>>{
           {}, {1, 5}, {2, 5, 1}, {1, 4, 1}, {1, 6, 1}, {1, 5, -1}}) {
    YOLOFilter filter(YOLOVersion::kV11, {"target"}, shape);
    auto result = filter(detection, 0.5f, transform);
    TEST_ASSERT(!result, "v11 rejects malformed shape or class count");
    TEST_ASSERT(result.error().code == VisionSimpleErrorCode::kModelError,
                "v11 shape is model error");
  }
  YOLOFilter v11(YOLOVersion::kV11, {"target"}, {1, 5, 1});
  TEST_ASSERT(!v11(std::span<const float>(detection).first(4), 0.5f, transform),
              "v11 rejects truncated span");
  TEST_ASSERT(!v11(detection, 0.5f, transform), "v11 rejects trailing data");
  YOLOFilter empty(YOLOVersion::kV10, {"target"}, {1, 0, 6});
  auto no_detections = empty({}, 0.5f, transform);
  TEST_ASSERT(no_detections && no_detections->results.empty(),
              "zero detections are successful empty output");
  TEST_PASS("rank, shape, class dimensions and exact span lengths");
  return 0;
}

int test_malformed_class_and_nonfinite_values() {
  const LetterboxTransform transform{{640, 640}, {640, 640}, {640, 640}, 1,
                                     1,          0,          0};
  YOLOFilter v10(YOLOVersion::kV10, {"target"}, {1, 1, 6});
  for (float class_id :
       {-1.0f, 1.0f, 0.5f, 1e30f, std::bit_cast<float>(0x7fc00000u),
        std::bit_cast<float>(0x7f800000u)}) {
    const std::array<float, 6> output{10, 10, 20, 20, 0.1f, class_id};
    auto result = v10(output, 0.5f, transform);
    TEST_ASSERT(!result,
                "invalid class rejected even below confidence threshold");
    TEST_ASSERT(result.error().code == VisionSimpleErrorCode::kModelError,
                "invalid class returns model error");
  }
  YOLOFilter v11(YOLOVersion::kV11, {"target"}, {1, 5, 1});
  std::array<float, 5> output{10, 10, 20, 20,
                              std::bit_cast<float>(0x7fc00000u)};
  TEST_ASSERT(!v11(output, 0.5f, transform),
              "non-finite class confidence rejected");
  output = {std::bit_cast<float>(0x7f800000u), 10, 20, 20, 0.9f};
  TEST_ASSERT(!v11(output, 0.5f, transform), "non-finite coordinates rejected");
  TEST_PASS("class indices and non-finite output rejected safely");
  return 0;
}

int test_v26_layouts_and_metadata() {
  const LetterboxTransform transform{{640, 640}, {640, 640}, {640, 640}, 1,
                                     1,          0,          0};
  bool end_to_end = false;
  TEST_ASSERT(detail::ParseYOLO26Export(
                  "{'nms': False, 'imgsz': (640, 640), 'half': True}",
                  "True", "", end_to_end) && end_to_end,
              "Python exporter metadata selects end-to-end decoding");
  YOLOFilter e2e(YOLOVersion::kV26, {"first", "second"}, {1, 6, 6},
                 YOLODetectionLayout::kEndToEnd);
  const std::array<float, 36> rows{
      60, 60, 140, 140, 0.9f, 0, 60, 60, 140, 140, 0.8f, 0,
      60, 60, 140, 140, 0.85f, 1, 60, 60, 140, 140, 0.5f, 0,
      60, 60, 140, 140, 0.1f, 0, 60, 60, 140, 140, 0.1f, 0};
  auto result = e2e(rows, 0.5f, transform);
  TEST_ASSERT(result && result->results.size() == 4,
              "YOLO26 end-to-end keeps overlaps and threshold equality");
  TEST_ASSERT_EQ(result->results[0].bbox, cv::Rect(60, 60, 80, 80),
                 "YOLO26 end-to-end boxes are xyxy");
  TEST_ASSERT_EQ(result->results[2].class_name, "second",
                 "YOLO26 end-to-end class mapping");
  TEST_ASSERT(detail::ParseYOLO26Export(
                  "{\"nms\": null, \"end2end\": false}", "false", "false",
                  end_to_end) && !end_to_end,
              "JSON export metadata selects raw decoding");
  YOLOFilter raw(YOLOVersion::kV26, {"first", "second"}, {1, 6, 6},
                 YOLODetectionLayout::kRaw);
  const std::array<float, 36> channels{
      100, 100, 100, 400, 400, 400, 100, 100, 100, 400, 400, 400,
      80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
      0.9f, 0.8f, 0.05f, 0.5f, 0.1f, 0.1f,
      0.05f, 0.05f, 0.85f, 0.1f, 0.1f, 0.1f};
  result = raw(channels, 0.5f, transform);
  TEST_ASSERT(result && result->results.size() == 2,
              "YOLO26 raw applies class-aware NMS and strict threshold");
  TEST_ASSERT_EQ(result->results[0].bbox, cv::Rect(60, 60, 80, 80),
                 "YOLO26 raw boxes are center xywh");
  YOLOFilter unspecified(YOLOVersion::kV26, {"first", "second"}, {1, 6, 6});
  TEST_ASSERT(!unspecified(channels, 0.5f, transform),
              "ambiguous shape alone never selects YOLO26 mode");
  auto malformed = rows;
  malformed[5] = 0.5f;
  TEST_ASSERT(!e2e(malformed, 0.5f, transform),
              "YOLO26 rejects fractional class indexes");
  malformed = channels;
  malformed[0] = std::numeric_limits<float>::max();
  malformed[12] = std::numeric_limits<float>::max();
  TEST_ASSERT(!raw(malformed, 0.5f, transform),
              "YOLO26 rejects coordinate arithmetic overflow");
  malformed = channels;
  malformed[35] = std::numeric_limits<float>::quiet_NaN();
  TEST_ASSERT(!raw(malformed, 0.5f, transform),
              "YOLO26 rejects nonfinite scores below threshold");
  for (const auto& shape : std::vector<std::vector<int64_t>>{
           {2, 6, 6}, {1, -1, 6}, {1, 6, -1},
           {1, std::numeric_limits<int64_t>::max(), 6}}) {
    YOLOFilter invalid_shape(YOLOVersion::kV26, {"first", "second"}, shape,
                             YOLODetectionLayout::kEndToEnd);
    TEST_ASSERT(!invalid_shape(rows, 0.5f, transform),
                "YOLO26 rejects dynamic, batched and oversized outputs");
  }
  TEST_ASSERT(!e2e(std::span(rows).first(35), 0.5f, transform),
              "YOLO26 rejects truncated tensors");
  const std::array<std::array<std::string_view, 3>, 17> invalid{{
      {"", "True", ""},
      {"{}", "True", ""},
      {"{'nms': False}", "", ""},
      {"{'nms': False}", "1", ""},
      {"{'nms': True}", "True", ""},
      {"{'nms': False}", "False", ""},
      {"{'nms': None}", "True", ""},
      {"{'nms': False}", "True", "True"},
      {"{'nms': False}", "True", "None"},
      {"{'nms': False}", "True", "garbage"},
      {"{'nms': False, 'nms': False}", "True", ""},
      {"{'nms': False, 'end2end': False}", "True", ""},
      {"{'nms': False, 'end2end': True, 'end2end': True}", "True", ""},
      {"{'nms': 'False'}", "True", ""},
      {"{'nms': False, 'half': garbage}", "True", ""},
      {"{'nms': False, 'other': {'broken', 2}}", "True", ""},
      {"{'nms': False} trailing", "True", ""},
  }};
  for (const auto& metadata : invalid)
    TEST_ASSERT(!detail::ParseYOLO26Export(metadata[0], metadata[1], metadata[2],
                                          end_to_end),
                "malformed, missing, embedded-NMS or conflicting export rejected");
  TEST_PASS("YOLO26 explicit layouts, NMS policy and metadata validation");
  return 0;
}

int main() {
  int failures = 0;
  failures += test_v11_class_aware_nms();
  failures += test_v10_coordinates_without_second_nms();
  failures += test_malformed_shapes_and_lengths();
  failures += test_malformed_class_and_nonfinite_values();
  failures += test_v26_layouts_and_metadata();
  return failures ? 1 : 0;
}
