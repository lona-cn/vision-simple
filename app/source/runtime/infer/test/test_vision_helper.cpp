#include <climits>
#include <cmath>
#include <iostream>

#include "Util.hpp"
#include "VisionHelper.hpp"

using namespace vision_simple;

int test_letterbox_markers() {
  VisionHelper helper;
  LetterboxTransform transform;
  cv::Mat wide(100, 200, CV_8UC3, cv::Scalar(64, 128, 192));
  wide(cv::Rect(40, 20, 40, 20)).setTo(cv::Scalar(0, 0, 255));
  auto& output = helper.Letterbox(wide, {100, 101}, transform);
  TEST_ASSERT_EQ(output.size(), cv::Size(100, 101), "wide output dimensions");
  TEST_ASSERT_EQ(output.at<cv::Vec3b>(24, 50), cv::Vec3b(0, 0, 0),
                 "top padding ends before content");
  TEST_ASSERT_EQ(output.at<cv::Vec3b>(25, 50), cv::Vec3b(64, 128, 192),
                 "content starts at actual top");
  TEST_ASSERT_EQ(output.at<cv::Vec3b>(74, 50), cv::Vec3b(64, 128, 192),
                 "last content row");
  TEST_ASSERT_EQ(output.at<cv::Vec3b>(75, 50), cv::Vec3b(0, 0, 0),
                 "odd padding remainder stays below");
  TEST_ASSERT_EQ(output.at<cv::Vec3b>(35, 20), cv::Vec3b(0, 0, 255),
                 "known marker top-left pixel");
  TEST_ASSERT_EQ(output.at<cv::Vec3b>(44, 39), cv::Vec3b(0, 0, 255),
                 "known marker bottom-right pixel");
  TEST_ASSERT_EQ(output.at<cv::Vec3b>(35, 19), cv::Vec3b(64, 128, 192),
                 "pixel outside marker");
  TEST_ASSERT_EQ(VisionHelper::ScaleCoords(transform, {20, 35, 40, 45}),
                 cv::Rect(40, 20, 40, 20), "pixel marker restores exactly");

  cv::Mat tall(200, 100, CV_8UC3, cv::Scalar(64, 128, 192));
  tall(cv::Rect(20, 40, 20, 40)).setTo(cv::Scalar(0, 255, 0));
  auto& portrait = helper.Letterbox(tall, {101, 100}, transform);
  TEST_ASSERT_EQ(portrait.at<cv::Vec3b>(50, 24), cv::Vec3b(0, 0, 0),
                 "portrait left padding");
  TEST_ASSERT_EQ(portrait.at<cv::Vec3b>(50, 25), cv::Vec3b(64, 128, 192),
                 "portrait first content column");
  TEST_ASSERT_EQ(portrait.at<cv::Vec3b>(20, 35), cv::Vec3b(0, 255, 0),
                 "portrait known marker");
  TEST_ASSERT_EQ(portrait.at<cv::Vec3b>(39, 44), cv::Vec3b(0, 255, 0),
                 "portrait marker final pixel");
  TEST_ASSERT_EQ(VisionHelper::ScaleCoords(transform, {35, 20, 45, 40}),
                 cv::Rect(20, 40, 20, 40), "portrait marker restores exactly");
  TEST_PASS("actual wide and portrait pixels agree with inverse geometry");
  return 0;
}

int test_axis_scales_and_rounding() {
  VisionHelper helper;
  LetterboxTransform transform;
  cv::Mat source(3, 7, CV_8UC3, cv::Scalar(20, 40, 60));
  auto& output = helper.Letterbox(source, {10, 10}, transform);
  TEST_ASSERT_EQ(output.at<cv::Vec3b>(2, 5), cv::Vec3b(0, 0, 0),
                 "integer resize top padding");
  TEST_ASSERT_EQ(output.at<cv::Vec3b>(3, 5), cv::Vec3b(20, 40, 60),
                 "integer resize first row");
  TEST_ASSERT_EQ(output.at<cv::Vec3b>(6, 5), cv::Vec3b(20, 40, 60),
                 "integer resize last row");
  TEST_ASSERT_EQ(output.at<cv::Vec3b>(7, 5), cv::Vec3b(0, 0, 0),
                 "integer resize bottom padding");
  TEST_ASSERT_EQ(VisionHelper::ScaleCoords(transform, {0, 3, 10, 7}),
                 cv::Rect(0, 0, 7, 3), "separate actual x and y gains");
  TEST_ASSERT_EQ(VisionHelper::ScaleCoords(transform, {0, 3, 10, 5}),
                 cv::Rect(0, 0, 7, 2),
                 "axis gain uses round rather than truncation");

  cv::Mat square(10, 10, CV_8UC3, cv::Scalar(1, 2, 3));
  helper.Letterbox(square, square.size(), transform);
  TEST_ASSERT_EQ(VisionHelper::ScaleCoords(transform, {0.5f, 0.5f, 2.5f, 3.5f}),
                 cv::Rect(1, 1, 2, 3), "round endpoints consistently");
  TEST_ASSERT_EQ(VisionHelper::ScaleCoords(transform, {-5, -5, 5, 5}),
                 cv::Rect(0, 0, 5, 5), "clip endpoints not old widths");
  TEST_ASSERT_EQ(VisionHelper::ScaleCoords(transform, {8, 8, 20, 20}),
                 cv::Rect(8, 8, 2, 2), "right and bottom clipping");
  TEST_ASSERT_EQ(VisionHelper::ScaleCoords(transform, {0, 0, 10, 10}),
                 cv::Rect(0, 0, 10, 10), "edge aligned box");
  TEST_ASSERT_EQ(VisionHelper::ScaleCoords(transform, {-20, 2, -1, 8}),
                 cv::Rect(), "fully outside box is empty");
  TEST_ASSERT_EQ(VisionHelper::ScaleCoords(transform, {3, 2, 3, 8}), cv::Rect(),
                 "zero width box is empty");
  TEST_ASSERT_EQ(VisionHelper::ScaleCoords(transform, {4, 2, 3, 8}), cv::Rect(),
                 "reversed endpoints are empty");
  TEST_ASSERT_EQ(VisionHelper::ScaleCoords(transform, {2.1f, 2, 2.2f, 8}),
                 cv::Rect(), "rounded zero width box is empty");
  TEST_PASS("actual axis scales, clipping and endpoint rounding");
  return 0;
}

int test_letterbox_rejection_and_recovery() {
  VisionHelper helper;
  LetterboxTransform transform;
  cv::Mat valid(10, 10, CV_8UC3, cv::Scalar(1, 2, 3));
  helper.Letterbox(valid, {10, 10}, transform);
  TEST_ASSERT(helper.Letterbox(cv::Mat{}, {10, 10}, transform).empty(),
              "empty source rejected");
  TEST_ASSERT_EQ(VisionHelper::ScaleCoords(transform, {0, 0, 5, 5}), cv::Rect(),
                 "failed transform cannot restore boxes");
  TEST_ASSERT(helper.Letterbox(valid, {0, 10}, transform).empty(),
              "zero target rejected");
  cv::Mat extreme(1, 10000, CV_8UC3);
  TEST_ASSERT(helper.Letterbox(extreme, {10, 10}, transform).empty(),
              "rounded zero resize rejected");
  auto& recovered = helper.Letterbox(valid, {10, 10}, transform);
  TEST_ASSERT_EQ(recovered.at<cv::Vec3b>(5, 5), cv::Vec3b(1, 2, 3),
                 "valid request after rejection");
  TEST_ASSERT_EQ(VisionHelper::ScaleCoords(transform, {0, 0, 10, 10}),
                 cv::Rect(0, 0, 10, 10), "recovered geometry");
  TEST_PASS("invalid dimensions return empty and allow reuse");
  return 0;
}

int test_iou() {
  VisionHelper helper;
  TEST_ASSERT_FLOAT_EQ(helper.ComputeIOU({0, 0, 10, 10}, {5, 5, 10, 10}),
                       25.0 / 175.0, 1e-12, "IoU uses set union");
  TEST_ASSERT_EQ(helper.ComputeIOU({0, 0, 10, 10}, {0, 0, 10, 10}), 1.0,
                 "identical boxes");
  TEST_ASSERT_EQ(helper.ComputeIOU({0, 0, 10, 10}, {10, 10, 10, 10}), 0.0,
                 "touching boxes");
  TEST_ASSERT_EQ(helper.ComputeIOU({0, 0, 0, 10}, {0, 0, 10, 10}), 0.0,
                 "zero area");
  TEST_ASSERT_EQ(helper.ComputeIOU({0, 0, -1, 10}, {0, 0, 10, 10}), 0.0,
                 "negative area");
  TEST_ASSERT_FLOAT_EQ(
      helper.ComputeIOU({0, 0, 100000, 100000}, {50000, 50000, 100000, 100000}),
      25.0 / 175.0, 1e-12, "areas exceed 32 bits");
  const cv::Rect beyond_int(INT_MAX - 10, INT_MAX - 10, 100000, 100000);
  TEST_ASSERT_EQ(helper.ComputeIOU(beyond_int, beyond_int), 1.0,
                 "endpoints exceed 32 bits");
  TEST_PASS("IoU exact union, degeneracy and wide arithmetic");
  return 0;
}

int test_iou_filter() {
  std::vector<cv::Rect> boxes{{0, 0, 10, 10}, {1, 1, 10, 10}, {30, 30, 10, 10}};
  VisionHelper helper;
  const auto filtered = helper.FilterByIOU(boxes, 0.5);
  TEST_ASSERT_EQ(filtered.size(), size_t{2}, "overlap filter count");
  TEST_ASSERT_EQ(filtered[0], boxes[0], "first box retained");
  TEST_ASSERT_EQ(filtered[1], boxes[2], "nonoverlapping box retained");
  TEST_PASS("IoU filtering retains distinct boxes");
  return 0;
}

int main() {
  int failures = 0;
  failures += test_letterbox_markers();
  failures += test_axis_scales_and_rounding();
  failures += test_letterbox_rejection_and_recovery();
  failures += test_iou();
  failures += test_iou_filter();
  return failures ? 1 : 0;
}
