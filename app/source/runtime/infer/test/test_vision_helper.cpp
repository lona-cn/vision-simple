#include <cmath>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "Util.hpp"
#include "VisionHelper.hpp"

using namespace vision_simple;

// Letterbox tests
int test_letterbox_square() {
  VisionHelper helper;
  cv::Mat src(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));
  cv::Mat& result = helper.Letterbox(src, cv::Size(100, 100));
  TEST_ASSERT_EQ(result.rows, 100, "letterbox square rows");
  TEST_ASSERT_EQ(result.cols, 100, "letterbox square cols");
  TEST_PASS("letterbox square (100x100 -> 100x100)");
  return 0;
}

int test_letterbox_wide() {
  VisionHelper helper;
  cv::Mat src(100, 200, CV_8UC3, cv::Scalar(128, 128, 128));
  cv::Mat& result = helper.Letterbox(src, cv::Size(100, 100));
  TEST_ASSERT_EQ(result.rows, 100, "lb wide rows");
  TEST_ASSERT_EQ(result.cols, 100, "lb wide cols");
  cv::Vec3b top = result.at<cv::Vec3b>(0, 50);
  TEST_ASSERT(top == cv::Vec3b(0, 0, 0), "lb wide top padding black");
  TEST_PASS("letterbox wide (200x100 -> 100x100)");
  return 0;
}

int test_letterbox_tall() {
  VisionHelper helper;
  cv::Mat src(200, 100, CV_8UC3, cv::Scalar(128, 128, 128));
  cv::Mat& result = helper.Letterbox(src, cv::Size(100, 100));
  TEST_ASSERT_EQ(result.rows, 100, "lb tall rows");
  TEST_ASSERT_EQ(result.cols, 100, "lb tall cols");
  cv::Vec3b left = result.at<cv::Vec3b>(50, 0);
  TEST_ASSERT(left == cv::Vec3b(0, 0, 0), "lb tall left padding black");
  TEST_PASS("letterbox tall (100x200 -> 100x100)");
  return 0;
}

int test_letterbox_channels() {
  VisionHelper helper;
  cv::Mat src(50, 50, CV_8UC3, cv::Scalar(64, 128, 192));
  cv::Mat& result = helper.Letterbox(src, cv::Size(50, 50));
  TEST_ASSERT_EQ(result.channels(), 3, "lb 3-channel");
  cv::Vec3b mid = result.at<cv::Vec3b>(25, 25);
  TEST_ASSERT_EQ(mid[0], (uint8_t)64, "lb B");
  TEST_ASSERT_EQ(mid[1], (uint8_t)128, "lb G");
  TEST_ASSERT_EQ(mid[2], (uint8_t)192, "lb R");
  TEST_PASS("letterbox channel preservation");
  return 0;
}

// IOU tests
int test_iou_perfect() {
  double iou = VisionHelper::ComputeIOU(cv::Rect(0,0,10,10), cv::Rect(0,0,10,10));
  TEST_ASSERT_FLOAT_EQ(iou, 1.0, 0.001, "IOU perfect");
  TEST_PASS("IOU perfect overlap = 1.0");
  return 0;
}

int test_iou_none() {
  double iou = VisionHelper::ComputeIOU(cv::Rect(0,0,10,10), cv::Rect(20,20,10,10));
  TEST_ASSERT_FLOAT_EQ(iou, 0.0, 0.001, "IOU none");
  TEST_PASS("IOU no overlap = 0.0");
  return 0;
}

int test_iou_partial() {
  double iou = VisionHelper::ComputeIOU(cv::Rect(0,0,10,10), cv::Rect(5,5,10,10));
  TEST_ASSERT_FLOAT_EQ(iou, 25.0/175.0, 0.01, "IOU partial");
  TEST_PASS("IOU partial overlap = 25/175");
  return 0;
}

// NMS
int test_nms_keep_all() {
  std::vector<cv::Rect> boxes = {cv::Rect(0,0,10,10), cv::Rect(20,20,10,10), cv::Rect(40,40,10,10)};
  VisionHelper helper;
  auto result = helper.FilterByIOU(boxes, 1.0);
  TEST_ASSERT_EQ(result.size(), (size_t)3, "NMS keep all");
  TEST_PASS("NMS thresh=1.0 keeps all");
  return 0;
}

int test_nms_filter() {
  std::vector<cv::Rect> boxes = {cv::Rect(0,0,10,10), cv::Rect(1,1,10,10), cv::Rect(30,30,10,10)};
  VisionHelper helper;
  auto result = helper.FilterByIOU(boxes, 0.5);
  TEST_ASSERT_EQ(result.size(), (size_t)2, "NMS filter");
  TEST_PASS("NMS thresh=0.5 filters overlapping");
  return 0;
}

int main() {
  int failures = 0;
  std::cout << "=== VisionHelper Unit Tests ===" << std::endl;
  std::cout << "-- Letterbox --" << std::endl;
  failures += test_letterbox_square();
  failures += test_letterbox_wide();
  failures += test_letterbox_tall();
  failures += test_letterbox_channels();
  std::cout << "-- ComputeIOU --" << std::endl;
  failures += test_iou_perfect();
  failures += test_iou_none();
  failures += test_iou_partial();
  std::cout << "-- FilterByIOU --" << std::endl;
  failures += test_nms_keep_all();
  failures += test_nms_filter();
  std::cout << (failures ? "\n*** FAILED ***" : "\n*** ALL PASSED ***") << std::endl;
  return failures ? 1 : 0;
}
