#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>

#include "InferPipeline.h"
#include "Infer.h"
#include "InferYOLOTask.h"
#include "Util.hpp"

using namespace vision_simple;
namespace fs = std::filesystem;
namespace {
bool Near(float a, float b, float tolerance = .025f) {
  return std::abs(a - b) <= tolerance;
}
bool Same(const YOLOTaskFrameResult& a, const YOLOTaskFrameResult& b) {
  if (a.index() != b.index()) return false;
  return std::visit(
      [&](const auto& left) {
        using Frame = std::decay_t<decltype(left)>;
        const auto& right = std::get<Frame>(b);
        if (left.results.size() != right.results.size()) return false;
        for (size_t i = 0; i < left.results.size(); ++i) {
          const auto& x = left.results[i];
          const auto& y = right.results[i];
          if (x.class_id != y.class_id || !Near(x.confidence, y.confidence))
            return false;
          if constexpr (std::is_same_v<Frame, YOLOOBBFrame>) {
            if (!Near(x.angle, y.angle)) return false;
            for (size_t j = 0; j < 4; ++j)
              if (!Near(x.corners[j].x, y.corners[j].x) ||
                  !Near(x.corners[j].y, y.corners[j].y))
                return false;
          } else {
            if (x.bbox != y.bbox) return false;
            if constexpr (std::is_same_v<Frame, YOLOSegmentationFrame>) {
              if (x.mask.type() != CV_8UC1 || x.mask.size() != y.mask.size() ||
                  cv::countNonZero(x.mask != y.mask))
                return false;
            } else {
              if (x.keypoints.size() != y.keypoints.size()) return false;
              for (size_t j = 0; j < x.keypoints.size(); ++j)
                if (!Near(x.keypoints[j].x, y.keypoints[j].x) ||
                    !Near(x.keypoints[j].y, y.keypoints[j].y) ||
                    !Near(x.keypoints[j].confidence, y.keypoints[j].confidence))
                  return false;
            }
          }
        }
        return true;
      },
      a);
}
int Exercise(InferContext& context, const fs::path& assets, YOLOTask task,
             const char* stem) {
  auto model = InferYOLOTask::Create(
      context, (assets / (std::string(stem) + ".onnx")).string(), task);
  if (!model) std::cerr << model.error().message << '\n';
  TEST_ASSERT(model, "create FP32 task model");
  auto bytes = ReadAll((assets / (std::string(stem) + "_fp16.onnx")).string());
  TEST_ASSERT(bytes, "read FP16 task model");
  auto half = InferYOLOTask::Create(context, bytes->span(), task);
  TEST_ASSERT(half, "create FP16 task model from memory");
  const cv::Mat square(64, 64, CV_8UC3, cv::Scalar::all(0));
  auto reference = (*model)->Run(square, .5f);
  auto half_result = (*half)->Run(square, .5f);
  TEST_ASSERT(reference && half_result && Same(*reference, *half_result),
              "FP16/FP32 decode agrees");
  if (task == YOLOTask::kSegmentation) {
    const auto& items = std::get<YOLOSegmentationFrame>(*reference).results;
    TEST_ASSERT(items.size() == 2,
                "segmentation removes duplicate and preserves second class");
    TEST_ASSERT(
        items[0].class_id == 1 && items[0].bbox == cv::Rect(40, 40, 16, 16) &&
            items[1].class_id == 0 && items[1].bbox == cv::Rect(4, 4, 24, 24),
        "segmentation restores sorted boxes");
    for (const auto& item : items) {
      TEST_ASSERT(
          item.mask.type() == CV_8UC1 && item.mask.size() == item.bbox.size(),
          "mask is binary owning bbox ROI");
      for (int y = 0; y < item.mask.rows; ++y)
        for (int x = 0; x < item.mask.cols; ++x) {
          const bool foreground = item.class_id == 1 ? x < 8 : x >= 12;
          TEST_ASSERT(
              item.mask.at<uint8_t>(y, x) == (foreground ? 255 : 0),
              "coefficient selects correct prototype and logit boundary");
        }
    }
    const cv::Mat saved = items[0].mask.clone();
    auto again =
        (*model)->Run(cv::Mat(57, 101, CV_8UC3, cv::Scalar::all(255)), .5f);
    TEST_ASSERT(again && !cv::countNonZero(items[0].mask != saved),
                "later workspace reuse does not mutate retained masks");
  } else if (task == YOLOTask::kPose) {
    const auto& items = std::get<YOLOPoseFrame>(*reference).results;
    TEST_ASSERT(items.size() == 2 && items[1].keypoints.size() == 2,
                "pose suppresses duplicate while retaining keypoints");
    TEST_ASSERT(Near(items[1].keypoints[0].x, 10) &&
                    Near(items[1].keypoints[0].y, 12) &&
                    Near(items[1].keypoints[1].x, 22) &&
                    Near(items[1].keypoints[1].y, 24) &&
                    Near(items[1].keypoints[1].confidence, .2f),
                "keypoint confidence below detection threshold is preserved");
    auto wide =
        (*model)->Run(cv::Mat(57, 101, CV_8UC3, cv::Scalar::all(0)), .5f);
    TEST_ASSERT(wide, "pose accepts non-square original");
    const auto& point = std::get<YOLOPoseFrame>(*wide).results[1].keypoints[0];
    TEST_ASSERT(
        Near(point.x, 10.f * 101 / 64) && Near(point.y, (12.f - 14) * 57 / 36),
        "pose inverse uses independently rounded x/y resize gains without "
        "clipping");
  } else {
    const auto& items = std::get<YOLOOBBFrame>(*reference).results;
    TEST_ASSERT(items.size() == 2 && Near(items[0].angle, -.785398f) &&
                    Near(items[1].angle, .785398f),
                "crossed boxes survive rotated IoU while parallel duplicate is "
                "suppressed");
    TEST_ASSERT(
        Near(items[1].corners[1].x - items[1].corners[0].x, 28.28427f) &&
            Near(items[1].corners[1].y - items[1].corners[0].y, 28.28427f),
        "ordered corners preserve oriented width edge");
    auto wide =
        (*model)->Run(cv::Mat(16, 128, CV_8UC3, cv::Scalar::all(0)), .5f);
    TEST_ASSERT(wide, "OBB supports severe letterbox padding");
    const auto& corners = std::get<YOLOOBBFrame>(*wide).results[1].corners;
    TEST_ASSERT(
        corners[0].y < 0 && corners[2].y > 16,
        "OBB corners remain outside image rather than clipping geometry");
  }
  auto pipeline = InferPipeline::Create({2, 2, 16});
  TEST_ASSERT(pipeline, "create bounded pipeline");
  std::vector<cv::Mat> images;
  for (const auto size :
       {cv::Size(64, 64), cv::Size(101, 57), cv::Size(128, 64),
        cv::Size(32, 96), cv::Size(64, 64)})
    images.emplace_back(size, CV_8UC3, cv::Scalar::all(0));
  const auto batch = (*pipeline)->Run(**model, images, .5f);
  TEST_ASSERT(batch && batch->size() == images.size(),
              "pipeline completes every task image");
  for (size_t i = 0; i < images.size(); ++i) {
    const auto direct = (*model)->Run(images[i], .5f);
    TEST_ASSERT(direct && Same(*direct, (*batch)[i]),
                "pipeline output geometry/masks and order match direct path");
  }
  const std::array<cv::Mat, 2> invalid{square, cv::Mat()};
  const auto failed = (*pipeline)->Run(**model, invalid, .5f);
  TEST_ASSERT(!failed && failed.error().image_index == 1,
              "invalid staged input reports correct index");
  TEST_ASSERT(!(*model)->Run(square, std::numeric_limits<float>::quiet_NaN()),
              "non-finite threshold rejected");
  const auto recovered = (*pipeline)->Run(**model, images, .5f);
  TEST_ASSERT(recovered && Same((*recovered)[0], *reference),
              "failed request releases reusable workspace and capacity");
  auto mismatch = InferYOLOTask::Create(
      context, (assets / (std::string(stem) + ".onnx")).string(),
      task == YOLOTask::kPose ? YOLOTask::kOBB : YOLOTask::kPose);
  TEST_ASSERT(!mismatch, "task metadata or output layout mismatch rejected");
  return 0;
}
int MetadataAndRecovery(InferContext& context, const fs::path& assets) {
  const auto path = [&](const char* name) {
    return (assets / (std::string("yolo_") + name + ".onnx")).string();
  };
  const std::vector<std::string> expected{
      "worker's glove", "say 'hi' \\ \xc3\xa9 \xf0\x9f\x98\x80"};
  auto named =
      InferYOLOTask::Create(context, path("names_obb"), YOLOTask::kOBB);
  TEST_ASSERT(named && (*named)->class_names() == expected,
              "quoted keys, apostrophes, escaped quotes/backslashes and "
              "Unicode preserve labels; quoted flag text is not NMS");
  auto legacy_bytes = ReadAll(path("names_detect"));
  TEST_ASSERT(legacy_bytes, "read legacy names fixture");
  auto legacy =
      InferYOLO::Create(context, legacy_bytes->span(), YOLOVersion::kV11);
  TEST_ASSERT(legacy && (*legacy)->class_names() == expected,
              "legacy detector uses identical lossless names parsing");
  for (const char* name :
       {"names_invalid", "names_unclosed", "names_escape_invalid"})
    TEST_ASSERT(
        !InferYOLOTask::Create(context, path(name), YOLOTask::kOBB),
        "invalid names reject instead of truncating or mislabeling classes");
  for (const char* name : {"pose_nms_python", "pose_nms_json", "pose_end2end",
                           "pose_nms_direct", "pose_end2end_args"})
    TEST_ASSERT(
        !InferYOLOTask::Create(context, path(name), YOLOTask::kPose),
        "exported NMS metadata rejects shape collision with raw pose channels");
  const cv::Mat black(64, 64, CV_8UC3, cv::Scalar::all(0));
  const cv::Mat white(64, 64, CV_8UC3, cv::Scalar::all(255));
  auto d2 = InferYOLOTask::Create(context, path("pose_d2"), YOLOTask::kPose);
  TEST_ASSERT(d2, "create two-dimensional keypoints model");
  auto points = (*d2)->Run(black, .5f);
  TEST_ASSERT(points, "decode two-dimensional keypoints");
  const auto& items = std::get<YOLOPoseFrame>(*points).results;
  TEST_ASSERT(items.size() == 2 && items[1].keypoints.size() == 2 &&
                  Near(items[1].keypoints[0].x, 10) &&
                  Near(items[1].keypoints[0].y, 12) &&
                  Near(items[1].keypoints[1].x, 22) &&
                  Near(items[1].keypoints[1].y, 24),
              "D2 channels preserve xy coordinates");
  for (const auto& item : items)
    for (const auto& point : item.keypoints)
      TEST_ASSERT(point.confidence == 1.f,
                  "D2 keypoints have implicit confidence one");
  auto nan = InferYOLOTask::Create(context, path("pose_nan"), YOLOTask::kPose);
  TEST_ASSERT(nan, "create input-dependent non-finite output model");
  auto before = (*nan)->Run(black, .5f);
  TEST_ASSERT(before, "finite output succeeds before runtime NaN");
  TEST_ASSERT(!(*nan)->Run(white, .5f),
              "actual non-finite ONNX output is rejected");
  auto after = (*nan)->Run(black, .5f);
  TEST_ASSERT(after && Same(*before, *after),
              "same model recovers after runtime NaN");
  auto pipeline = InferPipeline::Create({2, 2, 16});
  TEST_ASSERT(pipeline, "create non-finite recovery pipeline");
  const std::array<cv::Mat, 2> mixed{black, white};
  auto failed = (*pipeline)->Run(**nan, mixed, .5f);
  TEST_ASSERT(!failed && failed.error().image_index == 1,
              "pipeline attributes actual NaN output to failing image");
  const std::array<cv::Mat, 2> valid{black, black};
  auto recovered = (*pipeline)->Run(**nan, valid, .5f);
  TEST_ASSERT(recovered && recovered->size() == 2 &&
                  Same((*recovered)[0], *before) &&
                  Same((*recovered)[1], *before),
              "pipeline recovers capacity and workspaces after NaN output");
  return 0;
}
}  // namespace
int main(int argc, char** argv) {
  fs::path root = fs::current_path();
  if (argc == 3 && std::string_view(argv[1]) == "--project-root")
    root = fs::absolute(argv[2]);
  else if (argc != 1) {
    std::cerr << "usage: test_yolo_tasks [--project-root ROOT]\n";
    return 1;
  } else
    while (!fs::exists(root / "app/assets/test") && root != root.parent_path())
      root = root.parent_path();
  auto context =
      InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kCPU);
  TEST_ASSERT(context, "create CPU ORT context");
  const auto assets = root / "app/assets/test/reliability";
  if (Exercise(**context, assets, YOLOTask::kSegmentation, "yolo_seg"))
    return 1;
  if (Exercise(**context, assets, YOLOTask::kPose, "yolo_pose")) return 1;
  if (Exercise(**context, assets, YOLOTask::kOBB, "yolo_obb")) return 1;
  if (MetadataAndRecovery(**context, assets)) return 1;
  return 0;
}
