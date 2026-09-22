#include <atomic>
#include <barrier>
#include <bit>
#include <filesystem>
#include <iostream>
#include <thread>

#include "Infer.h"
#include "Util.hpp"

using namespace vision_simple;
namespace fs = std::filesystem;

namespace {
bool Same(const YOLOFrameResult& a, const YOLOFrameResult& b) {
  if (a.results.size() != b.results.size()) return false;
  for (size_t i = 0; i < a.results.size(); ++i) {
    const auto& x = a.results[i];
    const auto& y = b.results[i];
    if (x.class_id != y.class_id || x.bbox != y.bbox ||
        std::abs(x.confidence - y.confidence) > 1e-5f)
      return false;
  }
  return true;
}

bool Same(const OCRFrameResult& a, const OCRFrameResult& b) {
  if (a.results.size() != b.results.size()) return false;
  for (size_t i = 0; i < a.results.size(); ++i) {
    const auto& x = a.results[i];
    const auto& y = b.results[i];
    if (x.line != y.line || x.rect != y.rect ||
        std::abs(x.confidence - y.confidence) > 1e-5f)
      return false;
  }
  return true;
}

template <typename Model>
int CheckInputs(Model& model) {
  cv::Mat storage(96, 192, CV_8UC3, cv::Scalar::all(0));
  cv::putText(storage, "TEST", {12, 62}, cv::FONT_HERSHEY_SIMPLEX, 1.2,
              cv::Scalar::all(255), 2);
  const auto roi = storage(cv::Rect{8, 8, 160, 80});
  TEST_ASSERT(!roi.isContinuous(), "fixture must exercise a strided ROI");
  const auto reference = model.Run(roi.clone(), .5f);
  TEST_ASSERT(reference, "valid CPU inference succeeds");
  const int dimensions[] = {2, 3, 4};
  const std::vector<cv::Mat> bad_images{
      cv::Mat{}, cv::Mat(32, 32, CV_8UC1), cv::Mat(32, 32, CV_8UC4),
      cv::Mat(32, 32, CV_32FC3), cv::Mat(3, dimensions, CV_8UC3)};
  for (const auto& image : bad_images) {
    const auto rejected = model.Run(image, .5f);
    TEST_ASSERT(!rejected && rejected.error().code ==
                                 VisionSimpleErrorCode::kParameterError,
                "unsupported image returns parameter error");
    const auto recovered = model.Run(roi, .5f);
    TEST_ASSERT(
        recovered && Same(*reference, *recovered),
        "same instance recovers with strided ROI matching contiguous pixels");
  }
  const float bad_confidence[]{-.1f, 1.1f, std::bit_cast<float>(0x7fc00001u),
                               std::bit_cast<float>(0x7f800000u),
                               std::bit_cast<float>(0xff800000u)};
  for (const auto threshold : bad_confidence) {
    const auto rejected = model.Run(roi, threshold);
    TEST_ASSERT(!rejected && rejected.error().code ==
                                 VisionSimpleErrorCode::kParameterError,
                "non-finite or out-of-range confidence rejected under release "
                "fast-math");
    const auto recovered = model.Run(roi, .5f);
    TEST_ASSERT(recovered && Same(*reference, *recovered),
                "confidence error leaves instance reusable");
  }
  TEST_ASSERT(model.Run(roi, 0.f), "zero confidence is accepted");
  TEST_ASSERT(model.Run(roi, 1.f), "unit confidence is accepted");
  return 0;
}

template <typename Model>
int CheckConcurrentRecovery(Model& model) {
  const cv::Mat black(32, 32, CV_8UC3, cv::Scalar::all(0));
  const cv::Mat white(32, 32, CV_8UC3, cv::Scalar::all(255));
  const auto reference = model.Run(black, .5f);
  TEST_ASSERT(reference, "concurrent recovery baseline succeeds");
  std::atomic<bool> correct{true};
  std::barrier ready(4);
  std::vector<std::jthread> threads;
  for (int worker = 0; worker < 4; ++worker) {
    threads.emplace_back([&, worker] {
      ready.arrive_and_wait();
      for (int iteration = 0; iteration < 8; ++iteration) {
        const bool fail = (iteration + worker) % 2 == 0;
        const auto result = model.Run(fail ? white : black, .5f);
        if (fail ? (result ||
                    result.error().code != VisionSimpleErrorCode::kRuntimeError)
                 : (!result || !Same(*reference, *result))) {
          correct.store(false, std::memory_order_relaxed);
        }
      }
    });
  }
  threads.clear();
  TEST_ASSERT(
      correct.load(std::memory_order_relaxed),
      "concurrent calls preserve their pixels and recover from ORT errors");
  return 0;
}

int CheckRuntimeRecovery(InferContext& context, const fs::path& assets) {
  auto yolo = InferYOLO::Create(context,
                                (assets / "yolo_runtime_failure.onnx").string(),
                                YOLOVersion::kV10);
  TEST_ASSERT(yolo, "load genuine input-dependent YOLO ORT failure fixture");
  cv::Mat black(32, 32, CV_8UC3, cv::Scalar::all(0));
  cv::Mat white(32, 32, CV_8UC3, cv::Scalar::all(255));
  for (int repeat = 0; repeat < 2; ++repeat) {
    const auto failed = (*yolo)->Run(white, .5f);
    TEST_ASSERT(
        !failed && failed.error().code == VisionSimpleErrorCode::kRuntimeError,
        "input-dependent Gather raises a recoverable ORT Run error");
    const auto recovered = (*yolo)->Run(black, .5f);
    TEST_ASSERT(recovered && recovered->results.size() == 1 &&
                    recovered->results[0].bbox == cv::Rect(4, 4, 16, 16),
                "YOLO same-session binding recovers after ORT Run error");
  }
  const auto extreme = (*yolo)->Run(cv::Mat(1, 10000, CV_8UC3), .5f);
  TEST_ASSERT(!extreme && extreme.error().code ==
                              VisionSimpleErrorCode::kParameterError,
              "rounded-zero Letterbox dimension is a recoverable input error");
  TEST_ASSERT((*yolo)->Run(black, .5f),
              "Letterbox error leaves session reusable");
  TEST_ASSERT(CheckConcurrentRecovery(**yolo) == 0,
              "YOLO serializes full Run for direct clients");

  auto rec_data = ReadAll((assets / "ocr_rec_runtime_failure.onnx").string());
  TEST_ASSERT(rec_data, "recognition failure fixture exists");
  for (const auto* detector :
       {"ocr_det_runtime_failure.onnx", "ocr_det_box.onnx"}) {
    auto det_data = ReadAll((assets / detector).string());
    TEST_ASSERT(det_data, "detection fixture exists");
    auto ocr = InferOCR::Create(context, {{0, "A"}}, det_data->span(),
                                rec_data->span(), OCRModelType::kPPOCRv4);
    TEST_ASSERT(ocr, "create real OCR failure/recovery sessions");
    for (int repeat = 0; repeat < 2; ++repeat) {
      const auto failed = (*ocr)->Run(white, .5f);
      TEST_ASSERT(!failed && failed.error().code ==
                                 VisionSimpleErrorCode::kRuntimeError,
                  "OCR detection/recognition ORT Run error is recoverable");
      const auto recovered = (*ocr)->Run(black, .5f);
      TEST_ASSERT(recovered,
                  "same OCR instance succeeds after ORT Run failure");
      if (std::string_view(detector) == "ocr_det_box.onnx") {
        TEST_ASSERT(
            recovered->results.size() == 1 &&
                recovered->results[0].line == "A" &&
                recovered->results[0].rect == cv::Rect(0, 0, 32, 32),
            "recovered recognition returns expected text and in-image crop");
      } else {
        TEST_ASSERT(recovered->results.empty(),
                    "blank detection is a successful empty result");
      }
    }
    TEST_ASSERT(CheckConcurrentRecovery(**ocr) == 0,
                "OCR serializes detection and all recognition crops");
  }
  return 0;
}

int CheckYOLO26(InferContext& context, const fs::path& assets) {
  const auto path = [&](const std::string& name) {
    return (assets / "reliability" / ("yolo26_detect_" + name + ".onnx")).string();
  };
  for (const char* name : {"missing_task", "wrong_task", "missing_args",
                           "empty_args", "missing_mode", "invalid_mode",
                           "embedded_nms", "conflicting_mode", "raw_conflict"}) {
    const auto rejected = InferYOLO::Create(context, path(name), YOLOVersion::kV26);
    TEST_ASSERT(!rejected && rejected.error().code == VisionSimpleErrorCode::kModelError,
                "ambiguous detection layout requires valid consistent export metadata");
  }
  for (const char* mode : {"raw", "e2e"}) {
    for (const char* precision : {"", "_fp16"}) {
      auto model = InferYOLO::Create(context, path(std::string(mode) + precision),
                                     YOLOVersion::kV26);
      TEST_ASSERT(model && (*model)->version() == YOLOVersion::kV26,
                  "load explicit YOLO26 mode without masquerading as an old version");
      auto result = (*model)->Run(cv::Mat(64, 64, CV_8UC3, cv::Scalar::all(0)), .5f);
      const size_t count = std::string_view(mode) == "raw" ? 1 : 2;
      TEST_ASSERT(result && result->results.size() == count,
                  "metadata disambiguates identical [1,6,6] shapes and controls NMS");
      for (const auto& detection : result->results) {
        TEST_ASSERT(detection.class_id == 0 && detection.bbox == cv::Rect(4, 4, 24, 24),
                    "explicit layout preserves box coordinate semantics");
      }
      auto empty = (*model)->Run(cv::Mat(64, 64, CV_8UC3, cv::Scalar::all(0)), 1.f);
      TEST_ASSERT(empty && empty->results.empty(), "empty YOLO26 detections are successful");
    }
  }
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  fs::path root = fs::current_path();
  if (argc == 3 && std::string_view(argv[1]) == "--project-root") {
    root = fs::absolute(argv[2]);
  } else if (argc != 1) {
    std::cerr << "usage: test_infer_inputs [--project-root ROOT]\n";
    return 1;
  } else {
    while (!fs::exists(root / "app/assets/test") && root != root.parent_path())
      root = root.parent_path();
  }
  const auto assets = root / "app/assets/test";
  TEST_ASSERT(fs::exists(assets / "hd2-yolo11n-fp32.onnx"),
              "required model fixtures exist");
  auto context =
      InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kCPU);
  TEST_ASSERT(context, "create actual CPU context");
  TEST_ASSERT(
      !InferContext::Create(static_cast<InferFramework>(255), InferEP::kCPU),
      "unknown framework fails without terminating");
  const auto missing = InferYOLO::Create(
      **context, (assets / "missing.onnx").string(), YOLOVersion::kV11);
  TEST_ASSERT(!missing, "missing model is an ordinary creation error");
  std::vector<uint8_t> malformed{0, 1, 2, 3};
  TEST_ASSERT(
      !InferYOLO::Create(**context, std::span(malformed), YOLOVersion::kV11),
      "invalid ONNX bytes do not cross noexcept");
  auto yolo =
      InferYOLO::Create(**context, (assets / "hd2-yolo11n-fp32.onnx").string(),
                        YOLOVersion::kV11);
  TEST_ASSERT(yolo,
              "create repository YOLO CPU instance after creation errors");
  TEST_ASSERT(CheckInputs(**yolo) == 0, "YOLO input contract and recovery");
  auto ocr = InferOCR::Create(
      **context, (assets / "ppocr_keys_v1.txt").string(),
      (assets / "ppocr_det.onnx").string(),
      (assets / "ppocr_rec.onnx").string(), OCRModelType::kPPOCRv4);
  TEST_ASSERT(ocr, "create repository OCR CPU instance");
  TEST_ASSERT(CheckInputs(**ocr) == 0, "OCR input contract and recovery");
  TEST_ASSERT(CheckRuntimeRecovery(**context, assets) == 0,
              "ORT failure recovery");
  TEST_ASSERT(CheckYOLO26(**context, assets) == 0,
              "YOLO26 load-time metadata, ambiguous layouts and FP16");
  TEST_PASS(
      "CPU input boundaries, ROI, fast-math confidence, real ORT recovery");
  return 0;
}
