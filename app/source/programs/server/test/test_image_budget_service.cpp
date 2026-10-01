#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stop_token>
#include <string_view>

#include "InferenceService.h"

using namespace vision_simple;
namespace fs = std::filesystem;
namespace {
void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "image budget service regression: " << message << '\n';
    std::abort();
  }
}
struct FixtureDirectory {
  fs::path original = fs::current_path();
  fs::path path;
  explicit FixtureDirectory(const fs::path& root) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    for (size_t suffix = 0;; ++suffix) {
      path = fs::temp_directory_path() /
             ("vision-simple-image-budget-" + std::to_string(stamp) + "-" +
              std::to_string(suffix));
      if (fs::create_directory(path)) break;
    }
    fs::create_directory(path / "config");
    const auto model = root / "app/assets/test/yolo_runtime_failure.onnx";
    Check(fs::is_regular_file(model), "real reliability ONNX fixture exists");
    std::ofstream config(path / "config/models.yaml");
    config << "models:\n  - task: yolo\n    name: budget-test\n"
              "    version: kV10\n    files:\n      model: '"
           << model.generic_string() << "'\n";
    config.close();
    Check(static_cast<bool>(config), "write isolated model catalog");
    fs::current_path(path);
  }
  ~FixtureDirectory() {
    fs::current_path(original);
    fs::remove_all(path);
  }
};
ImageBudgetStatistics Budget(const InferenceService& service) {
  const auto stats = service.Stats();
  Check(static_cast<bool>(stats), "read shared image budget statistics");
  return stats->image_budget;
}
void Released(const InferenceService& service) {
  const auto budget = Budget(service);
  Check(budget.in_use_bytes == 0 && budget.active_requests == 0,
        "physical return refunds bytes and admission credit");
}
std::shared_ptr<InferenceService> Create(InferenceServiceOptions options) {
  options.idle_timeout = std::chrono::milliseconds{0};
  const auto service = InferenceService::Create(options);
  if (!service) std::cerr << service.error().message << '\n';
  Check(static_cast<bool>(service), "create real CPU service");
  return *service;
}
void Options() {
  const auto maximum = std::numeric_limits<size_t>::max();
  for (size_t value : {size_t{0}, maximum / 3 + 1, maximum}) {
    InferenceServiceOptions options;
    options.max_image_pixels = value;
    const auto result = InferenceService::Create(options);
    Check(!result && result.error().code == VisionSimpleErrorCode::kParameterError,
          "zero and overflowing BGR pixel limits fail Create");
  }
  for (size_t value : {size_t{0}, size_t{1}, size_t{2}}) {
    InferenceServiceOptions options;
    options.max_batch_decoded_bytes = value;
    auto result = InferenceService::Create(options);
    Check(!result && result.error().code == VisionSimpleErrorCode::kParameterError,
          "batch budget must fit one BGR pixel");
    options.max_batch_decoded_bytes = 3;
    options.max_inflight_decoded_bytes = value;
    result = InferenceService::Create(options);
    Check(!result && result.error().code == VisionSimpleErrorCode::kParameterError,
          "global budget must fit one BGR pixel");
  }
  InferenceServiceOptions tiny;
  tiny.max_image_pixels = 1;
  tiny.max_batch_decoded_bytes = tiny.max_inflight_decoded_bytes = 3;
  auto service = Create(tiny);
  Released(*service);
}
void InferenceControls() {
  auto service = Create({});
  const std::array<std::string, 1> malformed{"not-base64"};
  const std::array<cv::Mat, 1> frame{
      cv::Mat(32, 32, CV_8UC3, cv::Scalar::all(0))};
  const std::array<cv::Mat, 1> invalid_frame{cv::Mat{}};
  const auto reject = [](const ServiceResult<InferenceResponse>& result) {
    Check(!result && result.error().kind == ServiceFailure::kInvalidRequest &&
              !result.error().image_index,
          "detector controls reject before model lookup, image decode or empty return");
  };
  const auto reject_inputs = [&](InferenceKind kind, YOLOInferenceOptions controls) {
    reject(service->Run(kind, "missing-model", malformed, controls));
    reject(service->Run(kind, "budget-test", malformed, controls));
    reject(service->Run(kind, "missing-model", {}, controls));
    reject(service->Run(kind, "budget-test", {}, controls));
    reject(service->RunFrames(kind, "missing-model", frame, controls));
    reject(service->RunFrames(kind, "budget-test", invalid_frame, controls));
    reject(service->RunFrames(kind, "missing-model", {}, controls));
    reject(service->RunFrames(kind, "budget-test", {}, controls));
  };
  // Construct nonfinite/subnormal cases by bits so fast-math cannot erase them.
  for (uint32_t bits : {0x7fc00001u, 0x7f800000u, 0xff800000u, 0xbf800000u,
                        0x80000001u, 0x3f800001u}) {
    const float value = std::bit_cast<float>(bits);
    reject_inputs(InferenceKind::kYOLO, {.confidence = value});
    reject_inputs(InferenceKind::kYOLO, {.nms_iou = value});
  }
  for (const auto controls : {YOLOInferenceOptions{.confidence = 0.125f},
                              YOLOInferenceOptions{.nms_iou = 0.3f},
                              YOLOInferenceOptions{.confidence = 0.125f,
                                                   .nms_iou = 0.3f}})
    reject_inputs(InferenceKind::kOCR, controls);
  const auto guarded = service->Stats();
  Check(guarded && guarded->models.empty() &&
            guarded->image_budget.in_use_bytes == 0 &&
            guarded->image_budget.active_requests == 0 &&
            guarded->image_budget.peak_bytes == 0 &&
            guarded->image_budget.decode_calls == 0 &&
            guarded->image_budget.rejected_requests == 0,
        "rejected controls never load models, decode or consume image admission");
  const auto omitted_ocr = service->Run(InferenceKind::kOCR, "missing-model", {});
  Check(!omitted_ocr && omitted_ocr.error().kind == ServiceFailure::kUnknownModel,
        "omitted OCR controls retain ordinary model lookup");
  for (uint32_t bits : {0x80000000u, 0u, 0x3f800000u}) {
    const float value = std::bit_cast<float>(bits);
    const YOLOInferenceOptions controls{.confidence = value, .nms_iou = value};
    auto encoded_empty = service->Run(InferenceKind::kYOLO, "budget-test", {}, controls);
    Check(encoded_empty &&
              std::get<InferYOLOResponse>(encoded_empty->payload).results.empty(),
          "valid boundary controls accept an encoded empty batch");
    encoded_empty->Succeed();
    auto borrowed_empty = service->RunFrames(InferenceKind::kYOLO, "budget-test", {}, controls);
    Check(borrowed_empty &&
              std::get<InferYOLOResponse>(borrowed_empty->payload).results.empty(),
          "valid boundary controls accept a borrowed empty batch");
    borrowed_empty->Succeed();
    auto detected = service->RunFrames(InferenceKind::kYOLO, "budget-test", frame, controls);
    Check(detected &&
              std::get<InferYOLOResponse>(detected->payload).results.size() == 1,
          "valid boundary controls execute real detection");
    const auto& objects = std::get<InferYOLOResponse>(detected->payload).results[0];
    if (bits == 0x3f800000u) {
      Check(objects.empty(), "confidence one excludes the real 0.9 detection");
    } else {
      Check(objects.size() == 1 && objects[0].class_id == 0 &&
                objects[0].confidence == 0.9f && objects[0].bbox[0] == 4 &&
                objects[0].bbox[1] == 4 && objects[0].bbox[2] == 16 &&
                objects[0].bbox[3] == 16,
            "positive and negative zero retain the real detection geometry");
    }
    detected->Succeed();
    Released(*service);
  }
  auto recovered = service->RunFrames(InferenceKind::kYOLO, "budget-test", frame);
  Check(recovered && std::get<InferYOLOResponse>(recovered->payload).results.size() == 1,
        "default request recovers after rejected and boundary controls");
  const auto& objects = std::get<InferYOLOResponse>(recovered->payload).results[0];
  Check(objects.size() == 1 && objects[0].class_id == 0 &&
            objects[0].confidence == 0.9f && objects[0].bbox[0] == 4 &&
            objects[0].bbox[1] == 4 && objects[0].bbox[2] == 16 &&
            objects[0].bbox[3] == 16,
        "per-request controls do not mutate subsequent default inference");
  recovered->Succeed();
  Released(*service);
}
void Frames() {
  InferenceServiceOptions options;
  options.pipeline.max_batches = 1;
  options.max_image_pixels = 1024;
  options.max_batch_decoded_bytes = options.max_inflight_decoded_bytes = 3072;
  auto service = Create(options);
  cv::Mat backing(32, 64, CV_8UC3, cv::Scalar::all(0));
  const std::array<cv::Mat, 1> frame{backing(cv::Rect{0, 0, 32, 32})};
  Check(!frame[0].isContinuous(), "test noncontiguous borrowed ROI");
  auto result = service->RunFrames(InferenceKind::kYOLO, "budget-test", frame);
  Check(result && std::get<InferYOLOResponse>(result->payload).results.size() == 1 &&
            std::get<InferYOLOResponse>(result->payload).results[0].size() == 1,
        "exact pixel/byte boundary executes real inference on borrowed ROI");
  Released(*service);
  Check(Budget(*service).peak_bytes == 3072 && Budget(*service).decode_calls == 0,
        "frame accounting measures logical BGR bytes and never decodes");
  auto retained = service->Stats();
  Check(retained && retained->models.size() == 1 &&
            retained->models[0].active_requests == 1,
        "response retains model lease but not image credit");
  result->Succeed();
  auto second = service->RunFrames(InferenceKind::kYOLO, "budget-test", frame);
  Check(static_cast<bool>(second), "retained response does not exhaust request credit");
  second->Succeed();
  Released(*service);
  const std::array<cv::Mat, 2> over_batch{frame[0], frame[0]};
  auto rejected = service->RunFrames(InferenceKind::kYOLO, "budget-test", over_batch);
  Check(!rejected && rejected.error().kind == ServiceFailure::kImageLimit &&
            rejected.error().image_index == 1,
        "first cumulative batch overflow reports its image index");
  Released(*service);
  const std::array<cv::Mat, 1> over_pixel{backing};
  rejected = service->RunFrames(InferenceKind::kYOLO, "budget-test", over_pixel);
  Check(!rejected && rejected.error().kind == ServiceFailure::kImageLimit &&
            rejected.error().image_index == 0,
        "single image overflow rejects without modifying borrowed input");
  const std::array<cv::Mat, 2> invalid{frame[0], cv::Mat{}};
  rejected = service->RunFrames(InferenceKind::kYOLO, "budget-test", invalid);
  Check(!rejected && rejected.error().kind == ServiceFailure::kInvalidImage &&
            rejected.error().image_index == 1,
        "invalid frame after valid prefix releases admission credit");
  Released(*service);
  std::stop_source stop;
  stop.request_stop();
  rejected = service->RunFrames(InferenceKind::kYOLO, "budget-test", frame,
                               {}, {.stop = stop.get_token()});
  Check(!rejected && rejected.error().kind == ServiceFailure::kCancelled,
        "stopped request never reserves frame bytes");
  rejected = service->RunFrames(
      InferenceKind::kYOLO, "budget-test", frame, {},
      {.timeout = std::chrono::milliseconds{1},
       .started = std::chrono::steady_clock::now() - std::chrono::seconds{1}});
  Check(!rejected && rejected.error().kind == ServiceFailure::kTimedOut,
        "expired request never reserves frame bytes");
  Released(*service);
  Check(Budget(*service).rejected_requests == 2,
        "only budget rejection increments rejected requests");
  const std::array<cv::Mat, 1> native_failure{
      cv::Mat(32, 32, CV_8UC3, cv::Scalar::all(255))};
  rejected = service->RunFrames(InferenceKind::kYOLO, "budget-test", native_failure);
  Check(!rejected && rejected.error().kind == ServiceFailure::kInference,
        "real native inference failure drains charged input");
  Released(*service);
  auto recovered = service->RunFrames(InferenceKind::kYOLO, "budget-test", frame);
  Check(static_cast<bool>(recovered), "native failure refunds usable credit");
  recovered->Succeed();
  Released(*service);
}
void GlobalBoundary() {
  InferenceServiceOptions options;
  options.max_image_pixels = 1024;
  options.max_batch_decoded_bytes = 3072;
  options.max_inflight_decoded_bytes = 3071;
  auto service = Create(options);
  const std::array<cv::Mat, 1> frame{
      cv::Mat(32, 32, CV_8UC3, cv::Scalar::all(0))};
  auto rejected = service->RunFrames(InferenceKind::kYOLO, "budget-test", frame);
  Check(!rejected && rejected.error().kind == ServiceFailure::kBusy &&
            !rejected.error().image_index,
        "one byte over global quota is busy, not an image limit");
  Released(*service);
  Check(Budget(*service).peak_bytes == 0 && Budget(*service).rejected_requests == 1,
        "failed all-batch reservation never charges partial bytes");
  auto empty = service->RunFrames(InferenceKind::kYOLO, "budget-test", {});
  Check(empty && std::get<InferYOLOResponse>(empty->payload).results.empty(),
        "empty valid-model request does not need image quota");
  empty->Succeed();
  Released(*service);
}
}  // namespace
int main(int argc, char** argv) {
  fs::path root = fs::current_path();
  if (argc == 3 && std::string_view(argv[1]) == "--project-root")
    root = fs::absolute(argv[2]);
  else if (argc != 1) {
    std::cerr << "usage: test_image_budget_service [--project-root ROOT]\n";
    return 1;
  } else {
    while (!fs::exists(root / "app/assets/test") && root != root.parent_path())
      root = root.parent_path();
  }
  FixtureDirectory fixtures(root);
  Options();
  InferenceControls();
  Frames();
  GlobalBoundary();
  std::cout << "image budget service regression passed\n";
}
