#include <array>
#include <atomic>
#include <barrier>
#include <filesystem>
#include <iostream>
#include <thread>

#include "InferPipeline.h"
#include "Util.hpp"

using namespace vision_simple;
namespace fs = std::filesystem;

namespace {
bool Same(const YOLOFrameResult& a, const YOLOFrameResult& b) {
  if (a.results.size() != b.results.size()) return false;
  for (size_t i = 0; i < a.results.size(); ++i) {
    if (a.results[i].bbox != b.results[i].bbox ||
        a.results[i].class_id != b.results[i].class_id ||
        std::abs(a.results[i].confidence - b.results[i].confidence) > 1e-5f)
      return false;
  }
  return true;
}
bool Same(const OCRFrameResult& a, const OCRFrameResult& b) {
  if (a.results.size() != b.results.size()) return false;
  for (size_t i = 0; i < a.results.size(); ++i) {
    if (a.results[i].rect != b.results[i].rect ||
        a.results[i].line != b.results[i].line ||
        std::abs(a.results[i].confidence - b.results[i].confidence) > 1e-5f)
      return false;
  }
  return true;
}
template <typename Model>
int Ordered(InferPipeline& pipeline, Model& model) {
  std::vector<cv::Mat> images;
  // The tiny detector has a fixed 32x32 input; vary unpadded image height
  // within that extent to exercise ordering and different recognition widths.
  for (int height : {32, 24, 16, 8, 30, 20, 28, 32})
    images.emplace_back(height, 32, CV_8UC3, cv::Scalar::all(0));
  auto result = pipeline.Run(model, images, .5f);
  if (!result) std::cerr << "Pipeline failure: " << result.error().cause.message << '\n';
  TEST_ASSERT(result && result->size() == images.size(), "every input has a result");
  for (size_t i = 0; i < images.size(); ++i) {
    const auto direct = model.Run(images[i], .5f);
    TEST_ASSERT(direct && Same(*direct, (*result)[i]),
                "pipeline preserves input ordering and native output");
  }
  return 0;
}

int Interrupted(InferYOLO& model, bool close) {
  auto created = InferPipeline::Create({1, 1, 4096});
  TEST_ASSERT(created, "create single-credit pipeline");
  auto& pipeline = **created;
  const cv::Mat black(32, 32, CV_8UC3, cv::Scalar::all(0));
  std::vector<cv::Mat> images(4096, black);
  const std::array<cv::Mat, 1> probe{black};
  std::stop_source stop;
  std::atomic<bool> done{false};
  std::optional<PipelineResult<YOLOFrameResult>> result;
  std::jthread caller([&] {
    // A competing probe can win admission; retry that controlled rejection.
    do {
      result.emplace(pipeline.Run(model, images, .5f, {stop.get_token()}));
    } while (!*result && result->error().kind == PipelineFailureKind::kBusy);
    done.store(true, std::memory_order_release);
  });
  bool observed_busy = false;
  while (!done.load(std::memory_order_acquire)) {
    auto attempt = pipeline.Run(model, probe, .5f);
    if (!attempt && attempt.error().kind == PipelineFailureKind::kBusy) {
      observed_busy = true;
      if (close) pipeline.Close();
      else stop.request_stop();
      break;
    }
    std::this_thread::yield();
  }
  caller.join();
  TEST_ASSERT(observed_busy, "active batch rejects competing admission immediately");
  TEST_ASSERT(result && !*result && result->error().kind ==
                  (close ? PipelineFailureKind::kClosed : PipelineFailureKind::kCancelled),
              "admitted backpressured batch drains on cancellation or close");
  if (!close) {
    const auto recovery = pipeline.Run(model, probe, .5f);
    TEST_ASSERT(recovery && recovery->size() == 1 &&
                    (*recovery)[0].results.size() == 1,
                "cancelled batch releases its capacity and model workspace");
  } else {
    const auto rejected = pipeline.Run(model, probe, .5f);
    TEST_ASSERT(!rejected && rejected.error().kind == PipelineFailureKind::kClosed,
                "close permanently rejects new work");
  }
  return 0;
}
}

int main(int argc, char** argv) {
  fs::path root = fs::current_path();
  if (argc == 3 && std::string_view(argv[1]) == "--project-root")
    root = fs::absolute(argv[2]);
  else if (argc != 1) {
    std::cerr << "usage: test_pipeline [--project-root ROOT]\n";
    return 1;
  } else {
    while (!fs::exists(root / "app/assets/test") && root != root.parent_path())
      root = root.parent_path();
  }
  const auto assets = root / "app/assets/test";
  auto context = InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kCPU);
  TEST_ASSERT(context, "create CPU context");
  auto yolo = InferYOLO::Create(**context,
      (assets / "yolo_runtime_failure.onnx").string(), YOLOVersion::kV10);
  TEST_ASSERT(yolo, "create tiny real YOLO session");
  auto det = ReadAll((assets / "ocr_det_box.onnx").string());
  auto rec = ReadAll((assets / "ocr_rec_runtime_failure.onnx").string());
  TEST_ASSERT(det && rec, "read tiny OCR fixtures");
  auto ocr = InferOCR::Create(**context, {{0, "A"}}, det->span(), rec->span(),
                            OCRModelType::kPPOCRv4);
  TEST_ASSERT(ocr, "create tiny real OCR sessions");
  for (const auto options : {PipelineOptions{0, 1, 1}, PipelineOptions{65, 1, 1},
                            PipelineOptions{1, 0, 1}, PipelineOptions{1, 65, 1},
                            PipelineOptions{1, 1, 0}, PipelineOptions{1, 1, 4097}}) {
    const auto invalid = InferPipeline::Create(options);
    TEST_ASSERT(!invalid && invalid.error().code == VisionSimpleErrorCode::kParameterError,
                "invalid bounds are controlled parameter errors");
  }
  for (size_t capacity : {1u, 4u}) {
    auto pipeline = InferPipeline::Create({capacity, 4, 128});
    TEST_ASSERT(pipeline, "create bounded pipeline");
    const cv::Mat direct_image(32, 32, CV_8UC3, cv::Scalar::all(0));
    const auto yolo_reference = (*yolo)->Run(direct_image, .5f);
    const auto ocr_reference = (*ocr)->Run(direct_image, .5f);
    TEST_ASSERT(yolo_reference && ocr_reference, "direct inference baselines succeed");
    bool direct_correct = true;
    std::barrier start(3);
    int yolo_status = -1;
    int ocr_status = -1;
    std::jthread yolo_call([&] { start.arrive_and_wait(); yolo_status = Ordered(**pipeline, **yolo); });
    std::jthread ocr_call([&] { start.arrive_and_wait(); ocr_status = Ordered(**pipeline, **ocr); });
    std::jthread direct_call([&] {
      start.arrive_and_wait();
      for (int iteration = 0; iteration < 16; ++iteration) {
        const auto direct_yolo = (*yolo)->Run(direct_image, .5f);
        const auto direct_ocr = (*ocr)->Run(direct_image, .5f);
        if (!direct_yolo || !direct_ocr ||
            !Same(*direct_yolo, *yolo_reference) ||
            !Same(*direct_ocr, *ocr_reference))
          direct_correct = false;
      }
    });
    yolo_call.join();
    ocr_call.join();
    direct_call.join();
    TEST_ASSERT(direct_correct,
                "direct Run shares session gates without corrupting staged outputs");
    TEST_ASSERT(yolo_status == 0 && ocr_status == 0,
                "mixed real-model batches retain type and output order even at capacity one");
    const cv::Mat black(32, 32, CV_8UC3, cv::Scalar::all(0));
    const cv::Mat white(32, 32, CV_8UC3, cv::Scalar::all(255));
    const std::vector<cv::Mat> failing{black, white, cv::Mat{}, white};
    auto failed = (*pipeline)->Run(**yolo, failing, .5f);
    TEST_ASSERT(!failed && failed.error().kind == PipelineFailureKind::kInference &&
                    failed.error().image_index == 1 &&
                    failed.error().cause.code == VisionSimpleErrorCode::kRuntimeError,
                "earlier native error wins over later preprocessing failure");
    const std::vector<cv::Mat> excessive(129, black);
    auto oversized = (*pipeline)->Run(**yolo, excessive, .5f);
    TEST_ASSERT(!oversized && oversized.error().kind == PipelineFailureKind::kInvalidRequest,
                "batch storage limit rejects oversized input");
    auto empty = (*pipeline)->Run(**yolo, {}, .5f);
    TEST_ASSERT(empty && empty->empty(), "empty input produces empty output");
    std::stop_source stop;
    stop.request_stop();
    auto stopped = (*pipeline)->Run(**yolo, {}, .5f,
        {stop.get_token(), std::chrono::steady_clock::now()});
    TEST_ASSERT(!stopped && stopped.error().kind == PipelineFailureKind::kCancelled,
                "cancellation precedes deadline even for empty input");
    auto expired = (*pipeline)->Run(**yolo, failing, .5f,
        {{}, std::chrono::steady_clock::now()});
    TEST_ASSERT(!expired && expired.error().kind == PipelineFailureKind::kTimedOut,
                "deadline precedes native inference errors");
  }
  auto timed_pipeline = InferPipeline::Create({1, 1, 4096});
  TEST_ASSERT(timed_pipeline, "create deadline pipeline");
  const std::vector<cv::Mat> long_batch(
      4096, cv::Mat(32, 32, CV_8UC3, cv::Scalar::all(0)));
  const auto timed = (*timed_pipeline)->Run(**yolo, long_batch, .5f,
      {{}, std::chrono::steady_clock::now() + std::chrono::milliseconds(1)});
  TEST_ASSERT(!timed && timed.error().kind == PipelineFailureKind::kTimedOut,
              "deadline wakes capacity waits and drains admitted stages");
  const std::span<const cv::Mat> single(long_batch.data(), 1);
  const auto after_timeout = (*timed_pipeline)->Run(**yolo, single, .5f);
  TEST_ASSERT(after_timeout && (*after_timeout)[0].results.size() == 1,
              "timed-out run returns all credits before returning");
  TEST_ASSERT(Interrupted(**yolo, false) == 0, "backpressure cancellation and recovery");
  TEST_ASSERT(Interrupted(**yolo, true) == 0, "backpressure close and drain");
  TEST_PASS("bounded stages, mixed models, ordered failures, cancellation and close");
  return 0;
}
