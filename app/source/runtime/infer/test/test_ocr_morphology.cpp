#include <array>
#include <filesystem>
#include <future>
#include <string_view>

#include "Infer.h"
#include "InferPipeline.h"
#include "Util.hpp"

using namespace vision_simple;
namespace fs = std::filesystem;

namespace {
cv::Mat SourceImage() {
  cv::Mat image(256, 384, CV_8UC3);
  for (int band = 0; band < 4; ++band)
    image.rowRange(band * 64, (band + 1) * 64)
        .setTo(cv::Scalar::all(32 + band * 64));
  return image;
}

int Check(const OCRFrameResult& frame, const std::array<cv::Rect, 4>& boxes,
          bool wide_only = false) {
  TEST_ASSERT_EQ(frame.results.size(), wide_only ? 2u : 4u,
                 "strict pre-unclip region admission");
  std::array<bool, 4> seen{};
  for (const auto& line : frame.results) {
    const int band = (line.rect.y + line.rect.height / 2) / 64;
    TEST_ASSERT(band >= 0 && band < 4 && !seen[band],
                "distinct source region recognized once");
    TEST_ASSERT(!wide_only || band % 2 == 1, "only wide regions survive");
    seen[band] = true;
    TEST_ASSERT(line.rect == boxes[band], "exact morphology/unclip geometry");
    TEST_ASSERT_EQ(line.line, std::string(1, static_cast<char>('A' + band)),
                   "recognition still belongs to its source region");
    TEST_ASSERT_FLOAT_EQ(line.confidence, .9f, 1e-6f,
                         "recognition confidence unchanged");
  }
  return 0;
}

int Same(const OCRFrameResult& a, const OCRFrameResult& b) {
  TEST_ASSERT_EQ(a.results.size(), b.results.size(), "same ordered result count");
  for (size_t i = 0; i < a.results.size(); ++i) {
    TEST_ASSERT(a.results[i].rect == b.results[i].rect, "same ordered geometry");
    TEST_ASSERT_EQ(a.results[i].line, b.results[i].line, "same ordered text");
    TEST_ASSERT_EQ(a.results[i].confidence, b.results[i].confidence,
                   "same exact confidence");
  }
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  fs::path root = fs::current_path();
  if (argc == 3 && std::string_view(argv[1]) == "--project-root")
    root = fs::absolute(argv[2]);
  else if (argc != 1)
    return 1;
  const auto assets = root / "app/assets/test";
  auto context = InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kCPU);
  TEST_ASSERT(context, "real CPU ORT context");
  auto det = ReadAll((assets / "ocr_det_batch.onnx").string());
  auto rec = ReadAll((assets / "ocr_rec_batch.onnx").string());
  TEST_ASSERT(det && rec, "existing deterministic detector/input-dependent recognizer");
  const std::map<int, std::string> dictionary{{0, "A"}, {1, "B"}, {2, "C"}, {3, "D"}};
  const auto create = [&](OCRDetectionOptions options) {
    return InferOCR::Create(**context, dictionary, det->span(), rec->span(),
                            OCRModelType::kPPOCRv4, 0, options);
  };
  auto omitted = InferOCR::Create(**context, dictionary, det->span(), rec->span(),
                                  OCRModelType::kPPOCRv4);
  OCRDetectionOptions caller_options;
  auto explicit_default = create(caller_options);
  caller_options = {32, 8, 1048576};
  auto zero = create({32, 0, 64});
  auto one = create({2, 1, 64});
  auto odd = create({3, 1, 64});
  auto identity = create({1, 8, 64});
  auto below = create({1, 0, 1279});
  auto equal = create({1, 0, 1280});
  auto all_equal = create({1, 0, 2240});
  TEST_ASSERT(omitted && explicit_default && zero && one && odd && identity &&
                  below && equal && all_equal, "construct independent immutable models");
  const auto image = SourceImage();
  // Golden rectangles derive from the fixture's 64/112 x 20 masks, default
  // OpenCV anchors, and legacy DB unclip; no predictions generate expectations.
  const std::array<cv::Rect, 4> legacy_boxes{
      cv::Rect(19, 7, 93, 49), cv::Rect(18, 70, 143, 51),
      cv::Rect(19, 135, 93, 49), cv::Rect(18, 198, 143, 51)};
  const std::array<cv::Rect, 4> zero_boxes{
      cv::Rect(21, 9, 86, 42), cv::Rect(19, 71, 138, 46),
      cv::Rect(21, 137, 86, 42), cv::Rect(19, 199, 138, 46)};
  const std::array<cv::Rect, 4> one_boxes{
      cv::Rect(20, 8, 89, 45), cv::Rect(19, 71, 139, 47),
      cv::Rect(20, 136, 89, 45), cv::Rect(19, 199, 139, 47)};
  const std::array<cv::Rect, 4> odd_boxes{
      cv::Rect(19, 7, 90, 46), cv::Rect(17, 69, 142, 50),
      cv::Rect(19, 135, 90, 46), cv::Rect(17, 197, 142, 50)};
  auto legacy = (*omitted)->Run(image, .5f);
  auto copied = (*explicit_default)->Run(image, .5f);
  auto no_dilation = (*zero)->Run(image, .5f);
  auto once = (*one)->Run(image, .5f);
  auto odd_result = (*odd)->Run(image, .5f);
  auto unchanged = (*identity)->Run(image, .5f);
  auto admitted = (*below)->Run(image, .5f);
  auto boundary = (*equal)->Run(image, .5f);
  auto excluded = (*all_equal)->Run(image, .5f);
  TEST_ASSERT(legacy && copied && no_dilation && once && odd_result && unchanged &&
                  admitted && boundary && excluded, "execute real ORT inference");
  TEST_ASSERT(Check(*legacy, legacy_boxes) == 0 && Same(*legacy, *copied) == 0,
              "omitted/explicit legacy and caller mutation isolation");
  TEST_ASSERT(Check(*no_dilation, zero_boxes) == 0 &&
                  Same(*no_dilation, *unchanged) == 0,
              "zero iterations ignores kernel; kernel one is identity");
  TEST_ASSERT(Check(*once, one_boxes) == 0 && Check(*odd_result, odd_boxes) == 0,
              "iteration count and odd kernel control exact geometry");
  TEST_ASSERT(Check(*admitted, zero_boxes) == 0 &&
                  Check(*boundary, zero_boxes, true) == 0 && excluded->results.empty(),
              "area equality rejected before expansion, not contour area");

  auto pipeline = InferPipeline::Create();
  TEST_ASSERT(pipeline, "real staged pipeline");
  const std::array<cv::Mat, 3> frames{image, image, image};
  auto staged_default = (*pipeline)->Run(**omitted, frames, .5f);
  auto staged_zero = (*pipeline)->Run(**zero, frames, .5f);
  TEST_ASSERT(staged_default && staged_zero && staged_default->size() == 3 &&
                  staged_zero->size() == 3, "all staged frames completed");
  for (size_t i = 0; i < frames.size(); ++i)
    TEST_ASSERT(Same(*legacy, (*staged_default)[i]) == 0 &&
                    Same(*no_dilation, (*staged_zero)[i]) == 0,
                "staged options match synchronous model-owned options");

  auto direct = std::async(std::launch::async, [&] {
    for (int i = 0; i < 4; ++i) {
      auto value = (*omitted)->Run(image, .5f);
      if (!value || Same(*legacy, *value) != 0) return 1;
    }
    return 0;
  });
  auto staged = std::async(std::launch::async, [&] {
    auto value = (*pipeline)->Run(**zero, frames, .5f);
    if (!value || value->size() != frames.size()) return 1;
    for (const auto& frame : *value)
      if (Same(*no_dilation, frame) != 0) return 1;
    return 0;
  });
  auto shared_model = std::async(std::launch::async, [&] {
    auto value = (*pipeline)->Run(**omitted, frames, .5f);
    if (!value || value->size() != frames.size()) return 1;
    for (const auto& frame : *value)
      if (Same(*legacy, frame) != 0) return 1;
    return 0;
  });
  TEST_ASSERT(direct.get() == 0 && staged.get() == 0 && shared_model.get() == 0,
              "concurrent same/distinct model kernels/options stay isolated");

  for (const auto invalid : std::array<OCRDetectionOptions, 6>{
           OCRDetectionOptions{0, 3, 64}, {33, 3, 64}, {2, -1, 64},
           {2, 9, 64}, {2, 3, -1}, {2, 3, 1048577}}) {
    auto bytes = InferOCR::Create(**context, dictionary, std::span<uint8_t>{},
                                 std::span<uint8_t>{}, OCRModelType::kPPOCRv4,
                                 0, invalid);
    auto arithmetic = InferOCR::Create(**context, dictionary, std::span<float>{},
                                      std::span<float>{}, OCRModelType::kPPOCRv4,
                                      0, invalid);
    auto files = InferOCR::Create(**context, "", "", "",
                                 OCRModelType::kPPOCRv4, 0, invalid);
    TEST_ASSERT(!bytes && !arithmetic && !files &&
                    bytes.error().code == VisionSimpleErrorCode::kParameterError &&
                    arithmetic.error().code == VisionSimpleErrorCode::kParameterError &&
                    files.error().code == VisionSimpleErrorCode::kParameterError,
                "invalid options precede invalid models and file IO in every factory");
  }
  // Exercise the arithmetic-span and file factories with real models/options.
  auto signed_bytes = InferOCR::Create(
      **context, dictionary,
      std::span<int8_t>(reinterpret_cast<int8_t*>(det->span().data()), det->span().size()),
      std::span<int8_t>(reinterpret_cast<int8_t*>(rec->span().data()), rec->span().size()),
      OCRModelType::kPPOCRv4, 0, OCRDetectionOptions{32, 0, 64});
  auto sar_file = InferOCR::Create(
      **context, (assets / "ocr_sar_dictionary.txt").string(),
      (assets / "ocr_det_batch.onnx").string(),
      (assets / "ocr_rec_sar_batch.onnx").string(), OCRModelType::kPaddleSAR,
      0, OCRDetectionOptions{1, 0, 2240});
  TEST_ASSERT(signed_bytes && sar_file, "real arithmetic and file construction");
  auto arithmetic_result = (*signed_bytes)->Run(image, .5f);
  auto file_result = (*sar_file)->Run(image, .5f);
  TEST_ASSERT(arithmetic_result && Same(*no_dilation, *arithmetic_result) == 0 &&
                  file_result && file_result->results.empty(),
              "all factory forms apply morphology options");
  TEST_PASS("OCR immutable morphology, strict boundaries, factories and staged isolation");
  return 0;
}
