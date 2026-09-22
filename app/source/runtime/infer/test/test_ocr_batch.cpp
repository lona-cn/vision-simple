#include <array>
#include <filesystem>
#include <string_view>

#include "Infer.h"
#include "InferPipeline.h"
#include "Util.hpp"

using namespace vision_simple;
namespace fs = std::filesystem;

namespace {
cv::Mat SourceImage(bool reverse = false) {
  cv::Mat image(256, 384, CV_8UC3);
  for (int y = 0; y < image.rows; ++y) {
    const int band = std::min(y / 64, 3);
    const int value = 32 + 64 * (reverse ? 3 - band : band);
    image.row(y).setTo(cv::Scalar(value, value, value));
  }
  return image;
}

int CheckLines(const OCRFrameResult& result, bool sar = false,
               bool reverse = false) {
  TEST_ASSERT_EQ(result.results.size(), 4u,
                 "exactly four real crops, no padded samples");
  std::array<bool, 4> seen{};
  for (const auto& line : result.results) {
    // Identify the source region by its center, not the expanded crop margin.
    const auto band = std::min((line.rect.y + line.rect.height / 2) / 64, 3);
    TEST_ASSERT(band >= 0 && !seen[band],
                "each detected source region occurs once");
    seen[band] = true;
    const char character = static_cast<char>('A' + (reverse ? 3 - band : band));
    TEST_ASSERT_EQ(line.line, std::string(sar ? 2 : 1, character),
                   "decoded sample belongs to its source rectangle");
    TEST_ASSERT_FLOAT_EQ(line.confidence, .9f, 1e-6f, "per-sample confidence");
  }
  return 0;
}

int SameOrder(const OCRFrameResult& reference, const OCRFrameResult& batched) {
  TEST_ASSERT_EQ(reference.results.size(), batched.results.size(),
                 "same crop count");
  for (size_t i = 0; i < reference.results.size(); ++i) {
    TEST_ASSERT(reference.results[i].rect == batched.results[i].rect,
                "width grouping preserves detection order");
    TEST_ASSERT_EQ(reference.results[i].line, batched.results[i].line,
                   "batching preserves recognition");
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
  const auto dictionary =
      std::map<int, std::string>{{0, "A"}, {1, "B"}, {2, "C"}, {3, "D"}};
  auto det = ReadAll((assets / "ocr_det_batch.onnx").string());
  TEST_ASSERT(det, "load deterministic multi-crop detection model");
  auto single_context =
      InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kCPU);
  auto batch_context =
      InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kCPU,
                           {{"ocr_rec_batch_size", "4"}});
  if (!single_context) std::cerr << single_context.error().message << '\n';
  if (!batch_context) std::cerr << batch_context.error().message << '\n';
  TEST_ASSERT(single_context && batch_context, "create real ORT contexts");
  auto rec = ReadAll((assets / "ocr_rec_batch.onnx").string());
  TEST_ASSERT(rec, "load input-dependent recognition model");
  auto single = InferOCR::Create(**single_context, dictionary, det->span(),
                                 rec->span(), OCRModelType::kPPOCRv4);
  auto batch = InferOCR::Create(**batch_context, dictionary, det->span(),
                                rec->span(), OCRModelType::kPPOCRv4);
  TEST_ASSERT(single && batch, "create single and dynamic batch recognition");
  const auto image = SourceImage();
  const auto reversed = SourceImage(true);
  auto baseline = (*single)->Run(image, .5f);
  auto grouped = (*batch)->Run(image, .5f);
  TEST_ASSERT(baseline && grouped, "execute real mixed-width recognition");
  TEST_ASSERT(CheckLines(*baseline) == 0 && CheckLines(*grouped) == 0,
              "source-to-text correspondence");
  TEST_ASSERT(SameOrder(*baseline, *grouped) == 0,
              "original ordering restored");

  auto required_data =
      ReadAll((assets / "ocr_rec_batch_required.onnx").string());
  TEST_ASSERT(required_data, "load N>1-required graph");
  auto required_single =
      InferOCR::Create(**single_context, dictionary, det->span(),
                       required_data->span(), OCRModelType::kPPOCRv4);
  auto required_batch =
      InferOCR::Create(**batch_context, dictionary, det->span(),
                       required_data->span(), OCRModelType::kPPOCRv4);
  TEST_ASSERT(required_single && required_batch,
              "create runtime batch guard models");
  TEST_ASSERT(!(*required_single)->Run(image, .5f),
              "graph rejects N=1 in real ORT execution");
  auto actual_batch = (*required_batch)->Run(image, .5f);
  TEST_ASSERT(actual_batch && CheckLines(*actual_batch) == 0,
              "actual N>1 reaches ORT, not a loop over singles");

  auto fixed_data = ReadAll((assets / "ocr_rec_fixed_batch.onnx").string());
  TEST_ASSERT(fixed_data, "load fixed N=3/W=240 graph");
  auto fixed = InferOCR::Create(**single_context, dictionary, det->span(),
                                fixed_data->span(), OCRModelType::kPPOCRv4);
  TEST_ASSERT(fixed, "fixed batch metadata overrides configured single size");
  auto fixed_result = (*fixed)->Run(image, .5f);
  TEST_ASSERT(fixed_result && CheckLines(*fixed_result) == 0,
              "fixed N tail excludes dummy outputs");
  TEST_ASSERT(SameOrder(*baseline, *fixed_result) == 0,
              "fixed-width output order");

  auto sar_data = ReadAll((assets / "ocr_rec_sar_batch.onnx").string());
  TEST_ASSERT(sar_data, "load SAR prediction graph");
  auto sar = InferOCR::Create(**batch_context, dictionary, det->span(),
                              sar_data->span(), OCRModelType::kPaddleSAR);
  TEST_ASSERT(sar, "create registry-selected SAR backend");
  auto sar_result = (*sar)->Run(image, .5f);
  TEST_ASSERT(sar_result && CheckLines(*sar_result, true) == 0,
              "SAR retains repeats and stops at EOS end to end");

  auto pipeline = InferPipeline::Create();
  TEST_ASSERT(pipeline, "create real stage pipeline");
  const std::array<cv::Mat, 2> frames{image, reversed};
  auto piped = (*pipeline)->Run(**required_batch, frames, .5f);
  TEST_ASSERT(piped && piped->size() == 2,
              "pipeline performs genuine minibatches");
  TEST_ASSERT(
      CheckLines((*piped)[0]) == 0 && CheckLines((*piped)[1], false, true) == 0,
      "per-frame workspaces do not overwrite each other");
  auto sar_file = InferOCR::Create(
      **batch_context, (assets / "ocr_sar_dictionary.txt").string(),
      (assets / "ocr_det_batch.onnx").string(),
      (assets / "ocr_rec_sar_batch.onnx").string(), OCRModelType::kPaddleSAR);
  TEST_ASSERT(sar_file, "SAR file factory must not append a CTC space class");
  auto sar_file_result = (*sar_file)->Run(image, .5f);
  TEST_ASSERT(sar_file_result && CheckLines(*sar_file_result, true) == 0,
              "file-based SAR dictionary agrees with memory-based dictionary");
  auto reused = (*batch)->Run(reversed, .5f);
  TEST_ASSERT(reused && CheckLines(*reused, false, true) == 0,
              "reused input tensor replaces all previous pixels");

  for (const std::string value : {"0", "65", "-1", "4x"}) {
    auto bad_context =
        InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kCPU,
                             {{"ocr_rec_batch_size", value}});
    TEST_ASSERT(bad_context, "context creation permits model-specific options");
    TEST_ASSERT(!InferOCR::Create(**bad_context, dictionary, det->span(),
                                  rec->span(), OCRModelType::kPPOCRv4),
                "invalid recognition batch rejected at model creation");
  }
  TEST_ASSERT(!InferOCR::Create(**single_context, dictionary, det->span(),
                                rec->span(), OCRModelType::kEasyOCR),
              "unsupported EasyOCR never silently uses Paddle CTC");
  TEST_PASS(
      "real ORT mixed widths, batch identity, fixed tail, SAR, ordering and "
      "workspace reuse");
  return 0;
}
