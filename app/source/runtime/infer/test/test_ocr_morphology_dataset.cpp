#include <filesystem>
#include <iostream>
#include <map>
#include <string_view>

#include "Infer.h"
#include "InferPipeline.h"
#include "OCRTextCorpus.h"
#include "Util.hpp"

using namespace vision_simple;
namespace corpus = ocr_text_corpus;
namespace fs = std::filesystem;

namespace {
int CheckScorer() {
  const std::vector<cv::Rect> labels{{0, 0, 10, 10}, {5, 0, 10, 10}};
  const auto augment = corpus::Score(labels, {{2, 0, 10, 10}, {0, 0, 10, 10}});
  TEST_ASSERT(augment.tp == 2 && augment.fn == 0 && augment.fp == 0,
              "augmenting path recovers two matches that greedy matching loses");
  const auto duplicates = corpus::Score({labels[0]}, {labels[0], labels[0]});
  TEST_ASSERT(duplicates.tp == 1 && duplicates.fp == 1 && duplicates.fn == 0,
              "duplicate predictions cannot match a word twice");
  const auto merged = corpus::Score(labels, {{0, 0, 15, 10}});
  TEST_ASSERT(merged.tp == 1 && merged.fn == 1 && merged.fp == 0,
              "merged line matches at most one independently labeled word");
  TEST_ASSERT(corpus::Score({labels[0]}, {{0, 0, 20, 10}}).tp == 1 &&
                  corpus::Score({labels[0]}, {{0, 0, 21, 10}}).tp == 0,
              "IoU boundary is inclusive at exactly one half");
  const auto negative = corpus::Score({}, {labels[0], cv::Rect{}});
  TEST_ASSERT(negative.tp == 0 && negative.fp == 2 && negative.fn == 0,
              "all negative-image predictions including invalid geometry are false positives");
  TEST_ASSERT(corpus::Score(labels, {}).fn == 2,
              "missing every word is counted, not ignored");
  const auto disjoint = corpus::Score({labels[0]}, {{20, 0, 10, 10}});
  TEST_ASSERT(disjoint.tp == 0 && disjoint.fn == 1 && disjoint.fp == 1,
              "wrong-location detection contributes both a miss and false positive");
  bool bounded = false;
  try {
    corpus::Score({}, std::vector<cv::Rect>(corpus::kMaxRegions + 1));
  } catch (const std::length_error&) {
    bounded = true;
  }
  TEST_ASSERT(bounded, "oversized matching graph is rejected before allocation");
  return 0;
}

corpus::Counts ScoreFrame(const corpus::Frame& frame, const OCRFrameResult& result) {
  std::vector<cv::Rect> boxes;
  boxes.reserve(result.results.size());
  size_t empty = 0;
  for (const auto& prediction : result.results) {
    boxes.push_back(prediction.rect);  // Empty decoded text still participates.
    empty += prediction.line.empty();
  }
  auto counts = corpus::Score(frame.words, boxes);
  counts.empty_text = empty;
  return counts;
}
}  // namespace

int main(int argc, char** argv) {
  fs::path root = fs::current_path();
  if (argc == 3 && std::string_view(argv[1]) == "--project-root")
    root = fs::absolute(argv[2]);
  else if (argc != 1) {
    std::cerr << "usage: test_ocr_morphology_dataset [--project-root ROOT]\n";
    return 1;
  } else {
    while (!fs::exists(root / "app/assets/test") && root != root.parent_path())
      root = root.parent_path();
  }
  TEST_ASSERT(CheckScorer() == 0, "one-to-one scorer invariants");
  const auto assets = root / "app/assets/test";
  auto context = InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kCPU,
                                      {{"ocr_rec_batch_size", "1"}});
  if (!context) std::cerr << context.error().message << '\n';
  TEST_ASSERT(context, "create real CPU context");
  const auto create = [&](OCRDetectionOptions options) {
    return InferOCR::Create(**context, (assets / "ppocr_keys_v1.txt").string(),
                            (assets / "ppocr_det.onnx").string(),
                            (assets / "ppocr_rec.onnx").string(),
                            OCRModelType::kPPOCRv4, 0, options);
  };
  auto omitted = InferOCR::Create(**context, (assets / "ppocr_keys_v1.txt").string(),
                                  (assets / "ppocr_det.onnx").string(),
                                  (assets / "ppocr_rec.onnx").string(),
                                  OCRModelType::kPPOCRv4, 0);
  auto explicit_default = create({2, 3, 64});
  OCRDetectionOptions caller_options{1, 1, 16};
  auto alternative = create(caller_options);
  caller_options = {32, 8, 1048576};  // Mutation must not affect the admitted model.
  auto alternative_reference = create({1, 1, 16});
  if (!omitted) std::cerr << omitted.error().message << '\n';
  if (!explicit_default) std::cerr << explicit_default.error().message << '\n';
  if (!alternative) std::cerr << alternative.error().message << '\n';
  TEST_ASSERT(omitted && explicit_default && alternative && alternative_reference,
              "create trained PP-OCRv4 models with independent construction options");
  auto pipeline = InferPipeline::Create();
  TEST_ASSERT(pipeline, "create staged executor");
  std::map<std::string, corpus::Counts> defaults, alternatives;
  for (size_t i = 0; i < corpus::kCaseCount; ++i) {
    const auto frame = corpus::Render(i);  // At most one corpus frame resident.
    auto legacy = (*omitted)->Run(frame.image, .125f);
    auto explicit_result = (*explicit_default)->Run(frame.image, .125f);
    auto changed = (*alternative)->Run(frame.image, .125f);
    auto reference = (*alternative_reference)->Run(frame.image, .125f);
    TEST_ASSERT(legacy && explicit_result && changed && reference,
                "execute trained CPU inference for every case");
    const auto normalized_default = corpus::SerializeNormalizedResult(*legacy);
    const auto normalized_alternative = corpus::SerializeNormalizedResult(*changed);
    TEST_ASSERT(normalized_default == corpus::SerializeNormalizedResult(*explicit_result),
                "omitted and explicit defaults preserve bbox/text/exact confidence");
    TEST_ASSERT(normalized_alternative == corpus::SerializeNormalizedResult(*reference),
                "construction snapshots caller options across mutation and other models");
    const std::span<const cv::Mat> images(&frame.image, 1);
    auto staged_default = (*pipeline)->Run(**omitted, images, .125f);
    auto staged_alternative = (*pipeline)->Run(**alternative, images, .125f);
    TEST_ASSERT(staged_default && staged_alternative && staged_default->size() == 1 &&
                    staged_alternative->size() == 1,
                "staged execution completes both independent models");
    TEST_ASSERT(normalized_default == corpus::SerializeNormalizedResult(staged_default->front()) &&
                    normalized_alternative == corpus::SerializeNormalizedResult(staged_alternative->front()),
                "sync/staged geometry text and bit confidence agree for both options");
    auto default_after = (*omitted)->Run(frame.image, .125f);
    TEST_ASSERT(default_after && normalized_default == corpus::SerializeNormalizedResult(*default_after),
                "alternative model execution cannot mutate the default model");
    defaults[frame.cohort].Add(ScoreFrame(frame, *legacy));
    alternatives[frame.cohort].Add(ScoreFrame(frame, *changed));
    std::cout << "{\"kind\":\"normalized\",\"corpus\":\"hershey-word-v1\",\"case\":\""
              << frame.id << "\",\"cohort\":\"" << frame.cohort
              << "\",\"default\":" << normalized_default
              << ",\"alternative\":" << normalized_alternative << "}\n";
  }
  for (const auto& [cohort, counts] : defaults) {
    std::cout << "{\"kind\":\"corpus_metrics\",\"corpus\":\"hershey-word-v1\","
                 "\"iou\":0.5,\"recognition_confidence\":0.125,\"device\":0,"
                 "\"ocr_rec_batch_size\":1,\"model_type\":\"kPPOCRv4\",\"width\":640,\"height\":384,"
                 "\"label_granularity\":\"word_ink_bounds\",\"ignore_policy\":\"none\","
                 "\"synthetic_only\":true,\"cohort\":\"" << cohort
              << "\",\"default_2_3_64\":" << corpus::Metrics(counts)
              << ",\"alternative_1_1_16\":" << corpus::Metrics(alternatives.at(cohort)) << "}\n";
  }
  std::cout.flush();
  TEST_ASSERT(std::cout.good(), "checked metrics output");
  return 0;
}
