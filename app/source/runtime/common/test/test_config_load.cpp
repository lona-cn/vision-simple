#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>

#include "VisionSimpleConfig.h"

using namespace vision_simple;

namespace {
int failures = 0;
bool Require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "configuration test failed: " << message << '\n';
    ++failures;
  }
  return condition;
}

class TemporaryConfig {
 public:
  TemporaryConfig() {
    const auto root = std::filesystem::temp_directory_path();
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    for (unsigned attempt = 0; attempt < 100; ++attempt) {
      directory_ = root / ("vision-simple-config-" + std::to_string(stamp) +
                           "-" + std::to_string(attempt));
      if (std::filesystem::create_directory(directory_)) return;
    }
    directory_.clear();
    Require(false, "unable to create temporary config directory");
  }
  bool valid() const { return !directory_.empty(); }
  ~TemporaryConfig() {
    std::error_code ignored;
    if (valid()) std::filesystem::remove_all(directory_, ignored);
  }
  std::expected<Config, VisionSimpleError> Load(std::string_view yaml) {
    const auto path = directory_ / "models.yaml";
    {
      std::ofstream output(path, std::ios::binary | std::ios::trunc);
      output << yaml;
      output.close();
      if (!Require(bool(output), "temporary configuration write failed")) {
        return MK_VSERROR(VisionSimpleErrorCode::kIOError,
                          "temporary configuration write failed");
      }
    }
    const auto filename = path.string();
    return Config::Load({filename});
  }

 private:
  std::filesystem::path directory_;
};

constexpr std::string_view legacy = R"(yolo:
  - name: shared
    version: kV11
    path: missing-yolo.onnx
ocr:
  - name: shared
    version: kPPOCRv4
    det_path: missing-det.onnx
    rec_path: missing-rec.onnx
    char_dict_path: missing-dictionary.txt
)";
constexpr std::string_view unified = R"(models:
  - task: yolo
    name: shared
    version: kV11
    files:
      model: missing-yolo.onnx
  - task: ocr
    name: shared
    version: kPPOCRv4
    files:
      det: missing-det.onnx
      rec: missing-rec.onnx
      dictionary: missing-dictionary.txt
)";

void TestEquivalentCatalogs(TemporaryConfig& temporary) {
  const auto old_config = temporary.Load(legacy);
  const auto new_config = temporary.Load(unified);
  if (!Require(
          old_config.has_value() && new_config.has_value(),
          "legacy and unified configurations must load without model files"))
    return;
  const auto& old_models = old_config->model_config().models;
  const auto& new_models = new_config->model_config().models;
  if (!Require(
          old_models.size() == 2 && new_models.size() == 2,
          "same-name models belonging to distinct tasks must both survive"))
    return;
  for (size_t i = 0; i < old_models.size(); ++i) {
    const auto& a = old_models[i];
    const auto& b = new_models[i];
    Require(a.task == b.task && a.name == b.name && a.version == b.version &&
                a.files == b.files,
            "legacy and unified schemas must produce identical definitions");
  }
  Require(new_models[0].files.at("model") == "missing-yolo.onnx" &&
              new_models[1].files.at("dictionary") == "missing-dictionary.txt",
          "canonical resource roles must preserve configured paths");
  const auto& projections = new_config->model_config();
  Require(projections.yolo.size() == 1 && projections.ocr.size() == 1 &&
              projections.yolo[0].path == "missing-yolo.onnx" &&
              projections.ocr[0].det_path == "missing-det.onnx" &&
              projections.ocr[0].rec_path == "missing-rec.onnx" &&
              projections.ocr[0].char_dict_path == "missing-dictionary.txt",
          "legacy readers must see unified models through projections");
  const Config reconstructed{new_config->model_config()};
  const auto& reconstructed_models = reconstructed.model_config().models;
  Require(reconstructed_models.size() == new_models.size(),
          "reconstructing from projected public DTO must not duplicate models");
  for (size_t i = 0; i < new_models.size() && i < reconstructed_models.size();
       ++i)
    Require(reconstructed_models[i].task == new_models[i].task &&
                reconstructed_models[i].name == new_models[i].name &&
                reconstructed_models[i].version == new_models[i].version &&
                reconstructed_models[i].files == new_models[i].files,
            "public DTO round-trip preserves canonical model definitions");
}

void TestMixedAndInvalidCatalogs(TemporaryConfig& temporary) {
  const auto mixed = temporary.Load(R"(models:
  - task: yolo
    name: first
    version: unsupported-version
ocr:
  - name: second
    version: unsupported-version
)");
  if (!Require(mixed.has_value(),
               "resource and version checks must remain lazy"))
    return;
  const auto& models = mixed->model_config().models;
  Require(models.size() == 2 && models[0].task == "yolo" &&
              models[0].version == "unsupported-version" &&
              models[0].files.empty() && models[1].task == "ocr" &&
              models[1].files.at("det").empty(),
          "mixed input must preserve missing resources for lazy loading");

  for (const auto yaml : {
           R"(models:
  - task: yolo
    name: duplicate
    version: first
yolo:
  - name: duplicate
    version: second
)",
           R"(yolo:
  - name: duplicate
  - name: duplicate
)",
           R"(models:
  - task: ocr
    name: duplicate
  - task: ocr
    name: duplicate
)",
           R"(models:
  - name: missing-task
)",
           R"(models:
  - task: yolo
)",
           R"(models:
  - task: yolo
    name: "unterminated
)"}) {
    const auto invalid = temporary.Load(yaml);
    if (!Require(!invalid,
                 "duplicates, empty identities and malformed YAML must fail"))
      continue;
    Require(invalid.error().code == VisionSimpleErrorCode::kRuntimeError,
            "invalid YAML/catalog must be a configuration runtime error");
  }

  const auto unknown = temporary.Load(R"(models:
  - task: custom-task
    name: custom-model
    version: custom-version
)");
  Require(unknown.has_value() && unknown->model_config().models.size() == 1 &&
              unknown->model_config().models[0].task == "custom-task" &&
              unknown->model_config().yolo.empty() &&
              unknown->model_config().ocr.empty(),
          "configuration identity validation must not create a task registry");
}
void TestOCRDetectionConfiguration(TemporaryConfig& temporary) {
  const auto load_leaf = [&](std::string_view leaf) {
    return temporary.Load(std::string("models:\n  - task: ocr\n    name: tuned\n    ocr_detection: ") +
                          std::string(leaf) + "\n");
  };
  for (const auto leaf : {"{}", "{kernel_size: 2, dilation_iterations: 3, min_box_area: 64}"}) {
    const auto loaded = load_leaf(leaf);
    Require(loaded && loaded->model_config().models[0].ocr_detection ==
                          std::optional{OCRDetectionOptions{}},
            "empty and explicit default mappings select legacy geometry");
  }
  const auto omitted = temporary.Load("models:\n  - task: ocr\n    name: plain\n");
  const auto null_leaf = load_leaf("null");
  Require(omitted && null_leaf &&
              !omitted->model_config().models[0].ocr_detection &&
              !null_leaf->model_config().models[0].ocr_detection,
          "omitted and null detection options stay disengaged");
  for (const auto leaf : {"{kernel_size: 1, dilation_iterations: 0, min_box_area: 0}",
                          "{kernel_size: 32, dilation_iterations: 8, min_box_area: 1048576}",
                          "{kernel_size: '1', dilation_iterations: \"1\", min_box_area: '16'}"}) {
    const auto loaded = load_leaf(leaf);
    Require(loaded && loaded->model_config().models[0].ocr_detection &&
                loaded->model_config().models[0].ocr_detection->IsValid(),
            "inclusive ranges and quoted decimal scalars load");
  }
  for (const auto leaf : {"{kernel_size: 0}", "{kernel_size: 33}",
                          "{dilation_iterations: -1}", "{dilation_iterations: 9}",
                          "{min_box_area: -1}", "{min_box_area: 1048577}",
                          "{kernel_size: 999999999999999999999999}",
                          "{kernel_size: true}", "{kernel_size: 1.0}",
                          "{kernel_size: '1x'}", "{kernel_size: ''}",
                          "{kernel_size: '+1'}", "{kernel_size: ' 1'}",
                          "{kernel_size: 0x10}", "{unknown: 1}",
                          "{kernel_size: {nested: 1}}", "{kernel_size: [1]}",
                          "{kernel_size: null}", "[]", "true", "1"}) {
    const auto invalid = load_leaf(leaf);
    Require(!invalid && invalid.error().code == VisionSimpleErrorCode::kRuntimeError,
            "invalid detection mappings must fail at checked load boundary");
  }
  for (const auto task : {"yolo", "seg", "pose", "obb", "custom"}) {
    const auto invalid = temporary.Load(std::string("models:\n  - task: ") + task +
        "\n    name: guarded\n    ocr_detection: {}\n");
    Require(!invalid, "engaged detection options require an OCR task");
  }
  Require(!temporary.Load("yolo:\n  - name: guarded\n    ocr_detection: {}\n"),
          "legacy YOLO must not silently drop an engaged detection mapping");
  Require(temporary.Load("yolo:\n  - name: allowed\n    ocr_detection: null\n").has_value(),
          "null detection mapping remains omitted on non-OCR tasks");
  const auto tuned = temporary.Load(R"(ocr:
  - name: tuned
    version: kPPOCRv4
    det_path: det.onnx
    rec_path: rec.onnx
    char_dict_path: dictionary.txt
    ocr_detection: {kernel_size: 1, dilation_iterations: 1, min_box_area: 16}
)");
  if (!Require(tuned.has_value(), "legacy OCR customization loads")) return;
  const OCRDetectionOptions custom{1, 1, 16};
  const auto& catalog = tuned->model_config();
  Require(catalog.models[0].ocr_detection == std::optional{custom} &&
              catalog.ocr[0].ocr_detection == std::optional{custom},
          "legacy import and projection preserve all detection controls");
  const Config reconstructed{catalog};
  Require(reconstructed.model_config().models.size() == 1 &&
              reconstructed.model_config().models[0].ocr_detection == std::optional{custom} &&
              reconstructed.model_config().models[0].files == catalog.models[0].files,
          "reconstruction recognizes customized projections without duplication");
  auto changed = catalog;
  changed.ocr[0].ocr_detection = OCRDetectionOptions{3, 2, 100};
  const Config distinct{std::move(changed)};
  Require(distinct.model_config().models.size() == 2 &&
              distinct.model_config().models[0].ocr_detection == std::optional{custom} &&
              distinct.model_config().models[1].ocr_detection ==
                  std::optional{OCRDetectionOptions{3, 2, 100}},
          "different options are not mistaken for an existing projection");
  Require(omitted && !omitted->model_config().models[0].ocr_detection &&
              tuned->model_config().models[0].ocr_detection == std::optional{custom},
          "independent configurations do not share mutable detection state");
  const auto backend_policy = temporary.Load(R"(unrelated: ignored
models:
  - task: ocr
    name: policy
    extra: ignored
    ocr_detection: {kernel_size: 3, kernel_size: 1}
)");
  Require(backend_policy && backend_policy->model_config().models[0].ocr_detection ==
                                std::optional{OCRDetectionOptions{1, 3, 64}},
          "outer unknown keys and duplicate-last YAML map policy remain unchanged");
  const auto independent = temporary.Load(R"(models:
  - task: ocr
    name: first
    ocr_detection:
      kernel_size: 1
      dilation_iterations: 1
      min_box_area: 16
  - task: ocr
    name: second
    ocr_detection:
      kernel_size: 4
      dilation_iterations: 0
      min_box_area: 100
)");
  if (Require(independent.has_value(), "block mappings support distinct OCR models")) {
    const Config copied{independent->model_config()};
    const auto& models = copied.model_config().models;
    Require(models.size() == 2 &&
                models[0].ocr_detection == std::optional{custom} &&
                models[1].ocr_detection == std::optional{OCRDetectionOptions{4, 0, 100}},
            "canonical options remain isolated by model through reconstruction");
    Require(copied.model_config().ocr.size() == 2 &&
                copied.model_config().ocr[0].ocr_detection == models[0].ocr_detection &&
                copied.model_config().ocr[1].ocr_detection == models[1].ocr_detection,
            "multiple OCR projections preserve each model's own options");
  }
}
}  // namespace

int main() {
  try {
    TemporaryConfig temporary;
    if (!temporary.valid()) return 1;
    TestEquivalentCatalogs(temporary);
    TestMixedAndInvalidCatalogs(temporary);
    TestOCRDetectionConfiguration(temporary);
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "configuration test failed: " << error.what() << '\n';
    return 1;
  }
}
