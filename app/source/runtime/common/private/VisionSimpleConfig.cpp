#include "VisionSimpleConfig.h"

#include <ylt/struct_yaml/yaml_reader.h>

#include <charconv>
#include <mutex>
#include <stdexcept>
#include <set>
#include <shared_mutex>

#include "IOUtil.h"

namespace {
constexpr std::string_view MODEL_CONFIG_PATH = "config/models.yaml";
std::shared_mutex instance_mutex;
std::unique_ptr<vision_simple::Config> config_instance{nullptr};
// Checked input DTOs keep the YAML backend's outer-key policy unchanged.
using RawDetection = std::optional<std::map<std::string, std::string>>;
struct RawYOLOModelInfo {
  std::string name, version;
  std::string path;
  RawDetection ocr_detection;
};
struct RawOCRModelInfo {
  std::string name, version;
  std::string det_path, rec_path, char_dict_path;
  RawDetection ocr_detection;
};
struct RawModelDefinition {
  std::string task, name, version;
  std::map<std::string, std::string> files;
  RawDetection ocr_detection;
};
struct RawModelConfig {
  std::vector<RawYOLOModelInfo> yolo;
  std::vector<RawOCRModelInfo> ocr;
  std::vector<RawModelDefinition> models;
};

std::optional<vision_simple::OCRDetectionOptions> NormalizeDetection(
    const RawDetection& raw, std::string_view task) {
  if (!raw) return std::nullopt;
  if (task != "ocr")
    throw std::invalid_argument("ocr_detection requires an ocr model");
  vision_simple::OCRDetectionOptions options;
  for (const auto& [key, text] : *raw) {
    int* target = nullptr;
    if (key == "kernel_size") target = &options.kernel_size;
    else if (key == "dilation_iterations") target = &options.dilation_iterations;
    else if (key == "min_box_area") target = &options.min_box_area;
    else throw std::invalid_argument("unknown ocr_detection key");
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("ocr_detection requires decimal integers");
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), *target);
    if (error != std::errc{} || end != text.data() + text.size())
      throw std::invalid_argument("ocr_detection requires decimal integers");
  }
  if (!options.IsValid())
    throw std::invalid_argument("ocr_detection value is outside its range");
  return options;
}

vision_simple::ModelConfig NormalizeConfig(RawModelConfig raw) {
  vision_simple::ModelConfig result;
  result.models.reserve(raw.models.size());
  result.yolo.reserve(raw.yolo.size());
  result.ocr.reserve(raw.ocr.size());
  for (auto& model : raw.models) {
    auto options = NormalizeDetection(model.ocr_detection, model.task);
    result.models.push_back({std::move(model.task), std::move(model.name),
                             std::move(model.version), std::move(model.files),
                             std::move(options)});
  }
  for (auto& model : raw.yolo) {
    NormalizeDetection(model.ocr_detection, "yolo");
    result.yolo.push_back({std::move(model.name), std::move(model.version),
                           std::move(model.path)});
  }
  for (auto& model : raw.ocr) {
    auto options = NormalizeDetection(model.ocr_detection, "ocr");
    result.ocr.push_back({std::move(model.name), std::move(model.version),
                          std::move(model.det_path), std::move(model.rec_path),
                          std::move(model.char_dict_path), std::move(options)});
  }
  return result;
}
}  // namespace

vision_simple::Config::Config(ModelConfig model_config) {
  auto& models = model_config.models;
  models.reserve(models.size() + model_config.yolo.size() +
                 model_config.ocr.size());
  const auto canonical_count = models.size();
  const auto is_projection =
      [&](std::string_view task, std::string_view name,
          std::string_view version,
          std::initializer_list<std::pair<const char*, std::string_view>>
              files,
          const std::optional<OCRDetectionOptions>& detection = std::nullopt) {
        for (size_t i = 0; i < canonical_count; ++i) {
          const auto& model = models[i];
          if (model.task != task || model.name != name ||
              model.version != version || model.ocr_detection != detection)
            continue;
          bool equal = true;
          for (const auto& [role, value] : files) {
            const auto found = model.files.find(role);
            if (found == model.files.end() ? !value.empty()
                                           : found->second != value) {
              equal = false;
              break;
            }
          }
          if (equal) return true;
        }
        return false;
      };
  for (auto& legacy : model_config.yolo) {
    if (is_projection("yolo", legacy.name, legacy.version,
                      {{"model", legacy.path}}))
      continue;
    models.push_back({"yolo",
                      std::move(legacy.name),
                      std::move(legacy.version),
                      {{"model", std::move(legacy.path)}}});
  }
  for (auto& legacy : model_config.ocr) {
    if (is_projection("ocr", legacy.name, legacy.version,
                      {{"det", legacy.det_path},
                       {"rec", legacy.rec_path},
                       {"dictionary", legacy.char_dict_path}},
                      legacy.ocr_detection))
      continue;
    models.push_back({"ocr",
                      std::move(legacy.name),
                      std::move(legacy.version),
                      {{"det", std::move(legacy.det_path)},
                       {"rec", std::move(legacy.rec_path)},
                       {"dictionary", std::move(legacy.char_dict_path)}},
                      std::move(legacy.ocr_detection)});
  }
  model_config.yolo.clear();
  model_config.ocr.clear();
  for (const auto& model : models) {
    const auto file = [&model](const char* key) -> std::string {
      const auto found = model.files.find(key);
      return found == model.files.end() ? std::string{} : found->second;
    };
    if (model.task == "yolo") {
      model_config.yolo.push_back({model.name, model.version, file("model")});
    } else if (model.task == "ocr") {
      model_config.ocr.push_back({model.name, model.version, file("det"),
                                  file("rec"), file("dictionary"),
                                  model.ocr_detection});
    }
  }
  model_config_ = std::move(model_config);
}

std::expected<vision_simple::Config, vision_simple::VisionSimpleError>
vision_simple::Config::Load(const ConfigLoadOptions& options) noexcept {
  try {
    auto data_result = ReadAllString(std::string(options.model_config_path));
    if (!data_result) return std::unexpected(std::move(data_result.error()));
    std::string str{std::move(*data_result)};
    RawModelConfig raw;
    struct_yaml::from_yaml(raw, str);
    auto model_config = NormalizeConfig(std::move(raw));
    std::set<std::pair<std::string_view, std::string_view>> identities;
    const auto check_identity =
        [&](std::string_view task,
            std::string_view name) -> std::expected<void, VisionSimpleError> {
      if (task.empty() || name.empty()) {
        return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError,
                          "model task and name must be nonempty");
      }
      if (!identities.emplace(task, name).second) {
        return MK_VSERROR(
            VisionSimpleErrorCode::kRuntimeError,
            std::format("duplicate model declaration: {}/{}", task, name));
      }
      return {};
    };
    for (const auto& model : model_config.models) {
      auto valid = check_identity(model.task, model.name);
      if (!valid) return std::unexpected(std::move(valid.error()));
    }
    for (const auto& model : model_config.yolo) {
      auto valid = check_identity("yolo", model.name);
      if (!valid) return std::unexpected(std::move(valid.error()));
    }
    for (const auto& model : model_config.ocr) {
      auto valid = check_identity("ocr", model.name);
      if (!valid) return std::unexpected(std::move(valid.error()));
    }
    return Config{std::move(model_config)};
  } catch (const std::exception& e) {
    return MK_VSERROR(
        VisionSimpleErrorCode::kRuntimeError,
        std::format("unable to load model configuration: {}", e.what()));
  }
}

const vision_simple::ModelConfig& vision_simple::Config::model_config()
    const noexcept {
  return model_config_;
}

std::expected<std::reference_wrapper<const vision_simple::Config>,
              vision_simple::VisionSimpleError>
vision_simple::Config::Instance() noexcept {
  try {
    {
      std::shared_lock shared_lock(instance_mutex);
      if (config_instance) return *config_instance;
    }
    std::unique_lock lock(instance_mutex);
    if (config_instance) return *config_instance;
    auto cfg_opt = Load(ConfigLoadOptions{MODEL_CONFIG_PATH});
    if (!cfg_opt) return std::unexpected(std::move(cfg_opt.error()));
    config_instance = std::make_unique<Config>(std::move(*cfg_opt));
    return *config_instance;
  } catch (const std::exception& e) {
    return MK_VSERROR(
        VisionSimpleErrorCode::kRuntimeError,
        std::format("unable to initialize configuration: {}", e.what()));
  }
}
