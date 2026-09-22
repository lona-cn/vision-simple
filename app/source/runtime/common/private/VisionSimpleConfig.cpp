#include "VisionSimpleConfig.h"

#include <ylt/struct_yaml/yaml_reader.h>

#include <mutex>
#include <set>
#include <shared_mutex>

#include "IOUtil.h"

namespace {
constexpr std::string_view MODEL_CONFIG_PATH = "config/models.yaml";
std::shared_mutex instance_mutex;
std::unique_ptr<vision_simple::Config> config_instance{nullptr};
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
              files) {
        for (size_t i = 0; i < canonical_count; ++i) {
          const auto& model = models[i];
          if (model.task != task || model.name != name ||
              model.version != version)
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
                       {"dictionary", legacy.char_dict_path}}))
      continue;
    models.push_back({"ocr",
                      std::move(legacy.name),
                      std::move(legacy.version),
                      {{"det", std::move(legacy.det_path)},
                       {"rec", std::move(legacy.rec_path)},
                       {"dictionary", std::move(legacy.char_dict_path)}}});
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
                                  file("rec"), file("dictionary")});
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
    ModelConfig model_config;
    struct_yaml::from_yaml(model_config, str);
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
