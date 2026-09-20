#include "VisionSimpleConfig.h"

#include <ylt/struct_yaml/yaml_reader.h>

#include <mutex>
#include <shared_mutex>

#include "IOUtil.h"

namespace {
constexpr std::string_view MODEL_CONFIG_PATH = "config/models.yaml";
std::shared_mutex instance_mutex;
std::unique_ptr<vision_simple::Config> config_instance{nullptr};
}  // namespace

std::expected<vision_simple::Config, vision_simple::VisionSimpleError>
vision_simple::Config::Load(const ConfigLoadOptions& options) noexcept {
  try {
    auto data_result = ReadAllString(std::string(options.model_config_path));
    if (!data_result) return std::unexpected(std::move(data_result.error()));
    std::string str{std::move(*data_result)};
    ModelConfig model_config;
    struct_yaml::from_yaml(model_config, str);
    return Config{model_config};
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
