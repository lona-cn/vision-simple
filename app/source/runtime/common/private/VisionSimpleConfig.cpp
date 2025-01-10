#include "VisionSimpleConfig.h"
#include <mutex>
#include <shared_mutex>
#include <ylt/struct_yaml/yaml_reader.h>

#include "IOUtil.h"

namespace
{
    constexpr std::string_view MODEL_CONFIG_PATH = "config/models.yaml";
    std::shared_mutex instance_mutex;
    std::unique_ptr<vision_simple::Config> config_instance{nullptr};
}

std::expected<vision_simple::Config, vision_simple::VisionSimpleError> vision_simple::Config::Load(
    const ConfigLoadOptions& options) noexcept
{
    std::optional<ModelConfig> model_config_opt;
    auto data_result = ReadAllString(options.model_config_path.data());
    if (!data_result)return std::unexpected(std::move(data_result.error()));
    std::string str{std::move(*data_result)};
    std::error_code error_code{};
    ModelConfig model_config;
    struct_yaml::from_yaml(model_config, str, error_code);
    if (error_code)
        return std::unexpected(VisionSimpleError{
            VisionSimpleErrorCode::kRuntimeError,
            std::format("unable to deserialize yaml file:{},message:{}", options.model_config_path,
                        error_code.message())
        });
    return Config{model_config};
}

const vision_simple::ModelConfig& vision_simple::Config::model_config() const noexcept
{
    return model_config_;
}

std::expected<std::reference_wrapper<const vision_simple::Config>, vision_simple::VisionSimpleError>
vision_simple::Config::Instance() noexcept
{
    {
        std::shared_lock shared_lock(instance_mutex);
        if (config_instance)return *config_instance;
    }
    std::unique_lock lock(instance_mutex);
    if (config_instance)return *config_instance;
    auto cfg_opt = Load(ConfigLoadOptions{MODEL_CONFIG_PATH});
    if (!cfg_opt)return std::unexpected(std::move(cfg_opt.error()));
    config_instance = std::make_unique<Config>(std::move(*cfg_opt));
    return *config_instance;
}
