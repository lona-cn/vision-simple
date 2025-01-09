#pragma once
#include <expected>
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include "config.h"
#include "VisionSimpleError.h"

namespace vision_simple
{
    struct YOLOModelInfo
    {
        std::string name, version;
        std::string path;
    };

    struct OCRModelInfo
    {
        std::string name, version;
        std::string det_path, rec_path, char_dict_path;
    };

    struct ModelConfig
    {
        std::vector<YOLOModelInfo> yolo;
        std::vector<OCRModelInfo> ocr;
    };

    struct ConfigLoadOptions
    {
        std::string_view model_config_path;
    };

    class VISION_SIMPLE_API Config
    {
        ModelConfig model_config_;

        static std::expected<Config, VisionSimpleError> Load(const ConfigLoadOptions& options) noexcept;

    public:
        explicit Config(ModelConfig model_config)
            : model_config_(std::move(model_config))
        {
        }

        static std::expected<std::reference_wrapper<const Config>, VisionSimpleError> Instance() noexcept;

        const ModelConfig& model_config() const noexcept;
    };
}
