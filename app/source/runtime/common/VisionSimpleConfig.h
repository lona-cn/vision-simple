#pragma once
#include <expected>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "VisionSimpleError.h"
#include "config.h"

namespace vision_simple {
struct YOLOModelInfo {
  std::string name, version;
  std::string path;
};

struct OCRModelInfo {
  std::string name, version;
  std::string det_path, rec_path, char_dict_path;
};

struct ModelDefinition {
  std::string task, name, version;
  std::map<std::string, std::string> files;
};

struct ModelConfig {
  std::vector<YOLOModelInfo> yolo;
  std::vector<OCRModelInfo> ocr;
  // Canonical execution catalog; yolo/ocr are compatibility projections
  // after construction. Before construction all three are input lists.
  std::vector<ModelDefinition> models;
};

struct ConfigLoadOptions {
  std::string_view model_config_path;
};

class VISION_SIMPLE_API Config {
  ModelConfig model_config_;

 public:
  static std::expected<Config, VisionSimpleError> Load(
      const ConfigLoadOptions& options) noexcept;

  // Trusted DTO boundary: normalizes input lists and recognizes existing
  // compatibility projections. Use Load for checked YAML input.
  explicit Config(ModelConfig model_config);

  static std::expected<std::reference_wrapper<const Config>, VisionSimpleError>
  Instance() noexcept;

  const ModelConfig& model_config() const noexcept;
};
}  // namespace vision_simple
