#include "IOUtil.h"

#include <filesystem>
#include <limits>
#include <sstream>

std::expected<vision_simple::DataBuffer<unsigned char>,
              vision_simple::VisionSimpleError>
vision_simple::ReadAll(const std::string& path) noexcept {
  try {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
      return MK_VSERROR(VisionSimpleErrorCode::kIOError,
                        std::format("unable to open file '{}'", path));
    }
    const std::streamoff file_size = ifs.tellg();
    if (file_size <= 0) {
      return MK_VSERROR(
          VisionSimpleErrorCode::kIOError,
          std::format("file '{}' is empty or has an invalid size", path));
    }
    if (static_cast<uintmax_t>(file_size) >
            std::numeric_limits<size_t>::max() ||
        file_size > std::numeric_limits<std::streamsize>::max()) {
      return MK_VSERROR(VisionSimpleErrorCode::kIOError,
                        std::format("file '{}' is too large to read", path));
    }
    const auto size = static_cast<size_t>(file_size);
    ifs.seekg(std::ios::beg);
    auto buffer = std::make_unique<uint8_t[]>(size);
    if (!ifs.read(reinterpret_cast<char*>(buffer.get()),
                  static_cast<std::streamsize>(size))) {
      return MK_VSERROR(VisionSimpleErrorCode::kIOError,
                        std::format("unable to read file '{}'", path));
    }
    return DataBuffer{std::move(buffer), size};
  } catch (const std::exception& error) {
    return MK_VSERROR(
        VisionSimpleErrorCode::kIOError,
        std::format("unable to read file '{}': {}", path, error.what()));
  }
}

std::expected<std::string, vision_simple::VisionSimpleError>
vision_simple::ReadAllString(const std::string& path) noexcept {
  try {
    if (!std::filesystem::exists(path)) {
      return std::unexpected(
          VisionSimpleError{VisionSimpleErrorCode::kIOError,
                            std::format("file not exists '{}'", path)});
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) {
      return std::unexpected(
          VisionSimpleError{VisionSimpleErrorCode::kIOError,
                            std::format("unable to open file '{}'", path)});
    }
    std::stringstream file_str;
    std::string line;
    // TODO: optimize
    while (std::getline(file, line)) {
      std::string result;
      for (char c : line) {
        if (c != '\r') result.push_back(c);
      }
      file_str << std::format("{}\n", result);
    }
    return file_str.str();
  } catch (const std::exception& e) {
    return MK_VSERROR(
        VisionSimpleErrorCode::kIOError,
        std::format("unable to read file '{}': {}", path, e.what()));
  }
}

std::expected<std::vector<std::string>, vision_simple::VisionSimpleError>
vision_simple::ReadAllLines(const std::string& path) noexcept {
  try {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
      return MK_VSERROR(VisionSimpleErrorCode::kIOError,
                        std::format("unable to open file '{}'", path));
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line)) {
      std::string result;
      result.reserve(line.size());
      for (char c : line) {
        if (c != '\r') result.push_back(c);
      }
      lines.emplace_back(std::move(result));
    }
    if (ifs.bad()) {
      return MK_VSERROR(VisionSimpleErrorCode::kIOError,
                        std::format("unable to read file '{}'", path));
    }
    return lines;
  } catch (const std::exception& error) {
    return MK_VSERROR(
        VisionSimpleErrorCode::kIOError,
        std::format("unable to read file '{}': {}", path, error.what()));
  }
}
