#include "IOUtil.h"

#include <filesystem>

std::expected<vision_simple::DataBuffer<unsigned char>,
              vision_simple::VisionSimpleError>
vision_simple::ReadAll(const std::string& path) noexcept {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    return std::unexpected(
        VisionSimpleError{VisionSimpleErrorCode::kIOError,
                          std::format("unable to open file '{}'", path)});
  }
  const size_t size = ifs.tellg();
  if (size <= 0) {
    return std::unexpected(
        VisionSimpleError{VisionSimpleErrorCode::kIOError,
                          std::format("file:{} is empty,size:{}", path, size)});
  }
  ifs.seekg(std::ios::beg);
  auto buffer = std::make_unique<uint8_t[]>(size);
  ifs.read(reinterpret_cast<char*>(buffer.get()), static_cast<long long>(size));
  return DataBuffer{std::move(buffer), size};
}

std::expected<std::string, vision_simple::VisionSimpleError>
vision_simple::ReadAllString(const std::string& path) noexcept {
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
}

std::expected<std::vector<std::string>, vision_simple::VisionSimpleError>
vision_simple::ReadAllLines(const std::string& path) noexcept {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    return MK_VSERROR(VisionSimpleErrorCode::kIOError,
                      std::format("Unable to open file:{}", path));
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
  return lines;
}
