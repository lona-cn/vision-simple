#include "IOUtil.h"
#include <filesystem>

std::expected<vision_simple::DataBuffer<unsigned char>, vision_simple::VisionSimpleError> vision_simple::ReadAll(
    const std::string& path) noexcept
{
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs)
    {
        return std::unexpected(VisionSimpleError{
            VisionSimpleErrorCode::kIOError,
            std::format("unable to open file '{}'", path)
        });
    }
    const size_t size = ifs.tellg();
    if (size <= 0)
    {
        return std::unexpected(VisionSimpleError{
            VisionSimpleErrorCode::kIOError,
            std::format("file:{} is empty,size:{}", path, size)
        });
    }
    ifs.seekg(std::ios::beg);
    auto buffer = std::make_unique<uint8_t[]>(size);
    ifs.read(reinterpret_cast<char*>(buffer.get()),
             static_cast<long long>(size));
    return DataBuffer{std::move(buffer), size};
}

std::expected<std::string, vision_simple::VisionSimpleError> vision_simple::ReadAllString(
    const std::string& path) noexcept
{
    std::ifstream file(path);
    if (!file)
    {
        return std::unexpected(VisionSimpleError{
            VisionSimpleErrorCode::kIOError,
            std::format("Unable to open file:{}", path)
        });
    }

    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
}

std::expected<std::vector<std::string>, vision_simple::VisionSimpleError> vision_simple::ReadAllLines(
    const std::string& path) noexcept
{
    std::vector<std::string> lines;
    std::ifstream ifs(path);
    if (!ifs)
    {
        return MK_VSERROR(
            VisionSimpleErrorCode::kIOError,
            std::format("Unable to open file:{}", path));
    }
    std::string line;
    while (std::getline(ifs, line))
    {
        lines.push_back(line.substr(0, line.size()));
    }
    return lines;
}
