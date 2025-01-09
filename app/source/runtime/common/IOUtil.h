#pragma once
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <fstream>
#include "VisionSimpleError.h"

namespace vision_simple
{
    template <typename T>
    struct DataBuffer
    {
        std::unique_ptr<T[]> data;
        size_t size;

        size_t size_bytes() const noexcept { return size * sizeof(T); }


        template <typename AS = T>
        std::span<AS> span()
        {
            return std::span{reinterpret_cast<AS*>(data.get()), size * sizeof(T) / sizeof(AS)};
        }
    };

    std::expected<DataBuffer<uint8_t>, VisionSimpleError> ReadAll(const std::string& path) noexcept;
    std::expected<std::string, VisionSimpleError> ReadAllString(const std::string& path) noexcept;
}
