#pragma once
#include <algorithm>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <string>
#include <source_location>
#include "config.h"

namespace vision_simple
{
    enum class VisionSimpleErrorCode:uint8_t
    {
        kOK = 0,
        kUnimplementedError,
        kCustomError,
        kUnknownError,
        kRuntimeError,
        kRangeError,
        kParameterError,
        kIOError,
        kDeviceError,
        kModelError,
    };

    class VISION_SIMPLE_API VisionSimpleError
    {
    public:
        VisionSimpleErrorCode code;
        std::unique_ptr<void*> user_data;
        std::pmr::string message;
        VisionSimpleError(VisionSimpleErrorCode code, const std::string& message,
                          std::unique_ptr<void*> user_data = nullptr);
        VisionSimpleError(VisionSimpleErrorCode code, const char* message, std::unique_ptr<void*> user_data = nullptr);
        VisionSimpleError(VisionSimpleErrorCode code, std::pmr::string message,
                          std::unique_ptr<void*> user_data = nullptr);
        explicit operator bool() const noexcept;

        static VisionSimpleError Ok(std::string msg = "ok") noexcept;

        static VisionSimpleError Unimplemented(
            const std::source_location& location = std::source_location::current()) noexcept;
    };

#define MK_VSERROR(code,message) \
    std::unexpected{ \
        VisionSimpleError{ \
            (code), \
            (message) \
        }\
    }
}
