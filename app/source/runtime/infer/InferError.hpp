#pragma once
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <source_location>

enum class InferErrorCode:uint8_t
{
    kOK = 0,
    kUnimplementedError,
    kIOError,
    kDeviceError,
    kModelError,
    kParameterError,
    kRangeError,
    kRuntimeError,
    kCustomError,
    kUnknownError,
};

class InferError
{
public:
    InferErrorCode code;
    std::unique_ptr<void*> user_data;
    std::string message; //TODO: 考虑试试PMR版本？

    InferError(InferErrorCode code, std::string message, std::unique_ptr<void*> user_data = nullptr)
        : code(code),
          user_data(std::move(user_data)),
          message(std::move(message))

    {
    }

    static InferError Ok(std::string msg = "ok") noexcept
    {
        return InferError{InferErrorCode::kOK, std::move(msg)};
    }

    static InferError Unimplemented(const std::source_location& location = std::source_location::current()) noexcept
    {
        std::string msg = std::format("unimplemented function:{} {} {} {}",
                                      location.file_name(), location.line(),
                                      location.column(), location.function_name());
        return InferError{InferErrorCode::kUnimplementedError, std::move(msg)};
    }
};
