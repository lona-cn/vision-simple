#include "VisionSimpleError.h"

#include <memory_resource>

vision_simple::VisionSimpleError::VisionSimpleError(
    VisionSimpleErrorCode code, const std::string& message,
    std::unique_ptr<void*> user_data)
    : code(code),
      user_data(std::move(user_data)),
      message{message, std::pmr::new_delete_resource()} {}

vision_simple::VisionSimpleError::VisionSimpleError(
    VisionSimpleErrorCode code, const char* message,
    std::unique_ptr<void*> user_data)
    : code(code),
      user_data(std::move(user_data)),
      message{message, std::pmr::new_delete_resource()} {}

vision_simple::VisionSimpleError::VisionSimpleError(
    VisionSimpleErrorCode code, std::pmr::string message,
    std::unique_ptr<void*> user_data)
    : code(code),
      user_data(std::move(user_data)),
      message(std::move(message)) {}

vision_simple::VisionSimpleError::operator bool() const noexcept {
  return code == VisionSimpleErrorCode::kOK;
}

vision_simple::VisionSimpleError vision_simple::VisionSimpleError::Ok(
    std::string msg) noexcept {
  return VisionSimpleError{VisionSimpleErrorCode::kOK, msg};
}

vision_simple::VisionSimpleError
vision_simple::VisionSimpleError::Unimplemented(
    const std::source_location& location) noexcept {
  std::string msg =
      std::format("unimplemented function:{} {} {} {}", location.file_name(),
                  location.line(), location.column(), location.function_name());
  return VisionSimpleError{VisionSimpleErrorCode::kUnimplementedError, msg};
}
