#pragma once
#include <expected>

#include "VisionSimpleError.h"

namespace vision_simple
{
    template <typename T>
    using VSResult = std::expected<T, VisionSimpleError>;
}
