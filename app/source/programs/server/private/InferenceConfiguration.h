#pragma once
#include "InferenceService.h"
namespace vision_simple {
struct HTTPServerOptions;
// Shared normalization only; listener, management authority and logging stay HTTP-owned.
VSResult<InferenceServiceOptions> ParseInferenceServiceOptions(HTTPServerOptions& options) noexcept;
}
