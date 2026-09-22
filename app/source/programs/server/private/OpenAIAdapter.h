#pragma once
#include <hv/HttpServer.h>

#include "InferenceService.h"

namespace vision_simple {
void RegisterOpenAI(hv::HttpService& http,
                    std::shared_ptr<InferenceService> service);
}  // namespace vision_simple
