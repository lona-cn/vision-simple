#pragma once
#include <hv/HttpServer.h>

#include "InferenceService.h"

namespace vision_simple {
class HttpDispatch;
void RegisterOpenAI(hv::HttpService& http,
                    std::shared_ptr<InferenceService> service,
                    HttpDispatch& dispatch);
}  // namespace vision_simple
