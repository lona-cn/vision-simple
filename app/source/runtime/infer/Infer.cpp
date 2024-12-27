#include "Infer.h"
#include <magic_enum.hpp>
#include <memory>

#include "private/InferORT.h"
using namespace std;
using namespace cv;
using namespace vision_simple;

namespace vision_simple
{
    InferContext::CreateResult InferContext::Create(InferFramework framework, InferEP ep) noexcept
    {
        if (framework == InferFramework::kONNXRUNTIME && (ep == InferEP::kCPU || ep == InferEP::kDML || ep ==
            InferEP::kCUDA))
            return std::make_unique<InferContextONNXRuntime>(ep);
        return std::unexpected{
            InferError{
                InferErrorCode::kParameterError, std::format("unsupported framework({}) or ep({})",
                                                             magic_enum::enum_name(framework),
                                                             magic_enum::enum_name(ep))
            }
        };
    }
}
