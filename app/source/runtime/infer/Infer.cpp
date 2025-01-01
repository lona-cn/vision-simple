#include "Infer.h"
#include <magic_enum.hpp>
#include <memory>

#include "private/InferORT.h"
using namespace std;
using namespace cv;
using namespace vision_simple;

#define UNSUPPORTED(framework,ep) \
std::unexpected{ \
InferError{ \
    InferErrorCode::kParameterError, std::format("unsupported framework({}) or ep({})", \
                                                 magic_enum::enum_name((framework)), \
                                                 magic_enum::enum_name((ep))) \
} \
}

namespace vision_simple
{
    namespace
    {
        const std::map<InferFramework, std::vector<InferEP>> supported_framework_eps = {
            std::pair{
                InferFramework::kONNXRUNTIME,
                std::vector{InferEP::kCPU, InferEP::kDML, InferEP::kCUDA, InferEP::kTensorRT,}
            }
        };

        bool IsSupported(const InferFramework framework, const InferEP ep)
        {
            try
            {
                auto vec = supported_framework_eps.at(framework);
                return ranges::find(vec, ep) != vec.end();
            }
            catch (std::exception& _)
            {
                return false;
            }
        }
    }

    InferContext::InferContext(InferFramework framework, InferEP ep, InferArgs args): framework_(framework),
        ep_(ep), args_(std::move(args))
    {
    }

    InferFramework InferContext::framework() const noexcept
    {
        return framework_;
    }

    InferEP InferContext::execution_provider() const noexcept
    {
        return ep_;
    }

    const InferArgs& InferContext::args() const noexcept
    {
        return args_;
    }

    InferContext::CreateResult InferContext::Create(const InferFramework framework, const InferEP ep,
                                                    InferArgs args) noexcept
    {
        if (!IsSupported(framework, ep))return UNSUPPORTED(framework, ep);
        switch (framework)
        {
        case InferFramework::kCUSTOM_FRAMEWORK:
            return UNSUPPORTED(framework, ep);
        case InferFramework::kONNXRUNTIME:
            return std::make_unique<InferContextORT>(ep, std::move(args));
        case InferFramework::kTVM:
            return UNSUPPORTED(framework, ep);
        default:
            return UNSUPPORTED(framework, ep);
        }
    }
}
