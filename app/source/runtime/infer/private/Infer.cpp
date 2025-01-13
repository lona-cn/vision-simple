#include "Infer.h"

#include <magic_enum.hpp>
#include <memory>
#include <ranges>

#include "private/InferORT.h"
using namespace std;
using namespace cv;
using namespace vision_simple;

#define UNSUPPORTED(framework, ep)                           \
  std::unexpected {                                          \
    VisionSimpleError {                                      \
      VisionSimpleErrorCode::kParameterError,                \
          std::format("unsupported framework({}) or ep({})", \
                      magic_enum::enum_name((framework)),    \
                      magic_enum::enum_name((ep)))           \
    }                                                        \
  }

namespace vision_simple {
namespace {
const std::map<InferFramework, std::vector<InferEP>> supported_framework_eps = {
    std::pair{InferFramework::kONNXRUNTIME, std::vector{
                                                InferEP::kCPU,
                                                InferEP::kDML,
                                                InferEP::kCUDA,
                                                InferEP::kTensorRT,
                                            }}};

bool IsSupported(const InferFramework framework, const InferEP ep) {
  try {
    auto vec = supported_framework_eps.at(framework);
    return ranges::find(vec, ep) != vec.end();
  } catch (std::exception& _) {
    return false;
  }
}
}  // namespace

InferContext::InferContext(InferFramework framework, InferEP ep, InferArgs args)
    : framework_(framework), ep_(ep), args_(std::move(args)) {}

InferFramework InferContext::framework() const noexcept { return framework_; }

InferEP InferContext::execution_provider() const noexcept { return ep_; }

const InferArgs& InferContext::args() const noexcept { return args_; }

InferContext::CreateResult InferContext::Create(const InferFramework framework,
                                                const InferEP ep,
                                                InferArgs args) noexcept {
  if (!IsSupported(framework, ep)) return UNSUPPORTED(framework, ep);
  switch (framework) {
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

InferYOLO::CreateResult InferYOLO::Create(InferContext& context,
                                          const std::string& path,
                                          YOLOVersion version,
                                          size_t device_id) noexcept {
  auto data_result = ReadAll(path);
  if (!data_result) return std::unexpected(std::move(data_result.error()));
  return Create(context, data_result->span(), version, device_id);
}

InferOCR::CreateResult InferOCR::Create(InferContext& context,
                                        const std::string& char_dict_path,
                                        const std::string& det_path,
                                        const std::string& rec_path,
                                        OCRModelType model_type,
                                        size_t device_id) noexcept {
  auto char_dict_result = ReadAllLines(char_dict_path);
  if (!char_dict_result)
    return std::unexpected(std::move(char_dict_result.error()));
  auto det_data_result = ReadAll(det_path);
  if (!det_data_result)
    return std::unexpected(std::move(det_data_result.error()));
  auto rec_data_rect = ReadAll(rec_path);
  if (!rec_data_rect) return std::unexpected(std::move(rec_data_rect.error()));
  std::map<int, std::string> char_dict;
  for (auto [idx, c] : std::views::enumerate(*char_dict_result))
    char_dict.emplace(idx, c);
  return Create(context, char_dict, det_data_result->span(),
                rec_data_rect->span(), model_type, device_id);
}

// InferOCR::DetResult InferOCR::Det(const cv::Mat* images, size_t count)
// noexcept
// {
//     std::vector<std::reference_wrapper<const cv::Mat>> images_vec;
//     images_vec.reserve(count);
//     for (decltype(count) i = 0u; i < count; ++i)
//     {
//         images_vec.emplace_back(std::cref(images[i]));
//     }
//     return Det(images_vec);
// }
//
// InferOCR::RecResult InferOCR::Rec(const cv::Mat* images, size_t count, float
// confidence_threshold) noexcept
// {
//     std::vector<std::reference_wrapper<const cv::Mat>> images_vec;
//     images_vec.reserve(count);
//     for (decltype(count) i = 0u; i < count; ++i)
//     {
//         images_vec.emplace_back(std::cref(images[i]));
//     }
//     return Rec(images_vec, confidence_threshold);
// }
}  // namespace vision_simple
