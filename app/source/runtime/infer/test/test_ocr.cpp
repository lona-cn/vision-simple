#include <codecvt>
#include <cstdio>
#include <format>
#include <magic_enum.hpp>
#include <ranges>

#include "Util.hpp"
#define CHECK_RESULT(result)                                      \
  do {                                                            \
    if (!(result)) {                                              \
      auto&(err) = (result).error();                              \
      std::cout << std::format("fail code:{} message:{}",         \
                               magic_enum::enum_name((err).code), \
                               (err).message)                     \
                << std::endl;                                     \
      return -1;                                                  \
    }                                                             \
  } while (0)
using namespace vision_simple;

int main() {
  SetConsoleOutputCP(CP_UTF8);
  auto infer_ctx = vision_simple::InferContext::Create(
      vision_simple::InferFramework::kONNXRUNTIME,
      vision_simple::InferEP::kDML);
  CHECK_RESULT(infer_ctx);
  auto infer_ocr = vision_simple::InferOCR::Create(
      **infer_ctx, "assets/ppocr_keys_v1.txt", "assets/ppocr_det.onnx",
      "assets/ppocr_rec.onnx", vision_simple::OCRModelType::kPPOCRv4);
  CHECK_RESULT(infer_ocr);
  auto image{cv::imread((const char*)"assets/hd2.png")};
  std::chrono::high_resolution_clock::time_point begin =
      std::chrono::high_resolution_clock::now();
  auto result = (*infer_ocr)->Run(image, 0.5f);
  std::chrono::high_resolution_clock::time_point stop =
      std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff =
      std::chrono::duration_cast<std::chrono::duration<double>>(stop - begin);
  std::cout << diff.count() << std::endl;
  CHECK_RESULT(result);
  auto& results = result->results;
  for (const auto& ocr_result : results) {
    auto msg = std::format("----x:{} y:{} w:{} h:{} conf:{}\n----{}",
                           ocr_result.rect.x, ocr_result.rect.y,
                           ocr_result.rect.width, ocr_result.rect.height,
                           ocr_result.confidence, ocr_result.line);
    puts(msg.c_str());
  }
}
