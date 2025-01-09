#include <codecvt>
#include <stdio.h>
#include "Util.hpp"
#include <format>
#include <magic_enum.hpp>
#define CHECK_RESULT(result) do{\
     if (!(result)) \
     { \
         auto &(err) = (result).error(); \
         std::cout << std::format("fail code:{} message:{}", magic_enum::enum_name((err).code), (err).message) << std::endl; \
         return -1; \
     }}while(0)
using namespace vision_simple;

int main()
{
    SetConsoleOutputCP(CP_UTF8);
    auto char_dict_data = ReadAll("assets/ppocr_keys_v1.txt");
    CHECK_RESULT(char_dict_data);
    std::string str{reinterpret_cast<const char*>(char_dict_data->data.get()), char_dict_data->size};
    std::string wstr{reinterpret_cast<const char*>(char_dict_data->data.get()), char_dict_data->size};
    std::vector<std::string> lines;
    std::vector<std::wstring> wlines;
    std::stringstream ss(str);
    std::string line;
    // 按行读取，直到文件流结束
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    while (std::getline(ss, line))
    {
        lines.push_back(line.substr(0, line.size() - 1));
        // std::cout << line << "\n";
    }
    std::cout << std::endl;
    std::map<int, std::string> char_dict;
    for (auto [idx,c] : std::views::enumerate(lines))
    {
        // char_dict.insert(std::pair<int, std::string>(idx, c));
        char_dict.emplace(idx, c);
    }
    auto det_data = ReadAll("assets/ppocr_det.onnx");
    CHECK_RESULT(det_data);
    auto rec_data = ReadAll("assets/ppocr_rec.onnx");
    CHECK_RESULT(rec_data);
    auto infer_ctx = vision_simple::InferContext::Create(vision_simple::InferFramework::kONNXRUNTIME,
                                                         vision_simple::InferEP::kDML);
    CHECK_RESULT(infer_ctx);
    auto infer_ocr = vision_simple::InferOCR::Create(**infer_ctx, char_dict,
                                                     det_data->span(), rec_data->span(),
                                                     vision_simple::OCRModelType::kPPOCRv4);
    CHECK_RESULT(infer_ocr);
    auto image{cv::imread((const char*)"assets/hd2.png")};
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    auto result = (*infer_ocr)->Run(image, 0.5f);
    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = std::chrono::duration_cast<std::chrono::duration<double>>(stop - begin);
    std::cout << diff.count() << std::endl;
    CHECK_RESULT(result);
    auto& results = result->results;
    for (const auto& ocr_result : results)
    {
        auto msg = std::format("----x:{} y:{} w:{} h:{} conf:{}\n----{}", ocr_result.rect.x, ocr_result.rect.y,
                               ocr_result.rect.width, ocr_result.rect.height, ocr_result.confidence,
                               ocr_result.line);
        puts(msg.c_str());
    }
}
