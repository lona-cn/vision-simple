#include <fstream>
#include <Infer.h>

using namespace vision_simple;

template <typename T>
struct DataBuffer
{
    std::unique_ptr<T[]> data;
    size_t size;

    std::span<T> span()
    {
        return std::span{data.get(), size};
    }
};

std::expected<DataBuffer<uint8_t>, InferError> ReadAll(
    const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs)
    {
        std::unexpected(InferError{
            InferErrorCode::kIOError,
            std::format("unable to open file '{}'", path)
        });
    }
    const size_t size = ifs.tellg();
    if (size <= 0)
    {
        std::unexpected(InferError{
            InferErrorCode::kIOError,
            std::format("file:{} is empty,size:{}", path, size)
        });
    }
    ifs.seekg(std::ios::beg);
    auto buffer = std::make_unique<uint8_t[]>(size);
    ifs.read(reinterpret_cast<char*>(buffer.get()),
             static_cast<long long>(size));
    return DataBuffer{std::move(buffer), size};
}

void drawYOLOResults(cv::Mat& image, const std::vector<YOLOResult>& results)
{
    for (const auto& result : results)
    {
        // 绘制边界框
        cv::rectangle(image, result.bbox, cv::Scalar(0, 255, 0), 2);

        // 设置标签内容
        std::string label = result.class_name.data() + (" (" + std::to_string(result.confidence * 100) + "%)");

        // 计算文本背景框大小
        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        // 确定文本位置
        int top = std::max(result.bbox.y, labelSize.height);
        cv::Point labelOrigin(result.bbox.x, top);

        // 绘制文本背景框
        cv::rectangle(image, labelOrigin + cv::Point(0, baseLine),
                      labelOrigin + cv::Point(labelSize.width, -labelSize.height), cv::Scalar(0, 255, 0), cv::FILLED);

        // 绘制文本
        cv::putText(image, label, labelOrigin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

int main(int argc, char* argv[])
{
    auto data = ReadAll("assets/hd2-yolo11n-fp32.onnx");
    auto ctx = InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kCPU);
    auto infer_yolo = InferYOLO::Create(**ctx, data->span(), YOLOVersion::kV11);
    auto image = cv::imread("assets/hd2.png");
    auto result = infer_yolo->get()->Run(image, 0.625);
    drawYOLOResults(image, result->results);
    cv::namedWindow("YOLO Detection", cv::WINDOW_NORMAL); // 支持调整大小
    cv::resizeWindow("YOLO Detection", 1720, 720); // 设置窗口大小
    cv::imshow("YOLO Detection", image);
    cv::waitKey(0);
    auto video = cv::VideoCapture("assets/hd2.avi");
    while (video.read(image))
    {
        result = infer_yolo->get()->Run(image, 0.625);
        drawYOLOResults(image, result->results);
        cv::imshow("YOLO Detection", image);
        cv::waitKey(1);
    }
    return 0;
}
