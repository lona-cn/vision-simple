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

class DoubleBuffer
{
private:
    cv::Mat buffer[2]; // 双缓冲区
    int currentIndex = 0; // 当前缓冲区索引
    bool newDataAvailable = false; // 数据更新标志
    std::mutex mtx; // 互斥锁
    std::condition_variable dataReady; // 条件变量

public:
    // 写入数据，支持自定义处理函数
    void write(std::function<void(cv::Mat&)> processFrame)
    {
        std::unique_lock<std::mutex> lock(mtx);

        // 在备用缓冲区上执行自定义操作
        processFrame(buffer[1 - currentIndex]); // 用户定义的操作

        // 标记新数据可用
        newDataAvailable = true;

        // 通知消费者
        dataReady.notify_one();
    }

    // 读取数据
    cv::Mat& read()
    {
        std::unique_lock<std::mutex> lock(mtx);

        // 等待新数据
        dataReady.wait(lock, [this]() { return newDataAvailable; });

        // 切换到新缓冲区
        currentIndex = 1 - currentIndex;

        // 重置标志位
        newDataAvailable = false;

        // 返回缓冲区引用
        return buffer[currentIndex];
    }
};

class FPSCounter
{
    // 使用高精度时钟计算时间
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime{std::chrono::high_resolution_clock::now()};
    int frameCount{0}; // 当前计算时间窗口内的帧数
    double fps{0.0}; // 当前FPS值
    double timeWindow{1.0}; // 每次计算的时间窗口（秒）

public:
    // 更新帧数并计算FPS
    void update()
    {
        frameCount++; // 增加当前窗口内的帧数

        // 当前时间
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - startTime;

        // 如果当前时间窗口已过
        if (elapsed.count() >= timeWindow)
        {
            // 计算过去 timeWindow 秒内的平均 FPS
            fps = frameCount / elapsed.count();

            // 重置时间和帧数
            startTime = now; // 更新起始时间
            frameCount = 0; // 重置帧数
        }
    }

    // 获取当前FPS
    double getFPS() const
    {
        return fps;
    }

    // 在图像上显示FPS信息
    void display(cv::Mat& frame)
    {
        std::string fpsText = "Infer FPS: " + std::to_string(static_cast<int>(fps));

        // 确保窗口大小足够显示文字
        if (frame.rows > 30 && frame.cols > 100)
        {
            cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }
    }
};

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

using Finally = std::unique_ptr<char, std::function<void(void*)>>;

int main(int argc, char* argv[])
{
    auto data = ReadAll("assets/hd2-yolo11n-fp16.onnx");
#ifdef VISION_SIMPLE_WITH_DML
    auto ctx = InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kDML);
#else
    auto ctx = InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kCPU);
#endif
    auto infer_yolo = InferYOLO::Create(**ctx, data->span(), YOLOVersion::kV11);
    cv::namedWindow("YOLO Detection", cv::WINDOW_NORMAL); // 支持调整大小
    cv::resizeWindow("YOLO Detection", 1720, 720); // 设置窗口大小
    // auto image = cv::imread("assets/hd2.png");
    // auto result = infer_yolo->get()->Run(image, 0.625);
    // drawYOLOResults(image, result->results);
    // cv::imshow("YOLO Detection", image);
    // cv::waitKey(0);
    auto video = cv::VideoCapture("assets/hd2.avi");
    DoubleBuffer buf;
    std::atomic_bool exit_flag{false};
    std::jthread t{
        [&]
        {
            Finally finally{
                reinterpret_cast<char*>(42), [&](void*)
                {
                    exit_flag.store(true);
                }
            };
            FPSCounter fps_counter{};
            while (!exit_flag.load() && video.grab())
            {
                buf.write([&](cv::Mat& buf_image)
                {
                    video.retrieve(buf_image);
                    auto result = infer_yolo->get()->Run(buf_image, 0.625);
                    drawYOLOResults(buf_image, result->results);
                    fps_counter.update();
                    fps_counter.display(buf_image);
                });
            }
        }
    };
    while (!exit_flag.load())
    {
        cv::imshow("YOLO Detection", buf.read());
        if (cv::waitKey(1) == 27)
        {
            exit_flag.store(true);
        }
    }
    return 0;
}
