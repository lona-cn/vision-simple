#include <filesystem>
#include <fstream>
#include <semaphore>
#include <Infer.h>
#include <shared_mutex>

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
    const auto fs_size = std::filesystem::file_size(path);
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
    std::shared_mutex mtx; // 互斥锁

public:
    // 写入数据，支持自定义处理函数
    void Write(const std::function<void(cv::Mat&)>& process_frame)
    {
        // 在备用缓冲区上执行自定义操作
        process_frame(BackFrame()); // 用户定义的操作
        // 交换缓冲区
        std::unique_lock lock{mtx};
        currentIndex = 1 - currentIndex;
    }

    cv::Mat& FrontFrame()
    {
        std::shared_lock lock{mtx};
        return buffer[currentIndex];
    }

    cv::Mat& BackFrame()
    {
        std::shared_lock lock{mtx};
        return buffer[1 - currentIndex];
    }
};

template <typename T, size_t QUEUE_SIZE = 10>
class SafeQueue
{
    std::queue<T> queue_; // 队列存储
    std::counting_semaphore<QUEUE_SIZE> emptySlots_{QUEUE_SIZE}; // 空槽数，初始值为 QUEUE_SIZE
    std::counting_semaphore<QUEUE_SIZE> filledSlots_{0}; // 已用槽数，初始值为 0
    mutable std::shared_mutex mtx_; // 线程安全的互斥锁

public:
    SafeQueue() = default;

    // **1. PopFrontFor：超时等待并取出队首元素**
    template <class Rep, typename Period>
    std::optional<T> PopFrontFor(std::chrono::duration<Rep, Period> duration)
    {
        try
        {
            // 等待 filledSlots 信号量（已用槽数）减少，超时则返回空
            if (!filledSlots_.try_acquire_for(duration))
                return std::nullopt;
        }
        catch (const std::system_error& _)
        {
            return std::nullopt; // 出错返回
        }

        // 从队列中取元素
        std::unique_lock lock{mtx_};
        if (queue_.empty()) // 再次检查空队列情况
        {
            filledSlots_.release(); // 恢复信号量
            return std::nullopt;
        }
        auto item = std::move(queue_.front());
        queue_.pop();

        // 增加空槽数信号量
        emptySlots_.release();
        return item;
    }

    // **2. PushBack：向队尾添加元素，受信号量控制**
    template <class Rep, typename Period>
    bool PushBack(T&& item, T& out, std::chrono::duration<Rep, Period> duration)
    {
        try
        {
            // 等待空槽，如果没有空槽则阻塞，避免超过容量
            if (!emptySlots_.try_acquire_for(duration))return false;
        }
        catch (const std::system_error& _)
        {
            return false; // 操作失败
        }

        // 加入队列
        {
            std::unique_lock lock{mtx_};
            queue_.emplace(std::move(item));
        }

        // 增加已用槽数信号量
        filledSlots_.release();
        return true;
    }

    // **3. 获取队列大小**
    size_t Size() const
    {
        std::shared_lock lock{mtx_};
        return queue_.size();
    }

    // **4. 判断队列是否为空**
    bool IsEmpty() const
    {
        return Size() == 0;
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
    std::cout << "Read Model" << std::endl;
    auto data = ReadAll("assets/hd2-yolo11n-fp16.onnx");
    std::cout << "Create Infer Context" << std::endl;
#ifdef VISION_SIMPLE_WITH_DML
    auto ctx = InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kDML);
#else
    auto ctx = InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kCPU);
#endif
    std::cout << "Create Infer Instance" << std::endl;
    auto infer_yolo = InferYOLO::Create(**ctx, data->span(), YOLOVersion::kV11);
    const char* WINDIW_TITLE = "YOLO Detection";
    cv::namedWindow(WINDIW_TITLE, cv::WINDOW_NORMAL); // 支持调整大小
    cv::resizeWindow(WINDIW_TITLE, 1720, 720); // 设置窗口大小
    // auto image = cv::imread("assets/hd2.png");
    // auto result = infer_yolo->get()->Run(image, 0.625);
    // drawYOLOResults(image, result->results);
    // cv::imshow("YOLO Detection", image);
    // cv::waitKey(0);
    SafeQueue<cv::Mat> decode_queue, show_queue;
    std::atomic_bool exit_flag{false};
    std::jthread video_thread{
        [&]
        {
            Finally finally{
                reinterpret_cast<char*>(42), [&](void*)
                {
                    exit_flag.store(true);
                }
            };
            auto video = cv::VideoCapture("assets/hd2.avi");
            std::cout << "Video Decoding thread running" << std::endl;
            while (!exit_flag.load() && video.grab())
            {
                cv::Mat img;
                video.retrieve(img);
                // auto rect = cv::getWindowImageRect(WINDIW_TITLE);
                // cv::resize(img, img, rect.size());
                while (!exit_flag.load() && !decode_queue.PushBack(std::move(img), img, std::chrono::milliseconds(10)))
                {
                }
            }
        }
    };
    //TODO: multithread+reoredered frame
    std::jthread infer_thread{
        [&]
        {
            FPSCounter fps_counter{};
            while (!exit_flag.load())
            {
                auto front_frame_opt = decode_queue.PopFrontFor(std::chrono::milliseconds(10));
                if (!front_frame_opt)continue;
                auto result = infer_yolo->get()->Run(*front_frame_opt, 0.225f);
                drawYOLOResults(*front_frame_opt, result->results);
                fps_counter.update();
                fps_counter.display(*front_frame_opt);
                auto img = *std::move(front_frame_opt);
                while (!exit_flag.load() && !show_queue.PushBack(std::move(img), img, std::chrono::milliseconds(10)))
                {
                }
            }
        }
    };
    while (!exit_flag.load())
    {
        auto show_img_opt = show_queue.PopFrontFor(std::chrono::milliseconds(10));
        if (!show_img_opt)continue;
        auto& frame = *show_img_opt;
        if (!frame.empty())
        {
            cv::imshow(WINDIW_TITLE, frame);
        }
        if (cv::waitKey(1) == 27)
        {
            exit_flag.store(true);
        }
    }
    return 0;
}
