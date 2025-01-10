#pragma once
#ifndef __VISION_SIMPLE_TEST_UTIL_H__
#define __VISION_SIMPLE_TEST_UTIL_H__
#include <filesystem>
#include <fstream>
#include <semaphore>
#include <Infer.h>
#include <shared_mutex>
#include <cstddef>
#include <format>
#include <magic_enum.hpp>
#include "IOUtil.h"

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

using Finally = std::unique_ptr<char, std::function<void(void*)>>;
#endif
