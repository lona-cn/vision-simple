#include <Infer.h>

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <magic_enum.hpp>
#include <semaphore>
#include <shared_mutex>
#include <thread>

#include "Util.hpp"
using namespace vision_simple;

void drawYOLOResults(cv::Mat& image, const std::vector<YOLOResult>& results) {
  for (const auto& result : results) {
    // 绘制边界框
    cv::rectangle(image, result.bbox, cv::Scalar(0, 255, 0), 2);

    // 设置标签内容
    std::string label = result.class_name.data() +
                        (" (" + std::to_string(result.confidence * 100) + "%)");

    // 计算文本背景框大小
    int baseLine = 0;
    cv::Size labelSize =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    // 确定文本位置
    int top = std::max(result.bbox.y, labelSize.height);
    cv::Point labelOrigin(result.bbox.x, top);

    // 绘制文本背景框
    cv::rectangle(image, labelOrigin + cv::Point(0, baseLine),
                  labelOrigin + cv::Point(labelSize.width, -labelSize.height),
                  cv::Scalar(0, 255, 0), cv::FILLED);

    // 绘制文本
    cv::putText(image, label, labelOrigin, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1);
  }
}

int main(int argc, char* argv[]) {
  std::cout << "Load Model" << std::endl;
  auto data = ReadAll("assets/hd2-yolo11n-fp16.onnx");
  std::cout << "Create Infer Context" << std::endl;
#ifdef VISION_SIMPLE_WITH_DML
  auto ctx = InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kDML);
#elifdef VISION_SIMPLE_WITH_CUDA
  auto ctx = InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kCUDA);
  // #elifdef VISION_SIMPLE_WITH_TENSORRT
  // auto ctx = InferContext::Create(InferFramework::kONNXRUNTIME,
  // InferEP::kTensorRT);
#else
  auto ctx = InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kCPU);
#endif
  if (!ctx) {
    auto& error = ctx.error();
    std::cout << std::format(
                     "Failed to create Infer Context code:{} message:{}",
                     magic_enum::enum_name(error.code), error.message)
              << std::endl;
    return -1;
  }
  std::cout << std::format("framework:{} EP:{}",
                           magic_enum::enum_name((*ctx)->framework()),
                           magic_enum::enum_name((*ctx)->execution_provider()))
            << std::endl;
  std::cout << "Create Infer Instance" << std::endl;
  auto infer_yolo = InferYOLO::Create(**ctx, data->span(), YOLOVersion::kV11);
  if (!infer_yolo) {
    auto& error = infer_yolo.error();
    std::cout << std::format(
                     "Failed to create Infer Instance code:{} message:{}",
                     magic_enum::enum_name(error.code), error.message)
              << std::endl;
    return -1;
  }
  const char* WINDIW_TITLE = "YOLO Detection";
  cv::namedWindow(WINDIW_TITLE, cv::WINDOW_NORMAL);  // 支持调整大小
  cv::resizeWindow(WINDIW_TITLE, 1720, 720);         // 设置窗口大小
  // auto image = cv::imread("assets/hd2.png");
  // auto result = infer_yolo->get()->Run(image, 0.625);
  // drawYOLOResults(image, result->results);
  // cv::imshow("YOLO Detection", image);
  // cv::waitKey(0);
  SafeQueue<cv::Mat> decode_queue, show_queue;
  std::atomic_bool exit_flag{false};
  std::jthread video_thread{[&] {
    Finally finally{reinterpret_cast<char*>(42),
                    [&](void*) { exit_flag.store(true); }};
    auto video = cv::VideoCapture("assets/hd2.avi");
    std::cout << "Video Decoding thread running" << std::endl;
    while (!exit_flag.load() && video.grab()) {
      cv::Mat img;
      video.retrieve(img);
      while (!exit_flag.load() &&
             !decode_queue.PushBack(std::move(img), img,
                                    std::chrono::milliseconds(1))) {
      }
    }
  }};
  // TODO: multithread+reoredered frame
  std::jthread infer_thread{[&] {
    FPSCounter fps_counter{};
    while (!exit_flag.load()) {
      auto front_frame_opt =
          decode_queue.PopFrontFor(std::chrono::milliseconds(1));
      if (!front_frame_opt) continue;
      auto img = *std::move(front_frame_opt);
      auto result = infer_yolo->get()->Run(img, 0.225f);
      drawYOLOResults(img, result->results);
      fps_counter.update();
      fps_counter.display(img);
      while (!exit_flag.load() &&
             !show_queue.PushBack(std::move(img), img,
                                  std::chrono::milliseconds(1))) {
      }
    }
  }};
  while (!exit_flag.load()) {
    decltype(show_queue.PopFrontFor(
        std::chrono::milliseconds(10))) last_img_opt{std::nullopt};
    while (auto show_img_opt =
               show_queue.PopFrontFor(std::chrono::milliseconds(0))) {
      last_img_opt = std::move(show_img_opt);
    }
    if (!last_img_opt) continue;
    auto& frame = *last_img_opt;
    if (!frame.empty()) {
      cv::imshow(WINDIW_TITLE, frame);
    }
    if (cv::waitKey(1) == 27) {
      exit_flag.store(true);
    }
  }
  return 0;
}
