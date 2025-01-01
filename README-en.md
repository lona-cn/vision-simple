# vision-simple
english | [简体中文](./README.md)
</br>
Version: `0.2.0`

vision-simple is a C++23 library that provides a high-performance inference server for `YOLOv10`, `YOLOv11`, `PaddleOCR`, and `EasyOCR` with built-in HTTP API support. It supports multiple Execution Providers, including `DirectML` `CUDA` `TensorRT`, enabling flexible hardware acceleration. Designed for cross-platform deployment, it runs seamlessly on both Windows and Linux.


### A Simple Example of YOLOv11 Using DirectML
`test_yolo.cpp`
```cpp
#include <Infer.h>
#include <opencv2/opencv.hpp>

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

extern std::expected<DataBuffer<uint8_t>, InferError> ReadAll(const std::string& path);

int main(int argc,char *argv[]){
    //----read file----
    // read fp32 onnx model
    auto data = ReadAll("assets/hd2-yolo11n-fp32.onnx");
    // read test image
    auto image = cv::imread("assets/hd2.png");
    //----create context----
    // create inference context
    auto ctx = InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kDML);
    // create yolo inference instance
    auto infer_yolo = InferYOLO::Create(**ctx, data->span(), YOLOVersion::kV11);
    //----do inference----
    auto result = infer_yolo->get()->Run(image, 0.625);
    // do what u want
    return 0;
}
```

## support list
|type|status|
|-|-|
|YOLOv10|Y|
|YOLOv11|Y|
|EasyOCR|N|
|PaddleOCR|Y|
### inference frameworks
|framework|status|
|-|-|
|ONNXRuntime|Y|
|TVM|N|
### execution providers
|platform|CPU|DirectML|CUDA|TensorRT|Vulkan|OpenGL|OpenCL|
|-|-|-|-|-|-|-|-|
|windows|Y|Y|Y|?|N|N|N|
|linux|Y|N|Y|?|N|N|N|
|WSL|Y|N|Y|?|N|N|N|
## demo
### yolo(HellDivers2)
<center>vision-simple</center>

![hd2-yolo-img](doc/images/hd2-yolo.jpg)

<center>ultralytics</center>

![hd2-yolo-gif](doc/images/hd2-yolo.gif)

### PaddleOCR(HellDivers2)

![paddleocr](doc/images/ppocr.png)
## get started
### build
#### windows/amd64
* [xmake](https://xmake.io) >= 2.9.4
* msvc support c++23
* windows 11

```powershell
# setup sln
./scripts/setupdev-vs.bat
# test
xmake build test_yolo
xmake run test_yolo
```
#### linux/amd64
* [xmake](https://xmake.io) >= 2.9.4
* gcc-13
* debian12/ubuntu2022

```sh
# build release
./scripts/build-release.sh
# test
xmake build test_yolo
xmake run test_yolo
```

## docker
not support yet.

## Contact me
[email](amhakureireimu@gmail.com)