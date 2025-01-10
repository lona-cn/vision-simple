# vision-simple
[english](./README-en.md) | 简体中文
</br>
Version: `0.3.0`

`vision-simple`是一个基于C++23的跨平台视觉推理库，提供开箱即用的推理功能，目前支持常用的YOLO系列（YOLOv10和YOLOv11）和一些OCR（`PaddleOCR`）,**内建HTTP API**，并且基于`ONNXRuntime`支持多种Execution Provider，如`DirectML`、`CUDA`、`TensorRT`，以及一些特殊的设备（如RockChip的NPU）。

---

## 演示
### yolo

![hd2-yolo-gif](doc/images/hd2-yolo.gif)

### OCR

![http-inferocr](doc/images/http-inferocr.png)

![paddleocr](doc/images/ppocr.png)
---

## 支持情况

### 视觉模型
|type|status|
|-|-|
|YOLOv10|Y|
|YOLOv11|Y|
|EasyOCR|N|
|PaddleOCR|Y|

### 推理框架
|framework|status|
|-|-|
|ONNXRuntime|Y|
|TVM|N|

### EP
|platform|CPU|DirectML|CUDA|TensorRT|Vulkan|OpenGL|OpenCL|
|-|-|-|-|-|-|-|-|
|windows|Y|Y|Y|?|N|N|N|
|linux|Y|N|Y|?|N|N|N|
|WSL|Y|N|Y|?|N|N|N|

---

## 例子
### YOLOv11+ONNXRuntime+DirectML
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

### HTTP API+OCR
1. 启动server项目：
```powershell
# 构建
./scripts/build-release.bat
xmake build server
# 启动server
xmake run server
```
2. 打开[swagger在线编辑器](https://editor-next.swagger.io/)，并允许该网站的不安全内容
3. 复制[doc/openapi/server.yaml](doc/openapi/server.yaml)的内容到`swagger在线编辑器`
4. 在编辑器右侧选择感兴趣的API进行测试
![swagger-right](doc/images/swagger-right.png)

---

## 开发
### 构建项目
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
### HTTP API
api按照OAS描述，其文件位于[doc/openapi/server.yaml](doc/openapi/server.yaml)
## docker
暂未支持。

## 联系方式
QQ: 1307693959 </br>
[email](amhakureireimu@gmail.com)
