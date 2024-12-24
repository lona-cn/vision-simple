# vision-simple
a simple cross-platform inference engine,support `YOLOv10~11`, `PaddleOCR`/`EasyOCR` using `ONNXRuntime`/`TVM` with multiple exectuion providers.

## support list
|type|status|
|-|-|
|YOLOv10|Y|
|YOLOv11|Y|
|EasyOCR|N|
|PaddleOCR|N|
### inference frameworks
|framework|status|
|-|-|
|ONNXRuntime|Y|
|TVM|N|
### execution providers
|EP|status|
|-|-|
|CPU|Y|
|DirectML|Y|
|CUDA|N|
|TensorRT|N|
|Vulkan|N|
|OpenGL|N|
|OpenCL|N|
## demo
### yolo(HellDivers2)
<center>vision-simple</center>

![hd2-yolo-img](doc/images/hd2-yolo.jpg)

<center>ultralytics</center>

![hd2-yolo-gif](doc/images/hd2-yolo.gif)

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
### api examples

### YOLOv11 with DirectML
`test_yolo.cpp`
```cpp
#include <Infer.h>

using namespace vision_simple;

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