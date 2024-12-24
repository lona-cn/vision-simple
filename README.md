# vision-simple
Inference examples for YOLOv10, YOLOv11, PaddleOCR, and EasyOCR using ONNXRuntime-DirectML and TVM

## demo
### yolo(HellDivers2)
<center>vision-simple</center>

![hd2-yolo-img](doc/images/hd2-yolo.jpg)

<center>ultralytics</center>

![hd2-yolo-gif](doc/images/hd2-yolo.gif)

## HOWTO
### build
#### windows
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
### api example

### YOLOv11
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
    auto ctx = InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kCPU);
    // create yolo inference instance
    auto infer_yolo = InferYOLO::Create(**ctx, data->span(), YOLOVersion::kV11);
    //----do inference----
    auto result = infer_yolo->get()->Run(image, 0.625);
    // do what u want
    return 0;
}
```