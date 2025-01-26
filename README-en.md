# <div align="center">🚀 vision-simple 🚀</div>
english | [简体中文](./README.md)

<p align="center">
<a><img alt="GitHub License" src="https://img.shields.io/github/license/lona-cn/vision-simple"></a>
<a><img alt="GitHub Release" src="https://img.shields.io/github/v/release/lona-cn/vision-simple"></a>
<a><img alt="Docker pulls" src="https://img.shields.io/docker/pulls/lonacn/vision_simple"></a>
<a><img alt="GitHub Downloads (all assets, all releases)" src="https://img.shields.io/github/downloads/lona-cn/vision-simple/total"></a>
</p>
<p align="center">
<a><img alt="" src="https://img.shields.io/badge/yolo-v10-AD65F1.svg"></a>
<a><img alt="" src="https://img.shields.io/badge/yolo-v11-AD65F1.svg"></a>
<a><img alt="" src="https://img.shields.io/badge/paddle_ocr-v4-2932DF.svg"></a>
</p>

<p align="center">
<a><img alt="windows x64" src="https://img.shields.io/badge/windows-x64-brightgreen.svg"></a>
<a><img alt="linux x86_64" src="https://img.shields.io/badge/linux-x86_64-brightgreen.svg"></a>
<a><img alt="linux arm64" src="https://img.shields.io/badge/linux-arm64-brightgreen.svg"></a>
<a><img alt="linux arm64" src="https://img.shields.io/badge/linux-riscv64-brightgreen.svg"></a>
</p>

<p align="center">
<a><img alt="ort cpu" src="https://img.shields.io/badge/ort-cpu-880088.svg"></a>
<a><img alt="ort dml" src="https://img.shields.io/badge/ort-dml-blue.svg"></a>
<a><img alt="ort cuda" src="https://img.shields.io/badge/ort-cuda-green.svg"></a>
<a><img alt="ort rknpu" src="https://img.shields.io/badge/ort-rknpu-white.svg"></a>
</p>

`vision-simple` is a cross-platform visual inference library based on C++23, designed to provide **out-of-the-box** inference capabilities. With Docker, users can quickly set up inference services. This library currently supports popular YOLO models (including YOLOv10 and YOLOv11) and some OCR models (such as `PaddleOCR`). It features a **built-in HTTP API**, making the service more accessible. Additionally, `vision-simple` uses the `ONNXRuntime` engine, which supports multiple Execution Providers such as `DirectML`, `CUDA`, `TensorRT`, and can be compatible with specific hardware devices (such as RockChip's RKNPU), offering more efficient inference performance.

## <div align="center">🚀 Features </div>
- **Cross-platform**: Supports `windows/x64`, `linux/x86_64`, `linux/arm64`,and `linux/riscv64`
- **Multi-device**: Supports CPU, GPU, and RKNPU
- **Small size**: The statically compiled version is under 20 MiB, with YOLO and OCR inference occupying 300 MiB of memory
- **Fast deployment**:
  - **One-click compilation**: Provides verified build scripts for multiple platforms
  - **[Container deployment](https://hub.docker.com/r/lonacn/vision_simple)**: One-click deployment with `docker`, `podman`, or `containerd`
  - **[HTTP Service](doc/openapi/server.yaml)**: Offers a HTTP API for non-real-time applications

### <div align="center"> yolov11n 3440x1440@60fps+ </div>
![hd2-yolo-gif](doc/images/hd2-yolo.gif)

### <div align="center"> OCR (HTTP API) </div>

![http-inferocr](doc/images/http-inferocr.png)
## <div align="center">🚀 Using vision-simple </div>
### Deploy HTTP Service
1. Start the server project:
```powershell
docker run -it --rm --name vs -p 11451:11451 lonacn/vision_simple:0.4.0-cpu-x86_64
```
2. Open the Swagger online editor and allow the site’s unsafe content.
3. Copy the content from doc/openapi/server.yaml into the Swagger editor.
4. On the right panel of the editor, select the APIs you want to test
![swagger-right](doc/images/swagger-right.png)

## <div align="center">🚀 Quick Start for Development </div>
### YOLOv11 Inference Development
```cpp
#include <Infer.h>
#include <opencv2/opencv.hpp>
using namespace vision_simple;
template <typename T>
struct DataBuffer
{
    std::unique_ptr<T[]> data;
    size_t size;
    std::span<T> span(){return std::span{data.get(), size};}
};

extern std::expected<DataBuffer<uint8_t>, InferError> ReadAll(const std::string& path);

int main(int argc,char *argv[]){
    auto data = ReadAll("assets/hd2-yolo11n-fp32.onnx");
    auto image = cv::imread("assets/hd2.png");
    auto ctx = InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kDML);
    auto infer_yolo = InferYOLO::Create(**ctx, data->span(), YOLOVersion::kV11);
    auto result = infer_yolo->get()->Run(image, 0.625);
    // do what u want
    return 0;
}
```
### Build Project
#### windows/x64
- xmake >= 2.9.7
- msvc with C++23
- Windows 11
```powershell
# setup sln
./scripts/setupdev-vs.bat
# test
xmake build test_yolo
xmake run test_yolo
```
#### linux/x86_64
- xmake >= 2.9.7
- gcc-13
- Debian 12 / Ubuntu 2022
```sh
# build release
./scripts/build-release.sh
# test
xmake build test_yolo
xmake run test_yolo
```
### Docker Image
All `Dockerfiles` are located in the `docker/` directory.
```sh
# From the root directory of vision-simple
# Build the project
docker build -t vision-simple:latest -f  docker/Dockerfile.debian-bookworm-x86_64-cpu .
# Run the container, the default configuration will use CPU inference and listen on port 11451
docker run -it --rm -p 11451:11451 --name vs vision-simple
```

<div align="center">🚀 Contact</div>

![Discord](https://img.shields.io/discord/1327875843581808640)

<div align="center">📄 License</div>
The copyrights for the YOLO models and PaddleOCR models in this project belong to the original authors.

This project is licensed under the Apache-2.0 license.