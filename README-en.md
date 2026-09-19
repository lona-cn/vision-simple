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
### Deploy HTTP Service with docker
1. Start the server project:
```powershell
docker run -it --rm --name vs -p 11451:11451 lonacn/vision_simple:0.4.1-cpu-x86_64
```
2. Open the Swagger online editor and allow the site’s unsafe content.
3. Copy the content from doc/openapi/server.yaml into the Swagger editor.
4. On the right panel of the editor, select the APIs you want to test
![swagger-right](doc/images/swagger-right.png)

### HTTP v0 errors and batch semantics

Successful fields are unchanged: YOLO returns `class_names`/`results`; OCR returns `results`. `model` must be a nonempty string and `images` an array of strings. An existing model accepts an empty array. HTTP 200 guarantees one result per input image in the original order; no detections is a successful empty item.

Any image failure fails the entire batch, without partial results. Processing order is request validation, model lookup/loading, decoding all images, inference, then serialization. The first failing stage reports its lowest failing image index.

```json
{"error":{"code":"invalid_image","message":"Image cannot be decoded","image_index":1}}
```

| HTTP | `error.code` | `image_index` |
| --- | --- | --- |
| 400 | `invalid_request`, `unknown_model` | `null` |
| 400 | `invalid_image` | Zero-based image index |
| 500 | `model_load_failed`, `model_config_failed`, `internal_error` | `null` |
| 500 | `inference_failed` | Zero-based image index |

Clients relying on HTTP 200 with textual errors must migrate to HTTP status and `error.code`; do not branch on `message`. Third-party exception details remain in logs. See the [OpenAPI contract](doc/openapi/server.yaml).

### C++ migration

- `InferYOLO/InferOCR::Create/Run` signatures are unchanged. Run requires a nonempty two-dimensional `CV_8UC3` image and supports non-contiguous ROIs. Grayscale, BGRA, floating-point images and non-finite/out-of-range `[0,1]` confidence return parameter errors.
- YOLO v11 uses class-aware NMS; v10 accepts only end-to-end `[1,N,6]` output and does not repeat NMS. Confidence defaults, black Letterbox padding and OCR detection normalization are unchanged.
- OCR file-based Create follows the Paddle dictionary convention: the file excludes blank and the trailing space class; the loader appends space. Map-based callers supply every nonblank class with key `class_id - 1`.
- `HTTPServer::Run/StartAsync` return `HTTPServerResult<void>`; callers must check failures. Empty/overlong hosts and listen failures are rejected without silently binding wildcard. The repository's explicit `0.0.0.0` default is unchanged.
- Helper consumers must recompile and migrate to one geometry path:

```cpp
LetterboxTransform transform;
cv::Mat& padded = helper.Letterbox(image, target_size, transform);
if (padded.empty()) { /* reject invalid or rounded-zero image dimensions */ }
cv::Rect box = VisionHelper::ScaleCoords(transform, cv::Vec4f{x1, y1, x2, y2});
```

`ScaleCoords` accepts floating-point model-space xyxy, reverses actual per-axis scaling, clips endpoints, then rounds endpoints into integer xywh. Old geometry signatures, `DataConverter` and unimplemented uint8 no-op conversions were removed. `Cvt` supports bidirectional FP32/FP16 conversion.

This work does not address authentication/CORS, image/batch resource limits, thread pools, hot loading, the global error arena, full signal shutdown, RKNPU/container releases or arbitrary ONNX support. It is not a blanket production-security guarantee.

## <div align="center">🚀 Quick Start for Development </div>

### Build Project
#### windows/x64
- xmake >= 2.9.7
- msvc with C++23
- Windows 11
```powershell
# pull project
git clone https://github.com/lona-cn/vision-simple.git
cd vision-simple
# setup sln
./scripts/dev-vs.bat
# run server
xmake build server
xmake run server
```
#### linux/x86_64
- xmake >= 2.9.7
- gcc-13
- Debian 12 / Ubuntu 2022
```sh
# pull project
git clone https://github.com/lona-cn/vision-simple.git
cd vision-simple
# build release
./scripts/build-release.sh
# run server
xmake build server
xmake run server
```

#### linux/arm64 (Cross-compile)
- Cross-compile toolchain: `aarch64-linux-gnu-`

```sh
xmake f -p linux -a arm64 --cross=aarch64-linux-gnu- -m release
xmake build server
```

#### linux/riscv64 (Cross-compile)
- Cross-compile toolchain: `riscv64-linux-gnu-`

```sh
xmake f -p linux -a riscv64 --cross=riscv64-linux-gnu- -m release
xmake build server
```

### Enable Hardware Acceleration (Execution Provider)

```sh
# CUDA
xmake f --with_cuda=y -m release
xmake build server

# TensorRT
xmake f --with_tensorrt=y -m release
xmake build server

# RKNPU (Linux only)
xmake f --with_rknpu=y -m release
xmake build server
```

### Run Tests

```sh
# Build CPU regression targets individually
xmake build server
xmake build test_common
xmake build test_cvt
xmake build test_vision_helper
xmake build test_yolo_postprocess
xmake build test_ocr_decode
xmake build test_infer_inputs

xmake run test_common
xmake run test_cvt
xmake run test_vision_helper
xmake run test_yolo_postprocess
xmake run test_ocr_decode
xmake run test_infer_inputs
```

The Python 3 standard-library HTTP driver creates isolated configuration, ports and processes; it does not touch an existing port 11451 service. Missing models, dictionaries, images or failure fixtures fail the run rather than count as SKIP.

```powershell
python scripts/test_http_regression.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
```

For Linux or a custom build directory, discover the actual target:

```sh
server="$(xmake lua -q -c "import('core.project.config'); config.load(); import('core.project.project'); io.write(path.absolute(project.target('server'):targetfile()))")"
python3 scripts/test_http_regression.py --server "$server" --project-root .
```

`test_yolo`/`test_ocr` remain interactive demos, not headless acceptance tests. Tiny failure models are checked in; only regeneration requires the development package `onnx` and `scripts/generate_reliability_fixtures.py`, not a server runtime dependency.

### Docker Image
All `Dockerfiles` are located in the `docker/` directory.
```sh
# pull project
git clone https://github.com/lona-cn/vision-simple.git
cd vision-simple
# Build the project
docker build -t vision-simple:latest -f  docker/Dockerfile.debian-bookworm-x86_64-cpu .
# Run the container, the default configuration will use CPU inference and listen on port 11451
docker run -it --rm -p 11451:11451 --name vs vision-simple
```

#### Other Platforms / Hardware Acceleration

```sh
# ARM64 CPU
docker build -t vision-simple:arm64 -f docker/Dockerfile.debian-bookworm-arm64-cpu .

# RISC-V CPU
docker build -t vision-simple:riscv64 -f docker/Dockerfile.debian-sid-riscv64-cpu .

# x86_64 + CUDA/TensorRT
docker build -t vision-simple:cuda -f docker/Dockerfile.debian-bookworm-x86_64-cuda_trt .

# ARM64 + Rockchip NPU
docker build -t vision-simple:rknpu -f docker/Dockerfile.debian-bookworm-arm64-rknpu .
```

### dev YOLOv11 Inference with `vision-simple`
```cpp
#include <vision_simple/Infer.h>
#include <opencv2/opencv.hpp>
using namespace vision_simple;

int main() {
    auto ctx = InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kCPU);
    if (!ctx) return 1;
    auto model = InferYOLO::Create(**ctx, "assets/hd2-yolo11n-fp32.onnx", YOLOVersion::kV11);
    if (!model) return 1;
    auto image = cv::imread("assets/hd2.png");
    auto result = (*model)->Run(image, 0.625f);
    return result ? 0 : 1;
}
```

<div align="center">📄 License</div>
The copyrights for the YOLO models and PaddleOCR models in this project belong to the original authors.

This project is licensed under the Apache-2.0 license.