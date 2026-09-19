# <div align="center">🚀 vision-simple 🚀</div>
[english](./README-en.md) | 简体中文

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

`vision-simple` 是一个基于 C++23 的跨平台视觉推理库，旨在提供 **开箱即用** 的推理功能。通过 Docker用户可以快速搭建推理服务。该库目前支持常见的 YOLO 系列（包括 YOLOv10 和 YOLOv11），以及部分 OCR 模型（如 `PaddleOCR`）。**内建 HTTP API** 使得服务更加便捷。此外，`vision-simple` 采用 `ONNXRuntime` 引擎，支持多种 Execution Provider，如 `DirectML`、`CUDA`、`TensorRT`，并可与特定硬件设备（如 RockChip 的 RKNPU）兼容，提供更高效的推理性能。


## <div align="center">🚀 特性 </div>

- **跨平台**：支持`windows/x64`、`linux/x86_64`、`linux/arm64/v8`、`linux/riscv64`
- **多计算设备**：支持CPU、GPU、RKNPU
- **嵌入式设备**：目前已支持`rk3568`、`rv1106G3`（Luckfox Pico 1T算力版本）
- **小体积**：静态编译版本体积不到20MiB，推理YOLO和OCR占用300MiB内存
- **快速部署**：
  - **一键编译**：提供各个平台已验证的编译脚本
  - **[容器部署](https://hub.docker.com/r/lonacn/vision_simple)**：使用`docker`、`podman`、`containerd`一键部署
  - **[HTTP服务](doc/openapi/server.yaml)**：提供HTTP API供Web应用调用


### <div align="center"> YOLOv11 </div>
![hd2-yolo-gif](doc/images/hd2-yolo.gif)

### <div align="center"> OCR(HTTP API) </div>

![http-inferocr](doc/images/http-inferocr.png)
## <div align="center">🚀 快速使用 </div>
### docker部署HTTP服务
1. 启动server项目：
```sh
docker run -it --rm --name vs -p 11451:11451 lonacn/vision_simple:0.4.1-cpu-x86_64
```
2. 打开[swagger在线编辑器](https://editor-next.swagger.io/)，并允许该网站的不安全内容
3. 复制[doc/openapi/server.yaml](doc/openapi/server.yaml)的内容到`swagger在线编辑器`
4. 在编辑器右侧选择感兴趣的API进行测试：
![swagger-right](doc/images/swagger-right.png)

### HTTP v0 错误与批量语义

成功字段保持不变：YOLO 返回 `class_names`/`results`，OCR 返回 `results`。`model` 必须为非空字符串，`images` 必须为字符串数组；有效模型接受空数组。HTTP 200 保证结果数量与输入数量相等、顺序一致；某张图没有目标时，该项为空数组。

任一图片失败即整批失败，不返回部分结果。处理顺序为请求校验、模型查找/加载、全部图片解码、逐图推理、序列化；返回首先失败的阶段及该阶段最小图片索引。

```json
{"error":{"code":"invalid_image","message":"Image cannot be decoded","image_index":1}}
```

| HTTP | `error.code` | `image_index` |
| --- | --- | --- |
| 400 | `invalid_request`、`unknown_model` | `null` |
| 400 | `invalid_image` | 从 0 开始的图片索引 |
| 500 | `model_load_failed`、`model_config_failed`、`internal_error` | `null` |
| 500 | `inference_failed` | 从 0 开始的图片索引 |

旧客户端需从“HTTP 200 + 文本错误”迁移为检查 HTTP 状态和 `error.code`，不能依赖 `message` 文案。第三方异常细节仅保留在日志。完整契约见 [OpenAPI](doc/openapi/server.yaml)。

### C++ 迁移说明

- `InferYOLO/InferOCR::Create/Run` 签名不变。Run 接受非空二维 `CV_8UC3`，支持非连续 ROI；灰度、BGRA、浮点图像以及非有限或超出 `[0,1]` 的 confidence 返回参数错误。
- YOLO v11 默认按类别 NMS；v10 仅支持端到端 `[1,N,6]`，不再重复 NMS。原 confidence、黑色 Letterbox 填充和 OCR 检测归一化不变。
- OCR 文件路径 Create 使用 Paddle 字典文件约定：文件不含 blank 和末尾空格类别，由加载器补空格；直接传入 map 时，调用者须提供全部非 blank 类别，键为 `class_id - 1`。
- `HTTPServer::Run/StartAsync` 现在返回 `HTTPServerResult<void>`，调用者必须检查错误。空/超长 host、监听失败会受控失败，不会静默绑定 wildcard。仓库显式 `0.0.0.0` 默认配置未改变。
- helper 使用者必须重新编译并迁移到单一几何路径：

```cpp
LetterboxTransform transform;
cv::Mat& padded = helper.Letterbox(image, target_size, transform);
if (padded.empty()) { /* 拒绝无效或缩放后零尺寸的图片 */ }
cv::Rect box = VisionHelper::ScaleCoords(transform, cv::Vec4f{x1, y1, x2, y2});
```

`ScaleCoords` 接收模型空间浮点 xyxy，按实际轴向比例反算，裁剪端点后 round 为整数 xywh。旧几何签名、`DataConverter` 和未实现的 uint8 转换空操作已删除；`Cvt` 支持 FP32/FP16 双向转换。

本轮不解决认证/CORS、批量/图像资源限额、线程池、热加载、全局错误 arena、完整信号停机、RKNPU/容器发布或任意 ONNX 支持，不能据此宣称全面生产安全。


## <div align="center">🚀 快速开发 </div>

### 构建项目
#### windows/x64
* [xmake](https://xmake.io) >= 2.9.7
* msvc with c++23
* windows 11

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
* [xmake](https://xmake.io) >= 2.9.7
* gcc-13
* debian12/ubuntu2022

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

#### linux/arm64 (交叉编译)
* 交叉编译工具链: `aarch64-linux-gnu-`

```sh
xmake f -p linux -a arm64 --cross=aarch64-linux-gnu- -m release
xmake build server
```

#### linux/riscv64 (交叉编译)
* 交叉编译工具链: `riscv64-linux-gnu-`

```sh
xmake f -p linux -a riscv64 --cross=riscv64-linux-gnu- -m release
xmake build server
```

### 启用硬件加速 (Execution Provider)

```sh
# CUDA
xmake f --with_cuda=y -m release
xmake build server

# TensorRT
xmake f --with_tensorrt=y -m release
xmake build server

# RKNPU (仅 Linux)
xmake f --with_rknpu=y -m release
xmake build server
```

### 运行测试

```sh
# 构建 CPU 回归目标（逐个构建）
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

HTTP 回归使用 Python 3 标准库，独立创建临时配置、端口和进程，不触碰现有 11451 服务。模型、字典、图片和小型 ONNX 故障 fixture 必须存在；缺失即失败，不记为 SKIP。

```powershell
python scripts/test_http_regression.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
```

Linux 或自定义构建目录先查询实际可执行文件：

```sh
server="$(xmake lua -q -c "import('core.project.config'); config.load(); import('core.project.project'); io.write(path.absolute(project.target('server'):targetfile()))")"
python3 scripts/test_http_regression.py --server "$server" --project-root .
```

`test_yolo`/`test_ocr` 仍为交互演示，不作为上述 headless 验收。故障 fixture 已入库；仅重新生成时需要开发工具 `onnx` 和 `scripts/generate_reliability_fixtures.py`，不是服务运行依赖。

### 构建docker镜像
所有`Dockerfile`位于目录：`docker/`

```sh
# pull project
git clone https://github.com/lona-cn/vision-simple.git
cd vision-simple
# 构建项目
docker build -t vision-simple:latest -f  docker/Dockerfile.debian-bookworm-x86_64-cpu .
# 运行容器，默认配置会使用CPU推理并监听11451端口
docker run -it --rm -p 11451:11451 --name vs vision-simple
```

#### 其他平台 / 硬件加速

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

### 使用`vision-simple`进行YOLOv11推理

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

## <div align="center">📄 许可证</div>
项目内的YOLO模型和PaddleOCR模型版权归原项目所有

本项目使用**Apache-2.0**许可证
