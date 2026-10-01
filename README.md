# <div align="center">vision-simple</div>

[English](./README-en.md) | 简体中文

<p align="center">
<a><img alt="GitHub License" src="https://img.shields.io/github/license/lona-cn/vision-simple"></a>
<a><img alt="GitHub Release" src="https://img.shields.io/github/v/release/lona-cn/vision-simple"></a>
<a href="https://github.com/users/lona-cn/packages/container/package/vision-simple"><img alt="GHCR image" src="https://img.shields.io/badge/GHCR-vision--simple-2496ED"></a>
<a><img alt="GitHub Downloads" src="https://img.shields.io/github/downloads/lona-cn/vision-simple/total"></a>
</p>

`vision-simple` 是一个 C++23 视觉推理库，也可以作为独立 HTTP 服务运行。它用 ONNX Runtime 执行 YOLO 和 OCR 模型，提供目标检测、实例分割、姿态估计、旋转框检测和文字识别；在此基础上，还支持检测结果跟踪和视频画面文字转字幕。

如果你只想先试一次推理，从下面的快速上手开始即可。如果要接入现有程序，可以选择 C++、原生 HTTP、OpenAI-like 或 MCP SSE，不必分别部署几套推理服务。

本文对应**当前源码**。历史发布包或镜像可能还没有包含这里介绍的全部功能。

**导航**：[能做什么](#能做什么) · [快速上手](#快速上手) · [模型与推理](#模型与推理) · [接入方式](#接入方式) · [部署与诊断](#部署与诊断) · [构建变体与验证](#构建变体与验证)

## 能做什么

| 能力 | 当前支持 | 使用前要知道 |
| --- | --- | --- |
| 目标检测 | YOLOv10、YOLO11、YOLO26 | 使用符合输入、输出及元数据约定的 ONNX 模型，不是任意 ONNX 执行器 |
| 实例分割、姿态、旋转框 | YOLO11、YOLO26 | 分别使用 `seg`、`pose`、`obb` 任务及对应模型 |
| OCR | PP-OCR v3/v4 的 CTC 识别；受限的 Paddle SAR 识别 | 需要检测模型、识别模型和字典；EasyOCR 尚未实现 |
| 时序跟踪 | ByteTrack、BoT-SORT | 由调用者逐帧提交检测结果；不会自动运行检测或 ReID 模型 |
| 视频字幕 | OCR 提取画面文字，输出 SRT/WebVTT | 异步上传任务，不是语音转写；视频格式取决于构建平台 |
| 部署与诊断 | 按需加载、空闲卸载、统计、预检、预热和分阶段计时 | 健康检查不等于模型可用；取消和超时是合作式的 |

Windows x64 和 Linux x86_64 提供原生构建路径；另有 ARM64、ARMv7、RISC-V64 的构建配置。CPU、DirectML、CUDA、TensorRT、RKNPU 的实际可用性取决于构建依赖、驱动、硬件和模型。交叉编译成功不等于已在目标设备上跑过推理。

**效果示例**

![YOLO11 检测示例](doc/images/hd2-yolo.gif)

![OCR HTTP 接口示例](doc/images/http-inferocr.png)

## 快速上手

先用默认的 YOLO11 FP32 模型跑通请求，再换自己的模型或执行提供者，会更容易定位问题。

### 1. 获取源码和模型资源

准备 Git、Git LFS 和用于客户端示例的 Python 3。源码构建还需要 [xmake](https://xmake.io) 和支持 C++23 的编译器；容器部署需要 Docker。客户端在宿主机运行，不需要进入容器。

```sh
git clone --recurse-submodules https://github.com/lona-cn/vision-simple.git
cd vision-simple
git lfs install
git lfs pull
```

已有 checkout 可执行 `git submodule update --init --recursive` 补齐子模块。Git LFS 指针文件不是模型权重，不能拿来推理。首次构建会下载并编译依赖，需要可用网络和相应构建工具。

### 2. 启动服务：容器或源码二选一

#### 容器：本地构建 CPU 镜像

在仓库根目录运行：

```sh
docker build --platform linux/amd64 -t vision-simple:local -f docker/Dockerfile.debian-bookworm-x86_64-cpu .
docker run -it --rm --name vs -p 127.0.0.1:11451:11451 vision-simple:local
```

该镜像包含默认 YOLO11/PP-OCR 配置及测试资源，不包含 YOLO26 权重。x86_64 构建需要支持 AVX、AVX2、F16C 的 CPU。这里从源码构建，避免把旧镜像标签当作当前功能；发布镜像见后面的 GHCR 说明。

**推理和用户任务接口没有内建认证或租户隔离。** 示例只将端口发布到宿主回环地址；不要直接改为公网暴露。

#### Windows x64：源码 CPU 构建

准备 Visual Studio 2022 MSVC 工具链和 Windows SDK，在仓库根目录的 PowerShell 中运行：

```powershell
xmake f -p windows -a x64 --toolchain=msvc -m release --with_dml=n --with_cuda=n --with_tensorrt=n -y
xmake build server
Copy-Item app/assets/test/* build/windows/x64/release/assets/ -Recurse -Force
Set-Location build/windows/x64/release
.\vision_simple-server.exe
```

#### Linux x86_64：源码 CPU 构建

以下使用与 CI 相同的 GCC 14 工具链；请先安装编译器、Python 开发工具和依赖包构建工具。

```sh
xmake f -p linux -a x86_64 --toolchain=gcc --cc=gcc-14 --cxx=g++-14 -m release --with_cuda=n --with_tensorrt=n --with_rknpu=n -y
xmake build server
cp -R app/assets/test/. build/linux/x86_64/release/assets/
cd build/linux/x86_64/release
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ./vision_simple-server
```

默认监听 `127.0.0.1:11451`，使用 CPU。服务从**工作目录**读取 `config/server.yaml`、`config/models.yaml` 和日志配置；模型相对路径也从这里解析。上面的命令按默认输出目录编写，自定义输出时请使用实际产物目录。

构建会复制基础配置和主资源，但 server 目标不会自动复制测试模型，所以这里单独复制了测试资源。正式部署只需准备配置引用的权重和字典。再次构建可能覆盖配置，建议使用独立部署目录；修改配置或权重后重启服务。

### 3. 检查服务，再发起首次推理

保留服务终端，另开终端，在仓库根目录运行客户端。先检查连接：

```sh
curl http://127.0.0.1:11451/livez
curl http://127.0.0.1:11451/v1/models
```

PowerShell 使用 `curl.exe`，不要使用 `curl` 别名。`/livez` 只说明服务存活，模型目录只说明配置可发现；下一步的真实图片请求才能验证模型加载和推理。

下面的客户端只依赖 Python 标准库，保存为 `infer.py` 后，在仓库根目录执行 `python infer.py`。默认图片和模型来自刚才下载的测试资源，也可以把图片换成自己的：

```python
import base64
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

image_path = Path("app/assets/test/hd2.png")
payload = {
    "model": "hd2-fp32",
    "images": [base64.b64encode(image_path.read_bytes()).decode("ascii")],
    "confidence": 0.1,
    "nms_iou": 0.3,
}
request = Request(
    "http://127.0.0.1:11451/v1/infer/yolo",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urlopen(request, timeout=120) as response:
        print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
except HTTPError as error:
    print(error.code, error.read().decode("utf-8"))
    raise
```

成功响应包含 `class_names` 和按图片顺序排列的 `results`；`bbox` 为原图像素 `[x,y,width,height]`。某张图没有目标时，其结果是空数组，不是错误。原生接口的 `images` 使用**原始 base64**，不是 data URL。

试 OCR 时，将端点改为 `/v1/infer/ocr`、模型改为 `ppocr-v4`，并**删除** `confidence` 和 `nms_iou`。OCR 响应包含 `results`；这两个 YOLO 控制字段即使等于默认值，也会被 OCR 拒绝。

## 模型与推理

### 模型目录怎么配置

在服务工作目录的 `config/models.yaml` 中，用 `models` 列表声明模型：

```yaml
models:
  - task: yolo
    name: hd2-fp32
    version: kV11
    files:
      model: assets/hd2-yolo11n-fp32.onnx
  - task: ocr
    name: ppocr-v4
    version: kPPOCRv4
    files:
      det: assets/ppocr_det.onnx
      rec: assets/ppocr_rec.onnx
      dictionary: assets/ppocr_keys_v1.txt
```

模型由 `(task,name)` 唯一标识：同一任务不能重复声明同名模型，不同任务可以同名。原生 HTTP 和 MCP 推理使用配置原名；OpenAI-like 使用 `yolo:hd2-fp32` 这样的 `<task>:<name>` ID。旧 `yolo`/`ocr` 配置仍可读取，新配置建议统一使用 `models`。

当前任务是 `yolo`、`ocr`、`seg`、`pose`、`obb`，未知任务会被拒绝。模型按需加载，版本、文件和导出契约在加载时检查；目录里列出来不代表能加载。不支持热加载、TVM 或动态插件注册。

### 分割、姿态和旋转框

为所需任务添加模型条目，例如：

```yaml
models:
  - task: seg
    name: segment
    version: kV11
    files: {model: assets/yolo11n-seg.onnx}
  - task: pose
    name: pose
    version: kV11
    files: {model: assets/yolo11n-pose.onnx}
  - task: obb
    name: oriented
    version: kV11
    files: {model: assets/yolo11n-obb.onnx}
```

这些文件名是配置示例，不表示仓库附带对应权重。分别调用 `/v1/infer/seg`、`/v1/infer/pose`、`/v1/infer/obb`，正文格式与检测相同，模型名改为对应条目。

| 任务 | 结果中额外的几何信息 |
| --- | --- |
| `seg` | 整数 `bbox` 和 `mask_png_base64`；mask 是裁剪至框的 0/255 二值 PNG，放回原图时以框左上角为原点，不是全图 mask |
| `pose` | `bbox` 和 `keypoints:[{x,y,confidence}]`；关键点是原图浮点像素，可以在图像范围外 |
| `obb` | 四个有序 `corners` 和沿角点 0 → 1 的弧度 `angle`；不转换为轴对齐框，角点可以在图外 |

YOLO11 使用静态 `[1,3,H,W]` 输入、FP32/FP16 raw 输出和类别名称元数据；上述三种任务不接受内嵌 NMS。分割还需要匹配的 prototype，姿态需要关键点形状，OBB 需要角度通道。不支持动态输入、模型 batch > 1、分类、语义分割、深度估计或 YOLOE。

### 使用 YOLO26

YOLO26 使用 `version: kV26`，支持四种 YOLO 任务及 raw / NMS-free 导出。首次使用建议选 **FP32 + NMS-free**，先跑通检测。

在仓库根目录安装导出用依赖，再导出检测模型；这些不是服务运行依赖，首次运行会下载官方 nano 权重：

```sh
python -m pip install ultralytics==8.4.159 onnx==1.20.1 onnxruntime==1.24.3 torch==2.14.0 torchvision==0.29.0
python scripts/export_yolo26.py --output build/yolo26 --imgsz 640 --tasks detect --precisions 32
```

生成 `detect_raw_fp32.onnx` 和 `detect_e2e_fp32.onnx`。将选中的文件复制到服务的 `assets/`，在现有 `models` 列表追加条目后重启：

```yaml
  - task: yolo
    name: yolo26n
    version: kV26
    files: {model: assets/detect_e2e_fp32.onnx}
```

首次推理示例只需把 `model` 改为 `yolo26n`。导出参数 `detect` 对应服务任务 `yolo`；其他任务使用 `seg`、`pose`、`obb`。省略 `--tasks` 和 `--precisions` 时，脚本导出四任务 × 两种模式 × FP32/FP16，共 16 个模型，并生成含依赖版本、SHA256 和张量信息的 manifest。

| 任务 | raw 输出 | NMS-free 输出 |
| --- | --- | --- |
| `yolo` | `[1,4+nc,A]` | `[1,K,6]`：`xyxy,score,class_id` |
| `seg` | `[1,4+nc+nm,A]` + prototype | `[1,K,6+nm]` + prototype |
| `pose` | `[1,4+nc+nk*nd,A]` | `[1,K,6+nk*nd]`，`nd=2/3` |
| `obb` | `[1,5+nc,A]` | `[1,K,7]`：**`xywh,score,class_id,angle`** |

输入和输出需为静态、单张的 FP32/FP16 张量；`A`、`K` 不固定。保留 `names`、正确的 `task`、显式 `end2end` 和 `args.nms` 元数据，pose 还要有 `kpt_shape`。服务不会凭文件名或输出形状猜模式，缺失、矛盾的元数据或内嵌 NMS 会被拒绝。导出脚本使用固定 batch=1、opset=17、dynamic=False；不要直接套用其他导出器同名参数的含义。

已有 Windows x64 验证记录中，CPU 通过四任务的 raw/NMS-free FP32/FP16；DirectML（C++ ORT 1.20.0 / DML 1.15.4）通过 FP32 和 raw FP16，但 NMS-free FP16 在初始化时失败。**该 DML 环境请使用 FP32 或 raw FP16。** 这不是本次文档改写重新跑出的矩阵，也不覆盖 CUDA/TensorRT；部署前仍需用自己的模型、图片和设备验证。

YOLO26 权重和导出产物不随仓库分发，请单独核对 Ultralytics 软件与模型许可证。

### OCR 模型与调节参数

PP-OCR v3/v4 使用 CTC。Paddle SAR 使用独立解码器，但只支持单 float `[N,3,H,W]` 输入、单 float `[N,T,C]` 概率输出及 RGB、`[-1,1]` 归一化的识别契约；需要额外 attention/valid-ratio 输入的模型不能直接使用。EasyOCR 创建时明确返回不支持。

CTC 文件字典不包含 blank 和末尾空格类别，加载器会补空格；直接传 C++ map 时须提供全部非 blank 类别。SAR 字典只列普通字符，特殊类别在其后为 UKN、BOS/EOS、PAD，输出类别数为 `D+3`；不要混用两种字典约定。

服务选项 `ocr_rec_batch_size: "4"` 可启用识别裁剪批处理，范围 1–64、默认 1。动态 N 按预处理宽度分组，固定 N 服从模型声明并忽略尾批补样本；批大小收益取决于模型、图片和执行提供者。它不是 HTTP 图片批次上限。

文字检测的形态学参数属于**模型构造配置**，不是请求参数。在 OCR 条目中与 `files` 同级添加 `ocr_detection`：

```yaml
    ocr_detection:
      kernel_size: 2
      dilation_iterations: 3
      min_box_area: 64
```

| 参数 | 范围 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `kernel_size` | 1–32 | 2 | 正方形膨胀核 |
| `dilation_iterations` | 0–8 | 3 | 膨胀次数；0 不膨胀 |
| `min_box_area` | 0–1048576 | 64 | unclip 前包围框面积须严格大于此值，不是轮廓面积 |

省略、null 或 `{}` 保留默认值；未知字段、非整数和越界值会被拒绝。C++ 使用 `OCRDetectionOptions`，在 `InferOCR::Create` 的 `device_id` 后传入；模型保存参数快照，没有运行时 setter。请用实际图片评估参数调整后的检测和识别效果。

## 接入方式

### C++：直接调用库

下面使用 CPU 跑一张 YOLO11 图片。项目需链接 vision-simple 及其依赖，按当前 SDK 重新编译：

```cpp
#include <vision_simple/Infer.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    using namespace vision_simple;
    auto context = InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kCPU);
    if (!context) return 1;
    auto model = InferYOLO::Create(**context, "assets/hd2-yolo11n-fp32.onnx",
                                   YOLOVersion::kV11);
    if (!model) return 1;
    auto image = cv::imread("assets/hd2.png");
    auto result = (*model)->Run(image, YOLOInferenceOptions{.confidence = 0.625f});
    if (!result) return 1;
    for (const auto& object : result->results) {
        std::cout << object.class_name << ' ' << object.confidence << '\n';
    }
    return 0;
}
```

- `Create` 和 `Run` 返回 `VSResult` / `InferResult`（`std::expected`），请检查错误，不把异常作为公开错误协议。
- 输入须为非空二维 `CV_8UC3`，可以是非连续 ROI；灰度、BGRA、浮点图片需由调用者先转换。
- YOLO 的 `Run` 使用 `YOLOInferenceOptions`，`{}` 采用默认值，旧标量 confidence 调用需要迁移。OCR 仍使用 `Run(image, float confidence)`。
- 分割、姿态和 OBB 使用 `InferYOLOTask`；`Create(context, path, task, version, device_id)` 要显式传 `kV11` 或 `kV26`，不要省略版本。
- YOLO `class_name` 引用模型元数据，模型必须比结果视图活得更久；分割 mask 自持有像素。

批量阶段调度使用 `InferPipeline::Run`，需要计时可选 `RunMeasured`；测量结果持有普通结果和计时记录，普通 `Run` 不自动开启测量。`PipelineControl` 提供 stop token/deadline。调用期间模型和图片须保持有效且不被修改；`Close()` 后先等待调用者退出，再销毁流水线。取消不会强制中断已经执行的 ORT 调用。

其他 SDK 迁移点：`HTTPServer::Run/StartAsync` 返回错误结果，调用者须检查；几何 helper 使用 `LetterboxTransform`、`Letterbox(image,target,transform)` 和 `ScaleCoords(transform,float_xyxy)`，旧几何签名及 `DataConverter` 调用需迁移。`Cvt` 支持 FP32/FP16 双向转换。

### 原生 HTTP：直接拿结构化结果

| 操作 | 接口 |
| --- | --- |
| 五种任务推理 | `POST /v1/infer/{task}` |
| 兼容检测/OCR 接口 | `POST /v0/infer/yolo`、`POST /v0/infer/ocr` |
| 全任务目录 | `GET /v1/models?limit=100`；用 `next_cursor` 原样作为下一页 `after` |
| 旧目录 | `GET /v0/infer/models`，仅列 `yolo` / `ocr` |

推理正文包含 `model`、`images`，可选 `timeout_ms`（1–300000）。YOLO 任务还接受有限数值 `[0,1]` 的 `confidence`、`nms_iou`：默认 confidence 为 0.125，raw 检测默认 IoU 为 0.3，其他 raw 任务为 0.45。end-to-end 模型校验但不应用 `nms_iou`，不执行第二次 NMS。

YOLO11/26 raw 检测以 `score > confidence` 筛选，end-to-end 检测及其他任务以 `score >= confidence` 筛选。raw 检测先在浮点模型空间按类别 NMS，再映射、裁剪和取整；OBB 使用多边形 IoU，不等同 Ultralytics 的概率 IoU。

成功时结果数量和顺序与输入一致；任一图片失败则整批失败，不返回部分结果。有效模型允许空数组，但仍校验控制字段。原生 v1 未知模型返回 404，v0 和 OpenAI-like 返回 400。MCP 和 OpenAI 图片请求要求至少一张图片。

```json
{"error":{"code":"invalid_image","message":"Image cannot be decoded","image_index":1}}
```

| HTTP 状态 | 常见 `error.code` |
| --- | --- |
| 400 / 404 | `invalid_request`、`unknown_model` |
| 400 | `invalid_image`、`image_limit_exceeded` |
| 500 | `model_load_failed`、`model_config_failed`、`inference_failed`、`internal_error` |
| 503 | `service_overloaded`、`service_unavailable`、`request_cancelled` |
| 504 | `request_timeout` |

按 HTTP 状态和 `error.code` 处理，不依赖 `message` 文案。图片错误的 `image_index` 从 0 开始；非图片错误为 null。请求控制先于模型查找和解码校验，模型/配置错误先于图像预算接纳错误。完整字段、边界和错误契约以 [OpenAPI](doc/openapi/server.yaml) 为准。

### OpenAI-like：兼容视觉 Chat Completions 子集

此接口支持 `yolo`、`ocr`、`seg`、`pose`、`obb` 五种推理任务，使用 `<task>:<name>` 模型 ID；跟踪和字幕使用各自的原生 HTTP 接口。它不是聊天生成模型，文字提示不改变视觉推理行为，返回的 `choices[0].message.content` 是完整任务结果的 JSON 字符串。使用下面的 OpenAI Python SDK 示例前，先执行 `python -m pip install openai`。

```python
import base64
import json
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11451/v1", api_key="unused")
encoded = base64.b64encode(Path("app/assets/test/hd2.png").read_bytes()).decode("ascii")
response = client.chat.completions.create(
    model="yolo:hd2-fp32",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
    ]}],
    extra_body={"confidence": 0.1, "nms_iou": 0.3},
)
print(json.loads(response.choices[0].message.content))
```

SDK 是该客户端的额外依赖。它的 `extra_body` 会合并到请求顶层，不是线上 JSON 字段；`api_key` 在本服务中**不是认证凭证**。

接受内联 PNG/JPEG/WebP/BMP data URL，不抓取远程 URL，`detail` 仅支持省略或 `auto`；WebP 还取决于构建的 codec 支持。支持 `n:1`、text/json_object 响应格式及 `stream:true`。流式返回是在整批成功后发送完整结果和 `[DONE]`，不是逐 token 或逐图流；不支持生成参数、工具调用、JSON Schema 输出或 Responses API。

### MCP：传统 HTTP + SSE

客户端选择 **SSE** 传输并连接 `http://127.0.0.1:11451/mcp/sse`，不是 Streamable HTTP。仓库提供 [.mcp.json.example](.mcp.json.example)；先启动真实服务，再创建本地配置。不要覆盖已有配置：

```powershell
if (-not (Test-Path -LiteralPath .mcp.json)) {
    [IO.File]::Copy((Join-Path $PWD '.mcp.json.example'), (Join-Path $PWD '.mcp.json'), $false)
}
```

```sh
if [ ! -e .mcp.json ] && [ ! -L .mcp.json ]; then
    cp -nT .mcp.json.example .mcp.json
fi
```

Linux 示例使用 GNU coreutils。已有 `.mcp.json` 时，将 `vision-simple` 条目合并到现有 `mcpServers`；端口变更时修改 URL，保留 `/mcp/sse`。在仓库根目录打开 Claude Code，按客户端提示批准项目 MCP 服务；复制配置不会启动服务。

自行实现客户端时，从 `endpoint` 事件取得 POST 地址，完成 `initialize`、`notifications/initialized`，再调用工具。初始化需提供 `protocolVersion`、对象 `capabilities` 和含 `name`/`version` 的 `clientInfo`；支持 `2024-11-05`、`2025-03-26`、`2025-06-18`、`2025-11-25`。POST 202 只表示收到消息，JSON-RPC 结果从原 SSE 连接读取。

- `list_models`：分页发现配置模型。
- `infer_yolo`、`infer_ocr`、`infer_seg`、`infer_pose`、`infer_obb`：使用配置原名和原始 base64；YOLO 工具可传 `confidence` / `nms_iou`，OCR 不接受。没有跟踪或字幕工具。
- 新版本返回 `structuredContent` 及等价 JSON 文本，旧版本使用文本；执行失败为 `isError:true`，协议错误使用 JSON-RPC error。
- `notifications/cancelled` 按本会话 `requestId` 取消，断连取消本会话任务；原生阶段仍需结束后才能释放资源。

固定上限为 32 会话、2 执行 worker、16 排队任务、每会话 8 个活动工具请求，正文 64 MiB、待发送结果/写缓冲 8 MiB。每 15 秒心跳，持续写积压或空闲会关闭会话。Host/Origin 必须是受信任的后端 authority，端口也须匹配；代理需关闭 SSE 缓冲并允许长连接。这些检查不是用户认证。

### 跟踪：把检测结果串成轨迹

先创建会话，再提交每帧检测。下面以 ByteTrack 为例：

```text
POST /v1/tracking/sessions
{"algorithm":"bytetrack","options":{"min_hits":2}}

POST /v1/tracking/sessions/{id}/frames
{"frame_index":0,"timestamp":0.0,"detections":[{"class_id":0,"confidence":0.9,"bbox":[10,20,30,40]}]}
```

创建返回 201 和 `id`；推进返回 `tracks`，仅含本帧观测到的已确认轨迹。首帧轨迹立即确认，后续新轨迹受 `min_hits` 约束。还可 GET 状态/分页列表、POST `{id}/reset`、DELETE 会话。

- 同一会话的 `frame_index` 和以秒计的 `timestamp` 必须严格递增，被拒绝帧不推进状态；reset 清空序列和 ID。并发访问忙会话返回 409，不排队。
- BoT-SORT 的相机运动补偿需要每帧图片且尺寸一致；只提交 detections 时设置 `camera_motion:false`。`appearance:true` 需要每个检测提供一致维度、有限非零的 embedding；服务不附带 ReID 权重。
- 最多 32 会话，每帧最多 256 个检测、每会话最多 256 条轨迹，正文 4 MiB。会话在创建或最近成功 step/reset 后空闲 300 秒过期，读取不续期。
- 手动串联检测和跟踪时，可把 detector confidence 调低以保留 tracker 需要的低分候选；这是客户端策略，不是自动融合接口。

跟踪拒绝非空浏览器 Origin，没有用户隔离；请在可信网络或鉴权代理后使用。

### 视频字幕：提取画面里的文字

字幕任务复用 OCR 推理服务，流程是创建 → 上传 → 轮询 → 下载 → 删除。先配置可用的 OCR 权重与字典；`roi` 是归一化 `[x,y,width,height]`，默认取下半屏。

```sh
curl -sS -X POST http://127.0.0.1:11451/v1/subtitle/jobs -H 'Content-Type: application/json' -d '{"model":"ppocr-v4","sample_interval_ms":200,"roi":[0,0.5,1,0.5],"min_confidence":0.5,"stable_samples":2,"gap_samples":2}'
# 将 JOB_ID 换成创建响应中的 id；clip.avi 是自己的 MJPEG AVI 视频
curl -sS -X PUT http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID/video -H 'Content-Type: application/octet-stream' --data-binary @clip.avi
curl -sS http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID
```

创建选项直接放在顶层，不套 `options`。上传不接受 multipart、base64、本地服务器路径或远程 URL。202 只表示接受处理，不保证视频可解码；轮询直到 `completed`，若为 `failed` / `cancelled` 则检查 `error_code`，不要下载部分结果。上传中断后应新建任务，不能在原任务重试。

成功后，下面的标准库 Python 示例将 SRT 写入同目录临时文件，保存成功后才删除服务任务；Windows/Linux 均可使用：

```python
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

job_url = "http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID"
result = Path("clip.srt").resolve()
with urlopen(job_url, timeout=30) as response:
    import json
    if json.load(response)["state"] != "completed":
        raise RuntimeError("Job is not completed; server result retained")
fd, temporary = tempfile.mkstemp(dir=result.parent, prefix=result.name + ".download.")
try:
    with os.fdopen(fd, "wb") as output, urlopen(job_url + "/subtitles.srt", timeout=30) as response:
        while chunk := response.read(65536):
            output.write(chunk)
    os.replace(temporary, result)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
with urlopen(Request(job_url, method="DELETE"), timeout=30) as response:
    print("Saved", result, "deleted job:", response.status)
```

WebVTT 改用 `subtitles.vtt` 和 `clip.vtt`。要两种格式时，保存全部需要的文件后再删除。下载失败时保留原本地文件和服务任务，但服务结果仍受过期规则约束；成功但无字幕的 SRT 可以为空。

AVI 在两个平台使用可移植 MJPEG reader，仅支持对应 MJPEG 视频流；其他 AVI codec 会异步失败。MP4 仅限编译了 Media Foundation reader 的 Windows 构建，且依赖系统 codec；Linux 拒绝 MP4。MKV/ASF 不支持。

字幕服务单 worker，最多 8 个任务（包括等待上传和保留的终态结果）。每视频最多 64 MiB、1800 秒，每帧最多 16,777,216 像素。采样间隔 100–5000 ms，连续相同文字确认后才生成字幕，短暂噪声会被抑制。created/uploading 在最后接受活动 60 秒后可被清理，任务进入终态 300 秒后可被清理；读取和下载不续期。`expires_at` 是清理资格时间，不保证立即删除。运行任务先 POST `{id}/cancel` 并等待终态，再 DELETE。

## 部署与诊断

### 先明确安全边界

默认原生服务监听回环；Docker 在容器内监听 `0.0.0.0`，示例只映射宿主回环。公开访问需要 TLS 和外部鉴权代理。CORS、Origin、job/session ID、OpenAI SDK key 都不能代替用户认证或租户隔离。

只有 `GET /v0/infer/stats` 和 `POST /v0/infer/unload` 使用内建管理员策略，**默认关闭**，返回 `403 management_disabled`。如需启用，在部署配置的 `options` 中设置：

```yaml
  http_management_token_env: "VS_MANAGEMENT_TOKEN"
```

这只是环境变量名示例，不是默认名。启动前通过外部秘密管理向进程提供该变量，使用高熵、无空白的可见 ASCII 秘密；不要把秘密写入 YAML、镜像、URL 或日志。显式配置的变量缺失、为空或不安全时，服务启动失败，不回退到无认证；轮换后需重启。

请求发送 `Authorization: Bearer <secret>`，缺失/错误凭证返回 401。授权后还检查 Host/Origin：须匹配受信任的后端地址及端口，wildcard bind 不代表任意 Host 可信。不使用 Forwarded/X-* 作为授权依据。管理路径拒绝宽松 CORS；unload 需 `application/json`、正文最多 64 KiB。

推理代理默认应拒绝这两个管理路径，例如 nginx：

```nginx
location = /v0/infer/stats { return 403; }
location = /v0/infer/unload { return 403; }
```

确需管理转发时，单独做管理员鉴权，保留调用者 bearer，并将 Host/Origin 转为后端接受的 authority；不能向普通推理流量注入共享管理员秘密。旧 binary 可能忽略新 option、重新开放无鉴权管理，回滚前必须保持代理硬拒绝和后端隔离。

### 模型生命周期与资源限制

v0/v1、OpenAI-like、MCP 使用同一模型缓存和阶段流水线。模型首次请求加载，默认空闲 300 秒自动卸载；活动租约覆盖解码、推理和响应处理，不会卸载正在使用的实例。

授权后，stats 列出已加载实例的请求、失败、耗时及服务级 `image_budget`；不是配置目录。unload 接受 `{"kind":"yolo","model":"hd2-fp32"}`，支持五任务：空闲模型卸载返回 200，活动模型返回 `409 model_busy`，未加载返回 `404 model_not_loaded`，下次推理重新加载。实例统计在重载后重置，图像预算计数不随卸载重置。

常用 `config/server.yaml` 的 `options` 均为字符串：

| 选项 | 默认值 | 控制什么 |
| --- | --- | --- |
| `infer_idle_timeout_ms` / `infer_sweep_interval_ms` | `"300000"` / `"1000"` | 空闲卸载及扫描；timeout 为 `"0"` 关闭自动卸载 |
| `http_data_workers` / `http_data_queue_capacity` | `"4"` / `"4"` | 推理和视频上传的有界 HTTP 调度 |
| `http_control_workers` / `http_control_queue_capacity` | `"1"` / `"4"` | 管理与字幕元数据操作的独立有界调度 |
| `infer_pipeline_capacity` / `infer_pipeline_max_batches` | `"4"` / `"4"` | 驻留图片任务和已接纳批次，分别为 1–64 |
| `infer_max_batch_images` | `"128"` | 每批图片数，1–4096 |
| `infer_max_image_pixels` | `"16777216"` | 单张像素上限 |
| `infer_max_batch_decoded_bytes` | `"67108864"` | 单批解码输入字节，64 MiB |
| `infer_max_inflight_decoded_bytes` | `"268435456"` | 服务共享解码输入字节，256 MiB |
| `infer_timeout_ms` | `"60000"` | 默认推理 deadline，1–300000 ms |
| `ocr_rec_batch_size` | `"1"` | OCR 识别裁剪批大小，1–64 |

解码预算按 `width × height × 3` 估算 BGR 输入，在解码前检查 header 和预留整批额度；不等于 RSS、模型工作区、codec 临时内存或输出 mask 的总上限。超额图片返回 400，全局接纳不足返回 503 和 `Retry-After: 1`。跟踪服务有自己的限制，不能把上述预算当成全服务统一内存上限。

`/livez` 表示存活，`/readyz` 表示未停止且可接纳；二者绕过推理队列，不加载或预热模型，暂时队列满不改变 ready。HTTP deadline 从完整正文提交调度开始，包含排队、加载和解码；不能硬中断原生调用，取消/超时后仍需等资源实际排空。并发请求使用不同连接，顺序 keepalive 可用，不支持活动异步请求期间的同连接 pipeline 请求。

卸载释放 session 和工作区，但 ORT arena / provider 可能保留内存，不保证 RSS 或显存立即下降。

### 部署前做预检和预热

从准备好 `config/`、`assets/` 的服务工作目录运行，Windows 将可执行文件换为 `.\vision_simple-server.exe`：

```sh
./vision_simple-server --help
./vision_simple-server --diagnose preflight --model yolo:hd2-fp32 --model ocr:ppocr-v4
./vision_simple-server --diagnose warmup --model yolo:hd2-fp32 --image assets/hd2.png --timeout-ms 60000
```

`preflight` 检查必需文件并真实加载选定会话，不执行图片推理。`warmup` 在新服务缓存上执行一次完整 cold 批次，成功后重复一次 warm 批次；不启动 HTTP listener，也没有后台保温。选择 1–16 个不同的 `task:name`，默认不会加载整个目录。

JSON 报告区分配置存在、可加载和图片 smoke 成功，并记录加载、解码、预处理、排队、推理和后处理计时；阶段可以重叠，不能把计时简单相加当作批次耗时。报告只对本次调用有效，不证明 HTTP readiness 或 GPU placement。退出码 0 为所选目标及卸载成功，1 为操作失败，2 为参数错误。

OCR 调试时可显式导出预热图片与框叠加：

```sh
./vision_simple-server --diagnose warmup --model ocr:ppocr-v4 --image assets/hd2.png --debug-dir ocr-study-001 --debug-max-bytes 67108864 --debug-max-files 64
```

仅允许单个 OCR 模型、1–16 张图片；`--debug-dir` 是可信工作目录下**尚不存在**的新子目录名，不是任意路径。最多 64 MiB、64 文件，包含原图、框叠加和 manifest，不承诺检测 mask 或识别裁剪。只使用获准保存的图片，PNG 仍含敏感像素；默认不导出。退出 0 且 `debug.retained:true` 后可检查，之后由使用者清理这个确切的新目录，不要删除父目录或使用通配符。

## 构建变体与验证

### 执行提供者和平台

编译启用 provider 后，还要在运行配置的字符串 `options.infer_ep` 中选择它，`infer_device` 指定设备；默认仍是 `kCPU`。Windows DML 构建选项默认开启，不代表运行自动选择 DML。

| Provider | 构建选项 | 运行值 | 额外前提 |
| --- | --- | --- | --- |
| CPU | 快速上手的 CPU 配置 | `kCPU` | 匹配的 ONNX Runtime |
| DirectML | `--with_dml=y` | `kDML` | Windows、匹配的 DML 运行库和设备 |
| CUDA | `--with_cuda=y` | `kCUDA` | CUDA 运行库、驱动和硬件 |
| TensorRT | `--with_tensorrt=y` | `kTensorRT` | 匹配的 TensorRT/CUDA 及模型支持 |
| RKNPU | `--with_rknpu=y` | `kRKNPU` | Linux ARM/ARM64、RKNPU-enabled ORT、DDK/驱动及兼容模型 |

Windows 选择 CUDA/TensorRT 时同时设置 `--with_dml=n`，避免默认 DML 分支选择不同的 ORT 包。`InferContext::Capabilities()` 可查询编译和 runtime provider 能力；上下文创建成功、provider 可见、模型 smoke 成功是不同层次，CPU fallback 允许，均不证明所有算子在指定设备执行。

CI 使用 xmake 2.9.7 的 MSVC/GCC 配置，以及 xmake 3.1.1 的 Clang 18 + libc++ 配置。Linux Clang 可用以下命令替换快速上手的 GCC 配置，后续步骤相同：

```sh
xmake f -p linux -a x86_64 --toolchain=clang --cc=clang-18 --cxx=clang++-18 -m release --runtimes=c++_shared --with_cuda=n --with_tensorrt=n --with_rknpu=n -y
```

ARM64、RISC-V64 交叉构建分别使用 `--cross=aarch64-linux-gnu-`、`--cross=riscv64-linux-gnu-` 和相应 `-a`，还需配置目标依赖。交叉构建和产物架构检查不代替目标设备推理验收。也可从 [Releases](https://github.com/lona-cn/vision-simple/releases) 选择平台/架构/EP 匹配的归档，核对校验和和 `build-info.json`；归档不保证附带全部模型。

### GHCR 发布镜像

发布工作流构建并 smoke Linux CPU **amd64 / arm64**，合并到 `ghcr.io/lona-cn/vision-simple` 多架构 manifest。稳定版本更新 `latest`，预发布不更新；还有 `<version>-cpu`、`sha-<完整 commit SHA>-cpu` 和各平台专属标签。

```sh
docker pull ghcr.io/lona-cn/vision-simple:latest
docker run -it --rm -p 127.0.0.1:11451:11451 ghcr.io/lona-cn/vision-simple:latest
```

部署优先锁定成功发布摘要中的 `@sha256:...` digest，确认所选版本具备需要的功能。首次发布的 GHCR 包可能为 private，匿名拉取需先设置 public。amd64 镜像同样要求 AVX/AVX2/F16C。

ARMv7、RISC-V、CUDA/TensorRT 和 RKNPU 有独立 Dockerfile，但不属于上述 CPU manifest，不能据此声称已在对应设备完成运行验收。硬件 provider 镜像不能和同架构 CPU 镜像混作同一个自动选择变体。

### 运行验证

从仓库根目录另开终端，先完成 CPU 构建、Git LFS 资源下载和夹具预检：

```sh
python scripts/check_ci_fixtures.py --project-root . --layer native
python scripts/check_ci_fixtures.py --project-root . --layer http
xmake build test_yolo_postprocess
xmake run test_yolo_postprocess
xmake build test_pipeline
xmake run test_pipeline --project-root .
```

C++ 测试是普通可执行文件；还有 OCR batch/形态学、YOLO 多任务、配置、跟踪、字幕时间线与图像预算等目标。合成 OCR 几何研究可运行 `test_ocr_morphology_dataset --project-root .`，结果只代表该冻结样本协议，不是生产准确率。`test_yolo` / `test_ocr` 是交互演示，不作为 headless 验收。

文档校验需要 Python 3.10+，建议在虚拟环境中安装测试依赖。在 Windows 默认 CPU 构建下运行：

```powershell
python -m pip install -r scripts/requirements-doc-tests.txt
$server = 'build/windows/x64/release/vision_simple-server.exe'
python scripts/test_documentation_validation.py
python scripts/test_documentation_examples.py --server "$server" --project-root .
python scripts/test_diagnostics.py --server "$server" --project-root .
python scripts/test_protocol_regression.py --server "$server" --project-root .
```

Linux 可查询当前 xmake 配置的实际可执行文件，再运行相同 driver：

```sh
server="$(xmake lua -q -c "import('core.project.config'); config.load(); import('core.project.project'); io.write(path.absolute(project.target('server'):targetfile()))")"
python3 scripts/test_documentation_examples.py --server "$server" --project-root .
```

文档 driver 验证 OpenAPI/schema 示例、原样 YOLO/OCR 请求及 MCP 初始化/工具发现，**不自动执行 README 的所有代码块**。HTTP driver 创建隔离配置、临时端口和进程，不触碰已有 11451 服务；缺少模型或动态库会失败，不作为跳过成功。

其他场景见 [回归脚本](scripts/) 和 [CI 配置](.github/workflows/ci.yml)：管理授权、调度、模型目录、跟踪、字幕媒体、图像预算等均有对应 driver。字幕媒体回归另需 FFmpeg 和字体，Windows MP4 场景还需 Media Foundation。原生 CPU 回归、交叉编译架构检查、Docker smoke 和真实硬件推理是不同证据，不互相替代。

## 许可证

本项目使用 **Apache-2.0** 许可证。YOLO、PaddleOCR 模型及其他第三方资源的版权和许可归各自项目所有；使用、导出或部署权重时，请同时核对它们的许可要求。
