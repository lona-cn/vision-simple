# <div align="center">🚀 vision-simple 🚀</div>
[english](./README-en.md) | 简体中文

<p align="center">
<a><img alt="GitHub License" src="https://img.shields.io/github/license/lona-cn/vision-simple"></a>
<a><img alt="GitHub Release" src="https://img.shields.io/github/v/release/lona-cn/vision-simple"></a>
 <a href="https://github.com/users/lona-cn/packages/container/package/vision-simple"><img alt="GHCR image" src="https://img.shields.io/badge/GHCR-vision--simple-2496ED"></a>
<a><img alt="GitHub Downloads (all assets, all releases)" src="https://img.shields.io/github/downloads/lona-cn/vision-simple/total"></a>
</p>

<p align="center">
<a><img alt="" src="https://img.shields.io/badge/yolo-v10-AD65F1.svg"></a>
<a><img alt="" src="https://img.shields.io/badge/yolo-v11-AD65F1.svg"></a>
<a><img alt="YOLO26" src="https://img.shields.io/badge/yolo-26-AD65F1.svg"></a>
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

`vision-simple` 是基于 C++23 和 ONNXRuntime 的跨平台视觉推理库，提供 C++ API 与独立 HTTP 服务。支持 YOLO 检测、实例分割、姿态、旋转框及 OCR，并提供检测结果跟踪、视频画面文字提取、OpenAI-like 和 MCP SSE 接入。

本文面向希望构建、部署并发起首次推理的使用者，描述**当前源码**，不表示历史发布包或镜像已包含全部功能。

**快速入口**：[构建与启动](#构建项目) · [容器部署](#docker部署http服务) · [模型配置](#任务注册与统一模型配置) · [YOLO26](#yolo26yolov26) · [协议接入](#统一服务openai-like-与-mcp-sse) · [运行测试](#运行测试)

## 特性

- **推理任务**：YOLOv10/v11/26 检测；YOLO11/26 实例分割、姿态、旋转框；PP-OCR v3/v4 CTC 与受限 SAR recognition 契约。
- **有界服务**：模型按需加载、活动租约、空闲卸载、统计、分阶段流水线与合作式取消。
- **时序处理**：ByteTrack / BoT-SORT 跟踪会话；异步 OCR 视频字幕任务，输出 SRT/WebVTT，不包含语音转写。
- **部署平台**：Windows x64、Linux x86_64，以及 ARM64、ARMv7、RISC-V 64 交叉构建配置。交叉构建不等于目标硬件运行验证。
- **执行提供者**：CPU、DirectML、CUDA、TensorRT、RKNPU 构建选项；实际可用性取决于依赖、硬件和模型导出，见兼容性说明。不承诺固定体积、内存占用或帧率。
- **接入方式**：
  - C++23 API，使用 `std::expected` 返回错误。
  - [容器部署](https://github.com/users/lona-cn/packages/container/package/vision-simple)，GHCR 发布 Linux amd64、arm64 CPU 多架构镜像。
  - **[HTTP服务](doc/openapi/server.yaml)**：提供HTTP API供Web应用调用


### <div align="center"> YOLOv11 </div>
![hd2-yolo-gif](doc/images/hd2-yolo.gif)

### <div align="center"> OCR(HTTP API) </div>

![http-inferocr](doc/images/http-inferocr.png)
## 快速使用

### docker部署HTTP服务

使用当前源码构建 CPU 镜像，避免将历史 `0.4.1-cpu-x86_64` 标签误当作最新功能：

```sh
git clone --recurse-submodules https://github.com/lona-cn/vision-simple.git
cd vision-simple
git lfs install
git lfs pull
docker build --platform linux/amd64 -t vision-simple:local -f docker/Dockerfile.debian-bookworm-x86_64-cpu .
docker run -it --rm --name vs -p 127.0.0.1:11451:11451 vision-simple:local
```

需要 Docker、Git LFS 及支持 AVX/AVX2/F16C 的 x86_64 CPU。首次构建会下载并编译依赖；镜像包含默认 YOLO11/PP-OCR 配置及测试模型，不包含 YOLO26 权重。发布镜像的选择与校验见[GHCR 多平台发布](#ghcr-多平台发布)。

在另一终端执行 `curl http://127.0.0.1:11451/v0/infer/models`（Windows 可用 `curl.exe`）检查模型目录。目录成功只证明配置可发现，不证明权重加载或推理成功；完整请求示例见[发起推理](#发起推理)，默认检测模型改用 `hd2-fp32` 即可。

完整接口见 [OpenAPI](doc/openapi/server.yaml)。服务**没有认证或租户隔离**；不要直接暴露到公网。源码默认监听 `0.0.0.0`，容器示例仅发布回环端口；原生运行请将 `host` 改为 `"127.0.0.1"` 或使用鉴权代理。

### HTTP v0 错误与批量语义

成功字段保持不变：YOLO 返回 `class_names`/`results`，OCR 返回 `results`。`model` 必须为非空字符串，`images` 必须为字符串数组；有效模型接受空数组。HTTP 200 保证结果数量与输入数量相等、顺序一致；某张图没有目标时，该项为空数组。

任一图片失败即整批失败，不返回部分结果。处理顺序为请求校验、模型查找/加载、全部图片解码、有界流水线推理、序列化；返回首先失败的阶段及该阶段最小图片索引。

```json
{"error":{"code":"invalid_image","message":"Image cannot be decoded","image_index":1}}
```

| HTTP | `error.code` | `image_index` |
| --- | --- | --- |
| 400 | `invalid_request`、`unknown_model` | `null` |
| 400 | `invalid_image` | 从 0 开始的图片索引 |
| 500 | `model_load_failed`、`model_config_failed`、`internal_error` | `null` |
| 500 | `inference_failed` | 从 0 开始的图片索引 |
| 503 | `service_overloaded`、`service_unavailable`、`request_cancelled` | `null` |
| 504 | `request_timeout` | `null` |

旧客户端需从“HTTP 200 + 文本错误”迁移为检查 HTTP 状态和 `error.code`，不能依赖 `message` 文案。第三方异常细节仅保留在日志。完整契约见 [OpenAPI](doc/openapi/server.yaml)。

### 模型生命周期与并发

- `POST /v0/infer/unload` 接受 `{"kind":"yolo","model":"hd2-fp32"}`；`kind` 支持 `yolo`、`ocr`、`seg`、`pose`、`obb`。空闲模型卸载返回 `200 {"kind":"yolo","model":"hd2-fp32","unloaded":true}`；活动模型返回 `409 model_busy`；未加载返回 `404 model_not_loaded`。后续推理自动重新加载。卸载和 stats 覆盖五种任务，只有旧 `/v0/infer/models` 目录限于 `yolo`/`ocr`。
- `GET /v0/infer/stats?limit=100&offset=0` 返回 `models`、`total`、`limit`、`offset` 和 `idle_timeout_ms`；limit 范围 1–200。模型按 `(kind,name)` 排序，字段为 `kind`、`name`、`active_requests`、`requests`、`failures`、`total_duration_ms`、`last_used`（Unix 毫秒）。计数属于当前已加载实例，重新加载后重置；加载前的请求错误不计入实例统计，耗时含等待工作区的时间。
- `config/server.yaml` 的字符串 options：`infer_idle_timeout_ms: "300000"`，`infer_sweep_interval_ms: "1000"`。空闲时间从最后一次请求结束计算，使用单调时钟；timeout 为 `"0"` 关闭自动卸载，扫描间隔必须为正整数。
- 活动租约涵盖解码、等待推理、后处理及响应序列化/发送调用；手动和定时卸载均不删除活动实例。同步 C++ `Run` 每模型串行；HTTP 流水线使用独立任务工作区，同一会话的 ORT 执行受锁保护。
- 卸载释放 session 和模型工作区，但共享 ORT arena / provider 可能保留内存，不保证 RSS 或显存立即下降。YOLO 结果的 `class_name` 仍引用模型元数据，C++ 调用者须让模型活得比结果视图更久。
- 生命周期接口沿用现有无认证/CORS 策略，仅应部署在可信网络或受鉴权代理保护的环境。4 个 HTTP IO worker 全被推理占用时，管理请求会等待；目前没有保留独立管理通道。

### 有界推理流水线

HTTP v0 在全部图片解码成功后，使用前处理、ORT、后处理三个独立 worker；OCR 按检测及文本框 recognition minibatch 的依赖关系轮转阶段。图片结果按输入索引聚合；不同图片即使乱序完成，仍只返回最低失败索引，整批不返回部分结果。

服务 options 支持 `infer_pipeline_capacity: "4"`（驻留阶段任务数，1–64）、`infer_pipeline_max_batches: "4"`（已接纳批次数，1–64）、`infer_max_batch_images: "128"`（每批图片上限，1–4096）、`infer_timeout_ms: "60000"`（默认推理截止时间，1–300000 毫秒）。这些限制不等价于图像像素/字节数上限。

请求可附加整数 `timeout_ms`（1–300000）覆盖默认截止时间。计时从 handler 开始，包含模型加载和解码；截止时间控制推理阶段，不对序列化、网络传输或原生调用作硬实时保证。超时返回 `504 request_timeout`，容量耗尽返回 `503 service_overloaded` 和 `Retry-After: 1`；控制错误的 `image_index` 为 null。解码仍在调度前完成，保留 `invalid_image` 优先级。

C++ 调用者可使用 `InferPipeline::Create`，再调用 `Run(model, images, confidence, PipelineControl{stop_token, deadline})`；`Close()` 拒绝新任务并取消未完成批次，销毁前必须等待所有调用者退出。取消优先于超时，超时优先于关闭及普通推理错误。取消仅阻止后续阶段，`Run` 等待已经执行的原生阶段及任务析构完成后才返回，避免活动模型被释放。输入像素及模型须在整个调用期间保持有效，外部不得修改像素。普通 v0/v1 HTTP 断连不自动取消正在运行的任务；MCP 会话断连会发出合作式取消。

同步 `InferYOLO/InferOCR::Run` 签名不变，与流水线共享阶段算法和会话执行锁；流水线任务拥有独立工作区，每个模型最多缓存 2 个空闲流水线工作区。未加入流水线适配的自定义模型返回明确错误，不以同步调用伪装分阶段执行。

### 任务注册与统一模型配置

在 `config/models.yaml` 中声明模型，使用 `(task,name)` 标识。原生 v1 与 MCP 使用配置原名，OpenAI-like 使用 `<task>:<name>`。以下为默认配置的 FP32 检测与 OCR 条目：

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

旧 `yolo`/`ocr` 配置仍可读，也可与不冲突的新条目混用。重复 `(task,name)` 一律拒绝；不同任务允许同名。版本和模型资源检查仍在首次模型加载时执行。旧配置 DTO 保留兼容投影，执行路径只读取 canonical `models`，没有两套缓存或配置查找逻辑。

当前注册 `yolo`、`ocr`、`seg`、`pose`、`obb`，未知 task 在服务配置边界拒绝。各任务共用缓存、租约、统计和回收逻辑。实际推理后端为 ONNXRuntime，不支持 TVM；任务注册表不是动态插件系统。

### YOLO 分割、姿态与旋转框

使用自行提供的兼容导出模型添加 canonical 配置（以下文件名不表示仓库附带权重）：

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

- YOLO11 导出须具有静态 `[1,3,H,W]` 输入、静态 FP32/FP16 原始输出及类别名称元数据，不得包含导出 NMS/end-to-end 后处理。YOLO26 的两种导出模式见下节。分割需要预测和通道匹配的 mask prototype；姿态使用关键点通道（`kpt_shape` 支持 2 或 3 个分量）；OBB 需要一个角度通道。不支持动态维度、batch>1、任意 YOLO 架构或内嵌 NMS 图。
- 五个 task 均可调用 `POST /v1/infer/{task}`，正文为 `{"model":"配置原名","images":["原始base64"],"timeout_ms":60000}`。共用按输入顺序、整批失败语义和流水线限制；原生 v1 正文限制为 64 MiB。既有 v0 推理响应不变。
- 新任务返回 `class_names` 及逐图 `results`，各目标含 `class_id` 和 `confidence`。分割额外返回原图整数 `bbox:[x,y,width,height]` 和 `mask_png_base64`：裁剪至该框的 0/255 二值 PNG，并非全图 mask，放置时以框左上角为原点。C++ `InferYOLOTask` 使用独立 `YOLOTaskFrameResult` variant；分割 `CV_8UC1` mask 自持有像素，不依赖推理工作区。
- 姿态额外返回同样的框及 `keypoints:[{x,y,confidence}]`，坐标为原图浮点像素，不裁剪至图像范围。OBB 返回四个有序原图 `corners:[[x,y],...]` 和沿角点 0 → 1 的弧度 `angle`；角点可在图外，不转换为轴对齐包围框。
- OpenAI-like 使用带任务前缀的 ID，例如 `seg:segment`、`pose:pose`、`obb:oriented`；MCP 在旧工具之外生成 `infer_seg`、`infer_pose`、`infer_obb`。结果 JSON 保留各任务的几何信息。

### YOLO26（YOLOv26）

`kV26 = 26` 支持检测、实例分割、姿态与旋转框，保留 YOLOv10/YOLO11 的既有行为。模型须为单张、静态尺寸的 ONNX，输入/输出张量支持 FP32 或 FP16；FP16 权重不意味着所有 I/O 都是 FP16。

首次使用建议选择 **FP32 + NMS-free**，先跑通单个检测模型，再扩展其他任务。需要包含 YOLO26 实现的服务构建；历史发布包或 Docker 标签不代表已包含此功能。

| 服务 task | 原始输出（外部 NMS） | NMS-free 输出（不再执行 NMS） |
|---|---|---|
| `yolo` | `[1,4+nc,A]`，`xywh` + 类别分数 | `[1,K,6]`，`xyxy,score,class_id` |
| `seg` | `[1,4+nc+nm,A]` + prototype | `[1,K,6+nm]` + `[1,nm,Hm,Wm]` prototype |
| `pose` | `[1,4+nc+nk*nd,A]` | `[1,K,6+nk*nd]`，`nd=2/3` |
| `obb` | `[1,5+nc,A]` | `[1,K,7]`，**`xywh,score,class_id,angle`** |

`A`、`K` 不固定为 8400、300。YOLO26 必须保留 `names`、正确的 `task`、显式 `end2end` 和 `args.nms` 元数据；pose 还必须有 `kpt_shape`。缺失、矛盾、非法 shape、内嵌 NMS 均明确拒绝，不根据文件名或仅凭 `[1,K,6]` 猜测模式。

#### 导出模型

在仓库根目录执行。导出依赖仅用于开发，不是服务运行依赖；首次运行会下载官方 nano 权重。

```bash
python -m pip install ultralytics==8.4.159 onnx==1.20.1 onnxruntime==1.24.3 torch==2.14.0 torchvision==0.29.0
python scripts/export_yolo26.py --output build/yolo26 --imgsz 640
```

仅使用目标检测时，无需导出其余任务或半精度模型：

```bash
python scripts/export_yolo26.py --output build/yolo26 --imgsz 640 --tasks detect --precisions 32
```

此命令生成 `detect_raw_fp32.onnx` 和 `detect_e2e_fp32.onnx`。导出参数 `detect` 对应服务配置中的 `task: yolo`；其余任务均使用 `seg`、`pose`、`obb`。可在 `--tasks` 后指定多个任务，在 `--precisions` 后指定 `32`、`16` 或两者。

不指定 `--tasks`、`--precisions` 时，脚本默认导出四任务 × raw/e2e × FP32/FP16 共 16 个模型，固定 batch=1、opset=17、dynamic=False、simplify=False；生成的 `manifest.json` 记录依赖版本、权重与 ONNX SHA256、导出参数及实际张量元数据。该版本用 `nms=None` 选择 raw，`nms=False` 选择 NMS-free，`quantize=16` 选择半精度。CPU 半精度转换后会拓扑排序节点，修正转换器追加 I/O Cast 的顺序，然后执行 ONNX checker 和 CPU ORT 加载验证。不要删除模式元数据，也不要把旧导出器的同名参数语义直接套用。

#### 配置与启动

服务从**工作目录**读取 `config/server.yaml` 和 `config/models.yaml`，模型的相对路径也以工作目录为基准。将所需 ONNX 放入该目录的 `assets/`，在 `config/models.yaml` 的 `models` 列表中添加以下条目；只保留实际导出的模型，不要重复声明同一个 `(task,name)`。

源码构建会复制基础配置到可执行文件旁的 `config/`。Windows x64 Release 默认输出目录为 `build/windows/x64/release/`；部署时从包含 `config/`、`assets/` 的目录启动服务。再次构建可能重新复制基础配置，正式部署建议使用独立目录。

```yaml
models:
  - task: yolo
    name: yolo26n
    version: kV26
    files: {model: assets/detect_e2e_fp32.onnx}
  - task: seg
    name: yolo26n-seg
    version: kV26
    files: {model: assets/seg_e2e_fp32.onnx}
  - task: pose
    name: yolo26n-pose
    version: kV26
    files: {model: assets/pose_e2e_fp32.onnx}
  - task: obb
    name: yolo26n-obb
    version: kV26
    files: {model: assets/obb_e2e_fp32.onnx}
```

在 `config/server.yaml` 中选择已验证的执行提供者，例如 CPU：

```yaml
host: "127.0.0.1"
port: 11451
options:
  infer_framework: "kONNXRUNTIME"
  infer_ep: "kCPU"
  infer_device: "0"
```

启动构建出的 `vision_simple-server`（Windows 为 `.exe`）。模型在首次推理时加载；配置或模型文件变更后重启服务。只在可信网络或受鉴权代理保护的环境中开放非回环监听地址。

#### 发起推理

以下客户端仅依赖 Python 标准库。将 `image.jpg` 替换为本地图片路径；`model` 必须与上方配置的 `name` 一致：

```python
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

image_path = Path("image.jpg")
endpoint = "http://127.0.0.1:11451/v1/infer/yolo"
payload = {
    "model": "yolo26n",
    "images": [base64.b64encode(image_path.read_bytes()).decode("ascii")],
}
request = Request(
    endpoint,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request, timeout=120) as response:
    print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
```

`images` 使用原始 base64，**不是** `data:image/...;base64,...` URL。分割、姿态、旋转框分别改用 `/v1/infer/seg`、`/v1/infer/pose`、`/v1/infer/obb`，并选择对应模型名。检测结果中的 `bbox` 为原图像素 `[x,y,width,height]`；空检测返回该图片对应的空数组。

使用现有 `POST /v1/infer/{task}`；检测也支持原有 `/v0/infer/yolo`。响应结构、模型缓存、批次顺序与整批失败语义不变。C++ 检测仍使用 `InferYOLO::Create(context, path, YOLOVersion::kV26)`；其他任务改为显式版本，例如 `InferYOLOTask::Create(context, path, YOLOTask::kPose, YOLOVersion::kV26)`。旧任务调用者在 `task` 后补 `YOLOVersion::kV11`，可选 device_id 顺延。

后处理沿用本项目契约：raw 检测 NMS IoU 为 0.3，其他 raw 任务为 0.45；OBB 使用多边形 IoU，**不同于 Ultralytics 的概率 IoU**。NMS-free 不作二次抑制。Letterbox 使用黑色 padding，关键点保留图外坐标；mask 先插值 logits 再二值化、裁剪至整数 bbox。因此直接与 Ultralytics 默认 padding、mask 缩放、关键点裁剪比较不会逐像素一致，应统一预处理并明确这些差异。

#### 已验证兼容性与限制

**已验证范围**：Windows x64 Release，C++ ORT 1.20.0 / DirectML 1.15.4，对照端 Python ORT 1.24.3。

| EP | raw FP32 | NMS-free FP32 | raw FP16 | NMS-free FP16 |
|---|---|---|---|---|
| CPU | 四任务通过 | 四任务通过 | 四任务通过 | 四任务通过 |
| DirectML | 四任务对照通过 | 四任务对照通过 | 四任务推理通过 | 四任务均在 ORT 初始化时失败 |

该矩阵记录既有 Windows 验证范围，不是所有机器、驱动或导出模型的保证，也不是本次文档更新重新跑出的结果。**在上述 DML 环境请选择 FP32 或已验证的 raw FP16，不要部署这些 NMS-free FP16 产物**；初始化失败返回受控 `model_load_failed`，不会静默换模式。部署前请使用自己的图片、导出模型与执行提供者验证结果和资源占用。


不包括分类、语义分割、深度估计、YOLOE、动态输入或 batch>1。CUDA/TensorRT 不在上述 YOLO26 验证范围内。YOLO26 权重与导出产物不随仓库分发；部署者须核对 Ultralytics 软件及模型许可证。仓库中的 YOLO11/PP-OCR 测试资源另行通过 Git LFS 获取。

### 时序跟踪会话

跟踪由独立有状态 `TrackingService` 提供，不是推理任务，也不自动解码视频。调用者提交推理或其他检测器产生的检测结果；每个 C++ `Tracker`/HTTP 会话拥有独立的 ID、运动及外观状态。

| 操作 | HTTP 请求 |
|---|---|
| 创建 | `POST /v1/tracking/sessions`，正文 `{"algorithm":"bytetrack","options":{"min_hits":2}}`；返回 201、`id`、`algorithm`、`status` 及 `Location` |
| 推进一帧 | `POST /v1/tracking/sessions/{id}/frames`，正文示例如下 |
| 状态 | `GET /v1/tracking/sessions/{id}`；返回 `id`、`algorithm`、`status` |
| 列表 | `GET /v1/tracking/sessions?limit=100`；返回 ID 数组 `sessions` 和可空 `next_cursor`，下一页原样传入 `cursor`（limit 为 1–100） |
| 重置 | `POST /v1/tracking/sessions/{id}/reset`，正文 `{}`；返回 `{"reset":true}` |
| 删除 | `DELETE /v1/tracking/sessions/{id}`；返回 204 |

```json
{"frame_index":0,"timestamp":0.0,"detections":[{"class_id":0,"confidence":0.9,"bbox":[10,20,30,40]}]}
```

会话首帧建立的轨迹立即确认；后续新建轨迹才受 `min_hits` 确认门槛约束。空 `tracks` 也可能是成功结果，例如没有检测或新目标尚未确认。

- `timestamp` 是有限非负秒数，`frame_index` 是 0–9007199254740991 的整数；两者在同一会话内均须严格递增。经过的秒数控制运动预测，索引间隔计入过期帧数。被拒绝帧不推进状态；reset 清空序列、ID 和时间。推进返回 `frame_index`、`timestamp` 和 `tracks:[{track_id,class_id,confidence,bbox}]`，仅输出本帧观测到的已确认轨迹，不输出丢失预测。状态含可空 `last_frame_index`/`last_timestamp` 及 `active_tracks`/`lost_tracks`。
- 算法为 `"bytetrack"` 或 `"botsort"`。默认选项：`high_threshold:0.5`、`low_threshold:0.1`、`new_track_threshold:0.6`、`match_threshold:0.8`、`max_lost_frames:30`、`min_hits:2`、`max_tracks:256`、`max_detections:256`、`camera_motion:true`、`appearance:false`、`proximity_threshold:0.5`、`appearance_threshold:0.25`。阈值范围 [0,1]，须 low < high ≤ new；匹配阈值表示最大代价，不是最低 IoU。`min_hits` 为 1–10000，`max_lost_frames` 为 0–10000，两个容量选项均为 1–256。
- BoT-SORT 相机运动补偿要求每帧 `image` 提供原始 base64 PNG/JPEG，整个序列尺寸一致且至少 8×8；仅提交检测结果时设 `camera_motion:false`。解码前限制为最多 16,777,216 像素。框为有限浮点 xywh，宽高为正；confidence 范围 [0,1]，类别 ID 非负。
- BoT-SORT 的 `appearance:true` 要求每个检测含有限、非零 `embedding` 数组，最多 512 元素且会话内维度一致。外观特征由调用者提供，不附带 ReID 模型或权重，跟踪服务也不执行特征提取模型。
- 限额：32 个会话、每帧最多 256 个检测、每会话最多 256 条轨迹、正文 4 MiB。自创建或最近成功 step/reset 后 300 秒过期，在服务访问时清理；status/list 不续期。会话忙时并发访问返回 409，不排队；不同会话的轨迹身份互不共享。
- 错误为 `error.{code,message,image_index}`，`image_index` 为 null：400 `invalid_request`/`invalid_image`、404 `tracking_session_not_found`、409 `tracking_session_busy`/`frame_out_of_order`、503 `tracking_capacity`/`service_unavailable`（含 `Retry-After: 1`）、500 `tracking_failed`。拒绝未知 JSON 字段；POST 必须精确使用 `Content-Type: application/json`，GET/DELETE 不得带正文。正文超限返回 413，不支持的 `Expect` 返回 417。
- 跟踪业务请求拒绝非空浏览器 `Origin`（403），响应使用 `Cache-Control: no-store`；普通 OPTIONS 预检仍由全局 CORS 中间件处理。Origin/CORS 和会话 ID 都不是认证，列表没有租户隔离，必须使用可信网络或鉴权代理。

### 异步视频字幕

此功能通过 OCR 提取**画面内文字**，不是语音转写。先配置 OCR 模型及真实检测/识别权重和字典。以下为 POSIX shell 的 `curl` 示例（Windows 使用 `curl.exe`）；将 `JOB_ID` 替换为创建响应中的 `id`：

```sh
# 201 + Location；选项是顶层字段，不套 "options"
curl -sS -X POST http://127.0.0.1:11451/v1/subtitle/jobs -H 'Content-Type: application/json' -d '{"model":"ppocr-v4","sample_interval_ms":200,"roi":[0,0.5,1,0.5],"min_confidence":0.5,"stable_samples":2,"gap_samples":2}'
# 202 仅表示已接收处理，不保证视频解码成功
curl -sS -X PUT http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID/video -H 'Content-Type: application/octet-stream' --data-binary @clip.avi
curl -sS http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID
# 仅在 state == "completed" 后下载
curl -fS http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID/subtitles.srt -o clip.srt
curl -fS http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID/subtitles.vtt -o clip.vtt
# 或取消未完成的任务；轮询至终态后再删除
curl -sS -X POST http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID/cancel -H 'Content-Type: application/json' -d '{}'
curl -i -X DELETE http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID
```

- 创建仅接受 `model`、`sample_interval_ms`（整数 100–5000，默认 200）、`roi`、`min_confidence`（[0,1]，默认 0.5）、`stable_samples` 和 `gap_samples`（整数 2–10，均默认 2）。`model` 为配置中的 OCR 原名，1–256 字节。ROI 使用归一化 `[x,y,width,height]`，宽高为正且完整位于画面内；默认 `[0,0.5,1,0.5]` 为下半屏。全画面示例：`{"model":"ppocr-v4","roi":[0,0,1,1]}`。未知字段会被拒绝。
- 采样选择达到下一个间隔的首个解码帧，使用实际显示时间戳（毫秒），不是 `采样序号 × 间隔`。过滤低于 `min_confidence` 的 OCR 行（模型自身识别阈值仍生效），按从上到下、同行从左到右排列并规范化空白；不做模糊文本匹配。相同规范化文字连续出现 `stable_samples` 次才确认，字幕起点回溯到这组观测的第一次；确认新文字时，旧字幕在同一时刻结束。短暂替代文字被抑制；少于 `gap_samples` 次空观测可合并未改变的字幕，达到该阈值则在第一次空观测处结束。EOF 时若存在未确认的空白间隔则在其起点结束，否则末条字幕结束于解码流末尾。区间不重叠且时长为正，精度取决于采样与 OCR。
- 下载为经 UTF-8 校验的 SRT/WebVTT，规范化控制字符/空行并转义 `&`、`<`、`>`，防止识别文字变成标记或字幕结构。成功但没有字幕时，SRT 为空、WebVTT 仅含头部，不伪造字幕。
- 正常状态为 `created → uploading → queued → running → completed`，失败进入 `failed`。取消返回 202；正在执行原生处理时经历 `cancelling → cancelled`，是合作式取消，不强制中断解码器/ORT 调用。对终态任务取消不会改变结果。状态字段为 `id`、`state`、`uploaded_bytes`、`decoded_frames`、`sampled_frames`、`position_ms`、可空 `duration_ms`、`cue_count` 和可空 `error_code`。时长可能未知，计数/位置并非保证准确的百分比；运行中的 `cue_count` 不含尚未闭合的字幕。`GET /v1/subtitle/jobs?limit=100` 返回任务对象数组及可空 `next_cursor`，下一页原样传入 `cursor`（limit 为 1–100）。
- 创建 JSON 后以独立 PUT 上传原始字节，不接受 multipart/base64、服务器本地路径或远程 URL。仅 `created` 可开始上传；已消费、中断或失败的上传不能在同一任务重试，应新建任务并重新上传，创建操作不具幂等性。上传成功后仍可能异步失败，必须轮询 `state` 并检查 `error_code`（如 `upload_interrupted`、`unsupported_video`、`invalid_timestamps`、`ocr_failed`、`subtitle_limit`），不依赖错误文案；失败的部分结果不能下载。
- 单 worker，最多 **8 个任务**（含待上传及保留的终态结果）。上限为每视频 64 MiB、1800 秒、1,000,000 个解码帧、每帧 16,777,216 像素；每次观测最多 4096 行，单行及合并文字最多 4096 字节；最多 10,000 条字幕及累计 2 MiB 字幕文字。JSON 控制正文最多 64 KiB。输入文件存放于服务自建的私有临时目录，在完成/失败/取消/删除时移除，正常关闭时移除目录。created/uploading 无上传活动 60 秒过期，终态结果 300 秒过期；轮询和下载不续期。文件系统清理失败可能继续占用容量，进程崩溃不保证正常清理。
- HTTP 可用的视频解码依平台而定：所有平台提供有界 MJPEG AVI reader（从零开始的单一 MJPG/mjpg 视频流，含 OpenDML AVI/AVIX），拒绝其他 AVI codec；Windows 另通过 Media Foundation 读取首部为 `ftyp` box 的 MP4 系列文件，codec 取决于系统安装情况。ASF 虽有底层读取支持，但当前上传入口拒绝它；MKV 可通过上传嗅探，却会在解码阶段报 `unsupported_video`。识别容器或返回上传 202 不代表支持解码。不支持任意编码、播放列表、图像序列或 URL 抓取。
- MJPEG AVI 时间戳依据流的 rate/scale；Media Foundation 使用实际 sample 时间戳及正的 sample 时长，起点向下、终点向上取整至毫秒。完成后的 `duration_ms` 为实际视频结束时间，不是帧数估算值或较长的音轨/容器时长；原生解码的 duration 可在完成前一直为 null。
- 错误为 `error.{code,message,image_index}`，`image_index` 为 null：400 `invalid_request`；404 `model_not_found`/`subtitle_job_not_found`；409 `subtitle_job_busy`/`subtitle_not_ready`；413 `payload_too_large`；415 `unsupported_media_type`/`invalid_video`；503 `subtitle_capacity`/`service_unavailable`（`Retry-After: 1`）；500 `subtitle_failed`。创建/取消必须精确使用 `application/json`，上传使用 `application/octet-stream`；无正文操作拒绝正文，不支持的 `Expect` 返回 417。DELETE 仅对 created 或终态返回 204，其他状态需先取消并轮询。
- API **没有认证或租户隔离**。字幕业务请求对非空浏览器 `Origin` 返回 403，响应使用 `Cache-Control: no-store`；普通 OPTIONS 仍由全局 CORS 中间件处理。任务 ID 不是凭证，须使用可信网络或鉴权代理。

真实视频回归（在仓库根目录运行，需先构建 server）：

```sh
python scripts/test_subtitle_regression.py --server <server可执行文件> --project-root . --ffmpeg <ffmpeg可执行文件> --font <font.ttf>
```

需要 `app/assets/test` 中的真实 `ppocr_det.onnx`、`ppocr_rec.onnx`、`ppocr_keys_v1.txt`，以及带 drawtext/MJPEG/libx264/AAC 的 FFmpeg 和可用的 TrueType 字体。FFmpeg 位于 PATH 且系统有默认 Arial/DejaVuSans 字体时，可省略 `--ffmpeg`/`--font`。回归执行真实解码/OCR，检查 HELLO/WORLD 精确区间、单帧 NOISE 抑制、SRT/WebVTT 一致性、上传中断、取消、容量/结果隔离和临时文件清理；Windows 还覆盖带较长音轨的原生可变帧率视频。这不是 codec 质量或性能基准。

### OCR 解码与 recognition batch

- 模型类型注册表将 `kPPOCRv3`/`kPPOCRv4` 交给 CTC，将 `kPaddleSAR` 交给独立 SAR decoder。`kEasyOCR` 尚未实现，创建时明确拒绝，不再借用 Paddle/CTC。
- SAR 保留相邻重复字符，跳过 PAD，遇到首个 BOS/EOS 停止，未知字符输出 `<UKN>`。普通字符索引为 `0..D-1`，其后依次为 UKN、BOS/EOS、PAD；输出类别数必须为 `D+3`。置信度沿用本库严格阈值过滤，空结果置信度为 0。
- SAR 的文件字典精确列出普通字符（需要空格时自行列入），不附加 CTC 空格或特殊 token。服务器 `models.yaml` 的 `version: "kPaddleSAR"` 使用同一注册表。
- 当前 recognition 导出契约为一个 float `[N,3,H,W]` 输入和一个 float `[N,T,C]` 概率输出，预处理为 RGB、归一化至 `[-1,1]`；不宣称支持要求额外 valid-ratio/attention 输入或不同归一化的所有 Paddle 模型。非 CTC 验收使用真实 ORT 执行的确定性 SAR 预测图，不包含新训练权重。
- C++ 在 `InferContext::Create` 的 `InferArgs` 中设置 `{"ocr_rec_batch_size","4"}`；服务在 `options` 中设置 `ocr_rec_batch_size: "4"`。范围 1–64，默认 1；v0/OpenAI-like/MCP 共用该设置，不改变响应格式。
- 动态 N：把相同预处理宽度的框稳定分组，最多每组指定数量，执行真正的 `[N,3,H,W]` 推理，再恢复检测顺序。动态 H 默认 48，动态 W 保留原有高度倍数取整与整框 resize；不同宽度不额外混合 padding，因此不会引入新的有效长度截断。
- 固定 N 以模型维度为准（1–64），末尾不足时补归一化零样本并丢弃其输出。固定 H/W 按声明尺寸整框 resize；不凭空推断 `T × width_ratio`，所有真实样本解码完整 T（SAR 由 EOS 截止）。需要保持宽高比或额外 mask 的导出模型必须先匹配这一预处理契约。
- DBNet 检测在 recognition 裁剪前使用 1.5 unclip 比例扩展收缩后的文字区域，避免字形被裁掉（例如应为 `HELLO` 却只裁出 `E`）。同步与流水线 OCR 都使用修正后的几何，框和识别文字可能因此改变。
- `test_ocr_batch` 覆盖混合宽度、顺序、固定 N 尾部、输入缓存复用、SAR 文件字典及流水线隔离；其中 ONNX guard 会在 N=1 时真实失败，防止“循环单张”伪装 batching。批量大小的收益取决于裁剪尺寸、模型和执行提供者，需以实际负载测量。

### 统一服务、OpenAI-like 与 MCP SSE

v0、原生 v1、OpenAI-like 和 MCP 共用 `InferenceService`，不重复加载模型或维护第二套缓存，使用相同租约、统计、空闲卸载和流水线限制。跟踪状态由独立服务管理。发现接口仅列出配置，不证明模型文件可加载。

**OpenAI-like 子集**

- `GET /v1/models?limit=100` 返回 `object:"list"` 和 `data`；模型 ID 为 `<task>:<原名>`，task 包括 `yolo`、`ocr`、`seg`、`pose`、`obb`。limit 为 1–200；`has_more` 为 true 时，将 `next_cursor` 原样作为下一页 `after`，勿自行构造游标。
- `POST /v1/chat/completions` 接受 `model`、`messages`、可选 `stream`、`timeout_ms`、`n:1` 和 `response_format:{"type":"json_object"}`（或 `"text"`）。用户消息的 `image_url.url` 必须是 PNG/JPEG/WebP/BMP 的 base64 data URL，`detail` 仅支持省略或 `"auto"`；不抓取远程 URL。
- 至少提供一张图片；按消息及 content 数组中的图片顺序推理。文字上下文不改变检测/OCR行为，这不是聊天生成模型。其他生成参数、工具调用、JSON Schema 输出和 Responses API 不在兼容范围，不支持的参数返回 400，不静默假装生效。
- `choices[0].message.content` 是完整任务结果的 JSON 字符串：YOLO/新任务包含 `class_names/results`，OCR 包含 `results`；几何信息遵循上述各任务契约，不会编造 token usage。
- `stream:true` 返回 `chat.completion.chunk` SSE（role、完整结果 content、finish_reason）及 `[DONE]`；不是逐 token 或逐图流。整批成功后才提交流头，之前的失败仍返回正常 HTTP JSON 错误。
- 错误为 `{"error":{"type":...,"code":...,"message":...,"param":...,"image_index":...}}`。推理错误码沿用 v0；过载 503 带 `Retry-After: 1`。请求体在接收时限制为 64 MiB，超限 413；序列化结果限制为 64 MiB。

**MCP 传统 HTTP+SSE**

连接 `GET /mcp/sse`，读取 `endpoint` 事件中的相对 POST 地址，再向其发送 JSON-RPC 2.0。完成 `initialize`、`notifications/initialized` 后，使用 `tools/list` / `tools/call`；支持版本 `2024-11-05`、`2025-03-26`、`2025-06-18`、`2025-11-25`。这不是 Streamable HTTP，客户端须选择 SSE 传输。

初始化 `params` 必须包含 `protocolVersion`、对象 `capabilities` 和含 `name`/`version` 的 `clientInfo`。POST 返回 202 仅表示消息接收，JSON-RPC 响应从原 SSE 连接读取，不在 POST 正文中返回推理结果。

- `list_models`：可选 `limit`（1–200）和 `cursor`，返回 `data:[{id,kind,name}]` 及可选 `next_cursor`。
- 自动生成的 `infer_yolo` / `infer_ocr` / `infer_seg` / `infer_pose` / `infer_obb`：`{"model":"配置原名","images":["原始base64"],"timeout_ms":60000}`；不要传带任务前缀的 catalog ID 或 data URL。输入 JSON Schema 随工具发现返回。
- 新版本结果含 `structuredContent` 及等价 JSON 文本；旧版本通过文本保留完整结构。执行错误为 `isError:true`，包含稳定 `error.code`、`image_index` 和恢复建议；协议错误使用 JSON-RPC error，字符串/整数请求 ID 不混用。
- `notifications/cancelled` 的 `requestId` 取消本会话对应请求；断连取消该会话全部任务。取消及超时等待正在运行的原生阶段退出，不强制中断 ORT，也不影响其他会话的相同 ID。
- 固定上限：32 会话、2 执行 worker、16 排队任务、每会话 8 个活动工具请求、64 MiB POST 正文、8 MiB 待发送结果/写缓冲。每 15 秒心跳；持续积压写缓冲约 30 秒或无活动请求且 5 分钟无消息时关闭会话，定时检查可能延迟至下一次心跳。超过结果缓冲上限会关闭会话，需缩小批次重新连接。
- Host 只允许回环地址及显式非 wildcard 的监听 host，端口须匹配；提供 Origin 时必须是相应可信 HTTP(S) authority。MCP 不启用全局宽松 CORS。远程使用需显式绑定可信地址，或由鉴权代理重写为受信任的后端 Host/Origin。

仓库 `.mcp.json` 指向本地默认端口。OpenAI SDK 的 `api_key` 在本服务中不是认证凭证；三个协议均需可信网络或外部鉴权代理。代理需关闭 SSE 缓冲并允许长连接。验收命令见下方回归章节，多步骤 Agent 验收题见 `scripts/mcp_evals.xml`。

### C++ 迁移说明

- `InferYOLO/InferOCR::Create/Run` 签名不变。Run 接受非空二维 `CV_8UC3`，支持非连续 ROI；灰度、BGRA、浮点图像以及非有限或超出 `[0,1]` 的 confidence 返回参数错误。
- YOLO v11 默认按类别 NMS；v10 仅支持端到端 `[1,N,6]`，不再重复 NMS。原 confidence、黑色 Letterbox 填充和 OCR 检测归一化不变。
- YOLO26 检测使用 `YOLOVersion::kV26`；`InferYOLOTask::Create` 的路径和内存重载现在都要求在 `task` 后显式传入版本。旧分割／姿态／OBB 调用补 `YOLOVersion::kV11`，YOLO26 调用传 `YOLOVersion::kV26`，可选 `device_id` 放在版本之后。
- PP-OCR CTC 文件路径 Create 使用 Paddle 字典文件约定：文件不含 blank 和末尾空格类别，由加载器补空格；直接传入 map 时，调用者须提供全部非 blank 类别，键为 `class_id - 1`。SAR 使用上述独立字典约定。
- `HTTPServer::Run/StartAsync` 现在返回 `HTTPServerResult<void>`，调用者必须检查错误。空/超长 host、监听失败会受控失败，不会静默绑定 wildcard。仓库显式 `0.0.0.0` 默认配置未改变。
- helper 使用者必须重新编译并迁移到单一几何路径：

```cpp
LetterboxTransform transform;
cv::Mat& padded = helper.Letterbox(image, target_size, transform);
if (padded.empty()) { /* 拒绝无效或缩放后零尺寸的图片 */ }
cv::Rect box = VisionHelper::ScaleCoords(transform, cv::Vec4f{x1, y1, x2, y2});
```

`ScaleCoords` 接收模型空间浮点 xyxy，按实际轴向比例反算，裁剪端点后 round 为整数 xywh。旧几何签名、`DataConverter` 和未实现的 uint8 转换空操作已删除；`Cvt` 支持 FP32/FP16 双向转换。

服务没有内建认证、热加载或任意 ONNX 支持。v1/MCP 正文限制不覆盖旧 v0 路由；旧推理接口也没有通用解码后像素上限。跟踪 PNG/JPEG 和字幕视频帧具有上述像素限制，但这些限制不等于全面的生产安全保证。


## 开发与部署

### 构建项目

#### 获取源码与模型资源

安装 Git、Git LFS、[xmake](https://xmake.io) 及对应编译器。MSVC/GCC 配置使用 xmake ≥ 2.9.7；Clang 18 + libc++ 配置使用 xmake ≥ 3.1.1，与 CI 一致。

```sh
git clone --recurse-submodules https://github.com/lona-cn/vision-simple.git
cd vision-simple
git lfs install
git lfs pull
```

首次配置会下载依赖，需可用网络和依赖构建工具。现有 checkout 可先执行 `git submodule update --init --recursive`；LFS 指针文件不能作为 ONNX 权重使用。

#### windows/x64

使用支持 C++23 的 Visual Studio 2022 MSVC 工具链和 Windows SDK。以下显式选择 CPU 构建；DirectML 见下一节。

```powershell
xmake f -p windows -a x64 --toolchain=msvc -m release --with_dml=n --with_cuda=n --with_tensorrt=n -y
xmake build server
Copy-Item app/assets/test/* build/windows/x64/release/assets/ -Recurse -Force
# 在 build/windows/x64/release/config/server.yaml 中将 host 改为 "127.0.0.1"，再启动：
Set-Location build/windows/x64/release
.\vision_simple-server.exe
```

#### linux/x86_64

CI 的本机 CPU 配置为 Ubuntu 24.04 + GCC 14；也提供 Clang 18 + libc++ 18 配置。需安装相应 C/C++ 编译器及 Python 开发工具、包构建工具。

```sh
xmake f -p linux -a x86_64 --toolchain=gcc --cc=gcc-14 --cxx=g++-14 -m release --with_cuda=n --with_tensorrt=n --with_rknpu=n -y
xmake build server
cp -R app/assets/test/. build/linux/x86_64/release/assets/
# 在 build/linux/x86_64/release/config/server.yaml 中将 host 改为 "127.0.0.1"，再启动：
cd build/linux/x86_64/release
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ./vision_simple-server
```

Clang 用户在仓库根目录用以下命令替换 GCC 配置命令，后续构建与启动相同：

```sh
xmake f -p linux -a x86_64 --toolchain=clang --cc=clang-18 --cxx=clang++-18 -m release --runtimes=c++_shared --with_cuda=n --with_tensorrt=n --with_rknpu=n -y
```

libc++ 18 所需的实验库编译/链接标志由项目设置，无需手动修改源码。上述目录适用于默认输出配置；自定义 `-o` 时以实际 targetfile 为准。

**工作目录很重要**：server 从工作目录读取 `config/server.yaml`、`config/models.yaml` 和相对模型路径。构建会复制基础配置与主资源，但 server 目标不会自动复制测试模型，因此上面显式复制了测试资源；正式部署只需准备配置引用的权重和字典。重新构建可能覆盖配置，建议部署到独立目录；配置或模型变更后重启服务。

也可从 [GitHub Releases](https://github.com/lona-cn/vision-simple/releases) 选择平台、架构和 EP 变体匹配的归档；核对发布页校验和及归档内 `build-info.json` 的 commit/构建配置。归档仅收集当次构建目录中的配置和资源，不保证含全部模型；交叉构建产物仍须在目标硬件验证。

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

编译选项与运行配置必须匹配：启用构建选项后，还要在 `config/server.yaml` 的字符串 `options.infer_ep` 中选择 `kDML`、`kCUDA`、`kTensorRT` 或 `kRKNPU`，`infer_device` 选择设备。默认运行配置为 `kCPU`；Windows 的 DML 构建选项默认开启，不等于运行时自动选择 DML。

```powershell
# DirectML（Windows）
xmake f -p windows -a x64 -m release --with_dml=y
xmake build server
```

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

以下命令均从仓库根目录运行；如果刚按上文启动了服务，请另开终端并回到仓库根目录。先使用 CPU 配置构建，确保 Git LFS 模型资源已经下载。

```sh
# 构建 CPU 回归目标（逐个构建）
xmake build server
xmake build test_common
xmake build test_cvt
xmake build test_vision_helper
xmake build test_yolo_postprocess
xmake build test_ocr_decode
xmake build test_ocr_batch
xmake build test_infer_inputs
xmake build test_pipeline
xmake build test_yolo_tasks
xmake build test_tracker
xmake build test_config_load
xmake build test_subtitle_timeline

xmake run test_common
xmake run test_cvt
xmake run test_vision_helper
xmake run test_yolo_postprocess
xmake run test_ocr_decode
xmake run test_ocr_batch --project-root .
xmake run test_infer_inputs
xmake run test_pipeline
xmake run test_yolo_tasks --project-root .
xmake run test_tracker
xmake run test_config_load
xmake run test_subtitle_timeline
```

HTTP 回归使用 Python 3 标准库，独立创建临时配置、端口和进程，不触碰现有 11451 服务。模型、字典、图片和小型 ONNX 故障 fixture 必须存在；缺失即失败，不记为 SKIP。

```powershell
python scripts/test_http_regression.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
python scripts/test_protocol_regression.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
python scripts/test_yolo_tasks_http.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
python scripts/test_tracking_regression.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
python scripts/test_model_registry.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
```

Linux 或自定义构建目录先查询实际可执行文件：

```sh
server="$(xmake lua -q -c "import('core.project.config'); config.load(); import('core.project.project'); io.write(path.absolute(project.target('server'):targetfile()))")"
python3 scripts/test_http_regression.py --server "$server" --project-root .
python3 scripts/test_protocol_regression.py --server "$server" --project-root .
python3 scripts/test_yolo_tasks_http.py --server "$server" --project-root .
python3 scripts/test_tracking_regression.py --server "$server" --project-root .
python3 scripts/test_model_registry.py --server "$server" --project-root .
```

`test_yolo`/`test_ocr` 仍为交互演示，不作为上述 headless 验收。故障 fixture 已入库；仅重新生成时需要开发工具 `onnx` 和 `scripts/generate_reliability_fixtures.py`，不是服务运行依赖。

### 构建docker镜像
所有`Dockerfile`位于目录：`docker/`

```sh
# pull project
git clone --recurse-submodules https://github.com/lona-cn/vision-simple.git
cd vision-simple
git lfs pull
# 构建项目
docker build --platform linux/amd64 -t vision-simple:ci -f docker/Dockerfile.debian-bookworm-x86_64-cpu .
# 验证 HTTP 模型目录、Docker HEALTHCHECK、正常停止和容器清理
python3 scripts/test_docker_smoke.py --image vision-simple:ci
# 默认 CPU 推理；服务没有认证，先仅暴露本机端口
docker run -it --rm -p 127.0.0.1:11451:11451 --name vs vision-simple:ci
```

#### GHCR 多平台发布

`.github/workflows/docker.yml` 构建并发布 `ghcr.io/lona-cn/vision-simple` 的 Linux amd64、arm64 CPU 镜像：

- 推送 `vX.Y.Z` 或 `vX.Y.Z-prerelease` tag 触发；手动运行默认只构建、验收，显式选择 `publish=true` 才发布。发布使用自动提供的 `GITHUB_TOKEN` 和 `packages: write` 权限，不需要 PAT secrets。
- amd64、arm64 分别在原生 GitHub-hosted runner 构建，运行 HTTP/HEALTHCHECK smoke test、优雅停止和清理后，才推送已验收镜像。
- 每个平台分别发布 `<version>-cpu-amd64` / `<version>-cpu-arm64` 和 `sha-<完整 commit SHA>-cpu-amd64` / `sha-<完整 commit SHA>-cpu-arm64`。两边成功后合并为 `<version>-cpu`、`sha-<完整 commit SHA>-cpu` 多架构 manifest；稳定版本还更新 `latest`，预发布不更新。
- manifest 校验确认 amd64、arm64 两个平台的 registry digest 与 smoke-tested 镜像一致。只有 manifest 步骤成功才生成发布摘要；失败时已推送的平台专属 tag 不会自动回滚。
- `docker pull ghcr.io/lona-cn/vision-simple:latest` 会按客户端平台选择 amd64 或 arm64。部署优先使用成功摘要中的 `ghcr.io/lona-cn/vision-simple@sha256:...`。GHCR 包首次发布默认为 private；若需匿名拉取，在 GitHub 包设置中将其改为 public。
- ARMv7 Dockerfile 的产物路径仍指向 arm64，RISC-V 及 CUDA/TensorRT、RKNPU 镜像未纳入此次多架构发布；硬件加速后端需独立镜像变体和设备验收。

发布策略边界测试：`python3 -m unittest discover -s scripts -p test_docker_release.py`。amd64 CPU 构建启用 AVX/AVX2/F16C，需支持这些指令的 CPU。

#### 其他平台 / 硬件加速

GHCR 多架构 manifest 目前仅包含 Linux CPU `amd64` 和 `arm64`。ARMv7 Dockerfile 的产物路径仍指向 `arm64`，RISC-V 尚未通过目标设备运行验收；CUDA/TensorRT 和 RKNPU 需要对应硬件验证及独立镜像变体，不能与同架构 CPU 镜像合并到同一 manifest。

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
