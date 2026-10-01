# <div align="center">vision-simple</div>
English | [简体中文](./README.md)

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
<a><img alt="linux riscv64" src="https://img.shields.io/badge/linux-riscv64-brightgreen.svg"></a>
</p>
<p align="center">
<a><img alt="ort cpu" src="https://img.shields.io/badge/ort-cpu-880088.svg"></a>
<a><img alt="ort dml" src="https://img.shields.io/badge/ort-dml-blue.svg"></a>
<a><img alt="ort cuda" src="https://img.shields.io/badge/ort-cuda-green.svg"></a>
<a><img alt="ort rknpu" src="https://img.shields.io/badge/ort-rknpu-white.svg"></a>
</p>

vision-simple is a cross-platform C++23 vision inference library built on ONNX Runtime, with a C++ API and a standalone HTTP server. Use it to run compatible ONNX models, integrate structured vision results, or process detections and visible video text over time.

This guide describes **current source code**, not a guarantee that older release archives or container tags include every feature. It is intended for developers and operators making their first inference request and preparing a private deployment.

**Quick links:** [Quick start](#2-quick-start) · [Models](#3-models-and-inference) · [Integration](#4-integration) · [Deployment](#5-deployment-and-diagnostics) · [Builds and checks](#6-build-variants-and-checks) · [OpenAPI](doc/openapi/server.yaml)

## 1. Capabilities

| Area | What is available |
| --- | --- |
| Detection | YOLOv10, YOLO11 and YOLO26 |
| Instance segmentation, pose, oriented boxes | YOLO11 and YOLO26 |
| OCR | PP-OCR v3/v4 CTC; restricted Paddle SAR recognition exports |
| Tracking | Independent ByteTrack and BoT-SORT sessions using caller-supplied detections |
| Video subtitles | Asynchronous OCR of visible text, producing SRT and WebVTT; not speech transcription |
| Diagnostics | Model preflight, cold/warm inference timings and opt-in private OCR image exports |
| Integration | C++23 error-returning API, native HTTP, an OpenAI-like subset and legacy MCP HTTP+SSE |

Models load lazily. Native inference, OpenAI-like HTTP and MCP share model caching, active leases, idle eviction, statistics and bounded staged execution. Subtitle OCR also uses the inference service, while jobs have their own bounded worker and storage lifecycle. Tracking has separate state and limits.

Windows x64 and Linux x86_64 have native build configurations. ARM64, ARMv7 and RISC-V64 cross-build configurations also exist, but compiling an artifact does not verify inference on target hardware. CPU, DirectML, CUDA, TensorRT and RKNPU availability depends on the build, runtime, drivers and model.

The implemented inference backend is ONNX Runtime. TVM, EasyOCR, arbitrary ONNX architectures, classification, semantic segmentation, depth, YOLOE, dynamic YOLO shapes and YOLO tensor batches greater than one are outside the supported contract. No fixed binary size, memory use or frame rate is promised.

### Examples

![hd2-yolo-gif](doc/images/hd2-yolo.gif)

![http-inferocr](doc/images/http-inferocr.png)

## 2. Quick start

### Get the source and resources

Install Git and Git LFS, and Python 3 on the host for client examples. Source builds also require [xmake](https://xmake.io) and a C++23 compiler. CI uses xmake 2.9.7 for MSVC/GCC and 3.1.1 for Clang 18 with libc++; these are tested configurations, not verified minimum versions.

```sh
git clone --recurse-submodules https://github.com/lona-cn/vision-simple.git
cd vision-simple
git lfs install
git lfs pull
```

For an existing checkout, run `git submodule update --init --recursive`. Dependency configuration needs network access and package build tools. An LFS pointer is not a usable model file.

### Local CPU container

Build current source rather than assuming a historical image includes current functionality:

```sh
docker build --platform linux/amd64 -t vision-simple:local -f docker/Dockerfile.debian-bookworm-x86_64-cpu .
docker run -it --rm --name vs -p 127.0.0.1:11451:11451 vision-simple:local
```

This amd64 CPU image requires AVX/AVX2/F16C. The initial build downloads and compiles dependencies. Default configuration and YOLO11/PP-OCR test models are included; YOLO26 weights are not. The container binds its interfaces, but the command above publishes only host loopback. Inference and user-job APIs have no built-in authentication or tenant isolation; do not expose the backend directly to the Internet.

### Windows x64 source build

Use Visual Studio 2022 MSVC and a Windows SDK. Explicitly disable GPU providers for this CPU example:

```powershell
xmake f -p windows -a x64 --toolchain=msvc -m release --with_dml=n --with_cuda=n --with_tensorrt=n -y
xmake build server
Copy-Item app/assets/test/* build/windows/x64/release/assets/ -Recurse -Force
Set-Location build/windows/x64/release
.\vision_simple-server.exe
```

### Linux x86_64 source build

The native CPU configuration uses GCC 14; install the compiler and dependency build tools first.

```sh
xmake f -p linux -a x86_64 --toolchain=gcc --cc=gcc-14 --cxx=g++-14 -m release --with_cuda=n --with_tensorrt=n --with_rknpu=n -y
xmake build server
cp -R app/assets/test/. build/linux/x86_64/release/assets/
cd build/linux/x86_64/release
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ./vision_simple-server
```

For custom output directories, use the actual target path. The server reads `config/server.yaml`, `config/models.yaml` and relative model paths from its **working directory**. Building copies base configuration and main assets, but the server target does not automatically copy test models. The explicit copy above supplies the default test catalog. Production needs only its configured model files and dictionaries. Keep production configuration in a separate deployment directory because rebuilding may overwrite build-directory configuration.

### Check the service, then run inference

In another terminal, query `http://127.0.0.1:11451/livez`, `/readyz` and `/v1/models` with curl (`curl.exe` in PowerShell). Health means the process is alive or accepting work; discovery lists configuration. **Neither proves that weights load or inference succeeds.**

Save this Python 3 standard-library example as `infer.py` and run `python infer.py` from the repository root. The image is the same `hd2.png` copied into deployment `assets/`; change the path when using your own image.

```python
import base64
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

payload = {
    "model": "hd2-fp32",
    "images": [base64.b64encode(Path("app/assets/test/hd2.png").read_bytes()).decode("ascii")],
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
    print(f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}")
    raise
```

Native `images` contain **raw base64**, not data URLs. For OCR, use `/v1/infer/ocr`, set `model` to `ppocr-v4`, and **remove both `confidence` and `nms_iou`**. For segmentation, pose or OBB, configure compatible models first and use the corresponding task route.

## 3. Models and inference

### Configuration and model identity

Prefer the canonical `models` list in `config/models.yaml`:

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

Models are identified by `(task,name)`. Duplicate pairs are rejected; different tasks may share a name. Native v1 and MCP inference use the **raw name**, while OpenAI-like model IDs use `<task>:<name>`. Legacy `yolo`/`ocr` configuration lists remain readable and may coexist with non-conflicting canonical entries. Version and resource validation happen when loading, not merely listing a model.

Restart after changing configuration or model files; hot reload is unsupported. Check release checksums and `build-info.json` when using [release archives](https://github.com/lona-cn/vision-simple/releases). A versioned archive or image does not imply every configured model is present or that YOLO26 is implemented in that binary.

### Task geometry and export contract

| Task | Result geometry in original-image coordinates |
| --- | --- |
| `yolo` | Integer `bbox:[x,y,width,height]`, class and confidence |
| `ocr` | Recognized text, confidence and bounding rectangle |
| `seg` | Integer bbox plus cropped binary 0/255 PNG in `mask_png_base64`; place at the bbox origin |
| `pose` | Bbox and floating-point `keypoints:[{x,y,confidence}]`; points are not clipped |
| `obb` | Four ordered `corners:[[x,y],...]` and angle in radians along corner 0→1; corners may be outside the image |

YOLO11 requires static single-image `[1,3,H,W]` input, raw FP32/FP16 outputs, class-name metadata and no embedded NMS. Segmentation also requires matching mask prototypes; pose requires keypoint channels (`kpt_shape` supports 2 or 3 coordinates); OBB requires an angle channel. Add entries with `task: seg`, `pose` or `obb`, explicit `version: kV11` or `kV26`, and `files: {model: ...}` using your compatible exports.

Raw detection uses class-aware NMS on floating-point model-space boxes before clipping and rounding. Raw detection retains scores strictly greater than confidence; end-to-end detection and segmentation/pose/OBB use inclusive comparison. OBB uses polygon IoU, not Ultralytics probabilistic IoU. NMS-free exports do not receive a second NMS.

### YOLO26 exports

Start with **FP32 NMS-free detection** and validate it on your intended provider before adding tasks or FP16. Export from the repository root:

```sh
python -m pip install ultralytics==8.4.159 onnx==1.20.1 onnxruntime==1.24.3 torch==2.14.0 torchvision==0.29.0
python scripts/export_yolo26.py --output build/yolo26 --imgsz 640 --tasks detect --precisions 32
```

This produces `detect_raw_fp32.onnx` and `detect_e2e_fp32.onnx`. Exporter `detect` maps to service `yolo`. Omit the filters to export all four tasks, raw/e2e and FP32/FP16. The script fixes batch=1, opset=17 and static shapes, and writes a manifest with dependency versions, export arguments, hashes and actual tensor metadata. These Python packages are export-time dependencies, not server runtime dependencies.

| Task | Raw output, external NMS | NMS-free output |
| --- | --- | --- |
| `yolo` | `[1,4+nc,A]`, xywh and class scores | `[1,K,6]`, xyxy,score,class_id |
| `seg` | `[1,4+nc+nm,A]` plus prototypes | `[1,K,6+nm]` plus `[1,nm,Hm,Wm]` prototypes |
| `pose` | `[1,4+nc+nk*nd,A]` | `[1,K,6+nk*nd]`, nd=2/3 |
| `obb` | `[1,5+nc,A]` | `[1,K,7]`, **xywh,score,class_id,angle** |

`A` and `K` are not fixed at 8400 or 300. Preserve `names`, correct `task`, explicit `end2end`, `args.nms`, and pose `kpt_shape`. Missing/conflicting metadata, incompatible shapes and embedded NMS are rejected, not guessed from filenames. FP16 weights do not mean every I/O tensor is FP16.

Copy the selected export into deployment `assets/` and add, for example:

```yaml
models:
  - task: yolo
    name: yolo26n
    version: kV26
    files: {model: assets/detect_e2e_fp32.onnx}
```

Previous Windows x64 verification used C++ ORT 1.20.0 / DirectML 1.15.4: CPU ran raw and NMS-free FP32/FP16 for all four tasks; DirectML ran FP32 and raw FP16, while these NMS-free FP16 exports failed during initialization. Prefer FP32 on that DML stack. This is a compatibility boundary, not a guarantee for every driver/export, and does not verify CUDA/TensorRT. Loading failures return controlled errors rather than substituting modes.

YOLO26 weights are **not bundled**. Review the applicable Ultralytics software and model licenses. YOLO11/PP-OCR test resources are fetched separately through Git LFS.

### OCR decoding and batching

`kPPOCRv3` and `kPPOCRv4` use CTC. File dictionaries exclude blank and the trailing space class; the loader appends space. Map-based C++ dictionaries provide every nonblank class at key `class_id-1`. `kEasyOCR` is unimplemented and rejected at creation.

`kPaddleSAR` supports a restricted recognition contract: one float `[N,3,H,W]` input, one float `[N,T,C]` probability output and RGB preprocessing normalized to `[-1,1]`. Dictionaries contain ordinary characters only, including explicit space if needed; classes must equal dictionary size +3 (UKN, BOS/EOS, PAD). SAR preserves adjacent repeated characters, skips PAD and stops at the first BOS/EOS. Models needing extra attention/valid-ratio inputs or other preprocessing are unsupported; no trained SAR weights are bundled.

Set `ocr_rec_batch_size: "4"` in server `options` or the same key in C++ `InferArgs`; range 1–64, default 1. Dynamic-N crops are stably grouped by preprocessed width and restored to detection order. Fixed-N exports determine batch size (1–64); partial tails use normalized-zero dummy samples whose outputs are discarded. Fixed H/W resizes whole crops; no valid-length truncation is guessed. Batching gains depend on your crops, export and provider.

Detector morphology is a **model-construction option**, not an HTTP request control. Add this alongside `files` on an OCR entry:

```yaml
    ocr_detection:
      kernel_size: 2
      dilation_iterations: 3
      min_box_area: 64
```

| Field | Inclusive range | Default | Meaning |
| --- | --- | --- | --- |
| `kernel_size` | 1–32 | 2 | Square dilation kernel; 1 is identity |
| `dilation_iterations` | 0–8 | 3 | Zero skips dilation |
| `min_box_area` | 0–1048576 | 64 | Strict pre-unclip bounding-rectangle area threshold |

Omitted/null options and missing leaves use defaults; `{}` explicitly selects them. Bare or quoted decimal integers are accepted; invalid types, unknown leaves and out-of-range values fail. A nonnull morphology option on a non-OCR model fails. C++ factories take `OCRDetectionOptions` after the optional device ID and snapshot it immutably. DBNet uses unclip ratio 1.5. Recognition filtering remains 0.125 in the server; it filters tokens, not detector pixels or whole lines.

## 4. Integration

### C++ API

Link the built library and its dependencies. Check each `std::expected` result rather than dereferencing a failure:

```cpp
#include <vision_simple/Infer.h>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace vision_simple;

int main() {
    auto context = InferContext::Create(InferFramework::kONNXRUNTIME, InferEP::kCPU);
    if (!context) {
        std::cerr << "Context: " << context.error().message << '\n';
        return 1;
    }
    auto model = InferYOLO::Create(**context, "assets/hd2-yolo11n-fp32.onnx",
                                 YOLOVersion::kV11);
    if (!model) {
        std::cerr << "Model: " << model.error().message << '\n';
        return 1;
    }
    auto image = cv::imread("assets/hd2.png", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Cannot decode assets/hd2.png\n";
        return 1;
    }
    auto result = (*model)->Run(image, YOLOInferenceOptions{.confidence = 0.625f});
    if (!result) {
        std::cerr << "Inference: " << result.error().message << '\n';
        return 1;
    }
    for (const auto& object : result->results) {
        std::cout << object.class_name << " " << object.confidence
                  << " " << object.bbox << '\n';
    }
}
```

Rebuild/relink every SDK consumer when migrating. `InferYOLO::Run` and `InferYOLOTask::Run` now take `YOLOInferenceOptions`, not scalar confidence; `{}` preserves omission defaults. `InferOCR::Run(image,float)` is unchanged. `InferYOLOTask::Create` requires an explicit `YOLOVersion` after the task, before device ID: insert `kV11` for existing YOLO11 callers or `kV26` for YOLO26.

Inputs must be nonempty two-dimensional `CV_8UC3` BGR; non-contiguous ROIs work. Grayscale, BGRA and floating-point inputs are rejected. Models and images must remain alive and unmodified through inference. Detection `class_name` is a model-owned string view: keep the model alive while using it. Task segmentation masks own their pixels independently of workspaces.

`InferPipeline::Create` supplies bounded preprocessing, ORT and postprocessing workers. Call `Run(model, images, options, PipelineControl{stop_token, deadline})`; OCR retains scalar confidence. Ordered batch results fail as a whole on an image error. `RunMeasured` returns an owning result plus timing records: queue/execution counts and elapsed times are observations, not isolated device-kernel benchmarks. `Close()` rejects/cancels unfinished work; join callers before destruction. Cancellation and deadlines are cooperative and drain native work before returning.

`InferContext::Capabilities()` reports runtime/compiled provider facts without loading models. `HTTPServer::Run/StartAsync` return error-bearing results and must be checked. Geometry helpers use `LetterboxTransform`, `Letterbox(image,target,transform)` and `ScaleCoords(transform,float_xyxy)`; migrate old geometry signatures and removed `DataConverter` callers. `Cvt` supports bidirectional FP32/FP16 conversion.

### Native HTTP and batch errors

See the [OpenAPI specification](doc/openapi/server.yaml) for request and response schemas.

| Route | Purpose |
| --- | --- |
| `POST /v1/infer/{task}` | All five tasks using raw configured names |
| `POST /v0/infer/yolo`, `/v0/infer/ocr` | Legacy inference routes; no v0 seg/pose/obb routes |
| `GET /v0/infer/models` | Legacy yolo/ocr-only catalog |
| `GET /v1/models` | All-task, task-qualified catalog |
| `GET /v0/infer/stats?limit=100&offset=0` | Protected cache counters and image-budget statistics |
| `POST /v0/infer/unload` | Protected unload with `{"kind":"yolo","model":"hd2-fp32"}` |

Successful batches return one result per input image in original order; no detections is a valid empty item. A valid configured model accepts an empty native batch without decoding or image credit, but transport admission still applies. Any image failure rejects the whole batch: no partial output. Native errors use `error.{code,message,image_index}`; image failures identify an input index, while control/model failures have null index.

Common statuses are 400 `invalid_request`/`invalid_image`/`image_limit_exceeded`, 404 `unknown_model` on v1 (legacy v0 uses 400), 500 inference/load errors, 503 `service_overloaded` with `Retry-After: 1`, and 504 `request_timeout`. Unload returns 409 `model_busy` for leased models or 404 `model_not_loaded`; the next inference reloads an unloaded model. Loaded-instance statistics reset on reload; service image-budget counters do not.

YOLO-family requests accept numeric top-level `confidence` and `nms_iou` in [0,1]. Defaults are confidence 0.125 and raw NMS IoU 0.3 for detection, 0.45 for seg/pose/obb. End-to-end paths validate but do not apply NMS IoU. Controls are request-local; confidence does not change mask binarization or keypoint confidence. OCR rejects either field, even when explicitly set to a default. Wrong types and nonfinite/out-of-range values fail rather than falling back. Optional integer `timeout_ms` is 1–300000.

### OpenAI-like subset

This subset supports all five inference tasks; tracking and subtitles use their native HTTP APIs, not chat or MCP tools. Install the client dependency with `python -m pip install openai` and run the example from the repository root. `GET /v1/models?limit=100` lists `<task>:<name>` IDs; limit is 1–200. When `has_more` is true, pass `next_cursor` unchanged as `after`. Chat accepts inline base64 PNG/JPEG/WebP/BMP data URLs, not remote URLs. WebP requires codec-enabled builds. At least one image is required; order is preserved. Text does not guide inference: this is not a language model.

```python
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11451/v1", api_key="unused")
data = base64.b64encode(Path("app/assets/test/hd2.png").read_bytes()).decode("ascii")
response = client.chat.completions.create(
    model="yolo:hd2-fp32",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}
    ]}],
    response_format={"type": "json_object"},
    extra_body={"confidence": 0.1, "nms_iou": 0.3},
)
print(response.choices[0].message.content)
```

The SDK merges `extra_body` into the HTTP body; a literal wire `extra_body` field is unsupported. Remove detector controls for OCR. The content is the full task result encoded as JSON, not a narrative answer. Supported extras are `stream`, `timeout_ms`, `n:1`, response format `text`/`json_object`, and omitted/`auto` image detail. Other generation controls, tool calls, JSON Schema output and the Responses API are unsupported.

`stream:true` emits role, complete JSON content, finish reason and `[DONE]`; this is not token or per-image streaming. Headers commit only after whole-batch success, so earlier errors remain ordinary HTTP JSON. Body and serialized-result caps are 64 MiB. **SDK `api_key` is not authentication** on this backend.

### MCP HTTP+SSE

Use legacy SSE, not Streamable HTTP. Copy [.mcp.json.example](.mcp.json.example) only if local configuration is absent; never overwrite another client's settings:

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

If it exists, merge only the `vision-simple` entry into `mcpServers`. Start the actual server separately, open Claude Code at the project root and approve the project server. The template uses `type:"sse"` at `http://127.0.0.1:11451/mcp/sse`; adjust the port if needed.

Connect to `GET /mcp/sse`, read the `endpoint` event and POST JSON-RPC to that relative URI. Send `initialize` with `protocolVersion`, object `capabilities`, and `clientInfo` containing `name/version`, then `notifications/initialized`. HTTP 202 only acknowledges receipt; JSON-RPC responses arrive on the original SSE connection. Versions supported: `2024-11-05`, `2025-03-26`, `2025-06-18`, `2025-11-25`.

`list_models` accepts limit 1–200 and opaque cursor. `infer_yolo`, `infer_ocr`, `infer_seg`, `infer_pose`, `infer_obb` accept raw model names, raw-base64 `images` and optional timeout. Images must be nonempty; YOLO tools accept detector controls, OCR does not. Modern results provide `structuredContent` plus equivalent JSON text; older versions retain full JSON in text. Execution failures use `isError:true`; protocol failures use JSON-RPC errors. Cancellation targets the session's request ID; disconnect cancels its work, then native stages drain.

Limits are 32 sessions, 2 execution workers, 16 queued jobs, 8 active calls/session, 64 MiB POST bodies and 8 MiB output/socket buffering. Slow readers and oversized output close the session; reconnect with smaller batches. Idle sessions expire after about five minutes; heartbeats run every 15 seconds. Host and supplied Origin must match trusted loopback or an explicitly configured nonwildcard backend authority and port. Global permissive CORS does not bypass this check. For proxies, disable SSE buffering and allow long connections; external authentication remains necessary.

### Tracking workflow

Run a detector, then submit detections to an independent session. There is no fused detector/tracker or automatic video-decoding route.

```json
{"algorithm":"bytetrack","options":{"min_hits":2}}
```

POST this to `/v1/tracking/sessions`; retain the returned `id`. Submit frames to `/v1/tracking/sessions/{id}/frames`:

```json
{"frame_index":0,"timestamp":0.0,"detections":[{"class_id":0,"confidence":0.9,"bbox":[10,20,30,40]}]}
```

Both index and finite nonnegative timestamp must strictly increase. Rejected frames do not advance state. Timestamps drive motion; index gaps affect expiration. Results contain confirmed tracks observed this frame, not lost predictions. First-frame tracks are confirmed immediately; later tracks must meet `min_hits`. For detector chaining, set detector confidence below the tracker's low threshold (default 0.1) to preserve low-score candidates, accounting for strict raw-detection comparison.

GET the session for status; list with `/v1/tracking/sessions?limit=100` and opaque cursor. POST `{}` to its `/reset` endpoint to restart IDs/timing; DELETE removes it. BoT-SORT uses `algorithm:"botsort"`: camera motion defaults on and requires raw-base64 PNG/JPEG `image` on every frame, same dimensions and at least 8×8 pixels; use `camera_motion:false` for detection-only input. `appearance:true` requires caller-produced finite nonzero embeddings on every detection, consistent dimension ≤512. No ReID model is bundled.

Bounds: 32 sessions, 256 detections/frame and tracks/session, 4 MiB bodies and 16,777,216 image pixels. Sessions expire after 300 seconds since creation or last successful step/reset; reads do not renew them. Busy sessions return 409 rather than queueing. Threshold/capacity options and error schemas are in OpenAPI. Nonempty browser Origin is rejected; session IDs and CORS are not authentication or tenant isolation.

### Video subtitle workflow

Configure real OCR weights and dictionary first. Create, upload, poll, save each desired format, then delete the job. Replace `JOB_ID` with create's returned ID; `clip.avi` is your MJPEG AVI video. Use `curl.exe` in PowerShell.

```sh
curl --fail -sS -X POST http://127.0.0.1:11451/v1/subtitle/jobs -H 'Content-Type: application/json' -d '{"model":"ppocr-v4","sample_interval_ms":200,"roi":[0,0.5,1,0.5],"min_confidence":0.5,"stable_samples":2,"gap_samples":2}'
curl --fail -sS -X PUT http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID/video -H 'Content-Type: application/octet-stream' --data-binary @clip.avi
curl --fail -sS http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID
```

Poll until `state` is `completed`; a 202 upload is not proof of successful decoding. Options are top-level fields, not nested under `options`. This cross-platform Python standard-library client saves SRT to a same-directory temporary file, replaces the destination only after transfer/write success, then deletes the server job:

```python
import json
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

job_url = "http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID"
result = Path("clip.srt").resolve()
with urlopen(job_url, timeout=30) as response:
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

Empty SRT is valid. Use `subtitles.vtt` and `clip.vtt` for WebVTT, and save **all** wanted formats before DELETE. Download/write/replacement failure preserves the original local result and server job, subject to normal expiry. Never download directly over your only local copy or delete the job after a failed transfer.

Upload raw bytes, not multipart, base64, remote URLs or server paths. Only `created` jobs accept upload; interrupted/failed uploads need a new job. Typical states are `created → uploading → queued → running → completed`, or `failed`; inspect `error_code`. POST `{}` to `/v1/subtitle/jobs/{id}/cancel` for cooperative cancellation, then poll. DELETE accepts created or terminal jobs; cancel active jobs first. List uses `/v1/subtitle/jobs?limit=100` with opaque cursor.

One worker handles at most **8 jobs**, including retained results and pending uploads; successful upload completion determines FIFO order. Limits include 64 MiB/video, 1800 seconds, 1,000,000 decoded frames, 16,777,216 pixels/frame, 10,000 cues and 2 MiB accumulated text. Default sampling is 200 ms (range 100–5000); ROI is normalized xywh inside the frame, default bottom half. Stable/gap observations are 2–10, default 2. OCR text is normalized and escaped; no speech recognition or fuzzy matching is performed.

Both platforms decode bounded single-stream **MJPEG AVI**, including OpenDML; other AVI codecs fail asynchronously. MP4-family upload/decoding is available only in Windows builds with the Media Foundation reader and suitable native codecs. Linux rejects MP4; MKV/ASF are disabled on both. Header admission is not full codec validation, so accepted upload can still fail later. FFmpeg is a fixture-generation tool, not a runtime decoder dependency.

Created/uploading jobs become expiry-eligible after 60 seconds without accepted activity; terminal results become eligible 300 seconds after the job enters its terminal state. Queued/running/cancelling jobs do not expire. Reads/downloads do not extend retention; housekeeping or service access sweeps expired rows, and unlink failures can delay cleanup. An already accepted download may finish after expiry/deletion, while new downloads return 404. Preserve your saved result and delete promptly after successful saving. Jobs, IDs and Origin checks provide no authentication or tenant isolation.

## 5. Deployment and diagnostics

### Keep the backend private

Native defaults bind to `127.0.0.1`. Inference, tracking, subtitle, OpenAI and MCP APIs have **no built-in user authentication or tenant isolation**. Public access requires a TLS-terminating authenticated proxy with separate user and administrator policy. Origin restrictions, CORS, random IDs and SDK keys are not authentication.

Cache administration is disabled by default. To enable only stats/unload, set this in deployed `config/server.yaml` and provision the named environment variable before startup:

```yaml
options:
  http_management_token_env: "VS_MANAGEMENT_TOKEN"
```

The name is an example, not a default. It must be a valid environment-variable identifier ≤128 bytes; its secret must be 1–4096 visible ASCII bytes without whitespace. Invalid/missing configured secrets fail startup. Secrets are captured at startup and not stored in YAML. Send `Authorization: Bearer <secret>`; missing/wrong credentials yield 401. Even correct credentials must pass trusted Host and optional Origin checks: loopback or explicit nonwildcard bind authority with the listener port. Wildcard binds and forwarded client addresses do not grant trust.

Unless a separate administrator-authorized proxy route is configured, deny both exact paths:

```nginx
location = /v0/infer/stats { return 403; }
location = /v0/infer/unload { return 403; }
```

An authorized route must preserve the caller's bearer and rewrite Host/Origin only to a trusted backend authority. Authorize the original client Origin before rewriting; never inject a shared admin secret into general inference traffic. Keep secrets out of logs and command history.

Removing/emptying `http_management_token_env` and restarting disables management on current builds. Merely unsetting a still-configured variable fails startup. **Older binaries can ignore this option and restore unauthenticated management:** retain hard proxy denies and private-network isolation before rollback, then verify external requests remain rejected.

### Scheduling, budgets and health

Server options below are string values under `options`:

| Option | Default | Inclusive range / purpose |
| --- | --- | --- |
| `infer_idle_timeout_ms` | `"300000"` | Idle eviction; `"0"` disables automatic eviction |
| `infer_sweep_interval_ms` | `"1000"` | Idle-cache sweep interval |
| `http_data_workers` | `"4"` | 1–32 |
| `http_data_queue_capacity` | `"4"` | 1–128 |
| `http_control_workers` | `"1"` | 1–32 |
| `http_control_queue_capacity` | `"4"` | 1–128 |
| `infer_pipeline_capacity` | `"4"` | 1–64 resident frame tasks |
| `infer_pipeline_max_batches` | `"4"` | 1–64 admitted image batches |
| `infer_max_batch_images` | `"128"` | 1–4096 |
| `infer_timeout_ms` | `"60000"` | 1–300000 ms |
| `infer_max_image_pixels` | `"16777216"` | Positive integer, per image |
| `infer_max_batch_decoded_bytes` | `"67108864"` | Positive integer, per batch |
| `infer_max_inflight_decoded_bytes` | `"268435456"` | Positive integer, shared across inference adapters |
| `ocr_rec_batch_size` | `"1"` | 1–64 recognition crops |

Blocking work runs outside four listener IO loops. Data/control lanes independently bound resident handlers by workers + queue capacity; management can itself overload. Inference admission occurs after complete-body submission, before business JSON parsing, with 503 and `Retry-After: 1` on saturation. Sequential keepalive works; concurrent HTTP/1 pipelining on an active asynchronous connection is rejected by closing it—use separate connections.

The service preflights image headers before decoding and charges estimated BGR input bytes `width*height*3`. Request credit spans preflight through physical completion. Limits are **not RSS guarantees**: compressed bodies, codecs, ORT arenas/workspaces and output masks are separate. Header coverage is PNG/JPEG/BMP/PNM/PFM/HDR/Sun Raster; WebP requires codec support. TIFF/JP2/EXR/AVIF are not enabled. Payload decoding must still succeed as `CV_8UC3`; supported headers alone do not guarantee acceptance.

`/livez` returns 200 alive; `/readyz` returns 200 ready while accepting work and 503 while draining. Both bypass model/cache/dispatch locks; saturation does not make readiness false. Docker probes liveness with a two-second deadline. Deadlines include queue wait, model loading and decoding, not response delivery. Disconnect/timeout/shutdown cancellation is cooperative: native calls and owned inputs drain before credits are refunded; it is not hard ORT preemption.

### Preflight, warmup and private OCR exports

Run diagnostics from the deployment working directory with configuration and assets installed. No arguments start HTTP normally; diagnostic mode starts **no listener**.

```sh
./vision_simple-server --help
./vision_simple-server --diagnose preflight --model yolo:hd2-fp32 --model ocr:ppocr-v4
./vision_simple-server --diagnose warmup --model yolo:hd2-fp32 --image assets/hd2.png --timeout-ms 60000
./vision_simple-server --diagnose warmup --model ocr:ppocr-v4 --image assets/hd2.png --debug-dir ocr-study-001 --debug-max-bytes 67108864 --debug-max-files 64
```

Use `.\vision_simple-server.exe` on Windows. Select 1–16 distinct `task:name` models. Preflight actually initializes sessions without a frame; warmup executes a cold batch then the same batch warm, and unloads before advancing. Warmup accepts at most `min(infer_max_batch_images,128)` images, 48 MiB total raw files and 64 MiB base64. Existing decode/admission/pipeline limits still apply. Each attempted call has its own timeout; native drain may exceed it.

The JSON report distinguishes configured, loadable and smoke-tested states, normalized failures, runtime/provider capabilities and elapsed stage timings. It is valid **only for that invocation** and does not assess HTTP readiness. CPU fallback is allowed; successful provider selection or inference does not prove every operator's hardware placement. Cold means a fresh service cache, not cold OS/device caches. Exit 0 requires selected operations, unload and stdout commit to succeed; exit 1 is operational failure and exit 2 argument misuse.

OCR export is opt-in, requires exactly one OCR selection and 1–16 approved images, and persists sensitive input pixels. Use a **trusted current working directory** and a new portable ASCII basename (1–64 characters, alphanumeric first, then alphanumeric/underscore/hyphen; no paths, dots or reserved device names). The destination must not exist, including links/junctions. OS-private exclusive creation prevents intentional overwrite; it does not make an untrusted directory safe.

Hard maxima/defaults are 64 MiB and 64 files, counting actual encoded bytes and manifest. The export reuses the successful warm result and writes original PNGs, numeric box overlays and a manifest; it does not promise crops or detector masks. No recognized text or supplied paths appear in the report/manifest, but pixels remain sensitive. Failure rolls back owned artifacts; retain only on exit 0 with `debug.retained:true`. Successful exports need manual cleanup after confirming the exact unchanged directory: `rm -r -- './ocr-study-001'` or `Remove-Item -LiteralPath '.\ocr-study-001' -Recurse`. Never use a wildcard or substitute a parent directory.

## 6. Build variants and checks

### Providers and cross builds

Enable the needed provider at build time, then select it at runtime in `config/server.yaml`:

```yaml
host: "127.0.0.1"
port: 11451
options:
  infer_framework: "kONNXRUNTIME"
  infer_ep: "kCPU"
  infer_device: "0"
```

Build switches are `--with_dml=y`, `--with_cuda=y`, `--with_tensorrt=y` and `--with_rknpu=y`; runtime values are `kDML`, `kCUDA`, `kTensorRT`, `kRKNPU`. Windows builds default to compiling DML support, but runtime still defaults to CPU. On Windows, disable DML with `--with_dml=n` when building CUDA/TensorRT, because DML takes priority in dependency selection.

RKNPU requires its build macro, an RKNPU-enabled ORT, DDK/drivers and compatible models; context admission is not session/device verification. DML/CUDA/TensorRT device IDs must fit 0–INT_MAX; CPU ignores the ID and RKNPU requires device 0. CPU fallback is allowed, so provider selection is not guaranteed hardware placement.

```sh
# Native Clang 18 + libc++ alternative
xmake f -p linux -a x86_64 --toolchain=clang --cc=clang-18 --cxx=clang++-18 -m release --runtimes=c++_shared --with_cuda=n --with_tensorrt=n --with_rknpu=n -y
# Cross builds; configure and build each separately
xmake f -p linux -a arm64 --cross=aarch64-linux-gnu- -m release
xmake build server
xmake f -p linux -a riscv64 --cross=riscv64-linux-gnu- -m release
xmake build server
```

The project supplies libc++ 18's experimental-library flags. Native CPU tests, cross-artifact checks, provider compilation and real accelerated-device inference are separate evidence. Consult [Dockerfiles](docker/) for ARMv7, ARM64 CPU/RKNPU, RISC-V and CUDA/TensorRT variants; each needs its own runtime acceptance.

### Published containers

[GHCR](https://github.com/users/lona-cn/packages/container/package/vision-simple) publishes Linux CPU amd64 and arm64 manifests. Release tags use `<version>-cpu` and `sha-<full commit SHA>-cpu`, with `-amd64`/`-arm64` architecture suffixes. Stable releases update `latest`; prereleases do not. Prefer `ghcr.io/lona-cn/vision-simple@sha256:...` from a successful release summary for immutable deployment. A newly published private package must be made public for anonymous pulls.

Publication builds on native architecture runners, smoke-tests each image and checks manifest digests. This workflow policy is not a claim that a particular run passed. ARMv7/RISC-V and accelerated images are outside this manifest. The amd64 CPU image requires AVX/AVX2/F16C. Test a local image with `python3 scripts/test_docker_smoke.py --image vision-simple:local`.

### Representative checks

Run checks from the repository root with a CPU build and real Git LFS fixtures. Missing resources fail, including intentional invalid-model fixtures. Header/byte checks do not replace model inference or media prerequisites.

```sh
git lfs pull
python scripts/check_ci_fixtures.py --project-root . --layer native
python scripts/check_ci_fixtures.py --project-root . --layer http
xmake build test_pipeline
xmake run test_pipeline --project-root .
xmake build test_ocr_batch
xmake run test_ocr_batch --project-root .
xmake build test_ocr_morphology_dataset
xmake run test_ocr_morphology_dataset --project-root .
```

The morphology study renders a deterministic synthetic word-geometry corpus and prints metrics without persisting images. Its alternative 1/1/16 settings are an experiment, not new defaults or a universal accuracy recommendation. Common/conversion/geometry, trace IDs, postprocessing, decoder, configuration, task, image-budget, tracker and subtitle regressions are also available. Interactive `test_yolo`/`test_ocr` demos are not headless acceptance tests.

Documentation checks need Python ≥3.10 and test-only dependencies, preferably in a virtual environment. Set `server` to your built executable (Windows example below); the drivers launch isolated ports/processes and do not use an existing service at port 11451.

```powershell
python -m pip install -r scripts/requirements-doc-tests.txt
$server = 'build/windows/x64/release/vision_simple-server.exe'
python scripts/test_documentation_validation.py
python scripts/test_documentation_examples.py --server "$server" --project-root .
python scripts/test_http_regression.py --server "$server" --project-root .
python scripts/test_protocol_regression.py --server "$server" --project-root .
python scripts/test_tracking_regression.py --server "$server" --project-root . --yolo-model app/assets/test/hd2-yolo11n-fp32.onnx --yolo-image app/assets/test/hd2.png
python scripts/test_subtitle_regression.py --server "$server" --project-root .
```

For Linux/custom output, discover the current target and pass it to the same drivers:

```sh
server="$(xmake lua -q -c "import('core.project.config'); config.load(); import('core.project.project'); io.write(path.absolute(project.target('server'):targetfile()))")"
python3 scripts/test_documentation_examples.py --server "$server" --project-root .
```

The documentation driver validates OpenAPI, schemas/images, documented API examples and MCP handshake/discovery; it does **not** execute the code blocks in this README. Subtitle regression additionally needs FFmpeg with the required generation codecs and a usable font (Windows Arial/Linux DejaVuSans by default); use `--ffmpeg`/`--font` overrides. Windows MP4 scenarios need Media Foundation. These are test prerequisites, not server FFmpeg dependencies.

See [scripts](scripts/) for management security, dispatch, image-budget, model-registry, task, subtitle and release-policy drivers, and [MCP evaluations](scripts/mcp_evals.xml) for multi-step client scenarios. Native CI execution and cross-build architecture checks remain distinct; neither CPU regressions nor successful GPU-provider compilation establishes real CUDA/TensorRT/RKNPU inference.

## 7. License

This project is licensed under [Apache-2.0](LICENSE). Model copyrights belong to their original authors; the project license does not replace YOLO, PaddleOCR or other software/model licenses. Review those terms before redistributing weights or deploying exported models.
