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

`vision-simple` is a cross-platform C++23 vision inference library built on ONNXRuntime, with a C++ API and standalone HTTP server. It supports YOLO detection, instance segmentation, pose, oriented boxes and OCR, plus detection tracking, video text extraction, OpenAI-like HTTP and MCP SSE.

This guide is for users building, deploying and making their first inference request. It describes **current source code**, not a guarantee that older release archives or images include every feature.

**Quick links:** [Build and run](#build-project) · [Docker](#deploy-http-service-with-docker) · [Model configuration](#task-registry-and-unified-model-configuration) · [YOLO26](#yolo26-yolov26) · [Protocols](#shared-service-openai-like-http-and-mcp-sse) · [Tests](#run-tests)

## Features

- **Inference tasks:** YOLOv10/v11/26 detection; YOLO11/26 instance segmentation, pose and oriented boxes; PP-OCR v3/v4 CTC and a constrained SAR recognition contract.
- **Bounded service:** lazy model loading, active leases, idle eviction, statistics, staged pipelines and cooperative cancellation.
- **Temporal processing:** ByteTrack / BoT-SORT sessions; asynchronous OCR video subtitle jobs producing SRT/WebVTT, not speech transcription.
- **Platforms:** Windows x64, Linux x86_64, and cross-build configurations for ARM64, ARMv7 and RISC-V 64. Cross-building is not target-hardware runtime verification.
- **Execution providers:** CPU, DirectML, CUDA, TensorRT and RKNPU build options. Availability depends on dependencies, hardware and model exports; see compatibility notes. No fixed binary size, memory usage or frame rate is promised.
- **Integration:**
  - C++23 API returning errors through `std::expected`.
  - [Containers](https://hub.docker.com/r/lonacn/vision_simple); the current publication workflow covers Linux amd64 CPU only.
  - [HTTP API](doc/openapi/server.yaml) for application integration.

### <div align="center"> YOLOv11 </div>
![hd2-yolo-gif](doc/images/hd2-yolo.gif)

### <div align="center"> OCR (HTTP API) </div>

![http-inferocr](doc/images/http-inferocr.png)
## Quick start

### Deploy HTTP Service with docker

Build the CPU image from current source rather than treating the historical `0.4.1-cpu-x86_64` tag as the latest feature set:

```sh
git clone --recurse-submodules https://github.com/lona-cn/vision-simple.git
cd vision-simple
git lfs install
git lfs pull
docker build --platform linux/amd64 -t vision-simple:local -f docker/Dockerfile.debian-bookworm-x86_64-cpu .
docker run -it --rm --name vs -p 127.0.0.1:11451:11451 vision-simple:local
```

Requires Docker, Git LFS and an x86_64 CPU with AVX/AVX2/F16C. The first build downloads and compiles dependencies. The image includes default YOLO11/PP-OCR configuration and test models, not YOLO26 weights. See [Docker Hub publication](#docker-hub-publication) for published-image selection and verification.

In another terminal, run `curl http://127.0.0.1:11451/v0/infer/models` (`curl.exe` on Windows). Successful discovery proves only that configuration is listed, not that weights load or inference succeeds. Use the [inference example](#send-an-inference-request) below with `hd2-fp32` for the default detection model.

See [OpenAPI](doc/openapi/server.yaml) for the full API. The server has **no authentication or tenant isolation**; do not expose it directly to the Internet. Source defaults bind to `0.0.0.0`; this container example publishes only on loopback. For native execution, set `host: "127.0.0.1"` or use an authenticated proxy.

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
| 503 | `service_overloaded`, `service_unavailable`, `request_cancelled` | `null` |
| 504 | `request_timeout` | `null` |

Clients relying on HTTP 200 with textual errors must migrate to HTTP status and `error.code`; do not branch on `message`. Third-party exception details remain in logs. See the [OpenAPI contract](doc/openapi/server.yaml).

### Model lifecycle and concurrency

- `POST /v0/infer/unload` accepts `{"kind":"yolo","model":"hd2-fp32"}`; `kind` supports `yolo`, `ocr`, `seg`, `pose` and `obb`. An idle model returns `200 {"kind":"yolo","model":"hd2-fp32","unloaded":true}`; active models return `409 model_busy`; absent models return `404 model_not_loaded`. The next inference reloads transparently. Unload and stats cover all five tasks; only the legacy `/v0/infer/models` catalog is restricted to `yolo`/`ocr`.
- `GET /v0/infer/stats?limit=100&offset=0` returns `models`, `total`, `limit`, `offset`, and `idle_timeout_ms`; limit is 1–200. Entries sort by `(kind,name)` and contain `kind`, `name`, `active_requests`, `requests`, `failures`, `total_duration_ms`, and `last_used` (Unix milliseconds). Counters belong to the loaded instance and reset on reload. Errors before acquisition are excluded; duration includes waiting for the model workspace.
- String options in `config/server.yaml`: `infer_idle_timeout_ms: "300000"` and `infer_sweep_interval_ms: "1000"`. Idle time starts at request completion and uses a monotonic clock. Timeout `"0"` disables eviction; sweep interval must be positive.
- Active leases cover decoding, queued inference, postprocessing, and response serialization/send calls. Neither manual nor timer eviction removes active instances. Synchronous C++ `Run` calls serialize per model; HTTP pipeline tasks own workspaces and gate ORT execution per session.
- Unloading releases sessions and model workspaces, but shared ORT arenas/providers may retain allocations: RSS/VRAM need not fall immediately. YOLO `class_name` results still reference model metadata; C++ callers must keep the model alive longer than these views.
- Management endpoints retain the existing unauthenticated/CORS deployment model. Use a trusted network or authenticated proxy. If all four HTTP IO workers are occupied by inference, management traffic waits; there is no reserved management channel.

### Bounded inference pipeline

After every image has decoded successfully, HTTP v0 uses separate preprocessing, ORT, and postprocessing workers. OCR cycles through detection and crop-recognition minibatch dependencies. Outcomes aggregate by input index, not completion order: any failure rejects the entire batch and reports its lowest failing index.

Server options: `infer_pipeline_capacity: "4"` (resident frame tasks, 1–64), `infer_pipeline_max_batches: "4"` (admitted batches, 1–64), `infer_max_batch_images: "128"` (images per batch, 1–4096), and `infer_timeout_ms: "60000"` (default inference deadline, 1–300000 ms). These are not decoded pixel/byte limits.

An optional integer request field `timeout_ms` (1–300000) overrides the deadline. Time starts on handler entry and includes loading/decoding; it controls inference stages, not serialization/network delivery or hard preemption of native calls. Expiry returns `504 request_timeout`; exhausted admission returns `503 service_overloaded` with `Retry-After: 1`. Control errors have null `image_index`. Decode validation still precedes scheduling, preserving `invalid_image` precedence.

C++ callers create `InferPipeline`, then call `Run(model, images, confidence, PipelineControl{stop_token, deadline})`. `Close()` rejects new work and cancels unfinished batches; join callers before destruction. Cancellation takes precedence over timeout, then close, then ordinary inference failures. `Run` drains executing native stages and task destruction before returning, so cancellation never releases a model still in use. Models and input pixels must remain alive throughout the call; external aliases must not modify pixels. Ordinary v0/v1 HTTP disconnect does not cancel work; an MCP session disconnect does request cooperative cancellation.

Synchronous `InferYOLO/InferOCR::Run` signatures remain unchanged and share stage algorithms/session execution gates with the pipeline. Tasks own independent workspaces; each model retains at most two idle pipeline workspaces. Unsupported custom backends return an explicit error rather than wrapping synchronous inference as a fake pipeline.

### Task Registry and Unified Model Configuration

Declare models in `config/models.yaml`, identified by `(task,name)`. Native v1 and MCP use the configured name; OpenAI-like uses `<task>:<name>`. These are the default FP32 detection and OCR entries:

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

Legacy `yolo`/`ocr` lists remain readable and may coexist with non-conflicting canonical entries. Duplicate `(task,name)` declarations are rejected; different tasks may share a name. Version/resource validation remains lazy at first model load. Legacy public DTOs retain compatibility projections; execution reads only canonical `models`, with no parallel cache or configuration lookup path.

Registered tasks are `yolo`, `ocr`, `seg`, `pose` and `obb`; unknown tasks are rejected at the service configuration boundary. All tasks share caching, leases, statistics and eviction. ONNXRuntime is the implemented backend; TVM is unsupported, and the task registry is not a dynamic plugin system.

### YOLO segmentation, pose and oriented boxes

Add canonical entries using your own compatible exports (these example filenames are not bundled weights):

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

- YOLO11 exports must have a static `[1,3,H,W]` input and static raw FP32/FP16 outputs, class-name metadata, and no exported NMS/end-to-end postprocessing. YOLO26 supports the two modes described below. Segmentation requires predictions plus matching mask prototypes; pose uses keypoint channels (`kpt_shape` supports 2 or 3 coordinates); OBB requires one angle channel. Dynamic shapes, batch>1, arbitrary YOLO architectures and embedded-NMS graphs are not supported.
- `POST /v1/infer/{task}` accepts `{"model":"raw configured name","images":["raw base64"],"timeout_ms":60000}` for all five task IDs. It shares ordered, whole-batch failure semantics and pipeline limits; native v1 caps request bodies at 64 MiB. Existing v0 inference responses remain unchanged.
- New task responses contain `class_names` and per-image `results`. Every object has `class_id` and `confidence`. Segmentation adds integer original-image `bbox:[x,y,width,height]` and `mask_png_base64`: a binary 0/255 PNG cropped to that box, not a full-image mask. Place its top-left at the box origin. C++ `InferYOLOTask` returns a separate `YOLOTaskFrameResult` variant; each segmentation `CV_8UC1` mask owns its pixels independently of inference workspaces.
- Pose adds the same box and `keypoints:[{x,y,confidence}]` in original-image floating-point pixels. Keypoints are not clipped to the image. OBB instead returns four ordered original-image `corners:[[x,y],...]` and `angle` in radians along corner 0 → 1; corners may lie outside the image and are not replaced by an axis-aligned box.
- OpenAI-like model IDs are task-qualified (for example `seg:segment`, `pose:pose`, `obb:oriented`); MCP generates `infer_seg`, `infer_pose`, `infer_obb` alongside the existing tools. The result JSON retains each task's geometry.

### YOLO26 (YOLOv26)

`kV26 = 26` supports detection, instance segmentation, pose and oriented boxes, preserving existing YOLOv10/YOLO11 behavior. Models must use static, single-image ONNX tensors with FP32 or FP16 inputs/outputs. FP16 weights do not imply that every I/O tensor is FP16.

Start with **FP32 + NMS-free** detection before adding other tasks. The server build must contain the YOLO26 implementation; older release archives or Docker tags do not imply support.

| Service task | Raw output (external NMS) | NMS-free output (no additional NMS) |
|---|---|---|
| `yolo` | `[1,4+nc,A]`, `xywh` + class scores | `[1,K,6]`, `xyxy,score,class_id` |
| `seg` | `[1,4+nc+nm,A]` + prototypes | `[1,K,6+nm]` + `[1,nm,Hm,Wm]` prototypes |
| `pose` | `[1,4+nc+nk*nd,A]` | `[1,K,6+nk*nd]`, `nd=2/3` |
| `obb` | `[1,5+nc,A]` | `[1,K,7]`, **`xywh,score,class_id,angle`** |

Neither `A` nor `K` is hardcoded to 8400 or 300. Preserve `names`, the correct `task`, explicit `end2end`, and `args.nms` metadata; pose also requires `kpt_shape`. Missing/conflicting metadata, invalid shapes and embedded NMS are rejected rather than inferred from filenames or `[1,K,6]` alone.

#### Export models

Run from the repository root. Export tools are development dependencies, not service runtime dependencies; the first run downloads official nano checkpoints.

```bash
python -m pip install ultralytics==8.4.159 onnx==1.20.1 onnxruntime==1.24.3 torch==2.14.0 torchvision==0.29.0
python scripts/export_yolo26.py --output build/yolo26 --imgsz 640
```

For detection only, skip the other tasks and half-precision exports:

```bash
python scripts/export_yolo26.py --output build/yolo26 --imgsz 640 --tasks detect --precisions 32
```

This produces `detect_raw_fp32.onnx` and `detect_e2e_fp32.onnx`. Exporter task `detect` maps to service `task: yolo`; other task names are `seg`, `pose` and `obb`. Both `--tasks` and `--precisions` accept multiple values; precisions are `32` and `16`.

Without task or precision filters, the script exports 16 models: four tasks × raw/e2e × FP32/FP16, with batch=1, opset=17, dynamic=False and simplify=False. The generated `manifest.json` records dependency versions, checkpoint/ONNX SHA256 values, export arguments and actual tensor metadata. This exporter version uses `nms=None` for raw, `nms=False` for NMS-free and `quantize=16` for half precision. CPU FP16 conversion is followed by a topological sort to order appended I/O Cast nodes correctly, then ONNX checker and CPU ORT loading. Do not remove mode metadata or assume older exporters assign the same meaning to these arguments.

#### Configure and start

The server reads `config/server.yaml` and `config/models.yaml` from its **working directory**. Relative model paths also resolve there. Place the required ONNX files under its `assets/` directory and add these entries to the `models` list. Keep only models actually exported and never repeat a `(task,name)` pair.

Source builds copy base configuration beside the executable. The default Windows x64 Release output directory is `build/windows/x64/release/`. Launch from the directory containing `config/` and `assets/`; rebuilding may overwrite configuration, so use a separate deployment directory for production.

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

Select an execution provider in `config/server.yaml`, for example CPU:

```yaml
host: "127.0.0.1"
port: 11451
options:
  infer_framework: "kONNXRUNTIME"
  infer_ep: "kCPU"
  infer_device: "0"
```

Launch the built `vision_simple-server` (`.exe` on Windows). Models load on first inference; restart after changing configuration or model files. Bind beyond loopback only on a trusted network or behind an authenticated proxy.

#### Send an inference request

This client uses only the Python standard library. Replace `image.jpg` with a local image path; `model` must match the configured `name`:

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

`images` contains raw base64, **not** `data:image/...;base64,...` URLs. Use `/v1/infer/seg`, `/v1/infer/pose` or `/v1/infer/obb` and the corresponding model name for other tasks. Detection `bbox` uses original-image pixels `[x,y,width,height]`; no detections produces an empty array for that image.

Use existing `POST /v1/infer/{task}` routes; detection also supports `/v0/infer/yolo`. Responses, caching, ordered batches and whole-batch failure semantics are unchanged. C++ detection uses `InferYOLO::Create(context, path, YOLOVersion::kV26)`. Other tasks now take an explicit version, for example `InferYOLOTask::Create(context, path, YOLOTask::kPose, YOLOVersion::kV26)`. Existing task callers must insert `YOLOVersion::kV11` after `task`, before the optional device ID.

Postprocessing keeps this library's contract: raw detection uses NMS IoU 0.3; other raw tasks use 0.45. OBB uses polygon IoU, **not Ultralytics' probabilistic IoU**. NMS-free outputs never undergo another suppression pass. Letterbox uses black padding, keypoints retain out-of-frame coordinates, and masks interpolate logits before thresholding and cropping to integer bounding boxes. Comparisons against Ultralytics must align preprocessing and account for these documented differences.

#### Verified compatibility and limitations

**Verified environment:** Windows x64 Release, C++ ORT 1.20.0 / DirectML 1.15.4, with Python ORT 1.24.3 as the reference runtime.

| EP | Raw FP32 | NMS-free FP32 | Raw FP16 | NMS-free FP16 |
|---|---|---|---|---|
| CPU | All four tasks passed | All four tasks passed | All four tasks passed | All four tasks passed |
| DirectML | All four compared successfully | All four compared successfully | All four ran successfully | All four failed during ORT initialization |

This matrix records prior Windows verification, not a guarantee for every machine, driver or model export, and was not rerun for this documentation update. **Use FP32 or verified raw FP16 on the stated DML stack, not these NMS-free FP16 artifacts.** Initialization failures return controlled `model_load_failed` errors; no silent mode substitution occurs. Validate results and resource usage with your own images, exports and provider before deployment.

Classification, semantic segmentation, depth, YOLOE, dynamic shapes and batch>1 are outside this support. CUDA/TensorRT are outside the YOLO26 verification scope above. YOLO26 weights and exports are not distributed with this repository; review the applicable Ultralytics software/model licenses. YOLO11/PP-OCR test resources are separately fetched through Git LFS.

### Temporal tracking sessions

Tracking is a separate stateful `TrackingService`, not an inference task or an automatic video decoder. Supply detections from inference or another detector. Each C++ `Tracker`/HTTP session is an independent stream with its own IDs, motion and appearance state.

| Operation | HTTP request |
|---|---|
| Create | `POST /v1/tracking/sessions` with `{"algorithm":"bytetrack","options":{"min_hits":2}}`; returns 201, `id`, `algorithm`, `status` and `Location` |
| Step | `POST /v1/tracking/sessions/{id}/frames` with the frame below |
| Status | `GET /v1/tracking/sessions/{id}`; returns `id`, `algorithm`, `status` |
| List | `GET /v1/tracking/sessions?limit=100`; returns `sessions` (IDs) and nullable `next_cursor`; pass it unchanged as `cursor` (limit 1–100) |
| Reset | `POST /v1/tracking/sessions/{id}/reset` with `{}`; returns `{"reset":true}` |
| Delete | `DELETE /v1/tracking/sessions/{id}`; returns 204 |

```json
{"frame_index":0,"timestamp":0.0,"detections":[{"class_id":0,"confidence":0.9,"bbox":[10,20,30,40]}]}
```

Tracks created on the session's first frame are immediately confirmed; tracks created later must meet `min_hits`. An empty `tracks` array can still be successful, for example when there are no detections or a new target is not yet confirmed.

- `timestamp` is finite, nonnegative seconds; `frame_index` is an integer in 0–9007199254740991. Both must strictly increase within a session. Elapsed seconds control motion prediction; index gaps count toward expiration. Rejected frames do not advance state. Reset starts a fresh sequence, including IDs and timing. Step returns `frame_index`, `timestamp`, and `tracks:[{track_id,class_id,confidence,bbox}]`; only confirmed tracks observed this frame are emitted, not lost predictions. Status contains nullable `last_frame_index`/`last_timestamp` and `active_tracks`/`lost_tracks`.
- Algorithms: `"bytetrack"` or `"botsort"`. Defaults: `high_threshold:0.5`, `low_threshold:0.1`, `new_track_threshold:0.6`, `match_threshold:0.8`, `max_lost_frames:30`, `min_hits:2`, `max_tracks:256`, `max_detections:256`, `camera_motion:true`, `appearance:false`, `proximity_threshold:0.5`, `appearance_threshold:0.25`. Thresholds are in [0,1], with low < high ≤ new; matching thresholds are maximum costs, not minimum IoU. `min_hits` is 1–10000, `max_lost_frames` 0–10000, and both capacity options 1–256.
- BoT-SORT camera motion requires an `image` containing raw base64 PNG/JPEG on every frame, same dimensions throughout the sequence and at least 8×8 pixels; disable it with `camera_motion:false` when supplying detections only. Images are bounded to 16,777,216 decoded pixels before decoding. Boxes are finite floating-point xywh with positive dimensions; confidence is in [0,1] and class IDs are nonnegative.
- BoT-SORT `appearance:true` requires a finite, nonzero `embedding` array on every detection, at most 512 elements and a consistent dimension per session. Embeddings come from the caller: no ReID model or weights are bundled, and tracking does not run an embedding model.
- Limits: 32 sessions, 256 detections/frame and tracks/session, 4 MiB request body. Sessions expire after 300 seconds since creation or the last successful step/reset; expiration is swept on service access, and status/list do not renew it. Concurrent access to a busy session returns 409 rather than queueing; separate sessions do not share track identities.
- Errors use `error.{code,message,image_index}` with null `image_index`: 400 `invalid_request`/`invalid_image`, 404 `tracking_session_not_found`, 409 `tracking_session_busy`/`frame_out_of_order`, 503 `tracking_capacity`/`service_unavailable` (with `Retry-After: 1`), or 500 `tracking_failed`. Unknown JSON fields are rejected. POST requires exactly `Content-Type: application/json`; GET/DELETE cannot have bodies. Oversize bodies return 413; unsupported `Expect` returns 417.
- Tracking business requests reject nonempty browser `Origin` with 403 and use `Cache-Control: no-store`; ordinary OPTIONS preflight still passes through global CORS middleware. Neither Origin/CORS nor session IDs provide authentication, and listing is not tenant-scoped. Use a trusted network or authenticated proxy.

### Asynchronous video subtitles

This extracts **visible text with OCR**, not speech. Configure an OCR model and its real detection/recognition weights and dictionary first. The following POSIX-shell commands use `curl` (`curl.exe` on Windows); replace `JOB_ID` with the `id` returned by create:

```sh
# 201 + Location; options are top-level fields, not an "options" object
curl -sS -X POST http://127.0.0.1:11451/v1/subtitle/jobs -H 'Content-Type: application/json' -d '{"model":"ppocr-v4","sample_interval_ms":200,"roi":[0,0.5,1,0.5],"min_confidence":0.5,"stable_samples":2,"gap_samples":2}'
# 202 means accepted for processing, not successful decoding
curl -sS -X PUT http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID/video -H 'Content-Type: application/octet-stream' --data-binary @clip.avi
curl -sS http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID
# Download only after state == "completed"
curl -fS http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID/subtitles.srt -o clip.srt
curl -fS http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID/subtitles.vtt -o clip.vtt
# Alternatively cancel unfinished work; poll until terminal before deletion
curl -sS -X POST http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID/cancel -H 'Content-Type: application/json' -d '{}'
curl -i -X DELETE http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID
```

- Create accepts only `model`, `sample_interval_ms` (integer 100–5000, default 200), `roi`, `min_confidence` ([0,1], default 0.5), `stable_samples` and `gap_samples` (integers 2–10, both default 2). `model` is the configured raw OCR name, 1–256 bytes. ROI is normalized `[x,y,width,height]`, with positive dimensions entirely inside the frame; default `[0,0.5,1,0.5]` is the bottom half. Full-frame example: `{"model":"ppocr-v4","roi":[0,0,1,1]}`. Unknown fields are rejected.
- Sampling selects the first decoded frame at or after the next interval, using actual presentation timestamps in milliseconds, not `sample_index × interval`. OCR lines below `min_confidence` are removed (the model's own recognition filter still applies), arranged top-to-bottom/left-to-right and whitespace-normalized. There is no fuzzy text matching. Identical normalized text needs `stable_samples` consecutive observations; its cue starts at the first of those observations. A confirmed replacement ends the previous cue at that same timestamp. Transient alternatives are suppressed; fewer than `gap_samples` empty observations can bridge an unchanged cue, while a confirmed empty gap ends it at the first empty timestamp. At EOF, an outstanding gap closes there; otherwise the last cue ends at the decoded stream end. Intervals are nonoverlapping and positive; precision depends on sampling and OCR accuracy.
- Downloads contain validated UTF-8 SRT or WebVTT, with normalized control/blank lines and escaped `&`, `<`, `>` to prevent recognized text from becoming markup or cue syntax. Empty successful extraction is valid: empty SRT or a WebVTT header, not fabricated captions.
- Normal states are `created → uploading → queued → running → completed`; failures become `failed`. Cancellation returns 202 and moves active native processing through `cancelling → cancelled`; it is cooperative, not a hard interruption of a decoder/ORT call. Cancelling an already terminal job leaves its result unchanged. Status returns `id`, `state`, `uploaded_bytes`, `decoded_frames`, `sampled_frames`, `position_ms`, nullable `duration_ms`, `cue_count`, and nullable `error_code`. Duration may be unknown; counts/position are progress, not a guaranteed percentage, and `cue_count` while running excludes the open cue. `GET /v1/subtitle/jobs?limit=100` lists job objects and nullable `next_cursor`; pass it unchanged as `cursor` (limit 1–100).
- Upload is raw bytes in a separate PUT, not multipart/base64, a server-local path or a remote URL. Only `created` jobs accept upload; a consumed/interrupted/failed upload cannot restart on the same job. Retry by creating a new job and reuploading; create is not idempotent. A successful upload can still fail asynchronously, so poll `state` and inspect `error_code` (for example `upload_interrupted`, `unsupported_video`, `invalid_timestamps`, `ocr_failed`, `subtitle_limit`), not error-message wording. Partial failed results cannot be downloaded.
- One worker serves at most **8 jobs**, including pending uploads and retained terminal results. Limits: 64 MiB/video, 1800 seconds, 1,000,000 decoded frames, 16,777,216 pixels/frame, 4096 OCR lines/observation, 4096 bytes/line and combined observation, 10,000 cues and 2 MiB accumulated cue text. JSON control bodies are limited to 64 KiB. Input files live in a private server-owned temporary directory and are removed on completion/failure/cancellation/deletion; orderly shutdown removes the directory. Created/uploading jobs expire after 60 seconds without upload activity; terminal results expire after 300 seconds. Polling/downloading does not renew retention; filesystem cleanup failures can retain capacity, and a process crash is not an orderly cleanup guarantee.
- HTTP video decoding is platform-dependent. All platforms have a bounded MJPEG AVI reader (one MJPG/mjpg video stream starting at zero, including OpenDML AVI/AVIX); other AVI codecs are rejected. Windows additionally uses Media Foundation for MP4-family files with a leading `ftyp` box, subject to installed native codecs. Although the underlying reader supports ASF, the current upload entry point rejects it. MKV can pass upload sniffing but fails decoding with `unsupported_video`. Container recognition or upload HTTP 202 does not guarantee decoding. Arbitrary codecs, playlists, image sequences and URL fetching are unsupported.
- MJPEG AVI timestamps follow stream rate/scale; Media Foundation uses actual sample timestamps and positive sample durations, rounding starts down and ends up to milliseconds. `duration_ms` on completion is the actual video end, not an estimated frame count or a longer audio/container duration; native duration can remain null until completion.
- Errors use `error.{code,message,image_index}` with null `image_index`: 400 `invalid_request`; 404 `model_not_found`/`subtitle_job_not_found`; 409 `subtitle_job_busy`/`subtitle_not_ready`; 413 `payload_too_large`; 415 `unsupported_media_type`/`invalid_video`; 503 `subtitle_capacity`/`service_unavailable` (`Retry-After: 1`); 500 `subtitle_failed`. Create/cancel require exactly `application/json`, upload exactly `application/octet-stream`; bodyless operations reject bodies, and unsupported `Expect` returns 417. DELETE returns 204 only for a created or terminal job; cancel and poll other states first.
- This API has **no authentication or tenant isolation**. Subtitle business requests reject nonempty browser `Origin` with 403 and use `Cache-Control: no-store`; ordinary OPTIONS still passes through global CORS middleware. Job IDs are not credentials. Use a trusted network or authenticated proxy.

Real-video regression (run from the repository root with a built server):

```sh
python scripts/test_subtitle_regression.py --server <server-executable> --project-root . --ffmpeg <ffmpeg-executable> --font <font.ttf>
```

Requires real `ppocr_det.onnx`, `ppocr_rec.onnx` and `ppocr_keys_v1.txt` in `app/assets/test`, plus FFmpeg with drawtext/MJPEG/libx264/AAC and a usable TrueType font. `--ffmpeg`/`--font` may be omitted when FFmpeg is on PATH and the platform's default Arial/DejaVuSans font is available. The regression exercises real decoding/OCR, exact HELLO/WORLD intervals with a transient NOISE frame suppressed, SRT/WebVTT agreement, upload interruption, cancellation, capacity/isolation and temporary-file cleanup; Windows also exercises native variable-frame-rate video with a longer audio track. It is not a codec-quality or performance benchmark.

### OCR Decoders and Recognition Batching

- A model-type registry selects CTC for `kPPOCRv3`/`kPPOCRv4` and an independent SAR decoder for `kPaddleSAR`. `kEasyOCR` is not implemented and is rejected at creation instead of silently using Paddle/CTC.
- SAR preserves adjacent repeated characters, skips PAD, stops at the first BOS/EOS, and emits `<UKN>` for unknown characters. Ordinary characters occupy `0..D-1`, followed by UKN, BOS/EOS, PAD; output classes must equal `D+3`. The library retains strict confidence filtering and returns zero confidence for empty text.
- SAR dictionary files list exactly the ordinary characters (include space explicitly when needed), without appended CTC space or special tokens. Server `models.yaml` selects it using `version: "kPaddleSAR"`.
- Supported recognition exports have one float `[N,3,H,W]` input and one float `[N,T,C]` probability output, with RGB preprocessing normalized to `[-1,1]`. This does not claim support for every Paddle architecture requiring extra valid-ratio/attention inputs or different normalization. Non-CTC integration tests execute deterministic SAR prediction graphs in real ORT; no new trained weights are bundled.
- Set `{"ocr_rec_batch_size","4"}` in C++ `InferContext::Create`'s `InferArgs`, or `ocr_rec_batch_size: "4"` in server `options`. Range 1–64, default 1. v0/OpenAI-like/MCP share this setting without response changes.
- Dynamic N: stably group crops having the same preprocessed width, run actual `[N,3,H,W]` minibatches up to the requested size, and restore detection order. Dynamic H defaults to 48; dynamic W retains the original height-multiple rounding and whole-crop resize. Different widths are not mixed using extra padding, avoiding new valid-length truncation.
- Fixed N follows model metadata (1–64); partial tails receive normalized-zero dummy samples whose outputs are discarded. Fixed H/W resize the entire crop to those dimensions. No arbitrary `T × width_ratio` truncation is inferred: real samples decode all T, with SAR stopping at EOS. Exports requiring aspect-preserving padding or extra masks must match this preprocessing contract first.
- DBNet detection uses a 1.5 unclip ratio before recognition cropping, expanding the shrunken text region to avoid truncated glyphs (for example, an `E`-only crop instead of `HELLO`). This changes crop geometry and can change recognized text across both synchronous and pipeline OCR.
- `test_ocr_batch` covers mixed widths, ordering, fixed-N tails, workspace reuse, file-based SAR dictionaries and pipeline isolation. A real ONNX guard fails at N=1, preventing a single-image loop from masquerading as batching. Batching benefits depend on crop sizes, models and providers; measure your actual workload.

### Shared service, OpenAI-like HTTP and MCP SSE

v0, native v1, OpenAI-like and MCP call one `InferenceService`, sharing model loading, leases, stats, idle eviction and pipeline limits. Tracking state belongs to its separate service. Discovery lists configuration, not proof that every model file is loadable.

**OpenAI-like subset**

- `GET /v1/models?limit=100` returns `object:"list"` and `data`. IDs are `<task>:<raw-name>` for `yolo`, `ocr`, `seg`, `pose`, `obb`; `task` identifies the operation. Limit is 1–200. When `has_more` is true, pass the returned `next_cursor` unchanged as query parameter `after`.
- `POST /v1/chat/completions` accepts `model`, `messages`, optional `stream`, `timeout_ms`, `n:1` and `response_format:{"type":"json_object"}` (or `"text"`). User `image_url.url` parts must contain inline base64 PNG/JPEG/WebP/BMP data URLs. Only omitted/`"auto"` detail is supported; remote URLs are never fetched.
- At least one image is required. Images retain message/content order. Text does not alter detection/OCR: these are not language models. Other generation controls, tool calls, JSON Schema output and the Responses API are unsupported; unsupported parameters return 400 rather than silently doing nothing.
- `choices[0].message.content` is the full task result encoded as JSON: YOLO/new tasks include `class_names/results`, OCR includes `results`. Geometry follows the task contracts above; token usage is not fabricated.
- `stream:true` sends `chat.completion.chunk` SSE events (role, complete JSON content, finish reason), then `[DONE]`. This is neither token nor per-image streaming. Headers commit only after whole-batch success; earlier failures remain HTTP JSON errors.
- Errors contain `error.{type,code,message,param,image_index}`. Inference codes match v0; 503 includes `Retry-After: 1`. Request bodies are capped at 64 MiB while receiving (413); serialized inference results are capped at 64 MiB.

**Legacy MCP HTTP+SSE**

Connect to `GET /mcp/sse`, read its `endpoint` event, then POST JSON-RPC 2.0 to that relative URI. Complete `initialize` and `notifications/initialized` before `tools/list` / `tools/call`. Supported versions: `2024-11-05`, `2025-03-26`, `2025-06-18`, `2025-11-25`. Select SSE in clients; this is not Streamable HTTP.

Initialization `params` must include `protocolVersion`, an object `capabilities`, and `clientInfo` containing `name`/`version`. POST HTTP 202 acknowledges receipt only: read JSON-RPC responses from the original SSE connection, not the POST response body.

- `list_models`: optional `limit` (1–200) and opaque `cursor`; returns `data:[{id,kind,name}]` and optional `next_cursor`.
- Generated `infer_yolo` / `infer_ocr` / `infer_seg` / `infer_pose` / `infer_obb`: `{"model":"raw configured name","images":["raw base64"],"timeout_ms":60000}`. Do not pass prefixed catalog IDs or data URLs. Discovery includes input JSON Schemas.
- Modern results contain `structuredContent` and equivalent JSON text; legacy versions retain the full structure in text. Execution errors use `isError:true` with stable `error.code`, `image_index` and recovery advice. Protocol failures use JSON-RPC errors and preserve string versus integer IDs.
- `notifications/cancelled` cancels its session's `requestId`; disconnect cancels all that session's work. Native stages drain before cancellation/timeout returns; ORT is not forcibly interrupted and another session's equal ID is unaffected.
- Bounds: 32 sessions, 2 execution workers, 16 queued jobs, 8 active tool calls/session, 64 MiB POST body, 8 MiB queued output/socket buffers. Heartbeats occur every15s; sustained buffered writes close after about30s, and sessions with no active work expire after5min without messages (checked on heartbeat ticks). Oversized output closes the session; reconnect with a smaller batch.
- Host accepts loopback names and an explicitly configured nonwildcard bind host with matching port. Supplied Origin must identify a corresponding trusted HTTP(S) authority. MCP bypasses permissive global CORS. Remote access requires an explicit trusted bind address or an authenticated proxy rewriting trusted backend Host/Origin.

`.mcp.json` targets the local default port. OpenAI SDK `api_key` is not authentication here: use a trusted network or external authenticated proxy for every protocol. Disable SSE proxy buffering and allow long connections. Regression commands appear below; multi-step agent evaluations are in `scripts/mcp_evals.xml`.

### C++ migration

- `InferYOLO/InferOCR::Create/Run` signatures are unchanged. Run requires a nonempty two-dimensional `CV_8UC3` image and supports non-contiguous ROIs. Grayscale, BGRA, floating-point images and non-finite/out-of-range `[0,1]` confidence return parameter errors.
- YOLO v11 uses class-aware NMS; v10 accepts only end-to-end `[1,N,6]` output and does not repeat NMS. Confidence defaults, black Letterbox padding and OCR detection normalization are unchanged.
- YOLO26 detection uses `YOLOVersion::kV26`. Both path and memory overloads of `InferYOLOTask::Create` require an explicit version after `task`: use `YOLOVersion::kV11` for existing segmentation/pose/OBB callers and `YOLOVersion::kV26` for YOLO26. The optional `device_id` follows the version.
- PP-OCR CTC file-based Create follows the Paddle dictionary convention: the file excludes blank and the trailing space class; the loader appends space. Map-based callers supply every nonblank class with key `class_id - 1`. SAR follows its separate dictionary contract above.
- `HTTPServer::Run/StartAsync` return `HTTPServerResult<void>`; callers must check failures. Empty/overlong hosts and listen failures are rejected without silently binding wildcard. The repository's explicit `0.0.0.0` default is unchanged.
- Helper consumers must recompile and migrate to one geometry path:

```cpp
LetterboxTransform transform;
cv::Mat& padded = helper.Letterbox(image, target_size, transform);
if (padded.empty()) { /* reject invalid or rounded-zero image dimensions */ }
cv::Rect box = VisionHelper::ScaleCoords(transform, cv::Vec4f{x1, y1, x2, y2});
```

`ScaleCoords` accepts floating-point model-space xyxy, reverses actual per-axis scaling, clips endpoints, then rounds endpoints into integer xywh. Old geometry signatures, `DataConverter` and unimplemented uint8 no-op conversions were removed. `Cvt` supports bidirectional FP32/FP16 conversion.

The server has no built-in authentication, hot loading or arbitrary ONNX support. v1/MCP body limits do not cover legacy v0 routes, and legacy inference has no general decoded-pixel limit. Tracking PNG/JPEG and subtitle video frames have the bounds described above, not a blanket production-security guarantee.

## Development and deployment

### Build Project

#### Fetch source and model resources

Install Git, Git LFS, [xmake](https://xmake.io) and the appropriate compiler. Use xmake ≥ 2.9.7 for MSVC/GCC; Clang 18 + libc++ uses xmake ≥ 3.1.1, matching CI.

```sh
git clone --recurse-submodules https://github.com/lona-cn/vision-simple.git
cd vision-simple
git lfs install
git lfs pull
```

Initial configuration downloads dependencies and needs network access and dependency build tools. For an existing checkout, run `git submodule update --init --recursive` first. LFS pointer files cannot be used as ONNX weights.

#### windows/x64

Use a C++23-capable Visual Studio 2022 MSVC toolchain and Windows SDK. This explicitly selects a CPU build; see the next section for DirectML.

```powershell
xmake f -p windows -a x64 --toolchain=msvc -m release --with_dml=n --with_cuda=n --with_tensorrt=n -y
xmake build server
Copy-Item app/assets/test/* build/windows/x64/release/assets/ -Recurse -Force
# Set host to "127.0.0.1" in build/windows/x64/release/config/server.yaml before launching:
Set-Location build/windows/x64/release
.\vision_simple-server.exe
```

#### linux/x86_64

The native CPU CI configuration uses Ubuntu 24.04 + GCC 14; Clang 18 + libc++ 18 is also configured. Install the corresponding C/C++ compiler, Python development tools and package build tools.

```sh
xmake f -p linux -a x86_64 --toolchain=gcc --cc=gcc-14 --cxx=g++-14 -m release --with_cuda=n --with_tensorrt=n --with_rknpu=n -y
xmake build server
cp -R app/assets/test/. build/linux/x86_64/release/assets/
# Set host to "127.0.0.1" in build/linux/x86_64/release/config/server.yaml before launching:
cd build/linux/x86_64/release
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ./vision_simple-server
```

For Clang, replace the GCC configuration command with the following at the repository root; build and launch steps stay the same:

```sh
xmake f -p linux -a x86_64 --toolchain=clang --cc=clang-18 --cxx=clang++-18 -m release --runtimes=c++_shared --with_cuda=n --with_tensorrt=n --with_rknpu=n -y
```

The project supplies the experimental-library compile/link flags required by libc++ 18; no source edits are needed. These paths assume default output settings; with custom `-o`, use the actual targetfile.

**Working directory matters:** the server reads `config/server.yaml`, `config/models.yaml` and relative model paths from it. Building copies base configuration and main assets, but the server target does not automatically copy test models; the commands above explicitly copy test resources. Production deployments need only the weights and dictionary referenced by configuration. Rebuilding may overwrite configuration: use a separate deployment directory and restart after configuration/model changes.

Alternatively, choose a matching platform, architecture and EP archive from [GitHub Releases](https://github.com/lona-cn/vision-simple/releases). Check release checksums and the commit/build configuration in `build-info.json`. Archives collect configuration and resources present in that build directory; they do not guarantee every model is included. Cross-built artifacts still require validation on target hardware.

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

Build options and runtime configuration must agree: after enabling a provider, select `kDML`, `kCUDA`, `kTensorRT` or `kRKNPU` in the string option `infer_ep` in `config/server.yaml`, and choose a device with `infer_device`. Runtime defaults to `kCPU`. The Windows DML build option defaults to enabled, but that does not select DML at runtime.

```powershell
# DirectML (Windows)
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

# RKNPU (Linux only)
xmake f --with_rknpu=y -m release
xmake build server
```

### Run Tests

Run these commands from the repository root. If you just launched the server as above, open another terminal at the repository root. Configure a CPU build first and ensure Git LFS model resources have been downloaded.

```sh
# Build CPU regression targets individually
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

The Python 3 standard-library HTTP driver creates isolated configuration, ports and processes; it does not touch an existing port 11451 service. Missing models, dictionaries, images or failure fixtures fail the run rather than count as SKIP.

```powershell
python scripts/test_http_regression.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
python scripts/test_protocol_regression.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
python scripts/test_yolo_tasks_http.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
python scripts/test_tracking_regression.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
python scripts/test_model_registry.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
```

For Linux or a custom build directory, discover the actual target:

```sh
server="$(xmake lua -q -c "import('core.project.config'); config.load(); import('core.project.project'); io.write(path.absolute(project.target('server'):targetfile()))")"
python3 scripts/test_http_regression.py --server "$server" --project-root .
python3 scripts/test_protocol_regression.py --server "$server" --project-root .
python3 scripts/test_yolo_tasks_http.py --server "$server" --project-root .
python3 scripts/test_tracking_regression.py --server "$server" --project-root .
python3 scripts/test_model_registry.py --server "$server" --project-root .
```

`test_yolo`/`test_ocr` remain interactive demos, not headless acceptance tests. Tiny failure models are checked in; only regeneration requires the development package `onnx` and `scripts/generate_reliability_fixtures.py`, not a server runtime dependency.

### Docker Image
All `Dockerfiles` are located in the `docker/` directory.
```sh
# pull project
git clone --recurse-submodules https://github.com/lona-cn/vision-simple.git
cd vision-simple
git lfs pull
# Build the project
docker build --platform linux/amd64 -t vision-simple:ci -f docker/Dockerfile.debian-bookworm-x86_64-cpu .
# Check HTTP model discovery, Docker HEALTHCHECK, graceful shutdown, and cleanup
python3 scripts/test_docker_smoke.py --image vision-simple:ci
# CPU inference by default; bind locally because the service has no authentication
docker run -it --rm -p 127.0.0.1:11451:11451 --name vs vision-simple:ci
```

#### Docker Hub Publication

`.github/workflows/docker.yml` currently validates and publishes only the `linux/amd64` CPU image:

- Configure repository Actions secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` with write access to the destination repository. Optional Actions variable `DOCKERHUB_IMAGE` accepts `namespace/repository`, defaulting to `lonacn/vision_simple`; other registry addresses are rejected.
- Pushing a `vX.Y.Z` or `vX.Y.Z-prerelease` tag triggers publication. Manual runs default to build-and-test only; explicitly selecting `publish=true` requires a valid version tag, not a branch. SemVer `+build` suffixes are not accepted.
- Order: recursive checkout and Git LFS → build and load → pin local image ID → HTTP/container health and cleanup → login → push that tested image → verify registry digests. Publication does not perform a second build.
- Tags are `<version>-cpu-x86_64` and `sha-<full commit SHA>-cpu-x86_64`. Stable releases update `latest` last; prereleases do not. `latest` means the last successfully published stable release, not necessarily the highest historical version.
- Build, health, login, push, or digest failures fail the job. A publication success summary is written only after all checks pass. Updating multiple tags is not atomic; a failed run may have pushed some tags.
- Deploy using the summary's `lonacn/vision_simple@sha256:...` reference (or your configured repository), not a mutable tag. Debian/APT and online package inputs remain mutable; a commit tag does not guarantee bit-for-bit reproducibility.

Release-policy boundary tests: `python3 -m unittest discover -s scripts -p test_docker_release.py`. The x86 CPU build enables AVX/AVX2/F16C and requires a compatible host CPU.

#### Other Platforms / Hardware Acceleration

These Dockerfiles do not imply validated multi-platform publication. CPU `amd64` and `arm64` can share a manifest after both pass actual runtime validation; no multi-platform manifest is currently published by this workflow. ARMv7 still copies artifacts from an `arm64` directory and needs correction and validation; RISC-V has not been runtime-tested in this work. CUDA/TensorRT and RKNPU require their respective hardware and separate variant tags; same-architecture manifest entries cannot distinguish execution providers.

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

### YOLOv11 inference with `vision-simple`
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

## License
The copyrights for the YOLO models and PaddleOCR models in this project belong to the original authors.

This project is licensed under the Apache-2.0 license.