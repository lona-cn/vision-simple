# <div align="center">🚀 vision-simple 🚀</div>
english | [简体中文](./README.md)

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
  - [Containers](https://github.com/users/lona-cn/packages/container/package/vision-simple); GHCR publishes multi-platform Linux amd64 and arm64 CPU images.
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

Requires Docker, Git LFS and an x86_64 CPU with AVX/AVX2/F16C. The first build downloads and compiles dependencies. The image includes default YOLO11/PP-OCR configuration and test models, not YOLO26 weights. See [GHCR multi-platform publication](#ghcr-multi-platform-publication) for published-image selection and verification.

In another terminal, run `curl http://127.0.0.1:11451/v0/infer/models` (`curl.exe` on Windows). Successful discovery proves only that configuration is listed, not that weights load or inference succeeds. Use the [inference example](#send-an-inference-request) below with `hd2-fp32` for the default detection model.

See [OpenAPI](doc/openapi/server.yaml) for the full API. Inference and user-job APIs have **no authentication or tenant isolation**; do not expose the backend directly to the Internet. Native defaults bind to `127.0.0.1`; Docker explicitly binds container interfaces and this example publishes only host loopback. Shared-cache administration is disabled by default and requires explicit bearer activation below.

### HTTP v0 errors and batch semantics

For `yolo`, `seg`, `pose` and `obb`, native v0/v1 requests accept optional top-level numeric `confidence` and `nms_iou` in `[0,1]`. Omitted confidence is `0.125`; omitted raw NMS IoU is `0.3` for detection and `0.45` for segmentation/pose/OBB. End-to-end exports validate `nms_iou` but do not apply it: there is no second NMS. Controls apply to each image of this request only. Confidence changes object selection, not mask binarization or keypoint confidence. OCR rejects either explicit field, even at a default value; its existing `0.125` recognition filter suppresses individual tokens, not detection pixels or whole lines.

Boolean, null, string, array, object or finite out-of-range control values are invalid. Semantic validation occurs before model lookup or image decode, including empty batches: native/OpenAI return HTTP 400 `invalid_request` (ordinary JSON even with `stream:true`); MCP returns `isError:true` tool content in the negotiated text/structured format, not a new JSON-RPC error. Transport admission and cancellation/deadline/closed precedence remain unchanged; unrepresentable JSON numbers such as NaN, Infinity or overflow anywhere (including nested/unrelated fields and earlier duplicate keys) retain existing parse-error envelopes: native `invalid_request`, OpenAI `invalid_json`, MCP JSON-RPC `-32700` with `id:null`, never silent dropping or default fallback. Omission preserves existing JSON behavior.

Successful fields are unchanged: YOLO returns `class_names`/`results`; OCR returns `results`. `model` must be a nonempty string and `images` an array of strings. An existing model accepts an empty array. HTTP 200 guarantees one result per input image in the original order; no detections is a successful empty item.

Any image failure fails the entire batch, without partial results. Processing order is request validation, model lookup/loading, cancellation/deadline/closed checks, request-credit admission, ordered header preflight, whole-batch byte reservation, decoding all images, inference, then serialization. Model/configuration failures retain priority over image admission errors.

```json
{"error":{"code":"invalid_image","message":"Image cannot be decoded","image_index":1}}
```

| HTTP | `error.code` | `image_index` |
| --- | --- | --- |
| 400 | `invalid_request`, `unknown_model` | `null` |
| 400 | `invalid_image`, `image_limit_exceeded` | Zero-based image index |
| 500 | `model_load_failed`, `model_config_failed`, `internal_error` | `null` |
| 500 | `inference_failed` | Zero-based image index |
| 503 | `service_overloaded`, `service_unavailable`, `request_cancelled` | `null` |
| 504 | `request_timeout` | `null` |

Clients relying on HTTP 200 with textual errors must migrate to HTTP status and `error.code`; do not branch on `message`. Third-party exception details remain in logs. See the [OpenAPI contract](doc/openapi/server.yaml).

### Model lifecycle and concurrency

- `POST /v0/infer/unload` accepts `{"kind":"yolo","model":"hd2-fp32"}`; `kind` supports `yolo`, `ocr`, `seg`, `pose` and `obb`. An idle model returns `200 {"kind":"yolo","model":"hd2-fp32","unloaded":true}`; active models return `409 model_busy`; absent models return `404 model_not_loaded`. The next inference reloads transparently. Unload and stats cover all five tasks; only the legacy `/v0/infer/models` catalog is restricted to `yolo`/`ocr`.
- `GET /v0/infer/stats?limit=100&offset=0` returns `models`, `total`, `limit`, `offset`, `idle_timeout_ms`, and service-wide `image_budget` (described below); limit is 1–200. Entries sort by `(kind,name)` and contain `kind`, `name`, `active_requests`, `requests`, `failures`, `total_duration_ms`, and `last_used` (Unix milliseconds). Model counters belong to the loaded instance and reset on reload. Errors before acquisition are excluded; duration includes waiting for the model workspace.
- String options in `config/server.yaml`: `infer_idle_timeout_ms: "300000"` and `infer_sweep_interval_ms: "1000"`. Idle time starts at request completion and uses a monotonic clock. Timeout `"0"` disables eviction; sweep interval must be positive.
- Active leases cover decoding, queued inference, postprocessing, and response serialization/send calls. Neither manual nor timer eviction removes active instances. Synchronous C++ `Run` calls serialize per model; HTTP pipeline tasks own workspaces and gate ORT execution per session.
- Unloading releases sessions and model workspaces, but shared ORT arenas/providers may retain allocations: RSS/VRAM need not fall immediately. YOLO `class_name` results still reference model metadata; C++ callers must keep the model alive longer than these views.
- Stats/unload require the administrator policy below. Authorized blocking management uses the separate bounded control lane; health bypasses both lanes.

### Shared-cache administration and deployment

Only `GET /v0/infer/stats` and `POST /v0/infer/unload` are administrator operations. Inference, configured catalogs, health, tracking sessions, subtitle jobs, OpenAI-like HTTP and MCP retain their existing permissions: **no built-in user authentication or tenant isolation**. CORS, browser Origin restrictions, job/session IDs and an OpenAI SDK `api_key` are not authentication. Keep the backend private; public access needs a TLS-terminating authenticated proxy with separate inference and administrator policies.

Administration is disabled by default: omitted or empty string option `http_management_token_env` returns `403 management_disabled`. To enable it, add `http_management_token_env: "VS_MANAGEMENT_TOKEN"` under `options` in the deployed server configuration and externally provision that environment variable to the server process before startup. This is an example name, **not a default**. The name must match `[A-Za-z_][A-Za-z0-9_]*` and be at most 128 bytes; its value must be 1–4096 visible ASCII bytes with no whitespace/control characters. Use a high-entropy secret (recommend at least 32 random bytes encoded as visible ASCII). Do not put the value in YAML, image layers, commands, URLs, cookies or logs; the server does not generate a credential. An invalid name or missing/empty/unsafe configured value fails startup, never falls back to unauthenticated access. The secret is loaded once; rotate it by changing the externally supplied value and restarting.

Send `Authorization: Bearer <secret>`; header names and the Bearer scheme are case-insensitive, secret bytes are case-sensitive. Missing/wrong credentials return `401 management_unauthorized` with `WWW-Authenticate: Bearer realm="vision-simple-management"`. A correct credential is required even on loopback or behind a proxy. Next, Host must name `localhost`, `127.0.0.1`, `[::1]` or the explicitly configured nonwildcard bind host, with the backend listener port. A wildcard bind does not trust arbitrary authorities. An absent Origin permits CLI clients; a supplied Origin must be a trusted HTTP(S) backend authority, without a path/query/fragment/userinfo. Empty, `null` or malformed Origin and untrusted Host return `403 management_forbidden`. Forwarded/X-* headers are ignored; neither query strings nor cookies supply credentials.

The guard runs at header completion, before receiving the body, sending `100 Continue`, queue admission, JSON parsing or touching model/cache state. Rejection therefore precedes control-lane saturation and cannot reveal loaded/busy state. Unload requires exactly `Content-Type: application/json`, at most 64 KiB, and accepts only `Expect: 100-continue` after authorization; stats rejects a body. Incomplete rejected uploads are discarded or closed according to the HTTP parser's rejection rules. Management paths bypass permissive CORS and reject OPTIONS with 403; ordinary inference CORS is unchanged. Authorized requests retain pagination 400, unload 404/409/200, and bounded control overload `503 service_overloaded` with `Retry-After: 1`. Management errors use `error.{code,message,image_index}` with null `image_index` and do not echo credentials.
The native JSON/early-denial/no-Continue/no-permissive-CORS contract applies only when the management state handler is matched. Malformed HTTP framing may fail in libhv before a callback; malformed Host authority can instead rewrite routing to an unmatched generic route. In the observed slash-suffixed Host case, GET returned generic 404 HTML with permissive CORS/keepalive, and POST with Expect emitted generic 100 Continue before a final generic HTTP error after the body. These library responses are not management authorization or Stats/Unload execution; no model/cache administration occurs.

Native configuration binds `127.0.0.1:11451`. All Dockerfiles override only the staged listener host to `0.0.0.0` after copying the target configuration; management stays disabled, with no image credential. Publish only host loopback (`-p 127.0.0.1:11451:11451`) or a private network. A published port different from 11451 does not change the backend authority port.

For an nginx inference proxy, deny both exact administrative paths by default (queries still match these locations):

```nginx
location = /v0/infer/stats { return 403; }
location = /v0/infer/unload { return 403; }
```

Enable forwarding only on an explicitly administrator-authorized TLS/private proxy route, not the general inference route. Replace those deny locations only after configuring external administrator authorization; each authorized location must forward to the private backend and preserve the caller's bearer, for example these directives for a native loopback backend:

```nginx
proxy_pass http://127.0.0.1:11451;
proxy_set_header Authorization $http_authorization;
proxy_set_header Host 127.0.0.1:11451;
proxy_set_header Origin http://127.0.0.1:11451;
```

These directives are not a complete TLS/authentication configuration. The proxy must authorize the original client Origin before rewriting it; rewriting it is not authorization. CLI requests may instead preserve an absent Origin. Use the actual private backend address for container proxies while rewriting Host/Origin to an accepted backend authority. Do not inject a shared admin bearer into general inference traffic or rely on forwarded client IP/loopback as authentication. Keep tokens out of proxy access/error logs. MCP forwarding separately retains its existing Host/Origin checks and needs SSE buffering disabled and long connections.

For a fail-closed configuration rollback on this build, remove/empty `http_management_token_env` and restart: administration becomes disabled while inference remains available. Unsetting the variable while retaining its configured name instead fails startup. **Older binaries from before this protection ignore the option and restore unauthenticated administration.** Before a binary rollback, keep hard proxy denies on both exact administrator paths and private-backend isolation, then verify that external administrator requests are still rejected. Leaving the new option in YAML does not protect an old binary.

### HTTP scheduling and health

All endpoints share one listener and four IO loops. Blocking inference, model/cache operations, tracking and subtitle file/control work run outside IO loops. String options are `http_data_workers: "4"`, `http_data_queue_capacity: "4"`, `http_control_workers: "1"`, and `http_control_queue_capacity: "4"`; workers accept 1–32 and queue capacities 1–128. Data and control lanes are independently bounded. Each lane bounds accepted handler residents, including completed work awaiting its IO callback, by workers + queue capacity. Management can itself overload; it is not an unlimited priority channel.

Subtitle video uploads use the data lane so a paused upload cannot monopolize the default single control worker; subtitle metadata management stays on the control lane. Cancelling a paused upload wakes its worker; after buffers and files physically drain, it releases the transport slot and closes the incomplete PUT without requiring client disconnect.

Streaming video uploads may hold their transport handler while the body arrives; this does not acquire inference ImageCredit. Complete-body admission below describes inference/JSON requests.

Sequential HTTP keepalive remains supported. Sending a second pipelined request on a connection with an active asynchronous handler closes that connection safely; use separate connections for concurrent work.

Transport admission happens after the complete body arrives, before business JSON parsing. A full/stopping lane returns the endpoint's `503 service_overloaded` envelope with `Retry-After: 1`. Accepted inference retains model/configuration and empty/error precedence. Transport slots bound handler lifetimes, **not image pixels**: queued handlers do not acquire the shared service ImageCredit or decoded-input bytes. The service alone admits v0/v1/OpenAI/MCP image work using `infer_pipeline_max_batches` and the decoded-byte limits. Empty valid-model requests bypass image credit, but still need transport admission.

`GET /livez` returns `200 {"status":"alive"}`. `GET /readyz` returns `200 {"status":"ready"}` while accepting work and `503 {"status":"not_ready"}` during draining. Both bypass model loading, cache/inference locks and dispatch queues. Temporary queue saturation does not make readiness false. Neither endpoint guarantees model validity, loaded weights or warmup. Docker probes `/livez` with a two-second deadline to avoid restarting a live server merely because inference is busy.

HTTP inference deadlines start when the complete body is submitted; queue waiting, model loading and decoding count. Disconnect requests cooperative cancellation; already-running native calls and owned inputs must physically drain before image credit is refunded. Shutdown first marks not-ready/rejects admission, cancels queued/active handlers, stops adapters, drains workers and IO completions, then stops the listener.

Run `python scripts/test_http_dispatch.py --server <executable> --project-root .` with real PP-OCR fixtures. It prints raw idle/load RTT samples and P50/P95/P99, stages four observed image admissions, and checks transport overload, queue timeout, control traffic, shared MCP budget, slow clients and shutdown. No RSS or machine-specific millisecond target is asserted; health uses the consumer's Docker two-second deadline.


### Explicit model preflight, warmup and timing

For operators checking a selected model before deployment, the server executable has a local diagnostic mode. Run it from the same working directory as the HTTP service, with `config/server.yaml` and `config/models.yaml` in place and the configured model files installed. There is no arbitrary configuration-path option. The shipped catalog includes `yolo:hd2-fp32`, `yolo:hd2-fp16` and `ocr:ppocr-v4`; configuration is not proof that those files exist or work. Replace `fixture.jpg` below with your actual representative image.

```sh
./vision_simple-server --help
./vision_simple-server --diagnose preflight --model yolo:hd2-fp32 --model ocr:ppocr-v4
./vision_simple-server --diagnose warmup --model yolo:hd2-fp32 --image fixture.jpg --timeout-ms 60000
```

On Windows use `vision_simple-server.exe` instead. No arguments preserve ordinary HTTP startup and lazy loading; `--help` exits successfully before configuration, logging or listener startup. Diagnostic mode starts no listener and needs neither the HTTP management secret nor logger configuration. Unknown/malformed nonempty arguments exit 2 rather than starting the server. The CLI does not change HTTP routes, OpenAPI, base configuration or health semantics: health never automatically loads every model.

- Select 1–16 distinct `--model task:name` entries; tasks are `yolo`, `ocr`, `seg`, `pose` and `obb`. Nothing loads the full catalog by default. Repeated selections are rejected.
- `preflight` accepts no images and makes one empty measured service call per load attempt: it validates required file roles and actually initializes the selected session, but does not run a frame. `warmup` requires 1–`min(infer_max_batch_images,128)` repeated `--image` arguments. All images form the same ordered batch for each selected model. Total raw fixture bytes are capped at 48 MiB (50,331,648), and total base64 bytes at 64 MiB (67,108,864). Pixel, decoded-byte, request-credit and pipeline limits still apply through the existing service; diagnostics do not bypass admission, scheduling or leases.
- A warmup selection first executes the full batch on a fresh service cache, with **no empty preload**, then repeats it once on that service. The successful cold response holds its model lease through the warm call, preventing idle eviction between the pair even with a tiny idle timeout. Both responses are marked successful and released before explicit unload; selections run sequentially, so a cache limit of one remains usable. A failed cold pass has no fabricated warm pass. A loaded model is unloaded before advancing; unload failure is a failure, not a successful report.
- Each attempted call gets its own `--timeout-ms` (1–300000), or the configured inference timeout if omitted. Expiry is cooperative: native execution must drain and may overrun the deadline. Preflight costs one session load per selection; successful warmup costs one load and two complete fixture batches, including every OCR detection/recognition loop. There is no background keep-warm job or lasting cache promise. Repeating the command starts a new process/service cache; “cold” does not claim cold OS file caches, runtime/device caches or physically cold hardware.

The JSON report has `schema_version: 1`, `validity: "this_invocation_only"` and `http_readiness: "not_assessed"`. Selected models remain in selection order. `configured` is catalog membership; `loadable` is null until loading is attempted, then reflects actual load completion/cache reuse. `smoke_tested` requires successful fixture inference. States distinguish `not_configured`, `missing_files`, `load_failed`, `loadable`, `smoke_failed` and `smoke_tested`: existing but corrupt weights fail loading, while an input-dependent failure after a successful load fails smoke. Missing files list only role names (`model`, or OCR `det`/`rec`/`dictionary`). Passes contain normalized error codes and optional image indices, cache-hit status, batch size and per-frame object/line counts—not detections, masks, OCR text, pixels or base64. Reports and diagnostic errors do not echo physical paths, secrets, environment/configuration option maps or native exception messages.

Exit 0 means every selected model reached the requested state (`loadable` for preflight, `smoke_tested` for warmup), all attempted passes and cleanup succeeded, and the report was written and flushed to stdout successfully. A preflight timeout after successful session loading can still report state `loadable` and `loadable: true`, with a failed timeout pass and exit 1; loadability is not operational success. Exit 1 covers configuration/context/fixture, model, smoke, resource, timeout, unload or report write/flush failures; exit 2 is argument misuse before configuration. `--help` also returns 0 only after its output is successfully written and flushed. Output failures use a static stderr diagnostic, without echoing native errors or supplied values. A zero exit is not HTTP readiness, validity of unselected models or validity of other inputs. Fatal reports use static messages and codes `configuration_failed`, `context_failed`, `fixture_failed`, `diagnostic_failed` or `invalid_arguments`.

**Read capabilities in layers.** The report exposes framework/runtime version, actually compiled EP append support, the runtime's available-provider list, requested EP/device ID, context creation and `cpu_fallback_allowed`. Public C++ `InferContext::Capabilities()` queries those runtime facts without loading models. Context admission, compiled support, available providers and selected-model smoke success are different claims. CPU fallback is allowed; a provider request or successful smoke does not prove every operator ran on that GPU/device. No report claims hardware placement.

**Read elapsed timings, not kernel benchmarks.** Each cold/warm pass records request `wall_ns`; service `model_acquire`, `model_load`, `input_prepare` (base64/header preparation) and actual image `decode`; and pipeline `wall_ns`, resident `capacity_wait_ns`, task `setup_ns`, plus `preprocess`, `inference` and `postprocess`. Thus the five image-processing lanes are decode, preprocess, queue, inference and postprocess; each pipeline stage separates `queue_ns` (enqueue to Advance start) from `execution_ns`. Stage records include `calls`/`completed_calls`; pipeline includes `input_frames`/`completed_frames` and `complete`. Service-only stages use `elapsed_ns`. The four service stages are null when `calls` is zero; the whole pipeline is null only when not entered. Once entered, each pipeline stage remains an exact four-field record (`execution_ns`, `queue_ns`, `calls`, `completed_calls`), even when never executed. Stage `calls: 0` means execution was unreached, not measured instantaneous inference; `queue_ns` may still contain partial waiting for skipped work. Reached failures preserve partial work/wait counters after native drain. A genuine zero elapsed value is possible at clock resolution. OCR records every stage visit, including crop-recognition loops. Frame/stage sums overlap across workers and do **not** equal batch wall time. Inference elapsed time includes session-gate waiting, binding and output work; it is not pure ORT/GPU kernel time. Request wall includes result packing; packing is not a separate stage.

**Opt-in API decision.** Three designs were considered: an output-profile pointer added to controls, observer callbacks, and an owning measured result. `InferPipeline::RunMeasured` and private service `RunMeasured` use the third: a result value owning the ordinary result and fixed timing records. It avoids changing `PipelineControl`/`ServiceControl` layouts, output-pointer ownership and observer lifetime/callback concerns. Ordinary `Run` behavior stays unchanged; with measurement off, instrumentation adds no clock reads, heap allocations or result copies beyond the nullable-profile branch. New measured-API users must rebuild/relink against the updated library; there is no replaced old API needing a compatibility shim. Timing is an observer with a cost: compare measured/unmeasured raw samples and medians on representative fixtures before interpreting small differences. No universal overhead percentage or hardware-independent speed threshold is promised.

**Observed observer cost.** A Linux x86_64 CPU container on WSL2 kernel 6.6.87.2 used GCC 16.2.0 (`O3`, fast-math), ONNX Runtime 1.22.0, ONNXRUNTIME/CPU device 0 and the 384-byte `yolo26_detect_threshold_raw.onnx` kV26 raw-detection fixture (FP32 input `[1,3,64,64]`). The batch was one 32×32 all-black `CV_8UC3` BGR image, with confidence 0.5/NMS IoU 0.3; the service reused its PNG/base64 encoding prepared once. The same live context/model/cache was retained, with the first successful response lease held and idle timeout zero. Setup, loading and four warmup pairs were outside timing. Each API then ran 40 pairs, ordinary first on even pairs and measured first on odd pairs. External steady-clock timing covered only the call, excluding parity checks, `Succeed` and destruction. Medians were the floor of the mean of sorted samples 20 and 21. Ordinary/measured medians were 204.208/180.008 µs for the pipeline (204,208/180,008 ns; difference −24.200 µs), and 176.263/216.701 µs for the service (176,263/216,701 ns; difference +40.438 µs). All 80 pairs preserved exact class/confidence-bit/box parity and service class names; service measured calls reported warm cache hits and complete timing. The negative pipeline difference is scheduler noise, not a speedup claim. These fixture-specific observations are not a general overhead estimate, hardware-placement proof or performance guarantee; three-worker summed stage times still do not equal batch wall time.

### Private OCR warmup image export (explicit opt-in)

Export persists potentially sensitive input pixels: only use explicitly approved images in a trusted current working directory. Without `--debug-dir`, ordinary CLI/HTTP behavior persists no debug images and adds no debug allocations or filesystem probes.

```sh
./vision_simple-server --diagnose warmup --model ocr:ppocr-v4 --image fixture.jpg --debug-dir ocr-study-001 --debug-max-bytes 67108864 --debug-max-files 64
```

Windows uses `.\vision_simple-server.exe`. Export requires warmup, **exactly one OCR selection** and 1–16 images; existing batch/input budgets still apply. `--debug-max-bytes` is 1–67108864/default 67108864; `--debug-max-files` is 1–64/default 64. Both require `--debug-dir`; inclusive caps count the manifest, all files and actual encoded disk bytes, and cannot exceed the hard limits.

NAME is a portable ASCII basename of 1–64 characters, starting alphanumeric, remaining characters alphanumeric/underscore/hyphen. Reject dots, separators, absolute/drive paths and case-insensitive Windows reserved CON/PRN/AUX/NUL/COM1–9/LPT1–9 on ALL platforms. The direct CWD child must not exist as file/directory/symlink/junction: no overwrite/reuse or fallback path. POSIX requires CWD owned by current UID without group/world write; child directory mode 0700 / file mode 0600 use pinned no-follow relative exclusive creation. Windows uses a protected current-user-only DACL, pinned relative no-reparse handles and exclusive creation.

The existing successful warm payload supplies boxes; export makes **no second inference/detection**. Reuse encoded input, decode/encode at most one frame at a time, write original PNG then numeric-index box overlay on the SAME Mat. No detector masks or recognition crops are promised. Files `input-NNN.png`, `boxes-NNN.png` start at `000`, plus `manifest.json` (2*frames+1 files). Manifest fields: `schema_version:1`, effective `ocr_detection`, `recognition_confidence:0.125`, and `frames` with `index,width,height,box_count,boxes`; each box has `index,bbox:[x,y,width,height],confidence`. Report/manifest contain no recognized text/path/token/fixture base64; PNGs themselves still contain sensitive approved input pixels.

Stdout adds only `debug:{files,bytes,retained,error}`, error being a normalized code or null, without directory name/native exception. Retain only after desired model state, released leases, successful unload AND checked stdout write/flush. Model/decode/encode/IO/quota/exception/stdout failure exits 1 and rolls back only created known files and owned new root, never unrelated recursive remove_all. Argument misuse exits 2. Every write/close is checked; no partial export is intentionally retained on failure. A failed stdout commit may leave no usable report; trust exit status, not partial output.

After exit 0 and `retained:true`, inspect the export; successful exports need manual cleanup. Confirm it is still your unchanged, exact newly created directory, then for the name above use:

```sh
rm -r -- './ocr-study-001'
```

```powershell
Remove-Item -LiteralPath '.\ocr-study-001' -Recurse
```

Never substitute a parent directory, wildcard or unrelated existing folder.

### Bounded inference pipeline

After every image has decoded successfully, HTTP v0 uses separate preprocessing, ORT, and postprocessing workers. OCR cycles through detection and crop-recognition minibatch dependencies. Outcomes aggregate by input index, not completion order: any failure rejects the entire batch and reports its lowest failing index.

Server options: `infer_pipeline_capacity: "4"` (resident frame tasks, 1–64), `infer_pipeline_max_batches: "4"` (admitted batches, 1–64), `infer_max_batch_images: "128"` (images per batch, 1–4096), and `infer_timeout_ms: "60000"` (default inference deadline, 1–300000 ms). These are not decoded pixel/byte limits.

Decoded-input options (positive integer strings): `infer_max_image_pixels: "16777216"`, `infer_max_batch_decoded_bytes: "67108864"`, and `infer_max_inflight_decoded_bytes: "268435456"`. They apply to the shared v0/v1/OpenAI/MCP inference service, not the independent tracking service. Preflight strictly decodes base64 once and reads image headers without `imdecode`: PNG, JPEG, BMP, P1–P7, PF/Pf, Radiance HDR and Sun Raster are supported by this build; WebP requires codec-enabled builds. TIFF/JP2/EXR/AVIF are not enabled. Dimensions must be positive and at most `INT_MAX`, with checked pixel/byte arithmetic. Default OpenCV color decoding honors EXIF orientation; swapping width and height preserves the pixel charge. Decoded images are checked against header pixel counts and `CV_8UC3`.
This format list describes header-preflight coverage, not guaranteed acceptance of every codec payload. Decode failure or non-`CV_8UC3` output still returns `invalid_image`. In particular, OpenCV 4.10 decodes grayscale PFM (`Pf`) as `CV_8UC1` even with the color flag, so inference rejects it rather than adding format-specific normalization.

The estimate is `width * height * 3` for BGR input, not a bound on RSS, compressed request bytes, codec scratch space, model arenas, workspaces or response masks. The single-image limit or first cumulative batch overflow returns `400 image_limit_exceeded` with that input index, before any image decode; invalid base64/header returns `400 invalid_image` at the first invalid input. Request credit spans preflight through physical inference completion and is capped by `infer_pipeline_max_batches`, not resident-task capacity. Exhausted credit rejects before base64/codec work; exhausted global bytes reject after preflight but before decoding, both with `503 service_overloaded`.

Stats include service-wide `image_budget`: `in_use_bytes`, `peak_bytes`, `active_requests`, `decode_calls` (real `imdecode` starts only), and `rejected_requests` (budget admission refusals). Counters do not reset on model unload. Inputs and drained tasks are destroyed before refunds; completed responses hold model leases, not pixels/quota. Native HTTP and MCP disconnect request cooperative cancellation, but cannot interrupt ORT or refund before physical work drains.
The shared service and native v0/v1 inference support valid-model empty batches with empty results and control checks: no image credit, bytes or decode calls, even under image-budget saturation; transport admission still applies. MCP tools retain images minItems=1 and reject empty arrays as invalid_request. OpenAI chat also requires actual images rather than exposing native empty-batch semantics.

An optional integer `timeout_ms` (1–300000) overrides the deadline. HTTP time starts at complete-body dispatch submission and includes queue wait, loading and decoding, not serialization/network delivery or hard native preemption. Expiry returns `504 request_timeout`; admission exhaustion returns `503 service_overloaded` with `Retry-After: 1`. Control errors have null `image_index`. Cancellation precedes timeout, then close. With service credit available, ordered image validation precedes global-byte admission; with no credit, service overload precedes invalid images.

C++ YOLO/task callers create `InferPipeline`, then call `Run(model, images, YOLOInferenceOptions{.confidence = 0.1f}, PipelineControl{stop_token, deadline})`; OCR retains `Run(model, images, float confidence, control)`. Options are captured by value for the batch, not stored in scheduling `PipelineOptions` or model/context/cache state. `Close()` rejects/cancels unfinished work; join callers before destruction. Cancellation precedes timeout, close and ordinary inference errors. `Run` drains native stages and input destruction before returning. Models and pixels must remain valid and unmodified throughout; HTTP/MCP disconnect requests cooperative cancellation, never hard native preemption.

Synchronous `InferYOLO`, `InferYOLOTask` and `InferOCR::Run` share stage algorithms/session execution gates with the pipeline. YOLO calls now take `YOLOInferenceOptions`; OCR retains its scalar confidence signature. Tasks own independent workspaces; each model retains at most two idle pipeline workspaces. Unsupported custom backends return an explicit error rather than wrapping synchronous inference as a fake pipeline.

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

### OCR model-construction options

Optional `ocr_detection` belongs to a canonical `models` OCR entry or legacy `ocr` entry; it survives canonical/legacy projection, import/export, equality and configuration copies. It is not a request control. Add the following sibling of `files` to the canonical OCR example above (or to a legacy OCR entry):

```yaml
    ocr_detection:
      kernel_size: 1
      dilation_iterations: "1"
      min_box_area: 16
```

| Field | Inclusive range | Default | Meaning |
| --- | --- | --- | --- |
| `kernel_size` | 1–32 | 2 | Square MORPH_RECT dilation kernel |
| `dilation_iterations` | 0–8 | 3 | Passes; zero uses converted gray mask without dilation |
| `min_box_area` | 0–1048576 | 64 | STRICT pre-unclip `boundingRect.area() > min_box_area`, not contour area |

Omitted/null options inherit defaults; `ocr_detection: {}` explicitly selects defaults. Missing individual fields default. Bare/quoted fully consumed decimal integers are accepted. Unknown leaf keys, booleans, floats, nested values and out-of-range integers fail. An engaged leaf, even `{}`, on a non-OCR model fails; null means omitted. Outer YAML unknown-key and duplicate-key last-value policy is unchanged (duplicate model declarations still fail).

The lightweight public common header `OCRDetectionOptions.h` defines the aggregate `OCRDetectionOptions{kernel_size,dilation_iterations,min_box_area}` with constexpr validation/equality. Every file, byte-span and arithmetic-span template `InferOCR::Create` overload appends `OCRDetectionOptions detection_options = {}` **after** `device_id = 0`. Factories validate before IO/session creation and snapshot the value into immutable model-owned options/kernel; caller mutation cannot change it. Rebuild/relink SDK consumers; no old-symbol shims or setters.

Defaults stay 2/3/64 with the same anchor/border and three physical dilation passes. Kernel size 1 is mathematical identity and avoids unnecessary allocations/dilation. CV_8UC1 gray conversion adds no threshold/scaling. CTC/SAR, recognition batches/confidence, unclip 1.5, IoU 0.3, contour traversal/version branch, coordinate mapping/filter order, `Run(image,float)`, pipeline/measured calls and control layouts remain unchanged. Constructor values plus CLI overlay were selected instead of a callback-lifetime/debug-runtime port. HTTP/OpenAI/MCP inputs, per-request controls, HTTP model descriptors and OpenAPI are unchanged: no per-request morphology, setter, debug or detector HTTP API.

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
- `POST /v1/infer/{task}` accepts `{"model":"raw configured name","images":["raw base64"],"timeout_ms":60000}` for all five task IDs. It shares ordered, whole-batch failure semantics and pipeline limits; native v1 caps request bodies at 64 MiB. An unconfigured model for the selected task returns HTTP 404 `unknown_model` with `image_index:null`, including for empty batches. Legacy v0 and OpenAI-like chat retain HTTP 400 `unknown_model`; MCP reports a tool result with `isError:true` (not a JSON-RPC error). Existing v0 inference responses remain unchanged.
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
    "confidence": 0.1,
    "nms_iou": 0.3,
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

Postprocessing keeps this library's contract: YOLO11/YOLO26 raw detection uses strict `score > confidence`; end-to-end detection and segmentation/pose/OBB use inclusive `score >= confidence`. Raw detection runs class-aware NMS on unclipped, unrounded floating-point model-space boxes (omitted IoU `0.3`), then maps, clips and rounds survivors to the original image. Degenerate boxes and empty output boxes are discarded. With fixed model output, NMS candidate selection does not depend on original-image size; integer `bbox:[x,y,width,height]` is unchanged. Other raw tasks default to NMS IoU `0.45`. OBB uses polygon IoU, **not Ultralytics' probabilistic IoU**. Explicit `nms_iou` overrides raw suppression only; YOLO10/YOLO26 end-to-end (NMS-free) never undergo another suppression pass. Letterbox remains black; keypoints retain out-of-image coordinates, and masks interpolate logits before binarization/cropping. These differences preclude assuming identical Ultralytics results.

#### Verified compatibility and limitations

The default OpenCV dependency is 4.10.0. OCR selects LinkRuns from 4.10 onward; the older branch uses the same dilated mask, `Vec4i` hierarchy and two-level contour retrieval, without changing morphology defaults. Both contour APIs were exercised against the same rectangle fixture on 4.10.0; this does not certify an actual 4.9 build or future-major runtime compatibility.

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

Tracking `Step` consumes caller-supplied detections independently; there is no fused detector/tracker route. When manually chaining inference into tracking, choose detector `confidence` at or below the tracker's `low_threshold` (strict raw detection excludes the exact boundary; choose below it to retain equality) and forward the resulting low-score candidates. For example, `confidence:0.1` retains candidates between `0.1` and the default detector `0.125`. This is a client strategy, not a claim that all tracking inputs are truncated.

- `timestamp` is finite, nonnegative seconds; `frame_index` is an integer in 0–9007199254740991. Both must strictly increase within a session. Elapsed seconds control motion prediction; index gaps count toward expiration. Rejected frames do not advance state. Reset starts a fresh sequence, including IDs and timing. Step returns `frame_index`, `timestamp`, and `tracks:[{track_id,class_id,confidence,bbox}]`; only confirmed tracks observed this frame are emitted, not lost predictions. Status contains nullable `last_frame_index`/`last_timestamp` and `active_tracks`/`lost_tracks`.
- Algorithms: `"bytetrack"` or `"botsort"`. Defaults: `high_threshold:0.5`, `low_threshold:0.1`, `new_track_threshold:0.6`, `match_threshold:0.8`, `max_lost_frames:30`, `min_hits:2`, `max_tracks:256`, `max_detections:256`, `camera_motion:true`, `appearance:false`, `proximity_threshold:0.5`, `appearance_threshold:0.25`. Thresholds are in [0,1], with low < high ≤ new; matching thresholds are maximum costs, not minimum IoU. `min_hits` is 1–10000, `max_lost_frames` 0–10000, and both capacity options 1–256.
- BoT-SORT camera motion requires an `image` containing raw base64 PNG/JPEG on every frame, same dimensions throughout the sequence and at least 8×8 pixels; disable it with `camera_motion:false` when supplying detections only. Images are bounded to 16,777,216 decoded pixels before decoding. Boxes are finite floating-point xywh with positive dimensions; confidence is in [0,1] and class IDs are nonnegative.
- BoT-SORT `appearance:true` requires a finite, nonzero `embedding` array on every detection, at most 512 elements and a consistent dimension per session. Embeddings come from the caller: no ReID model or weights are bundled, and tracking does not run an embedding model.
- Limits: 32 sessions, 256 detections/frame and tracks/session, 4 MiB request body. Sessions expire after 300 seconds since creation or the last successful step/reset; expiration is swept on service access, and status/list do not renew it. Concurrent access to a busy session returns 409 rather than queueing; separate sessions do not share track identities.
- Errors use `error.{code,message,image_index}` with null `image_index`: 400 `invalid_request`/`invalid_image`, 404 `tracking_session_not_found`, 409 `tracking_session_busy`/`frame_out_of_order`, 503 `tracking_capacity`/`service_unavailable` (with `Retry-After: 1`), or 500 `tracking_failed`. Unknown JSON fields are rejected. POST requires exactly `Content-Type: application/json`; GET/DELETE cannot have bodies. Oversize bodies return 413; unsupported `Expect` returns 417.
- Tracking business requests reject nonempty browser `Origin` with 403 and use `Cache-Control: no-store`; ordinary OPTIONS preflight still passes through global CORS middleware. Neither Origin/CORS nor session IDs provide authentication, and listing is not tenant-scoped. Use a trusted network or authenticated proxy.

### Asynchronous video subtitles

This extracts **visible text with OCR**, not speech. Configure an OCR model and its real detection/recognition weights and dictionary first. The following Linux shell commands use `curl` and GNU `mktemp`/`mv`; replace `JOB_ID` with the `id` returned by create. A Windows PowerShell download example follows.

```sh
# 201 + Location; options are top-level fields, not an "options" object
curl -sS -X POST http://127.0.0.1:11451/v1/subtitle/jobs -H 'Content-Type: application/json' -d '{"model":"ppocr-v4","sample_interval_ms":200,"roi":[0,0.5,1,0.5],"min_confidence":0.5,"stable_samples":2,"gap_samples":2}'
# 202 means accepted for processing, not successful decoding
curl -sS -X PUT http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID/video -H 'Content-Type: application/octet-stream' --data-binary @clip.avi
curl -sS http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID
# Run only after GET reports state == "completed"; empty SRT is valid.
job_url=http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID
result=./clip.srt
# Same-directory temporary file: successful transfer + write + atomic rename before DELETE.
if tmp=$(mktemp "${result}.download.XXXXXX"); then
  if curl --fail --silent --show-error "$job_url/subtitles.srt" --output "$tmp" &&
     test -f "$tmp" && mv -fT -- "$tmp" "$result"; then
    curl --fail --silent --show-error -X DELETE "$job_url"
  else
    rm -f -- "$tmp"
    printf '%s\n' 'Download/save failed: original result and server job retained.' >&2
  fi
fi
# For WebVTT use subtitles.vtt and ./clip.vtt instead; do not delete until all wanted files are saved.
```

Windows PowerShell (Python 3 installed): use `curl.exe`, not the PowerShell `curl` alias. After polling until completed, use this download-and-save block; empty SRT is valid. For WebVTT change both the endpoint and destination extension. Save all wanted formats before deleting the job.

```powershell
$jobUrl = 'http://127.0.0.1:11451/v1/subtitle/jobs/JOB_ID'
$result = [IO.Path]::GetFullPath('clip.srt')
$tmp = Join-Path ([IO.Path]::GetDirectoryName($result)) ([IO.Path]::GetRandomFileName())
try {
    curl.exe --fail --silent --show-error "$jobUrl/subtitles.srt" --output "$tmp"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $tmp -PathType Leaf)) {
        throw 'Download/write failed; server job retained.'
    }
    python -c 'import os,sys; os.replace(sys.argv[1],sys.argv[2])' "$tmp" "$result"
    if ($LASTEXITCODE -ne 0) { throw 'Atomic replacement failed; server job retained.' }
    curl.exe --fail --silent --show-error -X DELETE "$jobUrl"
    if ($LASTEXITCODE -ne 0) { throw 'Local result saved; DELETE failed.' }
} finally {
    if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp }
}
```

- Create accepts only `model`, `sample_interval_ms` (integer 100–5000, default 200), `roi`, `min_confidence` ([0,1], default 0.5), `stable_samples` and `gap_samples` (integers 2–10, both default 2). `model` is the configured raw OCR name, 1–256 bytes. ROI is normalized `[x,y,width,height]`, with positive dimensions entirely inside the frame; default `[0,0.5,1,0.5]` is the bottom half. Full-frame example: `{"model":"ppocr-v4","roi":[0,0,1,1]}`. Unknown fields are rejected.
- Sampling selects the first decoded frame at or after the next interval, using actual presentation timestamps in milliseconds, not `sample_index × interval`. OCR lines below `min_confidence` are removed (the model's own recognition filter still applies), arranged top-to-bottom/left-to-right and whitespace-normalized. There is no fuzzy text matching. Identical normalized text needs `stable_samples` consecutive observations; its cue starts at the first of those observations. A confirmed replacement ends the previous cue at that same timestamp. Transient alternatives are suppressed; fewer than `gap_samples` empty observations can bridge an unchanged cue, while a confirmed empty gap ends it at the first empty timestamp. At EOF, an outstanding gap closes there; otherwise the last cue ends at the decoded stream end. Intervals are nonoverlapping and positive; precision depends on sampling and OCR accuracy.
- Downloads contain validated UTF-8 SRT or WebVTT, with normalized control/blank lines and escaped `&`, `<`, `>` to prevent recognized text from becoming markup or cue syntax. Empty successful extraction is valid: empty SRT or a WebVTT header, not fabricated captions.
- Normal states are `created → uploading → queued → running → completed`; failures become `failed`. Cancellation returns 202 and moves active native processing through `cancelling → cancelled`; it is cooperative, not a hard interruption of a decoder/ORT call. Cancelling an already terminal job leaves its result unchanged. Status returns `id`, `state`, `uploaded_bytes`, `decoded_frames`, `sampled_frames`, `position_ms`, nullable `duration_ms`, `cue_count`, nullable `error_code`, and nullable `expires_at`. Duration may be unknown; counts/position are progress, not a guaranteed percentage, and `cue_count` while running excludes the open cue. `GET /v1/subtitle/jobs?limit=100` lists job objects and nullable `next_cursor`; pass it unchanged as `cursor` (limit 1–100).
- Upload is raw bytes in a separate PUT, not multipart/base64, a server-local path or a remote URL. Only `created` jobs accept upload; a consumed/interrupted/failed upload cannot restart on the same job. Retry by creating a new job and reuploading; create is not idempotent. A successful upload can still fail asynchronously, so poll `state` and inspect `error_code` (for example `upload_interrupted`, `unsupported_video`, `invalid_timestamps`, `ocr_failed`, `subtitle_limit`), not error-message wording. Partial failed results cannot be downloaded.
- One worker serves at most **8 jobs**, including pending uploads and retained terminal results. Limits: 64 MiB/video, 1800 seconds, 1,000,000 decoded frames, 16,777,216 pixels/frame, 4096 OCR lines/observation, 4096 bytes/line and combined observation, 10,000 cues and 2 MiB accumulated cue text. JSON control bodies are limited to 64 KiB. Input files live in a private server-owned temporary directory; completion/failure/cancellation/deletion and orderly shutdown attempt cleanup. A process crash is not an orderly cleanup guarantee.
- Processing is FIFO by **successful upload completion and transition to queued**, not create time or random job ID. It guarantees dispatch order, not start/completion deadlines. List pagination remains lexical by job ID, independently of FIFO.
- Every job object includes nullable `expires_at`: UTC `YYYY-MM-DDTHH:MM:SS.mmmZ`. Created/uploading jobs become eligible for expiry **60 seconds after the last accepted activity** (creation, successful upload start, or successfully written nonempty chunk); empty chunks do not renew it. Completed/failed/cancelled results become eligible **300 seconds after actual terminal acknowledgement**, not the cancellation request. Queued/running/cancelling have `expires_at: null` and do not expire. GET, List, Download and idempotent terminal Cancel never renew retention.
- Expiry is cleanup eligibility, not exact-time removal: independent housekeeping checks every real second; create, GET, List, upload start and Download also sweep. Append, Finish, Cancel and Delete do not add a sweep, so an overdue but unswept operation may win the lock first. A row releases capacity only after input-file unlink succeeds; cleanup failures can keep it visible and consuming capacity beyond `expires_at`.
- Enforcement uses only the monotonic clock. The UTC metadata projection captures wall and monotonic time together at the activity/terminal transition and is not rebased on reads. A wall-clock jump can make the published date an estimate without changing retention. The private service clock is a typed source callback for native regression tests, not a per-job field, HTTP option or YAML knob.
- A Download accepted under the service lock retains shared ownership and may finish after concurrent expiry/deletion; new requests may already return 404. Save each wanted download successfully to a same-directory temporary file and atomically replace the destination before DELETE, as above. A failed download/write/replacement leaves the original local result and server job intact (subject to normal expiry).
- Container admission and actual decoding use the same build-specific capability policy:

  | Container | Windows native build | Linux build | Actual decoding after admission |
  | --- | --- | --- | --- |
  | AVI (`RIFF` / `AVI `) | Admitted | Admitted | Portable bounded MJPEG reader on both platforms: one MJPG/mjpg video stream starting at zero, valid rate/scale and complete declared frames, including OpenDML AVI/AVIX. Other AVI codecs fail asynchronously with `unsupported_video`; there is no native AVI fallback. |
  | MP4-family (leading `ftyp` box) | Admitted only with the compiled Media Foundation reader | HTTP 415 `invalid_video` before queueing | Windows native codecs determine decode availability; codec/container failures can still occur asynchronously. |
  | MKV / ASF | HTTP 415 `invalid_video` before queueing | HTTP 415 `invalid_video` before queueing | Disabled in both upload admission and the reader. |

- Admission only sniffs a bounded header against the received body size: AVI requires outer RIFF size ≥4 and ≤body size−8; MP4 requires first `ftyp` box size ≥12 and ≤body size. It is not codec inspection or full-file validation. A body-bounded fake AVI header (or fake MP4 header on Windows) may be admitted and then fail asynchronously; HTTP 202 means accepted, not successfully decoded. Malformed headers or containers unavailable in the build return HTTP 415 `invalid_video` before target-file rename/queueing: the job becomes `failed` with `error_code` set to `invalid_container` for malformed/unknown headers or `unsupported_video` for a recognized but unavailable container, zero `decoded_frames`/`sampled_frames`, and source-file cleanup. `uploaded_bytes` records bytes received, not retained files; cleanup failures retain capacity as described above. Failed-job downloads return 409 `subtitle_not_ready`, and reupload returns 409 `subtitle_job_busy`; create a new job to retry. Arbitrary codecs, playlists, image sequences and URL fetching are unsupported.
- MJPEG AVI timestamps follow stream rate/scale; Media Foundation uses actual sample timestamps and positive sample durations, rounding starts down and ends up to milliseconds. `duration_ms` on completion is the actual video end, not an estimated frame count or a longer audio/container duration; native duration can remain null until completion.
- Errors use `error.{code,message,image_index}` with null `image_index`: 400 `invalid_request`; 404 `model_not_found`/`subtitle_job_not_found`; 409 `subtitle_job_busy`/`subtitle_not_ready`; 413 `payload_too_large`; 415 `unsupported_media_type`/`invalid_video`; 503 `subtitle_capacity`/`service_unavailable` (`Retry-After: 1`); 500 `subtitle_failed`. Create/cancel require exactly `application/json`, upload exactly `application/octet-stream`; bodyless operations reject bodies, and unsupported `Expect` returns 417. DELETE returns 204 only for a created or terminal job; cancel and poll other states first.
- This API has **no authentication or tenant isolation**. Subtitle business requests reject nonempty browser `Origin` with 403 and use `Cache-Control: no-store`; ordinary OPTIONS still passes through global CORS middleware. Job IDs are not credentials. Use a trusted network or authenticated proxy.

Real-video regression (run from the repository root with a built server):

```sh
python scripts/test_subtitle_regression.py --server <server-executable> --project-root . --ffmpeg <ffmpeg-executable> --font <font.ttf>
```

Requires real `ppocr_det.onnx`, `ppocr_rec.onnx` and `ppocr_keys_v1.txt` in `app/assets/test`, plus FFmpeg with drawtext/MJPEG/libx264/AAC and a usable TrueType font. `--ffmpeg`/`--font` may be omitted when FFmpeg is on PATH and the platform's default Arial/DejaVuSans font is available. FFmpeg generates test fixtures; it is not an application decoder dependency. The regression exercises real decoding/OCR, exact HELLO/WORLD intervals with a transient NOISE frame suppressed, SRT/WebVTT agreement, build-specific container admission and failure cleanup, upload interruption, cancellation, capacity/isolation and temporary-file cleanup; Windows also exercises native variable-frame-rate video with a longer audio track. CI invokes this driver only on the existing native Linux x86_64 CPU and Windows x64 CPU `run_http` rows; cross-built architectures do not gain a runtime-validation claim. It is not a codec-quality or performance benchmark.

The Windows Server CI row conditionally installs the Media Foundation feature needed for its native MP4 scenario, and fails clearly if installation fails or requires a restart. Linux installs FFmpeg and DejaVuSans; Windows installs FFmpeg and uses Arial. These are regression prerequisites, not evidence that a hosted workflow or every Docker architecture has passed.

Issue #55 final full real-server subtitle regressions passed on native Windows CPU (30.59 seconds) and Linux x86_64 CPU (cached SDK, GCC 16.2, ORT 1.22; build plus full regression: 123.61 seconds), not hosted CI or all six production Docker builds. The first Windows run captured a socket reset (10054) of undetermined cause; the final diagnostic run passed without a production-source fix for that reset, so this does not claim the reset was fixed.

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

### Reproducible synthetic OCR geometry study

Approved `hershey-word-v1` freezes eight 640×384 cases: three ordinary images each with six words (scale 0.85/1/1.15, thickness 2), three dense images each with 24 words (0.48/0.55/0.62, thickness 1), and a separate two-image negative cohort (blank; low-contrast gradient/geometric background). Deterministic OpenCV Hershey SIMPLEX/LINE_8 visible words provide 90 tight rendered-ink word GT boxes, independently of detector predictions; ignore policy:none. No external fonts/images, additional downloads or new licensing claims.

Hold constant the repo's real trained `ppocr_det.onnx`, `ppocr_rec.onnx` and `ppocr_keys_v1.txt`, `kPPOCRv4`, CPU device 0, recognition confidence 0.125, explicit `ocr_rec_batch_size=1`. Compare default 2/3/64 to alternative 1/1/16. From repository root with installed assets and normal CPU build dependencies:

```sh
xmake build test_ocr_morphology_dataset
xmake run test_ocr_morphology_dataset --project-root .
```

The study renders in memory, prints JSON metrics and persists no images. Maximum-cardinality one-to-one bipartite matching at IoU≥0.5 counts matched TP, unmatched GT FN and unmatched predictions FP. **All returned boxes count, including empty recognized text** (separately counted). Recall=TP/GT, precision=TP/(TP+FP), FP/image=FP/images; zero denominators:null. Low-IoU expansion and splits/merges can produce FP/FN, not necessarily hallucinated words. These are synthetic word-ink geometry measurements, NOT production OCR accuracy or character-recognition quality.

Observed Linux CPU results distinguish immutable old #61 baseline (successful, 17.59 s) and new default/alternative study (exit 0, 9 s). Independent comparison found all eight new-default outputs EXACTLY equal to old baseline ordered bbox/fullUTF8 bytes/confidence float bits, with identical corpus pixels/annotations and default cohort counts. New study also exercises caller-option mutation/model isolation and synchronous/staged parity for both options.

| Run / cohort | Images | GT | TP | FN | FP | Recall | Precision | FP/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Old #61 baseline 2/3/64 / ordinary | 3 | 18 | 12 | 6 | 6 | 0.666667 | 0.666667 | 2 |
| Old #61 baseline 2/3/64 / dense | 3 | 72 | 0 | 72 | 72 | 0 | 0 | 24 |
| Old #61 baseline 2/3/64 / negative | 2 | 0 | 0 | 0 | 1 | null | 0 | 0.5 |
| New default 2/3/64 / ordinary | 3 | 18 | 12 | 6 | 6 | 0.666667 | 0.666667 | 2 |
| New default 2/3/64 / dense | 3 | 72 | 0 | 72 | 72 | 0 | 0 | 24 |
| New default 2/3/64 / negative | 2 | 0 | 0 | 0 | 1 | null | 0 | 0.5 |
| Alternative 1/1/16 / ordinary | 3 | 18 | 18 | 0 | 0 | 1 | 1 | 0 |
| Alternative 1/1/16 / dense | 3 | 72 | 52 | 20 | 20 | 0.722222 | 0.722222 | 6.66667 |
| Alternative 1/1/16 / negative | 2 | 0 | 0 | 0 | 1 | null | 0 | 0.5 |

Empty-text count is zero in every cohort/variant. Alternative is an experiment, not a changed default, required improvement gate or universal recommendation. Durations are observations, not performance thresholds. Mechanics fixtures for morphology boundaries, CTC/SAR, batches and concurrency are not trained-model accuracy evidence; this study does not certify every test/platform.

### Shared service, OpenAI-like HTTP and MCP SSE

v0, native v1, OpenAI-like and MCP call one `InferenceService`, sharing model loading, leases, stats, idle eviction and pipeline limits. Tracking state belongs to its separate service. Discovery lists configuration, not proof that every model file is loadable.

**OpenAI-like subset**

- `GET /v1/models?limit=100` returns `object:"list"` and `data`. IDs are `<task>:<raw-name>` for `yolo`, `ocr`, `seg`, `pose`, `obb`; `task` identifies the operation. Limit is 1–200. When `has_more` is true, pass the returned `next_cursor` unchanged as query parameter `after`.
- `POST /v1/chat/completions` accepts `model`, `messages`, optional `stream`, `timeout_ms`, `n:1` and `response_format:{"type":"json_object"}` (or `"text"`). User `image_url.url` parts must contain inline base64 PNG/JPEG/WebP/BMP data URLs. Only omitted/`"auto"` detail is supported; remote URLs are never fetched.
- YOLO-family model IDs accept the same top-level `confidence`/`nms_iou` fields and omission defaults; OCR IDs reject them. With an OpenAI Python SDK pass `extra_body={"confidence":0.1,"nms_iou":0.3}` to `chat.completions.create`: the SDK merges these into the top-level HTTP body; a literal wire `extra_body` object is not supported.
- At least one image is required. Images retain message/content order. Text does not alter detection/OCR: these are not language models. Other generation controls, tool calls, JSON Schema output and the Responses API are unsupported; unsupported parameters return 400 rather than silently doing nothing.
- `choices[0].message.content` is the full task result encoded as JSON: YOLO/new tasks include `class_names/results`, OCR includes `results`. Geometry follows the task contracts above; token usage is not fabricated.
- `stream:true` sends `chat.completion.chunk` SSE events (role, complete JSON content, finish reason), then `[DONE]`. This is neither token nor per-image streaming. Headers commit only after whole-batch success; earlier failures remain HTTP JSON errors.
- Errors contain `error.{type,code,message,param,image_index}`. Inference codes match v0; 503 includes `Retry-After: 1`. Request bodies are capped at 64 MiB while receiving (413); serialized inference results are capped at 64 MiB.

**Legacy MCP HTTP+SSE**

Connect to `GET /mcp/sse`, read its `endpoint` event, then POST JSON-RPC 2.0 to that relative URI. Complete `initialize` and `notifications/initialized` before `tools/list` / `tools/call`. Supported versions: `2024-11-05`, `2025-03-26`, `2025-06-18`, `2025-11-25`. Select SSE in clients; this is not Streamable HTTP.

Initialization `params` must include `protocolVersion`, an object `capabilities`, and `clientInfo` containing `name`/`version`. POST HTTP 202 acknowledges receipt only: read JSON-RPC responses from the original SSE connection, not the POST response body.

- `list_models`: optional `limit` (1–200) and opaque `cursor`; returns `data:[{id,kind,name}]` and optional `next_cursor`.
- Generated `infer_yolo` / `infer_ocr` / `infer_seg` / `infer_pose` / `infer_obb`: `{"model":"raw configured name","images":["raw base64"],"timeout_ms":60000}`. Do not pass prefixed catalog IDs or data URLs. Discovery includes input JSON Schemas.
- Only `infer_yolo`, `infer_seg`, `infer_pose` and `infer_obb` advertise/accept optional numeric `confidence` and `nms_iou`; for example add `"confidence":0.1,"nms_iou":0.3` to YOLO arguments. `infer_ocr` advertises neither and rejects explicit detector controls.
- Modern results contain `structuredContent` and equivalent JSON text; legacy versions retain the full structure in text. Execution errors use `isError:true` with stable `error.code`, `image_index` and recovery advice. Protocol failures use JSON-RPC errors and preserve string versus integer IDs.
- `notifications/cancelled` cancels its session's `requestId`; disconnect cancels all that session's work. Native stages drain before cancellation/timeout returns; ORT is not forcibly interrupted and another session's equal ID is unaffected.
- POST header admission preserves this order: trusted Host/Origin (403), JSON media type/UTF-8 charset (415), declared body length (413), stopping transport (503), then existing open session (404), before checking `Expect`. A single `100-continue` token is accepted case-insensitively with outer spaces/tabs and receives `100 Continue` only after those checks. Any other nonempty value delivered by the HTTP parser, including comma-separated/repeated tokens, returns HTTP 417 with the static `text/plain` body `Unsupported expectation`, without `100 Continue` or waiting for the upload body. The current libhv HTTP/1 parser discards leading spaces/tabs: a wire header containing only spaces/tabs becomes empty, like an absent/empty `Expect`, sends no `100 Continue` and follows ordinary body reception. HTTP errors with nonempty parser-delivered `Expect`, and all 413 errors, send `Connection: close`; other rejections retain the existing body consume/discard flow. These transport errors are not native error JSON or SSE JSON-RPC errors; closing the POST connection does not close the separate SSE session. Session lifecycle, execution queues and body/image limits are unchanged.
- TCP close regressions require EOF after a complete headers-only rejection. If the client has already sent its body, EOF or TCP RST is accepted only after validating the complete rejection response and `Connection: close`; resets before complete headers/body, timeouts and extra response bytes still fail.
- Bounds: 32 sessions, 2 execution workers, 16 queued jobs, 8 active tool calls/session, 64 MiB POST body, 8 MiB queued output/socket buffers. Heartbeats occur every15s; sustained buffered writes close after about30s, and sessions with no active work expire after5min without messages (checked on heartbeat ticks). Oversized output closes the session; reconnect with a smaller batch.
- Host accepts loopback names and an explicitly configured nonwildcard bind host with matching port. Supplied Origin must identify a corresponding trusted HTTP(S) authority. MCP bypasses permissive global CORS. Remote access requires an explicit trusted bind address or an authenticated proxy rewriting trusted backend Host/Origin.

**Claude Code project configuration:** the tracked [.mcp.json.example](.mcp.json.example) defines `mcpServers.vision-simple` with `type: "sse"` and loopback URL `http://127.0.0.1:11451/mcp/sse`. First start the actual server using [Build and run](#build-project), then create a local `.mcp.json` from the repository root:

```powershell
# Windows PowerShell: copy only when the destination is absent; never overwrite
if (-not (Test-Path -LiteralPath .mcp.json)) {
    [System.IO.File]::Copy((Join-Path $PWD '.mcp.json.example'), (Join-Path $PWD '.mcp.json'), $false)
}
```

```sh
# Linux (GNU coreutils): leave existing files, directories and symbolic links untouched
if [ ! -e .mcp.json ] && [ ! -L .mcp.json ]; then
    cp -nT .mcp.json.example .mcp.json
fi
```

If `.mcp.json` already exists, manually merge only the example's `vision-simple` entry into its existing `mcpServers`, preserving all other configuration; do not replace the file. If your server uses a port other than 11451, change the local entry's URL port, retaining `/mcp/sse`. Copying configuration does not start the server. Open Claude Code at the repository root and approve the project MCP server when prompted.

OpenAI SDK `api_key` is not authentication here: use a trusted network or external authenticated proxy for every protocol. Disable SSE proxy buffering and allow long connections. Regression commands appear below; multi-step agent evaluations are in `scripts/mcp_evals.xml`.

### C++ migration

- Rebuild the C++ SDK and every consumer: `InferYOLO::Run(image, YOLOInferenceOptions)` and `InferYOLOTask::Run(image, YOLOInferenceOptions)` replace scalar YOLO confidence arguments, with no compatibility overload. `YOLOInferenceOptions` is a public aggregate containing `std::optional<float> confidence` and `std::optional<float> nms_iou`; `{}` preserves omitted defaults. It owns its values, not borrowed option storage. These controls do not change `Create` signatures; `InferOCR::Run(image, float confidence)` is unchanged. Run requires a nonempty two-dimensional `CV_8UC3` image and supports non-contiguous ROIs. Grayscale, BGRA, floating-point images and supplied non-finite/out-of-range `[0,1]` controls return parameter errors.
- YOLO11/YOLO26 raw detection runs class-aware NMS in floating-point model space before clipping/rounding. v10 accepts only end-to-end `[1,N,6]` output; v10/v26 end-to-end never repeat NMS. Confidence threshold semantics, black Letterbox padding and OCR detection normalization are unchanged.
- YOLO26 detection uses `YOLOVersion::kV26`. Both path and memory overloads of `InferYOLOTask::Create` require an explicit version after `task`: use `YOLOVersion::kV11` for existing segmentation/pose/OBB callers and `YOLOVersion::kV26` for YOLO26. The optional `device_id` follows the version.
- PP-OCR CTC file-based Create follows the Paddle dictionary convention: the file excludes blank and the trailing space class; the loader appends space. Map-based callers supply every nonblank class with key `class_id - 1`. SAR follows its separate dictionary contract above.
- `HTTPServer::Run/StartAsync` return `HTTPServerResult<void>`; callers must check failures. Empty/overlong hosts and listen failures are rejected without silently binding wildcard. Native configuration now defaults to `127.0.0.1`; Docker applies an explicit container-only wildcard override.
- Helper consumers must recompile and migrate to one geometry path:

```cpp
LetterboxTransform transform;
cv::Mat& padded = helper.Letterbox(image, target_size, transform);
if (padded.empty()) { /* reject invalid or rounded-zero image dimensions */ }
cv::Rect box = VisionHelper::ScaleCoords(transform, cv::Vec4f{x1, y1, x2, y2});
```

`ScaleCoords` accepts floating-point model-space xyxy, reverses actual per-axis scaling, clips endpoints, then rounds endpoints into integer xywh. Old geometry signatures, `DataConverter` and unimplemented uint8 no-op conversions were removed. `Cvt` supports bidirectional FP32/FP16 conversion.

Inference and user-job APIs have no built-in user authentication; only the two shared-cache administration operations have the opt-in bearer policy above. Hot loading and arbitrary ONNX are unsupported. Shared v0/v1/OpenAI/MCP decoded-input limits and tracking/subtitle bounds are not a blanket production-security guarantee.

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

Public RKNPU support matrix (Linux ARM/ARM64):

| Build condition | `InferContext::Create` context admission | Model session initialization | Real-device inference |
| --- | --- | --- | --- |
| `VISION_SIMPLE_WITH_RKNPU` undefined | `kRKNPU` returns `kParameterError` | RKNPU session path is not entered | Unsupported |
| `--with_rknpu=y`, defining `VISION_SIMPLE_WITH_RKNPU` | Accepts ONNXRuntime + `kRKNPU` | Still requires RKNPU-enabled ORT, DDK/drivers and a compatible model; verify separately | Requires separate target-device acceptance; no hardware-verification claim |

`test_infer_inputs` covers RKNPU factory rejection/admission according to the build macro. It neither creates an RKNPU model session nor runs NPU inference; its retained CPU model inference does not verify RKNPU hardware. Successful context creation does not guarantee successful session initialization or real-device inference.

For ONNX Runtime model creation with DML, CUDA or TensorRT, the public `size_t device_id` must be in `0..INT_MAX` (inclusive). Larger values return `kParameterError` before provider initialization or narrowing, even if that provider was not compiled in. Passing the range check does not prove that the device exists: `0` and `INT_MAX` are representable, but provider support, runtime/drivers and actual device availability are checked separately and can return `kRuntimeError`. CPU continues to ignore the device ID, including `SIZE_MAX`; RKNPU retains its separate device-0-only rule.

`test_infer_inputs` checks these representation boundaries using model creation and compares CPU inference with device `0` and `SIZE_MAX`. Its representable GPU-ID cases allow provider initialization failure and do not claim GPU hardware is present or verified.

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

# RKNPU (Linux ARM/ARM64; target-device acceptance still required)
xmake f --with_rknpu=y -m release
xmake build server
```

### Run Tests

Run these commands from the repository root. If you just launched the server as above, open another terminal at the repository root. Configure a CPU build first and ensure Git LFS model resources have been downloaded.

**CI coverage layers:** execution gates and cross-build artifact checks are distinct. These definitions do not claim that a particular hosted run passed.

- Five native `build_tests` rows (Linux GCC Release, Clang/libc++ Release, GCC ASan+UBSan, Windows MSVC CPU and Windows MSVC DirectML) explicitly build and run **17 C++ executables**. Deterministic regressions cover common/conversion/vision helpers, trace-ID format, YOLO postprocessing, OCR decoding, configuration, tracking, subtitle timelines and image codecs. A separate CPU fixture-backed step runs inference inputs, pipelines, YOLO tasks, OCR batches, OCR morphology and its synthetic geometry study, and shared-service image budgets using real ONNX sessions, including deliberate failure/invalid models.
- Only the two CPU `run_http` rows (Linux GCC Release and Windows MSVC CPU Release) also run `test_subtitle_service`, making **18 C++ executables per CPU row**, plus **nine real-server HTTP drivers**: generic HTTP/startup, management security, bounded dispatch, subtitle media, protocol, model registry, tracking with a real detector and image, YOLO tasks and image budgets. Both rows also run a separate documentation-example driver for full OpenAPI/schema/image validation, exact YOLO/OCR requests, and MCP initialization/tool discovery, making **10 script drivers in total**; that test step uses Python 3.12 and test-only dependencies, adding no server runtime dependency. Subtitle media requires FFmpeg and a usable font; Windows MP4 decoding uses Media Foundation. Missing prerequisites fail, not skip.
- Documentation validation also includes separate static boundary unit tests for invalid schemas and rejected examples. These are not HTTP/real-server integration drivers and do not increase the HTTP driver count.
- ARM64 CPU/RKNPU, ARMv7 and RISC-V64 cross rows retain artifact architecture checks; they do not execute those binaries. The DirectML row does not establish GPU/DirectML inference coverage; these CPU tests do not verify real CUDA, TensorRT or RKNPU hardware inference either.
- Existing release-policy gates remain separate: **2 artifact tests + 11 Docker release tests**. The Docker workflow remains independent; its native CPU container smoke is not proof for every Dockerfile or hardware execution provider.

Image-codec regressions exercise actual PFM preflight for finite nonzero scales (including representable subnormals), signed zero, nonfinite values and malformed numeric syntax; parsing does not depend on floating-point `std::from_chars` support in libc++. Real YOLO metadata regressions cover successful creation and invalid-model rejection; the ASan+UBSan row also checks borrowed ONNX Runtime allocator lifetimes on these paths.

The public `LogContext::GenerateTraceId()` helper returns UUIDv4 format: 36 ASCII characters, lowercase hexadecimal in `8-4-4-4-12` groups, version `4`, variant `8/9/a/b`, and no NUL. Production code currently does not call it, and it does not automatically integrate with logging. It uses a noncryptographic PRNG and is not suitable for security tokens; `test_trace_id` checks scalar, batch and concurrent call formats, not uniqueness proven by finite samples.

Before model-backed runs, fetch Git LFS resources and run the fixture preflight below. Every required fixture must be a regular, nonempty file containing actual bytes, not an LFS pointer. This includes intentionally invalid ONNX metadata and runtime-failure fixtures: negative tests still require their inputs. Missing resources fail rather than become successful omissions.

```sh
git lfs pull
python scripts/check_ci_fixtures.py --project-root . --layer native
python scripts/check_ci_fixtures.py --project-root . --layer http
```

Both `--project-root` and `--layer` are required. `native` checks model/dictionary fixtures for the native CPU-session tests; `http` checks real-server model, image, reference/configuration and OpenAPI assets. These byte-level checks do not replace runtime inference or FFmpeg/font/media prerequisites. Tracking commands below deliberately pass both `--yolo-model` and `--yolo-image`; omitting them does not exercise the detector-to-tracker integration.

The subtitle driver accepts `--ffmpeg` and `--font` overrides. Its FFmpeg must support drawtext, MJPEG, libx264, AAC, msmpeg4v3 and ASF/Matroska muxers; default fonts are Windows Arial or Linux DejaVuSans. The workflow retains the existing FFmpeg/font setup and Windows Media Foundation prerequisite rather than skipping unavailable media paths.

The current bounded-dispatch driver uses separate consumer witnesses: four real native OCR batches exercise decoded-image occupancy and control-plane responsiveness; four real paused subtitle uploads independently hold the data workers while native queue admission, pre-parse overload rejection and expiry without decoding are checked. It does not require native image-budget saturation and a full transport queue to coincide. The historical measurements below remain evidence for their original run, not a timing promise for this CI driver.

The following dispatch timings are historical issue #52 evidence, collected before management authentication. The Linux runtime was a separate local CPU image, not a verification of today's six Dockerfiles or issue #53 administrator policy.

**HTTP dispatch verification (Windows x64, CPU, real PP-OCR):** the final complete driver passed in 113.23 s, covering bounded overload/queue deadlines, shared native/MCP image budget, empty-input contract differences, slow/disconnected clients, sequential keepalive, split/combined-write pipelining rejection and active+queued shutdown. Every loaded health sample was bracketed by four active requests reserving exactly 73,744,128 decoded-input bytes; the final 18 samples gave P50 14.4028 ms and P95/P99 15.4099 ms. Before offloading, five of six loaded probes timed out at the two-second consumer deadline. Generic HTTP regression with explicit workload quotas also passed (278.17 s), as did the enhanced paused-upload/cancellation subtitle regression (38.67 s). These are observed samples, not portable latency or RSS guarantees.

The native CPU run used the actually selected ORT SDK 1.20.0. Its matching DLL was staged beside the executable as a smoke prerequisite: file version `1.20.20241030.2.c4fb724`, source/destination SHA256 `09BFD8AE11E8E01FA5CD310B01FDB9384FD18EE61E7FAFF7E2F55D248B8C8E9B`. No build packaging rule or API version was changed for this staging. This evidence does not certify DirectML/CUDA/TensorRT/RKNPU execution, all platforms, bounded RSS or an online production Docker build.

**Linux/Docker CPU verification (linux/amd64 under WSL Debian):** the complete Linux builder dispatch driver passed in 114.42 s. Its four 16-image OCR requests held exactly 73,744,128 decoded-input bytes throughout all 18 loaded health samples (P50 0.313115 ms, P95/P99/max 2.63656 ms). A separate real PID1 runtime-container smoke passed readiness, model discovery and nonempty OCR warmup, then sustained four 32-image OCR requests at exactly 147,488,256 bytes. Its 18 health samples measured P50 0.743417 ms and P95/P99/max 1.083073 ms; models/v1 models/stats took 2.087264/1.657175/1.268436 ms. A fresh Docker HEALTHCHECK tick remained healthy with zero failures while all four requests were active. With four active and four waiting handlers, malformed excess admission returned 503 before business parsing; SIGTERM drained in 2.045 s, exited 15 without OOM/forced kill, and the owned container was removed.

This used a temporary cache-dependency/current-source recipe, GCC 16.2.0, Xmake 3.1.1 and official Linux ORT 1.22.0 (downloaded archive SHA256 `8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3`). Validation prerequisites included fixture transport, builder socket-inspection tools and explicit SDK shared-library staging; actual ELF dependencies resolved, and the pre-copy production output already resolved `libonnxruntime.so.1`. Runtime image digest began `ff27e5befdf60`; server SHA256 began `9401fbbef6de`. This verifies that local CPU amd64 recipe/runtime, not every provider/platform or the unchanged production Dockerfile's online build path.

```sh
# Build CPU regression targets individually
xmake build server
xmake build test_common
xmake build test_trace_id
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
xmake build test_image_codec
xmake build test_image_budget_service
xmake build test_subtitle_service

xmake run test_common
xmake run test_trace_id
xmake run test_cvt
xmake run test_vision_helper
xmake run test_yolo_postprocess
xmake run test_ocr_decode
xmake run test_ocr_batch --project-root .
xmake run test_infer_inputs --project-root .
xmake run test_pipeline --project-root .
xmake run test_yolo_tasks --project-root .
xmake run test_tracker
xmake run test_config_load
xmake run test_subtitle_timeline
xmake run test_image_codec
xmake run test_image_budget_service --project-root .
xmake run test_subtitle_service --project-root .
```

The Python 3 standard-library HTTP driver creates isolated configuration, ports and processes; it does not touch an existing port 11451 service. Missing models, dictionaries, images or failure fixtures fail the run rather than count as SKIP.

The management-security driver uses the standard real CPU model/image fixtures and explicitly stages a test-only secret in isolated temporary configuration/process environment, not a production credential. The Windows x64 native driver passed in 85.93 s; this does not verify a Docker/proxy deployment.

**Documentation-example regression:** install the test-only PyYAML, openapi-spec-validator and Pillow dependencies; these add no server runtime dependency. This driver validates the public MCP configuration, the complete OpenAPI 3.0 specification and local references, all schema/media examples, and full decoding of compact base64 images. It then sends the exact documented YOLO/OCR requests to an isolated CPU server process, checking HTTP 200, response schemas and one result per input, and performs MCP initialization and tool discovery. It does not touch an existing port 11451 service. Complete the CPU build, Git LFS download and HTTP fixture preflight above first:

This documentation driver requires Python 3.10 or newer; installing its test dependencies in a virtual environment is recommended.

```powershell
# Windows: use the native CPU build above; adjust for a custom build directory
python -m pip install -r scripts/requirements-doc-tests.txt
$server = 'build/windows/x64/release/vision_simple-server.exe'
python scripts/test_documentation_validation.py
python scripts/test_documentation_examples.py --server "$server" --project-root .
```

```sh
# Linux: discover the actual native executable from the current xmake configuration
python3 -m pip install -r scripts/requirements-doc-tests.txt
server="$(xmake lua -q -c "import('core.project.config'); config.load(); import('core.project.project'); io.write(path.absolute(project.target('server'):targetfile()))")"
python3 scripts/test_documentation_validation.py
python3 scripts/test_documentation_examples.py --server "$server" --project-root .
```

Both `--server` and `--project-root` are required for the documentation integration driver `test_documentation_examples.py`; the static boundary unit tests do not start a server.


```powershell
python scripts/test_http_regression.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
python scripts/test_management_security.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
python scripts/test_http_dispatch.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
python scripts/test_subtitle_regression.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
python scripts/test_protocol_regression.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
python scripts/test_image_budget.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
python scripts/test_yolo_tasks_http.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
python scripts/test_tracking_regression.py --server build/windows/x64/release/vision_simple-server.exe --project-root . --yolo-model app/assets/test/hd2-yolo11n-fp32.onnx --yolo-image app/assets/test/hd2.png
python scripts/test_model_registry.py --server build/windows/x64/release/vision_simple-server.exe --project-root .
```

For Linux or a custom build directory, discover the actual target:

```sh
server="$(xmake lua -q -c "import('core.project.config'); config.load(); import('core.project.project'); io.write(path.absolute(project.target('server'):targetfile()))")"
python3 scripts/test_http_regression.py --server "$server" --project-root .
python3 scripts/test_management_security.py --server "$server" --project-root .
python3 scripts/test_http_dispatch.py --server "$server" --project-root .
python3 scripts/test_subtitle_regression.py --server "$server" --project-root .
python3 scripts/test_protocol_regression.py --server "$server" --project-root .
python3 scripts/test_image_budget.py --server "$server" --project-root .
python3 scripts/test_yolo_tasks_http.py --server "$server" --project-root .
python3 scripts/test_tracking_regression.py --server "$server" --project-root . --yolo-model app/assets/test/hd2-yolo11n-fp32.onnx --yolo-image app/assets/test/hd2.png
python3 scripts/test_model_registry.py --server "$server" --project-root .
```

`test_yolo`/`test_ocr` remain interactive demos, not headless acceptance tests. Tiny failure models are checked in; only regeneration requires the development package `onnx` and `scripts/generate_reliability_fixtures.py`, not a server runtime dependency.

The real Windows CPU image-budget regression passed cross-v0/v1/OpenAI/MCP predecode refusal under small limits, exact batch/global boundaries, concurrent overload/recovery, real ORT exceptions and quota refunds after physical drain on cancellation, timeout, disconnect and shutdown. Existing protocol regressions also passed. A separate 16-pixel/48-byte smoke rejected 100 PNG headers declaring 2³¹ pixels: final `decode_calls=0`, `in_use_bytes=peak_bytes=0`, `rejected_requests=100`. Windows RSS was 51933184 bytes after the first request and 54845440 after the hundredth, with process peak unchanged at 71593984 bytes. This controlled refusal proves the codec was not started, not an exact RSS cap, acceptance of every format payload, or freedom from OOM for arbitrary input.

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

#### GHCR Multi-platform Publication

`.github/workflows/docker.yml` builds and publishes Linux amd64 and arm64 CPU images to `ghcr.io/lona-cn/vision-simple`:

- Pushing a `vX.Y.Z` or `vX.Y.Z-prerelease` tag triggers publication. Manual runs default to build-and-test only; explicitly selecting `publish=true` enables publication. Publishing uses the automatically provided `GITHUB_TOKEN` with `packages: write`; no PAT secrets are required.
- amd64 and arm64 are built on native GitHub-hosted runners. Each image passes the HTTP/HEALTHCHECK smoke test, graceful shutdown and cleanup before that tested image is pushed.
- Each architecture is published as `<version>-cpu-amd64` / `<version>-cpu-arm64` and `sha-<full commit SHA>-cpu-amd64` / `sha-<full commit SHA>-cpu-arm64`. After both builds succeed, those images are assembled as multi-platform `<version>-cpu` and `sha-<full commit SHA>-cpu` manifests. Stable releases also update `latest`; prereleases do not.
- Manifest verification checks both platform descriptors and confirms their registry digests match the smoke-tested images. The release summary is written only after all manifest tags pass. Platform-specific tags already pushed are not automatically rolled back if a later step fails.
- `docker pull ghcr.io/lona-cn/vision-simple:latest` selects amd64 or arm64 for the client platform. Prefer the immutable `ghcr.io/lona-cn/vision-simple@sha256:...` reference from the successful summary for deployment. A newly published GHCR package is private by default; change its visibility in GitHub package settings for anonymous pulls.
- The ARMv7 Dockerfile queries Xmake for the server's actual artifact directory and checks that the server is ELF32 ARM with readelf during the build; these checks do not establish container startup or target-device runtime acceptance. ARMv7, RISC-V, CUDA/TensorRT and RKNPU images remain outside this multi-platform publication; hardware-accelerated backends need separate image variants and device validation.

Release-policy boundary tests: `python3 -m unittest discover -s scripts -p test_docker_release.py`. The amd64 CPU build enables AVX/AVX2/F16C and requires a compatible CPU.

#### Other Platforms / Hardware Acceleration

The GHCR multi-platform manifest currently contains only Linux CPU `amd64` and `arm64`. The ARMv7 Dockerfile now queries the actual artifact directory and includes an ELF32 ARM build check. An image build using a local dependency cache completed under QEMU ARM emulation on WSL Debian, followed by smoke checks for ELF32 ARM, dynamic-library resolution and startup of the HTTP model catalog at `GET /v0/infer/models`. This smoke run used a temporary build recipe supplying local dependency sources, including the same ONNX Runtime version and Eigen commit; it does not verify the production Dockerfile's online dependency-download path, real ARM hardware or inference. RISC-V has not been runtime-tested on target hardware. CUDA/TensorRT and RKNPU require matching hardware validation and separate image variants; do not merge them with CPU images of the same architecture in one manifest.

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
    auto result = (*model)->Run(image, YOLOInferenceOptions{.confidence = 0.625f});
    return result ? 0 : 1;
}
```

## License
The copyrights for the YOLO models and PaddleOCR models in this project belong to the original authors.

This project is licensed under the Apache-2.0 license.