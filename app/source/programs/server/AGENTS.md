# Server contributor guide

This module builds the `server` target (`vision_simple-server`) with libhv. It hosts inference, protocol adapters, tracking sessions and subtitle jobs. Change the transport adapter and the shared service separately: do not put model loading or cache ownership in a route handler.

## Where to work

| Concern | Source |
| --- | --- |
| Listener, route mounting, legacy `/v0` handlers and `/v1/infer` streaming | `private/HTTPServer.cpp`, `private/HTTPServer.h` |
| Task names, model factories and task-specific result packing | `private/TaskRegistry.cpp`, `private/TaskRegistry.h` |
| Decode, inference pipeline, model leases, cache and statistics | `private/InferenceService.cpp`, `private/InferenceService.h` |
| Shared response serialization, error mapping and model pagination | `private/InferenceProtocol.h` |
| OpenAI-style models/chat, MCP SSE/tools, tracking, subtitle jobs | `private/OpenAIAdapter.cpp`, `private/MCPAdapter.cpp`, `private/TrackingAdapter.cpp`, `private/SubtitleAdapter.cpp` |
| Process startup and logging | `private/main.cpp`, `private/Logger.cpp` |

## Routes and wire contracts

- `POST /v0/infer/yolo` and `/v0/infer/ocr` accept JSON `{"model":"name","images":["<raw base64>",...],"timeout_ms":60000}`; `timeout_ms` is optional, 1–300000. `images` may be empty but cannot exceed `infer_max_batch_images`. YOLO responds with `class_names` and per-image `results` of `{class_id,confidence,bbox}`; OCR responds with per-image `results` of `{line,confidence,bbox}`. Boxes are `[x,y,width,height]`.
- `POST /v1/infer/{yolo|ocr|seg|pose|obb}` uses the same inference JSON and typed result shapes; these handlers enforce JSON content type and a 64 MiB body limit while streaming the upload. Segmentation includes `mask_png_base64`, pose includes `keypoints`, and OBB includes `corners` and `angle`. Task IDs come from `RegisteredTasks()`; update that registry rather than hard-coding a new `/v1` route.
- `GET /v0/infer/models` returns only the legacy `{"yolo":[...],"ocr":[...]}` view of configured models. `POST /v0/infer/unload` accepts `{"kind":"yolo","model":"name"}` (any registered task ID is valid) and returns `{kind,model,unloaded:true}`; a missing loaded entry is 404, an active one 409. `GET /v0/infer/stats?limit=100&offset=0` paginates loaded entries, not the configured catalog, returning `models`, `idle_timeout_ms`, `total`, `limit`, `offset`; limit is 1–200.
- `GET /v1/models` lists all configured tasks with IDs `task:name` and supports `limit` (1–200) plus opaque `after` cursor. `POST /v1/chat/completions` requires such a prefixed model ID and user `image_url` parts containing inline base64 image data URLs; text alone does not generate language. Its assistant `content` is serialized inference-result JSON, either in one completion or SSE chunks with `stream:true`. See `OpenAIAdapter.cpp` for accepted fields.
- MCP is `GET /mcp/sse` plus `POST /mcp/messages`; tools are `list_models` and `infer_<task ID>`, with raw model names and raw base64 image strings for inference. Tracking lives under `/v1/tracking/sessions` (create/list/get, frame step, reset, delete); subtitles under `/v1/subtitle/jobs` (create/list/get, video upload, cancel, delete, SRT/VTT download). Consult their adapters before changing payloads or session lifecycle.
- Legacy inference errors use `{"error":{"code":"...","message":"...","image_index":null}}` (index set for image-specific failures); HTTP status is set before `send()`. Error mappings are centralized in `InferenceProtocol.h`; adapter errors have their own contracts. Inference responses use `struct_json`, while handlers/adapters use `nlohmann::json` for parsing and other envelopes.

## Configuration and lifecycle

- Launch from a working directory with `config/server.yaml`, `config/models.yaml` and logger configuration. `main.cpp` loads server options, starts asynchronously, and stops on stdin newline or registered signals; shutdown stops subtitle, tracking and MCP adapters before the libhv listener and calls `hv::async::cleanup()`. Default/example config is under `app/config/base/`; model paths there are relative to the process working directory.
- `server.yaml` sets `host`, `port` and string-valued `options`: `static_path`, `infer_framework`, `infer_ep`, `infer_device`, `infer_idle_timeout_ms`, `infer_sweep_interval_ms`, `infer_pipeline_capacity`, `infer_pipeline_max_batches`, `infer_max_batch_images`, `infer_timeout_ms`, `ocr_rec_batch_size`. Validate new options at server/service construction, not per request.
- `models.yaml` uses `models: [{task,name,version,files}]`; `yolo` and `ocr` legacy sections are normalized by `Config` for compatibility. `(task,name)` is unique; `Config::Instance()` loads and retains the catalog on first use. Discovery does not load model weights. `InferenceService` lazily loads models by `(task,name)` and holds a mutex-protected cache; a response's lease protects active entries through serialization and records requests, failures and duration. Idle entries are swept after `infer_idle_timeout_ms` (zero disables sweeping); unload refuses active entries. Never mutate a loaded entry outside the service.

## Verify changes

- Build `server` with `xmake build server`; the executable is `vision_simple-server` (use your configured xmake target path rather than assuming a release directory). Run `xmake build test_subtitle_timeline` and `xmake run test_subtitle_timeline` for timeline changes.
- With a built executable and required real model/image fixtures, run `python scripts/test_http_regression.py --server <executable> --project-root .` from the repository root. Use the same arguments for `scripts/test_model_registry.py`, `scripts/test_protocol_regression.py`, `scripts/test_yolo_tasks_http.py`, `scripts/test_tracking_regression.py`, and `scripts/test_subtitle_regression.py` as applicable. The subtitle driver also requires FFmpeg and a font. These HTTP scripts launch isolated temporary server configurations/ports; missing fixtures fail rather than skip. `test_yolo` and `test_ocr` are interactive demos, not headless regression targets.
