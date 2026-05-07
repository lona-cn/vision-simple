# Server Module Knowledge Base

**Scope:** `app/source/programs/server`

## OVERVIEW
HTTP inference service on libhv exposing REST endpoints for YOLO/OCR batch inference with lazy model caching.

## STRUCTURE
```
private/
  main.cpp              # entry: YAML config, signals, libhv async cleanup
  HTTPServer.cpp        # routes, handlers, model cache, JSON (de)serialization
  HTTPServer.h          # HTTPServer interface + HTTPServerOptions
  Logger.cpp/.h/.hpp    # async logging wrapper
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Route registration | HTTPServer.cpp:188-211 | POST /v0/infer/yolo, /v0/infer/ocr; GET /v0/infer/models |
| YOLO handler | HTTPServer.cpp:287-356 | base64 -> cv::Mat -> InferYOLO::Run -> struct_json |
| OCR handler | HTTPServer.cpp:358-422 | base64 -> cv::Mat -> InferOCR::Run -> nlohmann JSON |
| Model list handler | HTTPServer.cpp:259-285 | reads Config::Instance(), returns {yolo:[],ocr:[]} |
| Model cache | HTTPServer.cpp:64-67, 69-176 | map<string, unique_ptr<InferYOLO/OCR>> + shared_mutex double-checked locking |
| Request/response structs | HTTPServer.cpp:25-57 | InferYOLORequest, InferOCRRequest, YOLODetectedObject, InferYOLOResponse, OCRLine, InferOCRResponse |
| Server options / YAML | HTTPServer.h:25-35, main.cpp:45-66 | struct_yaml parses config/server.yaml into HTTPServerOptions |
| JSON serialization | HTTPServer.cpp:11-12, 345-346, 411-412 | struct_json for YOLO; nlohmann macros for OCRLine/OCRResponse |
| Base64 decode | HTTPServer.cpp:9, 306-311 | turbobase64 tb64dec/tb64declen |

## CONVENTIONS
- Routes use `/v0/` prefix (not `/v1/`).
- Images travel as base64 strings inside JSON body, never multipart.
- Handlers return `int` HTTP status; errors sent via `ctx->sendString()`.
- `struct_json` (ylt) is primary serializer; `nlohmann` intrusive macros used only for OCR nested types because `struct_json` lacks equivalent vector nesting support.

## ANTI-PATTERNS
- DO NOT store raw `cv::Mat` in model cache — cache only `InferYOLO` / `InferOCR` instances.
- DO NOT skip `shared_lock` before `unique_lock` in cache lookup — double-checked locking pattern is required.
- DO NOT mix `struct_json` and `nlohmann` on the same type — choose one per struct.
