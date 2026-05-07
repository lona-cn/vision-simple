# INFER MODULE

## OVERVIEW
ONNXRuntime inference engine for YOLO (v10/v11) and OCR (PaddleOCR det+rec). 9 C++ files, plain-main tests, no gtest.

## STRUCTURE
```
app/source/runtime/infer/
├── Infer.h                 # public API: InferContext, InferYOLO, InferOCR
├── VisionHelper.hpp        # Letterbox, HWC2CHW_BGR2RGB, NMS, ScaleCoords
├── DXInfo.hpp              # Windows GPU enumeration
├── private/
│   ├── InferYOLO.h/.cpp    # YOLOFilter + InferYOLOOrtImpl
│   ├── InferOCR.h/.cpp     # InferOCROrtPaddleImpl (PIMPL)
│   ├── InferORT.h/.cpp     # ONNXRuntime context wrapper
│   └── Infer.cpp           # InferContext factory
├── test/
│   ├── test_yolo.cpp       # video decode + inference + display loop
│   ├── test_ocr.cpp        # det+rec end-to-end test
│   └── Util.hpp            # ReadAll, SafeQueue, FPSCounter
└── xmake.lua               # opencv + onnxruntime + magic_enum deps
```

## WHERE TO LOOK
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `YOLOFilter` | Class | `private/InferYOLO.h:11` | Version dispatch + NMS + coordinate rescale |
| `YOLOFilter::v10` / `v11` | Method | `private/InferYOLO.cpp:166` / `122` | Per-version post-processing logic |
| `InferYOLOOrtImpl` | Class | `private/InferYOLO.h:48` | ONNX YOLO session + IOBinding |
| `InferOCROrtPaddleImpl` | Class | `private/InferOCR.h:8` | PIMPL wrapper, delegates to `Impl` |
| `InferOCROrtPaddleImpl::Impl` | Struct | `private/InferOCR.cpp:29` | det session + rec session + pre/post-process |
| `InferContextORT` | Class | `private/InferORT.h` | Creates Ort::Session from model span |
| `VisionHelper::HWC2CHW_BGR2RGB<T>` | Template | `VisionHelper.hpp` | In-place BGR->RGB channel reorder |
| `Cvt::cvt` | Static | `VisionHelper.hpp:56` | fp32<->fp16 via AVX on x86_64 |

## CONVENTIONS
- **YOLO dispatch**: `YOLOFilter::operator()` branches on `YOLOVersion` enum to `v10()` or `v11()`
- **OCR two-stage**: `DetPreProcess` -> det.Run -> `DetPostProcess` -> per-box `RecPreProcess` -> rec.Run -> `RecPostProcess`
- **IOBinding**: All ORT inference uses `Ort::IoBinding` for zero-copy input/output
- **Template `Create` overloads**: `InferYOLO::Create<T>` and `InferOCR::Create<T>` accept any `std::is_arithmetic_v<T>` span, cast to `uint8_t`
- **Tests**: Plain `main()` with `CHECK_RESULT` macro, video loops with `std::jthread` + `SafeQueue`

## ANTI-PATTERNS
- DO NOT batch OCR rec inference, it degrades accuracy (comment at `InferOCR.cpp:272`)
- DO NOT use `try/catch` in new code, existing catches in factory map are for ORT C++ API boundary only
- DO NOT access `private/` headers from outside `infer/` module
- DO NOT forget `PadLength()` when sizing OCR det input, model expects 32-aligned dimensions
