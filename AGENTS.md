# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-08
**Commit:** 3913972
**Branch:** main

## OVERVIEW
vision-simple — C++23 vision inference library with YOLO/OCR support via ONNXRuntime, served through embedded HTTP API. Cross-platform: Windows(x64), Linux(x86_64/arm64/riscv64).

## STRUCTURE
```
./
├── app/source/
│   ├── runtime/
│   │   ├── infer/        # 推理引擎: YOLO, OCR, VisionHelper
│   │   └── common/       # 共享: Error, Config, IOUtil
│   └── programs/
│       ├── server/       # HTTP 推理服务 (libhv, port 11451)
│       └── demo/         # Demo 程序
├── xmake/                # xmake 构建配置
│   ├── project.lua       # 主构建配置
│   ├── options.lua       # 可选特性开关 (DML/CUDA/TensorRT/RKNPU)
│   ├── funcs/            # 构建函数
│   └── repo/             # 预编译包定义
├── docker/               # 6 个平台的 Dockerfile
├── scripts/              # 编译脚本 (Win/Linux)
└── xmake.lua             # 根构建入口
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| YOLO 推理实现 | `app/source/runtime/infer/private/InferYOLO.cpp` | 315 lines |
| OCR 推理实现 | `app/source/runtime/infer/private/InferOCR.cpp` | 303 lines |
| ONNX 执行提供者 | `app/source/runtime/infer/private/InferORT.cpp` | ONNXRuntime wrapper |
| HTTP 路由/API | `app/source/programs/server/private/HTTPServer.cpp` | 440 lines, libhv |
| 公开 API 头文件 | `app/source/runtime/infer/Infer.h` | InferYOLO, InferOCR, InferContext |
| 错误处理系统 | `app/source/runtime/common/VisionSimpleError.h` | VSResult<T> = expected<T, Error> |
| 构建选项 | `xmake/options.lua` | DML/CUDA/TensorRT/RKNPU 开关 |
| 测试文件 | `app/source/runtime/infer/test/` | C++ main() 测试, 非框架 |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `vision_simple` | Namespace | Infer.h:16 | 所有 API 的命名空间 |
| `InferContext` | Class | Infer.h:40 | 推理上下文: 框架+EP+参数 |
| `InferYOLO` | Class | Infer.h:82 | YOLO 推理接口 (Create + Run) |
| `InferOCR` | Class | Infer.h:133 | OCR 推理接口 (Create + Run) |
| `VisionHelper` | Class | VisionHelper.hpp:177 | 图像预处理: Letterbox, BGR2RGB, NMS |
| `Cvt` | Class | VisionHelper.hpp:56 | 图像数据类型转换 (fp32↔fp16↔u8) |
| `VSResult<T>` | Type alias | Infer.h:18 | `std::expected<T, VisionSimpleError>` |
| `VisionSimpleError` | Class | VisionSimpleError.h:27 | 统一错误类型 |
| `MK_VSERROR` | Macro | VisionSimpleError.h:46 | 创建错误结果 |
| `HTTPServerImpl` | Class | HTTPServer.cpp:59 | HTTP 服务实现 (libhv) |

## CONVENTIONS
- **C++23**: concepts (`requires`), `std::expected`, `std::span`, `std::pmr`, `std::source_location`
- **头文件**: `#pragma once` (不用 include guards)
- **命名**: 下划线后缀 `_` 标记成员变量; `kPascalCase` 标记枚举值
- **工厂模式**: 公开类通过 `static CreateResult Create(...)` 构造, 禁止拷贝, 允许移动
- **错误处理**: 禁止异常, 使用 `VSResult<T>` / `std::expected` + 显式 `if (!result)` 检查
- **`private/` 目录**: 内部实现隐藏 — 每个模块的实现文件放在 `private/` 子目录
- **`VISION_SIMPLE_API`**: DLL 导出宏 — 所有公开类/函数需标注
- **测试**: `test/` 子目录下普通 `main()` 可执行文件, 未使用 gtest/gmock 框架

## ANTI-PATTERNS (THIS PROJECT)
- DO NOT throw exceptions — 使用 `VSResult<T>` 返回错误
- DO NOT copy InferContext / InferYOLO / InferOCR — 拷贝构造已删除
- DO NOT include 内部头文件 (`private/`) 跨模块
- DO NOT use include guards — 用 `#pragma once`

## UNIQUE STYLES
- `unordered_map<string,string>` 用于 `InferArgs` — 类型擦除的参数传递
- `std::span<T>` 用于模型数据传递 — 零拷贝, 支持 `uint8_t` 和算术类型模板
- `std::expected` + `std::unexpected` 错误链, 配合 `MK_VSERROR` 宏
- `magic_enum` 库用于枚举反射日志

## COMMANDS
```bash
# 构建
xmake build server          # 构建 HTTP 服务

# 运行
xmake run server            # 运行 HTTP 服务 (port 11451)

# 配置 (Windows + DML)
xmake f -p windows -a x64 -m release --with_dml=y

# 配置 (Linux + CPU)
xmake f -p linux -a x86_64 --toolchain=gcc -m release

# Docker 构建
docker build -t vision-simple -f docker/Dockerfile.debian-bookworm-x86_64-cpu .

# Docker 运行
docker run -it --rm -p 11451:11451 vision-simple

# 测试 (手动运行可执行文件)
xmake build test_yolo
xmake run test_yolo
```

## NOTES
- Windows 开发: 使用 `scripts/dev-vs.bat` 生成 VS 项目
- 模型文件 `.onnx` 位于 `app/assets/test/` 目录
- 默认编译器: Windows=MSVC, Linux=GCC-13/Clang
- 3 个 Git Submodule: `third/` 包含第三方依赖 (libhv, struct_yaml, struct_json)
- 公开 API 头文件安装路径: `<vision_simple/Infer.h>` (通过 include 路径约定)
