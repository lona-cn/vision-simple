# PROJECT KNOWLEDGE BASE

## OVERVIEW
vision-simple 是 C++23 视觉推理库及内嵌 libhv HTTP 服务。推理由 ONNX Runtime 执行，公开 YOLO、OCR 接口；服务还提供模型目录、跟踪及字幕相关能力。支持 Windows x64 和 Linux 多架构；具体构建选项以 xmake 配置为准。

## NAVIGATION
- `app/source/runtime/infer/`：`Infer.h` 公开推理接口；`InferYOLOTask.h`、`InferPipeline.h`、`Tracker.h` 是任务/流水线/跟踪接口；实现见 `private/`。该目录的 `AGENTS.md` 说明推理细节。
- `app/source/runtime/common/`：`VisionSimpleError.h`、`VisionSimpleConfig.h`、`IOUtil.h` 及日志接口；`Config::Load` 从 YAML 加载模型定义，`ModelConfig::models` 为统一目录，`yolo`/`ocr` 为兼容投影。
- `app/source/programs/server/`：HTTP 路由、服务启动、模型管理；先读该目录的 `AGENTS.md` 再修改协议。
- `app/source/programs/demo/`：示例程序。
- `app/config/base/`：服务、模型和日志示例配置；默认监听端口由 `server.yaml` 指定（目前 11451）。
- `app/assets/test/`：模型及回归测试夹具；`app/assets/main/` 存放运行时资源。
- `xmake/project.lua`、`xmake/options.lua`、`xmake/funcs/`：语言/平台配置、执行提供者选项、目标及测试生成规则。
- `scripts/`：服务协议、模型注册、推理任务、跟踪、字幕和容器发布等 Python 回归脚本；`doc/openapi/` 是 HTTP 协议描述，`.github/workflows/` 是 CI。
- `docker/`：按平台划分的容器构建文件。

## CODE AND CHANGE CONVENTIONS
- 公开推理 API 使用 `InferContext::Create`、`InferYOLO::Create`、`InferOCR::Create` 与 `Run`；返回值为 `VSResult<T>` / `InferResult<T>`（`std::expected`），错误使用 `VisionSimpleError`，不要把异常作为公开错误协议。ONNX Runtime 异常在实现边界转换为错误结果。
- 公开接口头文件使用 `#pragma once`，导出符号使用 `VISION_SIMPLE_API`；`private/` 的实现头文件不跨模块引用。`InferContext`、`InferYOLO`、`InferOCR` 不可复制。
- 模型定义写入 `app/config/base/models.yaml` 的 `models` 列表（`task`、`name`、`version`、`files`）；更改模型 schema 同时检查配置加载器、服务注册逻辑、OpenAPI 和回归脚本。
- 修改 HTTP 请求/响应或路由时同步检查 `doc/openapi/server.yaml` 及 `scripts/` 中对应的协议、模型注册、任务、跟踪或字幕回归脚本；不要只改 handler。
- `xmake/funcs/funcs_target.lua` 为模块 `test/test_*.cpp` 自动创建同名可执行目标，测试仍是普通 C++ `main()`；`scripts/test_*.py` 覆盖跨进程协议与发布流程。

## BUILD AND VERIFICATION
```bash
xmake build server
xmake run server
xmake build test_yolo_postprocess
xmake run test_yolo_postprocess
```
运行服务需要相应配置、模型文件和动态库；具体复制规则见 `xmake/rules/`。变更推理/HTTP 行为时选对应的 C++ 目标或 Python 回归脚本，并用真实服务和模型验证端到端路径；不要把只检查接口存在的测试当作推理验证。
