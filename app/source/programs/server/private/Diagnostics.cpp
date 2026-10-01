#include "Diagnostics.h"
#include "DebugArtifacts.h"
#include "HTTPServer.h"
#include "InferenceConfiguration.h"
#include "InferenceProtocol.h"
#include "IOUtil.h"
#include "VisionSimpleConfig.h"
#include <nlohmann/json.hpp>
#include <magic_enum.hpp>
#include <ylt/struct_yaml/yaml_reader.h>
#include <turbobase64/turbob64.h>
#include <opencv2/core/utils/logger.hpp>
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>
#ifndef _WIN32
#include <csignal>
#include <utility>
#endif

namespace vision_simple {
namespace {
#ifndef _WIN32
struct DiagnosticPipeSignal {
  using Handler = void (*)(int);
  Handler previous = std::signal(SIGPIPE, SIG_IGN);
  bool Restore() noexcept {
    if (previous == SIG_ERR) return false;
    return std::signal(SIGPIPE, std::exchange(previous, SIG_ERR)) != SIG_ERR;
  }
  ~DiagnosticPipeSignal() {
    if (previous != SIG_ERR && !Restore())
      std::fputs("Diagnostic output signal cannot be restored.\n", stderr);
  }
};
#endif
using Json = nlohmann::json;
constexpr size_t kSelectionMax = 16, kRawMax = 48 * 1024 * 1024,
                 kEncodedMax = 64 * 1024 * 1024;
constexpr auto kHelp = R"(vision_simple-server
  (no arguments) Start the configured HTTP server.
  --help
  --diagnose preflight --model task:name [--model task:name ...] [--timeout-ms N]
  --diagnose warmup --model task:name [--model task:name ...] --image path
      [--image path ...] [--timeout-ms N]
      [--debug-dir NAME [--debug-max-bytes N] [--debug-max-files N]]
Debug: exactly one OCR warmup selection, 1..16 frames, new private CWD child.
NAME: 1..64 ASCII alnum/_/- starting alnum; Windows reserved names forbidden.
Bytes: 1..67108864 (default 67108864); files: 1..64 (default 64).
Input PNGs may be sensitive. Manifest has boxes/confidence, never recognized text.
POSIX CWD must be owned by this user and not group/world writable.
Success retains output for manual cleanup; failures roll back owned output.
Uses config/server.yaml and config/models.yaml in the current directory.
Select 1..16 unique registered task:model pairs; no default full-model loading.
Preflight performs one session load with zero frames, not a smoke test.
Warmup performs one full cold and, only after success, one full warm batch.
Cold means a fresh local service cache, not a cold OS, driver or runtime cache.
Images: 1..min(configured batch limit,128), 48 MiB raw / 64 MiB encoded total.
Each call uses configured resource budgets and a cooperative timeout (1..300000 ms);
native draining may overrun it. No hard kill, retries or HTTP health side effects.
Reports expire with this invocation; repeating starts a fresh local service cache.
Provider availability/context creation does not guarantee hardware placement.
Stage wall times include native gate/binding/output work, not pure ORT kernel time;
stage/frame totals overlap and are not the batch wall time.
Pipeline calls=0 means no stage execution; queue residence may still be recorded.
Zero-call service stages are null; zero-call pipeline stages retain their counters.
Exit: 0 desired states and unload succeeded, 1 operational failure, 2 invalid arguments.
)";
struct Selection { const TaskDescriptor* task; std::string model; };
struct Arguments {
  std::string mode;
  std::vector<Selection> models;
  std::vector<std::string> images;
  std::optional<std::chrono::milliseconds> timeout;
  std::optional<std::string> debug_dir;
  std::optional<size_t> debug_max_bytes, debug_max_files;
};
bool WriteOutput(std::string_view body) noexcept {
  const bool written = std::fwrite(body.data(), 1, body.size(), stdout) == body.size();
  const bool newline = std::fputc('\n', stdout) != EOF;
  const bool flushed = std::fflush(stdout) == 0;
  if (written && newline && flushed) return true;
  std::fputs("Diagnostic output cannot be written.\n", stderr);
  return false;
}
int Fatal(const char* code, const char* message, int status = 1) noexcept {
  // Only static strings reach this allocation-free last-resort error path.
  char body[512];
  const int size = std::snprintf(body, sizeof(body),
      "{\"schema_version\":1,\"error\":{\"code\":\"%s\",\"message\":\"%s\"}}", code, message);
  if (size < 0 || static_cast<size_t>(size) >= sizeof(body)) {
    std::fputs("Diagnostic output cannot be written.\n", stderr);
    return 1;
  }
  return WriteOutput(std::string_view(body, static_cast<size_t>(size))) ? status : 1;
}
bool ParseArguments(int argc, char* argv[], Arguments& out) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view flag{argv[i]};
    if (i + 1 == argc) return false;
    const std::string_view value{argv[++i]};
    if (flag == "--diagnose") {
      if (!out.mode.empty() || (value != "preflight" && value != "warmup")) return false;
      out.mode = value;
    } else if (flag == "--model") {
      const auto colon = value.find(':');
      if (colon == std::string_view::npos || colon + 1 == value.size() || out.models.size() == kSelectionMax) return false;
      const auto* task = FindTask(value.substr(0, colon));
      if (!task) return false;
      const auto model = value.substr(colon + 1);
      for (const auto& selected : out.models)
        if (selected.task == task && selected.model == model) return false;
      out.models.push_back({task, std::string(model)});
    } else if (flag == "--image") {
      if (value.empty() || out.images.size() == 128) return false;
      out.images.emplace_back(value);
    } else if (flag == "--timeout-ms") {
      if (out.timeout) return false;
      uint64_t ms = 0;
      const auto parsed = std::from_chars(value.data(), value.data() + value.size(), ms);
      if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || ms == 0 || ms > 300000) return false;
      out.timeout = std::chrono::milliseconds(ms);
    } else if (flag == "--debug-dir") {
      if (out.debug_dir || !DebugArtifacts::ValidName(value)) return false;
      out.debug_dir = value;
    } else if (flag == "--debug-max-bytes" || flag == "--debug-max-files") {
      auto& limit = flag == "--debug-max-bytes" ? out.debug_max_bytes : out.debug_max_files;
      size_t count = 0;
      const auto parsed = std::from_chars(value.data(), value.data() + value.size(), count);
      const size_t max = flag == "--debug-max-bytes" ? 67108864 : 64;
      if (limit || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !count || count > max) return false;
      limit = count;
    } else return false;
  }
  if (!out.debug_dir && (out.debug_max_bytes || out.debug_max_files)) return false;
  if (out.debug_dir && (out.mode != "warmup" || out.models.size() != 1 ||
      out.models.front().task->kind != InferenceKind::kOCR || out.images.empty() || out.images.size() > 16)) return false;
  return !out.mode.empty() && !out.models.empty() &&
      (out.mode == "warmup" ? !out.images.empty() : out.images.empty());
}
std::optional<std::string> Encode(std::span<const unsigned char> bytes) {
  std::string result(tb64enclen(bytes.size()), '\0');
  if (tb64enc(bytes.data(), bytes.size(),
      reinterpret_cast<unsigned char*>(result.data())) != result.size()) return std::nullopt;
  return result;
}
bool ReadFixtures(const Arguments& args, std::vector<std::string>& encoded) {
  size_t raw_total = 0, encoded_total = 0;
  for (const auto& path : args.images) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec) || ec) return false;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const auto end = file.tellg();
    if (end <= 0 || static_cast<uint64_t>(end) > kRawMax - raw_total) return false;
    const auto count = static_cast<size_t>(end);
    const auto encoded_count = ((count + 2) / 3) * 4;
    if (encoded_count > kEncodedMax - encoded_total) return false;
    std::vector<unsigned char> bytes(count);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(count))) return false;
    if (file.peek() != std::char_traits<char>::eof()) return false;
    raw_total += count;
    encoded_total += encoded_count;
    auto image = Encode(bytes);
    if (!image) return false;
    encoded.push_back(std::move(*image));
  }
  return true;
}
Json Stage(const ServiceStageTiming& value) {
  if (!value.calls) return nullptr;
  return {{"elapsed_ns", value.elapsed_ns}, {"calls", value.calls}, {"completed_calls", value.completed_calls}};
}
Json Stage(const PipelineStageTiming& value) {
  return {{"execution_ns", value.execution_ns}, {"queue_ns", value.queue_ns}, {"calls", value.calls}, {"completed_calls", value.completed_calls}};
}
Json Timing(const RequestTiming& value) {
  Json pipeline = nullptr;
  if (value.pipeline_entered) {
    const auto& p = value.pipeline;
    pipeline = {{"wall_ns", p.wall_ns}, {"capacity_wait_ns", p.capacity_wait_ns},
      {"setup_ns", p.setup_ns}, {"preprocess", Stage(p.preprocess)},
      {"inference", Stage(p.inference)}, {"postprocess", Stage(p.postprocess)},
      {"input_frames", p.input_frames}, {"completed_frames", p.completed_frames}, {"complete", p.complete}};
  }
  return {{"wall_ns", value.wall_ns}, {"model_acquire", Stage(value.model_acquire)},
    {"model_load", Stage(value.model_load)}, {"input_prepare", Stage(value.input_prepare)},
    {"decode", Stage(value.decode)}, {"pipeline", std::move(pipeline)}};
}
Json Error(const ServiceError& error) {
  Json index = nullptr;
  if (error.image_index) index = *error.image_index;
  return {{"code", DescribeError(error.kind).code}, {"image_index", std::move(index)}};
}
Json Pass(const char* phase, const MeasuredInferenceResult& measured, size_t batch) {
  Json counts = Json::array(), error = nullptr;
  if (measured.result) {
    std::visit([&](const auto& payload) {
      for (const auto& frame : payload.results) counts.push_back(frame.size());
    }, measured.result->payload);
  } else error = Error(measured.result.error());
  return {{"phase", phase}, {"success", bool(measured.result)}, {"error", std::move(error)},
    {"cache_hit", measured.timing.cache_hit}, {"batch_size", batch},
    {"result_counts", std::move(counts)}, {"timing", Timing(measured.timing)}};
}
int Diagnose(const Arguments& args, std::optional<DebugArtifacts>& artifacts) {
  // Native codecs can log exception details before service error normalization.
  // This explicit local mode exits without starting HTTP; leave server logging unchanged.
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
  HTTPServerOptions http{.host = "", .port = 11451, .options = {}};
  VSResult<InferenceServiceOptions> options;
  try {
    auto yaml = ReadAllString("config/server.yaml");
    if (!yaml) return Fatal("configuration_failed", "Diagnostic configuration cannot be read");
    struct_yaml::from_yaml(http, *yaml);
    options = ParseInferenceServiceOptions(http);
  } catch (...) { return Fatal("configuration_failed", "Diagnostic configuration cannot be parsed"); }
  if (!options) return Fatal("configuration_failed", "Diagnostic configuration is invalid");
  const size_t batch_max = std::min(size_t{128}, options->pipeline.max_batch_images);
  if (args.images.size() > batch_max) return Fatal("invalid_arguments", "Fixture count exceeds the configured batch limit", 2);
  auto config = Config::Instance();
  if (!config) return Fatal("configuration_failed", "Diagnostic model configuration cannot be read");
  auto capabilities = InferContext::Capabilities(options->framework);
  if (!capabilities) return Fatal("context_failed", "Runtime capabilities cannot be queried");
  std::vector<std::string> images;
  try {
    if (!ReadFixtures(args, images)) return Fatal("fixture_failed", "Diagnostic fixtures cannot be read within input limits");
  } catch (...) { return Fatal("fixture_failed", "Diagnostic fixtures cannot be read within input limits"); }
  auto service = InferenceService::Create(*options);
  if (!service) return Fatal("context_failed", "Diagnostic inference context cannot be created");
  Json compiled = Json::array();
  for (auto ep : capabilities->compiled_execution_providers) compiled.push_back(magic_enum::enum_name(ep));
  Json report = {{"schema_version", 1}, {"mode", args.mode}, {"validity", "this_invocation_only"},
    {"http_readiness", "not_assessed"},
    {"capabilities", {{"framework", magic_enum::enum_name(capabilities->framework)},
      {"runtime_version", capabilities->runtime_version}, {"compiled_execution_providers", std::move(compiled)},
      {"available_execution_providers", capabilities->available_execution_providers},
      {"cpu_fallback_allowed", capabilities->cpu_fallback_allowed},
      {"requested_ep", magic_enum::enum_name(options->ep)}, {"device_id", options->device_id}, {"context_created", true}}},
    {"limits", {{"selection_max", kSelectionMax}, {"batch_max", batch_max}, {"encoded_bytes_max", kEncodedMax},
      {"timeout_ms", args.timeout.value_or(options->request_timeout).count()},
      {"passes_per_model", args.mode == "preflight" ? 1 : 2}}}, {"models", Json::array()}};
  bool successful = true;
  const char* debug_error = nullptr;
  if (args.debug_dir) {
    artifacts.emplace();
    const auto initialized = artifacts->Init(*args.debug_dir, args.debug_max_bytes.value_or(67108864), args.debug_max_files.value_or(64));
    if (!initialized) debug_error = initialized.error();
  }
  for (const auto& selection : args.models) {
    const auto& definitions = config->get().model_config().models;
    const auto found = std::ranges::find_if(definitions, [&](const auto& d) {
      return d.task == selection.task->id && d.name == selection.model;
    });
    Json entry = {{"task", selection.task->id}, {"model", selection.model}, {"configured", found != definitions.end()},
      {"state", "not_configured"}, {"missing_roles", Json::array()}, {"loadable", nullptr},
      {"smoke_tested", false}, {"passes", Json::array()}, {"unloaded", false}};
    if (found != definitions.end()) {
      for (auto role : selection.task->required_files) {
        const auto file = found->files.find(std::string(role));
        std::error_code ec;
        if (file == found->files.end() || file->second.empty() ||
            !std::filesystem::is_regular_file(file->second, ec) || ec) entry["missing_roles"].push_back(role);
      }
      if (!entry["missing_roles"].empty()) entry["state"] = "missing_files";
      else {
        auto cold = (*service)->RunMeasured(selection.task->kind, selection.model, images, {},
          ServiceControl{.timeout = args.timeout});
        entry["passes"].push_back(Pass(args.mode == "preflight" ? "preflight" : "cold", cold, images.size()));
        if (!cold.result) successful = false;
        const bool loaded = cold.timing.cache_hit || cold.timing.model_load.completed_calls != 0;
        // Session success is independent of decode, inference and resource failure.
        if (cold.timing.model_load.calls || cold.timing.cache_hit) entry["loadable"] = loaded;
        entry["state"] = loaded ? (args.mode == "preflight" ? "loadable" : "smoke_failed") : "load_failed";
        if (cold.result) {
          cold.result->Succeed();
          if (args.mode == "warmup") {
            // Keep the first successful response lease through the second call,
            // even when idle expiry/sweep are configured to one millisecond.
            auto warm = (*service)->RunMeasured(selection.task->kind, selection.model, images, {},
              ServiceControl{.timeout = args.timeout});
            entry["passes"].push_back(Pass("warm", warm, images.size()));
            if (!warm.result) successful = false;
            if (warm.result) {
              if (artifacts && !debug_error) {
                const auto* payload = std::get_if<InferOCRResponse>(&warm.result->payload);
                if (!payload) debug_error = "debug_payload";
                else {
                  const auto exported = artifacts->Export(images, *payload,
                      found->ocr_detection.value_or(OCRDetectionOptions{}), options->max_image_pixels);
                  if (!exported) debug_error = exported.error();
                }
              }
              warm.result->Succeed();
              entry["smoke_tested"] = true;
              entry["state"] = "smoke_tested";
              warm.result->completion.reset();
            }
          }
          cold.result->completion.reset();
        }
        if (loaded) {
          auto unloaded = (*service)->Unload(selection.task->kind, selection.model);
          entry["unloaded"] = bool(unloaded);
          if (!unloaded) entry["unload_error"] = Error(unloaded.error());
        }
      }
    }
    if (entry["state"] != (args.mode == "preflight" ? "loadable" : "smoke_tested") || !entry["unloaded"].get<bool>()) successful = false;
    report["models"].push_back(std::move(entry));
  }
  if (artifacts) {
    if (!successful && !debug_error) debug_error = "debug_model";
    if (debug_error) successful = false;
    if (!successful) {
      const auto cleaned = artifacts->Rollback();
      if (!cleaned) debug_error = cleaned.error();
    }
    report["debug"] = {{"files", artifacts->Files()}, {"bytes", artifacts->Bytes()},
        {"retained", successful}, {"error", debug_error ? Json(debug_error) : Json(nullptr)}};
  }
  const auto body = report.dump();
  if (!WriteOutput(body)) {
    if (artifacts && !artifacts->Rollback()) std::fputs("Diagnostic artifact cleanup failed.\n", stderr);
    return 1;
  }
  return successful ? 0 : 1;
}
}  // namespace
int RunDiagnosticsCLI(int argc, char* argv[]) noexcept {
  // Explicit diagnostics exit before HTTP startup. Convert a closed stdout reader
  // into checked EPIPE output failure so owned artifacts can roll back normally.
#ifndef _WIN32
  DiagnosticPipeSignal pipe_signal;
  if (pipe_signal.previous == SIG_ERR) {
    std::fputs("Diagnostic output signal cannot be configured.\n", stderr);
    return 1;
  }
#endif
  std::optional<DebugArtifacts> artifacts;
  const int status = [&]() noexcept {
    try {
      if (argc == 2 && std::string_view(argv[1]) == "--help") {
        const std::string_view help{kHelp};
        return WriteOutput(help.substr(0, help.size() - 1)) ? 0 : 1;
      }
      Arguments arguments;
      if (!ParseArguments(argc, argv, arguments)) return Fatal("invalid_arguments", "Diagnostic arguments are invalid", 2);
      return Diagnose(arguments, artifacts);
    } catch (...) {
      return Fatal("diagnostic_failed", "Diagnostic operation failed");
    }
  }();
#ifndef _WIN32
  if (!pipe_signal.Restore()) {
    std::fputs("Diagnostic output signal cannot be restored.\n", stderr);
    return 1;
  }
#endif
  if (status == 0 && artifacts) artifacts->Retain();
  return status;
}
}  // namespace vision_simple
