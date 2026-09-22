#include "HTTPServer.h"

#include <hv/HttpServer.h>
#include <hv/hlog.h>
#include <hv/hv.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <format>
#include <magic_enum.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <string_view>

#include "HTTPExpectation.h"
#include "InferenceProtocol.h"
#include "LogFacade.h"
#include "Logger.h"
#include "MCPAdapter.h"
#include "OpenAIAdapter.h"
#include "SubtitleAdapter.h"
#include "TrackingAdapter.h"
#define LOG_DOMAIN_NAME "HTTPServer"
namespace vision_simple {
namespace {
struct InferRequest {
  std::string model;
  std::vector<std::string> images;
  std::optional<std::chrono::milliseconds> timeout;
};

enum class FailureStage {
  Request,
  ModelConfig,
  Serialization,
  LifecycleRequest,
  Pagination
};

struct RequestStage {
  FailureStage failure = FailureStage::Request;
  std::optional<size_t> image_index;
};

struct PreparedResponse {
  http_status status = HTTP_STATUS_OK;
  std::string body;
};

void LogFailure(std::string_view detail) noexcept {
  LogFacade::Error("http", detail);
}

PreparedResponse ServiceErrorResponse(const ServiceError& error) {
  const auto description = DescribeError(error.kind);
  // Fixed strings only: error output must survive serializer failures.
  return {static_cast<http_status>(description.status),
          std::format(
              R"({{"error":{{"code":"{}","message":"{}","image_index":{}}}}})",
              description.code, description.message,
              error.image_index ? std::to_string(*error.image_index) : "null")};
}
PreparedResponse ErrorResponse(const RequestStage& stage) {
  switch (stage.failure) {
    case FailureStage::Request:
      return ServiceErrorResponse({ServiceFailure::kInvalidRequest, {}});
    case FailureStage::ModelConfig:
      return ServiceErrorResponse({ServiceFailure::kModelConfig, {}});
    case FailureStage::Serialization:
      return ServiceErrorResponse({ServiceFailure::kInternal, {}});
    case FailureStage::LifecycleRequest:
      return {
          HTTP_STATUS_BAD_REQUEST,
          R"({"error":{"code":"invalid_request","message":"Request must contain a registered task kind and a nonempty model","image_index":null}})"};
    case FailureStage::Pagination:
      return {
          HTTP_STATUS_BAD_REQUEST,
          R"({"error":{"code":"invalid_request","message":"Limit must be 1 to 200 and offset must be nonnegative integers","image_index":null}})"};
  }
  return ServiceErrorResponse({ServiceFailure::kInternal, {}});
}
template <typename Handler>
int HandleRequest(const HttpContextPtr& ctx, RequestStage stage,
                  Handler&& handler) {
  PreparedResponse response;
  try {
    response = handler(stage);
  } catch (const std::exception& error) {
    LogFailure(error.what());
    response = ErrorResponse(stage);
  } catch (...) {
    LogFailure("Unknown HTTP request failure");
    response = ErrorResponse(stage);
  }
  // This is the only response owner. All processing and serialization finish
  // before libhv's send() ends the response; its returned status is not a
  // setter.
  ctx->setStatus(response.status);
  ctx->setContentType(APPLICATION_JSON);
  if (response.status == HTTP_STATUS_SERVICE_UNAVAILABLE)
    ctx->setHeader("Retry-After", "1");
  ctx->response->body = std::move(response.body);
  return ctx->send();
}

std::optional<InferRequest> ParseRequest(const std::string& body,
                                         size_t max_images) {
  const auto json = nlohmann::json::parse(body);
  if (!json.is_object()) return std::nullopt;
  const auto model = json.find("model");
  const auto images = json.find("images");
  if (model == json.end() || !model->is_string() ||
      model->get_ref<const std::string&>().empty() || images == json.end() ||
      !images->is_array() || images->size() > max_images) {
    return std::nullopt;
  }
  // Validate every field before model lookup, including later image entries.
  for (const auto& image : *images) {
    if (!image.is_string()) return std::nullopt;
  }
  std::optional<std::chrono::milliseconds> timeout;
  if (const auto it = json.find("timeout_ms"); it != json.end()) {
    if (!it->is_number_integer()) return std::nullopt;
    const auto value = it->get<int64_t>();
    if (value <= 0 || value > 300000) return std::nullopt;
    timeout = std::chrono::milliseconds(value);
  }
  return InferRequest{model->get<std::string>(),
                      images->get<std::vector<std::string>>(), timeout};
}
template <typename T>
bool ParseInteger(std::string_view text, T& value) {
  if (text.empty()) return false;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}
}  // namespace
class HTTPServerImpl : public HTTPServer {
  HTTPServerOptions options_;
  hv::HttpService http_service_;
  hv::HttpServer http_server_;
  std::shared_ptr<InferenceService> service_;
  std::unique_ptr<MCPAdapter> mcp_;
  std::unique_ptr<TrackingAdapter> tracking_;
  std::unique_ptr<SubtitleAdapter> subtitles_;

 public:
  HTTPServerImpl(HTTPServerOptions&& options,
                 std::shared_ptr<InferenceService> service,
                 std::unique_ptr<MCPAdapter> mcp,
                 std::unique_ptr<TrackingAdapter> tracking,
                 std::unique_ptr<SubtitleAdapter> subtitles)
      : options_(std::move(options)),
        service_(std::move(service)),
        mcp_(std::move(mcp)),
        tracking_(std::move(tracking)),
        subtitles_(std::move(subtitles)) {
    http_service_.Static(
        "/", options_.options.at(std::string(HTTPSERVER_OPT_KEY_STATIC_DIR))
                 .c_str());
    // register handles
    // /v0/infer/yolo
    http_service_.POST("/v0/infer/yolo", [this](const HttpContextPtr& ctx) {
      return this->HandleInferYOLO(ctx);
    });
    // /v0/infer/ocr
    http_service_.POST("/v0/infer/ocr", [this](const HttpContextPtr& ctx) {
      return this->HandleInferOCR(ctx);
    });
    for (const auto& task : RegisteredTasks()) {
      http_service_.POST(
          ("/v1/infer/" + std::string(task.id)).c_str(),
          [this, kind = task.kind](const HttpContextPtr& ctx,
                                   http_parser_state phase, const char* data,
                                   size_t size) {
            return HandleStreamingInfer(ctx, kind, phase, data, size);
          });
    }
    // /v0/infer/models
    http_service_.GET("/v0/infer/models", [this](const HttpContextPtr& ctx) {
      return this->HandleInferModels(ctx);
    });
    http_service_.POST("/v0/infer/unload", [this](const HttpContextPtr& ctx) {
      return HandleUnload(ctx);
    });
    http_service_.GET("/v0/infer/stats", [this](const HttpContextPtr& ctx) {
      return HandleStats(ctx);
    });
    http_service_.Use([](const HttpContextPtr& ctx) {
      Logger::Instance()->get().Info(
          LOG_DOMAIN_NAME,
          std::format("{}:{} -> {}", ctx->ip(), ctx->port(),
                      std::string_view(ctx->request->url)
                          .substr(0, ctx->request->url.find('?'))));
      return HTTP_STATUS_NEXT;
    });
    http_service_.AllowCORS();
    // libhv's CORS middleware answers OPTIONS before route handlers run.
    // Keep its v0 behavior, but never let it bypass MCP Origin validation.
    auto cors = std::move(http_service_.middleware.back().sync_handler);
    http_service_.middleware.back().sync_handler =
        [cors = std::move(cors)](HttpRequest* request, HttpResponse* response) {
          const auto path = std::string_view(request->path);
          if (path == "/mcp" || path.starts_with("/mcp/") ||
              path.starts_with("/mcp?"))
            return HTTP_STATUS_NEXT;
          return cors(request, response);
        };
    http_service_.enable_access_log = 0;

    http_server_.setHost(options_.host.c_str());
    http_server_.port = options_.port;
    http_server_.service = &http_service_;
    // Context handlers execute on libhv IO threads, not its async pool.
    // Multiple workers allow lifecycle routes to run alongside inference.
    http_server_.setThreadNum(4);
    logger_set_handler(
        hv_default_logger(), [](int log_level, const char* buf, int len) {
          auto msg = std::string(buf, len);
          msg.erase(std::remove(msg.begin(), msg.end(), '\r'), msg.end());
          msg.erase(std::remove(msg.begin(), msg.end(), '\n'), msg.end());
          auto logger_instance = Logger::Instance();
          if (log_level == LOG_LEVEL_DEBUG) {
            logger_instance->get().Debug(
                "libhv", msg.substr(msg.find_first_of("DEBUG") + 7));
          } else if (log_level == LOG_LEVEL_WARN) {
            logger_instance->get().Warn(
                "libhv", msg.substr(msg.find_first_of("WARN") + 6));
          } else if (log_level == LOG_LEVEL_ERROR) {
            logger_instance->get().Error(
                "libhv", msg.substr(msg.find_first_of("ERROR") + 7));
          } else if (log_level == LOG_LEVEL_FATAL) {
            logger_instance->get().Fatal(
                "libhv", msg.substr(msg.find_first_of("FATAL") + 7));
          } else if (log_level != LOG_LEVEL_SILENT) {
            logger_instance->get().Info(
                "libhv", msg.substr(msg.find_first_of("INFO") + 6));
          }
        });

    RegisterOpenAI(http_service_, service_);
    mcp_->Mount(http_service_, http_server_);
    tracking_->Mount(http_service_);
    subtitles_->Mount(http_service_);
  }
  ~HTTPServerImpl() override { Stop(); }
  const HTTPServerOptions& options() const noexcept override {
    return options_;
  }

  HTTPServerResult<void> Run() noexcept override { return Start(true); }

  HTTPServerResult<void> StartAsync() noexcept override { return Start(false); }

  void Stop() noexcept override {
    if (subtitles_) subtitles_->Stop();
    if (tracking_) tracking_->Stop();
    if (mcp_) mcp_->Stop();
    http_server_.stop();
  }
  HTTPServerResult<void> Start(bool wait) noexcept {
    try {
      const int result = http_server_run(&http_server_, wait ? 1 : 0);
      if (result != 0) {
        return MK_VSERROR(
            VisionSimpleErrorCode::kRuntimeError,
            std::format("unable to listen on {}:{} (libhv error {})",
                        http_server_.host, http_server_.port, result));
      }
      if (!wait) {
        LogFacade::Info("http",
                        std::format("listening on {}:{}", http_server_.host,
                                    http_server_.port));
      }
      return {};
    } catch (const std::exception& error) {
      return MK_VSERROR(
          VisionSimpleErrorCode::kRuntimeError,
          std::format("HTTP server startup failed: {}", error.what()));
    }
  }

  int HandleInferModels(const HttpContextPtr& ctx) {
    return HandleRequest(ctx, {FailureStage::ModelConfig, {}},
                         [this](RequestStage& stage) -> PreparedResponse {
                           auto catalog = service_->ListModels();
                           if (!catalog)
                             return ServiceErrorResponse(catalog.error());
                           stage = {FailureStage::Serialization, {}};
                           nlohmann::json legacy{
                               {"yolo", nlohmann::json::array()},
                               {"ocr", nlohmann::json::array()}};
                           for (const auto& model : catalog->models)
                             if (model.task == "yolo" || model.task == "ocr")
                               legacy[model.task].push_back(model.name);
                           return {HTTP_STATUS_OK, legacy.dump()};
                         });
  }
  int HandleUnload(const HttpContextPtr& ctx) {
    return HandleRequest(
        ctx, {FailureStage::LifecycleRequest, std::nullopt},
        [this, &ctx](RequestStage& stage) -> PreparedResponse {
          const auto body = nlohmann::json::parse(ctx->body());
          if (!body.is_object()) return ErrorResponse(stage);
          const auto kind = body.find("kind");
          const auto model = body.find("model");
          if (kind == body.end() || !kind->is_string() || model == body.end() ||
              !model->is_string() ||
              model->get_ref<const std::string&>().empty())
            return ErrorResponse(stage);
          const auto& kind_name = kind->get_ref<const std::string&>();
          const auto& name = model->get_ref<const std::string&>();
          const auto* task = FindTask(kind_name);
          if (!task) return ErrorResponse(stage);
          // Prepare the response before mutating the cache.
          stage = {FailureStage::Serialization, std::nullopt};
          PreparedResponse response{
              HTTP_STATUS_OK,
              nlohmann::json{
                  {"kind", kind_name}, {"model", name}, {"unloaded", true}}
                  .dump()};

          auto unloaded = service_->Unload(task->kind, name);
          if (!unloaded) return ServiceErrorResponse(unloaded.error());
          return response;
        });
  }
  int HandleStats(const HttpContextPtr& ctx) {
    return HandleRequest(
        ctx, {FailureStage::Pagination, std::nullopt},
        [this, &ctx](RequestStage& stage) -> PreparedResponse {
          size_t limit = 100, offset = 0;
          const auto& params = ctx->params();
          if (const auto it = params.find("limit"); it != params.end()) {
            if (!ParseInteger(it->second, limit) || limit == 0 || limit > 200)
              return ErrorResponse(stage);
          }
          if (const auto it = params.find("offset"); it != params.end()) {
            if (!ParseInteger(it->second, offset)) return ErrorResponse(stage);
          }
          stage = {FailureStage::Serialization, std::nullopt};

          auto stats = service_->Stats(limit, offset);
          if (!stats) return ServiceErrorResponse(stats.error());
          PreparedResponse response;
          struct_json::to_json(*stats, response.body);
          return response;
        });
  }
  int HandleInfer(const HttpContextPtr& ctx, InferenceKind kind) {
    const auto started = std::chrono::steady_clock::now();
    std::optional<InferenceResponse> inference;
    return HandleRequest(ctx, {}, [&](RequestStage& stage) -> PreparedResponse {
      auto request = ParseRequest(
          ctx->body(), service_->options().pipeline.max_batch_images);
      if (!request) return ErrorResponse(stage);
      auto result = service_->Run(
          kind, request->model, request->images,
          ServiceControl{.timeout = request->timeout, .started = started});
      if (!result) return ServiceErrorResponse(result.error());
      inference.emplace(std::move(*result));
      stage = {FailureStage::Serialization, {}};
      PreparedResponse response{HTTP_STATUS_OK, SerializeInference(*inference)};
      inference->Succeed();
      return response;
    });
  }
  int HandleStreamingInfer(const HttpContextPtr& ctx, InferenceKind kind,
                           http_parser_state phase, const char* data,
                           size_t size) {
    constexpr size_t kBodyLimit = 64 * 1024 * 1024;
    constexpr auto kBodyTooLarge = static_cast<http_status>(413);
    if (phase == HP_ERROR) return HTTP_STATUS_UNFINISHED;
    if (ctx->response->status_code >= 400) return ctx->response->status_code;
    const auto expectation = ctx->header("Expect");
    const bool continue_expected =
        ParseHTTPExpectation(expectation) == HTTPExpectation::kContinue;
    const auto reject = [&](http_status status, const char* code,
                            const char* message) {
      ctx->setStatus(status);
      ctx->setContentType(APPLICATION_JSON);
      ctx->response->body = nlohmann::json{{"error",
                                            {{"code", code},
                                             {"message", message},
                                             {"image_index", nullptr}}}}
                                .dump();
      if (status == kBodyTooLarge || !expectation.empty()) {
        ctx->setHeader("Connection", "close");
        return ctx->send();
      }
      return static_cast<int>(status);
    };
    if (phase == HP_HEADERS_COMPLETE) {
      if (!expectation.empty() && !continue_expected)
        return reject(static_cast<http_status>(417), "expectation_failed",
                      "Unsupported expectation");
      if (!ctx->is(APPLICATION_JSON))
        return reject(HTTP_STATUS_UNSUPPORTED_MEDIA_TYPE,
                      "unsupported_media_type", "Use application/json");
      const auto declared = ctx->header("Content-Length");
      uint64_t length = 0;
      if (!declared.empty() &&
          (!ParseInteger(declared, length) || length > kBodyLimit))
        return reject(kBodyTooLarge, "request_too_large",
                      "Request body exceeds 64 MiB");
      if (continue_expected)
        ctx->writer->write("HTTP/1.1 100 Continue\r\n\r\n");
    } else if (phase == HP_BODY) {
      if (size > kBodyLimit - ctx->request->body.size())
        return reject(kBodyTooLarge, "request_too_large",
                      "Request body exceeds 64 MiB");
      ctx->request->body.append(data, size);
    } else if (phase == HP_MESSAGE_COMPLETE) {
      return HandleInfer(ctx, kind);
    }
    return HTTP_STATUS_UNFINISHED;
  }
  int HandleInferYOLO(const HttpContextPtr& ctx) {
    return HandleInfer(ctx, InferenceKind::kYOLO);
  }
  int HandleInferOCR(const HttpContextPtr& ctx) {
    return HandleInfer(ctx, InferenceKind::kOCR);
  }
};
}  // namespace vision_simple
const std::string& vision_simple::HTTPServerOptions::OptionOrPut(
    const std::string& key, const std::string& default_value) {
  if (auto it = options.find(key); it != options.end()) {
    return it->second;
  }
  options.try_emplace(key, default_value);
  return options[key];
}

const std::string& vision_simple::HTTPServerOptions::OptionOrPut(
    std::string_view key, std::string_view default_value) {
  return OptionOrPut(std::string(key), std::string(default_value));
}

vision_simple::HTTPServerResult<std::unique_ptr<vision_simple::HTTPServer>>
vision_simple::HTTPServer::Create(HTTPServerOptions&& options) try {
  // libhv 1.3.3 setHost uses strcpy into host[64]; validate before calling it.
  if (options.host.empty() ||
      options.host.size() >= sizeof(http_server_t::host) ||
      options.host.find('\0') != std::string::npos) {
    return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                      "host must contain 1 to 63 bytes and no embedded NUL");
  }
  if (options.port == 0) {
    return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                      "port must be greater than zero");
  }
  options.OptionOrPut(HTTPSERVER_OPT_KEY_STATIC_DIR,
                      HTTPSERVER_OPT_DEFVAL_STATIC_DIR);
  const auto& device_text = options.OptionOrPut(
      HTTPSERVER_OPT_KEY_INFER_DEVICE, HTTPSERVER_OPT_DEFVAL_INFER_DEVICE);
  const auto& idle_text =
      options.OptionOrPut(HTTPSERVER_OPT_KEY_INFER_IDLE_TIMEOUT_MS,
                          HTTPSERVER_OPT_DEFVAL_INFER_IDLE_TIMEOUT_MS);
  const auto& sweep_text =
      options.OptionOrPut(HTTPSERVER_OPT_KEY_INFER_SWEEP_INTERVAL_MS,
                          HTTPSERVER_OPT_DEFVAL_INFER_SWEEP_INTERVAL_MS);
  int device_id = 0;
  uint64_t idle_ms = 0, sweep_ms = 0;
  // Bound conversions and steady-clock arithmetic, including wait deadlines.
  const auto max_ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::duration::max())
          .count() /
      2);
  if (!ParseInteger(device_text, device_id) || device_id < 0 ||
      !ParseInteger(idle_text, idle_ms) || idle_ms > max_ms ||
      !ParseInteger(sweep_text, sweep_ms) || sweep_ms == 0 ||
      sweep_ms > max_ms) {
    return MK_VSERROR(
        VisionSimpleErrorCode::kParameterError,
        "infer_device and infer_idle_timeout_ms must be nonnegative integers; "
        "infer_sweep_interval_ms must be a positive integer within clock "
        "range");
  }
  PipelineOptions pipeline_options;
  uint64_t timeout_ms = 0;
  size_t ocr_rec_batch_size = 1;
  if (!ParseInteger(
          options.OptionOrPut(HTTPSERVER_OPT_KEY_PIPELINE_CAPACITY,
                              HTTPSERVER_OPT_DEFVAL_PIPELINE_CAPACITY),
          pipeline_options.capacity) ||
      !ParseInteger(options.OptionOrPut(HTTPSERVER_OPT_KEY_PIPELINE_BATCHES,
                                        HTTPSERVER_OPT_DEFVAL_PIPELINE_BATCHES),
                    pipeline_options.max_batches) ||
      !ParseInteger(options.OptionOrPut(HTTPSERVER_OPT_KEY_MAX_BATCH_IMAGES,
                                        HTTPSERVER_OPT_DEFVAL_MAX_BATCH_IMAGES),
                    pipeline_options.max_batch_images) ||
      !ParseInteger(options.OptionOrPut(HTTPSERVER_OPT_KEY_TIMEOUT_MS,
                                        HTTPSERVER_OPT_DEFVAL_TIMEOUT_MS),
                    timeout_ms) ||
      !ParseInteger(
          options.OptionOrPut(HTTPSERVER_OPT_KEY_OCR_REC_BATCH_SIZE,
                              HTTPSERVER_OPT_DEFVAL_OCR_REC_BATCH_SIZE),
          ocr_rec_batch_size) ||
      ocr_rec_batch_size == 0 || ocr_rec_batch_size > 64 || timeout_ms == 0 ||
      timeout_ms > 300000) {
    return MK_VSERROR(
        VisionSimpleErrorCode::kParameterError,
        "pipeline limits must be integers and infer_timeout_ms "
        "must be 1 to 300000; ocr_rec_batch_size must be 1 to 64");
  }
  auto infer_fw_str =
      options.OptionOrPut(HTTPSERVER_OPT_KEY_INFER_FRAMEWORK,
                          HTTPSERVER_OPT_DEFVAL_INFER_FRAMEWORK);
  auto infer_ep_str = options.OptionOrPut(HTTPSERVER_OPT_KEY_INFER_EP,
                                          HTTPSERVER_OPT_DEFVAL_INFER_EP);
  auto infer_fw = magic_enum::enum_cast<InferFramework>(infer_fw_str);
  auto infer_ep = magic_enum::enum_cast<InferEP>(infer_ep_str);
  if (!infer_fw || !infer_ep)
    return std::unexpected(VisionSimpleError{
        VisionSimpleErrorCode::kParameterError,
        std::format("unsupported infer_framework:{} or infer_ep:{}",
                    infer_fw_str, infer_ep_str)});

  auto service = InferenceService::Create(InferenceServiceOptions{
      .framework = *infer_fw,
      .ep = *infer_ep,
      .device_id = device_id,
      .idle_timeout = std::chrono::milliseconds(idle_ms),
      .sweep_interval = std::chrono::milliseconds(sweep_ms),
      .pipeline = pipeline_options,
      .request_timeout = std::chrono::milliseconds(timeout_ms),
      .ocr_rec_batch_size = ocr_rec_batch_size});
  if (!service) return std::unexpected(std::move(service.error()));
  Logger::Instance()->get().Info(
      LOG_DOMAIN_NAME, std::format("Execution Provider:{}", infer_ep_str));
  auto mcp = MCPAdapter::Create(*service, options.host, options.port);
  if (!mcp) return std::unexpected(std::move(mcp.error()));
  auto tracking = TrackingAdapter::Create();
  if (!tracking) return std::unexpected(std::move(tracking.error()));
  auto subtitles = SubtitleAdapter::Create(*service);
  if (!subtitles) return std::unexpected(std::move(subtitles.error()));
  return std::make_unique<HTTPServerImpl>(
      std::move(options), std::move(*service), std::move(*mcp),
      std::move(*tracking), std::move(*subtitles));
} catch (const std::exception& error) {
  return MK_VSERROR(
      VisionSimpleErrorCode::kRuntimeError,
      std::format("unable to create HTTP server: {}", error.what()));
}
