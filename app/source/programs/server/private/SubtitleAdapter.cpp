#include "SubtitleAdapter.h"

#include <hv/HttpServer.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <limits>
#include <nlohmann/json.hpp>
#include <string_view>

#include "HTTPExpectation.h"
#include "SubtitleService.h"

namespace vision_simple {
namespace {
using Json = nlohmann::json;
constexpr size_t kJsonLimit = 64 * 1024;
constexpr size_t kUploadLimit = 64 * 1024 * 1024;
constexpr std::string_view kRoot = "/v1/subtitle/jobs";

bool TokenValid(std::string_view value) {
  return value.size() == 32 &&
         std::all_of(value.begin(), value.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}
bool Fields(const Json& value, std::initializer_list<std::string_view> fields) {
  if (!value.is_object()) return false;
  for (auto it = value.begin(); it != value.end(); ++it)
    if (std::find(fields.begin(), fields.end(), it.key()) == fields.end())
      return false;
  return true;
}
bool UnitNumber(const Json& value) {
  if (!value.is_number()) return false;
  const double number = value.get<double>();
  return (std::bit_cast<uint64_t>(number) & 0x7ff0000000000000ULL) !=
             0x7ff0000000000000ULL &&
         number >= 0 && number <= 1;
}
bool Options(const Json& body, SubtitleOptions& options) {
  if (!Fields(body, {"model", "sample_interval_ms", "roi", "min_confidence",
                     "stable_samples", "gap_samples"}) ||
      !body.contains("model") || !body["model"].is_string())
    return false;
  options.model = body["model"].get<std::string>();
  for (auto [name, target] :
       {std::pair{"sample_interval_ms", &options.sample_interval_ms},
        {"stable_samples", &options.stable_samples},
        {"gap_samples", &options.gap_samples}}) {
    if (!body.contains(name)) continue;
    const auto& value = body[name];
    if (!value.is_number_integer() ||
        (!value.is_number_unsigned() && value.get<int64_t>() < 1) ||
        value.get<uint64_t>() < 1 ||
        value.get<uint64_t>() > std::numeric_limits<unsigned>::max())
      return false;
    *target = value.get<unsigned>();
  }
  if (body.contains("min_confidence")) {
    if (!UnitNumber(body["min_confidence"])) return false;
    options.min_confidence = body["min_confidence"].get<double>();
  }
  if (body.contains("roi")) {
    const auto& roi = body["roi"];
    if (!roi.is_array() || roi.size() != 4) return false;
    for (size_t i = 0; i < 4; ++i) {
      if (!UnitNumber(roi[i])) return false;
      options.roi[i] = roi[i].get<double>();
    }
  }
  return true;
}
const char* StateName(SubtitleJobState state) {
  switch (state) {
    case SubtitleJobState::kCreated:
      return "created";
    case SubtitleJobState::kUploading:
      return "uploading";
    case SubtitleJobState::kQueued:
      return "queued";
    case SubtitleJobState::kRunning:
      return "running";
    case SubtitleJobState::kCancelling:
      return "cancelling";
    case SubtitleJobState::kCancelled:
      return "cancelled";
    case SubtitleJobState::kCompleted:
      return "completed";
    case SubtitleJobState::kFailed:
      return "failed";
  }
  return "failed";
}
Json Info(const SubtitleJobInfo& info) {
  return {
      {"id", info.id},
      {"state", StateName(info.state)},
      {"uploaded_bytes", info.uploaded_bytes},
      {"decoded_frames", info.decoded_frames},
      {"sampled_frames", info.sampled_frames},
      {"position_ms", info.position_ms},
      {"duration_ms",
       info.duration_ms ? Json(*info.duration_ms) : Json(nullptr)},
      {"cue_count", info.cue_count},
      {"error_code", info.error_code ? Json(*info.error_code) : Json(nullptr)}};
}
int Send(const HttpContextPtr& ctx, int status, std::string body,
         bool finish = true) {
  ctx->response->status_code = static_cast<http_status>(status);
  ctx->setContentType(APPLICATION_JSON);
  ctx->setHeader("Cache-Control", "no-store");
  ctx->response->body = std::move(body);
  return finish ? ctx->send() : status;
}
int Error(const HttpContextPtr& ctx, int status, const char* code,
          const char* message, bool finish = true) {
  if (status == 503) ctx->setHeader("Retry-After", "1");
  return Send(
      ctx, status,
      Json{{"error",
            {{"code", code}, {"message", message}, {"image_index", nullptr}}}}
          .dump(),
      finish);
}
int Error(const HttpContextPtr& ctx, SubtitleFailure failure,
          bool finish = true) {
  switch (failure) {
    case SubtitleFailure::kInvalid:
      return Error(ctx, 400, "invalid_request", "Invalid subtitle request",
                   finish);
    case SubtitleFailure::kMissing:
      return Error(ctx, 404, "subtitle_job_not_found", "Subtitle job not found",
                   finish);
    case SubtitleFailure::kUnknownModel:
      return Error(ctx, 404, "model_not_found", "OCR model not found", finish);
    case SubtitleFailure::kBusy:
      return Error(ctx, 409, "subtitle_job_busy", "Subtitle job is busy",
                   finish);
    case SubtitleFailure::kNotReady:
      return Error(ctx, 409, "subtitle_not_ready", "Subtitles are not ready",
                   finish);
    case SubtitleFailure::kTooLarge:
      return Error(ctx, 413, "payload_too_large",
                   "Subtitle resource limit exceeded", finish);
    case SubtitleFailure::kInvalidVideo:
      return Error(ctx, 415, "invalid_video", "Unsupported or invalid video",
                   finish);
    case SubtitleFailure::kCapacity:
      return Error(ctx, 503, "subtitle_capacity",
                   "Subtitle job capacity reached", finish);
    case SubtitleFailure::kClosed:
      return Error(ctx, 503, "service_unavailable",
                   "Subtitle service is stopping", finish);
    case SubtitleFailure::kFailed:
      return Error(ctx, 500, "subtitle_failed", "Subtitle operation failed",
                   finish);
  }
  return Error(ctx, 500, "subtitle_failed", "Subtitle operation failed",
               finish);
}

// An aliasing request pointer makes this guard live exactly as long as the
// context's request, including disconnect and exception paths. The original
// request is retained without a context/guard reference cycle.
struct RequestState {
  HttpRequestPtr original;
  std::shared_ptr<SubtitleService> service;
  std::string id;
  size_t received = 0;
  bool uploading = false;
  ~RequestState() { Abort(); }
  void Abort() noexcept {
    if (uploading) {
      uploading = false;
      service->AbortUpload(id);
    }
  }
};
}  // namespace

struct SubtitleAdapter::State {
  std::shared_ptr<SubtitleService> service;
  enum class Route {
    kCreate,
    kList,
    kGet,
    kUpload,
    kCancel,
    kDelete,
    kSrt,
    kVtt
  };

  int Receive(const HttpContextPtr& ctx, Route route, http_parser_state phase,
              const char* data, size_t size) {
    auto* request = static_cast<RequestState*>(ctx->userdata);
    if (phase == HP_ERROR) {
      if (request) request->Abort();
      return HTTP_STATUS_UNFINISHED;
    }
    if (ctx->response->status_code >= 400) return ctx->response->status_code;
    try {
      const bool upload = route == Route::kUpload;
      const bool json = route == Route::kCreate || route == Route::kCancel;
      const bool has_body = upload || json;
      const size_t limit = upload ? kUploadLimit : kJsonLimit;
      const auto expectation = ctx->header("Expect");
      const auto reject = [&](int status, const char* code,
                              const char* message) {
        if (request) request->Abort();
        const bool immediate =
            !has_body || status == 413 || !expectation.empty();
        if (immediate) ctx->setHeader("Connection", "close");
        return Error(ctx, status, code, message, immediate);
      };
      const auto reject_failure = [&](SubtitleFailure failure) {
        if (request) request->Abort();
        const bool immediate = !has_body ||
                               failure == SubtitleFailure::kTooLarge ||
                               !expectation.empty();
        if (immediate) ctx->setHeader("Connection", "close");
        return Error(ctx, failure, immediate);
      };
      if (phase == HP_HEADERS_COMPLETE) {
        auto owned = std::make_shared<RequestState>();
        owned->original = ctx->request;
        owned->service = service;
        owned->id = ctx->param("id");
        request = owned.get();
        ctx->request = HttpRequestPtr(owned, owned->original.get());
        ctx->userdata = request;
        if (!ctx->header("Origin").empty())
          return reject(403, "invalid_request",
                        "Browser origins are not accepted");
        const auto expected = ParseHTTPExpectation(expectation);
        if (expected == HTTPExpectation::kUnsupported ||
            (!has_body &&
             ctx->headers().find("Expect") != ctx->headers().end()))
          return reject(417, "invalid_request", "Unsupported expectation");
        uint64_t length = 0;
        const auto declared = ctx->headers().find("Content-Length");
        if (declared != ctx->headers().end()) {
          const auto& value = declared->second;
          const auto parsed = std::from_chars(
              value.data(), value.data() + value.size(), length);
          if (parsed.ec != std::errc{} ||
              parsed.ptr != value.data() + value.size())
            return reject(400, "invalid_request", "Invalid Content-Length");
        }
        if (!has_body) {
          if (length ||
              ctx->headers().find("Transfer-Encoding") != ctx->headers().end())
            return reject(400, "invalid_request",
                          "This request must not have a body");
        } else {
          if (length > limit)
            return reject(413, "payload_too_large",
                          "Request body exceeds its limit");
          if (ctx->header("Content-Type") !=
              (upload ? "application/octet-stream" : "application/json"))
            return reject(415, "unsupported_media_type",
                          "Unsupported Content-Type");
        }
        if (route != Route::kCreate && route != Route::kList &&
            !TokenValid(request->id))
          return reject_failure(SubtitleFailure::kInvalid);
        if (upload) {
          auto result = service->BeginUpload(request->id);
          if (!result) return reject_failure(result.error());
          request->uploading = true;
        }
        if (expected == HTTPExpectation::kContinue)
          ctx->writer->write("HTTP/1.1 100 Continue\r\n\r\n");
      } else if (phase == HP_BODY) {
        if (!request) return reject_failure(SubtitleFailure::kFailed);
        if (!has_body && size)
          return reject(400, "invalid_request",
                        "This request must not have a body");
        if (request->received > limit || size > limit - request->received)
          return reject(413, "payload_too_large",
                        "Request body exceeds its limit");
        request->received += size;
        if (upload) {
          if (!request->uploading)
            return reject_failure(SubtitleFailure::kBusy);
          auto result = service->AppendUpload(
              request->id, std::span<const char>(data, size));
          if (!result) return reject_failure(result.error());
        } else if (json) {
          ctx->request->body.append(data, size);
        }
      } else if (phase == HP_MESSAGE_COMPLETE) {
        if (!request) return reject_failure(SubtitleFailure::kFailed);
        if (upload) {
          if (!request->uploading)
            return reject_failure(SubtitleFailure::kBusy);
          auto result = service->FinishUpload(request->id);
          if (!result) return reject_failure(result.error());
          request->uploading = false;
          return Status(ctx, request->id, 202);
        }
        return Handle(ctx, route, request->id);
      }
      return HTTP_STATUS_UNFINISHED;
    } catch (...) {
      if (request) request->Abort();
      ctx->setHeader("Connection", "close");
      return Error(ctx, SubtitleFailure::kFailed);
    }
  }

  int Status(const HttpContextPtr& ctx, const std::string& id, int status) {
    auto result = service->Get(id);
    return result ? Send(ctx, status, Info(*result).dump())
                  : Error(ctx, result.error());
  }
  int Handle(const HttpContextPtr& ctx, Route route, const std::string& id) {
    if (route == Route::kCreate || route == Route::kCancel) {
      const auto body = Json::parse(ctx->body(), nullptr, false);
      if (body.is_discarded()) return Error(ctx, SubtitleFailure::kInvalid);
      if (route == Route::kCreate) {
        SubtitleOptions options;
        if (!Options(body, options))
          return Error(ctx, SubtitleFailure::kInvalid);
        auto result = service->Add(std::move(options));
        if (!result) return Error(ctx, result.error());
        ctx->setHeader("Location", std::string(kRoot) + "/" + result->id);
        return Send(ctx, 201, Info(*result).dump());
      }
      if (!body.is_object() || !body.empty())
        return Error(ctx, SubtitleFailure::kInvalid);
      auto result = service->Cancel(id);
      return result ? Status(ctx, id, 202) : Error(ctx, result.error());
    }
    if (route == Route::kList) {
      size_t limit = 100;
      const auto limit_it = ctx->params().find("limit");
      if (limit_it != ctx->params().end()) {
        const auto& text = limit_it->second;
        const auto parsed =
            std::from_chars(text.data(), text.data() + text.size(), limit);
        if (parsed.ec != std::errc{} ||
            parsed.ptr != text.data() + text.size() || limit < 1 || limit > 100)
          return Error(ctx, SubtitleFailure::kInvalid);
      }
      std::string cursor;
      const auto cursor_it = ctx->params().find("cursor");
      if (cursor_it != ctx->params().end()) {
        const auto& text = cursor_it->second;
        if (!text.starts_with("s1.") ||
            !TokenValid(std::string_view(text).substr(3)))
          return Error(ctx, SubtitleFailure::kInvalid);
        cursor = text.substr(3);
      }
      auto result = service->List(limit, cursor);
      if (!result) return Error(ctx, result.error());
      Json jobs = Json::array();
      for (const auto& job : result->jobs) jobs.push_back(Info(job));
      return Send(ctx, 200,
                  Json{{"jobs", std::move(jobs)},
                       {"next_cursor", result->next_cursor.empty()
                                           ? Json(nullptr)
                                           : Json("s1." + result->next_cursor)}}
                      .dump());
    }
    if (route == Route::kGet) return Status(ctx, id, 200);
    if (route == Route::kDelete) {
      auto result = service->Delete(id);
      return result ? Send(ctx, 204, {}) : Error(ctx, result.error());
    }
    const bool webvtt = route == Route::kVtt;
    auto result = service->Download(id, webvtt);
    if (!result) return Error(ctx, result.error());
    ctx->response->status_code = HTTP_STATUS_OK;
    ctx->setHeader("Cache-Control", "no-store");
    ctx->setHeader("Content-Type", webvtt ? "text/vtt; charset=utf-8"
                                          : "text/plain; charset=utf-8");
    ctx->setHeader("Content-Disposition", "attachment; filename=\"" + id +
                                              (webvtt ? ".vtt\"" : ".srt\""));
    ctx->response->body = std::move(*result);
    return ctx->send();
  }
};

SubtitleAdapter::SubtitleAdapter(std::shared_ptr<State> state)
    : state_(std::move(state)) {}
VSResult<std::unique_ptr<SubtitleAdapter>> SubtitleAdapter::Create(
    std::shared_ptr<InferenceService> inference) noexcept {
  try {
    auto service = SubtitleService::Create(std::move(inference));
    if (!service)
      return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError,
                        "Unable to create subtitle service");
    auto state = std::make_shared<State>();
    state->service = std::move(*service);
    return std::unique_ptr<SubtitleAdapter>(
        new SubtitleAdapter(std::move(state)));
  } catch (...) {
    return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError,
                      "Unable to create subtitle service");
  }
}
SubtitleAdapter::~SubtitleAdapter() { Stop(); }
void SubtitleAdapter::Mount(hv::HttpService& service) {
  const auto handler = [state = state_](State::Route route) {
    return [state, route](const HttpContextPtr& ctx, http_parser_state phase,
                          const char* data, size_t size) {
      return state->Receive(ctx, route, phase, data, size);
    };
  };
  service.POST("/v1/subtitle/jobs", handler(State::Route::kCreate));
  service.GET("/v1/subtitle/jobs", handler(State::Route::kList));
  service.GET("/v1/subtitle/jobs/:id", handler(State::Route::kGet));
  service.PUT("/v1/subtitle/jobs/:id/video", handler(State::Route::kUpload));
  service.POST("/v1/subtitle/jobs/:id/cancel", handler(State::Route::kCancel));
  service.Delete("/v1/subtitle/jobs/:id", handler(State::Route::kDelete));
  service.GET("/v1/subtitle/jobs/:id/subtitles.srt",
              handler(State::Route::kSrt));
  service.GET("/v1/subtitle/jobs/:id/subtitles.vtt",
              handler(State::Route::kVtt));
}
void SubtitleAdapter::Stop() noexcept {
  if (state_) state_->service->Stop();
}
}  // namespace vision_simple
