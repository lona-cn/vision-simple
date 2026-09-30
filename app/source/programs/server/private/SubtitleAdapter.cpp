#include "SubtitleAdapter.h"

#include <hv/HttpServer.h>
#include <hv/EventLoop.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string_view>

#include "HTTPExpectation.h"
#include "HttpDispatch.h"
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

// Own parser chunks, never the parser buffer or a socket. File operations
// belong to the accepted dispatch worker, including physical abort/close.
struct RequestState {
  HttpRequestPtr original;
  std::string id;
  size_t received = 0;
  std::mutex mutex;
  std::condition_variable ready;
  std::deque<std::string> chunks;
  size_t buffered = 0;
  bool done = false;
  bool cancelled = false;
  bool resume_pending = false;
  // Intrusive membership exists only while a data worker owns Upload.
  RequestState* next_upload = nullptr;
  void Abort() noexcept {
    std::lock_guard lock(mutex);
    cancelled = true;
    ready.notify_all();
  }
};
}  // namespace

struct SubtitleAdapter::State : std::enable_shared_from_this<SubtitleAdapter::State> {
  std::shared_ptr<SubtitleService> service;
  HttpDispatch* dispatch = nullptr;
  std::mutex uploads_mutex;
  RequestState* uploads = nullptr;
  void AbortUploads(const std::string& id) noexcept {
    // Caller holds uploads_mutex; entries are bounded by data worker count.
    for (auto* request = uploads; request; request = request->next_upload)
      if (request->id == id) request->Abort();
  }
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
  HttpDispatch::Reply Upload(const std::shared_ptr<RequestState>& request,
                             hv::EventLoop* loop, std::weak_ptr<hv::HttpContext> weak,
                             bool expect_continue, std::stop_token stop) {
    // Cancellation returns no Reply: after this worker physically drains its
    // file/buffers, HttpDispatch finalizes the incomplete socket on its IO loop.
    std::stop_callback cancel(stop, [request] { request->Abort(); });
    bool active = false;
    struct Cleanup {
      State& owner;
      std::shared_ptr<RequestState> request;
      bool& active;
      ~Cleanup() {
        {
          std::unique_lock lock(request->mutex);
          request->cancelled = true;
          request->ready.wait(lock, [&] { return !request->resume_pending; });
          request->chunks.clear();
          request->buffered = 0;
        }
        std::lock_guard lock(owner.uploads_mutex);
        if (active) owner.service->AbortUpload(request->id);
        auto** entry = &owner.uploads;
        while (*entry && *entry != request.get()) entry = &(*entry)->next_upload;
        if (*entry) *entry = request->next_upload;
      }
    } cleanup{*this, request, active};
    {
      std::lock_guard lock(request->mutex);
      if (request->cancelled) return {};
    }
    // Serialize BeginUpload plus registration with public cancel/delete. A
    // queued upload cancelled before this point fails BeginUpload instead.
    auto begin = [&] {
      std::lock_guard lock(uploads_mutex);
      auto result = service->BeginUpload(request->id);
      if (result) {
        request->next_upload = uploads;
        uploads = request.get();
      }
      return result;
    }();
    if (!begin) return [failure = begin.error()](const HttpContextPtr& ctx) {
      if (ctx->response->status_code >= 400) return;
      ctx->setHeader("Connection", "close");
      Error(ctx, failure);
    };
    active = true;
    const auto resume = [&](bool send_continue) {
      {
        std::lock_guard lock(request->mutex);
        if (request->cancelled || request->done || request->resume_pending) return;
        request->resume_pending = true;
      }
      try {
        loop->queueInLoop([weak, request, send_continue] {
          auto ctx = weak.lock();
          bool resume_read = false;
          {
            std::lock_guard lock(request->mutex);
            request->resume_pending = false;
            resume_read = !request->cancelled && !request->done;
            request->ready.notify_all();
          }
          if (!ctx || !resume_read || !ctx->writer->isConnected() ||
              ctx->response->status_code >= 400) return;
          if (send_continue) ctx->writer->write("HTTP/1.1 100 Continue\r\n\r\n");
          hio_read_start(ctx->writer->io());
        });
      } catch (...) {
        std::lock_guard lock(request->mutex);
        request->resume_pending = false;
        request->ready.notify_all();
        throw;
      }
    };
    resume(expect_continue);
    for (;;) {
      std::string chunk;
      {
        std::unique_lock lock(request->mutex);
        request->ready.wait(lock, [&] {
          return request->cancelled || request->done || !request->chunks.empty();
        });
        if (request->cancelled) return {};
        if (request->chunks.empty()) break;
        chunk = std::move(request->chunks.front());
        request->chunks.pop_front();
      }
      auto appended = service->AppendUpload(request->id,
                                           std::span<const char>(chunk.data(), chunk.size()));
      {
        std::lock_guard lock(request->mutex);
        request->buffered -= chunk.size();
      }
      if (!appended) return [failure = appended.error()](const HttpContextPtr& ctx) {
        if (ctx->response->status_code >= 400) return;
        ctx->setHeader("Connection", "close");
        Error(ctx, failure);
      };
      resume(false);
    }
    if (stop.stop_requested()) return {};
    auto finished = service->FinishUpload(request->id);
    if (finished) active = false;
    auto reply = finished ? Status(request->id, 202) : Failure(finished.error());
    return [reply = std::move(reply)](const HttpContextPtr& ctx) mutable {
      if (ctx->response->status_code >= 400) return;
      reply(ctx);
      if (ctx->writer->isConnected()) hio_read_start(ctx->writer->io());
    };
  }

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
          auto* loop = hv::tlsEventLoop();
          if (!loop || loop->loop() != hevent_loop(ctx->writer->io()))
            return reject(503, "service_overloaded", "HTTP dispatch unavailable");
          hio_read_stop(ctx->writer->io());
          auto self = shared_from_this();
          if (!dispatch->Submit(ctx, HttpDispatch::Lane::kData,
              [self, owned, loop, weak = std::weak_ptr<hv::HttpContext>(ctx),
               expect_continue = expected == HTTPExpectation::kContinue]
              (std::stop_token stop, HttpDispatch::Clock::time_point) {
                return self->Upload(owned, loop, weak, expect_continue, stop);
              }, false)) {
            ctx->setHeader("Connection", "close");
            return Error(ctx, 503, "service_overloaded", "HTTP dispatch capacity reached");
          }
        } else if (expected == HTTPExpectation::kContinue) {
          ctx->writer->write("HTTP/1.1 100 Continue\r\n\r\n");
        }
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
          hio_read_stop(ctx->writer->io());
          bool full = false;
          {
            std::lock_guard lock(request->mutex);
            full = size > 1024 * 1024 - request->buffered;
            if (!full && !request->cancelled) {
              request->chunks.emplace_back(data, size);
              request->buffered += size;
              request->ready.notify_one();
            }
          }
          if (full) {
            ctx->setHeader("Connection", "close");
            request->Abort();
            return Error(ctx, 503, "service_overloaded", "Upload buffer capacity reached");
          }
        } else if (json) {
          ctx->request->body.append(data, size);
        }
      } else if (phase == HP_MESSAGE_COMPLETE) {
        if (!request) return reject_failure(SubtitleFailure::kFailed);
        if (upload) {
          std::lock_guard lock(request->mutex);
          request->done = true;
          request->ready.notify_one();
          return HTTP_STATUS_UNFINISHED;
        }
        auto self = shared_from_this();
        if (!dispatch->Submit(ctx, HttpDispatch::Lane::kControl,
            [self, route, id = request->id, body = std::move(ctx->request->body),
             params = ctx->params()](std::stop_token stop, HttpDispatch::Clock::time_point) mutable {
              if (stop.stop_requested()) return HttpDispatch::Reply{};
              return self->Handle(route, id, std::move(body), params);
            })) return Error(ctx, 503, "service_overloaded", "HTTP dispatch capacity reached");
        return HTTP_STATUS_UNFINISHED;
      }
      return HTTP_STATUS_UNFINISHED;
    } catch (...) {
      if (request) request->Abort();
      ctx->setHeader("Connection", "close");
      return Error(ctx, SubtitleFailure::kFailed);
    }
  }

  using Reply = HttpDispatch::Reply;
  static Reply Failure(SubtitleFailure failure) {
    return [failure](const HttpContextPtr& ctx) { Error(ctx, failure); };
  }
  static Reply Response(int status, std::string body) {
    return [status, body = std::move(body)](const HttpContextPtr& ctx) mutable {
      Send(ctx, status, std::move(body));
    };
  }
  Reply Status(const std::string& id, int status) {
    auto result = service->Get(id);
    return result ? Response(status, Info(*result).dump())
                  : Failure(result.error());
  }
  Reply Handle(Route route, const std::string& id, std::string body_text,
               const hv::QueryParams& params) {
    if (route == Route::kCreate || route == Route::kCancel) {
      const auto body = Json::parse(body_text, nullptr, false);
      if (body.is_discarded()) return Failure(SubtitleFailure::kInvalid);
      if (route == Route::kCreate) {
        SubtitleOptions options;
        if (!Options(body, options)) return Failure(SubtitleFailure::kInvalid);
        auto result = service->Add(std::move(options));
        if (!result) return Failure(result.error());
        return [location = std::string(kRoot) + "/" + result->id,
                body = Info(*result).dump()](const HttpContextPtr& ctx) mutable {
          ctx->setHeader("Location", location);
          Send(ctx, 201, std::move(body));
        };
      }
      if (!body.is_object() || !body.empty())
        return Failure(SubtitleFailure::kInvalid);
      std::lock_guard lock(uploads_mutex);
      auto result = service->Cancel(id);
      if (result) AbortUploads(id);
      return result ? Status(id, 202) : Failure(result.error());
    }
    if (route == Route::kList) {
      size_t limit = 100;
      const auto limit_it = params.find("limit");
      if (limit_it != params.end()) {
        const auto& text = limit_it->second;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), limit);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
            limit < 1 || limit > 100) return Failure(SubtitleFailure::kInvalid);
      }
      std::string cursor;
      const auto cursor_it = params.find("cursor");
      if (cursor_it != params.end()) {
        const auto& text = cursor_it->second;
        if (!text.starts_with("s1.") || !TokenValid(std::string_view(text).substr(3)))
          return Failure(SubtitleFailure::kInvalid);
        cursor = text.substr(3);
      }
      auto result = service->List(limit, cursor);
      if (!result) return Failure(result.error());
      Json jobs = Json::array();
      for (const auto& job : result->jobs) jobs.push_back(Info(job));
      return Response(200, Json{{"jobs", std::move(jobs)},
          {"next_cursor", result->next_cursor.empty() ? Json(nullptr)
                                                    : Json("s1." + result->next_cursor)}}.dump());
    }
    if (route == Route::kGet) return Status(id, 200);
    if (route == Route::kDelete) {
      std::lock_guard lock(uploads_mutex);
      auto result = service->Delete(id);
      if (result) AbortUploads(id);
      return result ? Response(204, {}) : Failure(result.error());
    }
    const bool webvtt = route == Route::kVtt;
    auto result = service->Download(id, webvtt);
    if (!result) return Failure(result.error());
    return [webvtt, id, body = std::move(*result)](const HttpContextPtr& ctx) mutable {
      ctx->response->status_code = HTTP_STATUS_OK;
      ctx->setHeader("Cache-Control", "no-store");
      ctx->setHeader("Content-Type", webvtt ? "text/vtt; charset=utf-8" : "text/plain; charset=utf-8");
      ctx->setHeader("Content-Disposition", "attachment; filename=\"" + id +
                                              (webvtt ? ".vtt\"" : ".srt\""));
      ctx->response->body = std::move(body);
      ctx->send();
    };
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
void SubtitleAdapter::Mount(hv::HttpService& service, HttpDispatch& dispatch) {
  state_->dispatch = &dispatch;
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
