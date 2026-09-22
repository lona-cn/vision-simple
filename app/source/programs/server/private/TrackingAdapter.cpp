#include "TrackingAdapter.h"

#include <hv/HttpServer.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <random>
#include <string_view>

#include "HTTPExpectation.h"
#include "TrackingService.h"

namespace vision_simple {
namespace {
using Json = nlohmann::json;
constexpr size_t kBodyLimit = 4 * 1024 * 1024;
constexpr std::string_view kRoot = "/v1/tracking/sessions";
std::string Token() {
  std::random_device random;
  constexpr char hex[] = "0123456789abcdef";
  std::string token;
  token.reserve(32);
  for (int i = 0; i < 4; ++i) {
    const auto value = random();
    for (int shift = 0; shift < 32; shift += 4)
      token += hex[(value >> shift) & 15];
  }
  return token;
}
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
bool Unsigned(const Json& value, uint64_t low, uint64_t high) {
  if (!value.is_number_integer()) return false;
  if (!value.is_number_unsigned() && value.get<int64_t>() < 0) return false;
  const auto n = value.get<uint64_t>();
  return n >= low && n <= high;
}
bool Number(const Json& value, double low, double high) {
  if (!value.is_number()) return false;
  const double n = value.get<double>();
  return (std::bit_cast<uint64_t>(n) & 0x7ff0000000000000ULL) !=
             0x7ff0000000000000ULL &&
         n >= low && n <= high;
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
int Error(const HttpContextPtr& ctx, TrackingFailure failure) {
  switch (failure) {
    case TrackingFailure::kInvalid:
      return Error(ctx, 400, "invalid_request", "Invalid tracking request");
    case TrackingFailure::kInvalidImage:
      return Error(ctx, 400, "invalid_image", "Invalid encoded image");
    case TrackingFailure::kMissing:
      return Error(ctx, 404, "tracking_session_not_found",
                   "Tracking session not found");
    case TrackingFailure::kBusy:
      return Error(ctx, 409, "tracking_session_busy",
                   "Tracking session is busy");
    case TrackingFailure::kOrder:
      return Error(ctx, 409, "frame_out_of_order",
                   "Frame index and timestamp must increase strictly");
    case TrackingFailure::kCapacity:
      return Error(ctx, 503, "tracking_capacity",
                   "Tracking session capacity reached");
    case TrackingFailure::kClosed:
      return Error(ctx, 503, "service_unavailable",
                   "Tracking service is stopping");
    default:
      return Error(ctx, 500, "tracking_failed", "Tracking failed");
  }
}
const char* Algorithm(TrackerAlgorithm algorithm) {
  return algorithm == TrackerAlgorithm::kByteTrack ? "bytetrack" : "botsort";
}
Json Status(const TrackingStatus& status) {
  return {
      {"last_frame_index", status.last_frame_index
                               ? Json(*status.last_frame_index)
                               : Json(nullptr)},
      {"last_timestamp",
       status.last_timestamp ? Json(*status.last_timestamp) : Json(nullptr)},
      {"active_tracks", status.active_tracks},
      {"lost_tracks", status.lost_tracks}};
}
bool Options(const Json& body, TrackerOptions& options) {
  if (!Fields(body, {"algorithm", "options"}) || !body.contains("algorithm") ||
      !body["algorithm"].is_string())
    return false;
  const auto algorithm = body["algorithm"].get<std::string>();
  if (algorithm == "bytetrack")
    options.algorithm = TrackerAlgorithm::kByteTrack;
  else if (algorithm == "botsort")
    options.algorithm = TrackerAlgorithm::kBoTSORT;
  else
    return false;
  if (!body.contains("options")) return true;
  const auto& fields = body["options"];
  if (!Fields(fields,
              {"high_threshold", "low_threshold", "new_track_threshold",
               "match_threshold", "max_lost_frames", "min_hits", "max_tracks",
               "max_detections", "camera_motion", "appearance",
               "proximity_threshold", "appearance_threshold"}))
    return false;
  for (auto [name, target] :
       {std::pair{"high_threshold", &options.high_threshold},
        {"low_threshold", &options.low_threshold},
        {"new_track_threshold", &options.new_track_threshold},
        {"match_threshold", &options.match_threshold},
        {"proximity_threshold", &options.proximity_threshold},
        {"appearance_threshold", &options.appearance_threshold}}) {
    if (fields.contains(name)) {
      if (!Number(fields[name], 0, 1)) return false;
      *target = fields[name].get<float>();
    }
  }
  for (auto [name, target] :
       {std::pair{"camera_motion", &options.camera_motion},
        {"appearance", &options.appearance}}) {
    if (fields.contains(name)) {
      if (!fields[name].is_boolean()) return false;
      *target = fields[name].get<bool>();
    }
  }
  for (auto [name, target] :
       {std::pair{"max_lost_frames", &options.max_lost_frames},
        {"min_hits", &options.min_hits}}) {
    if (fields.contains(name)) {
      if (!Unsigned(fields[name], name == std::string_view("min_hits") ? 1 : 0,
                    UINT32_MAX))
        return false;
      *target = fields[name].get<uint32_t>();
    }
  }
  for (auto [name, target] : {std::pair{"max_tracks", &options.max_tracks},
                              {"max_detections", &options.max_detections}}) {
    if (fields.contains(name)) {
      if (!Unsigned(fields[name], 1, 256)) return false;
      *target = fields[name].get<size_t>();
    }
  }
  return true;
}
struct Frame {
  uint64_t index;
  double timestamp;
  std::vector<TrackingDetection> detections;
};
bool ParseFrame(const Json& body, Frame& frame) {
  if (!Fields(body, {"frame_index", "timestamp", "detections", "image"}) ||
      !body.contains("frame_index") ||
      !Unsigned(body["frame_index"], 0, 9007199254740991ULL) ||
      !body.contains("timestamp") ||
      !Number(body["timestamp"], 0, std::numeric_limits<double>::max()) ||
      !body.contains("detections") || !body["detections"].is_array() ||
      body["detections"].size() > 256 ||
      (body.contains("image") &&
       (!body["image"].is_string() ||
        body["image"].get_ref<const std::string&>().size() > kBodyLimit)))
    return false;
  frame.index = body["frame_index"].get<uint64_t>();
  frame.timestamp = body["timestamp"].get<double>();
  frame.detections.reserve(body["detections"].size());
  for (const auto& item : body["detections"]) {
    if (!Fields(item, {"class_id", "confidence", "bbox", "embedding"}) ||
        !item.contains("class_id") ||
        !Unsigned(item["class_id"], 0, INT32_MAX) ||
        !item.contains("confidence") || !Number(item["confidence"], 0, 1) ||
        !item.contains("bbox") || !item["bbox"].is_array() ||
        item["bbox"].size() != 4)
      return false;
    float coords[4];
    for (size_t i = 0; i < 4; ++i) {
      if (!Number(item["bbox"][i], -std::numeric_limits<float>::max(),
                  std::numeric_limits<float>::max()))
        return false;
      coords[i] = item["bbox"][i].get<float>();
    }
    if (coords[2] <= 0 || coords[3] <= 0 ||
        (std::bit_cast<uint32_t>(coords[0] + coords[2]) & 0x7f800000U) ==
            0x7f800000U ||
        (std::bit_cast<uint32_t>(coords[1] + coords[3]) & 0x7f800000U) ==
            0x7f800000U)
      return false;
    TrackingDetection detection{item["class_id"].get<int32_t>(),
                                item["confidence"].get<float>(),
                                {coords[0], coords[1], coords[2], coords[3]},
                                {}};
    if (item.contains("embedding")) {
      const auto& embedding = item["embedding"];
      if (!embedding.is_array() || embedding.empty() || embedding.size() > 512)
        return false;
      double norm = 0;
      for (const auto& value : embedding) {
        if (!Number(value, -std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max()))
          return false;
        const auto number = value.get<float>();
        detection.embedding.push_back(number);
        norm += static_cast<double>(number) * number;
      }
      if (norm == 0) return false;
    }
    frame.detections.push_back(std::move(detection));
  }
  return true;
}
}  // namespace
struct TrackingAdapter::State {
  TrackingService service;
  enum class Route { kCreate, kList, kGet, kStep, kReset, kDelete };
  int Receive(const HttpContextPtr& ctx, Route route, http_parser_state phase,
              const char* data, size_t size) {
    if (phase == HP_ERROR) return HTTP_STATUS_UNFINISHED;
    if (ctx->response->status_code >= 400) return ctx->response->status_code;
    try {
      const bool post = route == Route::kCreate || route == Route::kStep ||
                        route == Route::kReset;
      const auto expectation = ctx->header("Expect");
      const bool continue_expected =
          ParseHTTPExpectation(expectation) == HTTPExpectation::kContinue;
      const auto reject = [&](int status, const char* message) {
        const bool immediate = !post || status == 413 || !expectation.empty();
        if (immediate) ctx->setHeader("Connection", "close");
        return Error(ctx, status, "invalid_request", message, immediate);
      };
      if (phase == HP_HEADERS_COMPLETE) {
        if (!ctx->header("Origin").empty())
          return reject(403, "Browser origins are not accepted");
        if (!expectation.empty() && !continue_expected)
          return reject(417, "Unsupported expectation");
        const auto declared = ctx->header("Content-Length");
        uint64_t length = 0;
        if (!declared.empty()) {
          const auto parsed = std::from_chars(
              declared.data(), declared.data() + declared.size(), length);
          if (parsed.ec != std::errc{} ||
              parsed.ptr != declared.data() + declared.size())
            return reject(post ? 413 : 400, "Invalid Content-Length");
        }
        if (!post) {
          if (length || ctx->request->headers.find("Transfer-Encoding") !=
                            ctx->request->headers.end())
            return reject(
                400, "Tracking GET and DELETE requests must not have a body");
        } else {
          if (length > kBodyLimit)
            return reject(413, "Tracking body exceeds 4 MiB");
          if (ctx->header("Content-Type") != "application/json")
            return reject(400, "Use Content-Type: application/json");
        }
        if (continue_expected)
          ctx->writer->write("HTTP/1.1 100 Continue\r\n\r\n");
      } else if (phase == HP_BODY) {
        if (!post && size)
          return reject(
              400, "Tracking GET and DELETE requests must not have a body");
        if (ctx->request->body.size() > kBodyLimit ||
            size > kBodyLimit - ctx->request->body.size())
          return reject(413, "Tracking body exceeds 4 MiB");
        if (post) ctx->request->body.append(data, size);
      } else if (phase == HP_MESSAGE_COMPLETE) {
        return Handle(ctx, route);
      }
      return HTTP_STATUS_UNFINISHED;
    } catch (...) {
      ctx->setHeader("Connection", "close");
      return Error(ctx, TrackingFailure::kFailed);
    }
  }
  int Handle(const HttpContextPtr& ctx, Route route) {
    try {
      if (!ctx->header("Origin").empty())
        return Error(ctx, 403, "invalid_request",
                     "Browser origins are not accepted");
      const bool post = route == Route::kCreate || route == Route::kStep ||
                        route == Route::kReset;
      Json body;
      if (post) {
        if (ctx->header("Content-Type") != "application/json" ||
            ctx->body().size() > kBodyLimit)
          return Error(ctx, TrackingFailure::kInvalid);
        body = Json::parse(ctx->body(), nullptr, false);
        if (body.is_discarded()) return Error(ctx, TrackingFailure::kInvalid);
      }
      const auto id = ctx->param("id");
      if (route == Route::kCreate) {
        TrackerOptions options;
        if (!Options(body, options))
          return Error(ctx, TrackingFailure::kInvalid);
        auto new_id = Token();
        auto response = Json{
            {"id", new_id},
            {"algorithm", Algorithm(options.algorithm)},
            {"status",
             Status({})}}.dump();
        auto location = std::string(kRoot) + "/" + new_id;
        auto result = service.Add(new_id, options);
        if (!result) return Error(ctx, result.error());
        ctx->setHeader("Location", location.c_str());
        return Send(ctx, 201, std::move(response));
      }
      if (route == Route::kList) {
        size_t limit = 100;
        const auto text = ctx->param("limit");
        if (!text.empty()) {
          auto parsed =
              std::from_chars(text.data(), text.data() + text.size(), limit);
          if (parsed.ec != std::errc{} ||
              parsed.ptr != text.data() + text.size() || !limit || limit > 100)
            return Error(ctx, TrackingFailure::kInvalid);
        }
        auto cursor = ctx->param("cursor");
        if (!cursor.empty()) {
          if (!cursor.starts_with("t1.") ||
              !TokenValid(std::string_view(cursor).substr(3)))
            return Error(ctx, TrackingFailure::kInvalid);
          cursor.erase(0, 3);
        }
        auto page = service.List(limit, cursor);
        if (!page) return Error(ctx, page.error());
        Json response{{"sessions", page->ids},
                      {"next_cursor", page->next_cursor.empty()
                                          ? Json(nullptr)
                                          : Json("t1." + page->next_cursor)}};
        return Send(ctx, 200, response.dump());
      }
      if (route == Route::kGet) {
        auto info = service.Get(id);
        if (!info) return Error(ctx, info.error());
        return Send(ctx, 200,
                    Json{{"id", info->id},
                         {"algorithm", Algorithm(info->algorithm)},
                         {"status", Status(info->status)}}
                        .dump());
      }
      if (route == Route::kReset) {
        if (!body.is_object() || !body.empty())
          return Error(ctx, TrackingFailure::kInvalid);
        std::string response = "{\"reset\":true}";
        auto result = service.Reset(id);
        return result ? Send(ctx, 200, std::move(response))
                      : Error(ctx, result.error());
      }
      if (route == Route::kDelete) {
        auto result = service.Delete(id);
        return result ? Send(ctx, 204, {}) : Error(ctx, result.error());
      }
      Frame frame;
      if (!ParseFrame(body, frame))
        return Error(ctx, TrackingFailure::kInvalid);
      std::optional<std::string_view> image;
      if (body.contains("image"))
        image = body["image"].get_ref<const std::string&>();
      auto result = service.Step(id, frame.index, frame.timestamp,
                                 frame.detections, image);
      if (!result) return Error(ctx, result.error());
      Json tracks = Json::array();
      for (const auto& track : result->tracks)
        tracks.push_back({{"track_id", track.track_id},
                          {"class_id", track.class_id},
                          {"confidence", track.confidence},
                          {"bbox",
                           {track.bbox.x, track.bbox.y, track.bbox.width,
                            track.bbox.height}}});
      return Send(ctx, 200,
                  Json{{"frame_index", result->frame_index},
                       {"timestamp", result->timestamp},
                       {"tracks", std::move(tracks)}}
                      .dump());
    } catch (...) {
      return Error(ctx, TrackingFailure::kFailed);
    }
  }
};
TrackingAdapter::TrackingAdapter(std::shared_ptr<State> state)
    : state_(std::move(state)) {}
VSResult<std::unique_ptr<TrackingAdapter>> TrackingAdapter::Create() noexcept {
  try {
    return std::unique_ptr<TrackingAdapter>(
        new TrackingAdapter(std::make_shared<State>()));
  } catch (...) {
    return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError,
                      "Unable to create tracking service");
  }
}
TrackingAdapter::~TrackingAdapter() { Stop(); }
void TrackingAdapter::Mount(hv::HttpService& service) {
  const auto state = state_;
  service.POST("/v1/tracking/sessions", [state](const HttpContextPtr& ctx,
                                                http_parser_state phase,
                                                const char* data, size_t size) {
    return state->Receive(ctx, State::Route::kCreate, phase, data, size);
  });
  service.GET("/v1/tracking/sessions", [state](const HttpContextPtr& ctx,
                                               http_parser_state phase,
                                               const char* data, size_t size) {
    return state->Receive(ctx, State::Route::kList, phase, data, size);
  });
  service.GET("/v1/tracking/sessions/:id",
              [state](const HttpContextPtr& ctx, http_parser_state phase,
                      const char* data, size_t size) {
                return state->Receive(ctx, State::Route::kGet, phase, data,
                                      size);
              });
  service.POST("/v1/tracking/sessions/:id/frames",
               [state](const HttpContextPtr& ctx, http_parser_state phase,
                       const char* data, size_t size) {
                 return state->Receive(ctx, State::Route::kStep, phase, data,
                                       size);
               });
  service.POST("/v1/tracking/sessions/:id/reset",
               [state](const HttpContextPtr& ctx, http_parser_state phase,
                       const char* data, size_t size) {
                 return state->Receive(ctx, State::Route::kReset, phase, data,
                                       size);
               });
  service.Delete("/v1/tracking/sessions/:id",
                 [state](const HttpContextPtr& ctx, http_parser_state phase,
                         const char* data, size_t size) {
                   return state->Receive(ctx, State::Route::kDelete, phase,
                                         data, size);
                 });
}
void TrackingAdapter::Stop() noexcept {
  if (state_) state_->service.Stop();
}
}  // namespace vision_simple
