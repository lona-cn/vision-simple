#include "OpenAIAdapter.h"

#include <atomic>
#include <charconv>
#include <chrono>
#include <nlohmann/json.hpp>

#include "HTTPExpectation.h"
#include "HttpDispatch.h"
#include "InferenceProtocol.h"
#include "InferenceRequestOptions.h"
#include "LogFacade.h"

namespace vision_simple {
namespace {
using Json = nlohmann::json;
constexpr size_t kMaxBodyBytes = 64 * 1024 * 1024;

struct AdapterError {
  int status = 400;
  std::string_view code = "invalid_request";
  std::string_view message = "Invalid visual Chat Completions request";
  std::string param;
  std::optional<size_t> image_index;
};
std::string ErrorBody(const AdapterError& error) {
  Json detail{
      {"code", error.code},
      {"message", error.message},
      {"type", error.status < 500 ? "invalid_request_error" : "server_error"},
      {"param", error.param.empty() ? Json(nullptr) : Json(error.param)},
      {"image_index",
       error.image_index ? Json(*error.image_index) : Json(nullptr)}};
  return Json{{"error", std::move(detail)}}.dump();
}
int SendError(const HttpContextPtr& ctx, const AdapterError& error,
              bool finish = true) {
  ctx->setStatus(static_cast<http_status>(error.status));
  if (error.status == 503) ctx->setHeader("Retry-After", "1");
  ctx->setContentType(APPLICATION_JSON);
  ctx->response->body = ErrorBody(error);
  return finish ? ctx->send() : error.status;
}
HttpDispatch::Reply ErrorReply(const AdapterError& error) {
  return [status = error.status, body = ErrorBody(error)](
             const HttpContextPtr& ctx) mutable {
    ctx->setStatus(static_cast<http_status>(status));
    if (status == 503) ctx->setHeader("Retry-After", "1");
    ctx->setContentType(APPLICATION_JSON);
    ctx->response->body = std::move(body);
    ctx->send();
  };
}
HttpDispatch::Reply ErrorReply(const ServiceError& error) {
  const auto info = DescribeError(error.kind);
  return ErrorReply(
      AdapterError{info.status, info.code, info.message,
                   error.kind == ServiceFailure::kUnknownModel ? "model" : "",
                   error.image_index});
}
HttpDispatch::Reply InternalError(const std::exception& error) {
  LogFacade::Error("openai", error.what());
  return ErrorReply(
      AdapterError{
          500, "internal_error", "Request could not be completed", {}, {}});
}

struct ChatRequest {
  std::string id;
  std::string model;
  InferenceKind kind;
  std::vector<std::string> images;
  bool stream = false;
  std::optional<std::chrono::milliseconds> timeout;
  YOLOInferenceOptions options;
};
std::expected<ChatRequest, AdapterError> ParseChat(const Json& json,
                                                   size_t max_images) {
  const auto invalid = [](std::string param, std::string_view message) {
    return std::unexpected(
        AdapterError{400, "invalid_request", message, std::move(param), {}});
  };
  if (!json.is_object()) return invalid({}, "Request must be an object");
  for (auto it = json.begin(); it != json.end(); ++it) {
    const auto& key = it.key();
    if (key != "model" && key != "messages" && key != "stream" &&
        key != "timeout_ms" && key != "response_format" && key != "n" &&
        key != "confidence" && key != "nms_iou")
      return invalid(key,
                     "This parameter is not supported by the vision adapter");
  }
  const auto model = json.find("model"), messages = json.find("messages");
  if (model == json.end() || !model->is_string())
    return invalid("model", "Select a model ID returned by /v1/models");
  ChatRequest request;
  request.id = model->get<std::string>();
  const auto colon = request.id.find(':');
  const auto* task =
      colon == std::string::npos
          ? nullptr
          : FindTask(std::string_view(request.id).substr(0, colon));
  if (!task || colon + 1 == request.id.size())
    return invalid("model",
                   "Model IDs must use a registered task prefix; discover "
                   "available IDs with /v1/models");
  request.kind = task->kind;
  request.model = request.id.substr(colon + 1);
  const auto options = ParseInferenceOptions(json, request.kind);
  if (!options)
    return invalid(options.error().parameter, options.error().message);
  request.options = *options;
  if (const auto stream = json.find("stream"); stream != json.end()) {
    if (!stream->is_boolean())
      return invalid("stream", "stream must be a boolean");
    request.stream = stream->get<bool>();
  }
  if (const auto n = json.find("n"); n != json.end()) {
    if (!n->is_number_integer() || *n != 1)
      return invalid("n", "The vision adapter supports exactly one completion");
  }
  if (const auto timeout = json.find("timeout_ms"); timeout != json.end()) {
    if (!timeout->is_number_integer() || *timeout < 1 || *timeout > 300000)
      return invalid("timeout_ms",
                     "timeout_ms must be an integer from 1 to 300000");
    request.timeout = std::chrono::milliseconds(timeout->get<int64_t>());
  }
  if (const auto format = json.find("response_format"); format != json.end()) {
    if (!format->is_object() || format->size() != 1 ||
        !format->contains("type") ||
        ((*format)["type"] != "json_object" && (*format)["type"] != "text"))
      return invalid("response_format",
                     "Only text and json_object are supported; content is "
                     "always structured result JSON");
  }
  if (messages == json.end() || !messages->is_array() || messages->empty())
    return invalid(
        "messages",
        "messages must be a nonempty array with user image_url content");
  for (const auto& message : *messages) {
    if (!message.is_object() || !message.contains("role") ||
        !message["role"].is_string() || !message.contains("content"))
      return invalid("messages", "Each message must contain role and content");
    const auto& role = message["role"].get_ref<const std::string&>();
    if (role != "user" && role != "system" && role != "developer" &&
        role != "assistant")
      return invalid("messages",
                     "Tool and function messages are not supported");
    const auto& content = message["content"];
    if (content.is_string())
      continue;  // Text is context only; these are not language models.
    if (!content.is_array())
      return invalid("messages",
                     "Message content must be text or a content-part array");
    for (const auto& part : content) {
      if (!part.is_object() || !part.contains("type") ||
          !part["type"].is_string())
        return invalid("messages", "Content parts must have a string type");
      if (part["type"] == "text") {
        if (!part.contains("text") || !part["text"].is_string())
          return invalid("messages", "Text parts must contain a text string");
        continue;
      }
      if (part["type"] != "image_url" || role != "user" ||
          !part.contains("image_url") || !part["image_url"].is_object())
        return invalid("messages",
                       "Only text and user image_url parts are supported");
      const auto& image = part["image_url"];
      if (!image.contains("url") || !image["url"].is_string())
        return invalid("messages",
                       "image_url.url must be an inline image data URL");
      if (image.contains("detail") && image["detail"] != "auto")
        return invalid("messages",
                       "Only detail:auto is supported; resolution is "
                       "determined by the vision model");
      const auto& url = image["url"].get_ref<const std::string&>();
      const auto comma = url.find(',');
      if (comma == std::string::npos ||
          (url.substr(0, comma) != "data:image/png;base64" &&
           url.substr(0, comma) != "data:image/jpeg;base64" &&
           url.substr(0, comma) != "data:image/webp;base64" &&
           url.substr(0, comma) != "data:image/bmp;base64"))
        return invalid("messages",
                       "Use a base64 PNG, JPEG, WebP or BMP data URL; remote "
                       "URLs are never fetched");
      if (request.images.size() >= max_images)
        return invalid("messages",
                       "Image count exceeds the configured batch limit");
      request.images.emplace_back(url.substr(comma + 1));
    }
  }
  if (request.images.empty())
    return invalid("messages",
                   "At least one image is required; text-only generation is "
                   "not supported");
  return request;
}

HttpDispatch::Reply ListModels(std::optional<std::string> limit_text, std::string after,
                               InferenceService& service) {
  try {
    size_t limit = 100;
    if (limit_text) {
      const auto parsed = std::from_chars(
          limit_text->data(), limit_text->data() + limit_text->size(), limit);
      if (parsed.ec != std::errc{} ||
          parsed.ptr != limit_text->data() + limit_text->size())
        return ErrorReply(AdapterError{400, "invalid_request",
                                      "limit must be an integer from 1 to 200",
                                      "limit", {}});
    }
    auto catalog = service.ListModels();
    if (!catalog) return ErrorReply(catalog.error());
    auto page = PaginateModels(*catalog, limit, after);
    if (!page)
      return ErrorReply(AdapterError{400, "invalid_request",
                                    "Invalid limit or catalog cursor", "after", {}});
    auto data = Json::array();
    for (const auto& model : page->data)
      data.push_back({{"id", model.id},
                      {"object", "model"},
                      {"created", 0},
                      {"owned_by", "vision-simple"},
                      {"task", model.kind}});
    Json body{{"object", "list"},
              {"data", std::move(data)},
              {"has_more", !page->next_cursor.empty()}};
    if (!page->next_cursor.empty()) body["next_cursor"] = page->next_cursor;
    return [body = body.dump()](const HttpContextPtr& ctx) mutable {
      ctx->setContentType(APPLICATION_JSON);
      ctx->setHeader("Cache-Control", "no-store");
      ctx->response->body = std::move(body);
      ctx->send();
    };
  } catch (const std::exception& error) {
    return InternalError(error);
  }
}

HttpDispatch::Reply Chat(std::string body, InferenceService& service,
                         std::stop_token stop, HttpDispatch::Clock::time_point started) {
  try {
    const auto json = ParseInferenceJson(body);
    if (json.is_discarded())
      return ErrorReply(AdapterError{
          400, "invalid_json", "Request body must be valid JSON", {}, {}});
    auto request = ParseChat(json, service.options().pipeline.max_batch_images);
    if (!request) return ErrorReply(request.error());
    auto result = service.Run(
        request->kind, request->model, request->images, request->options,
        ServiceControl{.stop = stop, .timeout = request->timeout, .started = started});
    if (!result) return ErrorReply(result.error());
    auto content = SerializeInference(*result);
    if (content.size() > kMaxBodyBytes)
      return ErrorReply(AdapterError{500, "response_too_large",
                                    "Result exceeds 64 MiB; submit fewer images", {}, {}});
    static std::atomic<uint64_t> sequence{0};
    const auto created =
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    const auto id = "chatcmpl-" + std::to_string(created) + "-" +
                    std::to_string(sequence.fetch_add(1));
    if (!request->stream) {
      Json response{
          {"id", id},
          {"object", "chat.completion"},
          {"created", created},
          {"model", request->id},
          {"choices", Json::array({{{"index", 0},
                                    {"message",
                                     {{"role", "assistant"},
                                      {"content", std::move(content)}}},
                                    {"finish_reason", "stop"}}})}};
      return [body = response.dump(), completion = std::move(result->completion)](
                 const HttpContextPtr& ctx) mutable {
        ctx->setContentType(APPLICATION_JSON);
        ctx->response->body = std::move(body);
        if (completion) completion->Succeed();
        ctx->send();
      };
    }
    const auto chunk = [&](Json delta, Json finish) {
      return "data: " +
             Json{{"id", id},
                  {"object", "chat.completion.chunk"},
                  {"created", created},
                  {"model", request->id},
                  {"choices",
                   Json::array({{{"index", 0},
                                 {"delta", std::move(delta)},
                                 {"finish_reason", std::move(finish)}}})}}
                 .dump() +
             "\n\n";
    };
    // Prepare the complete bounded SSE batch before committing any headers.
    return [role = chunk({{"role", "assistant"}}, nullptr),
            data = chunk({{"content", std::move(content)}}, nullptr),
            finish = chunk(Json::object(), "stop"),
            completion = std::move(result->completion)](
               const HttpContextPtr& ctx) {
      ctx->setHeader("Content-Type", "text/event-stream");
      ctx->setHeader("Cache-Control", "no-cache, no-transform");
      ctx->setHeader("X-Accel-Buffering", "no");
      ctx->writer->WriteChunked(role);
      ctx->writer->WriteChunked(data);
      ctx->writer->WriteChunked(finish);
      ctx->writer->WriteChunked("data: [DONE]\n\n");
      if (completion) completion->Succeed();
      ctx->writer->End();
    };
  } catch (const std::exception& error) {
    return InternalError(error);
  }
}
}  // namespace

void RegisterOpenAI(hv::HttpService& http,
                    std::shared_ptr<InferenceService> service,
                    HttpDispatch& dispatch) {
  http.GET("/v1/models", [service, &dispatch](const HttpContextPtr& ctx) {
    if (!dispatch.Submit(
            ctx, HttpDispatch::Lane::kControl,
            [service, limit = ctx->params().contains("limit")
                                  ? std::optional<std::string>(ctx->param("limit"))
                                  : std::optional<std::string>{},
             after = ctx->param("after")](
                std::stop_token, HttpDispatch::Clock::time_point) mutable {
              return ListModels(std::move(limit), std::move(after), *service);
            }))
      return SendError(ctx, AdapterError{503, "service_overloaded",
                                        "HTTP dispatch capacity reached", {}, {}});
    return HTTP_STATUS_UNFINISHED;
  });
  http.POST(
      "/v1/chat/completions",
      [service, &dispatch](const HttpContextPtr& ctx, http_parser_state phase,
                const char* data, size_t size) -> int {
        if (phase == HP_ERROR) return HTTP_STATUS_UNFINISHED;
        if (ctx->response->status_code >= 400)
          return ctx->response->status_code;
        const auto expectation = ctx->header("Expect");
        const bool continue_expected =
            ParseHTTPExpectation(expectation) == HTTPExpectation::kContinue;
        const auto reject = [&](AdapterError error) {
          const bool immediate = error.status == 413 || !expectation.empty();
          if (immediate) ctx->setHeader("Connection", "close");
          return SendError(ctx, error, immediate);
        };
        if (phase == HP_HEADERS_COMPLETE) {
          if (!expectation.empty() && !continue_expected)
            return reject(
                {417, "expectation_failed", "Unsupported expectation", {}, {}});
          if (!ctx->is(APPLICATION_JSON))
            return reject({415,
                           "unsupported_media_type",
                           "Use Content-Type: application/json",
                           {},
                           {}});
          const auto declared = ctx->header("Content-Length");
          if (!declared.empty()) {
            uint64_t length = 0;
            const auto parsed = std::from_chars(
                declared.data(), declared.data() + declared.size(), length);
            if (parsed.ec != std::errc{} ||
                parsed.ptr != declared.data() + declared.size() ||
                length > kMaxBodyBytes)
              return reject({413,
                             "request_too_large",
                             "Request body exceeds 64 MiB",
                             {},
                             {}});
          }
          if (continue_expected)
            ctx->writer->write("HTTP/1.1 100 Continue\r\n\r\n");
        } else if (phase == HP_BODY) {
          if (size > kMaxBodyBytes - ctx->request->body.size())
            return reject({413,
                           "request_too_large",
                           "Request body exceeds 64 MiB",
                           {},
                           {}});
          ctx->request->body.append(data, size);
        } else if (phase == HP_MESSAGE_COMPLETE) {
          if (!dispatch.Submit(
                  ctx, HttpDispatch::Lane::kData,
                  [service, body = std::move(ctx->request->body)](
                      std::stop_token stop, HttpDispatch::Clock::time_point started) mutable {
                    return Chat(std::move(body), *service, stop, started);
                  }))
            return SendError(ctx, AdapterError{503, "service_overloaded",
                                              "HTTP dispatch capacity reached", {}, {}});
          return HTTP_STATUS_UNFINISHED;
        }
        return HTTP_STATUS_UNFINISHED;
      });
}
}  // namespace vision_simple
