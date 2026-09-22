#include "MCPAdapter.h"

#include <hv/EventLoop.h>
#include <hv/HttpServer.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <nlohmann/json.hpp>
#include <random>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

#include "InferenceProtocol.h"

namespace vision_simple {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr size_t kBodyLimit = 64 * 1024 * 1024;
constexpr size_t kOutboundLimit = 8 * 1024 * 1024;
constexpr size_t kSessions = 32, kQueue = 16, kOutstanding = 8;
constexpr std::array<std::string_view, 4> kVersions{"2024-11-05", "2025-03-26",
                                                    "2025-06-18", "2025-11-25"};

std::string Lower(std::string value) {
  for (char& c : value)
    if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
  return value;
}
std::string Token() {
  // MSVC's random_device is backed by the operating system's secure generator.
  std::random_device random;
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(32);
  for (int i = 0; i < 4; ++i) {
    const auto word = random();
    for (int shift = 0; shift < 32; shift += 4)
      result += hex[(word >> shift) & 15];
  }
  return result;
}
bool Integer(const Json& value) {
  return value.is_number_integer() || value.is_number_unsigned();
}
bool Id(const Json& value) { return value.is_string() || Integer(value); }
bool Fields(const Json& value,
            std::initializer_list<std::string_view> allowed) {
  if (!value.is_object()) return false;
  for (auto it = value.begin(); it != value.end(); ++it)
    if (std::find(allowed.begin(), allowed.end(), it.key()) == allowed.end())
      return false;
  return true;
}
bool Bounded(const Json& value, uint64_t low, uint64_t high) {
  if (!Integer(value)) return false;
  if (value.is_number_integer() && !value.is_number_unsigned() &&
      value.get<int64_t>() < 0)
    return false;
  const auto number = value.get<uint64_t>();
  return number >= low && number <= high;
}
Json RpcResult(const Json& id, Json result) {
  return {{"jsonrpc", "2.0"}, {"id", id}, {"result", std::move(result)}};
}
Json RpcError(const Json& id, int code, std::string_view message) {
  return {{"jsonrpc", "2.0"},
          {"id", id},
          {"error", {{"code", code}, {"message", message}}}};
}
Json ToolResult(Json data, bool modern, bool error = false) {
  Json result{
      {"content", Json::array({{{"type", "text"}, {"text", data.dump()}}})},
      {"isError", error}};
  if (modern) result["structuredContent"] = std::move(data);
  return result;
}
Json ToolError(ServiceError error, bool modern) {
  const auto description = DescribeError(error.kind);
  std::string message(description.message);
  switch (error.kind) {
    case ServiceFailure::kUnknownModel:
      message +=
          "; call list_models and use its raw name for the correct task.";
      break;
    case ServiceFailure::kInvalidImage:
      message += "; supply a base64-encoded supported image, not a URL.";
      break;
    case ServiceFailure::kBusy:
    case ServiceFailure::kModelBusy:
      message += "; wait for outstanding work to finish before retrying.";
      break;
    case ServiceFailure::kTimedOut:
      message +=
          "; reduce the image batch or increase timeout_ms (maximum 300000).";
      break;
    case ServiceFailure::kCancelled:
      message += "; submit a new request if the result is still needed.";
      break;
    case ServiceFailure::kInvalidRequest:
      message +=
          "; check the tool input schema and configured image-batch limit.";
      break;
    default:
      message +=
          "; check server availability and configured model files before "
          "retrying.";
      break;
  }
  Json detail{{"code", description.code}, {"message", message}};
  detail["image_index"] =
      error.image_index ? Json(*error.image_index) : Json(nullptr);
  return ToolResult({{"error", std::move(detail)}}, modern, true);
}
Json Tools(size_t max_images) {
  const Json paging{{"type", "object"},
                    {"additionalProperties", false},
                    {"properties",
                     {{"limit",
                       {{"type", "integer"},
                        {"minimum", 1},
                        {"maximum", 200},
                        {"default", 100},
                        {"description", "Maximum model entries returned."}}},
                      {"cursor",
                       {{"type", "string"},
                        {"description",
                         "Opaque next_cursor from the preceding list_models "
                         "result; omit for the first page."}}}}}};
  const Json infer{
      {"type", "object"},
      {"additionalProperties", false},
      {"required", {"model", "images"}},
      {"properties",
       {{"model",
         {{"type", "string"},
          {"minLength", 1},
          {"description",
           "Raw configured model name returned by list_models, without its "
           "task: catalog ID prefix."}}},
        {"images",
         {{"type", "array"},
          {"minItems", 1},
          {"maxItems", max_images},
          {"description",
           "Images in input order, each encoded as raw base64 (not a URL or "
           "data URI). Results preserve this order."},
          {"items",
           {{"type", "string"},
            {"minLength", 1},
            {"maxLength", kBodyLimit},
            {"description", "Base64 bytes of one supported image."}}}}},
        {"timeout_ms",
         {{"type", "integer"},
          {"minimum", 1},
          {"maximum", 300000},
          {"description",
           "Total cooperative deadline in milliseconds including transport "
           "queue time; omit for the server default."}}}}}};
  Json tools = Json::array(
      {{{"name", "list_models"},
        {"description",
         "Discover configured inference models before inference. Follow "
         "next_cursor for more entries; use each entry's raw name with its "
         "matching infer_<kind> tool, not its prefixed catalog ID."},
        {"inputSchema", paging}}});
  for (const auto& task : RegisteredTasks())
    tools.push_back({{"name", "infer_" + std::string(task.id)},
                     {"description", task.description},
                     {"inputSchema", infer}});
  return tools;
}
}  // namespace

struct MCPAdapter::State : std::enable_shared_from_this<MCPAdapter::State> {
  enum class Phase { kInit, kAwaitInitialized, kInitialized, kClosed };
  struct Session {
    std::string id;
    std::shared_ptr<hv::EventLoop> loop;
    HttpResponseWriterPtr writer;  // Accessed and released only on loop.
    hv::TimerID timer = INVALID_TIMER_ID;
    Phase phase =
        Phase::kInit;  // All remaining fields guarded by State::mutex.
    bool modern = false;
    Clock::time_point last_activity = Clock::now();
    Clock::time_point stalled_since{};  // Loop only.
    size_t outbound = 0;
    std::unordered_map<std::string, std::stop_source> active;
    std::unordered_map<std::string, size_t> pending_ids;
  };
  struct Job {
    std::shared_ptr<Session> session;
    Json id;
    std::string key;
    std::string tool;
    Json arguments;
    std::stop_source cancel;
    Clock::time_point started;
    bool modern;
  };
  std::shared_ptr<InferenceService> service;
  std::string host;
  uint16_t port;
  std::mutex mutex;
  std::condition_variable ready;
  std::unordered_map<std::string, std::shared_ptr<Session>> sessions;
  std::deque<Job> jobs;
  std::vector<std::jthread> workers;
  bool stopping = false;
  std::once_flag stopped;

  State(std::shared_ptr<InferenceService> value, std::string name,
        uint16_t number)
      : service(std::move(value)), host(Lower(std::move(name))), port(number) {}

  bool Authority(std::string authority, uint16_t default_port) const {
    authority = Lower(std::move(authority));
    std::string_view view(authority), name;
    std::string_view port_text;
    if (view.starts_with('[')) {
      auto end = view.find(']');
      if (end == std::string_view::npos) return false;
      name = view.substr(1, end - 1);
      if (end + 1 < view.size()) {
        if (view[end + 1] != ':') return false;
        port_text = view.substr(end + 2);
        if (port_text.empty()) return false;
      }
    } else {
      auto colon = view.find(':');
      name = view.substr(0, colon);
      if (colon != std::string_view::npos) {
        port_text = view.substr(colon + 1);
        if (port_text.empty()) return false;
      }
    }
    unsigned number = default_port;
    if (!port_text.empty()) {
      auto parsed = std::from_chars(
          port_text.data(), port_text.data() + port_text.size(), number);
      if (parsed.ec != std::errc{} ||
          parsed.ptr != port_text.data() + port_text.size())
        return false;
    }
    auto explicit_host = std::string_view(host);
    if (explicit_host.starts_with('[') && explicit_host.ends_with(']'))
      explicit_host = explicit_host.substr(1, explicit_host.size() - 2);
    const bool configured = explicit_host != "0.0.0.0" &&
                            explicit_host != "::" && explicit_host != "*" &&
                            !explicit_host.empty();
    return number == port &&
           (name == "localhost" || name == "127.0.0.1" || name == "::1" ||
            (configured && name == explicit_host));
  }
  bool Trusted(const HttpContextPtr& ctx) const {
    if (!Authority(ctx->header("Host"), 80)) return false;
    const auto origin = ctx->header("Origin");
    if (origin.empty()) return !ctx->headers().contains("Origin");
    std::string_view rest(origin);
    uint16_t default_port;
    if (rest.starts_with("http://")) {
      rest.remove_prefix(7);
      default_port = 80;
    } else if (rest.starts_with("https://")) {
      rest.remove_prefix(8);
      default_port = 443;
    } else
      return false;
    if (rest.find_first_of("/?#@\\") != std::string_view::npos) return false;
    return Authority(std::string(rest), default_port);
  }
  static int HttpError(const HttpContextPtr& ctx, int status,
                       std::string_view message) {
    ctx->response->status_code = static_cast<http_status>(status);
    ctx->response->body = message;
    // A normal header rejection must consume/discard the request body before
    // libhv sends the response. Closing here resets split header/body uploads.
    if (status == 413 || Lower(ctx->header("Expect")) == "100-continue") {
      ctx->response->SetHeader("Connection", "close");
      ctx->writer->End();
    }
    return status;
  }
  // Must run on the session's loop. Close first removes routing and cancels
  // work.
  void Close(const std::shared_ptr<Session>& session) {
    {
      std::lock_guard lock(mutex);
      session->phase = Phase::kClosed;
      sessions.erase(session->id);
      for (auto& [key, source] : session->active) source.request_stop();
    }
    if (session->timer != INVALID_TIMER_ID) {
      session->loop->killTimer(session->timer);
      session->timer = INVALID_TIMER_ID;
    }
    auto writer = std::move(session->writer);
    if (writer) {
      writer->onclose = nullptr;
      writer->close();
    }
  }
  void Send(const std::shared_ptr<Session>& session, Json message,
            std::shared_ptr<InferenceCompletion> completion = {},
            std::string key = {}) {
    auto frame = std::make_shared<std::string>(
        "event: message\ndata: " + message.dump() + "\n\n");
    const std::string response_id = message.contains("id") && Id(message["id"])
                                        ? message["id"].dump()
                                        : std::string{};
    bool overflow = false;
    {
      std::lock_guard lock(mutex);
      if (session->phase == Phase::kClosed) return;
      overflow = frame->size() > kOutboundLimit - session->outbound;
      if (!overflow)
        session->outbound += frame->size();
      else
        session->phase = Phase::kClosed;
      if (!overflow && !response_id.empty())
        ++session->pending_ids[response_id];
    }
    auto self = shared_from_this();
    session->loop->queueInLoop([self, session, frame, overflow, response_id,
                                completion = std::move(completion),
                                key = std::move(key)] {
      if (overflow) {
        self->Close(session);
        return;
      }
      bool closed;
      {
        std::lock_guard lock(self->mutex);
        session->outbound -= frame->size();
        if (!response_id.empty()) {
          auto found = session->pending_ids.find(response_id);
          if (found != session->pending_ids.end() && --found->second == 0)
            session->pending_ids.erase(found);
        }
        closed = session->phase == Phase::kClosed;
      }
      if (closed || !session->writer) return;
      auto& writer = session->writer;
      if (!writer->isOpened() ||
          writer->writeBufsize() > kOutboundLimit - frame->size() ||
          writer->write(*frame) < 0) {
        self->Close(session);
        return;
      }
      if (completion) completion->Succeed();
      if (!key.empty()) {
        std::lock_guard lock(self->mutex);
        session->active.erase(key);
      }
    });
  }
  int Open(const HttpContextPtr& ctx, std::shared_ptr<hv::EventLoop> loop) {
    if (!Trusted(ctx)) return HttpError(ctx, 403, "Untrusted Host or Origin");
    if (!loop) return HttpError(ctx, 503, "IO loop unavailable");
    auto session = std::make_shared<Session>();
    session->id = Token();
    session->loop = std::move(loop);
    {
      std::lock_guard lock(mutex);
      if (stopping || sessions.size() >= kSessions)
        return HttpError(ctx, 503, "MCP session capacity reached");
      while (sessions.contains(session->id)) session->id = Token();
      sessions.emplace(session->id, session);
    }
    session->writer = ctx->writer;
    auto weak = weak_from_this();
    std::weak_ptr<Session> weak_session = session;
    session->writer->onclose = [weak, weak_session] {
      if (auto self = weak.lock())
        if (auto live = weak_session.lock()) self->Close(live);
    };
    session->writer->setMaxWriteBufsize(static_cast<uint32_t>(kOutboundLimit));
    session->writer->setKeepaliveTimeout(0);
    session->writer->WriteHeader("Cache-Control", "no-cache, no-transform");
    session->writer->WriteHeader("X-Accel-Buffering", "no");
    session->writer->WriteHeader("Connection", "keep-alive");
    session->writer->EndHeaders("Content-Type", "text/event-stream");
    if (session->writer->write(
            "event: endpoint\ndata: /mcp/messages?session_id=" + session->id +
            "\n\n") < 0) {
      Close(session);
      return HTTP_STATUS_UNFINISHED;
    }
    session->timer = session->loop->setInterval(15000, [weak, weak_session](
                                                           hv::TimerID) {
      auto self = weak.lock();
      auto live = weak_session.lock();
      if (!self || !live) return;
      bool expire;
      {
        std::lock_guard lock(self->mutex);
        expire = self->stopping || live->phase == Phase::kClosed ||
                 (live->active.empty() &&
                  Clock::now() - live->last_activity > std::chrono::minutes(5));
      }
      if (expire || !live->writer || !live->writer->isOpened()) {
        self->Close(live);
        return;
      }
      const auto buffered = live->writer->writeBufsize();
      if (buffered) {
        if (live->stalled_since == Clock::time_point{})
          live->stalled_since = Clock::now();
        if (buffered >= kOutboundLimit ||
            Clock::now() - live->stalled_since >= std::chrono::seconds(30)) {
          self->Close(live);
          return;
        }
      } else
        live->stalled_since = {};
      if (live->writer->write(": keep-alive\n\n") < 0) self->Close(live);
    });
    return HTTP_STATUS_UNFINISHED;
  }

  void Receive(const std::shared_ptr<Session>& session,
               const std::string& body) {
    const auto started = Clock::now();
    Json message = Json::parse(body, nullptr, false);
    if (message.is_discarded()) {
      Send(session, RpcError(nullptr, -32700, "Parse error"));
      return;
    }
    if (!message.is_object()) {
      Send(session,
           RpcError(nullptr, -32600, "Expected one JSON-RPC request object"));
      return;
    }
    const bool notification = !message.contains("id");
    const Json id = notification ? Json(nullptr) : message["id"];
    if (!message.contains("jsonrpc") || message["jsonrpc"] != "2.0" ||
        !message.contains("method") || !message["method"].is_string() ||
        (!notification && !Id(id))) {
      Send(session, RpcError(!notification && Id(id) ? id : Json(nullptr),
                             -32600, "Invalid JSON-RPC request"));
      return;
    }
    auto error = [&](int code, std::string_view text) {
      if (!notification) Send(session, RpcError(id, code, text));
    };
    if (!Fields(message, {"jsonrpc", "id", "method", "params"})) {
      error(-32600, "Invalid request fields");
      return;
    }
    const auto& method = message["method"].get_ref<const std::string&>();
    Json params = message.contains("params") ? std::move(message["params"])
                                             : Json::object();
    if (!params.is_object()) {
      error(-32602, "params must be an object");
      return;
    }
    std::unique_lock lock(mutex);
    if (session->phase == Phase::kClosed || stopping) return;
    session->last_activity = started;
    const auto key = notification ? std::string{} : id.dump();
    if (!notification &&
        (session->active.contains(key) || session->pending_ids.contains(key))) {
      lock.unlock();
      error(-32600, "Request ID is already in flight in this session");
      return;
    }
    if (method == "notifications/cancelled" && notification) {
      if (params.contains("requestId") && Id(params["requestId"])) {
        auto found = session->active.find(params["requestId"].dump());
        if (found != session->active.end()) found->second.request_stop();
      }
      return;
    }
    if (method == "notifications/initialized" && notification) {
      if (session->phase == Phase::kAwaitInitialized)
        session->phase = Phase::kInitialized;
      return;
    }
    if (notification) return;
    if (method == "initialize") {
      if (session->phase != Phase::kInit) {
        lock.unlock();
        error(-32600, "Session is already initialized");
        return;
      }
      if (!params.contains("protocolVersion") ||
          !params["protocolVersion"].is_string() ||
          !params.contains("capabilities") ||
          !params["capabilities"].is_object() ||
          !params.contains("clientInfo") || !params["clientInfo"].is_object() ||
          !params["clientInfo"].contains("name") ||
          !params["clientInfo"]["name"].is_string() ||
          !params["clientInfo"].contains("version") ||
          !params["clientInfo"]["version"].is_string()) {
        lock.unlock();
        error(-32602,
              "initialize requires protocolVersion, capabilities and "
              "clientInfo name/version");
        return;
      }
      std::string version = params["protocolVersion"].get<std::string>();
      if (std::find(kVersions.begin(), kVersions.end(), version) ==
          kVersions.end())
        version = kVersions.back();
      session->modern = version >= "2025-06-18";
      session->phase = Phase::kAwaitInitialized;
      lock.unlock();
      Send(session,
           RpcResult(id,
                     {{"protocolVersion", version},
                      {"capabilities", {{"tools", {{"listChanged", false}}}}},
                      {"serverInfo",
                       {{"name", "vision-simple"}, {"version", "1.0.0"}}},
                      {"instructions",
                       "Object detection and OCR only. Discover configured raw "
                       "model names with list_models before inference."}}));
      return;
    }
    if (method == "ping") {
      lock.unlock();
      Send(session, RpcResult(id, Json::object()));
      return;
    }
    if (session->phase != Phase::kInitialized) {
      lock.unlock();
      error(-32600, "Complete initialize and notifications/initialized first");
      return;
    }
    if (method == "tools/list") {
      lock.unlock();
      if (!Fields(params, {"_meta"}) ||
          (params.contains("_meta") && !params["_meta"].is_object())) {
        error(-32602, "tools/list has no cursor: all tools fit one page");
        return;
      }
      Send(session,
           RpcResult(id,
                     {{"tools",
                       Tools(service->options().pipeline.max_batch_images)}}));
      return;
    }
    if (method != "tools/call") {
      lock.unlock();
      error(-32601, "Method not found");
      return;
    }
    if (!Fields(params, {"name", "arguments", "_meta"}) ||
        !params.contains("name") || !params["name"].is_string() ||
        (params.contains("_meta") && !params["_meta"].is_object())) {
      lock.unlock();
      error(-32602, "tools/call requires a tool name and object arguments");
      return;
    }
    auto tool = params["name"].get<std::string>();
    auto arguments = params.contains("arguments")
                         ? std::move(params["arguments"])
                         : Json::object();
    if (tool != "list_models" &&
        (!tool.starts_with("infer_") ||
         !FindTask(std::string_view(tool).substr(6)))) {
      lock.unlock();
      error(-32602, "Unknown tool; call tools/list");
      return;
    }
    if (!arguments.is_object()) {
      lock.unlock();
      error(-32602, "arguments must be an object");
      return;
    }
    const bool modern = session->modern;
    if (session->active.size() >= kOutstanding || jobs.size() >= kQueue) {
      lock.unlock();
      Send(session,
           RpcResult(id, ToolError({ServiceFailure::kBusy, {}}, modern)));
      return;
    }
    std::stop_source cancel;
    session->active.emplace(key, cancel);
    jobs.push_back({session, id, key, std::move(tool), std::move(arguments),
                    cancel, started, modern});
    lock.unlock();
    ready.notify_one();
  }

  void Execute(Job& job) {
    const auto& args = job.arguments;
    auto invalid = [&] {
      Send(job.session,
           RpcResult(job.id, ToolError({ServiceFailure::kInvalidRequest, {}},
                                       job.modern)),
           {}, job.key);
    };
    if (job.cancel.stop_requested()) {
      Send(job.session,
           RpcResult(job.id,
                     ToolError({ServiceFailure::kCancelled, {}}, job.modern)),
           {}, job.key);
      return;
    }
    if (job.tool == "list_models") {
      if (!Fields(args, {"limit", "cursor"}) ||
          (args.contains("limit") && !Bounded(args["limit"], 1, 200)) ||
          (args.contains("cursor") && !args["cursor"].is_string())) {
        invalid();
        return;
      }
      auto catalog = service->ListModels();
      if (!catalog) {
        Send(job.session,
             RpcResult(job.id, ToolError(catalog.error(), job.modern)), {},
             job.key);
        return;
      }
      auto page = PaginateModels(*catalog, args.value("limit", size_t{100}),
                                 args.value("cursor", std::string{}));
      if (!page) {
        invalid();
        return;
      }
      Json data{{"data", Json::array()}};
      for (auto& item : page->data)
        data["data"].push_back(
            {{"id", item.id}, {"kind", item.kind}, {"name", item.name}});
      if (!page->next_cursor.empty()) data["next_cursor"] = page->next_cursor;
      Send(job.session,
           RpcResult(job.id, ToolResult(std::move(data), job.modern)), {},
           job.key);
      return;
    }
    if (!Fields(args, {"model", "images", "timeout_ms"}) ||
        !args.contains("model") || !args["model"].is_string() ||
        args["model"].get_ref<const std::string&>().empty() ||
        !args.contains("images") || !args["images"].is_array() ||
        args["images"].empty() ||
        args["images"].size() > service->options().pipeline.max_batch_images ||
        (args.contains("timeout_ms") &&
         !Bounded(args["timeout_ms"], 1, 300000))) {
      invalid();
      return;
    }
    std::vector<std::string> images;
    images.reserve(args["images"].size());
    for (auto& image : job.arguments["images"]) {
      if (!image.is_string() || image.get_ref<const std::string&>().empty()) {
        invalid();
        return;
      }
      images.emplace_back(std::move(image.get_ref<std::string&>()));
    }
    ServiceControl control;
    control.stop = job.cancel.get_token();
    control.started = job.started;
    if (args.contains("timeout_ms"))
      control.timeout =
          std::chrono::milliseconds(args["timeout_ms"].get<int64_t>());
    const auto* task = FindTask(std::string_view(job.tool).substr(6));
    if (!task) {
      invalid();
      return;
    }
    auto response =
        service->Run(task->kind, args["model"].get_ref<const std::string&>(),
                     images, control);
    if (!response) {
      Send(job.session,
           RpcResult(job.id, ToolError(response.error(), job.modern)), {},
           job.key);
      return;
    }
    auto data = Json::parse(SerializeInference(*response));
    Send(job.session,
         RpcResult(job.id, ToolResult(std::move(data), job.modern)),
         std::move(response->completion), job.key);
  }
  void Worker() {
    for (;;) {
      Job job;
      {
        std::unique_lock lock(mutex);
        ready.wait(lock, [&] { return stopping || !jobs.empty(); });
        if (stopping && jobs.empty()) return;
        job = std::move(jobs.front());
        jobs.pop_front();
      }
      try {
        Execute(job);
      } catch (...) {
        try {
          Send(job.session,
               RpcResult(job.id, ToolError({ServiceFailure::kInternal, {}},
                                           job.modern)),
               {}, job.key);
        } catch (...) {
          auto self = shared_from_this();
          job.session->loop->queueInLoop(
              [self, session = job.session] { self->Close(session); });
        }
      }
    }
  }
  int Post(const HttpContextPtr& ctx, http_parser_state phase, const char* data,
           size_t size) {
    if (phase == HP_ERROR) return HTTP_STATUS_UNFINISHED;
    if (ctx->response->status_code >= 400) return ctx->response->status_code;
    if (phase == HP_HEADERS_COMPLETE) {
      if (!Trusted(ctx)) return HttpError(ctx, 403, "Untrusted Host or Origin");
      auto type = Lower(ctx->header("Content-Type"));
      auto semicolon = type.find(';');
      auto media = type.substr(0, semicolon);
      while (!media.empty() && media.back() == ' ') media.pop_back();
      if (media != "application/json")
        return HttpError(ctx, 415, "Use application/json with UTF-8 JSON");
      if (semicolon != std::string::npos) {
        auto parameters = type.substr(semicolon + 1);
        parameters.erase(std::remove(parameters.begin(), parameters.end(), ' '),
                         parameters.end());
        if (parameters != "charset=utf-8" && parameters != "charset=\"utf-8\"")
          return HttpError(ctx, 415, "Only UTF-8 JSON is supported");
      }
      const auto declared = ctx->header("Content-Length");
      if (!declared.empty()) {
        uint64_t count = 0;
        const auto parsed = std::from_chars(
            declared.data(), declared.data() + declared.size(), count);
        if (parsed.ec != std::errc{} ||
            parsed.ptr != declared.data() + declared.size() ||
            count > kBodyLimit)
          return HttpError(ctx, 413, "MCP body exceeds 64 MiB");
      }
      {
        std::lock_guard lock(mutex);
        if (stopping) return HttpError(ctx, 503, "MCP transport is stopping");
        auto found = sessions.find(ctx->param("session_id"));
        if (found == sessions.end() || found->second->phase == Phase::kClosed)
          return HttpError(ctx, 404,
                           "Unknown MCP session; connect to /mcp/sse first");
      }
      if (Lower(ctx->header("Expect")) == "100-continue")
        ctx->writer->write("HTTP/1.1 100 Continue\r\n\r\n");
    } else if (phase == HP_BODY) {
      if (size > kBodyLimit - ctx->request->body.size())
        return HttpError(ctx, 413, "MCP body exceeds 64 MiB");
      ctx->request->body.append(data, size);
    } else if (phase == HP_MESSAGE_COMPLETE) {
      std::shared_ptr<Session> session;
      {
        std::lock_guard lock(mutex);
        auto found = sessions.find(ctx->param("session_id"));
        if (stopping || found == sessions.end() ||
            found->second->phase == Phase::kClosed)
          return HttpError(ctx, 404, "MCP session is closed");
        session = found->second;
      }
      try {
        Receive(session, ctx->request->body);
      } catch (...) {
        Send(session, RpcError(nullptr, -32603, "Internal error"));
      }
      ctx->response->status_code = HTTP_STATUS_ACCEPTED;
      ctx->writer->End();
      return HTTP_STATUS_ACCEPTED;
    }
    return HTTP_STATUS_UNFINISHED;
  }
  void Stop() noexcept {
    std::call_once(stopped, [&] {
      std::vector<std::shared_ptr<Session>> closing;
      {
        std::lock_guard lock(mutex);
        stopping = true;
        for (auto& [id, session] : sessions) {
          closing.push_back(session);
          for (auto& [key, source] : session->active) source.request_stop();
        }
        jobs.clear();
      }
      ready.notify_all();
      workers.clear();  // jthread joins all service calls while IO loops still
                        // run.
      auto self = shared_from_this();
      std::vector<std::future<void>> closed;
      for (auto& session : closing) {
        auto done = std::make_shared<std::promise<void>>();
        closed.push_back(done->get_future());
        session->loop->runInLoop([self, session, done] {
          self->Close(session);
          done->set_value();
        });
      }
      for (auto& done : closed) done.wait();
    });
  }
};

MCPAdapter::MCPAdapter(std::shared_ptr<State> state)
    : state_(std::move(state)) {}
VSResult<std::unique_ptr<MCPAdapter>> MCPAdapter::Create(
    std::shared_ptr<InferenceService> service, std::string host,
    uint16_t port) noexcept {
  try {
    auto adapter = std::unique_ptr<MCPAdapter>(new MCPAdapter(
        std::make_shared<State>(std::move(service), std::move(host), port)));
    for (int i = 0; i < 2; ++i)
      adapter->state_->workers.emplace_back(
          [state = adapter->state_] { state->Worker(); });
    return adapter;
  } catch (const std::exception& error) {
    return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError, error.what());
  }
}
MCPAdapter::~MCPAdapter() { Stop(); }
void MCPAdapter::Mount(hv::HttpService& service, hv::HttpServer& server) {
  const auto state = state_;
  service.GET("/mcp/sse", [state, &server](const HttpContextPtr& ctx) {
    // Server owns this route's executing IO loop; Stop drains before server
    // stop.
    return state->Open(ctx, server.loop());
  });
  service.POST("/mcp/messages",
               [state](const HttpContextPtr& ctx, http_parser_state phase,
                       const char* data, size_t size) {
                 return state->Post(ctx, phase, data, size);
               });
}
void MCPAdapter::Stop() noexcept {
  if (state_) state_->Stop();
}
}  // namespace vision_simple
