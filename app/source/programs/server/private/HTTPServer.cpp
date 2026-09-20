#include "HTTPServer.h"

#include <hv/HttpServer.h>
#include <hv/hlog.h>
#include <hv/hv.h>
#include <turbobase64/turbob64.h>
#include <ylt/struct_json/json_writer.h>

#include <nlohmann/json.hpp>
#include <magic_enum.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <optional>
#include <shared_mutex>
#include <string_view>

#include "IOUtil.h"
#include "Infer.h"
#include "LogFacade.h"
#include "Logger.h"
#include "VisionSimpleConfig.h"
#define LOG_DOMAIN_NAME "HTTPServer"

namespace vision_simple {
namespace {
struct InferRequest {
  std::string model;
  std::vector<std::string> images;
};

enum class FailureStage {
  Request,
  UnknownModel,
  Image,
  ModelLoad,
  ModelConfig,
  Inference,
  Serialization
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

PreparedResponse ErrorResponse(const RequestStage& stage) {
  std::string_view code;
  std::string_view message;
  http_status status = HTTP_STATUS_INTERNAL_SERVER_ERROR;
  switch (stage.failure) {
    case FailureStage::Request:
      code = "invalid_request";
      message =
          "Request must contain a nonempty model and an array of image strings";
      status = HTTP_STATUS_BAD_REQUEST;
      break;
    case FailureStage::UnknownModel:
      code = "unknown_model";
      message = "Model is not configured";
      status = HTTP_STATUS_BAD_REQUEST;
      break;
    case FailureStage::Image:
      code = "invalid_image";
      message = "Image cannot be decoded";
      status = HTTP_STATUS_BAD_REQUEST;
      break;
    case FailureStage::ModelLoad:
      code = "model_load_failed";
      message = "Model cannot be loaded";
      break;
    case FailureStage::ModelConfig:
      code = "model_config_failed";
      message = "Model configuration cannot be read";
      break;
    case FailureStage::Inference:
      code = "inference_failed";
      message = "Image inference failed";
      break;
    case FailureStage::Serialization:
      code = "internal_error";
      message = "Response cannot be serialized";
      break;
  }
  // Only fixed server-owned strings and an integer are interpolated here.
  // Error responses do not depend on the serializer which may have just failed.
  return {status,
          std::format(
              R"({{"error":{{"code":"{}","message":"{}","image_index":{}}}}})",
              code, message,
              stage.image_index ? std::to_string(*stage.image_index) : "null")};
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
  }
  // This is the only response owner. All processing and serialization finish
  // before libhv's send() ends the response; its returned status is not a
  // setter.
  ctx->setStatus(response.status);
  ctx->setContentType(APPLICATION_JSON);
  ctx->response->body = std::move(response.body);
  return ctx->send();
}

std::optional<InferRequest> ParseRequest(const std::string& body) {
  const auto json = nlohmann::json::parse(body);
  if (!json.is_object()) return std::nullopt;
  const auto model = json.find("model");
  const auto images = json.find("images");
  if (model == json.end() || !model->is_string() ||
      model->get_ref<const std::string&>().empty() || images == json.end() ||
      !images->is_array()) {
    return std::nullopt;
  }
  // Validate every field before model lookup, including later image entries.
  for (const auto& image : *images) {
    if (!image.is_string()) return std::nullopt;
  }
  return InferRequest{model->get<std::string>(),
                      images->get<std::vector<std::string>>()};
}

int Base64Digit(unsigned char value) noexcept {
  if (value >= 'A' && value <= 'Z') return value - 'A';
  if (value >= 'a' && value <= 'z') return value - 'a' + 26;
  if (value >= '0' && value <= '9') return value - '0' + 52;
  if (value == '+') return 62;
  if (value == '/') return 63;
  return -1;
}

bool IsBase64(std::string_view text) noexcept {
  if (text.empty() || text.size() % 4 != 0) return false;
  const size_t padding =
      text.back() == '=' ? (text[text.size() - 2] == '=' ? 2 : 1) : 0;
  const size_t digits = text.size() - padding;
  for (size_t i = 0; i < digits; ++i) {
    if (Base64Digit(static_cast<unsigned char>(text[i])) < 0) return false;
  }
  // Reject non-canonical padding bits as well as interior/excess '='.
  const int last = Base64Digit(static_cast<unsigned char>(text[digits - 1]));
  return (padding != 2 || (last & 15) == 0) &&
         (padding != 1 || (last & 3) == 0);
}

std::optional<std::vector<cv::Mat>> DecodeImages(
    const std::vector<std::string>& encoded, RequestStage& stage) {
  stage = {FailureStage::Image, std::nullopt};
  std::vector<cv::Mat> images;
  images.reserve(encoded.size());
  for (size_t i = 0; i < encoded.size(); ++i) {
    stage.image_index = i;
    const auto& text = encoded[i];
    if (!IsBase64(text)) return std::nullopt;
    const auto* data = reinterpret_cast<const unsigned char*>(text.data());
    const size_t length = tb64declen(data, text.size());
    if (length == 0) return std::nullopt;
    std::vector<uint8_t> bytes(length);
    const size_t decoded = tb64dec(data, text.size(), bytes.data());
    if (decoded == 0 || decoded != length) return std::nullopt;
    auto image = cv::imdecode(bytes, cv::IMREAD_COLOR);
    if (image.empty() || image.dims != 2 || image.rows <= 0 ||
        image.cols <= 0 || image.type() != CV_8UC3) {
      return std::nullopt;
    }
    images.emplace_back(std::move(image));
  }
  return images;
}
}  // namespace

struct YOLODetectedObject {
  int32_t class_id;
  float confidence;
  int bbox[4];
};

struct InferYOLOResponse {
  std::vector<std::string_view> class_names;
  std::vector<std::vector<YOLODetectedObject>> results;
};

struct OCRLine {
  std::string line;
  float confidence;
  int bbox[4];
};

struct InferOCRResponse {
  std::vector<std::vector<OCRLine>> results;
};

class HTTPServerImpl : public HTTPServer {
  HTTPServerOptions options_;
  hv::HttpService http_service_;
  hv::HttpServer http_server_;
  std::unique_ptr<InferContext> infer_context_;
  std::shared_mutex yolo_models_cache_mutex_;
  std::map<std::string, std::unique_ptr<InferYOLO>> yolo_models_cache_;
  std::shared_mutex ocr_models_cache_mutex_;
  std::map<std::string, std::unique_ptr<InferOCR>> ocr_models_cache_;

  VSResult<std::optional<std::reference_wrapper<InferYOLO>>> GetYOLOModel(
      const std::string& name) {
    {
      std::shared_lock lock{yolo_models_cache_mutex_};
      if (auto it = yolo_models_cache_.find(name);
          it != yolo_models_cache_.end())
        return *it->second;
    }
    std::unique_lock lock{yolo_models_cache_mutex_};
    if (auto it = yolo_models_cache_.find(name); it != yolo_models_cache_.end())
      return *it->second;
    // load model
    auto config_result = Config::Instance();
    if (!config_result)
      return std::unexpected(std::move(config_result.error()));
    auto& config = config_result->get();
    if (auto it = std::ranges::find_if(
            config.model_config().yolo,
            [&name](const auto& item) { return name == item.name; });
        it != config.model_config().yolo.end()) {
      auto& model_info = *it;
      auto version_opt = magic_enum::enum_cast<YOLOVersion>(model_info.version);
      if (!version_opt)
        return std::unexpected{VisionSimpleError{
            VisionSimpleErrorCode::kModelError,
            std::format("unknown yolo version: {}", model_info.version)}};
      auto version = *version_opt;
      auto data_result = ReadAll(model_info.path);
      if (!data_result) return std::unexpected(std::move(data_result.error()));
      auto& device_str = options_.OptionOrPut(
          HTTPSERVER_OPT_KEY_INFER_DEVICE, HTTPSERVER_OPT_DEFVAL_INFER_DEVICE);
      int device_id{0};
      try {
        device_id = std::stoi(device_str);
      } catch (std::exception& _) {
        return std::unexpected{
            VisionSimpleError{VisionSimpleErrorCode::kParameterError,
                              "device_id is not a integer: " + device_str}};
      }
      auto infer_yolo_result = InferYOLO::Create(
          *infer_context_, data_result->span(), version, device_id);
      if (!infer_yolo_result) {
        return std::unexpected{VisionSimpleError{
            VisionSimpleErrorCode::kModelError,
            std::format("unable to create infer yolo model:{},message:{} ",
                        name, infer_yolo_result.error().message)}};
      }
      yolo_models_cache_.emplace(name, std::move(*infer_yolo_result));
      LogFacade::Info("http", std::format("yolo model '{}' loaded", name));
      return *yolo_models_cache_[name];
    }
    return std::optional<std::reference_wrapper<InferYOLO>>{};
  }

  VSResult<std::optional<std::reference_wrapper<InferOCR>>> GetOCRModel(
      const std::string& name) {
    {
      std::shared_lock lock{ocr_models_cache_mutex_};
      if (auto it = ocr_models_cache_.find(name); it != ocr_models_cache_.end())
        return *it->second;
    }
    std::unique_lock lock{ocr_models_cache_mutex_};
    if (auto it = ocr_models_cache_.find(name); it != ocr_models_cache_.end())
      return *it->second;
    auto config_result = Config::Instance();
    if (!config_result)
      return std::unexpected(std::move(config_result.error()));
    auto& config = config_result->get();
    if (auto it = std::ranges::find_if(
            config.model_config().ocr,
            [&name](const auto& item) { return name == item.name; });
        it != config.model_config().ocr.end()) {
      const auto& model_info = *it;
      auto model_type_opt =
          magic_enum::enum_cast<OCRModelType>(model_info.version);
      if (!model_type_opt)
        return MK_VSERROR(
            VisionSimpleErrorCode::kParameterError,
            std::format("unknown version:{}", model_info.version));
      auto model_type = *model_type_opt;
      auto& device_str = options_.OptionOrPut(
          HTTPSERVER_OPT_KEY_INFER_DEVICE, HTTPSERVER_OPT_DEFVAL_INFER_DEVICE);
      int device_id{0};
      try {
        device_id = std::stoi(device_str);
      } catch (std::exception& _) {
        return std::unexpected{
            VisionSimpleError{VisionSimpleErrorCode::kParameterError,
                              "device_id is not a integer: " + device_str}};
      }
      auto infer_ocr_result = InferOCR::Create(
          *infer_context_, model_info.char_dict_path, model_info.det_path,
          model_info.rec_path, model_type, device_id);
      if (!infer_ocr_result) {
        return std::unexpected{VisionSimpleError{
            VisionSimpleErrorCode::kModelError,
            std::format("unable to create infer ocr model:{},message:{} ", name,
                        infer_ocr_result.error().message)}};
      }
      ocr_models_cache_.emplace(name, std::move(*infer_ocr_result));
      LogFacade::Info("http",
                      std::format("ocr model loaded (det={}, rec={})",
                                  model_info.det_path, model_info.rec_path));
      return *ocr_models_cache_[name];
    }
    return std::optional<std::reference_wrapper<InferOCR>>{};
  }

 public:
  explicit HTTPServerImpl(HTTPServerOptions&& options,
                          std::unique_ptr<InferContext>&& infer_context)
      : HTTPServer{},
        options_(std::move(options)),
        http_service_(),
        http_server_(),
        infer_context_{std::move(infer_context)},
        yolo_models_cache_{} {
    // static resource
    http_service_.Static("/", options_
                                  .OptionOrPut(HTTPSERVER_OPT_KEY_STATIC_DIR,
                                               HTTPSERVER_OPT_DEFVAL_STATIC_DIR)
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
    // /v0/infer/models
    http_service_.GET("/v0/infer/models", [this](const HttpContextPtr& ctx) {
      return this->HandleInferModels(ctx);
    });
    http_service_.Use([](const HttpContextPtr& ctx) {
      Logger::Instance()->get().Info(
          LOG_DOMAIN_NAME,
          std::format("{}:{} -> {}", ctx->ip(), ctx->port(), ctx->url()));
      return HTTP_STATUS_NEXT;
    });
    http_service_.AllowCORS();
    http_service_.enable_access_log = 0;

    http_server_.setHost(options_.host.c_str());
    http_server_.port = options_.port;
    http_server_.service = &http_service_;
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
  }

  const HTTPServerOptions& options() const noexcept override {
    return options_;
  }

  HTTPServerResult<void> Run() noexcept override { return Start(true); }

  HTTPServerResult<void> StartAsync() noexcept override { return Start(false); }

  void Stop() noexcept override { http_server_.stop(); }

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
    return HandleRequest(
        ctx, {FailureStage::ModelConfig, std::nullopt},
        [](RequestStage& stage) -> PreparedResponse {
          auto config_result = Config::Instance();
          if (!config_result) {
            LogFailure(config_result.error().message);
            return ErrorResponse(stage);
          }
          const auto& model_config = config_result->get().model_config();
          std::map<std::string_view, std::vector<std::string_view>> model_list;
          auto& yolo_list = model_list["yolo"];
          auto& ocr_list = model_list["ocr"];
          for (const auto& info : model_config.yolo)
            yolo_list.emplace_back(info.name);
          for (const auto& info : model_config.ocr)
            ocr_list.emplace_back(info.name);
          stage = {FailureStage::Serialization, std::nullopt};
          PreparedResponse response;
          struct_json::to_json(model_list, response.body);
          return response;
        });
  }

  int HandleInferYOLO(const HttpContextPtr& ctx) {
    return HandleRequest(
        ctx, {}, [this, &ctx](RequestStage& stage) -> PreparedResponse {
          auto request = ParseRequest(ctx->body());
          if (!request) return ErrorResponse(stage);
          stage = {FailureStage::ModelConfig, std::nullopt};
          if (auto config = Config::Instance(); !config) {
            LogFailure(config.error().message);
            return ErrorResponse(stage);
          }
          stage = {FailureStage::ModelLoad, std::nullopt};
          auto model = GetYOLOModel(request->model);
          if (!model) {
            LogFailure(model.error().message);
            return ErrorResponse(stage);
          }
          if (!*model)
            return ErrorResponse({FailureStage::UnknownModel, std::nullopt});
          auto& infer = model->value().get();
          auto images = DecodeImages(request->images, stage);
          if (!images) return ErrorResponse(stage);
          std::vector<YOLOFrameResult> all_results;
          all_results.reserve(images->size());
          for (size_t i = 0; i < images->size(); ++i) {
            stage = {FailureStage::Inference, i};
            auto result = infer.Run((*images)[i], 0.125f);
            if (!result) {
              LogFailure(result.error().message);
              return ErrorResponse(stage);
            }
            all_results.emplace_back(std::move(*result));
          }
          stage = {FailureStage::Serialization, std::nullopt};
          InferYOLOResponse body;
          body.class_names.assign(infer.class_names().cbegin(),
                                  infer.class_names().cend());
          body.results.reserve(all_results.size());
          for (const auto& frame : all_results) {
            std::vector<YOLODetectedObject> objects;
            objects.reserve(frame.results.size());
            for (const auto& result : frame.results) {
              const auto& box = result.bbox;
              objects.emplace_back(
                  YOLODetectedObject{result.class_id,
                                     result.confidence,
                                     {box.x, box.y, box.width, box.height}});
            }
            body.results.emplace_back(std::move(objects));
          }
          PreparedResponse response;
          struct_json::to_json(body, response.body);
          return response;
        });
  }

  int HandleInferOCR(const HttpContextPtr& ctx) {
    return HandleRequest(
        ctx, {}, [this, &ctx](RequestStage& stage) -> PreparedResponse {
          auto request = ParseRequest(ctx->body());
          if (!request) return ErrorResponse(stage);
          stage = {FailureStage::ModelConfig, std::nullopt};
          if (auto config = Config::Instance(); !config) {
            LogFailure(config.error().message);
            return ErrorResponse(stage);
          }
          stage = {FailureStage::ModelLoad, std::nullopt};
          auto model = GetOCRModel(request->model);
          if (!model) {
            LogFailure(model.error().message);
            return ErrorResponse(stage);
          }
          if (!*model)
            return ErrorResponse({FailureStage::UnknownModel, std::nullopt});
          auto& infer = model->value().get();
          auto images = DecodeImages(request->images, stage);
          if (!images) return ErrorResponse(stage);
          std::vector<OCRFrameResult> all_results;
          all_results.reserve(images->size());
          for (size_t i = 0; i < images->size(); ++i) {
            stage = {FailureStage::Inference, i};
            auto result = infer.Run((*images)[i], 0.125f);
            if (!result) {
              LogFailure(result.error().message);
              return ErrorResponse(stage);
            }
            all_results.emplace_back(std::move(*result));
          }
          stage = {FailureStage::Serialization, std::nullopt};
          InferOCRResponse body;
          body.results.reserve(all_results.size());
          for (auto& frame : all_results) {
            std::vector<OCRLine> lines;
            lines.reserve(frame.results.size());
            for (auto& result : frame.results) {
              const auto& box = result.rect;
              lines.emplace_back(
                  OCRLine{std::move(result.line),
                          result.confidence,
                          {box.x, box.y, box.width, box.height}});
            }
            body.results.emplace_back(std::move(lines));
          }
          PreparedResponse response;
          struct_json::to_json(body, response.body);
          return response;
        });
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
  auto infer_context = InferContext::Create(*infer_fw, *infer_ep);
  if (!infer_context)
    return std::unexpected(VisionSimpleError{
        VisionSimpleErrorCode::kModelError,
        std::format("unable to create infer context with {}:{},error:{}",
                    infer_fw_str, infer_ep_str,
                    infer_context.error().message)});
  Logger::Instance()->get().Info(
      LOG_DOMAIN_NAME, std::format("Execution Provider:{}", infer_ep_str));
  return std::make_unique<HTTPServerImpl>(std::move(options),
                                          std::move(*infer_context));
} catch (const std::exception& error) {
  return MK_VSERROR(
      VisionSimpleErrorCode::kRuntimeError,
      std::format("unable to create HTTP server: {}", error.what()));
}
