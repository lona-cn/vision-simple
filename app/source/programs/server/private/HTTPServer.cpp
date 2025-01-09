#include "HTTPServer.h"

#include <span>
#include <hv/hv.h>
// #include <hv/HttpService.h>
#include <magic_enum.hpp>
#include <shared_mutex>
#include <hv/HttpServer.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <turbobase64/turbob64.h>
#include "Infer.h"
#include "VisionSimpleConfig.h"
#include <ylt/struct_json/json_reader.h>
#include <ylt/struct_json/json_writer.h>

#include "IOUtil.h"

namespace vision_simple
{
    struct InferYOLORequest
    {
        std::string model;
        std::vector<std::string> images;
    };

    struct YOLODetectedObject
    {
        int32_t class_id;
        float confidence;
        int bbox[4];
    };

    struct InferYOLOResponse
    {
        std::vector<std::string_view> class_names;
        std::vector<std::vector<YOLODetectedObject>> results;
    };

    class HTTPServerImpl : public HTTPServer
    {
        HTTPServerOptions options_;
        hv::HttpService http_service_;
        hv::HttpServer http_server_;
        std::unique_ptr<InferContext> infer_context_;
        std::shared_mutex yolo_models_cache_mutex_;
        std::map<std::string, std::unique_ptr<InferYOLO>> yolo_models_cache_;
        // std::map<std::string, std::unique_ptr<InferOCR>> ocr_models;

        std::expected<std::reference_wrapper<InferYOLO>, VisionSimpleError>
        GetYOLOModel(const std::string& name)
        {
            {
                std::shared_lock lock{yolo_models_cache_mutex_};
                if (auto it = yolo_models_cache_.find(name); it != yolo_models_cache_.end())return *it->second;
            }
            std::unique_lock lock{yolo_models_cache_mutex_};
            if (auto it = yolo_models_cache_.find(name); it != yolo_models_cache_.end())return *it->second;
            // load model
            auto config_result = Config::Instance();
            if (!config_result)return std::unexpected(std::move(config_result.error()));
            auto& config = *config_result;
            if (auto it = std::ranges::find_if(config.get().model_config().yolo, [&name](const auto& item)
            {
                return name == item.name;
            }); it != config.get().model_config().yolo.end())
            {
                auto& model_info = *it;
                auto version_opt = magic_enum::enum_cast<YOLOVersion>(model_info.version);
                if (!version_opt)
                    return std::unexpected{
                        VisionSimpleError{
                            VisionSimpleErrorCode::kModelError,
                            std::format("unknown yolo version: {}", model_info.version)
                        }
                    };
                auto version = *version_opt;
                auto data_result = ReadAll(name);
                if (!data_result)
                    return std::unexpected{
                        VisionSimpleError{
                            VisionSimpleErrorCode::kIOError,
                            "Failed to read YOLO model from file: " + name
                        }
                    };
                auto& device_str = options_.OptionOrPut(HTTPSERVER_OPTKEY_INFER_DEVICE,
                                                        HTTPSERVER_OPT_DEFVAL_INFER_DEVICE);
                int device{0};
                try
                {
                    device = std::stoi(device_str);
                }
                catch (std::exception& _)
                {
                    return std::unexpected{
                        VisionSimpleError{
                            VisionSimpleErrorCode::kParameterError,
                            "device_id is not a integer: " + device_str
                        }
                    };
                }
                auto infer_yolo_result = InferYOLO::Create(*infer_context_, data_result->span(), version, device);
                if (!infer_yolo_result)
                {
                    return std::unexpected{
                        VisionSimpleError{
                            VisionSimpleErrorCode::kModelError,
                            std::format("unable to create infer yolo model:{},message:{} ", name,
                                        infer_yolo_result.error().message)
                        }
                    };
                }
                yolo_models_cache_.emplace(name, std::move(*infer_yolo_result));
                return *yolo_models_cache_[name];
            }
            return std::unexpected{
                VisionSimpleError{
                    VisionSimpleErrorCode::kModelError,
                    "unable to find model: " + name
                }
            };
        }

    public:
        explicit HTTPServerImpl(HTTPServerOptions&& options, std::unique_ptr<InferContext>&& infer_context):
            HTTPServer{}, options_(std::move(options)),
            http_service_(),
            http_server_(),
            infer_context_{std::move(infer_context)},
            yolo_models_cache_{}

        {
            // initialize infer
            // register handles
            // static resource
            http_service_.Static(
                "/", options_.OptionOrPut(HTTPSERVER_OPT_KEY_STATIC_DIR,
                                          HTTPSERVER_OPT_DEFVAL_STATIC_DIR).c_str());
            // /infer/yolo
            http_service_.POST("/v1/infer/yolo", [this](const HttpContextPtr& ctx)
            {
                // ctx->request
                const auto& str = ctx->body();
                InferYOLORequest parsed_request;
                std::error_code error_code;
                struct_json::from_json(parsed_request, str, error_code);
                if (error_code)
                {
                    ctx->sendString(error_code.message());
                    return 400;
                }
                auto infer_result = GetYOLOModel(parsed_request.model);
                if (!infer_result)
                {
                    ctx->sendString(infer_result.error().message.c_str());
                    return 400;
                }
                auto& infer = infer_result->get();
                std::vector<cv::Mat> images;
                images.reserve(parsed_request.images.size());
                for (const auto& image_b64 : parsed_request.images)
                {
                    auto data_len = tb64declen(reinterpret_cast<const unsigned char*>(image_b64.c_str()),
                                               image_b64.size());
                    std::vector<uint8_t> data_vec(data_len);
                    tb64dec(reinterpret_cast<const unsigned char*>(image_b64.c_str()), image_b64.size(),
                            data_vec.data());
                    try
                    {
                        images.emplace_back(imdecode(data_vec, cv::IMREAD_COLOR));
                    }
                    catch (std::exception& e)
                    {
                        ctx->sendString(e.what());
                        return 400;
                    }
                }
                std::vector<YOLOFrameResult> all_results;
                all_results.reserve(images.size());
                for (const auto& image : images)
                {
                    if (auto result = infer.Run(image, 0.125))
                    {
                        all_results.emplace_back(*std::move(result));
                    }
                }
                InferYOLOResponse response;
                response.class_names = std::vector<std::string_view>(infer.class_names().cbegin(),
                                                                     infer.class_names().cend());
                response.results.reserve(all_results.size());
                //TODO: openmp
                for (const auto& frame_result : all_results)
                {
                    std::vector<YOLODetectedObject> objects;
                    objects.reserve(frame_result.results.size());
                    for (const auto& result : frame_result.results)
                    {
                        auto& bbox = result.bbox;
                        objects.emplace_back(YOLODetectedObject{
                            .class_id = result.class_id,
                            .confidence = result.confidence,
                            .bbox = {bbox.x, bbox.y, bbox.width, bbox.height},
                        });
                    }
                    response.results.emplace_back(std::move(objects));
                }
                std::string json_str;
                struct_json::to_json(std::move(response), json_str);
                ctx->send(json_str, APPLICATION_JSON);
                return 200;
            });

            // /infer/models
            http_service_.GET("/v1/infer/models", [this](const HttpContextPtr& ctx)
            {
                auto config_result = Config::Instance();
                if (!config_result)
                {
                    auto& err = config_result.error();
                    ctx->send(err.message.c_str());
                    return 500;
                }
                const auto& model_config = config_result->get().model_config();
                std::map<std::string_view, std::vector<std::string_view>> model_list;
                auto& yolo_list = model_list["yolo"];
                auto& ppocr_list = model_list["ppocr"];
                for (const auto& model_info : model_config.yolo)
                {
                    yolo_list.emplace_back(model_info.name);
                }
                for (const auto& model_info : model_config.ocr)
                {
                    ppocr_list.emplace_back(model_info.name);
                }
                ctx->sendJson(model_list);
                return 200;
            });

            http_service_.AllowCORS();


            http_server_.port = options_.port;
            http_server_.service = &http_service_;
        }

        const HTTPServerOptions& options() const noexcept override { return options_; }

        void Run() noexcept override
        {
            http_server_run(&http_server_);
        }

        void StartAsync() noexcept override
        {
            http_server_run(&http_server_, 0);
        }

        void Stop() noexcept override
        {
            http_server_.stop();
        }
    };
}

const std::string& vision_simple::HTTPServerOptions::OptionOrPut(const std::string& key,
                                                                 const std::string& default_value)
{
    if (auto it = options.find(key); it != options.end())
    {
        return it->second;
    }
    options.try_emplace(key, default_value);
    return options[key];
}

const std::string& vision_simple::HTTPServerOptions::OptionOrPut(const std::string_view& key,
                                                                 const std::string_view& default_value)
{
    return OptionOrPut(std::string(key), std::string(default_value));
}

vision_simple::HTTPServerResult<std::unique_ptr<vision_simple::HTTPServer>> vision_simple::HTTPServer::Create(
    HTTPServerOptions&& options)
{
    auto infer_fw_str = options.OptionOrPut(HTTPSERVER_OPTKEY_INFER_FRAMEWORK, HTTPSERVER_OPT_DEFVAL_INFER_FRAMEWORK);
    auto infer_ep_str = options.OptionOrPut(HTTPSERVER_OPTKEY_INFER_EP, HTTPSERVER_OPT_DEFVAL_INFER_EP);
    auto infer_fw = magic_enum::enum_cast<InferFramework>(infer_fw_str);
    auto infer_ep = magic_enum::enum_cast<InferEP>(infer_ep_str);
    if (!infer_fw || !infer_ep)
        return std::unexpected(
            VisionSimpleError{
                VisionSimpleErrorCode::kParameterError, std::format("unsupported infer_framework:{} or infer_ep:{}",
                                                                    infer_fw_str, infer_ep_str)
            }
        );
    auto infer_context = InferContext::Create(*infer_fw, *infer_ep);
    if (!infer_context)
        return std::unexpected(
            VisionSimpleError{
                VisionSimpleErrorCode::kModelError, std::format("unable to create infer context with {}:{},error:{}",
                                                                infer_fw_str, infer_ep_str,
                                                                infer_context.error().message)
            }
        );
    return std::make_unique<HTTPServerImpl>(std::move(options), std::move(*infer_context));
}
