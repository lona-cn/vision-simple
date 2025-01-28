#pragma once
#include <expected>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "VisionSimpleCommon.h"

namespace vision_simple
{
    template <typename T>
    using HTTPServerResult = VSResult<T>;

    constexpr std::string_view HTTPSERVER_OPT_KEY_STATIC_DIR{"static_path"};
    constexpr std::string_view HTTPSERVER_OPT_KEY_INFER_FRAMEWORK{"infer_framework"};
    constexpr std::string_view HTTPSERVER_OPT_KEY_INFER_EP{"infer_ep"};
    constexpr std::string_view HTTPSERVER_OPT_KEY_INFER_DEVICE{"infer_device"};

    constexpr std::string_view HTTPSERVER_OPT_DEFVAL_STATIC_DIR{"assets/static"};
    constexpr std::string_view HTTPSERVER_OPT_DEFVAL_INFER_FRAMEWORK{"kONNXRUNTIME"};
    constexpr std::string_view HTTPSERVER_OPT_DEFVAL_INFER_EP{"kCPU"};
    constexpr std::string_view HTTPSERVER_OPT_DEFVAL_INFER_DEVICE{"0"};

    struct HTTPServerOptions
    {
        using ServerOptions = std::map<std::string, std::string>;

        std::string host;
        uint16_t port;
        ServerOptions options;

        const std::string& OptionOrPut(const std::string& key, const std::string& default_value);
        const std::string& OptionOrPut(std::string_view key, std::string_view default_value);
    };

    class HTTPServer
    {
    public:
        static HTTPServerResult<std::unique_ptr<HTTPServer>> Create(HTTPServerOptions&& options);
        HTTPServer() = default;
        HTTPServer(const HTTPServer& other) = delete;
        HTTPServer(HTTPServer&& other) noexcept = default;
        HTTPServer& operator=(const HTTPServer& other) = delete;
        HTTPServer& operator=(HTTPServer&& other) noexcept = default;
        virtual ~HTTPServer() = default;
        virtual const HTTPServerOptions& options() const noexcept = 0;
        virtual void Run() noexcept = 0;
        virtual void StartAsync() noexcept = 0;
        virtual void Stop() noexcept = 0;
    };
}
