#pragma once
#include <string>
#include <expected>
#include <memory>

namespace vision_simple
{
    class HTTPServer
    {
    public:
        // using CreateResult = std::expected<std::unique_ptr<HTTPServer>, >;
        void Serve();

        static void Create(const std::string& host, uint16_t port) noexcept;
    };
}
