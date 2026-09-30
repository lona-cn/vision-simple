#pragma once

#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace vision_simple {
inline bool HTTPASCIIEqual(std::string_view a, std::string_view b) noexcept {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    const auto lower = [](char c) {
      return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c;
    };
    if (lower(a[i]) != lower(b[i])) return false;
  }
  return true;
}

// Shared backend-authority policy; this is not caller authentication.
class HTTPAuthority {
  std::string host_;
  uint16_t port_;

 public:
  HTTPAuthority(std::string host, uint16_t port)
      : host_(std::move(host)), port_(port) {}

  bool Accepts(std::string_view view, uint16_t default_port) const noexcept {
    std::string_view name, port_text;
    if (view.starts_with('[')) {
      const auto end = view.find(']');
      if (end == std::string_view::npos) return false;
      name = view.substr(1, end - 1);
      // Brackets denote an IPv6 literal, never a DNS or IPv4 authority.
      if (name.find(':') == std::string_view::npos) return false;
      if (end + 1 < view.size()) {
        if (view[end + 1] != ':') return false;
        port_text = view.substr(end + 2);
        if (port_text.empty()) return false;
      }
    } else {
      const auto colon = view.find(':');
      name = view.substr(0, colon);
      if (colon != std::string_view::npos) {
        port_text = view.substr(colon + 1);
        if (port_text.empty()) return false;
      }
    }
    unsigned number = default_port;
    if (!port_text.empty()) {
      const auto parsed = std::from_chars(
          port_text.data(), port_text.data() + port_text.size(), number);
      if (parsed.ec != std::errc{} ||
          parsed.ptr != port_text.data() + port_text.size()) return false;
    }
    auto explicit_host = std::string_view(host_);
    if (explicit_host.starts_with('[') && explicit_host.ends_with(']'))
      explicit_host = explicit_host.substr(1, explicit_host.size() - 2);
    const bool configured = explicit_host != "0.0.0.0" &&
                            explicit_host != "::" && explicit_host != "*" &&
                            !explicit_host.empty();
    return number == port_ &&
           (HTTPASCIIEqual(name, "localhost") || name == "127.0.0.1" ||
            name == "::1" || (configured && HTTPASCIIEqual(name, explicit_host)));
  }

  bool Trusted(std::string_view host, bool origin_present,
               std::string_view origin) const noexcept {
    if (!Accepts(host, 80)) return false;
    if (origin.empty()) return !origin_present;
    uint16_t default_port;
    if (origin.starts_with("http://")) {
      origin.remove_prefix(7);
      default_port = 80;
    } else if (origin.starts_with("https://")) {
      origin.remove_prefix(8);
      default_port = 443;
    } else return false;
    if (origin.find_first_of("/?#@\\") != std::string_view::npos) return false;
    return Accepts(origin, default_port);
  }
};
}  // namespace vision_simple
