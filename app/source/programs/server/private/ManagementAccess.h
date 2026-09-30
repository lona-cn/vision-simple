#pragma once

#include <cstdlib>
#include <expected>
#include <string>
#include <string_view>
#include <utility>

#include "HTTPAuthority.h"

namespace vision_simple {
class ManagementAccess {
  std::string token_;

 public:
  static std::expected<ManagementAccess, const char*> Load(
      const std::string& environment) {
    if (environment.empty()) return ManagementAccess{};
    const auto letter = [](unsigned char c) {
      return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
    };
    if (environment.size() > 128 || !letter(environment.front()))
      return std::unexpected("Invalid management token environment name");
    for (unsigned char c : environment)
      if (!letter(c) && !(c >= '0' && c <= '9'))
        return std::unexpected("Invalid management token environment name");
    const char* value = std::getenv(environment.c_str());
    if (!value || !*value)
      return std::unexpected("Management token environment is missing or empty");
    size_t length = 0;
    for (; value[length] && length <= 4096; ++length)
      if (static_cast<unsigned char>(value[length]) < 0x21 ||
          static_cast<unsigned char>(value[length]) > 0x7e)
        return std::unexpected("Management token must contain visible ASCII without whitespace and at most 4096 bytes");
    if (length > 4096)
      return std::unexpected("Management token must contain visible ASCII without whitespace and at most 4096 bytes");
    ManagementAccess access;
    access.token_.assign(value, length);
    return access;
  }

  bool Enabled() const noexcept { return !token_.empty(); }

  bool Authorized(std::string_view authorization) const noexcept {
    if (!Enabled() || authorization.size() < 7 ||
        !HTTPASCIIEqual(authorization.substr(0, 6), "Bearer") ||
        authorization[6] != ' ') return false;
    const auto candidate = authorization.substr(7);
    // Fixed work for the configured secret length; neither matching prefix nor
    // candidate content changes the amount of comparison work. Volatile keeps
    // this reduction from becoming an early-exit library comparison.
    volatile size_t difference = candidate.size() ^ token_.size();
    for (size_t i = 0; i < token_.size(); ++i) {
      const unsigned char c = i < candidate.size() ? candidate[i] : 0;
      difference = difference | (c ^ static_cast<unsigned char>(token_[i]));
    }
    return difference == 0;
  }
};
}  // namespace vision_simple
