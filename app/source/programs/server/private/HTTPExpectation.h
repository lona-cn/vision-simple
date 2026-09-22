#pragma once

#include <string_view>

namespace vision_simple {
enum class HTTPExpectation { kNone, kContinue, kUnsupported };
inline HTTPExpectation ParseHTTPExpectation(std::string_view value) noexcept {
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
    value.remove_prefix(1);
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t'))
    value.remove_suffix(1);
  if (value.empty()) return HTTPExpectation::kNone;
  constexpr std::string_view token = "100-continue";
  if (value.size() != token.size()) return HTTPExpectation::kUnsupported;
  for (size_t i = 0; i < token.size(); ++i) {
    const char c =
        value[i] >= 'A' && value[i] <= 'Z' ? value[i] + ('a' - 'A') : value[i];
    if (c != token[i]) return HTTPExpectation::kUnsupported;
  }
  return HTTPExpectation::kContinue;
}
}  // namespace vision_simple
