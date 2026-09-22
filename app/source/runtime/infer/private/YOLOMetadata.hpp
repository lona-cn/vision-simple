#pragma once

#include <charconv>
#include <climits>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vision_simple::detail {
// Python repr and JSON share the delimiters used by Ultralytics metadata.
class YOLOMetadataReader {
 public:
  explicit YOLOMetadataReader(std::string_view text) : text_(text) {}
  void Space() {
    while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\t' ||
                                   text_[pos_] == '\r' || text_[pos_] == '\n'))
      ++pos_;
  }
  bool Take(char c) {
    Space();
    if (pos_ == text_.size() || text_[pos_] != c) return false;
    ++pos_;
    return true;
  }
  bool End() {
    Space();
    return pos_ == text_.size();
  }
  bool Quoted() {
    Space();
    return pos_ < text_.size() && (text_[pos_] == '\'' || text_[pos_] == '"');
  }
  bool String(std::string& result) {
    if (!Quoted()) return false;
    const char quote = text_[pos_++];
    result.clear();
    while (pos_ < text_.size()) {
      char c = text_[pos_++];
      if (c == quote) return true;
      if (static_cast<unsigned char>(c) < 32) return false;
      if (c != '\\') {
        result += c;
        continue;
      }
      if (pos_ == text_.size()) return false;
      c = text_[pos_++];
      switch (c) {
        case '\\':
        case '\'':
        case '"':
        case '/':
          result += c;
          break;
        case 'a':
          result += '\a';
          break;
        case 'b':
          result += '\b';
          break;
        case 'f':
          result += '\f';
          break;
        case 'n':
          result += '\n';
          break;
        case 'r':
          result += '\r';
          break;
        case 't':
          result += '\t';
          break;
        case 'v':
          result += '\v';
          break;
        case 'x':
        case 'u':
        case 'U': {
          uint32_t point = 0;
          if (!Hex(c == 'x' ? 2 : c == 'u' ? 4 : 8, point)) return false;
          if (point >= 0xd800 && point <= 0xdbff && c == 'u') {
            if (pos_ + 2 > text_.size() || text_[pos_] != '\\' ||
                text_[pos_ + 1] != 'u')
              return false;
            pos_ += 2;
            uint32_t low = 0;
            if (!Hex(4, low) || low < 0xdc00 || low > 0xdfff) return false;
            point = 0x10000 + ((point - 0xd800) << 10) + low - 0xdc00;
          }
          if (point > 0x10ffff || (point >= 0xd800 && point <= 0xdfff))
            return false;
          if (point < 0x80)
            result += static_cast<char>(point);
          else if (point < 0x800) {
            result += static_cast<char>(0xc0 | (point >> 6));
            result += static_cast<char>(0x80 | (point & 63));
          } else {
            if (point >= 0x10000) {
              result += static_cast<char>(0xf0 | (point >> 18));
              result += static_cast<char>(0x80 | ((point >> 12) & 63));
            } else
              result += static_cast<char>(0xe0 | (point >> 12));
            result += static_cast<char>(0x80 | ((point >> 6) & 63));
            result += static_cast<char>(0x80 | (point & 63));
          }
          break;
        }
        default:
          return false;  // Never silently alter an unsupported escape.
      }
    }
    return false;
  }
  template <class Visitor>
  bool Dictionary(Visitor visit) {
    if (!Take('{')) return false;
    if (Take('}')) return End();
    do {
      std::string key;
      if (Quoted()) {
        if (!String(key)) return false;
      } else {
        Space();
        const size_t start = pos_;
        while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9')
          ++pos_;
        if (start == pos_) return false;
        key = text_.substr(start, pos_ - start);
      }
      if (!Take(':')) return false;
      Space();
      const size_t start = pos_;
      if (!Value(0) || !visit(key, text_.substr(start, pos_ - start)))
        return false;
      if (Take('}')) return End();
      if (!Take(',')) return false;
      if (Take('}')) return End();
    } while (true);
  }

 private:
  bool Hex(size_t count, uint32_t& value) {
    if (count > text_.size() - pos_) return false;
    while (count--) {
      const char c = text_[pos_++];
      const int digit = c >= '0' && c <= '9'   ? c - '0'
                        : c >= 'a' && c <= 'f' ? c - 'a' + 10
                        : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                               : -1;
      if (digit < 0) return false;
      value = (value << 4) | static_cast<uint32_t>(digit);
    }
    return true;
  }
  bool Value(size_t depth) {
    Space();
    if (pos_ == text_.size() || depth > 64) return false;
    if (Quoted()) {
      std::string ignored;
      return String(ignored);
    }
    const char c = text_[pos_];
    if (c == '{' || c == '[' || c == '(') {
      ++pos_;
      const char close = c == '{' ? '}' : c == '[' ? ']' : ')';
      while (!Take(close)) {
        if (!Value(depth + 1)) return false;
        if (Take(close)) return true;
        if (!Take(',') && !(c == '{' && Take(':'))) return false;
      }
      return true;
    }
    const size_t start = pos_;
    while (pos_ < text_.size()) {
      const char next = text_[pos_];
      if (next == ',' || next == ':' || next == '}' || next == ']' ||
          next == ')' || next == ' ' || next == '\t' || next == '\r' ||
          next == '\n')
        break;
      if (next == '\'' || next == '"' || next == '{' || next == '[' ||
          next == '(')
        return false;
      ++pos_;
    }
    return pos_ != start;
  }
  std::string_view text_;
  size_t pos_ = 0;
};

inline bool ParseYOLOClassNames(std::string_view text,
                                std::vector<std::string>& names) {
  names.clear();
  YOLOMetadataReader reader(text);
  const bool valid =
      reader.Dictionary([&](const std::string& key, std::string_view value) {
        size_t index = 0;
        const auto parsed =
            std::from_chars(key.data(), key.data() + key.size(), index);
        if (parsed.ec != std::errc{} || parsed.ptr != key.data() + key.size() ||
            index != names.size() || names.size() >= INT_MAX)
          return false;
        std::string name;
        YOLOMetadataReader quoted(value);
        if (!quoted.String(name) || !quoted.End() || name.empty()) return false;
        names.push_back(std::move(name));
        return true;
      });
  return valid && !names.empty();
}

inline bool YOLOMetadataTrue(std::string_view text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) return false;
  text = text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
  return text == "True" || text == "true" || text == "1";
}
inline bool ParseYOLOExportArgs(std::string_view text, bool& exported_nms) {
  YOLOMetadataReader reader(text);
  return reader.Dictionary([&](const std::string& key, std::string_view value) {
    if ((key == "nms" || key == "end2end") && YOLOMetadataTrue(value))
      exported_nms = true;
    return true;
  });
}
}  // namespace vision_simple::detail
