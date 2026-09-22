#include "SubtitleVideoReader.h"

#include <fstream>
#include <vector>

#include "ImageCodec.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
// Media Foundation's public types include BITMAPINFOHEADER.
#ifdef NOGDI
#undef NOGDI
#endif
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <windows.h>
#include <wrl/client.h>
#endif

namespace vision_simple {
namespace {
constexpr uint64_t kPixels = 16777216;
constexpr uint64_t kBytes = 64ULL * 1024 * 1024;
constexpr uint32_t Four(char a, char b, char c, char d) {
  return uint32_t(uint8_t(a)) | (uint32_t(uint8_t(b)) << 8) |
         (uint32_t(uint8_t(c)) << 16) | (uint32_t(uint8_t(d)) << 24);
}
uint32_t U32(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}
bool Size(uint32_t w, uint32_t h) {
  return w && h && uint64_t(w) * h <= kPixels;
}
}  // namespace
struct SubtitleVideoReader::Impl {
  std::stop_token cancel;
  const char* failure = "unreadable_video";
  bool avi = false;
  std::ifstream input;
  uint64_t bytes = 0, cursor = 0, frames = 0, emitted = 0;
  uint32_t rate = 0, scale = 0, expected = 0, extended = 0;
  uint32_t width = 0, height = 0, stream_count = 0, video_stream = UINT32_MAX;
  std::vector<uint8_t> encoded;
  cv::Mat pixels;
  struct Range {
    uint64_t end, next;
    bool movi;
  };
  std::vector<Range> ranges;
  std::optional<int64_t> hint;
#ifdef _WIN32
  Microsoft::WRL::ComPtr<IMFSourceReader> reader;
  bool com = false, mf = false;
  bool native_eof = false;
  LONG stride = 0;
  int64_t previous_ticks = -1;
  ~Impl() {
    reader.Reset();
    if (mf) MFShutdown();
    if (com) CoUninitialize();
  }
#endif
  bool Get(uint64_t offset, void* data, size_t count) {
    if (cancel.stop_requested() || offset > bytes || count > bytes - offset)
      return false;
    input.seekg(static_cast<std::streamoff>(offset));
    input.read(static_cast<char*>(data), static_cast<std::streamsize>(count));
    return bool(input);
  }
  bool VideoChunk(uint32_t id) const {
    if (video_stream > 99) return false;
    return (id & 0xffff) == uint32_t('0' + video_stream / 10) +
                                (uint32_t('0' + video_stream % 10) << 8) &&
           ((id >> 16) == uint32_t('d') + (uint32_t('c') << 8) ||
            (id >> 16) == uint32_t('d') + (uint32_t('b') << 8));
  }
  bool Stream(uint64_t begin, uint64_t end) {
    const uint32_t index = stream_count++;
    bool header = false, format = false, video = false;
    uint32_t codec = 0;
    for (uint64_t at = begin; at < end;) {
      uint8_t h[8];
      if (end - at < 8 || !Get(at, h, 8)) return false;
      const uint32_t id = U32(h), size = U32(h + 4);
      const uint64_t next = at + 8 + size + (size & 1);
      if (next > end) return false;
      if (id == Four('s', 't', 'r', 'h')) {
        uint8_t data[56];
        if (header || size < sizeof(data) || !Get(at + 8, data, sizeof(data)))
          return false;
        header = true;
        video = U32(data) == Four('v', 'i', 'd', 's');
        if (video) {
          if (video_stream != UINT32_MAX || index > 99) return false;
          video_stream = index;
          codec = U32(data + 4);
          scale = U32(data + 20);
          rate = U32(data + 24);
          // Nonzero initial video sample offsets are not represented by this
          // CFR reader.
          if (U32(data + 28) != 0) return false;
          expected = U32(data + 32);
        }
      } else if (id == Four('s', 't', 'r', 'f')) {
        if (!header || format) return false;
        format = true;
        if (video) {
          uint8_t data[40];
          if (size < sizeof(data) || !Get(at + 8, data, sizeof(data)) ||
              U32(data) < 40 || U32(data) > size)
            return false;
          width = U32(data + 4);
          height = U32(data + 8);
          const auto compression = U32(data + 16);
          if ((codec != Four('M', 'J', 'P', 'G') &&
               codec != Four('m', 'j', 'p', 'g') && codec != 0) ||
              (compression != Four('M', 'J', 'P', 'G') &&
               compression != Four('m', 'j', 'p', 'g'))) {
            failure = "unsupported_video";
            return false;
          }
          if (!Size(width, height)) {
            failure = "frame_dimensions";
            return false;
          }
        }
      }
      at = next;
    }
    return header && (!video || (format && rate && scale));
  }
  bool Scan(uint64_t begin, uint64_t end, bool movi, unsigned depth) {
    if (depth > 16) return false;
    for (uint64_t at = begin; at < end;) {
      uint8_t h[12];
      if (end - at < 8 || !Get(at, h, 8)) return false;
      const uint32_t id = U32(h), size = U32(h + 4);
      const uint64_t next = at + 8 + size + (size & 1);
      if (next > end) return false;
      if (id == Four('L', 'I', 'S', 'T') || id == Four('R', 'I', 'F', 'F')) {
        if (size < 4 || !Get(at + 8, h + 8, 4)) return false;
        const uint32_t type = U32(h + 8);
        if (id == Four('R', 'I', 'F', 'F') &&
            (depth != 0 || type != (at == 0 ? Four('A', 'V', 'I', ' ')
                                            : Four('A', 'V', 'I', 'X'))))
          return false;
        if (type == Four('s', 't', 'r', 'l')) {
          if (movi || !Stream(at + 12, at + 8 + size)) return false;
        } else if (!Scan(at + 12, at + 8 + size,
                         movi || type == Four('m', 'o', 'v', 'i'), depth + 1))
          return false;
      } else if (id == Four('d', 'm', 'l', 'h') && !movi) {
        uint8_t data[4];
        if (size < 4 || extended || !Get(at + 8, data, 4)) return false;
        extended = U32(data);
      } else if (movi && VideoChunk(id)) {
        if (!size || ++frames > 1000000) {
          failure = "frame_limit";
          return false;
        }
      }
      at = next;
    }
    return true;
  }
  bool OpenAvi() {
    failure = "incomplete_video";
    if (!Scan(0, bytes, false, 0) || video_stream == UINT32_MAX || !frames ||
        frames != (extended ? extended : expected))
      return false;
    const uint64_t ticks = frames * uint64_t(scale) * 1000;
    hint = static_cast<int64_t>((ticks + rate - 1) / rate);
    if (*hint > 1800000) {
      failure = "duration_limit";
      return false;
    }
    ranges.push_back({bytes, bytes, false});
    return true;
  }
  Result ReadAvi(Frame& frame) {
    while (!ranges.empty()) {
      if (cancel.stop_requested()) return Result::kCancelled;
      if (cursor == ranges.back().end) {
        cursor = ranges.back().next;
        ranges.pop_back();
        continue;
      }
      uint8_t h[12];
      if (!Get(cursor, h, 8)) return Result::kError;
      const uint32_t id = U32(h), size = U32(h + 4);
      const uint64_t payload = cursor + 8, next = payload + size + (size & 1);
      if (next > ranges.back().end) return Result::kError;
      if (id == Four('L', 'I', 'S', 'T') || id == Four('R', 'I', 'F', 'F')) {
        if (size < 4 || !Get(payload, h + 8, 4)) return Result::kError;
        const uint32_t type = U32(h + 8);
        const bool movi =
            ranges.back().movi || type == Four('m', 'o', 'v', 'i');
        if (type == Four('s', 't', 'r', 'l')) {
          cursor = next;
          continue;
        }
        ranges.push_back({payload + size, next, movi});
        cursor = payload + 4;
        continue;
      }
      cursor = next;
      if (!ranges.back().movi || !VideoChunk(id)) continue;
      encoded.resize(size);
      if (!Get(payload, encoded.data(), size)) return Result::kError;
      if (size < 4 || encoded[0] != 0xff || encoded[1] != 0xd8 ||
          encoded[size - 2] != 0xff || encoded[size - 1] != 0xd9)
        return Result::kError;
      auto image = DecodeImageBytes(encoded, kPixels, &pixels);
      if (!image || uint32_t(image->cols) != width ||
          uint32_t(image->rows) != height) {
        failure = "frame_dimensions";
        return Result::kError;
      }
      frame.image = std::move(*image);
      frame.start_ms =
          static_cast<int64_t>(emitted * uint64_t(scale) * 1000 / rate);
      frame.end_ms = static_cast<int64_t>(
          (++emitted * uint64_t(scale) * 1000 + rate - 1) / rate);
      return Result::kFrame;
    }
    return emitted == frames ? Result::kEof : Result::kError;
  }
#ifdef _WIN32
  bool Dimensions(IMFMediaType* type) {
    UINT32 w = 0, h = 0;
    if (FAILED(MFGetAttributeSize(type, MF_MT_FRAME_SIZE, &w, &h)) ||
        !Size(w, h)) {
      failure = "frame_dimensions";
      return false;
    }
    width = w;
    height = h;
    return true;
  }
  bool OutputType() {
    Microsoft::WRL::ComPtr<IMFMediaType> type;
    if (cancel.stop_requested() ||
        FAILED(reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                           &type)) ||
        !Dimensions(type.Get()))
      return false;
    GUID subtype{};
    if (FAILED(type->GetGUID(MF_MT_SUBTYPE, &subtype)) ||
        subtype != MFVideoFormat_RGB32)
      return false;
    UINT32 value = 0;
    if (SUCCEEDED(type->GetUINT32(MF_MT_DEFAULT_STRIDE, &value)))
      stride = static_cast<LONG>(value);
    else if (FAILED(MFGetStrideForBitmapInfoHeader(MFVideoFormat_RGB32.Data1,
                                                   width, &stride)))
      return false;
    return stride != LONG_MIN &&
           uint64_t(stride < 0 ? -stride : stride) >= uint64_t(width) * 4;
  }
  bool OpenNative(const std::filesystem::path& path) {
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr)) return false;
    com = true;
    if (cancel.stop_requested() ||
        FAILED(MFStartup(MF_VERSION, MFSTARTUP_FULL)))
      return false;
    mf = true;
    Microsoft::WRL::ComPtr<IMFAttributes> attributes;
    if (FAILED(MFCreateAttributes(&attributes, 1)) ||
        FAILED(attributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING,
                                     TRUE)) ||
        cancel.stop_requested() ||
        FAILED(MFCreateSourceReaderFromURL(path.c_str(), attributes.Get(),
                                           &reader)))
      return false;
    if (cancel.stop_requested() ||
        FAILED(
            reader->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE)) ||
        FAILED(reader->SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                          TRUE)))
      return false;
    Microsoft::WRL::ComPtr<IMFMediaType> native, output;
    if (cancel.stop_requested() ||
        FAILED(reader->GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                          0, &native)) ||
        !Dimensions(native.Get()) || FAILED(MFCreateMediaType(&output)) ||
        FAILED(output->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)) ||
        FAILED(output->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32)) ||
        cancel.stop_requested() ||
        FAILED(reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                           nullptr, output.Get())))
      return false;
    return OutputType();
  }
  Result ReadNative(Frame& frame) {
    for (;;) {
      if (native_eof) return Result::kEof;
      if (cancel.stop_requested()) return Result::kCancelled;
      DWORD flags = 0, stream = 0;
      LONGLONG timestamp = 0;
      Microsoft::WRL::ComPtr<IMFSample> sample;
      HRESULT hr = reader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0,
                                      &stream, &flags, &timestamp, &sample);
      if (cancel.stop_requested()) return Result::kCancelled;
      if (FAILED(hr) || (flags & MF_SOURCE_READERF_ERROR))
        return Result::kError;
      if (flags & MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED) {
        Microsoft::WRL::ComPtr<IMFMediaType> type;
        if (FAILED(reader->GetNativeMediaType(stream, 0, &type)) ||
            !Dimensions(type.Get()))
          return Result::kError;
      }
      if ((flags & (MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED |
                    MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED)) &&
          !OutputType())
        return Result::kError;
      if (!sample) {
        if (flags & MF_SOURCE_READERF_ENDOFSTREAM) return Result::kEof;
        if (flags & MF_SOURCE_READERF_STREAMTICK) continue;
        return Result::kError;
      }
      LONGLONG duration = 0;
      if (FAILED(sample->GetSampleTime(&timestamp)) ||
          FAILED(sample->GetSampleDuration(&duration)) || timestamp < 0 ||
          timestamp <= previous_ticks || duration <= 0 ||
          timestamp > 18000000000LL || duration > 18000000000LL - timestamp) {
        failure = "invalid_timestamps";
        return Result::kError;
      }
      previous_ticks = timestamp;
      Microsoft::WRL::ComPtr<IMFMediaBuffer> buffer;
      if (cancel.stop_requested()) return Result::kCancelled;
      if (FAILED(sample->ConvertToContiguousBuffer(&buffer)))
        return Result::kError;
      BYTE* data = nullptr;
      DWORD length = 0;
      LONG pitch = stride;
      Microsoft::WRL::ComPtr<IMF2DBuffer2> surface;
      const bool two_dimensional = SUCCEEDED(buffer.As(&surface));
      if (cancel.stop_requested()) return Result::kCancelled;
      BYTE* allocation = nullptr;
      if (two_dimensional) {
        if (FAILED(surface->Lock2DSize(MF2DBuffer_LockFlags_Read, &data, &pitch,
                                       &allocation, &length)))
          return Result::kError;
      } else {
        if (FAILED(buffer->Lock(&data, nullptr, &length)))
          return Result::kError;
        allocation = data;
      }
      struct Unlock {
        IMFMediaBuffer* buffer;
        IMF2DBuffer2* surface;
        ~Unlock() {
          if (surface)
            surface->Unlock2D();
          else
            buffer->Unlock();
        }
      } unlock{buffer.Get(), surface.Get()};
      if (pitch == LONG_MIN ||
          uint64_t(pitch < 0 ? -pitch : pitch) < uint64_t(width) * 4)
        return Result::kError;
      const uint64_t row_offset =
          uint64_t(pitch < 0 ? -pitch : pitch) * (height - 1);
      if (!two_dimensional && pitch < 0) {
        if (row_offset >= length) return Result::kError;
        data += row_offset;
      }
      const auto base = reinterpret_cast<uintptr_t>(allocation);
      const auto top = reinterpret_cast<uintptr_t>(data);
      if (top < base || top - base > length) return Result::kError;
      const uint64_t offset = top - base;
      if ((pitch < 0 && row_offset > offset) ||
          (pitch >= 0 && row_offset > length - offset))
        return Result::kError;
      const uint64_t last = pitch < 0 ? offset : offset + row_offset;
      if (uint64_t(width) * 4 > length - last) return Result::kError;
      pixels.create(height, width, CV_8UC3);
      for (uint32_t y = 0; y < height; ++y) {
        if (cancel.stop_requested()) return Result::kCancelled;
        const BYTE* src = data + int64_t(y) * pitch;
        auto* dst = pixels.ptr<uint8_t>(y);
        for (uint32_t x = 0; x < width; ++x) {
          dst[x * 3] = src[x * 4];
          dst[x * 3 + 1] = src[x * 4 + 1];
          dst[x * 3 + 2] = src[x * 4 + 2];
        }
      }
      frame.image = pixels;
      frame.start_ms = timestamp / 10000;
      frame.end_ms = (timestamp + duration + 9999) / 10000;
      native_eof = (flags & MF_SOURCE_READERF_ENDOFSTREAM) != 0;
      return Result::kFrame;
    }
  }
#endif
};
SubtitleVideoReader::SubtitleVideoReader() : impl_(std::make_unique<Impl>()) {}
SubtitleVideoReader::~SubtitleVideoReader() = default;
const char* SubtitleVideoReader::Open(const std::filesystem::path& path,
                                      std::stop_token cancel) {
  auto& p = *impl_;
  p.cancel = cancel;
  p.input.open(path, std::ios::binary | std::ios::ate);
  if (!p.input) return p.failure;
  const auto length = p.input.tellg();
  if (length < 12 || uint64_t(length) > kBytes) return p.failure;
  p.bytes = static_cast<uint64_t>(length);
  uint8_t signature[12];
  if (!p.Get(0, signature, 12)) return p.failure;
  p.avi = U32(signature) == Four('R', 'I', 'F', 'F') &&
          U32(signature + 8) == Four('A', 'V', 'I', ' ');
  if (p.avi) return p.OpenAvi() ? nullptr : p.failure;
  // Only sniffed binary containers reach the native resolver; never playlists,
  // image sequences, arbitrary URL schemes, or decoder-plugin fallbacks.
  const bool mp4 = U32(signature + 4) == Four('f', 't', 'y', 'p');
  const bool asf =
      U32(signature) == 0x75b22630 && U32(signature + 4) == 0x11cf668e;
  p.input.close();
  if (!mp4 && !asf) return "unsupported_video";
#ifdef _WIN32
  return p.OpenNative(path) ? nullptr : p.failure;
#else
  return "unsupported_video";
#endif
}
SubtitleVideoReader::Result SubtitleVideoReader::Read(Frame& frame) {
  if (impl_->cancel.stop_requested()) return Result::kCancelled;
  frame.image.release();
  if (impl_->avi) return impl_->ReadAvi(frame);
#ifdef _WIN32
  return impl_->ReadNative(frame);
#else
  return Result::kError;
#endif
}
const char* SubtitleVideoReader::error() const { return impl_->failure; }
std::optional<int64_t> SubtitleVideoReader::duration_hint() const {
  return impl_->hint;
}
}  // namespace vision_simple
