#include "DebugArtifacts.h"
#include "ImageCodec.h"
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <array>
#include <cstdio>
#include <limits>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <aclapi.h>
#include <winternl.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace vision_simple {
#ifdef _WIN32
namespace {
// Relative native exclusive creation binds children to pinned directory handles.
HANDLE OpenChild(HANDLE parent, std::string_view name, ACCESS_MASK access,
    ULONG disposition, ULONG options, PSECURITY_DESCRIPTOR security = nullptr) noexcept {
  using CreateFile = NTSTATUS (NTAPI*)(PHANDLE, ACCESS_MASK, POBJECT_ATTRIBUTES,
      PIO_STATUS_BLOCK, PLARGE_INTEGER, ULONG, ULONG, ULONG, ULONG, PVOID, ULONG);
  const auto create = reinterpret_cast<CreateFile>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtCreateFile"));
  if (!create || name.size() > 64) return INVALID_HANDLE_VALUE;
  std::array<wchar_t, 64> buffer{};
  for (size_t i = 0; i < name.size(); ++i) buffer[i] = static_cast<unsigned char>(name[i]);
  UNICODE_STRING string{static_cast<USHORT>(name.size() * sizeof(wchar_t)),
      static_cast<USHORT>(name.size() * sizeof(wchar_t)), buffer.data()};
  OBJECT_ATTRIBUTES attributes{};
  InitializeObjectAttributes(&attributes, &string, OBJ_CASE_INSENSITIVE, parent, security);
  IO_STATUS_BLOCK status{};
  HANDLE file = INVALID_HANDLE_VALUE;
  // FILE_OPEN_REPARSE_POINT | FILE_SYNCHRONOUS_IO_NONALERT; deny delete sharing.
  const auto result = create(&file, access | SYNCHRONIZE, &attributes, &status,
      nullptr, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ | FILE_SHARE_WRITE,
      disposition, options | 0x00200000 | 0x00000020, nullptr, 0);
  return result >= 0 ? file : INVALID_HANDLE_VALUE;
}
}
#endif
struct DebugArtifacts::Impl {
  size_t max_bytes = 0, max_files = 0, bytes = 0, files = 0;
  bool retained = false;
  std::array<std::string, 64> names;
#ifdef _WIN32
  HANDLE parent = INVALID_HANDLE_VALUE, root = INVALID_HANDLE_VALUE;
  bool Cleanup() noexcept {
    bool ok = true;
    if (!retained && root != INVALID_HANDLE_VALUE) {
      for (size_t i = 0; i < files; ++i) {
        HANDLE file = OpenChild(root, names[i], DELETE, 1, 0x40);
        if (file == INVALID_HANDLE_VALUE) { ok = false; continue; }
        FILE_DISPOSITION_INFO disposition{TRUE};
        if (!SetFileInformationByHandle(file, FileDispositionInfo, &disposition, sizeof(disposition))) ok = false;
        if (!CloseHandle(file)) ok = false;
      }
      FILE_DISPOSITION_INFO disposition{TRUE};
      if (!SetFileInformationByHandle(root, FileDispositionInfo, &disposition, sizeof(disposition))) ok = false;
    }
    if (root != INVALID_HANDLE_VALUE && !CloseHandle(root)) ok = false;
    root = INVALID_HANDLE_VALUE;
    if (parent != INVALID_HANDLE_VALUE && !CloseHandle(parent)) ok = false;
    parent = INVALID_HANDLE_VALUE;
    return ok;
  }
#else
  int parent = -1, root = -1;
  std::string name;
  struct stat identity{};
  bool Cleanup() noexcept {
    bool ok = true;
    if (!retained && root >= 0) {
      for (size_t i = 0; i < files; ++i)
        if (unlinkat(root, names[i].c_str(), 0) != 0) ok = false;
      struct stat current{};
      if (fstatat(parent, name.c_str(), &current, AT_SYMLINK_NOFOLLOW) != 0 ||
          current.st_dev != identity.st_dev || current.st_ino != identity.st_ino) ok = false;
      else if (unlinkat(parent, name.c_str(), AT_REMOVEDIR) != 0) ok = false;
    }
    if (root >= 0 && close(root) != 0) ok = false;
    root = -1;
    if (parent >= 0 && close(parent) != 0) ok = false;
    parent = -1;
    return ok;
  }
#endif
  ~Impl() { if (!Cleanup()) std::fputs("Diagnostic artifact cleanup failed.\n", stderr); }
  std::expected<void, const char*> Write(std::string name, std::span<const unsigned char> data) {
    if (files >= max_files || data.size() > max_bytes - bytes) return std::unexpected("debug_quota");
    // Save the static name before opening: no allocation can strand a created file.
    names[files] = std::move(name);
#ifdef _WIN32
    HANDLE file = OpenChild(root, names[files], GENERIC_WRITE, 2, 0x40);
    if (file == INVALID_HANDLE_VALUE) return std::unexpected("debug_io");
    ++files;
    DWORD written = 0;
    const bool ok = WriteFile(file, data.data(), static_cast<DWORD>(data.size()), &written, nullptr) && written == data.size();
    const bool flushed = FlushFileBuffers(file) != 0;
    const bool closed = CloseHandle(file) != 0;
#else
    const int file = openat(root, names[files].c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (file < 0) return std::unexpected("debug_io");
    ++files;
    size_t offset = 0;
    bool ok = true;
    while (offset < data.size()) {
      const auto count = write(file, data.data() + offset, data.size() - offset);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) { ok = false; break; }
      offset += static_cast<size_t>(count);
    }
    const bool flushed = fsync(file) == 0;
    const bool closed = close(file) == 0;
#endif
    if (!ok || !flushed || !closed) return std::unexpected("debug_io");
    bytes += data.size();
    return {};
  }
};
DebugArtifacts::DebugArtifacts() : impl_(std::make_unique<Impl>()) {}
DebugArtifacts::~DebugArtifacts() = default;
bool DebugArtifacts::ValidName(std::string_view name) noexcept {
  auto alnum = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'); };
  if (name.empty() || name.size() > 64 || !alnum(name.front())) return false;
  std::array<char, 64> upper{};
  for (size_t i = 0; i < name.size(); ++i) {
    if (!alnum(name[i]) && name[i] != '_' && name[i] != '-') return false;
    upper[i] = name[i] >= 'a' && name[i] <= 'z' ? name[i] - ('a' - 'A') : name[i];
  }
  const std::string_view folded(upper.data(), name.size());
  if (folded == "CON" || folded == "PRN" || folded == "AUX" || folded == "NUL") return false;
  return !(folded.size() == 4 && (folded.starts_with("COM") || folded.starts_with("LPT")) && folded[3] >= '1' && folded[3] <= '9');
}
std::expected<void, const char*> DebugArtifacts::Init(std::string_view name, size_t max_bytes, size_t max_files) noexcept {
  try {
    if (!ValidName(name) || !max_bytes || max_bytes > 67108864 || !max_files || max_files > 64) return std::unexpected("debug_directory");
    impl_->max_bytes = max_bytes; impl_->max_files = max_files;
#ifdef _WIN32
    impl_->parent = CreateFileW(L".", FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (impl_->parent == INVALID_HANDLE_VALUE) return std::unexpected("debug_directory");
    BY_HANDLE_FILE_INFORMATION info{};
    if (!GetFileInformationByHandle(impl_->parent, &info) || (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) return std::unexpected("debug_directory");
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return std::unexpected("debug_directory");
    DWORD required = 0;
    GetTokenInformation(token, TokenUser, nullptr, 0, &required);
    if (!required || required > 65536) { CloseHandle(token); return std::unexpected("debug_directory"); }
    alignas(TOKEN_USER) std::array<unsigned char, 65536> storage{};
    const bool token_ok = GetTokenInformation(token, TokenUser, storage.data(), required, &required) != 0;
    CloseHandle(token);
    if (!token_ok) return std::unexpected("debug_directory");
    EXPLICIT_ACCESSW access{};
    access.grfAccessPermissions = FILE_ALL_ACCESS;
    access.grfAccessMode = SET_ACCESS;
    access.grfInheritance = SUB_CONTAINERS_AND_OBJECTS_INHERIT;
    access.Trustee.TrusteeForm = TRUSTEE_IS_SID;
    access.Trustee.TrusteeType = TRUSTEE_IS_USER;
    access.Trustee.ptstrName = static_cast<LPWSTR>(reinterpret_cast<TOKEN_USER*>(storage.data())->User.Sid);
    PACL acl = nullptr;
    if (SetEntriesInAclW(1, &access, nullptr, &acl) != ERROR_SUCCESS) return std::unexpected("debug_directory");
    SECURITY_DESCRIPTOR descriptor{};
    const bool security_ok = InitializeSecurityDescriptor(&descriptor, SECURITY_DESCRIPTOR_REVISION) &&
        SetSecurityDescriptorOwner(&descriptor, reinterpret_cast<TOKEN_USER*>(storage.data())->User.Sid, FALSE) &&
        SetSecurityDescriptorDacl(&descriptor, TRUE, acl, FALSE) &&
        SetSecurityDescriptorControl(&descriptor, SE_DACL_PROTECTED, SE_DACL_PROTECTED);
    if (security_ok) impl_->root = OpenChild(impl_->parent, name,
        FILE_READ_ATTRIBUTES | FILE_LIST_DIRECTORY | DELETE, 2, 1, &descriptor);
    LocalFree(acl);
    if (impl_->root == INVALID_HANDLE_VALUE) return std::unexpected("debug_directory");
    if (!GetFileInformationByHandle(impl_->root, &info) || (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) return std::unexpected("debug_directory");
#else
    impl_->name = name;
    impl_->parent = open(".", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    struct stat parent_info{};
    if (impl_->parent < 0 || fstat(impl_->parent, &parent_info) != 0 ||
        parent_info.st_uid != geteuid() || (parent_info.st_mode & (S_IWGRP | S_IWOTH))) return std::unexpected("debug_directory");
    if (mkdirat(impl_->parent, impl_->name.c_str(), 0700) != 0) return std::unexpected("debug_directory");
    impl_->root = openat(impl_->parent, impl_->name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (impl_->root < 0) { unlinkat(impl_->parent, impl_->name.c_str(), AT_REMOVEDIR); return std::unexpected("debug_directory"); }
    if (fstat(impl_->root, &impl_->identity) != 0) return std::unexpected("debug_directory");
#endif
    return {};
  } catch (...) { return std::unexpected("debug_directory"); }
}
std::expected<void, const char*> DebugArtifacts::Export(std::span<const std::string> images,
    const InferOCRResponse& response, OCRDetectionOptions options, size_t max_pixels) noexcept {
  try {
    if (response.results.size() != images.size()) return std::unexpected("debug_payload");
    nlohmann::json manifest = {{"schema_version", 1}, {"ocr_detection", {{"kernel_size", options.kernel_size},
        {"dilation_iterations", options.dilation_iterations}, {"min_box_area", options.min_box_area}}},
        {"recognition_confidence", 0.125}, {"frames", nlohmann::json::array()}};
    for (size_t i = 0; i < images.size(); ++i) {
      auto prepared = PrepareEncodedImage(images[i]);
      if (!prepared || prepared->pixels > max_pixels) return std::unexpected("debug_decode");
      // Match service decode: its prepared header already enforces pixel admission.
      // The optional codec limit is PNG/JPEG-only and changes orientation handling.
      auto image = DecodeImageBytes(prepared->bytes);
      if (!image || image->dims != 2 || image->type() != CV_8UC3 ||
          image->total() != prepared->pixels) return std::unexpected("debug_decode");
      std::vector<unsigned char> png;
      if (!cv::imencode(".png", *image, png)) return std::unexpected("debug_encode");
      char filename[32];
      std::snprintf(filename, sizeof(filename), "input-%03zu.png", i);
      auto written = impl_->Write(filename, png);
      if (!written) return written;
      nlohmann::json frame = {{"index", i}, {"width", image->cols}, {"height", image->rows},
          {"box_count", response.results[i].size()}, {"boxes", nlohmann::json::array()}};
      for (size_t j = 0; j < response.results[i].size(); ++j) {
        const auto& line = response.results[i][j];
        frame["boxes"].push_back({{"index", j}, {"bbox", {line.bbox[0], line.bbox[1], line.bbox[2], line.bbox[3]}}, {"confidence", line.confidence}});
        const cv::Rect box(line.bbox[0], line.bbox[1], line.bbox[2], line.bbox[3]);
        cv::rectangle(*image, box, cv::Scalar(0, 0, 255), 2);
        cv::putText(*image, std::to_string(j), box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
      }
      if (!cv::imencode(".png", *image, png)) return std::unexpected("debug_encode");
      std::snprintf(filename, sizeof(filename), "boxes-%03zu.png", i);
      written = impl_->Write(filename, png);
      if (!written) return written;
      manifest["frames"].push_back(std::move(frame));
    }
    const auto body = manifest.dump();
    return impl_->Write("manifest.json", {reinterpret_cast<const unsigned char*>(body.data()), body.size()});
  } catch (...) { return std::unexpected("debug_export"); }
}
std::expected<void, const char*> DebugArtifacts::Rollback() noexcept {
  if (!impl_->Cleanup()) return std::unexpected("debug_cleanup");
  return {};
}
void DebugArtifacts::Retain() noexcept { impl_->retained = true; }
size_t DebugArtifacts::Files() const noexcept { return impl_->files; }
size_t DebugArtifacts::Bytes() const noexcept { return impl_->bytes; }
}  // namespace vision_simple
