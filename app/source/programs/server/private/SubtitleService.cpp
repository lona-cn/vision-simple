#include "SubtitleService.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <opencv2/core.hpp>
#include <random>
#include <stop_token>
#include <thread>

#include "InferenceService.h"
#include "SubtitleTimeline.h"
#include "SubtitleVideoReader.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#endif

namespace vision_simple {
namespace {
using Clock = std::chrono::steady_clock;
constexpr uint64_t kUploadLimit = 64ULL * 1024 * 1024;
constexpr int64_t kDurationLimit = 1800000;
std::string Token() {
  std::random_device random;
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(32);
  for (int i = 0; i < 4; ++i) {
    const auto word = random();
    for (int shift = 0; shift < 32; shift += 4)
      result += hex[(word >> shift) & 15];
  }
  return result;
}
bool Terminal(SubtitleJobState state) {
  return state == SubtitleJobState::kCancelled ||
         state == SubtitleJobState::kCompleted ||
         state == SubtitleJobState::kFailed;
}
bool PrivateDirectory(const std::filesystem::path& path) {
#ifdef _WIN32
  // Supply a protected owner-only DACL at creation (never an inherited public
  // directory followed by a permission change). Resolve Advapi dynamically so
  // the server's existing link interface remains unchanged.
  HMODULE module =
      LoadLibraryExW(L"advapi32.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!module) return false;
  using Convert = BOOL(WINAPI*)(LPCWSTR, DWORD, PSECURITY_DESCRIPTOR*, PULONG);
  auto convert = reinterpret_cast<Convert>(GetProcAddress(
      module, "ConvertStringSecurityDescriptorToSecurityDescriptorW"));
  PSECURITY_DESCRIPTOR descriptor = nullptr;
  bool success = false;
  if (convert && convert(L"D:P(A;OICI;FA;;;OW)", 1, &descriptor, nullptr)) {
    SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES), descriptor,
                                   FALSE};
    success = CreateDirectoryW(path.c_str(), &attributes) != FALSE;
    LocalFree(descriptor);
  }
  FreeLibrary(module);
  return success;
#else
  return ::mkdir(path.c_str(), 0700) == 0;
#endif
}
bool ValidOptions(const SubtitleOptions& options) {
  if (options.model.empty() || options.model.size() > 256 ||
      options.sample_interval_ms < 100 || options.sample_interval_ms > 5000 ||
      !std::isfinite(options.min_confidence) || options.min_confidence < 0 ||
      options.min_confidence > 1 || options.stable_samples < 2 ||
      options.stable_samples > 10 || options.gap_samples < 2 ||
      options.gap_samples > 10)
    return false;
  for (double value : options.roi)
    if (!std::isfinite(value)) return false;
  return options.roi[0] >= 0 && options.roi[1] >= 0 && options.roi[2] > 0 &&
         options.roi[3] > 0 && options.roi[0] + options.roi[2] <= 1 &&
         options.roi[1] + options.roi[3] <= 1;
}
// Keep text bounded before timeline normalization. Invalid UTF-8 is rejected by
// the timeline rather than cutting through a multibyte code point.
bool JoinLines(const InferOCRResponse& response, double confidence,
               std::string& text) {
  if (response.results.size() != 1) return false;
  std::vector<const OCRLine*> lines;
  for (const auto& line : response.results.front()) {
    if (!std::isfinite(line.confidence) || line.confidence < confidence ||
        line.line.empty())
      continue;
    if (lines.size() == 4096 || line.line.size() > 4096) return false;
    lines.push_back(&line);
  }
  std::sort(lines.begin(), lines.end(), [](const OCRLine* a, const OCRLine* b) {
    if (a->bbox[1] != b->bbox[1]) return a->bbox[1] < b->bbox[1];
    return a->bbox[0] < b->bbox[0];
  });
  // Form rows using vertical overlap against the row's first line. Unlike an
  // approximate-y comparator, this preserves strict ordering for std::sort.
  for (size_t begin = 0; begin < lines.size();) {
    size_t end = begin + 1;
    const auto* anchor = lines[begin];
    const int64_t bottom =
        int64_t(anchor->bbox[1]) + std::max(1, anchor->bbox[3]);
    while (end < lines.size() && lines[end]->bbox[1] < bottom &&
           int64_t(lines[end]->bbox[1]) + std::max(1, lines[end]->bbox[3]) >
               anchor->bbox[1])
      ++end;
    std::sort(lines.begin() + begin, lines.begin() + end,
              [](const OCRLine* a, const OCRLine* b) {
                return a->bbox[0] < b->bbox[0];
              });
    for (size_t i = begin; i < end; ++i) {
      const size_t separator = text.empty() ? 0 : 1;
      if (text.size() + separator + lines[i]->line.size() > 4096) return false;
      if (separator) text += i == begin ? '\n' : ' ';
      text += lines[i]->line;
    }
    begin = end;
  }
  return true;
}
}  // namespace

struct SubtitleService::Impl {
  struct Job {
    SubtitleJobInfo info;
    SubtitleOptions options;
    Clock::time_point touched = Clock::now();
    std::filesystem::path path;
    std::ofstream upload;
    std::stop_source cancel;
    std::vector<SubtitleCue> cues;
  };
  std::shared_ptr<InferenceService> inference;
  std::filesystem::path root;
  std::mutex mutex, stop_mutex;
  std::condition_variable wake;
  std::map<std::string, std::shared_ptr<Job>> jobs;
  std::thread worker, sweeper;
  bool closed = false;

  ~Impl() {
    // Also protects construction failures before a SubtitleService exists.
    if (!root.empty()) {
      std::error_code error;
      std::filesystem::remove_all(root, error);
    }
  }
  void RemoveInput(Job& job) {
    if (job.upload.is_open()) job.upload.close();
    if (!job.path.empty()) {
      std::error_code error;
      std::filesystem::remove(job.path, error);
      if (!error) job.path.clear();
    }
  }
  void Fail(Job& job, const char* code) {
    RemoveInput(job);
    job.info.state = SubtitleJobState::kFailed;
    job.info.error_code = code;
    job.touched = Clock::now();
  }
  void Sweep() {
    const auto now = Clock::now();
    for (auto it = jobs.begin(); it != jobs.end();) {
      auto& job = *it->second;
      const bool idle = job.info.state == SubtitleJobState::kCreated ||
                        job.info.state == SubtitleJobState::kUploading;
      if ((idle && now - job.touched >= std::chrono::seconds(60)) ||
          (Terminal(job.info.state) &&
           now - job.touched >= std::chrono::seconds(300))) {
        RemoveInput(job);
        // A failed unlink must not free capacity for another 64 MiB file.
        if (job.path.empty()) {
          it = jobs.erase(it);
          continue;
        }
      }
      ++it;
    }
  }
  void Housekeep() noexcept {
    try {
      std::unique_lock lock(mutex);
      while (!closed) {
        wake.wait_for(lock, std::chrono::seconds(1), [this] { return closed; });
        if (!closed) Sweep();
      }
    } catch (...) {
    }
  }
  const char* Process(const std::shared_ptr<Job>& job,
                      std::vector<SubtitleCue>& cues) {
    SubtitleVideoReader video;
    if (const char* error = video.Open(job->path, job->cancel.get_token()))
      return job->cancel.stop_requested() ? nullptr : error;
    if (const auto duration = video.duration_hint()) {
      std::lock_guard lock(mutex);
      job->info.duration_ms = *duration;
    }
    SubtitleTimeline timeline(job->options.stable_samples,
                              job->options.gap_samples);
    int64_t previous = -1, next_sample = 0, end = 0;
    uint64_t decoded = 0, sampled = 0;
    SubtitleVideoReader::Frame decoded_frame;
    while (!job->cancel.stop_requested()) {
      const auto result = video.Read(decoded_frame);
      if (result == SubtitleVideoReader::Result::kCancelled) return nullptr;
      if (result == SubtitleVideoReader::Result::kError) return video.error();
      if (result == SubtitleVideoReader::Result::kEof) {
        if (!decoded) return "unreadable_video";
        break;
      }
      if (job->cancel.stop_requested()) return nullptr;
      const cv::Mat& frame = decoded_frame.image;
      if (++decoded > 1000000) return "frame_limit";
      const auto timestamp = decoded_frame.start_ms;
      if (timestamp < 0 || timestamp < previous ||
          decoded_frame.end_ms <= timestamp)
        return "invalid_timestamps";
      end = std::max(end, decoded_frame.end_ms);
      if (end > kDurationLimit) return "duration_limit";
      previous = timestamp;
      if (timestamp >= next_sample) {
        const auto& roi = job->options.roi;
        const int left = static_cast<int>(std::floor(roi[0] * frame.cols));
        const int top = static_cast<int>(std::floor(roi[1] * frame.rows));
        const int right = std::min(
            frame.cols,
            static_cast<int>(std::ceil((roi[0] + roi[2]) * frame.cols)));
        const int bottom = std::min(
            frame.rows,
            static_cast<int>(std::ceil((roi[1] + roi[3]) * frame.rows)));
        cv::Mat cropped =
            frame(cv::Rect(left, top, right - left, bottom - top));
        auto response =
            inference->RunFrames(InferenceKind::kOCR, job->options.model,
                                 std::span<const cv::Mat>(&cropped, 1),
                                 ServiceControl{job->cancel.get_token()});
        if (!response)
          return job->cancel.stop_requested() ? nullptr : "ocr_failed";
        const auto* result = std::get_if<InferOCRResponse>(&response->payload);
        std::string text;
        if (!result || !JoinLines(*result, job->options.min_confidence, text))
          return "ocr_text_limit";
        if (!timeline.Observe(timestamp, std::move(text)))
          return "subtitle_limit";
        response->Succeed();
        ++sampled;
        next_sample = timestamp + job->options.sample_interval_ms;
      }
      {
        std::lock_guard lock(mutex);
        job->info.decoded_frames = decoded;
        job->info.sampled_frames = sampled;
        job->info.position_ms = timestamp;
        job->info.cue_count = timeline.cues().size();
      }
    }
    if (job->cancel.stop_requested()) return nullptr;
    {
      std::lock_guard lock(mutex);
      job->info.duration_ms = end;
      job->info.position_ms = end;
    }
    if (!timeline.Finish(end)) return "subtitle_limit";
    cues = timeline.cues();
    return nullptr;
  }
  void Work() noexcept {
    for (;;) {
      std::shared_ptr<Job> job;
      try {
        {
          std::unique_lock lock(mutex);
          Sweep();
          if (closed) return;
          for (auto& [id, candidate] : jobs)
            if (candidate->info.state == SubtitleJobState::kQueued) {
              job = candidate;
              job->info.state = SubtitleJobState::kRunning;
              break;
            }
          if (!job) {
            wake.wait_for(lock, std::chrono::seconds(1));
            continue;
          }
        }
        std::vector<SubtitleCue> cues;
        const char* error = nullptr;
        try {
          error = Process(job, cues);
        } catch (...) {
          error = "processing_failed";
        }
        std::lock_guard lock(mutex);
        RemoveInput(*job);
        if (job->cancel.stop_requested()) {
          job->info.state = SubtitleJobState::kCancelled;
        } else if (error || !job->path.empty()) {
          Fail(*job, error ? error : "storage_failed");
        } else {
          job->cues = std::move(cues);
          job->info.cue_count = job->cues.size();
          job->info.state = SubtitleJobState::kCompleted;
        }
        job->touched = Clock::now();
      } catch (...) {
        // Keep the worker alive even if allocation for an error report fails.
        if (job) {
          std::lock_guard lock(mutex);
          RemoveInput(*job);
          job->info.state = SubtitleJobState::kFailed;
          job->touched = Clock::now();
        }
      }
    }
  }
};

SubtitleService::SubtitleService(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
SubtitleService::~SubtitleService() { Stop(); }
SubtitleResult<std::shared_ptr<SubtitleService>> SubtitleService::Create(
    std::shared_ptr<InferenceService> inference) noexcept {
  try {
    if (!inference) return std::unexpected(SubtitleFailure::kInvalid);
    auto impl = std::make_unique<Impl>();
    impl->inference = std::move(inference);
    const auto path = std::filesystem::temp_directory_path() /
                      ("vision-subtitles-" + Token());
    if (!PrivateDirectory(path))
      return std::unexpected(SubtitleFailure::kFailed);
    impl->root = path;
    auto result =
        std::shared_ptr<SubtitleService>(new SubtitleService(std::move(impl)));
    result->impl_->worker =
        std::thread([ptr = result->impl_.get()] { ptr->Work(); });
    result->impl_->sweeper =
        std::thread([ptr = result->impl_.get()] { ptr->Housekeep(); });
    return result;
  } catch (...) {
    return std::unexpected(SubtitleFailure::kFailed);
  }
}
SubtitleResult<SubtitleJobInfo> SubtitleService::Add(
    SubtitleOptions options) noexcept {
  try {
    if (!ValidOptions(options))
      return std::unexpected(SubtitleFailure::kInvalid);
    auto catalog = impl_->inference->ListModels();
    if (!catalog) return std::unexpected(SubtitleFailure::kFailed);
    if (std::none_of(catalog->models.begin(), catalog->models.end(),
                     [&](const auto& model) {
                       return model.task == "ocr" &&
                              model.name == options.model;
                     }))
      return std::unexpected(SubtitleFailure::kUnknownModel);
    auto job = std::make_shared<Impl::Job>();
    job->options = std::move(options);
    job->info.id = Token();
    std::lock_guard lock(impl_->mutex);
    if (impl_->closed) return std::unexpected(SubtitleFailure::kClosed);
    impl_->Sweep();
    if (impl_->jobs.size() >= 8)
      return std::unexpected(SubtitleFailure::kCapacity);
    auto info = job->info;
    if (!impl_->jobs.emplace(info.id, std::move(job)).second)
      return std::unexpected(SubtitleFailure::kFailed);
    return info;
  } catch (...) {
    return std::unexpected(SubtitleFailure::kFailed);
  }
}
SubtitleResult<SubtitleJobInfo> SubtitleService::Get(
    const std::string& id) noexcept {
  try {
    std::lock_guard lock(impl_->mutex);
    impl_->Sweep();
    const auto it = impl_->jobs.find(id);
    if (it == impl_->jobs.end())
      return std::unexpected(SubtitleFailure::kMissing);
    return it->second->info;
  } catch (...) {
    return std::unexpected(SubtitleFailure::kFailed);
  }
}
SubtitleResult<SubtitleJobPage> SubtitleService::List(
    size_t limit, const std::string& after) noexcept {
  try {
    if (!limit || limit > 100)
      return std::unexpected(SubtitleFailure::kInvalid);
    std::lock_guard lock(impl_->mutex);
    impl_->Sweep();
    SubtitleJobPage page;
    auto it = impl_->jobs.upper_bound(after);
    for (; it != impl_->jobs.end() && page.jobs.size() < limit; ++it)
      page.jobs.push_back(it->second->info);
    if (it != impl_->jobs.end()) page.next_cursor = page.jobs.back().id;
    return page;
  } catch (...) {
    return std::unexpected(SubtitleFailure::kFailed);
  }
}
SubtitleResult<void> SubtitleService::BeginUpload(
    const std::string& id) noexcept {
  try {
    std::lock_guard lock(impl_->mutex);
    if (impl_->closed) return std::unexpected(SubtitleFailure::kClosed);
    impl_->Sweep();
    auto it = impl_->jobs.find(id);
    if (it == impl_->jobs.end())
      return std::unexpected(SubtitleFailure::kMissing);
    auto& job = *it->second;
    if (job.info.state != SubtitleJobState::kCreated)
      return std::unexpected(SubtitleFailure::kBusy);
    job.path = impl_->root / (job.info.id + ".upload");
    job.upload.open(job.path, std::ios::binary | std::ios::trunc);
    if (!job.upload) {
      impl_->Fail(job, "storage_failed");
      return std::unexpected(SubtitleFailure::kFailed);
    }
    job.info.state = SubtitleJobState::kUploading;
    job.touched = Clock::now();
    return {};
  } catch (...) {
    AbortUpload(id);
    return std::unexpected(SubtitleFailure::kFailed);
  }
}
SubtitleResult<void> SubtitleService::AppendUpload(
    const std::string& id, std::span<const char> bytes) noexcept {
  try {
    std::lock_guard lock(impl_->mutex);
    if (impl_->closed) return std::unexpected(SubtitleFailure::kClosed);
    auto it = impl_->jobs.find(id);
    if (it == impl_->jobs.end())
      return std::unexpected(SubtitleFailure::kMissing);
    auto& job = *it->second;
    if (job.info.state != SubtitleJobState::kUploading)
      return std::unexpected(SubtitleFailure::kBusy);
    if (bytes.size() > kUploadLimit - job.info.uploaded_bytes) {
      impl_->Fail(job, "upload_limit");
      return std::unexpected(SubtitleFailure::kTooLarge);
    }
    if (!bytes.empty())
      job.upload.write(bytes.data(),
                       static_cast<std::streamsize>(bytes.size()));
    if (!job.upload) {
      impl_->Fail(job, "storage_failed");
      return std::unexpected(SubtitleFailure::kFailed);
    }
    job.info.uploaded_bytes += bytes.size();
    if (!bytes.empty()) job.touched = Clock::now();
    return {};
  } catch (...) {
    AbortUpload(id);
    return std::unexpected(SubtitleFailure::kFailed);
  }
}
SubtitleResult<void> SubtitleService::FinishUpload(
    const std::string& id) noexcept {
  try {
    std::lock_guard lock(impl_->mutex);
    if (impl_->closed) return std::unexpected(SubtitleFailure::kClosed);
    auto it = impl_->jobs.find(id);
    if (it == impl_->jobs.end())
      return std::unexpected(SubtitleFailure::kMissing);
    auto& job = *it->second;
    if (job.info.state != SubtitleJobState::kUploading)
      return std::unexpected(SubtitleFailure::kBusy);
    job.upload.close();
    if (!job.upload) {
      impl_->Fail(job, "storage_failed");
      return std::unexpected(SubtitleFailure::kFailed);
    }
    char header[16]{};
    std::ifstream input(job.path, std::ios::binary);
    input.read(header, sizeof(header));
    const auto size = input.gcount();
    input.close();
    const char* extension = nullptr;
    if (size >= 12 && !std::memcmp(header, "RIFF", 4) &&
        !std::memcmp(header + 8, "AVI ", 4)) {
      const auto* h = reinterpret_cast<const unsigned char*>(header);
      const uint64_t riff_size = uint64_t(h[4]) | uint64_t(h[5]) << 8 |
                                 uint64_t(h[6]) << 16 | uint64_t(h[7]) << 24;
      if (riff_size >= 4 && riff_size + 8 <= job.info.uploaded_bytes)
        extension = ".avi";
    } else if (size >= 12 && !std::memcmp(header + 4, "ftyp", 4)) {
      const auto* h = reinterpret_cast<const unsigned char*>(header);
      const uint64_t box_size = uint64_t(h[0]) << 24 | uint64_t(h[1]) << 16 |
                                uint64_t(h[2]) << 8 | h[3];
      if (box_size >= 12 && box_size <= job.info.uploaded_bytes)
        extension = ".mp4";
    } else if (size >= 4 && !std::memcmp(header, "\x1a\x45\xdf\xa3", 4))
      extension = ".mkv";
    if (!extension) {
      impl_->Fail(job, "invalid_container");
      return std::unexpected(SubtitleFailure::kInvalidVideo);
    }
    const auto target = impl_->root / (job.info.id + extension);
    std::error_code error;
    std::filesystem::rename(job.path, target, error);
    if (error) {
      impl_->Fail(job, "storage_failed");
      return std::unexpected(SubtitleFailure::kFailed);
    }
    job.path = target;
    job.info.state = SubtitleJobState::kQueued;
    job.touched = Clock::now();
    impl_->wake.notify_one();
    return {};
  } catch (...) {
    AbortUpload(id);
    return std::unexpected(SubtitleFailure::kFailed);
  }
}
void SubtitleService::AbortUpload(const std::string& id) noexcept {
  try {
    std::lock_guard lock(impl_->mutex);
    const auto it = impl_->jobs.find(id);
    if (it != impl_->jobs.end() &&
        it->second->info.state == SubtitleJobState::kUploading)
      impl_->Fail(*it->second, "upload_interrupted");
  } catch (...) {
  }
}
SubtitleResult<void> SubtitleService::Cancel(const std::string& id) noexcept {
  try {
    std::lock_guard lock(impl_->mutex);
    const auto it = impl_->jobs.find(id);
    if (it == impl_->jobs.end())
      return std::unexpected(SubtitleFailure::kMissing);
    auto& job = *it->second;
    if (Terminal(job.info.state)) return {};
    job.cancel.request_stop();
    if (job.info.state == SubtitleJobState::kRunning ||
        job.info.state == SubtitleJobState::kCancelling)
      job.info.state = SubtitleJobState::kCancelling;
    else {
      impl_->RemoveInput(job);
      job.info.state = SubtitleJobState::kCancelled;
    }
    job.touched = Clock::now();
    impl_->wake.notify_one();
    return {};
  } catch (...) {
    return std::unexpected(SubtitleFailure::kFailed);
  }
}
SubtitleResult<void> SubtitleService::Delete(const std::string& id) noexcept {
  try {
    std::lock_guard lock(impl_->mutex);
    const auto it = impl_->jobs.find(id);
    if (it == impl_->jobs.end())
      return std::unexpected(SubtitleFailure::kMissing);
    auto& job = *it->second;
    if (!Terminal(job.info.state) &&
        job.info.state != SubtitleJobState::kCreated)
      return std::unexpected(SubtitleFailure::kBusy);
    impl_->RemoveInput(job);
    if (!job.path.empty()) return std::unexpected(SubtitleFailure::kFailed);
    impl_->jobs.erase(it);
    return {};
  } catch (...) {
    return std::unexpected(SubtitleFailure::kFailed);
  }
}
SubtitleResult<std::string> SubtitleService::Download(const std::string& id,
                                                      bool webvtt) noexcept {
  try {
    std::shared_ptr<Impl::Job> job;
    {
      std::lock_guard lock(impl_->mutex);
      impl_->Sweep();
      const auto it = impl_->jobs.find(id);
      if (it == impl_->jobs.end())
        return std::unexpected(SubtitleFailure::kMissing);
      job = it->second;
      if (job->info.state != SubtitleJobState::kCompleted)
        return std::unexpected(SubtitleFailure::kNotReady);
    }
    return EncodeSubtitles(job->cues, webvtt);
  } catch (...) {
    return std::unexpected(SubtitleFailure::kFailed);
  }
}
void SubtitleService::Stop() noexcept {
  if (!impl_) return;
  try {
    std::lock_guard stopping(impl_->stop_mutex);
    {
      std::lock_guard lock(impl_->mutex);
      impl_->closed = true;
      for (auto& [id, job] : impl_->jobs) {
        if (Terminal(job->info.state)) continue;
        job->cancel.request_stop();
        if (job->info.state == SubtitleJobState::kRunning ||
            job->info.state == SubtitleJobState::kCancelling)
          job->info.state = SubtitleJobState::kCancelling;
        else {
          impl_->RemoveInput(*job);
          job->info.state = SubtitleJobState::kCancelled;
        }
        job->touched = Clock::now();
      }
    }
    impl_->wake.notify_all();
    if (impl_->worker.joinable()) impl_->worker.join();
    if (impl_->sweeper.joinable()) impl_->sweeper.join();
    std::lock_guard lock(impl_->mutex);
    for (auto& [id, job] : impl_->jobs) impl_->RemoveInput(*job);
    std::error_code error;
    std::filesystem::remove_all(impl_->root, error);
  } catch (...) {
  }
}
}  // namespace vision_simple
