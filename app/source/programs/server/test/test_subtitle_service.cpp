#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string_view>
#include <thread>

#include "InferenceService.h"
#include "SubtitleService.h"

using namespace vision_simple;
using namespace std::chrono_literals;
namespace fs = std::filesystem;
namespace {
void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "subtitle service regression: " << message << '\n';
    std::abort();
  }
}
template <class T> void Ok(const SubtitleResult<T>& result, const char* message) {
  Check(result.has_value(), message);
}
template <class T> void Error(const SubtitleResult<T>& result,
                             SubtitleFailure failure, const char* message) {
  Check(!result && result.error() == failure, message);
}
struct FixtureDirectory {
  fs::path original = fs::current_path(), path;
  explicit FixtureDirectory(const fs::path& root) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    for (size_t suffix = 0;; ++suffix) {
      path = fs::temp_directory_path() /
             ("vision-simple-subtitle-test-" + std::to_string(stamp) + "-" +
              std::to_string(suffix));
      if (fs::create_directory(path)) break;
    }
    fs::create_directory(path / "config");
    std::ofstream catalog(path / "config/models.yaml");
    catalog << "models:\n  - task: ocr\n    name: subtitle-test\n"
               "    version: kPPOCRv4\n    files:\n";
    for (const auto& [key, file] : {
             std::pair{"det", "ppocr_det.onnx"},
             std::pair{"rec", "ppocr_rec.onnx"},
             std::pair{"dictionary", "ppocr_keys_v1.txt"}}) {
      const auto asset = root / "app/assets/test" / file;
      Check(fs::is_regular_file(asset), "repository PP-OCR asset exists");
      catalog << "      " << key << ": '" << asset.generic_string() << "'\n";
    }
    catalog.close();
    Check(bool(catalog), "write isolated real OCR catalog before singleton use");
    fs::current_path(path);
  }
  ~FixtureDirectory() {
    fs::current_path(original);
    fs::remove_all(path);
  }
};
struct Time {
  std::atomic<int64_t> steady{0}, utc{1790812800000LL};
  static SubtitleServiceClock::SteadyTime Steady(void* context) noexcept {
    return SubtitleServiceClock::SteadyTime{std::chrono::milliseconds{
        static_cast<Time*>(context)->steady.load()}};
  }
  static SubtitleServiceClock::UtcTime Utc(void* context) noexcept {
    return SubtitleServiceClock::UtcTime{std::chrono::milliseconds{
        static_cast<Time*>(context)->utc.load()}};
  }
  SubtitleServiceClock Clock() { return {Steady, Utc, this}; }
  void Advance(int64_t milliseconds) {
    utc.fetch_add(milliseconds);
    steady.fetch_add(milliseconds);
  }
};
struct Harness {
  Time time;  // Must outlive the service's two real threads.
  std::shared_ptr<SubtitleService> service;
  explicit Harness(const std::shared_ptr<InferenceService>& inference) {
    auto result = SubtitleService::Create(inference, time.Clock());
    Ok(result, "create real subtitle worker and independent sweeper");
    service = *result;
  }
  SubtitleJobInfo Add() {
    SubtitleOptions options;
    options.model = "subtitle-test";
    options.roi = {0, 0, 1, 1};
    options.sample_interval_ms = 100;
    options.min_confidence = 0.5;
    auto result = service->Add(std::move(options));
    Ok(result, "admit subtitle job");
    return *result;
  }
  SubtitleJobInfo Get(const std::string& id) {
    auto result = service->Get(id);
    Ok(result, "read live job");
    return *result;
  }
};
bool Terminal(SubtitleJobState state) {
  return state == SubtitleJobState::kCompleted ||
         state == SubtitleJobState::kCancelled || state == SubtitleJobState::kFailed;
}
template <class Predicate>
SubtitleJobInfo Wait(Harness& h, const std::string& id, Predicate predicate) {
  const auto end = std::chrono::steady_clock::now() + 120s;
  for (;;) {
    auto info = h.Get(id);
    if (predicate(info)) return info;
    Check(std::chrono::steady_clock::now() < end, "worker reaches expected state");
    std::this_thread::sleep_for(2ms);
  }
}
SubtitleJobInfo Done(Harness& h, const std::string& id) {
  return Wait(h, id, [](const auto& info) { return Terminal(info.state); });
}
void Put32(std::string& bytes, size_t offset, uint32_t value) {
  for (size_t i = 0; i < 4; ++i) bytes[offset + i] = char(value >> (8 * i));
}
std::string Chunk(std::string_view name, const std::string& payload) {
  std::string result(name);
  result.resize(8);
  Put32(result, 4, static_cast<uint32_t>(payload.size()));
  result += payload;
  if (payload.size() & 1) result += '\0';
  return result;
}
std::string Video(unsigned count) {
  // Structurally complete CFR MJPEG AVI, using the same RIFF chunk layout as
  // test_subtitle_regression.py; no codec/backend or scheduler substitution.
  cv::Mat frame(160, 640, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::putText(frame, "HELLO", {165, 110}, cv::FONT_HERSHEY_SIMPLEX, 2.5,
              cv::Scalar(0, 0, 0), 5, cv::LINE_AA);
  std::vector<uint8_t> jpeg;
  Check(cv::imencode(".jpg", frame, jpeg), "encode real JPEG frames");
  std::string main(56, '\0'), stream(56, '\0'), bitmap(40, '\0');
  Put32(main, 0, 100000); Put32(main, 16, count); Put32(main, 24, 1);
  Put32(main, 32, 640); Put32(main, 36, 160);
  stream.replace(0, 8, "vidsMJPG");
  Put32(stream, 20, 1); Put32(stream, 24, 10); Put32(stream, 32, count);
  Put32(bitmap, 0, 40); Put32(bitmap, 4, 640); Put32(bitmap, 8, 160);
  bitmap[12] = 1; bitmap[14] = 24; bitmap.replace(16, 4, "MJPG");
  const auto headers = Chunk("LIST", "hdrl" + Chunk("avih", main) +
      Chunk("LIST", "strl" + Chunk("strh", stream) + Chunk("strf", bitmap)));
  std::string frames = "movi";
  const auto encoded = Chunk("00dc", std::string(jpeg.begin(), jpeg.end()));
  frames.reserve(4 + count * encoded.size());
  for (unsigned i = 0; i < count; ++i) frames += encoded;
  return Chunk("RIFF", "AVI " + headers + Chunk("LIST", frames));
}
void Upload(Harness& h, const std::string& id, const std::string& video) {
  Ok(h.service->BeginUpload(id), "begin real disk upload");
  Ok(h.service->AppendUpload(id, video), "append actual video bytes");
  Ok(h.service->FinishUpload(id), "successfully queue actual video");
}
fs::path Input(const std::string& id) {
  for (const auto& entry : fs::directory_iterator(fs::temp_directory_path())) {
    if (!entry.is_directory() ||
        !entry.path().filename().string().starts_with("vision-subtitles-")) continue;
    std::error_code error;
    for (fs::directory_iterator it(entry.path(), error), end; !error && it != end;
         it.increment(error))
      if (it->path().stem() == id) return it->path();
  }
  return {};
}
void IdleAndWallClock(const std::shared_ptr<InferenceService>& inference) {
  Harness h(inference);
  const auto created = h.Add();
  Check(created.expires_at == Time::Utc(&h.time) + 60s, "created expiry uses captured UTC");
  h.time.Advance(59999);
  Check(h.Get(created.id).expires_at == created.expires_at, "59.999s read does not renew created expiry");
  auto list = h.service->List(1, "");
  Ok(list, "list before exact idle boundary");
  Check(list->jobs.front().expires_at == created.expires_at, "list preserves expiry metadata");
  h.time.utc.fetch_add(86400000);
  Check(h.Get(created.id).expires_at == created.expires_at, "forward wall jump neither expires nor rebases metadata");
  h.time.utc.fetch_sub(172800000);
  h.time.Advance(1);
  Error(h.service->Get(created.id), SubtitleFailure::kMissing, "60.000s expires despite backward wall jump");

  const auto upload = h.Add();
  h.time.Advance(50000);
  Ok(h.service->BeginUpload(upload.id), "begin upload renews created lifetime");
  const auto begun = h.Get(upload.id);
  Check(begun.expires_at == Time::Utc(&h.time) + 60s, "begin captures current wall time");
  const auto path = Input(upload.id);
  Check(!path.empty() && fs::is_regular_file(path), "upload owns a real private input file");
  h.time.Advance(59999);
  Ok(h.service->AppendUpload(upload.id, std::span<const char>{}), "empty append accepted");
  Check(h.Get(upload.id).expires_at == begun.expires_at, "empty append does not renew");
  h.time.Advance(1);
  Error(h.service->Download(upload.id, false), SubtitleFailure::kMissing, "download sweep expires exact upload boundary");
  Check(!fs::exists(path), "expiry removes physical upload before row disappears");

  const auto renewed = h.Add();
  Ok(h.service->BeginUpload(renewed.id), "begin second upload");
  h.time.Advance(59999);
  const std::string bytes = "real nonempty disk write";
  Ok(h.service->AppendUpload(renewed.id, bytes), "nonempty append renews idle deadline");
  const auto touched = h.Get(renewed.id);
  Check(touched.expires_at == Time::Utc(&h.time) + 60s, "nonempty append captures new deadline");
  h.time.Advance(59999);
  Check(h.Get(renewed.id).uploaded_bytes == bytes.size(), "renewed upload survives old creation deadline");
  h.time.Advance(1);
  Error(h.service->BeginUpload(renewed.id), SubtitleFailure::kMissing, "begin sweep expires at renewed boundary");
}
void TerminalAndCapacity(const std::shared_ptr<InferenceService>& inference,
                         const std::string& video) {
  Harness h(inference);
  const auto cancelled = h.Add();
  Ok(h.service->Cancel(cancelled.id), "cancel created job");
  const auto failed = h.Add();
  Ok(h.service->BeginUpload(failed.id), "begin invalid video");
  Ok(h.service->AppendUpload(failed.id, std::string_view("invalid video")), "write invalid video");
  Error(h.service->FinishUpload(failed.id), SubtitleFailure::kInvalidVideo, "invalid container fails synchronously");
  const auto completed = h.Add();
  Upload(h, completed.id, video);
  const auto result = Done(h, completed.id);
  Check(result.state == SubtitleJobState::kCompleted && result.sampled_frames == 3,
        "real PP-OCR video completes with all three frames sampled");
  auto transcript = h.service->Download(completed.id, false);
  Ok(transcript, "completed job downloads SRT");
  Check(transcript->find("HELLO") != std::string::npos,
        "real OCR transcript contains the rendered HELLO cue");
  auto vtt = h.service->Download(completed.id, true);
  Ok(vtt, "completed job downloads VTT");
  Check(vtt->starts_with("WEBVTT\n") && vtt->find("HELLO") != std::string::npos,
        "VTT preserves actual recognized cue");
  const auto expiry = Time::Utc(&h.time) + 300s;
  for (const auto& id : {cancelled.id, failed.id, completed.id})
    Check(h.Get(id).expires_at == expiry, "all terminal states retain 300s from actual acknowledgement");
  for (int i = 0; i < 5; ++i) h.Add();
  SubtitleOptions options; options.model = "subtitle-test";
  Error(h.service->Add(options), SubtitleFailure::kCapacity, "eight rows include retained terminal results");
  h.time.Advance(60000);
  const auto admitted = h.Add();
  Check(h.Get(cancelled.id).state == SubtitleJobState::kCancelled &&
        h.Get(failed.id).state == SubtitleJobState::kFailed &&
        h.Get(completed.id).state == SubtitleJobState::kCompleted,
        "selective idle sweep keeps terminal rows while freeing admission");
  Ok(h.service->Delete(admitted.id), "remove replacement created job");
  h.time.Advance(239999);
  for (const auto& id : {cancelled.id, failed.id}) {
    Error(h.service->Download(id, false), SubtitleFailure::kNotReady, "terminal nonresult download denied");
    Ok(h.service->Cancel(id), "repeated terminal cancel accepted without renewal");
    Check(h.Get(id).expires_at == expiry, "terminal read and cancel preserve deadline");
  }
  Check(h.Get(completed.id).expires_at == expiry, "completed survives 299.999s");
  Ok(h.service->Download(completed.id, true), "download at 299.999s does not renew");
  h.time.Advance(1);
  for (const auto& id : {cancelled.id, failed.id, completed.id})
    Error(h.service->Download(id, false), SubtitleFailure::kMissing, "terminal expires at 300.000s including downloaded result");
  for (int i = 0; i < 8; ++i) h.Add();
  Error(h.service->Add(options), SubtitleFailure::kCapacity, "all eight slots reusable after terminal cleanup");
}
void FifoAndActiveLifetime(const std::shared_ptr<InferenceService>& inference,
                          const std::string& long_video) {
  Harness h(inference);
  const auto blocker = h.Add();
  Upload(h, blocker.id, long_video);
  Wait(h, blocker.id, [](const auto& info) { return info.state == SubtitleJobState::kRunning; });
  auto a = h.Add(), b = h.Add(), c = h.Add();
  // Reverse lexical order deliberately, and move the first-created row last.
  std::array<std::string, 3> order{a.id, b.id, c.id};
  std::sort(order.begin(), order.end(), std::greater<>());
  if (order == std::array<std::string, 3>{a.id, b.id, c.id})
    std::swap(order[0], order[1]);
  for (const auto& id : order) Upload(h, id, long_video);
  for (const auto& id : order) {
    const auto info = h.Get(id);
    Check(info.state == SubtitleJobState::kQueued && !info.expires_at,
          "queued rows have no expiry while real worker is occupied");
  }
  auto page = h.service->List(2, "");
  Ok(page, "page lexical metadata independently of FIFO");
  Check(page->jobs[0].id < page->jobs[1].id && page->next_cursor == page->jobs[1].id,
        "metadata pagination remains lexical");
  h.time.Advance(600000);
  Check(h.Get(blocker.id).state == SubtitleJobState::kRunning && !h.Get(blocker.id).expires_at,
        "running has no idle or terminal expiry even after ten virtual minutes");
  for (const auto& id : order)
    Check(h.Get(id).state == SubtitleJobState::kQueued, "queued survives ten virtual minutes");
  const auto idle = h.Add();
  Ok(h.service->BeginUpload(idle.id), "begin idle upload beside occupied worker");
  const auto idle_path = Input(idle.id);
  Check(!idle_path.empty(), "independent sweep fixture owns input");
  h.time.Advance(60000);
  const auto sweep_deadline = std::chrono::steady_clock::now() + 5s;
  while (fs::exists(idle_path) && std::chrono::steady_clock::now() < sweep_deadline)
    std::this_thread::sleep_for(10ms);
  Check(!fs::exists(idle_path), "independent real sweeper unlinks expired upload while OCR worker remains occupied");
  Error(h.service->Get(idle.id), SubtitleFailure::kMissing, "independent sweeper removes row with input");
  Check(h.Get(blocker.id).state == SubtitleJobState::kRunning, "sweep does not wait for active OCR completion");
  Ok(h.service->Cancel(blocker.id), "cancel real in-flight OCR blocker");
  const auto cancelling = h.Get(blocker.id);
  if (cancelling.state == SubtitleJobState::kCancelling)
    Check(!cancelling.expires_at, "cancelling has no expiry until worker acknowledgement");
  Check(Done(h, blocker.id).state == SubtitleJobState::kCancelled, "worker acknowledges cancellation");
  for (size_t i = 0; i < order.size(); ++i) {
    auto next = Wait(h, order[i], [](const auto& info) { return info.state != SubtitleJobState::kQueued; });
    Check(next.state == SubtitleJobState::kRunning, "worker starts in successful FinishUpload order, not random ID order");
    for (size_t j = i + 1; j < order.size(); ++j)
      Check(h.Get(order[j]).state == SubtitleJobState::kQueued, "later uploads stay queued until earlier job returns");
    Ok(h.service->Cancel(order[i]), "cancel running FIFO job");
    Check(Done(h, order[i]).state == SubtitleJobState::kCancelled, "cancelled FIFO job acknowledges before next starts");
  }
  const auto acknowledged = h.Get(blocker.id).expires_at;
  Check(acknowledged == Time::Utc(&h.time) + 300s,
        "terminal deadline begins at actual acknowledgement, not queued time");
}
void ExpiryRaces(const std::shared_ptr<InferenceService>& inference,
                 const std::string& video) {
  for (bool finish : {false, true}) {
    for (int iteration = 0; iteration < 8; ++iteration) {
      Harness h(inference);
      const auto job = h.Add();
      Ok(h.service->BeginUpload(job.id), "race opens upload");
      Ok(h.service->AppendUpload(job.id, video), "race writes valid video");
      const auto path = Input(job.id);
      Check(!path.empty(), "race input exists before arbitration");
      h.time.Advance(60000);
      std::barrier gate(3);
      SubtitleResult<void> mutation;
      SubtitleResult<SubtitleJobInfo> observed;
      std::thread mutator([&] {
        gate.arrive_and_wait();
        mutation = finish ? h.service->FinishUpload(job.id) : h.service->Cancel(job.id);
      });
      std::thread reader([&] { gate.arrive_and_wait(); observed = h.service->Get(job.id); });
      gate.arrive_and_wait();
      mutator.join(); reader.join();
      if (!mutation) {
        Error(mutation, SubtitleFailure::kMissing, "sweeper wins mutex race and mutation sees missing");
        Error(h.service->Get(job.id), SubtitleFailure::kMissing, "expired winner cannot resurrect a row");
      } else {
        const auto terminal = Done(h, job.id);
        Check(terminal.state == (finish ? SubtitleJobState::kCompleted : SubtitleJobState::kCancelled),
              "mutation winner reaches its legal terminal state rather than expiring in flight");
        Check(terminal.expires_at == Time::Utc(&h.time) + 300s,
              "race winner receives acknowledgement-based terminal retention");
        if (observed) Check(observed->state != SubtitleJobState::kUploading,
                            "reader after winning mutation never reports stale uploading state");
        Ok(h.service->Delete(job.id), "delete race winner");
      }
      Check(!fs::exists(path) && Input(job.id).empty(), "race leaves no upload or renamed orphan input");
      std::array<std::string, 8> replacements;
      for (auto& id : replacements) id = h.Add().id;
      for (const auto& id : replacements) Ok(h.service->Delete(id), "race reclaimed all capacity");
    }
  }
}
}  // namespace
int main(int argc, char** argv) {
  fs::path root = fs::current_path();
  if (argc == 3 && std::string_view(argv[1]) == "--project-root") root = fs::absolute(argv[2]);
  else if (argc != 1) {
    std::cerr << "usage: test_subtitle_service [--project-root directory]\n";
    return 2;
  } else {
    while (!fs::exists(root / "app/assets/test") && root.has_parent_path() && root != root.parent_path())
      root = root.parent_path();
  }
  FixtureDirectory fixtures(root);
  InferenceServiceOptions options;
  options.idle_timeout = 0ms;
  auto inference = InferenceService::Create(options);
  Check(inference.has_value(), "create actual shared CPU inference service");
  const auto short_video = Video(3), long_video = Video(1200);
  IdleAndWallClock(*inference);
  TerminalAndCapacity(*inference, short_video);
  FifoAndActiveLifetime(*inference, long_video);
  ExpiryRaces(*inference, short_video);
  std::cout << "subtitle service regression passed: idle/terminal boundaries, UTC jumps, real OCR, FIFO, races and cleanup\n";
}
