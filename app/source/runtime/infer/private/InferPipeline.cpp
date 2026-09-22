#include "InferPipeline.h"

#include <array>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "InferTask.h"

namespace vision_simple {
namespace {
PipelineFailure Failure(PipelineFailureKind kind, const char* message) {
  return {kind, std::nullopt,
          VisionSimpleError{kind == PipelineFailureKind::kInvalidRequest
                                ? VisionSimpleErrorCode::kParameterError
                                : VisionSimpleErrorCode::kRuntimeError,
                            message}};
}
}  // namespace

struct InferPipeline::Impl {
  struct Batch {
    PipelineControl control;
    std::vector<detail::FrameResult> results;
    std::optional<PipelineFailure> failure;
    std::optional<PipelineFailureKind> interrupted;
    size_t active = 0;
  };
  struct Slot {
    Batch* batch = nullptr;
    size_t index = 0;
    std::unique_ptr<detail::FrameTask> task;
  };
  // Every queue can hold every resident slot. Transitions never need another
  // credit or allocation, including OCR's postprocess -> preprocess loop.
  struct Queue {
    std::array<size_t, 64> items{};
    size_t head = 0;
    size_t count = 0;
    void Push(size_t slot) noexcept {
      items[(head + count++) % items.size()] = slot;
    }
    size_t Pop() noexcept {
      const auto slot = items[head];
      head = (head + 1) % items.size();
      --count;
      return slot;
    }
  };

  explicit Impl(PipelineOptions value) : options(value) {}
  ~Impl() {
    Close();
    for (auto& worker : workers)
      if (worker.joinable()) worker.join();
  }
  PipelineOptions options;
  std::mutex mutex;
  std::condition_variable_any changed;
  std::array<Slot, 64> slots;
  std::array<Queue, 3> queues;
  std::array<std::thread, 3> workers;
  size_t resident = 0;
  size_t batches = 0;
  bool closed = false;

  void Close() noexcept {
    {
      std::lock_guard lock(mutex);
      closed = true;
    }
    changed.notify_all();
  }
  void Observe(Batch& batch) noexcept {
    // Re-observe until finalization so a stop request outranks an earlier
    // timeout or close while executing native work is being drained.
    if (batch.control.stop.stop_requested())
      batch.interrupted = PipelineFailureKind::kCancelled;
    else if (batch.interrupted != PipelineFailureKind::kCancelled &&
             std::chrono::steady_clock::now() >= batch.control.deadline)
      batch.interrupted = PipelineFailureKind::kTimedOut;
    else if (!batch.interrupted && closed)
      batch.interrupted = PipelineFailureKind::kClosed;
  }
  bool Skip(Batch& batch, size_t index) noexcept {
    Observe(batch);
    return batch.interrupted ||
           (batch.failure && (!batch.failure->image_index ||
                              index > *batch.failure->image_index));
  }
  void Record(Batch& batch, size_t index, VisionSimpleError error) {
    if (!batch.failure ||
        (batch.failure->image_index && index < *batch.failure->image_index))
      batch.failure.emplace(PipelineFailure{PipelineFailureKind::kInference,
                                            index, std::move(error)});
  }
  void Release(size_t slot) noexcept {
    auto& entry = slots[slot];
    // Destroy before releasing the batch reference or capacity. A returning
    // Run must not leave a task destructor accessing its leased model.
    entry.task.reset();
    --entry.batch->active;
    --resident;
    entry.batch = nullptr;
    changed.notify_all();
  }
  void Work(size_t lane) noexcept {
    std::unique_lock lock(mutex);
    for (;;) {
      changed.wait(lock, [&] { return closed || queues[lane].count; });
      if (!queues[lane].count) {
        if (closed) return;
        continue;
      }
      const size_t slot = queues[lane].Pop();
      auto& entry = slots[slot];
      auto& batch = *entry.batch;
      if (Skip(batch, entry.index)) {
        Release(slot);
        continue;
      }
      lock.unlock();
      auto next = entry.task->Advance();
      lock.lock();
      if (!next) Record(batch, entry.index, std::move(next.error()));
      if (Skip(batch, entry.index) || !next) {
        Release(slot);
      } else if (*next) {
        queues[static_cast<size_t>(**next)].Push(slot);
        changed.notify_all();
      } else {
        batch.results[entry.index] = entry.task->TakeResult();
        Release(slot);
      }
    }
  }

  template <typename Result, typename Model>
  PipelineResult<Result> Run(Model& model, std::span<const cv::Mat> images,
                             float threshold,
                             PipelineControl control) noexcept {
    Batch batch;
    batch.control = control;
    std::unique_lock lock(mutex);
    Observe(batch);
    if (batch.interrupted)
      return std::unexpected(Failure(*batch.interrupted, "Interrupted"));
    if (images.size() > options.max_batch_images)
      return std::unexpected(
          Failure(PipelineFailureKind::kInvalidRequest, "Batch too large"));
    if (images.empty()) return std::vector<Result>{};
    if (batches == options.max_batches)
      return std::unexpected(
          Failure(PipelineFailureKind::kBusy, "Pipeline busy"));
    ++batches;
    // Stop-aware waits cannot lose a notification between check and sleep.
    try {
      batch.results.resize(images.size());
      for (size_t index = 0; index < images.size(); ++index) {
        while (resident == options.capacity && !Skip(batch, index)) {
          changed.wait_until(lock, control.stop, control.deadline, [&] {
            return resident < options.capacity || Skip(batch, index);
          });
        }
        if (Skip(batch, index)) break;
        size_t slot = 0;
        while (slots[slot].batch) ++slot;
        auto& entry = slots[slot];
        entry.batch = &batch;
        entry.index = index;
        ++resident;
        ++batch.active;
        lock.unlock();
        auto task = detail::MakeFrameTask(model, images[index], threshold);
        lock.lock();
        if (!task) {
          Record(batch, index, std::move(task.error()));
          Release(slot);
          break;
        }
        entry.task = std::move(*task);
        if (Skip(batch, index)) {
          Release(slot);
          break;
        }
        queues[0].Push(slot);
        changed.notify_all();
      }
    } catch (...) {
      if (!lock.owns_lock()) lock.lock();
      batch.failure.emplace(
          Failure(PipelineFailureKind::kInference, "Pipeline error"));
      changed.notify_all();
    }
    // All admitted stages (and task destructors) finish before releasing input
    // and model leases. An interrupted batch waits without an expired deadline.
    while (batch.active) {
      Observe(batch);
      if (batch.interrupted)
        changed.wait(lock);
      else
        changed.wait_until(lock, control.stop, control.deadline, [&] {
          Observe(batch);
          return !batch.active || batch.interrupted.has_value();
        });
    }
    Observe(batch);
    --batches;
    changed.notify_all();
    if (batch.interrupted)
      return std::unexpected(Failure(*batch.interrupted, "Interrupted"));
    if (batch.failure) return std::unexpected(std::move(*batch.failure));
    try {
      std::vector<Result> results;
      results.reserve(images.size());
      for (auto& result : batch.results)
        results.push_back(std::move(std::get<Result>(result)));
      Observe(batch);
      if (batch.interrupted)
        return std::unexpected(Failure(*batch.interrupted, "Interrupted"));
      return results;
    } catch (...) {
      return std::unexpected(
          Failure(PipelineFailureKind::kInference, "Pipeline error"));
    }
  }
};

InferPipeline::InferPipeline(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
InferPipeline::~InferPipeline() = default;
InferPipeline::CreateResult InferPipeline::Create(
    PipelineOptions options) noexcept {
  if (!options.capacity || options.capacity > 64 || !options.max_batches ||
      options.max_batches > 64 || !options.max_batch_images ||
      options.max_batch_images > 4096)
    return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                      "Invalid pipeline limits");
  try {
    auto impl = std::make_unique<Impl>(options);
    for (size_t lane = 0; lane < impl->workers.size(); ++lane)
      impl->workers[lane] =
          std::thread([state = impl.get(), lane] { state->Work(lane); });
    return std::unique_ptr<InferPipeline>(new InferPipeline(std::move(impl)));
  } catch (...) {
    return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError,
                      "Pipeline creation failed");
  }
}
void InferPipeline::Close() noexcept { impl_->Close(); }
PipelineResult<YOLOFrameResult> InferPipeline::Run(
    InferYOLO& model, std::span<const cv::Mat> images, float threshold,
    PipelineControl control) noexcept {
  return impl_->Run<YOLOFrameResult>(model, images, threshold, control);
}
PipelineResult<OCRFrameResult> InferPipeline::Run(
    InferOCR& model, std::span<const cv::Mat> images, float threshold,
    PipelineControl control) noexcept {
  return impl_->Run<OCRFrameResult>(model, images, threshold, control);
}
PipelineResult<YOLOTaskFrameResult> InferPipeline::Run(
    InferYOLOTask& model, std::span<const cv::Mat> images, float threshold,
    PipelineControl control) noexcept {
  return impl_->Run<YOLOTaskFrameResult>(model, images, threshold, control);
}
}  // namespace vision_simple
