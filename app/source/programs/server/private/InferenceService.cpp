#include "InferenceService.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <format>
#include <magic_enum.hpp>
#include <map>
#include <mutex>
#include <string_view>
#include <thread>

#include "IOUtil.h"
#include "ImageCodec.h"
#include "LogFacade.h"
#include "TaskRegistry.h"
#include "VisionSimpleConfig.h"

namespace vision_simple {
namespace {
using Clock = std::chrono::steady_clock;
void LogFailure(std::string_view detail) noexcept {
  LogFacade::Error("inference", detail);
}
std::optional<std::vector<cv::Mat>> PrepareImages(
    std::span<const std::string> encoded, ServiceError& stage) {
  stage = {ServiceFailure::kInvalidImage, std::nullopt};
  std::vector<cv::Mat> images;
  images.reserve(encoded.size());
  for (size_t i = 0; i < encoded.size(); ++i) {
    stage.image_index = i;
    auto image = DecodeEncodedImage(encoded[i]);
    if (!image) return std::nullopt;
    images.emplace_back(std::move(*image));
  }
  return images;
}
std::optional<std::span<const cv::Mat>> PrepareImages(
    std::span<const cv::Mat> images, ServiceError& stage) {
  stage = {ServiceFailure::kInvalidImage, std::nullopt};
  for (size_t i = 0; i < images.size(); ++i) {
    stage.image_index = i;
    const auto& image = images[i];
    if (image.empty() || image.dims != 2 || image.rows <= 0 ||
        image.cols <= 0 || image.type() != CV_8UC3)
      return std::nullopt;
  }
  return images;
}
int64_t UnixMilliseconds() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

struct ModelKey {
  std::string task, name;
};
struct ModelKeyView {
  std::string_view task, name;
};
struct ModelKeyLess {
  using is_transparent = void;
  template <typename Left, typename Right>
  bool operator()(const Left& left, const Right& right) const noexcept {
    return std::pair{std::string_view(left.task), std::string_view(left.name)} <
           std::pair{std::string_view(right.task),
                     std::string_view(right.name)};
  }
};
}  // namespace
struct InferenceService::Impl : std::enable_shared_from_this<Impl> {
  InferenceServiceOptions options;
  std::unique_ptr<InferContext> infer_context_;
  struct ModelEntry {
    RegisteredModel model;
    uint64_t active_requests = 0;
    uint64_t requests = 0;
    uint64_t failures = 0;
    double total_duration_ms = 0;
    int64_t last_used = UnixMilliseconds();
    Clock::time_point idle_since = Clock::now();
  };
  std::mutex cache_mutex_;
  std::map<ModelKey, ModelEntry, ModelKeyLess> models_cache_;
  const ModelConfig* configuration_ = nullptr;
  std::unique_ptr<InferPipeline> pipeline_;
  int device_id_;
  std::mutex sweep_mutex_;
  std::condition_variable sweep_cv_;
  bool sweep_stopping_ = false;
  std::thread sweep_thread_;
  class ModelLease final : public InferenceCompletion {
    std::shared_ptr<Impl> owner_;
    ModelEntry* entry_;
    Clock::time_point started_ = Clock::now();
    std::atomic<bool> succeeded_{false};

   public:
    ModelLease(std::shared_ptr<Impl> owner, ModelEntry& entry)
        : owner_(std::move(owner)), entry_(&entry) {
      ++entry_->active_requests;
      ++entry_->requests;
    }
    ~ModelLease() override {
      std::lock_guard lock{owner_->cache_mutex_};
      --entry_->active_requests;
      if (!succeeded_.load(std::memory_order_relaxed)) ++entry_->failures;
      const auto now = Clock::now();
      entry_->total_duration_ms +=
          std::chrono::duration<double, std::milli>(now - started_).count();
      entry_->idle_since = now;
      entry_->last_used = UnixMilliseconds();
    }
    RegisteredModel& get() { return entry_->model; }
    void Succeed() noexcept override {
      succeeded_.store(true, std::memory_order_relaxed);
    }
  };
  VSResult<std::reference_wrapper<const ModelConfig>> GetConfiguration() {
    const std::lock_guard lock{cache_mutex_};
    if (configuration_) return *configuration_;
    auto config = Config::Instance();
    if (!config) return std::unexpected(std::move(config.error()));
    const auto& catalog = config->get().model_config();
    for (const auto& definition : catalog.models) {
      if (!FindTask(definition.task))
        return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                          "Configured inference task is not registered");
    }
    configuration_ = &catalog;
    return catalog;
  }

  VSResult<std::shared_ptr<ModelLease>> GetModel(const TaskDescriptor& task,
                                                 const std::string& name,
                                                 const ModelConfig& config) {
    const std::lock_guard lock{cache_mutex_};
    const auto cached = models_cache_.find(ModelKeyView{task.id, name});
    if (cached != models_cache_.end())
      return std::make_shared<ModelLease>(shared_from_this(), cached->second);
    const auto definition =
        std::ranges::find_if(config.models, [&](const auto& value) {
          return value.task == task.id && value.name == name;
        });
    if (definition == config.models.end()) return std::shared_ptr<ModelLease>{};
    auto model = task.load(*infer_context_, *definition, device_id_);
    if (!model) return std::unexpected(std::move(model.error()));
    auto [loaded, inserted] = models_cache_.try_emplace(
        ModelKey{std::string(task.id), name}, ModelEntry{std::move(*model)});
    return std::make_shared<ModelLease>(shared_from_this(), loaded->second);
  }

  template <typename Image>
  ServiceResult<InferenceResponse> Run(InferenceKind kind,
                                       const std::string& model,
                                       std::span<const Image> input,
                                       ServiceControl control) noexcept {
    ServiceError stage{ServiceFailure::kInvalidRequest, {}};
    try {
      const auto* task = FindTask(kind);
      const auto timeout = control.timeout.value_or(options.request_timeout);
      if (!task || model.empty() ||
          input.size() > options.pipeline.max_batch_images ||
          timeout.count() <= 0 || timeout.count() > 300000)
        return std::unexpected(stage);
      stage.kind = ServiceFailure::kModelConfig;
      auto config = GetConfiguration();
      if (!config) {
        LogFailure(config.error().message);
        return std::unexpected(stage);
      }
      const auto duration =
          std::chrono::duration_cast<Clock::duration>(timeout);
      const auto deadline =
          control.started > Clock::time_point::max() - duration
              ? Clock::time_point::max()
              : control.started + duration;
      const PipelineControl pipeline_control{.stop = control.stop,
                                             .deadline = deadline};
      stage.kind = ServiceFailure::kModelLoad;
      auto loaded = GetModel(*task, model, config->get());
      if (!loaded) {
        LogFailure(loaded.error().message);
        return std::unexpected(stage);
      }
      if (!*loaded)
        return std::unexpected(ServiceError{ServiceFailure::kUnknownModel, {}});
      auto lease = std::move(*loaded);
      auto images = PrepareImages(input, stage);
      if (!images) return std::unexpected(stage);
      stage = {ServiceFailure::kInference, {}};
      auto batch = RunRegisteredTask(lease->get(), *pipeline_, *images,
                                     pipeline_control);
      if (!batch) return std::unexpected(std::move(batch.error()));
      stage.kind = ServiceFailure::kInternal;
      return InferenceResponse{std::move(*batch), std::move(lease)};
    } catch (const std::exception& error) {
      LogFailure(error.what());
    } catch (...) {
      LogFailure("Unknown inference failure");
    }
    return std::unexpected(stage);
  }

  Impl(InferenceServiceOptions value, std::unique_ptr<InferContext> context,
       std::unique_ptr<InferPipeline> pipeline)
      : options(value),
        infer_context_(std::move(context)),
        pipeline_(std::move(pipeline)),
        device_id_(value.device_id) {}
  void StartSweep() {
    if (options.idle_timeout.count() != 0) {
      sweep_thread_ = std::thread([this] {
        std::unique_lock wait_lock{sweep_mutex_};
        while (!sweep_cv_.wait_for(wait_lock, options.sweep_interval,
                                   [this] { return sweep_stopping_; })) {
          wait_lock.unlock();
          {
            std::lock_guard lock{cache_mutex_};
            const auto now = Clock::now();
            const auto expired = [&](const auto& item) {
              return item.second.active_requests == 0 &&
                     now - item.second.idle_since >= options.idle_timeout;
            };
            std::erase_if(models_cache_, expired);
          }
          wait_lock.lock();
        }
      });
    }
  }
  ~Impl() {
    {
      std::lock_guard lock{sweep_mutex_};
      sweep_stopping_ = true;
    }
    sweep_cv_.notify_all();
    if (sweep_thread_.joinable()) sweep_thread_.join();
  }
};
InferenceService::InferenceService(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
InferenceService::~InferenceService() = default;
const InferenceServiceOptions& InferenceService::options() const noexcept {
  return impl_->options;
}
VSResult<std::shared_ptr<InferenceService>> InferenceService::Create(
    InferenceServiceOptions options) noexcept {
  const auto max_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          Clock::duration::max())
                          .count() /
                      2;
  if (!magic_enum::enum_contains(options.framework) ||
      !magic_enum::enum_contains(options.ep) || options.device_id < 0 ||
      options.idle_timeout.count() < 0 ||
      options.idle_timeout.count() > max_ms ||
      options.sweep_interval.count() <= 0 ||
      options.sweep_interval.count() > max_ms ||
      options.request_timeout.count() <= 0 ||
      options.request_timeout.count() > 300000 ||
      options.ocr_rec_batch_size == 0 || options.ocr_rec_batch_size > 64 ||
      options.pipeline.capacity == 0 || options.pipeline.capacity > 64 ||
      options.pipeline.max_batches == 0 || options.pipeline.max_batches > 64 ||
      options.pipeline.max_batch_images == 0 ||
      options.pipeline.max_batch_images > 4096)
    return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                      "Invalid inference service options");
  try {
    auto pipeline = InferPipeline::Create(options.pipeline);
    if (!pipeline) return std::unexpected(std::move(pipeline.error()));
    auto context = InferContext::Create(
        options.framework, options.ep,
        {{"ocr_rec_batch_size", std::to_string(options.ocr_rec_batch_size)}});
    if (!context) {
      LogFailure(context.error().message);
      return MK_VSERROR(VisionSimpleErrorCode::kModelError,
                        "Unable to create inference context");
    }
    auto impl = std::make_shared<Impl>(options, std::move(*context),
                                       std::move(*pipeline));
    impl->StartSweep();
    return std::shared_ptr<InferenceService>(
        new InferenceService(std::move(impl)));
  } catch (const std::exception& error) {
    LogFailure(error.what());
  } catch (...) {
    LogFailure("Unknown service creation failure");
  }
  return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError,
                    "Unable to create inference service");
}
ServiceResult<InferenceResponse> InferenceService::Run(
    InferenceKind kind, const std::string& model,
    std::span<const std::string> encoded, ServiceControl control) noexcept {
  return impl_->Run(kind, model, encoded, control);
}
ServiceResult<InferenceResponse> InferenceService::RunFrames(
    InferenceKind kind, const std::string& model,
    std::span<const cv::Mat> images, ServiceControl control) noexcept {
  return impl_->Run(kind, model, images, control);
}
ServiceResult<ModelCatalog> InferenceService::ListModels() const noexcept {
  ServiceError stage{ServiceFailure::kModelConfig, {}};
  try {
    auto config = impl_->GetConfiguration();
    if (!config) {
      LogFailure(config.error().message);
      return std::unexpected(stage);
    }
    stage.kind = ServiceFailure::kInternal;
    ModelCatalog result;
    result.models.reserve(config->get().models.size());
    for (const auto& info : config->get().models)
      result.models.push_back({info.task, info.name});
    return result;
  } catch (const std::exception& error) {
    LogFailure(error.what());
  } catch (...) {
    LogFailure("Unknown catalog failure");
  }
  return std::unexpected(stage);
}
ServiceResult<void> InferenceService::Unload(
    InferenceKind kind, const std::string& model) noexcept {
  const auto* task = FindTask(kind);
  if (!task || model.empty())
    return std::unexpected(ServiceError{ServiceFailure::kInvalidRequest, {}});
  try {
    std::lock_guard lock{impl_->cache_mutex_};
    const auto it = impl_->models_cache_.find(ModelKeyView{task->id, model});
    if (it == impl_->models_cache_.end())
      return std::unexpected(ServiceError{ServiceFailure::kModelNotLoaded, {}});
    if (it->second.active_requests)
      return std::unexpected(ServiceError{ServiceFailure::kModelBusy, {}});
    impl_->models_cache_.erase(it);
    return {};
  } catch (const std::exception& error) {
    LogFailure(error.what());
  } catch (...) {
    LogFailure("Unknown unload failure");
  }
  return std::unexpected(ServiceError{ServiceFailure::kInternal, {}});
}
ServiceResult<ServiceStatistics> InferenceService::Stats(
    size_t limit, size_t offset) const noexcept {
  if (!limit || limit > 200)
    return std::unexpected(ServiceError{ServiceFailure::kInvalidRequest, {}});
  try {
    std::lock_guard lock{impl_->cache_mutex_};
    ServiceStatistics result{{},
                             impl_->options.idle_timeout.count(),
                             impl_->models_cache_.size(),
                             limit,
                             offset};
    size_t index = 0;
    for (const auto& [key, entry] : impl_->models_cache_) {
      if (index++ < offset || result.models.size() >= limit) continue;
      result.models.push_back({key.task, key.name, entry.active_requests,
                               entry.requests, entry.failures,
                               entry.total_duration_ms, entry.last_used});
    }
    return result;
  } catch (const std::exception& error) {
    LogFailure(error.what());
  } catch (...) {
    LogFailure("Unknown statistics failure");
  }
  return std::unexpected(ServiceError{ServiceFailure::kInternal, {}});
}
}  // namespace vision_simple
