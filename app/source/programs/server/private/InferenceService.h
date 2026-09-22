#pragma once

#include <chrono>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <variant>
#include <vector>

#include "Infer.h"
#include "InferPipeline.h"

namespace vision_simple {
enum class InferenceKind { kYOLO, kOCR, kSegmentation, kPose, kOBB };
enum class ServiceFailure {
  kInvalidRequest,
  kUnknownModel,
  kInvalidImage,
  kModelLoad,
  kModelConfig,
  kInference,
  kInternal,
  kModelBusy,
  kModelNotLoaded,
  kBusy,
  kCancelled,
  kTimedOut,
  kClosed
};
struct ServiceError {
  ServiceFailure kind;
  std::optional<size_t> image_index;
};
template <typename T>
using ServiceResult = std::expected<T, ServiceError>;

struct YOLODetectedObject {
  int32_t class_id;
  float confidence;
  int bbox[4];
};
struct InferYOLOResponse {
  std::vector<std::string> class_names;
  std::vector<std::vector<YOLODetectedObject>> results;
};
struct OCRLine {
  std::string line;
  float confidence;
  int bbox[4];
};
struct InferOCRResponse {
  std::vector<std::vector<OCRLine>> results;
};
struct SegmentedObject {
  int32_t class_id;
  float confidence;
  int bbox[4];
  std::string mask_png_base64;
};
struct InferSegmentationResponse {
  std::vector<std::string> class_names;
  std::vector<std::vector<SegmentedObject>> results;
};
struct PoseKeypoint {
  float x, y, confidence;
};
struct PoseObject {
  int32_t class_id;
  float confidence;
  int bbox[4];
  std::vector<PoseKeypoint> keypoints;
};
struct InferPoseResponse {
  std::vector<std::string> class_names;
  std::vector<std::vector<PoseObject>> results;
};
struct RotatedObject {
  int32_t class_id;
  float confidence;
  float corners[4][2];
  float angle;
};
struct InferOBBResponse {
  std::vector<std::string> class_names;
  std::vector<std::vector<RotatedObject>> results;
};

// Keeps model/context ownership and request accounting alive through transport
// serialization. An abandoned or failed response counts as a failed request.
class InferenceCompletion {
 public:
  virtual ~InferenceCompletion() = default;
  virtual void Succeed() noexcept = 0;
};
struct InferenceResponse {
  std::variant<InferYOLOResponse, InferOCRResponse, InferSegmentationResponse,
               InferPoseResponse, InferOBBResponse>
      payload;
  std::shared_ptr<InferenceCompletion> completion;
  void Succeed() noexcept {
    if (completion) completion->Succeed();
  }
};
struct CatalogModel {
  std::string task;
  std::string name;
};
struct ModelCatalog {
  std::vector<CatalogModel> models;
};
struct ModelStatistics {
  std::string kind;
  std::string name;
  uint64_t active_requests;
  uint64_t requests;
  uint64_t failures;
  double total_duration_ms;
  int64_t last_used;
};
struct ServiceStatistics {
  std::vector<ModelStatistics> models;
  int64_t idle_timeout_ms;
  size_t total;
  size_t limit;
  size_t offset;
};
struct InferenceServiceOptions {
  InferFramework framework = InferFramework::kONNXRUNTIME;
  InferEP ep = InferEP::kCPU;
  int device_id = 0;
  std::chrono::milliseconds idle_timeout{300000};
  std::chrono::milliseconds sweep_interval{1000};
  PipelineOptions pipeline;
  std::chrono::milliseconds request_timeout{60000};
  size_t ocr_rec_batch_size = 1;
};
struct ServiceControl {
  std::stop_token stop;
  std::optional<std::chrono::milliseconds> timeout;
  std::chrono::steady_clock::time_point started =
      std::chrono::steady_clock::now();
};

// Protocol-independent owner of loading, decoding, scheduling and lifecycle.
// All methods support concurrent callers. Transport owners must join callers
// before destroying the service; returned responses retain their model lease.
class InferenceService {
 public:
  static VSResult<std::shared_ptr<InferenceService>> Create(
      InferenceServiceOptions options) noexcept;
  ~InferenceService();
  InferenceService(const InferenceService&) = delete;
  InferenceService& operator=(const InferenceService&) = delete;
  ServiceResult<InferenceResponse> Run(InferenceKind kind,
                                       const std::string& model,
                                       std::span<const std::string> images,
                                       ServiceControl control = {}) noexcept;
  // Borrows BGR frames (including noncontiguous ROIs) until synchronous return.
  // Call Succeed() on the returned response after consuming its payload.
  ServiceResult<InferenceResponse> RunFrames(
      InferenceKind kind, const std::string& model,
      std::span<const cv::Mat> images, ServiceControl control = {}) noexcept;
  ServiceResult<ModelCatalog> ListModels() const noexcept;
  ServiceResult<void> Unload(InferenceKind kind,
                             const std::string& model) noexcept;
  ServiceResult<ServiceStatistics> Stats(size_t limit = 100,
                                         size_t offset = 0) const noexcept;
  const InferenceServiceOptions& options() const noexcept;

 private:
  struct Impl;
  explicit InferenceService(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};
}  // namespace vision_simple
