#pragma once
#include <ylt/struct_json/json_writer.h>

#include <algorithm>
#include <string_view>

#include "InferenceService.h"
#include "TaskRegistry.h"

namespace vision_simple {
struct ServiceErrorDescription {
  int status;
  std::string_view code;
  std::string_view message;
};
inline ServiceErrorDescription DescribeError(ServiceFailure kind) noexcept {
  switch (kind) {
    case ServiceFailure::kInvalidRequest:
      return {400, "invalid_request",
              "Request model, images or inference controls are invalid"};
    case ServiceFailure::kUnknownModel:
      return {400, "unknown_model", "Model is not configured"};
    case ServiceFailure::kInvalidImage:
      return {400, "invalid_image", "Image cannot be decoded"};
    case ServiceFailure::kModelLoad:
      return {500, "model_load_failed", "Model cannot be loaded"};
    case ServiceFailure::kModelConfig:
      return {500, "model_config_failed", "Model configuration cannot be read"};
    case ServiceFailure::kInference:
      return {500, "inference_failed", "Image inference failed"};
    case ServiceFailure::kModelBusy:
      return {409, "model_busy", "Model has active requests"};
    case ServiceFailure::kModelNotLoaded:
      return {404, "model_not_loaded", "Model is not loaded"};
    case ServiceFailure::kBusy:
      return {503, "service_overloaded", "Inference pipeline is at capacity"};
    case ServiceFailure::kCancelled:
      return {503, "request_cancelled", "Inference request was cancelled"};
    case ServiceFailure::kTimedOut:
      return {504, "request_timeout",
              "Inference request exceeded its deadline"};
    case ServiceFailure::kClosed:
      return {503, "service_unavailable", "Inference pipeline is closed"};
    case ServiceFailure::kInternal:
      return {500, "internal_error", "Response cannot be serialized"};
  }
  return {500, "internal_error", "Request failed"};
}
inline std::string SerializeInference(const InferenceResponse& response) {
  std::string body;
  std::visit([&](const auto& value) { struct_json::to_json(value, body); },
             response.payload);
  return body;
}

struct ProtocolModel {
  std::string id;
  std::string kind;
  std::string name;
};
struct ModelPage {
  std::vector<ProtocolModel> data;
  std::string next_cursor;
};

// Shared opaque, versioned catalog cursor for HTTP and MCP. The cursor names a
// sort position, not an offset, so removing a preceding model does not skip
// rows.
inline ServiceResult<ModelPage> PaginateModels(const ModelCatalog& catalog,
                                               size_t limit,
                                               std::string_view cursor = {}) {
  if (limit == 0 || limit > 200)
    return std::unexpected(ServiceError{ServiceFailure::kInvalidRequest, {}});
  std::string after;
  if (!cursor.empty()) {
    if (!cursor.starts_with("m1_") || (cursor.size() - 3) % 2 != 0)
      return std::unexpected(ServiceError{ServiceFailure::kInvalidRequest, {}});
    const auto digit = [](char value) -> int {
      if (value >= '0' && value <= '9') return value - '0';
      if (value >= 'a' && value <= 'f') return value - 'a' + 10;
      return -1;
    };
    for (size_t i = 3; i < cursor.size(); i += 2) {
      const int high = digit(cursor[i]), low = digit(cursor[i + 1]);
      if (high < 0 || low < 0)
        return std::unexpected(
            ServiceError{ServiceFailure::kInvalidRequest, {}});
      after.push_back(static_cast<char>((high << 4) | low));
    }
    const auto colon = after.find(':');
    if (colon == std::string::npos || colon + 1 == after.size() ||
        !FindTask(std::string_view(after).substr(0, colon)))
      return std::unexpected(ServiceError{ServiceFailure::kInvalidRequest, {}});
  }
  std::vector<ProtocolModel> models;
  models.reserve(catalog.models.size());
  for (const auto& model : catalog.models)
    models.push_back({model.task + ":" + model.name, model.task, model.name});
  std::ranges::sort(models, {}, &ProtocolModel::id);
  const auto first =
      std::ranges::upper_bound(models, after, {}, &ProtocolModel::id);
  ModelPage page;
  page.data.reserve(std::min(limit, static_cast<size_t>(models.end() - first)));
  auto it = first;
  for (; it != models.end() && page.data.size() < limit; ++it)
    page.data.emplace_back(std::move(*it));
  if (it != models.end()) {
    constexpr std::string_view digits = "0123456789abcdef";
    page.next_cursor = "m1_";
    for (unsigned char value : page.data.back().id) {
      page.next_cursor += digits[value >> 4];
      page.next_cursor += digits[value & 15];
    }
  }
  return page;
}
}  // namespace vision_simple
