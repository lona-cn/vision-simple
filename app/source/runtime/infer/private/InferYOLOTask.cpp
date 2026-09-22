#include "InferYOLOTask.h"

#include <algorithm>
#include <charconv>
#include <climits>
#include <cmath>
#include <limits>
#include <mutex>
#include <regex>

#include "InferORT.h"
#include "InferTask.h"
#include "InferValidation.hpp"
#include "VisionHelper.hpp"
#include "YOLOMetadata.hpp"

namespace vision_simple {
namespace {
auto ModelError(const char* message) {
  return std::unexpected(
      VisionSimpleError{VisionSimpleErrorCode::kModelError, message});
}
bool FloatType(ONNXTensorElementDataType type) {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}
bool ShapeSize(const std::vector<int64_t>& shape, size_t& count) {
  count = 1;
  for (auto dimension : shape) {
    if (dimension <= 0 || dimension > INT_MAX ||
        count > std::numeric_limits<size_t>::max() / sizeof(float) /
                    static_cast<size_t>(dimension))
      return false;
    count *= static_cast<size_t>(dimension);
  }
  return true;
}

class TaskModel final : public InferYOLOTask {
 public:
  struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    ONNXTensorElementDataType type;
    size_t count;
  };
  std::unique_ptr<Ort::Session> session;
  Ort::Allocator allocator;
  Tensor input;
  std::vector<Tensor> outputs;
  YOLOTask kind;
  std::vector<std::string> names;
  size_t extra = 0, keypoints = 0, dimensions = 0;
  bool end_to_end = false;
  std::mutex session_mutex, pool_mutex;

  struct Workspace {
    Ort::Value input{nullptr};
    std::vector<Ort::Value> outputs;
    Ort::IoBinding binding;
    VisionHelper helper;
    LetterboxTransform transform;
    cv::Mat converted, logits;
    std::vector<std::vector<float>> fp32;
    explicit Workspace(TaskModel& model) : binding(*model.session) {
      input =
          Ort::Value::CreateTensor(model.allocator, model.input.shape.data(),
                                   model.input.shape.size(), model.input.type);
      fp32.resize(model.outputs.size());
      for (const auto& spec : model.outputs) {
        outputs.emplace_back(Ort::Value::CreateTensor(
            model.allocator, spec.shape.data(), spec.shape.size(), spec.type));
        binding.BindOutput(spec.name.c_str(), outputs.back());
      }
    }
  };
  std::array<std::unique_ptr<Workspace>, 2> idle;

  TaskModel(std::unique_ptr<Ort::Session> value, Ort::Allocator alloc,
            YOLOTask task)
      : session(std::move(value)), allocator(std::move(alloc)), kind(task) {}
  YOLOTask task() const noexcept override { return kind; }
  const std::vector<std::string>& class_names() const noexcept override {
    return names;
  }
  std::unique_ptr<Workspace> Acquire() {
    {
      std::lock_guard lock(pool_mutex);
      for (auto& entry : idle)
        if (entry) return std::move(entry);
    }
    std::lock_guard lock(session_mutex);
    return std::make_unique<Workspace>(*this);
  }
  void Release(std::unique_ptr<Workspace> workspace) noexcept {
    std::lock_guard lock(pool_mutex);
    for (auto& entry : idle)
      if (!entry) {
        entry = std::move(workspace);
        return;
      }
  }
  VSResult<void> Pre(Workspace& ws, const cv::Mat& image, float threshold) {
    auto valid = ValidateInferInput(image, threshold);
    if (!valid) return std::unexpected(std::move(valid.error()));
    auto& chw = ws.helper.Letterbox(
        image,
        {static_cast<int>(input.shape[3]), static_cast<int>(input.shape[2])},
        ws.transform);
    if (chw.empty())
      return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                        "Image dimensions cannot be letterboxed");
    ws.helper.HWC2CHW_BGR2RGB<uint8_t>(chw, chw);
    if (input.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      chw.convertTo(ws.converted, CV_32F, 1.0 / 255);
      Cvt::cvt(std::span<const float>(ws.converted.ptr<float>(), input.count),
               ws.input.GetTensorMutableData<Ort::Float16_t>());
    } else {
      cv::Mat target(chw.size(), CV_32FC3,
                     ws.input.GetTensorMutableData<float>());
      chw.convertTo(target, CV_32F, 1.0 / 255);
    }
    return {};
  }
  void Execute(Workspace& ws) {
    std::lock_guard lock(session_mutex);
    ws.binding.BindInput(input.name.c_str(), ws.input);
    Ort::RunOptions options;
    session->Run(options, ws.binding);
  }
  struct Candidate {
    size_t index;
    int32_t label;
    float score;
    cv::Vec4f xyxy;
    cv::Rect bbox;
    std::array<cv::Point2f, 4> corners;
  };
  static double IoU(const Candidate& a, const Candidate& b, bool rotated) {
    if (rotated) {
      std::vector<cv::Point2f> intersection;
      const double overlap = cv::intersectConvexConvex(
          cv::Mat(4, 1, CV_32FC2, const_cast<cv::Point2f*>(a.corners.data())),
          cv::Mat(4, 1, CV_32FC2, const_cast<cv::Point2f*>(b.corners.data())),
          intersection, true);
      const double area_a = std::abs(cv::contourArea(
          cv::Mat(4, 1, CV_32FC2, const_cast<cv::Point2f*>(a.corners.data()))));
      const double area_b = std::abs(cv::contourArea(
          cv::Mat(4, 1, CV_32FC2, const_cast<cv::Point2f*>(b.corners.data()))));
      const double total = area_a + area_b - overlap;
      return total > 0 ? overlap / total : 0;
    }
    const double width = std::max(0.0, double(std::min(a.xyxy[2], b.xyxy[2])) -
                                           std::max(a.xyxy[0], b.xyxy[0]));
    const double height = std::max(0.0, double(std::min(a.xyxy[3], b.xyxy[3])) -
                                            std::max(a.xyxy[1], b.xyxy[1]));
    const double overlap = width * height;
    const double total =
        (double(a.xyxy[2]) - a.xyxy[0]) * (double(a.xyxy[3]) - a.xyxy[1]) +
        (double(b.xyxy[2]) - b.xyxy[0]) * (double(b.xyxy[3]) - b.xyxy[1]) -
        overlap;
    return total > 0 ? overlap / total : 0;
  }
  RunResult Post(Workspace& ws, float threshold) {
    auto values = ws.binding.GetOutputValues();
    if (values.size() != outputs.size())
      return ModelError("Unexpected YOLO task outputs");
    std::vector<std::span<const float>> data;
    for (size_t i = 0; i < values.size(); ++i) {
      if (!values[i].IsTensor())
        return ModelError("YOLO task output is not a tensor");
      const auto info = values[i].GetTensorTypeAndShapeInfo();
      if (info.GetShape() != outputs[i].shape ||
          info.GetElementType() != outputs[i].type ||
          info.GetElementCount() != outputs[i].count)
        return ModelError("YOLO task output shape/type mismatch");
      if (outputs[i].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        ws.fp32[i].resize(outputs[i].count);
        Cvt::cvt(
            std::span<const Ort::Float16_t>(
                values[i].GetTensorData<Ort::Float16_t>(), outputs[i].count),
            ws.fp32[i].data());
        data.emplace_back(ws.fp32[i]);
      } else
        data.emplace_back(values[i].GetTensorData<float>(), outputs[i].count);
      for (float value : data.back())
        if (!IsFinite(value)) return ModelError("Non-finite YOLO task output");
    }
    const auto& transform = ws.transform;
    const size_t n = static_cast<size_t>(
        outputs[0].shape[end_to_end ? 1 : 2]);
    const size_t channels = static_cast<size_t>(
        outputs[0].shape[end_to_end ? 2 : 1]);
    const size_t extra_begin = end_to_end ? 6 : 4 + names.size();
    const auto at = [&](size_t row, size_t col) {
      return data[0][end_to_end ? col * channels + row : row * n + col];
    };
    const auto restore = [&](double x, double y) {
      return cv::Point2f(
          static_cast<float>((x - transform.left) / transform.gain_x),
          static_cast<float>((y - transform.top) / transform.gain_y));
    };
    std::vector<Candidate> candidates;
    for (size_t i = 0; i < n; ++i) {
      size_t label = 0;
      float score;
      if (end_to_end) {
        const float class_id = at(5, i);
        if (class_id < 0 || double(class_id) >= double(names.size()) ||
            std::floor(class_id) != class_id)
          return ModelError("Invalid YOLO end-to-end class index");
        label = static_cast<size_t>(class_id);
        score = at(4, i);
      } else {
        for (size_t c = 1; c < names.size(); ++c)
          if (at(4 + c, i) > at(4 + label, i)) label = c;
        score = at(4 + label, i);
      }
      if (score < threshold) continue;
      // OBB's dist2rbox emits xywh in both modes; the other end-to-end
      // heads emit xyxy. All task extras are already decoded by the graph.
      const bool xyxy = end_to_end && kind != YOLOTask::kOBB;
      const double w = xyxy ? double(at(2, i)) - at(0, i) : at(2, i);
      const double h = xyxy ? double(at(3, i)) - at(1, i) : at(3, i);
      const double x = xyxy ? double(at(0, i)) + w / 2 : at(0, i);
      const double y = xyxy ? double(at(1, i)) + h / 2 : at(1, i);
      if (w <= 0 || h <= 0) continue;
      Candidate candidate{
          i,
          static_cast<int32_t>(label),
          score,
          {static_cast<float>(x - w / 2), static_cast<float>(y - h / 2),
           static_cast<float>(x + w / 2), static_cast<float>(y + h / 2)},
          {},
          {}};
      for (float v : candidate.xyxy.val)
        if (!IsFinite(v)) return ModelError("YOLO box geometry overflow");
      if (kind == YOLOTask::kOBB) {
        const double angle = at(extra_begin, i), cosine = std::cos(angle),
                     sine = std::sin(angle);
        constexpr int signs[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        for (size_t j = 0; j < 4; ++j) {
          const double dx = signs[j][0] * w / 2, dy = signs[j][1] * h / 2;
          candidate.corners[j] =
              restore(x + dx * cosine - dy * sine, y + dx * sine + dy * cosine);
          if (!IsFinite(candidate.corners[j].x) ||
              !IsFinite(candidate.corners[j].y) ||
              std::abs(candidate.corners[j].x) > 1e15f ||
              std::abs(candidate.corners[j].y) > 1e15f)
            return ModelError("YOLO rotated box geometry overflow");
        }
      } else {
        candidate.bbox = VisionHelper::ScaleCoords(transform, candidate.xyxy);
        if (candidate.bbox.empty()) continue;
      }
      candidates.push_back(candidate);
    }
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a.score > b.score; });
    std::vector<Candidate> kept;
    for (const auto& candidate : candidates) {
      bool suppress = false;
      if (!end_to_end) {
        for (const auto& selected : kept)
          if (candidate.label == selected.label &&
              IoU(candidate, selected, kind == YOLOTask::kOBB) > .45) {
            suppress = true;
            break;
          }
      }
      if (!suppress) kept.push_back(candidate);
    }
    if (kind == YOLOTask::kOBB) {
      YOLOOBBFrame result;
      for (const auto& c : kept) {
        const auto edge = c.corners[1] - c.corners[0];
        result.results.push_back(
            {c.label, c.score, c.corners, std::atan2(edge.y, edge.x)});
      }
      return result;
    }
    if (kind == YOLOTask::kPose) {
      YOLOPoseFrame result;
      for (const auto& c : kept) {
        YOLOPoseResult item{c.label, c.score, c.bbox, {}};
        item.keypoints.reserve(keypoints);
        for (size_t k = 0; k < keypoints; ++k) {
          const size_t row = extra_begin + k * dimensions;
          const auto point = restore(at(row, c.index), at(row + 1, c.index));
          if (!IsFinite(point.x) || !IsFinite(point.y))
            return ModelError("YOLO keypoint geometry overflow");
          item.keypoints.push_back(
              {point.x, point.y, dimensions == 3 ? at(row + 2, c.index) : 1.f});
        }
        result.results.push_back(std::move(item));
      }
      return result;
    }
    YOLOSegmentationFrame result;
    const int ph = static_cast<int>(outputs[1].shape[2]),
              pw = static_cast<int>(outputs[1].shape[3]);
    const size_t plane = static_cast<size_t>(ph) * pw;
    ws.logits.create(ph, pw, CV_32FC1);
    for (const auto& c : kept) {
      float* logits = ws.logits.ptr<float>();
      for (size_t p = 0; p < plane; ++p) {
        double sum = 0;
        for (size_t m = 0; m < extra; ++m)
          sum += double(at(extra_begin + m, c.index)) *
                 data[1][m * plane + p];
        logits[p] = static_cast<float>(sum);
        if (!IsFinite(logits[p])) return ModelError("YOLO mask logit overflow");
      }
      cv::Mat mask(c.bbox.size(), CV_8UC1);
      // Bilinear, half-pixel pixel-center convention: map original centers
      // through exact gain_x/gain_y and floating letterbox offsets directly
      // into prototype space. Clamp at prototype edges, interpolate logits,
      // then threshold >0. No rounded prototype crop or binary-mask resize.
      for (int y = 0; y < mask.rows; ++y) {
        auto* dst = mask.ptr<uint8_t>(y);
        const double sy = std::clamp(
            ((c.bbox.y + y + .5) * transform.gain_y + transform.top) * ph /
                    input.shape[2] -
                .5,
            0.0, double(ph - 1));
        const int y0 = static_cast<int>(sy), y1 = std::min(y0 + 1, ph - 1);
        const double fy = sy - y0;
        for (int x = 0; x < mask.cols; ++x) {
          const double sx = std::clamp(
              ((c.bbox.x + x + .5) * transform.gain_x + transform.left) * pw /
                      input.shape[3] -
                  .5,
              0.0, double(pw - 1));
          const int x0 = static_cast<int>(sx), x1 = std::min(x0 + 1, pw - 1);
          const double fx = sx - x0;
          const double top = logits[size_t(y0) * pw + x0] * (1 - fx) +
                             logits[size_t(y0) * pw + x1] * fx;
          const double bottom = logits[size_t(y1) * pw + x0] * (1 - fx) +
                                logits[size_t(y1) * pw + x1] * fx;
          dst[x] = top * (1 - fy) + bottom * fy > 0 ? 255 : 0;
        }
      }
      result.results.push_back({c.label, c.score, c.bbox, std::move(mask)});
    }
    return result;
  }
  class Task final : public detail::FrameTask {
    TaskModel& model;
    const cv::Mat& image;
    float threshold;
    std::unique_ptr<Workspace> ws;
    detail::PipelineLane lane = detail::PipelineLane::kPreprocess;
    YOLOTaskFrameResult result;

   public:
    Task(TaskModel& m, const cv::Mat& img, float t)
        : model(m), image(img), threshold(t), ws(m.Acquire()) {}
    ~Task() override { model.Release(std::move(ws)); }
    VSResult<std::optional<detail::PipelineLane>> Advance() noexcept override {
      try {
        if (lane == detail::PipelineLane::kPreprocess) {
          auto prepared = model.Pre(*ws, image, threshold);
          if (!prepared) return std::unexpected(std::move(prepared.error()));
          lane = detail::PipelineLane::kInference;
          return lane;
        }
        if (lane == detail::PipelineLane::kInference) {
          model.Execute(*ws);
          lane = detail::PipelineLane::kPostprocess;
          return lane;
        }
        auto decoded = model.Post(*ws, threshold);
        if (!decoded) return std::unexpected(std::move(decoded.error()));
        result = std::move(*decoded);
        return std::nullopt;
      } catch (const std::exception& e) {
        return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError, e.what());
      }
    }
    detail::FrameResult TakeResult() noexcept override {
      return std::move(result);
    }
  };
  RunResult Run(const cv::Mat& image, float threshold) noexcept override {
    try {
      Task task(*this, image, threshold);
      for (;;) {
        auto next = task.Advance();
        if (!next) return std::unexpected(std::move(next.error()));
        if (!*next) return std::get<YOLOTaskFrameResult>(task.TakeResult());
      }
    } catch (const std::exception& e) {
      return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError, e.what());
    }
  }
};
}  // namespace

VSResult<std::unique_ptr<detail::FrameTask>> detail::MakeFrameTask(
    InferYOLOTask& model, const cv::Mat& image, float threshold) noexcept {
  try {
    auto* native = dynamic_cast<TaskModel*>(&model);
    if (!native)
      return MK_VSERROR(VisionSimpleErrorCode::kUnimplementedError,
                        "Unsupported staged YOLO task backend");
    return std::make_unique<TaskModel::Task>(*native, image, threshold);
  } catch (const std::exception& e) {
    return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError, e.what());
  }
}

InferYOLOTask::CreateResult InferYOLOTask::Create(InferContext& context,
                                                  std::span<uint8_t> bytes,
                                                  YOLOTask task,
                                                  YOLOVersion version,
                                                  size_t device_id) noexcept {
  try {
    if (version != YOLOVersion::kV11 && version != YOLOVersion::kV26)
      return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                        "Unsupported YOLO task version");
    if (task != YOLOTask::kSegmentation && task != YOLOTask::kPose &&
        task != YOLOTask::kOBB)
      return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                        "Unsupported YOLO task");
    auto* ort = dynamic_cast<InferContextORT*>(&context);
    if (!ort)
      return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                        "YOLO tasks require ONNX Runtime");
    auto created = ort->CreateSession(bytes, device_id);
    if (!created) return std::unexpected(std::move(created.error()));
    auto& session = **created;
    const size_t count = task == YOLOTask::kSegmentation ? 2 : 1;
    if (session.GetInputCount() != 1 || session.GetOutputCount() != count)
      return ModelError("YOLO task requires one input and task outputs");
    Ort::Allocator allocator(session, ort->env_memory_info());
    const auto metadata = session.GetModelMetadata();
    const auto args =
        metadata.LookupCustomMetadataMapAllocated("args", allocator);
    bool end_to_end = false;
    if (version == YOLOVersion::kV26) {
      const auto end2end =
          metadata.LookupCustomMetadataMapAllocated("end2end", allocator);
      const auto nms =
          metadata.LookupCustomMetadataMapAllocated("nms", allocator);
      if (!detail::ParseYOLO26Export(args ? args.get() : "",
                                     end2end ? end2end.get() : "",
                                     nms ? nms.get() : "", end_to_end))
        return ModelError("Unsupported YOLO26 export metadata");
    } else {
      bool exported_nms = false;
      for (const char* key : {"nms", "end2end"}) {
        const auto flag =
            metadata.LookupCustomMetadataMapAllocated(key, allocator);
        if (flag && detail::YOLOMetadataTrue(flag.get())) exported_nms = true;
      }
      if (args && !detail::ParseYOLOExportArgs(args.get(), exported_nms))
        return ModelError("Invalid YOLO export args metadata");
      if (exported_nms)
        return ModelError("YOLO11 tasks require raw outputs");
    }
    auto model = std::make_unique<TaskModel>(std::move(*created),
                                             std::move(allocator), task);
    model->end_to_end = end_to_end;
    const auto load = [&](size_t index, bool input,
                          TaskModel::Tensor& spec) -> bool {
      auto info = input ? session.GetInputTypeInfo(index)
                        : session.GetOutputTypeInfo(index);
      if (info.GetONNXType() != ONNX_TYPE_TENSOR) return false;
      const auto tensor = info.GetTensorTypeAndShapeInfo();
      spec.shape = tensor.GetShape();
      spec.type = tensor.GetElementType();
      spec.name =
          (input ? session.GetInputNameAllocated(index, model->allocator)
                 : session.GetOutputNameAllocated(index, model->allocator))
              .get();
      return FloatType(spec.type) && ShapeSize(spec.shape, spec.count);
    };
    if (!load(0, true, model->input) || model->input.shape.size() != 4 ||
        model->input.shape[0] != 1 || model->input.shape[1] != 3)
      return ModelError("YOLO task requires static [1,3,H,W] FP32/FP16 input");
    model->outputs.resize(count);
    for (size_t i = 0; i < count; ++i)
      if (!load(i, false, model->outputs[i]))
        return ModelError("YOLO task requires static FP32/FP16 tensor outputs");
    const auto task_name =
        metadata.LookupCustomMetadataMapAllocated("task", model->allocator);
    const std::string_view expected = task == YOLOTask::kSegmentation
                                          ? "segment"
                                      : task == YOLOTask::kPose ? "pose"
                                                                : "obb";
    if ((!task_name && version == YOLOVersion::kV26) ||
        (task_name && std::string_view(task_name.get()) != expected))
      return ModelError("YOLO task metadata does not match requested task");
    const auto names =
        metadata.LookupCustomMetadataMapAllocated("names", model->allocator);
    if (!names) return ModelError("Missing YOLO class names metadata");
    if (!detail::ParseYOLOClassNames(names.get(), model->names))
      return ModelError("Invalid YOLO class names metadata");
    const auto& shape = model->outputs[0].shape;
    const size_t base_channels = end_to_end ? 6 : 4 + model->names.size();
    if (shape.size() != 3 || shape[0] != 1 ||
        shape[end_to_end ? 2 : 1] <= static_cast<int64_t>(base_channels))
      return ModelError("Unsupported YOLO task prediction shape");
    model->extra =
        static_cast<size_t>(shape[end_to_end ? 2 : 1]) - base_channels;
    if (task == YOLOTask::kSegmentation) {
      const auto& proto = model->outputs[1].shape;
      if (proto.size() != 4 || proto[0] != 1 ||
          proto[1] != static_cast<int64_t>(model->extra))
        return ModelError(
            "YOLO prototype channels do not match mask coefficients");
    } else if (task == YOLOTask::kOBB) {
      if (model->extra != 1)
        return ModelError("YOLO OBB requires one angle channel");
    } else {
      model->dimensions = 3;
      const auto kpt_shape = metadata.LookupCustomMetadataMapAllocated(
          "kpt_shape", model->allocator);
      if (!kpt_shape && version == YOLOVersion::kV26)
        return ModelError("Missing YOLO26 keypoint shape metadata");
      if (kpt_shape) {
        const std::string value(kpt_shape.get());
        std::smatch match;
        const std::regex pattern(
            R"(^\s*[\[(]\s*(\d+)\s*,\s*([23])\s*[\])]\s*$)");
        if (!std::regex_match(value, match, pattern))
          return ModelError("Invalid YOLO kpt_shape metadata");
        const auto number = match[1].str();
        const auto parsed = std::from_chars(
            number.data(), number.data() + number.size(), model->keypoints);
        if (parsed.ec != std::errc{} || model->keypoints == 0)
          return ModelError("Invalid YOLO keypoint count");
        model->dimensions = match[2].str() == "2" ? 2 : 3;
        if (model->extra % model->dimensions ||
            model->keypoints != model->extra / model->dimensions)
          return ModelError(
              "YOLO keypoint metadata does not match output channels");
      } else {
        if (model->extra % 3)
          return ModelError(
              "YOLO pose without metadata requires XYZ-confidence triplets");
        model->keypoints = model->extra / 3;
      }
    }
    return model;
  } catch (const Ort::Exception& e) {
    return MK_VSERROR(VisionSimpleErrorCode::kModelError, e.what());
  } catch (const std::exception& e) {
    return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError, e.what());
  }
}

InferYOLOTask::CreateResult InferYOLOTask::Create(InferContext& context,
                                                  const std::string& path,
                                                  YOLOTask task,
                                                  YOLOVersion version,
                                                  size_t device_id) noexcept {
  try {
    auto bytes = ReadAll(path);
    if (!bytes) return std::unexpected(std::move(bytes.error()));
    return Create(context, bytes->span(), task, version, device_id);
  } catch (const std::exception& e) {
    return MK_VSERROR(VisionSimpleErrorCode::kModelError, e.what());
  }
}
}  // namespace vision_simple
