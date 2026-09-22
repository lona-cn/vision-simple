/*
 * ByteTrack/BoT-SORT algorithm adaptation:
 * https://github.com/ifzhang/ByteTrack (XYAH Kalman, two-stage association)
 * https://github.com/NirAharon/BoT-SORT (XYWH Kalman, GMC, appearance fusion)
 * MIT License
 * Copyright (c) 2021 Yifu Zhang
 * Copyright (c) 2022 Nir Aharon
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "Tracker.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace vision_simple {
namespace {
using M8 = cv::Matx<double, 8, 8>;
using V8 = cv::Vec<double, 8>;
using M4 = cv::Matx<double, 4, 4>;
using M84 = cv::Matx<double, 8, 4>;
bool Finite(double x) {
  return (std::bit_cast<uint64_t>(x) & UINT64_C(0x7ff0000000000000)) !=
         UINT64_C(0x7ff0000000000000);
}
bool Unit(float x) { return Finite(x) && x >= 0 && x <= 1; }
enum class Life { kActive, kLost, kRemoved };
struct Track {
  uint64_t id = 0, born = 0, observed = 0;
  uint32_t hits = 1;
  int32_t class_id = 0;
  float confidence = 0;
  bool confirmed = false;
  Life life = Life::kActive;
  V8 mean = V8::all(0);
  M8 covariance = M8::zeros();
  std::vector<double> feature;
};
cv::Vec4d Measurement(const cv::Rect2f& b, bool bot) {
  return {double(b.x) + double(b.width) / 2, double(b.y) + double(b.height) / 2,
          bot ? double(b.width) : double(b.width) / b.height, b.height};
}
cv::Rect2d Box(const Track& t, bool bot) {
  const double h = std::max(1e-4, t.mean[3]);
  const double w = std::max(1e-4, bot ? t.mean[2] : t.mean[2] * h);
  return {t.mean[0] - w / 2, t.mean[1] - h / 2, w, h};
}
double IoU(const cv::Rect2d& a, const cv::Rect2d& b) {
  const double w = std::max(
      0.0, std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x));
  const double h = std::max(
      0.0, std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y));
  const double intersection = w * h;
  return intersection /
         (a.width * a.height + b.width * b.height - intersection);
}
double Scale(const Track& t, int i, bool bot) {
  return std::max(1e-4, bot && i % 2 == 0 ? t.mean[2] : t.mean[3]);
}
void Initialize(Track& t, const TrackingDetection& d, bool bot) {
  auto z = Measurement(d.bbox, bot);
  for (int i = 0; i < 4; ++i) t.mean[i] = z[i];
  for (int i = 0; i < 4; ++i) {
    const double pos = !bot && i == 2 ? 1e-2 : Scale(t, i, bot) / 10;
    const double vel = !bot && i == 2 ? 1e-5 : Scale(t, i, bot) / 16;
    t.covariance(i, i) = pos * pos;
    t.covariance(i + 4, i + 4) = vel * vel;
  }
}
void Predict(Track& t, bool bot, double dt) {
  if (t.life == Life::kLost) {
    t.mean[7] = 0;
    if (bot) t.mean[6] = 0;
  }
  M8 f = M8::eye();
  for (int i = 0; i < 4; ++i) f(i, i + 4) = dt;
  M8 q = M8::zeros();
  for (int i = 0; i < 4; ++i) {
    double p = !bot && i == 2 ? 1e-2 : Scale(t, i, bot) / 20;
    double v = !bot && i == 2 ? 1e-5 : Scale(t, i, bot) / 160;
    // At dt=1 this is the reference diagonal process covariance.
    q(i, i) = p * p * dt;
    q(i + 4, i + 4) = v * v * dt;
  }
  t.mean = f * t.mean;
  t.covariance = f * t.covariance * f.t() + q;
}
bool Correct(Track& t, const TrackingDetection& d, bool bot) {
  M4 s, r = M4::zeros();
  M84 cross;
  for (int i = 0; i < 4; ++i) {
    const double sigma = !bot && i == 2 ? 0.1 : Scale(t, i, bot) / 20;
    r(i, i) = sigma * sigma;
    for (int j = 0; j < 4; ++j) s(i, j) = t.covariance(i, j) + r(i, j);
    for (int j = 0; j < 8; ++j) cross(j, i) = t.covariance(j, i);
  }
  bool ok = false;
  auto inverse = s.inv(cv::DECOMP_CHOLESKY, &ok);
  if (!ok) return false;
  auto k = cross * inverse;
  auto z = Measurement(d.bbox, bot);
  for (int i = 0; i < 4; ++i) z[i] -= t.mean[i];
  t.mean += k * z;
  // Joseph form avoids covariance loss of positive semidefiniteness.
  M8 residual = M8::eye();
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j) residual(i, j) -= k(i, j);
  t.covariance = residual * t.covariance * residual.t() + k * r * k.t();
  return true;
}
bool Healthy(const Track& t) {
  for (double v : t.mean.val)
    if (!Finite(v) || std::abs(v) > 1e14) return false;
  for (double v : t.covariance.val)
    if (!Finite(v) || std::abs(v) > 1e30) return false;
  return t.mean[2] > 0 && t.mean[3] > 0;
}
void Smooth(Track& t, const std::vector<double>& f) {
  if (f.empty()) return;
  if (t.feature.empty()) {
    t.feature = f;
    return;
  }
  double norm = 0;
  for (size_t i = 0; i < f.size(); ++i) {
    t.feature[i] = 0.9 * t.feature[i] + 0.1 * f[i];
    norm += t.feature[i] * t.feature[i];
  }
  norm = std::sqrt(norm);
  for (double& x : t.feature) x /= norm;
}
// Rectangular Hungarian shortest augmenting paths. Each row has an unmatched
// dummy column costing limit/2; real columns receive the other limit/2 credit.
// Thus an edge is chosen exactly when its total cost beats leaving both ends
// unmatched. Forbidden edges cannot force another row into a bad match.
std::vector<int> Assign(const std::vector<std::vector<double>>& costs,
                        size_t columns, double limit) {
  const size_t n = costs.size(), m = columns + n;
  std::vector<double> u(n + 1), v(m + 1);
  std::vector<size_t> p(m + 1), way(m + 1);
  for (size_t row = 1; row <= n; ++row) {
    p[0] = row;
    size_t j0 = 0;
    std::vector<double> minimum(m + 1, 1e9);
    std::vector<bool> used(m + 1, false);
    do {
      used[j0] = true;
      size_t i0 = p[j0], j1 = 0;
      double delta = 1e9;
      for (size_t j = 1; j <= m; ++j)
        if (!used[j]) {
          double c = limit / 2 + 1e-10;
          if (j <= columns)
            c = costs[i0 - 1][j - 1] <= limit ? costs[i0 - 1][j - 1] - limit / 2
                                              : 1e6;
          const double reduced = c - u[i0] - v[j];
          if (reduced < minimum[j]) {
            minimum[j] = reduced;
            way[j] = j0;
          }
          if (minimum[j] < delta) {
            delta = minimum[j];
            j1 = j;
          }
        }
      for (size_t j = 0; j <= m; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else
          minimum[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      size_t j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }
  std::vector<int> result(n, -1);
  for (size_t j = 1; j <= columns; ++j)
    if (p[j] && costs[p[j] - 1][j - 1] <= limit)
      result[p[j] - 1] = static_cast<int>(j - 1);
  return result;
}
cv::Matx23d Motion(const cv::Mat& previous, const cv::Mat& current,
                   const std::vector<cv::Point2f>& points) {
  cv::Matx23d identity(1, 0, 0, 0, 1, 0);
  if (previous.empty() || points.size() < 6) return identity;
  std::vector<cv::Point2f> forward, backward;
  std::vector<uchar> good, back_good;
  std::vector<float> errors;
  cv::calcOpticalFlowPyrLK(previous, current, points, forward, good, errors);
  cv::calcOpticalFlowPyrLK(current, previous, forward, backward, back_good,
                           errors);
  std::vector<cv::Point2f> from, to;
  for (size_t i = 0; i < points.size(); ++i) {
    if (!good[i] || !back_good[i] || cv::norm(points[i] - backward[i]) > 1.5)
      continue;
    if (!Finite(forward[i].x) || !Finite(forward[i].y)) continue;
    from.push_back(points[i]);
    to.push_back(forward[i]);
  }
  if (from.size() < 6) return identity;
  cv::Mat inliers;
  auto affine = cv::estimateAffinePartial2D(from, to, inliers, cv::RANSAC, 2.0,
                                            2000, 0.99, 10);
  if (affine.empty() || cv::countNonZero(inliers) < 6 ||
      cv::countNonZero(inliers) * 2 < static_cast<int>(from.size()))
    return identity;
  cv::Matx23d h;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) {
      h(i, j) = affine.at<double>(i, j);
      if (!Finite(h(i, j))) return identity;
    }
  return h;
}
void Warp(Track& t, const cv::Matx23d& h) {
  M8 transform = M8::zeros();
  for (int block = 0; block < 4; ++block)
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        // Centers/velocities rotate; axis-aligned extents use absolute
        // rotation.
        transform(block * 2 + i, block * 2 + j) =
            block % 2 ? std::abs(h(i, j)) : h(i, j);
  t.mean = transform * t.mean;
  t.mean[0] += h(0, 2);
  t.mean[1] += h(1, 2);
  t.covariance = transform * t.covariance * transform.t();
}
}  // namespace

struct Tracker::Impl {
  explicit Impl(TrackerOptions o) : options(o) {}
  TrackerOptions options;
  mutable std::mutex mutex;
  struct State {
    std::vector<Track> tracks;
    std::optional<uint64_t> index;
    std::optional<double> timestamp;
    uint64_t next_id = 1;
    size_t embedding_dimension = 0;
    cv::Mat gray;
    std::vector<cv::Point2f> points;
  } state;
};
Tracker::Tracker(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
Tracker::~Tracker() = default;
Tracker::CreateResult Tracker::Create(TrackerOptions o) noexcept {
  if ((o.algorithm != TrackerAlgorithm::kByteTrack &&
       o.algorithm != TrackerAlgorithm::kBoTSORT) ||
      !Unit(o.low_threshold) || !Unit(o.high_threshold) ||
      !Unit(o.new_track_threshold) || !Unit(o.match_threshold) ||
      !Unit(o.proximity_threshold) || !Unit(o.appearance_threshold) ||
      o.low_threshold >= o.high_threshold ||
      o.new_track_threshold < o.high_threshold || !o.max_tracks ||
      o.max_tracks > 256 || !o.max_detections || o.max_detections > 256 ||
      !o.min_hits || o.min_hits > 10000 || o.max_lost_frames > 10000 ||
      (o.appearance && o.algorithm != TrackerAlgorithm::kBoTSORT))
    return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                      "Invalid tracker options");
  try {
    return std::unique_ptr<Tracker>(new Tracker(std::make_unique<Impl>(o)));
  } catch (const std::exception& e) {
    return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError, e.what());
  }
}
const TrackerOptions& Tracker::options() const noexcept {
  return impl_->options;
}
void Tracker::Reset() noexcept {
  std::scoped_lock lock(impl_->mutex);
  impl_->state = {};
}
TrackingStatus Tracker::Status() const noexcept {
  std::scoped_lock lock(impl_->mutex);
  TrackingStatus result{impl_->state.index, impl_->state.timestamp};
  for (const auto& t : impl_->state.tracks)
    if (t.life == Life::kActive)
      ++result.active_tracks;
    else if (t.life == Life::kLost)
      ++result.lost_tracks;
  return result;
}
VSResult<TrackingFrameResult> Tracker::Step(
    uint64_t index, double timestamp,
    std::span<const TrackingDetection> detections,
    const cv::Mat& image) noexcept {
  std::scoped_lock lock(impl_->mutex);
  const auto& o = impl_->options;
  const auto& accepted = impl_->state;
  const bool bot = o.algorithm == TrackerAlgorithm::kBoTSORT;
  if (!Finite(timestamp) || (accepted.index && index <= *accepted.index) ||
      (accepted.timestamp && timestamp <= *accepted.timestamp))
    return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                      "Frame index and timestamp must increase strictly");
  // Velocities use reference-frame units (30 Hz); elapsed seconds, not index
  // gaps, determine motion. Index gaps independently determine expiration.
  const double dt =
      accepted.timestamp ? (timestamp - *accepted.timestamp) * 30.0 : 1.0;
  if (!Finite(dt) || dt <= 0 || dt > 1e6 ||
      detections.size() > o.max_detections)
    return MK_VSERROR(VisionSimpleErrorCode::kRangeError,
                      "Tracker timestep or detection limit exceeded");
  if (bot && o.camera_motion &&
      (image.empty() || image.type() != CV_8UC3 || image.cols < 8 ||
       image.rows < 8 ||
       (!accepted.gray.empty() && image.size() != accepted.gray.size())))
    return MK_VSERROR(
        VisionSimpleErrorCode::kParameterError,
        "Camera motion requires same-sized BGR images of at least 8x8");
  try {
    size_t dimension = accepted.embedding_dimension;
    std::vector<std::vector<double>> features(detections.size());
    for (size_t i = 0; i < detections.size(); ++i) {
      const auto& d = detections[i];
      if (d.class_id < 0 || !Unit(d.confidence) || !Finite(d.bbox.x) ||
          !Finite(d.bbox.y) || !Finite(d.bbox.width) ||
          !Finite(d.bbox.height) || std::abs(d.bbox.x) > 1e7 ||
          std::abs(d.bbox.y) > 1e7 || d.bbox.width < 1e-4 ||
          d.bbox.height < 1e-4 || d.bbox.width > 1e7 || d.bbox.height > 1e7 ||
          d.embedding.size() > 512)
        return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                          "Invalid tracking detection");
      double norm = 0;
      for (float x : d.embedding) {
        if (!Finite(x))
          return MK_VSERROR(VisionSimpleErrorCode::kParameterError,
                            "Nonfinite appearance embedding");
        norm += double(x) * x;
      }
      if (o.appearance) {
        if (d.embedding.empty() || norm <= 0 ||
            (dimension && dimension != d.embedding.size()))
          return MK_VSERROR(
              VisionSimpleErrorCode::kParameterError,
              "Appearance embeddings must be nonzero and session-sized");
        dimension = d.embedding.size();
        norm = std::sqrt(norm);
        features[i].reserve(dimension);
        for (float x : d.embedding) features[i].push_back(x / norm);
      }
    }
    // All changes, including GMC history, IDs and learned features, are staged.
    // Allocation/OpenCV/numerical/capacity failures leave the accepted state
    // intact.
    auto next = accepted;
    next.embedding_dimension = dimension;
    cv::Matx23d warp(1, 0, 0, 0, 1, 0);
    if (bot && o.camera_motion) {
      cv::Mat gray;
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
      warp = Motion(accepted.gray, gray, accepted.points);
      cv::Mat mask(gray.size(), CV_8UC1, cv::Scalar(255));
      for (const auto& d : detections) {
        const double x1 = std::clamp(double(d.bbox.x), 0.0, double(gray.cols));
        const double y1 = std::clamp(double(d.bbox.y), 0.0, double(gray.rows));
        const double x2 =
            std::clamp(double(d.bbox.x) + d.bbox.width, 0.0, double(gray.cols));
        const double y2 = std::clamp(double(d.bbox.y) + d.bbox.height, 0.0,
                                     double(gray.rows));
        if (x2 > x1 && y2 > y1)
          cv::rectangle(
              mask,
              cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                       static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)),
              cv::Scalar(0), cv::FILLED);
      }
      next.points.clear();
      cv::goodFeaturesToTrack(gray, next.points, 1000, 0.01, 8, mask);
      next.gray = std::move(gray);
    }
    std::vector<size_t> pool, tentative, high, low;
    for (size_t i = 0; i < next.tracks.size(); ++i) {
      auto& t = next.tracks[i];
      // A contiguous active observation can continue with max_lost_frames=0.
      const uint64_t gap = index - t.observed;
      if ((t.life == Life::kLost && gap > o.max_lost_frames) ||
          (t.life == Life::kActive && gap > 1 && gap > o.max_lost_frames) ||
          (!t.confirmed && gap > 1)) {
        t.life = Life::kRemoved;
        continue;
      }
      if (gap > 1) t.life = Life::kLost;
      // Reference two-hit probation has no prediction. Longer configurable
      // probation must learn velocity rather than repeatedly averaging boxes.
      if (t.confirmed || o.min_hits > 2) Predict(t, bot, dt);
      if (t.confirmed)
        pool.push_back(i);
      else
        tentative.push_back(i);
      if (bot && o.camera_motion) Warp(t, warp);
      if (!Healthy(t))
        return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError,
                          "Tracker prediction became numerically invalid");
    }
    for (size_t i = 0; i < detections.size(); ++i)
      if (detections[i].confidence >= o.high_threshold)
        high.push_back(i);
      else if (detections[i].confidence >= o.low_threshold)
        low.push_back(i);
    std::vector<bool> matched(next.tracks.size(), false),
        consumed(detections.size(), false);
    auto associate = [&](const std::vector<size_t>& rows,
                         const std::vector<size_t>& cols, double limit,
                         bool fuse) {
      std::vector<std::vector<double>> costs(
          rows.size(), std::vector<double>(cols.size(), 1e6));
      for (size_t r = 0; r < rows.size(); ++r)
        for (size_t c = 0; c < cols.size(); ++c) {
          const auto& t = next.tracks[rows[r]];
          const auto& d = detections[cols[c]];
          if (consumed[cols[c]] || t.class_id != d.class_id) continue;
          double overlap = IoU(Box(t, bot), cv::Rect2d(d.bbox));
          double cost = 1 - overlap * (fuse ? d.confidence : 1.0);
          if (fuse && o.appearance && 1 - overlap <= o.proximity_threshold &&
              !t.feature.empty()) {
            const auto& f = features[cols[c]];
            double cosine = std::inner_product(t.feature.begin(),
                                               t.feature.end(), f.begin(), 0.0);
            double appearance = (1 - std::clamp(cosine, -1.0, 1.0)) / 2;
            if (appearance <= o.appearance_threshold)
              cost = std::min(cost, appearance);
          }
          costs[r][c] = cost;
        }
      auto assignment = Assign(costs, cols.size(), limit);
      for (size_t r = 0; r < rows.size(); ++r)
        if (assignment[r] >= 0) {
          const size_t c = cols[assignment[r]];
          auto& t = next.tracks[rows[r]];
          if (!Correct(t, detections[c], bot)) return false;
          // Low-score recovery is geometry-only: do not learn an occluder's
          // appearance from detections excluded from the reliable feature
          // stage.
          if (fuse) Smooth(t, features[c]);
          t.life = Life::kActive;
          t.observed = index;
          t.confidence = detections[c].confidence;
          if (t.hits < o.min_hits) ++t.hits;
          t.confirmed = t.confirmed || t.hits >= o.min_hits;
          matched[rows[r]] = true;
          consumed[c] = true;
        }
      return true;
    };
    if (!associate(pool, high, o.match_threshold, true))
      return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError,
                        "Kalman correction failed");
    std::vector<size_t> remaining;
    for (size_t r : pool)
      if (!matched[r] && next.tracks[r].life == Life::kActive)
        remaining.push_back(r);
    if (!associate(remaining, low, 0.5, false) ||
        !associate(tentative, high, 0.7, true))
      return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError,
                        "Kalman correction failed");
    for (size_t r : pool)
      if (!matched[r]) {
        next.tracks[r].life =
            index - next.tracks[r].observed > o.max_lost_frames ? Life::kRemoved
                                                                : Life::kLost;
      }
    for (size_t r : tentative)
      if (!matched[r]) next.tracks[r].life = Life::kRemoved;
    for (size_t c : high)
      if (!consumed[c] && detections[c].confidence >= o.new_track_threshold) {
        if (next.next_id == std::numeric_limits<uint64_t>::max())
          return MK_VSERROR(VisionSimpleErrorCode::kRangeError,
                            "Track ID space exhausted");
        Track t;
        t.id = next.next_id++;
        t.born = t.observed = index;
        t.class_id = detections[c].class_id;
        t.confidence = detections[c].confidence;
        t.confirmed = !accepted.index || o.min_hits == 1;
        Initialize(t, detections[c], bot);
        Smooth(t, features[c]);
        next.tracks.push_back(std::move(t));
      }
    // Reference duplicate suppression compares active versus lost tracks only;
    // overlapping active same-class objects must remain independently
    // trackable.
    for (auto& a : next.tracks)
      if (a.life == Life::kActive)
        for (auto& b : next.tracks)
          if (b.life == Life::kLost && a.class_id == b.class_id &&
              IoU(Box(a, bot), Box(b, bot)) > 0.85) {
            if (a.observed - a.born >= b.observed - b.born)
              b.life = Life::kRemoved;
            else {
              a.life = Life::kRemoved;
              break;
            }
          }
    std::erase_if(next.tracks,
                  [](const Track& t) { return t.life == Life::kRemoved; });
    if (next.tracks.size() > o.max_tracks)
      return MK_VSERROR(VisionSimpleErrorCode::kRangeError,
                        "Tracker capacity exceeded; frame was not accepted");
    TrackingFrameResult result{index, timestamp, {}};
    for (const auto& t : next.tracks) {
      if (!Healthy(t))
        return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError,
                          "Tracker state became numerically invalid");
      if (t.life == Life::kActive && t.confirmed && t.observed == index)
        result.tracks.push_back(
            {t.id, t.class_id, t.confidence, cv::Rect2f(Box(t, bot))});
    }
    next.index = index;
    next.timestamp = timestamp;
    impl_->state = std::move(next);
    return result;
  } catch (const std::exception& e) {
    return MK_VSERROR(VisionSimpleErrorCode::kRuntimeError, e.what());
  }
}
}  // namespace vision_simple
