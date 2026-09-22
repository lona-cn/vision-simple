#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <vector>

#include "Tracker.h"

using namespace vision_simple;
namespace {
void Check(bool ok, const char* message) {
  if (!ok) {
    std::cerr << "tracker regression: " << message << '\n';
    std::abort();
  }
}
std::unique_ptr<Tracker> Make(TrackerOptions options = {}) {
  auto result = Tracker::Create(options);
  Check(bool(result), "create tracker");
  return std::move(*result);
}
TrackingDetection Det(float x = 0, float confidence = .9f, int cls = 0,
                      float width = 20) {
  return {cls, confidence, {x, 40, width, 20}, {}};
}
TrackingFrameResult Step(Tracker& tracker, uint64_t index,
                         std::initializer_list<TrackingDetection> detections,
                         const cv::Mat& image = {}) {
  auto result =
      tracker.Step(index, double(index) / 30,
                   std::span(detections.begin(), detections.size()), image);
  Check(bool(result), "accept valid frame");
  return std::move(*result);
}
uint64_t Only(const TrackingFrameResult& result) {
  Check(result.tracks.size() == 1, "exactly one observed confirmed track");
  return result.tracks.front().track_id;
}
void RecoveryAndConfirmation() {
  auto tracker = Make();
  auto id = Only(Step(*tracker, 1, {Det()}));
  Step(*tracker, 2, {});
  Check(Step(*tracker, 3, {Det(0, .2f)}).tracks.empty(),
        "lost prediction cannot use low score");
  Check(Only(Step(*tracker, 4, {Det()})) == id,
        "high score recovers lost identity");
  tracker->Reset();
  Check(!tracker->Status().last_frame_index &&
            tracker->Status().active_tracks == 0,
        "reset clears sequence");
  Check(Only(Step(*tracker, 40, {Det()})) == 1,
        "first sequence frame confirms regardless of index");
  auto other = Make();
  Step(*other, 1, {});
  Check(Step(*other, 2, {Det()}).tracks.empty(),
        "later birth initially unconfirmed");
  Check(Only(Step(*other, 3, {Det()})) == 1, "second hit confirms birth");
  other->Reset();
  Step(*other, 1, {});
  Step(*other, 2, {Det()});
  Step(*other, 3, {});
  Check(Step(*other, 4, {Det()}).tracks.empty(),
        "missed tentative track removed");
  Check(Only(Step(*other, 5, {Det()})) == 2, "new tentative track has new ID");
}
void ExtendedConfirmation() {
  TrackerOptions options;
  options.min_hits = 20;
  auto tracker = Make(options);
  Step(*tracker, 0, {});
  for (uint64_t frame = 1; frame < 20; ++frame)
    Check(Step(*tracker, frame, {Det(float(frame - 1) * 2)}).tracks.empty(),
          "long probation does not confirm before required observations");
  const auto confirmed = Step(*tracker, 20, {Det(38)});
  Check(Only(confirmed) == 1 && confirmed.tracks[0].bbox.x > 37,
        "moving tentative track learns velocity and confirms without ID churn");
}
void ExpirationAndGaps() {
  TrackerOptions o;
  o.min_hits = 1;
  o.max_lost_frames = 2;
  auto t = Make(o);
  auto id = Only(Step(*t, 1, {Det()}));
  Check(Only(Step(*t, 5, {Det()})) != id,
        "explicit index gap expires track before matching");
  auto old = Only(Step(*t, 6, {Det()}));
  Step(*t, 7, {});
  Step(*t, 8, {});
  Step(*t, 9, {});
  Check(Only(Step(*t, 10, {Det()})) != old, "lost expiry produces new ID");
}
void TimestampMotion() {
  auto regular = Make(), delayed = Make();
  for (uint64_t frame = 1; frame <= 3; ++frame) {
    Step(*regular, frame, {Det(float(frame - 1) * 2)});
    Step(*delayed, frame, {Det(float(frame - 1) * 2)});
  }
  const auto normal = Step(*regular, 4, {Det(6)});
  std::array detection{Det(6)};
  auto later = delayed->Step(4, 8.0 / 30, detection);
  Check(bool(later) && Only(*later) == Only(normal),
        "elapsed-time prediction preserves identity");
  Check(std::abs(later->tracks[0].bbox.x - normal.tracks[0].bbox.x) > .01f,
        "motion prediction uses elapsed seconds rather than only frame index");
}
void GlobalAndClassAssignment() {
  auto t = Make();
  auto first = Step(*t, 1, {Det(0, 1), Det(10, 1)});
  Check(first.tracks.size() == 2, "two overlapping active objects retained");
  auto second = Step(*t, 2, {Det(4, .99f), Det(-8, 1)});
  Check(second.tracks.size() == 2,
        "global assignment preserves both feasible identities");
  Check(second.tracks[0].track_id == first.tracks[0].track_id &&
            second.tracks[0].confidence == 1 &&
            second.tracks[1].track_id == first.tracks[1].track_id &&
            second.tracks[1].confidence == .99f,
        "optimal assignment beats row-greedy IoU");
  t->Reset();
  first = Step(*t, 1, {Det(0, .9f, 0), Det(10, .9f, 1)});
  second = Step(*t, 2, {Det(10, .9f, 0), Det(0, .9f, 1)});
  Check(second.tracks.size() == 2 && second.tracks[0].class_id == 0 &&
            second.tracks[1].class_id == 1 &&
            second.tracks[0].bbox.x > first.tracks[0].bbox.x &&
            second.tracks[1].bbox.x < first.tracks[1].bbox.x,
        "class isolation overrides geometrically closer alternative");
}
void TransactionAndCapacity() {
  auto t = Make(), control = Make();
  Step(*t, 1, {Det()});
  Step(*control, 1, {Det()});
  std::array bad{Det()};
  bad[0].confidence = std::numeric_limits<float>::quiet_NaN();
  Check(!t->Step(2, 2.0 / 30, bad), "NaN rejected under fast floating point");
  std::array good{Det(2)};
  Check(!t->Step(1, 2.0 / 30, good), "duplicate index rejected");
  Check(!t->Step(2, 1.0 / 30, good), "nonmonotonic timestamp rejected");
  Check(!t->Step(2, std::numeric_limits<double>::infinity(), good),
        "infinite time rejected");
  auto actual = Step(*t, 2, {Det(2)}), expected = Step(*control, 2, {Det(2)});
  Check(Only(actual) == Only(expected) &&
            actual.tracks[0].bbox == expected.tracks[0].bbox,
        "rejected frames do not alter next accepted estimate");
  TrackerOptions o;
  o.max_tracks = 1;
  t = Make(o);
  Step(*t, 1, {Det()});
  std::array overflow{Det(), Det(100)};
  Check(!t->Step(2, 2.0 / 30, overflow),
        "capacity is explicit error, not eviction");
  Check(t->Status().last_frame_index == 1 && Only(Step(*t, 2, {Det()})) == 1,
        "capacity failure is transactional");
  o.high_threshold = std::numeric_limits<float>::quiet_NaN();
  Check(!Tracker::Create(o), "nonfinite option rejected");
  o = {};
  o.max_tracks = 257;
  Check(!Tracker::Create(o), "track bound enforced");
}
void Appearance() {
  TrackerOptions o;
  o.algorithm = TrackerAlgorithm::kBoTSORT;
  o.camera_motion = false;
  o.appearance = true;
  auto t = Make(o);
  auto left = Det(0), right = Det(5);
  left.embedding = {1, 0};
  right.embedding = {0, 1};
  auto first = Step(*t, 1, {left, right});
  left.bbox.x = 5;
  right.bbox.x = 0;
  left.confidence = .95f;
  right.confidence = .85f;
  auto crossed = Step(*t, 2, {left, right});
  Check(crossed.tracks.size() == 2 &&
            crossed.tracks[0].track_id == first.tracks[0].track_id &&
            crossed.tracks[0].confidence == .95f &&
            crossed.tracks[1].confidence == .85f,
        "supplied appearance disambiguates close same-class crossing");
  auto geometry = Make();
  Step(*geometry, 1, {Det(0), Det(5)});
  auto switched = Step(*geometry, 2, {Det(5, .95f), Det(0, .85f)});
  Check(switched.tracks[0].confidence == .85f,
        "geometry-only control takes spatial match");
  auto bad = left;
  bad.embedding = {0, 0};
  std::array invalid{bad};
  Check(!t->Step(3, .1, invalid), "zero appearance rejected");
  invalid[0].embedding = {1, 0, 0};
  Check(!t->Step(3, .1, invalid), "embedding dimension cannot change");
  invalid[0].embedding = {1, std::numeric_limits<float>::infinity()};
  Check(!t->Step(3, .1, invalid), "nonfinite embedding rejected");
  Check(t->Status().last_frame_index == 2, "invalid embeddings preserve time");
  Check(Step(*t, 3, {left, right}).tracks.size() == 2,
        "valid frame accepted after invalid embedding");
  t->Reset();
  left.embedding = {1, 0, 0};
  Check(Only(Step(*t, 1, {left})) == 1,
        "reset clears appearance dimensionality and IDs");
}
void ReliableAppearanceRecovery() {
  TrackerOptions options;
  options.algorithm = TrackerAlgorithm::kBoTSORT;
  options.camera_motion = false;
  options.appearance = true;
  auto tracker = Make(options);
  auto left = Det(0), right = Det(5);
  left.embedding = {1, 0};
  right.embedding = {0, 1};
  const auto first = Step(*tracker, 1, {left, right});
  auto occluded_left = left, occluded_right = right;
  occluded_left.confidence = occluded_right.confidence = .2f;
  occluded_left.embedding = right.embedding;
  occluded_right.embedding = left.embedding;
  for (uint64_t frame = 2; frame <= 31; ++frame) {
    const auto recovered =
        Step(*tracker, frame, {occluded_left, occluded_right});
    Check(recovered.tracks.size() == 2 &&
              recovered.tracks[0].track_id == first.tracks[0].track_id &&
              recovered.tracks[1].track_id == first.tracks[1].track_id,
          "low-score geometry recovery keeps both established identities");
  }
  left.bbox.x = 5;
  right.bbox.x = 0;
  left.confidence = .95f;
  right.confidence = .85f;
  const auto crossed = Step(*tracker, 32, {left, right});
  Check(crossed.tracks.size() == 2 &&
            crossed.tracks[0].track_id == first.tracks[0].track_id &&
            crossed.tracks[0].confidence == .95f &&
            crossed.tracks[1].confidence == .85f,
        "low-score occluder embeddings never overwrite reliable ReID history");
}
void CameraMotion() {
  cv::Mat image(240, 320, CV_8UC3);
  cv::RNG random(12345);
  random.fill(image, cv::RNG::UNIFORM, 0, 256);
  cv::GaussianBlur(image, image, {3, 3}, .6);
  cv::Mat shifted;
  cv::Matx23d translation(1, 0, 14, 0, 1, 0);
  cv::warpAffine(image, shifted, translation, image.size());
  TrackerOptions o;
  o.algorithm = TrackerAlgorithm::kBoTSORT;
  auto camera = Make(o);
  auto before = Det(120, .9f, 0, 10), after = Det(134, .9f, 0, 10);
  auto id = Only(Step(*camera, 1, {before}, image));
  Check(Only(Step(*camera, 2, {after}, shifted)) == id,
        "sparse-flow RANSAC compensates non-overlapping camera shift");
  o.camera_motion = false;
  auto fixed = Make(o);
  Step(*fixed, 1, {before});
  Check(Step(*fixed, 2, {after}).tracks.empty(),
        "without camera compensation shifted birth remains unconfirmed");
  std::array dets{after};
  Check(!camera->Step(3, .1, dets), "GMC requires image every frame");
  Check(!camera->Step(3, .1, dets, cv::Mat(10, 10, CV_8UC3)),
        "GMC rejects changed dimensions");
  Check(!camera->Step(3, .1, dets, cv::Mat(image.size(), CV_8UC1)),
        "GMC rejects non-BGR image");
  Check(camera->Status().last_frame_index == 2 &&
            Only(Step(*camera, 3, {after}, shifted)) == id,
        "image errors preserve accepted motion history");
  camera->Reset();
  cv::Mat blank(80, 80, CV_8UC3, cv::Scalar(0));
  const auto blank_id = Only(Step(*camera, 1, {Det()}, blank));
  Check(Only(Step(*camera, 2, {Det()}, blank)) == blank_id,
        "featureless frames use evidence-free identity warp");
}
void Isolation() {
  std::array<uint64_t, 8> final_ids{};
  std::vector<std::jthread> workers;
  for (size_t i = 0; i < final_ids.size(); ++i)
    workers.emplace_back([&, i] {
      auto t = Make();
      for (uint64_t f = 1; f <= 40; ++f)
        final_ids[i] = Only(Step(*t, f, {Det(float(f) / 4)}));
    });
  workers.clear();
  for (auto id : final_ids)
    Check(id == 1, "concurrent instances have independent ID/state allocation");
}
}  // namespace
int main() {
  RecoveryAndConfirmation();
  ExpirationAndGaps();
  GlobalAndClassAssignment();
  TransactionAndCapacity();
  TimestampMotion();
  Appearance();
  CameraMotion();
  Isolation();
  ExtendedConfirmation();
  ReliableAppearanceRecovery();
  for (auto algorithm :
       {TrackerAlgorithm::kByteTrack, TrackerAlgorithm::kBoTSORT}) {
    TrackerOptions o;
    o.algorithm = algorithm;
    o.camera_motion = false;
    auto t = Make(o);
    auto id = Only(Step(*t, 1, {Det()}));
    Check(Only(Step(*t, 2, {Det(1, .2f)})) == id,
          "both algorithms retain high-to-low identity");
  }
  std::cout << "tracker behavioral regressions passed\n";
}
