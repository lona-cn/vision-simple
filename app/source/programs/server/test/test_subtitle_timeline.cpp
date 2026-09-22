#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "SubtitleTimeline.h"

using namespace vision_simple;
namespace {
void Check(bool ok, const char* message) {
  if (!ok) {
    std::cerr << "subtitle timeline regression: " << message << '\n';
    std::abort();
  }
}
void Cue(const SubtitleTimeline& timeline, size_t index, int64_t start,
         int64_t end, const std::string& text) {
  Check(index < timeline.cues().size(), "expected cue exists");
  const auto& cue = timeline.cues()[index];
  Check(cue.start_ms == start && cue.end_ms == end && cue.text == text,
        "cue text and exact interval");
}
void StabilityAndNoise() {
  SubtitleTimeline timeline;
  Check(timeline.Observe(0, "caption"), "first caption sample");
  Check(timeline.Observe(40, "caption"), "caption confirmation");
  Check(timeline.Observe(80, "hallucination"), "transient changed text");
  Check(timeline.Observe(120, "caption"), "recover caption");
  Check(timeline.Observe(160, ""), "transient blank");
  Check(timeline.Observe(200, "caption"), "recover blank");
  for (int64_t time = 240; time < 1000; time += 40)
    Check(timeline.Observe(time, "caption"), "stable repeated samples");
  Check(timeline.Finish(1001), "finish stable caption");
  Check(timeline.cues().size() == 1,
        "no per-frame or transient duplicate cues");
  Cue(timeline, 0, 0, 1001, "caption");
}
void TransitionsAndGaps() {
  SubtitleTimeline timeline;
  Check(timeline.Observe(0, "unconfirmed"), "initial noise");
  Check(timeline.Observe(10, "A"), "first real caption");
  Check(timeline.Observe(20, "A"), "confirm first caption");
  Check(timeline.Observe(30, "B"), "new caption starts");
  Check(timeline.Observe(40, "B"), "confirm persistent change");
  Check(timeline.Observe(50, ""), "gap starts");
  Check(timeline.Observe(60, ""), "confirm gap");
  Check(timeline.Observe(70, "B"), "same text after real gap");
  Check(timeline.Observe(80, "B"), "confirm reappearance");
  Check(timeline.Observe(90, "discard"), "unconfirmed final change");
  Check(timeline.Finish(100), "finish at actual end");
  Check(timeline.cues().size() == 3, "persistent changes and gaps split cues");
  Cue(timeline, 0, 10, 30, "A");
  Cue(timeline, 1, 30, 50, "B");
  Cue(timeline, 2, 70, 100, "B");

  SubtitleTimeline pending_gap;
  Check(pending_gap.Observe(1, "A") && pending_gap.Observe(2, "A") &&
            pending_gap.Observe(3, "") && pending_gap.Finish(10),
        "finish with a pending blank");
  Cue(pending_gap, 0, 1, 3, "A");
  SubtitleTimeline unconfirmed;
  Check(unconfirmed.Observe(1, "noise") && unconfirmed.Finish(10),
        "finish unconfirmed initial text");
  Check(unconfirmed.cues().empty(), "initial hallucination discarded");
  SubtitleTimeline consecutive(3, 3);
  Check(consecutive.Observe(0, "A") && consecutive.Observe(1, "A") &&
            consecutive.Observe(2, "") && consecutive.Observe(3, "A") &&
            consecutive.Observe(4, "A") && consecutive.Finish(5),
        "interrupted confirmation sequence");
  Check(consecutive.cues().empty(), "only consecutive samples confirm text");
}
void TextAndCodec() {
  const std::string unicode = "\xe4\xbd\xa0\xe5\xa5\xbd \xf0\x9f\x8e\xac";
  SubtitleTimeline timeline(1, 1);
  const std::string source = " \t<b>& " + unicode +
                             "</b>\r\n\r\n  1\n"
                             "00:00:00,000 --> 00:00:01,000\n\n\t";
  Check(timeline.Observe(3600001, source) && timeline.Finish(3661234),
        "normalize unicode multiline caption");
  const std::string text = "<b>& " + unicode +
                           "</b>\n1\n"
                           "00:00:00,000 --> 00:00:01,000";
  Cue(timeline, 0, 3600001, 3661234, text);
  const std::string escaped =
      "&lt;b&gt;&amp; " + unicode +
      "&lt;/b&gt;\n1\n00:00:00,000 --&gt; 00:00:01,000\n\n";
  Check(EncodeSubtitles(timeline.cues(), false) ==
            "1\n01:00:00,001 --> 01:01:01,234\n" + escaped,
        "SRT exact timing, numbering, safe markup and no blank-line injection");
  Check(EncodeSubtitles(timeline.cues(), true) ==
            "WEBVTT\n\n01:00:00.001 --> 01:01:01.234\n" + escaped,
        "VTT exact timing, header and safe unicode text");
  Check(EncodeSubtitles({{0, 1, "A"}, {1, 1000, "B"}}, false) ==
            "1\n00:00:00,000 --> 00:00:00,001\nA\n\n"
            "2\n00:00:00,001 --> 00:00:01,000\nB\n\n",
        "numbering and adjacent positive intervals");
  SubtitleTimeline normalized;
  Check(normalized.Observe(0, " A\t B \r\n\n C ") &&
            normalized.Observe(1, "A B\nC") && normalized.Finish(2),
        "normalization precedes equality confirmation");
  Cue(normalized, 0, 0, 2, "A B\nC");
  SubtitleTimeline controls(1, 1);
  Check(controls.Observe(0, std::string("A\0B\x7f C", 7)) && controls.Finish(1),
        "normalize embedded controls");
  Cue(controls, 0, 0, 1, "A B C");
}
void InvalidInputAndBoundaries() {
  SubtitleTimeline invalid_stable(0, 2), invalid_gap(2, 0);
  Check(!invalid_stable.Observe(0, "A") && !invalid_stable.Finish(1) &&
            !invalid_gap.Observe(0, "A") && !invalid_gap.Finish(1),
        "zero confirmation options rejected");
  SubtitleTimeline timeline(1, 1);
  Check(!timeline.Observe(-1, "A") && !timeline.Finish(-1),
        "negative time rejected");
  for (const std::string malformed :
       {std::string("\xc0\xaf"), std::string("\xe0\x80\x80"),
        std::string("\xed\xa0\x80"), std::string("\xf4\x90\x80\x80"),
        std::string("\x80"), std::string("\xf0\x9f"), std::string("\xc2x")})
    Check(!timeline.Observe(0, malformed),
          "malformed UTF-8 rejected without consuming timestamp");
  Check(!timeline.Observe(0, std::string(4097, 'a')), "observation byte limit");
  Check(timeline.Observe(0, std::string(4096, 'a')),
        "exact observation boundary");
  Check(!timeline.Observe(0, "A") && !timeline.Observe(-1, "A"),
        "duplicate and decreasing observations rejected");
  Check(timeline.Observe(2, "A") && !timeline.Finish(1) && timeline.Finish(3),
        "finish cannot precede last observation");
  Check(!timeline.Observe(4, "B") && !timeline.Finish(4),
        "finished timeline is sealed");
  SubtitleTimeline zero(1, 1);
  Check(zero.Observe(0, "A") && zero.Finish(0) && zero.cues().empty(),
        "zero duration caption never emitted");
  SubtitleTimeline maximum(1, 1);
  const auto end = std::numeric_limits<int64_t>::max();
  Check(maximum.Observe(end - 1, "A") && maximum.Finish(end),
        "maximum time without overflow");
  Check(EncodeSubtitles(maximum.cues(), false) ==
            "1\n2562047788015:12:55,806 --> 2562047788015:12:55,807\nA\n\n",
        "large integer timestamp formatting does not overflow");
  Check(EncodeSubtitles({{-1, 1, "A"}}, false).empty() &&
            EncodeSubtitles({{0, 0, "A"}}, false).empty() &&
            EncodeSubtitles({{0, 3, "A"}, {2, 4, "B"}}, false).empty(),
        "codec rejects negative, zero and overlapping intervals");
}
void ResourceLimits() {
  SubtitleTimeline cues(1, 1);
  for (int64_t i = 0; i < 10000; ++i)
    Check(cues.Observe(i, i % 2 ? "A" : "B"), "accept bounded cue count");
  Check(!cues.Observe(10000, "B"), "reject 10001st cue including active cue");
  SubtitleTimeline exact_cues(1, 1);
  for (int64_t i = 0; i < 10000; ++i)
    Check(exact_cues.Observe(i, i % 2 ? "A" : "B"), "exact cue count sequence");
  Check(exact_cues.Finish(10000) && exact_cues.cues().size() == 10000,
        "exact cue limit is finishable");
  SubtitleTimeline bytes(1, 1);
  for (int64_t i = 0; i < 512; ++i)
    Check(bytes.Observe(i, std::string(4096, i % 2 ? 'A' : 'B')),
          "accept exactly two MiB of text");
  Check(!bytes.Observe(512, "overflow"), "reject total text overflow");
  SubtitleTimeline exact_bytes(1, 1);
  for (int64_t i = 0; i < 512; ++i)
    Check(exact_bytes.Observe(i, std::string(4096, i % 2 ? 'A' : 'B')),
          "exact text budget sequence");
  Check(exact_bytes.Finish(512) && exact_bytes.cues().size() == 512,
        "exact text budget is finishable");
}
}  // namespace
int main() {
  StabilityAndNoise();
  TransitionsAndGaps();
  TextAndCodec();
  InvalidInputAndBoundaries();
  ResourceLimits();
  std::cout << "subtitle timeline regressions passed\n";
}
