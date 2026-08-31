#include "graphics/host_gpu/contentVersionMap.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Libs::Graphics::BuildContentReadPlan;
using Libs::Graphics::ContentDomain;
using Libs::Graphics::ContentImageCandidate;
using Libs::Graphics::ContentRange;
using Libs::Graphics::ContentReadPlan;
using Libs::Graphics::ContentSerial;
using Libs::Graphics::ContentSourceKind;
using Libs::Graphics::ContentVersionMap;
using Libs::Graphics::ContentVersionTracker;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "ContentVersionMapTests: failed: %s\n", text);
    std::abort();
  }
}

const auto &SourceAt(const ContentReadPlan &plan, uint64_t address) {
  const auto source =
      std::ranges::find_if(plan.spans, [address](const auto &span) {
        return span.range.address <= address && address < span.range.End();
      });
  Check(source != plan.spans.end(),
        "plan does not cover the requested address");
  return *source;
}

void TestSplitAndCoalesce() {
  ContentVersionMap versions;
  Check(versions.Stamp(ContentDomain::Cpu, {0x1000, 0x1000}, 1),
        "initial CPU stamp failed");
  Check(versions.Stamp(ContentDomain::Buffer, {0x1400, 0x400}, 2),
        "middle Buffer stamp failed");
  Check(versions.SegmentCount() == 3,
        "middle stamp did not split both boundaries");
  const auto split = versions.Query({0x1000, 0x1000});
  Check(split.size() == 3 && split[0].range == ContentRange{0x1000, 0x400} &&
            split[1].range == ContentRange{0x1400, 0x400} &&
            split[2].range == ContentRange{0x1800, 0x800},
        "split query returned wrong ranges");
  Check(split[0].versions.cpu == 1 && split[0].versions.buffer == 0 &&
            split[1].versions.cpu == 1 && split[1].versions.buffer == 2,
        "split query returned wrong versions");

  Check(versions.Stamp(ContentDomain::Buffer, {0x1000, 0x400}, 2),
        "left Buffer stamp failed");
  Check(versions.Stamp(ContentDomain::Buffer, {0x1800, 0x800}, 2),
        "right Buffer stamp failed");
  Check(versions.SegmentCount() == 1,
        "equal adjacent segments did not coalesce");
  const auto coalesced = versions.Query({0x1000, 0x1000});
  Check(coalesced.size() == 1 && coalesced[0].versions.cpu == 1 &&
            coalesced[0].versions.buffer == 2,
        "coalesced segment lost a domain version");
}

void TestGapsAndInvalidRanges() {
  ContentVersionMap versions;
  Check(versions.Stamp(ContentDomain::Buffer, {0x2100, 0x100}, 7),
        "gap test stamp failed");
  const auto query = versions.Query({0x2000, 0x300});
  Check(query.size() == 3 && query[0].range == ContentRange{0x2000, 0x100} &&
            query[1].range == ContentRange{0x2100, 0x100} &&
            query[2].range == ContentRange{0x2200, 0x100},
        "query did not materialize both gaps");
  Check(query[0].versions == Libs::Graphics::ContentVersions{} &&
            query[1].versions.buffer == 7 &&
            query[2].versions == Libs::Graphics::ContentVersions{},
        "gap query assigned an invalid source");
  const auto incomplete = BuildContentReadPlan(versions, {0x2000, 0x300});
  Check(incomplete.valid_range && !incomplete.complete,
        "planner did not report uncovered gaps");

  const auto old_count = versions.SegmentCount();
  Check(!versions.Stamp(ContentDomain::Cpu, {0, 1}, 8),
        "zero address was accepted");
  Check(!versions.Stamp(ContentDomain::Cpu, {0x3000, 0}, 8),
        "zero size was accepted");
  Check(!versions.Stamp(ContentDomain::Cpu, {UINT64_MAX - 7, 8}, 8),
        "wrapping range was accepted");
  Check(!versions.Stamp(ContentDomain::Cpu, {0x3000, 1}, 0),
        "zero content serial was accepted");
  Check(versions.SegmentCount() == old_count, "invalid stamp mutated the map");
  Check(versions.Query({UINT64_MAX - 7, 8}).empty(),
        "invalid query returned slices");
  Check(!BuildContentReadPlan(versions, {UINT64_MAX - 7, 8}).valid_range,
        "planner accepted an overflowing request");

  ContentSerial first = 0;
  ContentSerial second = 0;
  Check(versions.NextSerial(first) && versions.NextSerial(second) &&
            first == 8 && second == 9,
        "serial allocation is not monotonic");

  ContentVersionMap external_serial;
  Check(external_serial.Stamp(ContentDomain::Cpu, {0x4000, 1}, 70),
        "external serial stamp failed");
  ContentSerial after_external = 0;
  Check(external_serial.NextSerial(after_external) && after_external == 71,
        "serial allocation collided with an externally stamped version");
}

void TestTraceShapeAndStableSelection() {
  constexpr ContentRange image29{0x000000028d700000, 0x0000000003c00000};
  constexpr ContentRange image60{0x000000028d0f0000, 0x0000000000780000};
  constexpr ContentRange image61{0x000000028d870000, 0x0000000000780000};
  constexpr ContentRange image62{0x000000028dff0000, 0x0000000000780000};
  constexpr ContentRange image63{0x000000028e770000, 0x0000000000780000};
  constexpr ContentRange newer_buffer{0x000000028e100000, 0x0000000000010000};

  ContentVersionMap versions;
  Check(versions.Stamp(ContentDomain::Buffer, image29, 5),
        "older containing Buffer stamp failed");
  Check(versions.Stamp(ContentDomain::Buffer, newer_buffer, 30),
        "newer Buffer subspan stamp failed");
  std::vector<ContentImageCandidate> images{
      {image29, 10, 29}, {image60, 20, 60}, {image61, 21, 61},
      {image62, 22, 62}, {image63, 23, 63},
  };
  const auto plan = BuildContentReadPlan(versions, image29, images);
  Check(plan.valid_range && plan.complete, "trace-shaped plan is incomplete");
  Check(SourceAt(plan, image29.address).source_id == 60,
        "nested image60 did not beat containing image29");
  Check(SourceAt(plan, image61.address).source_id == 61,
        "nested image61 was not selected");
  Check(SourceAt(plan, image62.address).source_id == 62,
        "nested image62 was not selected");
  Check(SourceAt(plan, newer_buffer.address).kind ==
                ContentSourceKind::Buffer &&
            SourceAt(plan, newer_buffer.address).serial == 30,
        "newer Buffer subspan did not beat nested image62");
  Check(SourceAt(plan, image63.address).source_id == 63,
        "nested image63 was not selected");
  Check(SourceAt(plan, image63.End()).source_id == 29,
        "containing image29 did not cover the remaining tail");

  std::ranges::reverse(images);
  const auto reversed = BuildContentReadPlan(versions, image29, images);
  Check(reversed.valid_range == plan.valid_range &&
            reversed.complete == plan.complete && reversed.spans == plan.spans,
        "candidate input order changed source selection");
}

void TestSecondFrameAndTieSelection() {
  constexpr ContentRange image29{0x000000028d700000, 0x0000000003c00000};
  constexpr ContentRange first_write{0x000000028d700000, 0x0000000000f00000};
  constexpr ContentRange image61{0x000000028d870000, 0x0000000000780000};

  ContentVersionMap versions;
  Check(versions.Stamp(ContentDomain::Buffer, first_write, 40),
        "first-frame Buffer result stamp failed");
  std::array images{ContentImageCandidate{image29, 50, 29},
                    ContentImageCandidate{image61, 51, 61}};
  const auto second_frame = BuildContentReadPlan(versions, image29, images);
  Check(second_frame.complete &&
            SourceAt(second_frame, image29.address).source_id == 29 &&
            SourceAt(second_frame, image61.address).source_id == 61,
        "second frame did not select newer rendered images");

  Check(versions.Stamp(ContentDomain::Buffer, first_write, 60),
        "second-frame Buffer result stamp failed");
  const auto after_dispatch = BuildContentReadPlan(versions, image29, images);
  Check(SourceAt(after_dispatch, image29.address).kind ==
                ContentSourceKind::Buffer &&
            SourceAt(after_dispatch, image61.address).kind ==
                ContentSourceKind::Buffer &&
            SourceAt(after_dispatch, first_write.End()).source_id == 29,
        "second-frame Buffer snapshot did not win only its stamped range");

  ContentVersionMap tie_versions;
  Check(tie_versions.Stamp(ContentDomain::Cpu, {0x4000, 0x100}, 70),
        "tie CPU stamp failed");
  Check(tie_versions.Stamp(ContentDomain::Buffer, {0x4000, 0x100}, 70),
        "tie Buffer stamp failed");
  std::array tied_images{ContentImageCandidate{{0x4000, 0x100}, 70, 9},
                         ContentImageCandidate{{0x4000, 0x100}, 70, 3}};
  const auto tie =
      BuildContentReadPlan(tie_versions, {0x4000, 0x100}, tied_images);
  Check(tie.complete && tie.spans.size() == 1 &&
            tie.spans.front().kind == ContentSourceKind::Buffer,
        "stable tie policy did not prefer the destination Buffer domain");

  ContentVersionMap image_tie_versions;
  const auto image_tie =
      BuildContentReadPlan(image_tie_versions, {0x4000, 0x100}, tied_images);
  Check(image_tie.complete && image_tie.spans.size() == 1 &&
            image_tie.spans.front().kind == ContentSourceKind::Image &&
            image_tie.spans.front().source_id == 3,
        "stable image tie policy did not choose the lowest source id");
}

void TestTrackedInterestAndLateRegistration() {
  ContentVersionTracker tracker;
  Check(tracker.Enabled(), "production tracker is not active");
  Check(tracker.Stamp(ContentDomain::Cpu, {0x8000, 0x100}, "outside") == 0 &&
            tracker.Query({0x8000, 0x100}).front().versions ==
                Libs::Graphics::ContentVersions{},
        "CPU write outside tracked interest changed content versions");

  Check(tracker.EnsureTracked({0x8000, 0x100}),
        "late tracked-range registration failed");
  const auto baseline = tracker.Query({0x8000, 0x100});
  Check(baseline.size() == 1 && baseline.front().versions.cpu == 1 &&
            baseline.front().versions.buffer == 0,
        "late registration did not seed one CPU baseline");
  Check(tracker.EnsureTracked({0x8080, 0x40}),
        "re-registering tracked interest failed");
  Check(tracker.ReserveSerial() == 2,
        "re-registering existing interest consumed a serial");

  ContentVersionTracker split_baseline;
  Check(split_baseline.EnsureTracked({0x8080, 0x40}) &&
            split_baseline.EnsureTracked({0x8000, 0x100}),
        "overlapping late registration failed");
  const auto split = split_baseline.Query({0x8000, 0x100});
  Check(split.size() == 3 && split[0].versions.cpu == 2 &&
            split[1].versions.cpu == 1 && split[2].versions.cpu == 2,
        "one late registration did not seed all new spans with one serial");
}

void TestSharedOperationSerialAndLockedPlan() {
  ContentVersionTracker tracker;
  Check(tracker.EnsureTracked({0x9000, 0x100}) &&
            tracker.EnsureTracked({0x9200, 0x100}),
        "multi-output tracked-range registration failed");
  const auto operation = tracker.ReserveSerial();
  Check(
      operation == 3 &&
          tracker.StampAt(ContentDomain::Buffer, {0x9000, 0x100}, operation,
                          "output-a") &&
          tracker.StampAt(ContentDomain::Buffer, {0x9200, 0x100}, operation,
                          "output-b") &&
          tracker.TraceImageAt({0x9000, 0x100}, operation, "image-output", 7) &&
          !tracker.TraceImageAt({0x9100, 0x100}, operation, "outside-image", 8),
      "one operation serial was not accepted for multiple outputs");
  const auto first = tracker.Query({0x9000, 0x100});
  const auto second = tracker.Query({0x9200, 0x100});
  Check(first.front().versions.buffer == operation &&
            second.front().versions.buffer == operation &&
            tracker.ReserveSerial() == operation + 1,
        "multi-output stamps did not share one stable serial");

  const auto plan = tracker.BuildPlan({0x9000, 0x100});
  Check(plan.complete && plan.spans.size() == 1 &&
            plan.spans.front().kind == ContentSourceKind::Buffer &&
            plan.spans.front().serial == operation,
        "tracker-owned read plan did not observe the stamped output");
}

void TestPartialTrackedIntersections() {
  ContentVersionTracker tracker;
  Check(tracker.EnsureTracked({0xa000, 0x100}) &&
            tracker.EnsureTracked({0xa200, 0x100}),
        "split tracked ranges were not registered");
  const auto serial =
      tracker.Stamp(ContentDomain::Buffer, {0x9f80, 0x400}, "wide-write");
  Check(serial == 4,
        "wide Buffer write did not allocate baseline and write serials");
  const auto query = tracker.Query({0x9f80, 0x400});
  Check(query.size() == 5 && query[0].versions.cpu == 3 &&
            query[0].versions.buffer == serial &&
            query[1].range == ContentRange{0xa000, 0x100} &&
            query[1].versions.buffer == serial && query[2].versions.cpu == 3 &&
            query[2].versions.buffer == serial &&
            query[3].range == ContentRange{0xa200, 0x100} &&
            query[3].versions.buffer == serial && query[4].versions.cpu == 3 &&
            query[4].versions.buffer == serial,
        "Buffer write did not establish tracking across every gap");
}

void TestUnmapErasesInterestAndVersions() {
  ContentVersionTracker tracker;
  Check(tracker.EnsureTracked({0xb000, 0x300}),
        "unmap test registration failed");
  Check(tracker.Stamp(ContentDomain::Buffer, {0xb000, 0x300}, "initial") == 2,
        "unmap test initial stamp failed");
  Check(tracker.Erase({0xb100, 0x100}), "tracked unmap failed");
  const auto after_unmap = tracker.Query({0xb000, 0x300});
  Check(after_unmap.size() == 3 &&
            after_unmap[1].range == ContentRange{0xb100, 0x100} &&
            after_unmap[1].versions == Libs::Graphics::ContentVersions{},
        "unmap retained content versions");

  const auto write =
      tracker.Stamp(ContentDomain::Cpu, {0xb000, 0x300}, "post-unmap");
  const auto clipped = tracker.Query({0xb000, 0x300});
  Check(write == 3 && clipped.size() == 3 && clipped[0].versions.cpu == write &&
            clipped[1].versions == Libs::Graphics::ContentVersions{} &&
            clipped[2].versions.cpu == write,
        "CPU write restored interest in an unmapped interval");

  Check(tracker.EnsureTracked({0xb100, 0x100}),
        "re-registering unmapped interval failed");
  const auto registered = tracker.Query({0xb000, 0x300});
  Check(registered.size() == 3 && registered[1].versions.cpu == 4 &&
            registered[1].versions.buffer == 0,
        "re-registering did not establish a fresh CPU baseline");
}

void TestBufferWriteBeforeImageRegistration() {
  ContentVersionTracker tracker;
  constexpr ContentRange write{0xd080, 0x40};
  constexpr ContentRange image{0xd000, 0x100};

  const auto buffer =
      tracker.Stamp(ContentDomain::Buffer, write, "dma-before-image");
  const auto before_registration = tracker.Query(write);
  Check(buffer == 2 && before_registration.size() == 1 &&
            before_registration.front().versions.cpu == 1 &&
            before_registration.front().versions.buffer == buffer,
        "Buffer write before image registration was not tracked");

  Check(tracker.EnsureTracked(image), "late image registration failed");
  const auto registered = tracker.Query(image);
  Check(registered.size() == 3 && registered[0].versions.cpu == 3 &&
            registered[0].versions.buffer == 0 &&
            registered[1].range == write && registered[1].versions.cpu == 1 &&
            registered[1].versions.buffer == buffer &&
            registered[2].versions.cpu == 3 &&
            registered[2].versions.buffer == 0,
        "late image registration lost the earlier Buffer write");

  const auto plan = tracker.BuildPlan(image);
  Check(plan.complete &&
            SourceAt(plan, image.address).kind == ContentSourceKind::Cpu &&
            SourceAt(plan, write.address).kind == ContentSourceKind::Buffer &&
            SourceAt(plan, write.End()).kind == ContentSourceKind::Cpu,
        "late image registration selected the wrong backing source");
}

void TestOlderReservedSerialDoesNotDowngrade() {
  ContentVersionTracker tracker;
  constexpr ContentRange range{0xe000, 0x100};
  Check(tracker.EnsureTracked(range), "monotonic stamp registration failed");

  const auto older_buffer = tracker.ReserveSerial();
  const auto newer_buffer =
      tracker.Stamp(ContentDomain::Buffer, range, "newer-buffer");
  Check(older_buffer == 2 && newer_buffer == 3 &&
            tracker.StampAt(ContentDomain::Buffer, range, older_buffer,
                            "late-older-buffer") &&
            tracker.Query(range).front().versions.buffer == newer_buffer,
        "older reserved serial downgraded the Buffer domain");

  const auto older_cpu = tracker.ReserveSerial();
  const auto newer_cpu = tracker.Stamp(ContentDomain::Cpu, range, "newer-cpu");
  Check(older_cpu == 4 && newer_cpu == 5 &&
            tracker.StampAt(ContentDomain::Cpu, range, older_cpu,
                            "late-older-cpu") &&
            tracker.Query(range).front().versions.cpu == newer_cpu,
        "older reserved serial downgraded the CPU domain");
}

void TestTrackedInvalidRanges() {
  ContentVersionTracker tracker;
  Check(!tracker.EnsureTracked({0, 1}) &&
            !tracker.EnsureTracked({UINT64_MAX - 7, 8}) &&
            tracker.StampImage({0, 1}, "invalid", 9) == 0 &&
            !tracker.StampAt(ContentDomain::Cpu, {0xc000, 1}, 0, "invalid"),
        "tracker accepted an invalid range or serial");
}

} // namespace

void RunContentVersionMapTests() {
  TestSplitAndCoalesce();
  TestGapsAndInvalidRanges();
  TestTraceShapeAndStableSelection();
  TestSecondFrameAndTieSelection();
  TestTrackedInterestAndLateRegistration();
  TestSharedOperationSerialAndLockedPlan();
  TestPartialTrackedIntersections();
  TestUnmapErasesInterestAndVersions();
  TestBufferWriteBeforeImageRegistration();
  TestOlderReservedSerialDoesNotDowngrade();
  TestTrackedInvalidRanges();
}
