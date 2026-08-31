#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_CONTENTVERSIONMAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_CONTENTVERSIONMAP_H_

#include <compare>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <span>
#include <string_view>
#include <vector>

namespace Libs::Graphics {

using ContentSerial = uint64_t;

struct ContentRange {
	uint64_t address = 0;
	uint64_t size    = 0;

	[[nodiscard]] constexpr bool Valid() const noexcept {
		return address != 0 && size != 0 && size <= UINT64_MAX - address;
	}
	[[nodiscard]] constexpr uint64_t End() const noexcept { return address + size; }
	auto                             operator<=>(const ContentRange&) const = default;
};

enum class ContentDomain : uint8_t { Cpu, Buffer };

struct ContentVersions {
	ContentSerial cpu                                       = 0;
	ContentSerial buffer                                    = 0;
	auto          operator<=>(const ContentVersions&) const = default;
};

struct ContentVersionSlice {
	ContentRange    range;
	ContentVersions versions;
	auto            operator<=>(const ContentVersionSlice&) const = default;
};

class ContentVersionMap final {
public:
	[[nodiscard]] bool NextSerial(ContentSerial& serial) noexcept;
	[[nodiscard]] bool Stamp(ContentDomain domain, ContentRange range, ContentSerial serial);
	[[nodiscard]] bool Erase(ContentRange range);

	[[nodiscard]] std::vector<ContentVersionSlice> Query(ContentRange range) const;
	[[nodiscard]] size_t SegmentCount() const noexcept { return m_segments.size(); }

private:
	struct Segment {
		uint64_t        end = 0;
		ContentVersions versions;
	};

	void SplitAt(uint64_t address);
	void Coalesce();

	std::map<uint64_t, Segment> m_segments;
	ContentSerial               m_last_serial = 0;
};

enum class ContentSourceKind : uint8_t { None, Cpu, Buffer, Image };

struct ContentImageCandidate {
	ContentRange  range;
	ContentSerial serial                                          = 0;
	uint64_t      source_id                                       = 0;
	auto          operator<=>(const ContentImageCandidate&) const = default;
};

struct ContentSourceSpan {
	ContentRange      range;
	ContentSourceKind kind                                        = ContentSourceKind::None;
	ContentSerial     serial                                      = 0;
	uint64_t          source_id                                   = 0;
	auto              operator<=>(const ContentSourceSpan&) const = default;
};

struct ContentReadPlan {
	bool                           valid_range = false;
	bool                           complete    = false;
	std::vector<ContentSourceSpan> spans;
};

[[nodiscard]] ContentReadPlan
BuildContentReadPlan(const ContentVersionMap& versions, ContentRange range,
                     std::span<const ContentImageCandidate> images = {});

class ContentVersionTracker final {
public:
	explicit ContentVersionTracker(bool force_enabled = false);
	~ContentVersionTracker();
	ContentVersionTracker(const ContentVersionTracker&)            = delete;
	ContentVersionTracker& operator=(const ContentVersionTracker&) = delete;

	[[nodiscard]] bool          EnsureTracked(ContentRange range);
	[[nodiscard]] ContentSerial ReserveSerial();
	[[nodiscard]] bool StampAt(ContentDomain domain, ContentRange range, ContentSerial serial,
	                           std::string_view event, uint64_t source_id = 0);
	[[nodiscard]] ContentSerial Stamp(ContentDomain domain, ContentRange range,
	                                  std::string_view event, uint64_t source_id = 0);
	[[nodiscard]] bool          TraceImageAt(ContentRange range, ContentSerial serial,
	                                         std::string_view event, uint64_t source_id);
	[[nodiscard]] ContentSerial StampImage(ContentRange range, std::string_view event,
	                                       uint64_t source_id);
	[[nodiscard]] bool          Erase(ContentRange range);
	[[nodiscard]] std::vector<ContentVersionSlice> Query(ContentRange range) const;
	[[nodiscard]] ContentReadPlan
	BuildPlan(ContentRange range, std::span<const ContentImageCandidate> images = {}) const;
	[[nodiscard]] bool Enabled() const noexcept { return true; }
	[[nodiscard]] bool TraceEnabled() const noexcept { return m_trace != nullptr; }

private:
	[[nodiscard]] ContentSerial ReserveSerialLocked();
	[[nodiscard]] bool EnsureTrackedAtLocked(ContentRange range, ContentSerial baseline_serial);
	[[nodiscard]] bool StampAtLocked(const ContentDomain* domain, ContentRange range,
	                                 ContentSerial serial, std::string_view event,
	                                 uint64_t source_id);
	[[nodiscard]] std::vector<ContentRange> TrackedIntersectionsLocked(ContentRange range) const;
	[[nodiscard]] std::vector<ContentRange> UntrackedRangesLocked(ContentRange range) const;
	void                                    AddTrackedLocked(ContentRange range);
	void                                    SubtractTrackedLocked(ContentRange range);
	void TraceLocked(const ContentDomain* domain, ContentRange range, ContentSerial serial,
	                 std::string_view event, uint64_t source_id);

	mutable std::mutex           m_lock;
	ContentVersionMap            m_versions;
	std::map<uint64_t, uint64_t> m_tracked_ranges;
	void*                        m_trace    = nullptr;
	uint64_t                     m_sequence = 0;
	ContentRange                 m_trace_filter;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_CONTENTVERSIONMAP_H_
