#include "graphics/host_gpu/contentVersionMap.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>

namespace Libs::Graphics {

bool ContentVersionMap::NextSerial(ContentSerial& serial) noexcept {
	if (m_last_serial == std::numeric_limits<ContentSerial>::max()) {
		return false;
	}
	serial = ++m_last_serial;
	return true;
}

namespace {

bool ParseEnvironmentNumber(const char* name, uint64_t& result) {
	const auto* value = std::getenv(name);
	if (value == nullptr || *value == '\0') {
		return false;
	}
	errno          = 0;
	char*      end = nullptr;
	const auto raw = std::strtoull(value, &end, 0);
	if (errno != 0 || end == value || *end != '\0') {
		return false;
	}
	result = raw;
	return true;
}

} // namespace

ContentVersionTracker::ContentVersionTracker(bool force_enabled) {
	(void)force_enabled;
	const auto* path = std::getenv("KYTY_CONTENT_VERSION_LOG");
	if (path != nullptr && *path != '\0' && std::filesystem::path(path).is_absolute()) {
		m_trace = std::fopen(path, "wb");
	}
	uint64_t trace_address = 0;
	uint64_t trace_size    = 0;
	if (ParseEnvironmentNumber("KYTY_CONTENT_VERSION_TRACE_ADDRESS", trace_address) &&
	    ParseEnvironmentNumber("KYTY_CONTENT_VERSION_TRACE_SIZE", trace_size)) {
		const ContentRange filter {trace_address, trace_size};
		if (filter.Valid()) {
			m_trace_filter = filter;
		}
	}
}

ContentVersionTracker::~ContentVersionTracker() {
	if (m_trace != nullptr) {
		std::fclose(static_cast<std::FILE*>(m_trace));
	}
}

ContentSerial ContentVersionTracker::ReserveSerialLocked() {
	ContentSerial serial = 0;
	return m_versions.NextSerial(serial) ? serial : 0;
}

ContentSerial ContentVersionTracker::ReserveSerial() {
	std::scoped_lock lock {m_lock};
	return ReserveSerialLocked();
}

std::vector<ContentRange>
ContentVersionTracker::TrackedIntersectionsLocked(ContentRange range) const {
	std::vector<ContentRange> result;
	if (!range.Valid()) {
		return result;
	}
	auto current = m_tracked_ranges.upper_bound(range.address);
	if (current != m_tracked_ranges.begin()) {
		--current;
		if (current->second <= range.address) {
			++current;
		}
	}
	for (; current != m_tracked_ranges.end() && current->first < range.End(); ++current) {
		const auto begin = std::max(range.address, current->first);
		const auto end   = std::min(range.End(), current->second);
		if (begin < end) {
			result.push_back({begin, end - begin});
		}
	}
	return result;
}

std::vector<ContentRange> ContentVersionTracker::UntrackedRangesLocked(ContentRange range) const {
	std::vector<ContentRange> result;
	if (!range.Valid()) {
		return result;
	}
	uint64_t cursor = range.address;
	for (const auto tracked: TrackedIntersectionsLocked(range)) {
		if (cursor < tracked.address) {
			result.push_back({cursor, tracked.address - cursor});
		}
		cursor = tracked.End();
	}
	if (cursor < range.End()) {
		result.push_back({cursor, range.End() - cursor});
	}
	return result;
}

void ContentVersionTracker::AddTrackedLocked(ContentRange range) {
	auto current = m_tracked_ranges.lower_bound(range.address);
	if (current != m_tracked_ranges.begin() && std::prev(current)->second >= range.address) {
		current = std::prev(current);
	}
	uint64_t begin = range.address;
	uint64_t end   = range.End();
	while (current != m_tracked_ranges.end() && current->first <= end) {
		begin   = std::min(begin, current->first);
		end     = std::max(end, current->second);
		current = m_tracked_ranges.erase(current);
	}
	m_tracked_ranges.emplace(begin, end);
}

void ContentVersionTracker::SubtractTrackedLocked(ContentRange range) {
	auto current = m_tracked_ranges.lower_bound(range.address);
	if (current != m_tracked_ranges.begin() && std::prev(current)->second > range.address) {
		current = std::prev(current);
	}
	while (current != m_tracked_ranges.end() && current->first < range.End()) {
		const auto begin = current->first;
		const auto end   = current->second;
		current          = m_tracked_ranges.erase(current);
		if (begin < range.address) {
			m_tracked_ranges.emplace(begin, range.address);
		}
		if (end > range.End()) {
			m_tracked_ranges.emplace(range.End(), end);
			break;
		}
	}
}

void ContentVersionTracker::TraceLocked(const ContentDomain* domain, ContentRange range,
                                        ContentSerial serial, std::string_view event,
                                        uint64_t source_id) {
	if (m_trace == nullptr || (m_trace_filter.Valid() && (range.End() <= m_trace_filter.address ||
	                                                      m_trace_filter.End() <= range.address))) {
		return;
	}
	const char* domain_name = domain == nullptr               ? "image"
	                          : *domain == ContentDomain::Cpu ? "cpu"
	                                                          : "buffer";
	auto*       file        = static_cast<std::FILE*>(m_trace);
	std::fprintf(file,
	             "seq=%" PRIu64 " serial=%" PRIu64 " domain=%s event=%.*s "
	             "addr=0x%016" PRIx64 " size=0x%016" PRIx64 " source=0x%016" PRIx64 "\n",
	             m_sequence++, serial, domain_name, static_cast<int>(event.size()), event.data(),
	             range.address, range.size, source_id);
	std::fflush(file);
}

bool ContentVersionTracker::StampAtLocked(const ContentDomain* domain, ContentRange range,
                                          ContentSerial serial, std::string_view event,
                                          uint64_t source_id) {
	if (!range.Valid() || serial == 0) {
		return false;
	}
	const auto intersections = TrackedIntersectionsLocked(range);
	if (intersections.empty()) {
		return false;
	}
	for (const auto intersection: intersections) {
		if (domain != nullptr && !m_versions.Stamp(*domain, intersection, serial)) {
			return false;
		}
		TraceLocked(domain, intersection, serial, event, source_id);
	}
	return true;
}

bool ContentVersionTracker::EnsureTrackedAtLocked(ContentRange  range,
                                                  ContentSerial baseline_serial) {
	if (!range.Valid() || baseline_serial == 0) {
		return false;
	}
	const auto additions = UntrackedRangesLocked(range);
	for (const auto addition: additions) {
		if (!m_versions.Stamp(ContentDomain::Cpu, addition, baseline_serial)) {
			return false;
		}
		const auto domain = ContentDomain::Cpu;
		TraceLocked(&domain, addition, baseline_serial, "track-init", 0);
	}
	AddTrackedLocked(range);
	return true;
}

bool ContentVersionTracker::EnsureTracked(ContentRange range) {
	if (!range.Valid()) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	const auto       additions = UntrackedRangesLocked(range);
	if (additions.empty()) {
		return true;
	}
	const auto serial = ReserveSerialLocked();
	if (serial == 0) {
		return false;
	}
	return EnsureTrackedAtLocked(range, serial);
}

bool ContentVersionTracker::StampAt(ContentDomain domain, ContentRange range, ContentSerial serial,
                                    std::string_view event, uint64_t source_id) {
	std::scoped_lock lock {m_lock};
	if (domain == ContentDomain::Buffer && !EnsureTrackedAtLocked(range, serial)) {
		return false;
	}
	return StampAtLocked(&domain, range, serial, event, source_id);
}

ContentSerial ContentVersionTracker::Stamp(ContentDomain domain, ContentRange range,
                                           std::string_view event, uint64_t source_id) {
	std::scoped_lock lock {m_lock};
	if (!range.Valid()) {
		return 0;
	}
	if (domain == ContentDomain::Buffer) {
		if (!UntrackedRangesLocked(range).empty()) {
			const auto baseline_serial = ReserveSerialLocked();
			if (baseline_serial == 0 || !EnsureTrackedAtLocked(range, baseline_serial)) {
				return 0;
			}
		}
	} else if (TrackedIntersectionsLocked(range).empty()) {
		return 0;
	}
	const auto serial = ReserveSerialLocked();
	if (serial == 0 || !StampAtLocked(&domain, range, serial, event, source_id)) {
		return 0;
	}
	return serial;
}

bool ContentVersionTracker::TraceImageAt(ContentRange range, ContentSerial serial,
                                         std::string_view event, uint64_t source_id) {
	std::scoped_lock lock {m_lock};
	return StampAtLocked(nullptr, range, serial, event, source_id);
}

ContentSerial ContentVersionTracker::StampImage(ContentRange range, std::string_view event,
                                                uint64_t source_id) {
	std::scoped_lock lock {m_lock};
	if (TrackedIntersectionsLocked(range).empty()) {
		return 0;
	}
	const auto serial = ReserveSerialLocked();
	return serial != 0 && StampAtLocked(nullptr, range, serial, event, source_id) ? serial : 0;
}

bool ContentVersionTracker::Erase(ContentRange range) {
	if (!range.Valid()) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	SubtractTrackedLocked(range);
	return m_versions.Erase(range);
}

std::vector<ContentVersionSlice> ContentVersionTracker::Query(ContentRange range) const {
	std::scoped_lock lock {m_lock};
	return m_versions.Query(range);
}

ContentReadPlan
ContentVersionTracker::BuildPlan(ContentRange                           range,
                                 std::span<const ContentImageCandidate> images) const {
	std::scoped_lock lock {m_lock};
	return BuildContentReadPlan(m_versions, range, images);
}

void ContentVersionMap::SplitAt(uint64_t address) {
	auto after = m_segments.upper_bound(address);
	if (after == m_segments.begin()) {
		return;
	}
	auto current = std::prev(after);
	if (address <= current->first || address >= current->second.end) {
		return;
	}
	const auto old_end      = current->second.end;
	const auto old_versions = current->second.versions;
	current->second.end     = address;
	m_segments.emplace(address, Segment {old_end, old_versions});
}

void ContentVersionMap::Coalesce() {
	for (auto current = m_segments.begin(); current != m_segments.end();) {
		auto next = std::next(current);
		if (next != m_segments.end() && current->second.end == next->first &&
		    current->second.versions == next->second.versions) {
			current->second.end = next->second.end;
			m_segments.erase(next);
			continue;
		}
		current = next;
	}
}

bool ContentVersionMap::Stamp(ContentDomain domain, ContentRange range, ContentSerial serial) {
	if (!range.Valid() || serial == 0) {
		return false;
	}
	m_last_serial = std::max(m_last_serial, serial);
	SplitAt(range.address);
	SplitAt(range.End());

	uint64_t cursor = range.address;
	while (cursor < range.End()) {
		auto current = m_segments.lower_bound(cursor);
		if (current == m_segments.end() || current->first != cursor) {
			const auto end =
			    current == m_segments.end() ? range.End() : std::min(range.End(), current->first);
			current = m_segments.emplace(cursor, Segment {end, {}}).first;
		}
		auto& version = domain == ContentDomain::Cpu ? current->second.versions.cpu
		                                             : current->second.versions.buffer;
		version       = std::max(version, serial);
		cursor        = current->second.end;
	}
	Coalesce();
	return true;
}

bool ContentVersionMap::Erase(ContentRange range) {
	if (!range.Valid()) {
		return false;
	}
	SplitAt(range.address);
	SplitAt(range.End());
	for (auto current = m_segments.lower_bound(range.address);
	     current != m_segments.end() && current->first < range.End();) {
		current = m_segments.erase(current);
	}
	Coalesce();
	return true;
}

std::vector<ContentVersionSlice> ContentVersionMap::Query(ContentRange range) const {
	std::vector<ContentVersionSlice> result;
	if (!range.Valid()) {
		return result;
	}

	uint64_t cursor  = range.address;
	auto     current = m_segments.upper_bound(cursor);
	if (current != m_segments.begin()) {
		--current;
		if (current->second.end <= cursor) {
			++current;
		}
	}
	while (cursor < range.End()) {
		if (current == m_segments.end() || current->first > cursor) {
			const auto end =
			    current == m_segments.end() ? range.End() : std::min(range.End(), current->first);
			result.push_back({{cursor, end - cursor}, {}});
			cursor = end;
			continue;
		}
		const auto end = std::min(range.End(), current->second.end);
		result.push_back({{cursor, end - cursor}, current->second.versions});
		cursor = end;
		++current;
	}
	return result;
}

namespace {

struct SourceChoice {
	ContentSourceKind kind      = ContentSourceKind::None;
	ContentSerial     serial    = 0;
	uint64_t          source_id = 0;
};

int SourcePriority(ContentSourceKind kind) {
	switch (kind) {
		case ContentSourceKind::Buffer: return 3;
		case ContentSourceKind::Image: return 2;
		case ContentSourceKind::Cpu: return 1;
		case ContentSourceKind::None: return 0;
	}
	return 0;
}

void ConsiderSource(SourceChoice& selected, ContentSourceKind kind, ContentSerial serial,
                    uint64_t source_id = 0) {
	if (serial == 0 || serial < selected.serial) {
		return;
	}
	if (serial > selected.serial || SourcePriority(kind) > SourcePriority(selected.kind) ||
	    (serial == selected.serial && kind == selected.kind && kind == ContentSourceKind::Image &&
	     source_id < selected.source_id)) {
		selected = {kind, serial, source_id};
	}
}

} // namespace

ContentReadPlan BuildContentReadPlan(const ContentVersionMap& versions, ContentRange range,
                                     std::span<const ContentImageCandidate> images) {
	ContentReadPlan result;
	if (!range.Valid()) {
		return result;
	}
	result.valid_range = true;

	const auto            slices = versions.Query(range);
	std::vector<uint64_t> boundaries {range.address, range.End()};
	boundaries.reserve(2 + slices.size() * 2 + images.size() * 2);
	for (const auto& slice: slices) {
		boundaries.push_back(slice.range.address);
		boundaries.push_back(slice.range.End());
	}
	for (const auto& image: images) {
		if (image.serial == 0) {
			continue;
		}
		if (!image.range.Valid()) {
			result.valid_range = false;
			result.spans.clear();
			return result;
		}
		const auto begin = std::max(range.address, image.range.address);
		const auto end   = std::min(range.End(), image.range.End());
		if (begin < end) {
			boundaries.push_back(begin);
			boundaries.push_back(end);
		}
	}
	std::ranges::sort(boundaries);
	boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());

	result.complete  = true;
	size_t slice_idx = 0;
	for (size_t index = 0; index + 1 < boundaries.size(); index++) {
		const auto begin = boundaries[index];
		const auto end   = boundaries[index + 1];
		if (begin == end) {
			continue;
		}
		while (slice_idx + 1 < slices.size() && begin >= slices[slice_idx].range.End()) {
			++slice_idx;
		}
		SourceChoice selected;
		if (slice_idx < slices.size() && slices[slice_idx].range.address <= begin &&
		    slices[slice_idx].range.End() >= end) {
			ConsiderSource(selected, ContentSourceKind::Cpu, slices[slice_idx].versions.cpu);
			ConsiderSource(selected, ContentSourceKind::Buffer, slices[slice_idx].versions.buffer);
		}
		for (const auto& image: images) {
			if (image.serial != 0 && image.range.address <= begin && image.range.End() >= end) {
				ConsiderSource(selected, ContentSourceKind::Image, image.serial, image.source_id);
			}
		}
		ContentSourceSpan span {
		    {begin, end - begin}, selected.kind, selected.serial, selected.source_id};
		result.complete &= span.kind != ContentSourceKind::None;
		if (!result.spans.empty()) {
			auto& previous = result.spans.back();
			if (previous.range.End() == span.range.address && previous.kind == span.kind &&
			    previous.serial == span.serial && previous.source_id == span.source_id) {
				previous.range.size += span.range.size;
				continue;
			}
		}
		result.spans.push_back(span);
	}
	return result;
}

} // namespace Libs::Graphics
