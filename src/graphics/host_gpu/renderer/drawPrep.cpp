#include "graphics/host_gpu/renderer/drawPrep.h"

#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/pm4.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>

namespace Libs::Graphics::DrawPrep {

namespace {

// Results waiting for the GPU thread; the scanner waits while there are more.
constexpr size_t MaxPending = 8192;

// Results the GPU thread is done with go back to the scanner with their buffers, so that neither
// thread frees what the other allocated (slow across threads on the Windows heap).
constexpr size_t MaxRecycled = 4096;

struct Store {
	std::mutex                mutex;
	std::condition_variable   space;
	std::array<std::deque<PacketResult>, MaxQueues> pending; // per queue
	std::vector<PacketResult> recycled;
};

Store& GetStore() {
	static Store store;
	return store;
}

thread_local Key  t_current {};
thread_local bool t_has_current = false;
// GPU thread progress per queue, packed: submission (40 bits), ordinal (24 bits).
std::array<std::atomic<uint64_t>, MaxQueues> g_progress {};

std::array<std::atomic<uint64_t>, static_cast<size_t>(Counter::Count)> g_counters {};

struct GoodPairs {
	std::shared_mutex            mutex;
	std::unordered_set<uint64_t> pairs;
};

GoodPairs& GetGoodPairs() {
	static GoodPairs pairs;
	return pairs;
}

uint64_t PairKey(uint64_t vs_addr, uint64_t ps_addr) {
	// Shader addresses are 256-byte aligned guest addresses below 2^48.
	return (vs_addr >> 8u) * 0x9e3779b97f4a7c15ull ^ (ps_addr >> 8u);
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		// On by default since 2026-10-09 (verify: 0 mismatches over the new games of 2026-10-08
		// and -09; +9 % at the Leap Attack prompt, +13 % in the fight); KYTY_DRAW_PREP=0 turns it off.
		const char* value = std::getenv("KYTY_DRAW_PREP");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

int VerifyMode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_DRAW_PREP_VERIFY");
		if (value == nullptr || value[0] == '\0' || value[0] == '0') {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

bool CountsAsDrawOrDispatch(uint32_t opcode) {
	switch (opcode) {
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DISPATCH_DRAW_PREAMBLE:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
		case Pm4::IT_DISPATCH_DRAW:
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT: return true;
		default: return false;
	}
}

void ReadSet::Add(uint64_t address, const void* data, uint32_t size) {
	if (size == 0) {
		return;
	}
	const auto offset = static_cast<uint32_t>(m_bytes.size());
	m_bytes.resize(m_bytes.size() + size);
	std::memcpy(m_bytes.data() + offset, data, size);
	if (!m_ranges.empty()) {
		auto& last = m_ranges.back();
		if (last.address + last.size == address && last.offset + last.size == offset) {
			last.size += size;
			return;
		}
	}
	m_ranges.push_back({address, offset, size});
}

// Page by page: a page that is GPU-clean as a whole is compared straight from its backing bytes
// (one dirty-range query per page, as the walk's page cache does); otherwise the range's bytes are
// read with the exact clean read.
bool ReadSet::Validate() const {
	KYTY_PROFILER_BLOCK("DrawPrep::Validate");
	constexpr uint64_t PageSize = 4096;
	struct Page {
		uint64_t       page    = UINT64_MAX;
		const uint8_t* backing = nullptr;
	};
	std::array<Page, 8> pages {};
	size_t              next = 0;
	const auto map = [&](uint64_t page) {
		for (const auto& entry: pages) {
			if (entry.page == page) {
				return entry.backing;
			}
		}
		auto& slot = pages[next++ % pages.size()];
		slot       = {page, LibKernel::Memory::FindGpuCleanBacking(page, PageSize)};
		return slot.backing;
	};
	thread_local std::vector<uint8_t> current;
	for (const auto& range: m_ranges) {
		const uint8_t* expected = m_bytes.data() + range.offset;
		uint64_t       address  = range.address;
		uint64_t       left     = range.size;
		while (left != 0) {
			const uint64_t page    = address & ~(PageSize - 1u);
			const uint64_t in_page = std::min<uint64_t>(left, page + PageSize - address);
			if (const auto* backing = map(page); backing != nullptr) {
				if (std::memcmp(backing + (address - page), expected, in_page) != 0) {
					return false;
				}
			} else {
				if (current.size() < in_page) {
					current.resize(in_page);
				}
				if (!LibKernel::Memory::TryReadGpuCleanBacking(address, current.data(), in_page) ||
				    std::memcmp(current.data(), expected, in_page) != 0) {
					return false;
				}
			}
			address += in_page;
			expected += in_page;
			left -= in_page;
		}
	}
	return true;
}

void SetCurrent(const Key& key) {
	t_current     = key;
	t_has_current = true;
	g_progress[key.queue % MaxQueues].store((key.seq << 24u) | (key.ordinal & 0xffffffu),
	                                        std::memory_order_relaxed);
}

void ClearCurrent() {
	t_has_current = false;
}

const Key* Current() {
	return t_has_current ? &t_current : nullptr;
}

Key GpuProgress(uint32_t queue) {
	const auto packed = g_progress[queue % MaxQueues].load(std::memory_order_relaxed);
	return {.queue   = queue,
	        .seq     = packed >> 24u,
	        .ordinal = static_cast<uint32_t>(packed & 0xffffffu)};
}

void Publish(PacketResult&& result) {
	auto&            store = GetStore();
	std::unique_lock lock(store.mutex);
	// While the GPU thread is far behind, wait for it a little; a result that does not fit is
	// dropped (the GPU thread walks that draw itself), so the scanner never waits for good.
	auto& pending = store.pending[result.key.queue % MaxQueues];
	for (int i = 0; i < 100 && pending.size() >= MaxPending; i++) {
		store.space.wait_for(lock, std::chrono::milliseconds(1));
	}
	if (pending.size() >= MaxPending) {
		return;
	}
	pending.push_back(std::move(result));
}

void Recycle(PacketResult& result) {
	auto&           store = GetStore();
	std::lock_guard lock(store.mutex);
	if (!store.recycled.empty()) {
		result = std::move(store.recycled.back());
		store.recycled.pop_back();
	}
	result.key         = {};
	result.stage_count = 0;
	for (auto& stage: result.stages) {
		stage.ok       = false;
		stage.consumed = false;
	}
}

StageResult* Take(uint32_t stage) {
	KYTY_PROFILER_BLOCK("DrawPrep::Take");
	const auto* key = Current();
	Add(Counter::Takes);
	if (key == nullptr) {
		Add(Counter::MissNone);
		return nullptr;
	}
	auto&           store   = GetStore();
	std::lock_guard lock(store.mutex);
	auto&           pending = store.pending[key->queue % MaxQueues];
	bool            popped = false;
	while (!pending.empty() && pending.front().key.Before(*key)) {
		if (store.recycled.size() < MaxRecycled) {
			store.recycled.push_back(std::move(pending.front()));
		}
		pending.pop_front();
		popped = true;
	}
	if (popped) {
		store.space.notify_one();
	}
	if (pending.empty()) {
		Add(Counter::MissNone);
		return nullptr;
	}
	auto& front = pending.front();
	if (front.key.seq != key->seq || front.key.ordinal != key->ordinal ||
	    front.key.packet != key->packet) {
		Add(Counter::MissNone);
		return nullptr;
	}
	for (uint32_t i = 0; i < front.stage_count; i++) {
		auto& result = front.stages[i];
		if (result.ok && result.stage == stage) {
			// The deque keeps references stable across the scanner's push_back; only this thread
			// pops, so the result stays valid until the next Take.
			return &result;
		}
	}
	Add(Counter::MissNone);
	return nullptr;
}

void Add(Counter counter, uint64_t value) {
	g_counters[static_cast<size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
}

void LogCountersIfDue() {
	static std::chrono::steady_clock::time_point last {};
	static std::array<uint64_t, static_cast<size_t>(Counter::Count)> previous {};
	const auto now = std::chrono::steady_clock::now();
	if (last.time_since_epoch().count() == 0) {
		last = now;
		return;
	}
	if (now - last < std::chrono::seconds(10)) {
		return;
	}
	last = now;
	std::array<uint64_t, static_cast<size_t>(Counter::Count)> delta {};
	for (size_t i = 0; i < delta.size(); i++) {
		const auto value = g_counters[i].load(std::memory_order_relaxed);
		delta[i]         = value - previous[i];
		previous[i]      = value;
	}
	const auto c = [&](Counter counter) { return static_cast<unsigned long long>(delta[static_cast<size_t>(counter)]); };
	LOGF("DrawPrep (10 s): scanned %llu, prepared %llu stages, skipped behind %llu / unknown pair "
	     "%llu / unsupported %llu / no entry %llu, walk failed %llu, tainted %llu (indirect %llu, "
	     "context %llu); GPU takes %llu, "
	     "used %llu (again %llu), miss none %llu / entry %llu / inputs %llu / reads %llu; verify "
	     "%llu checks, "
	     "%llu mismatches\n",
	     c(Counter::Scanned), c(Counter::Prepared), c(Counter::SkipBehind),
	     c(Counter::SkipUnknownPair), c(Counter::SkipUnsupported), c(Counter::SkipNoEntry),
	     c(Counter::FailWalk), c(Counter::Tainted), c(Counter::TaintIndirect),
	     c(Counter::TaintContext), c(Counter::Takes), c(Counter::Used) + c(Counter::Reused),
	     c(Counter::Reused), c(Counter::MissNone), c(Counter::MissEntry), c(Counter::MissInputs),
	     c(Counter::MissReads),
	     c(Counter::VerifyChecks), c(Counter::VerifyMismatches));
}

void NoteGoodPair(uint64_t vs_addr, uint64_t ps_addr) {
	thread_local std::unordered_set<uint64_t> known;
	const auto                                key = PairKey(vs_addr, ps_addr);
	if (!known.insert(key).second) {
		return;
	}
	auto&           pairs = GetGoodPairs();
	std::unique_lock lock(pairs.mutex);
	pairs.pairs.insert(key);
}

bool IsGoodPair(uint64_t vs_addr, uint64_t ps_addr) {
	auto&            pairs = GetGoodPairs();
	std::shared_lock lock(pairs.mutex);
	return pairs.pairs.contains(PairKey(vs_addr, ps_addr));
}

} // namespace Libs::Graphics::DrawPrep
