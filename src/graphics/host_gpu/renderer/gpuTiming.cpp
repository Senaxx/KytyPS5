#include "graphics/host_gpu/renderer/gpuTiming.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::GpuTiming {

namespace {

constexpr uint32_t BlockQueries = 2048;
constexpr uint32_t BlockCount   = 32;

struct Pair {
	uint32_t before = 0;
	uint32_t after  = 0;
	uint64_t key    = 0;
	uint32_t kind   = 0;
};

struct Block {
	uint64_t          tick = 0;
	uint32_t          used = 0;
	int64_t           open = -1; // query index of a Before without its After yet
	std::vector<Pair> pairs;
	std::atomic<bool> in_flight {false};
};

struct Stat {
	double   ms    = 0.0;
	uint64_t count = 0;
	uint32_t kind  = 0;
};

struct State {
	vk::Device                           device;
	vk::QueryPool                        pool;
	double                               period_ns = 1.0;
	std::array<Block, BlockCount>        blocks;
	uint32_t                             next_block = 0;
	Block*                               current    = nullptr;
	std::mutex                           stats_mutex;
	std::unordered_map<uint64_t, Stat>   stats; // key 0: GPU time between timed commands
	double                               total_ms = 0.0;
};

State& GetState() {
	static State state;
	return state;
}

struct Window {
	uint64_t first = 0;
	uint64_t count = 0;
};

const Window& GetWindow() {
	static const Window window = [] {
		Window result;
		const char* value = std::getenv("KYTY_GPU_TIMING");
		if (value != nullptr) {
			char* end    = nullptr;
			result.first = std::strtoull(value, &end, 10);
			result.count = end != nullptr && *end == ':' ? std::strtoull(end + 1, nullptr, 10) : 1;
		}
		return result;
	}();
	return window;
}

void ReadBlock(State& state, Block& block) {
	if (block.used != 0) {
		std::vector<uint64_t> values(block.used);
		const auto result = state.device.getQueryPoolResults(
		    state.pool, static_cast<uint32_t>(&block - state.blocks.data()) * BlockQueries,
		    block.used, values.size() * sizeof(uint64_t), values.data(), sizeof(uint64_t),
		    vk::QueryResultFlagBits::e64);
		if (result == vk::Result::eSuccess) {
			std::lock_guard lock(state.stats_mutex);
			uint64_t        previous_after = 0;
			for (const auto& pair: block.pairs) {
				const auto before = values[pair.before];
				const auto after  = values[pair.after];
				if (after >= before) {
					auto& stat = state.stats[pair.key];
					stat.ms += static_cast<double>(after - before) * state.period_ns / 1e6;
					stat.count++;
					stat.kind = pair.kind;
				}
				if (previous_after != 0 && before >= previous_after) {
					auto& gap = state.stats[0];
					gap.ms += static_cast<double>(before - previous_after) * state.period_ns / 1e6;
					gap.count++;
				}
				previous_after = after;
			}
			if (!block.pairs.empty()) {
				const auto first = values[block.pairs.front().before];
				const auto last  = values[block.pairs.back().after];
				if (last >= first) {
					state.total_ms += static_cast<double>(last - first) * state.period_ns / 1e6;
				}
			}
		}
	}
	block.pairs.clear();
	block.used = 0;
	block.open = -1;
	block.in_flight.store(false, std::memory_order_release);
}

// The block for the recording in progress, or null when none is free.
Block* CurrentBlock(RenderContext& context, CommandBuffer& buffer) {
	auto&      state     = GetState();
	auto&      scheduler = context.GetCommandScheduler();
	const auto tick      = scheduler.CurrentTick();
	if (state.current != nullptr && state.current->tick == tick &&
	    state.current->used + 2 <= BlockQueries) {
		return state.current;
	}
	if (!state.pool) {
		const auto& graphics = context.GetGraphics();
		state.device         = graphics.device;
		state.period_ns      = graphics.physical_device_properties.limits.timestampPeriod;
		vk::QueryPoolCreateInfo info {};
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = BlockQueries * BlockCount;
		if (state.device.createQueryPool(&info, nullptr, &state.pool) != vk::Result::eSuccess) {
			Detail::g_active.store(false);
			return nullptr;
		}
	}
	auto& block = state.blocks[state.next_block % BlockCount];
	if (block.in_flight.load(std::memory_order_acquire)) {
		state.current = nullptr;
		return nullptr;
	}
	state.next_block++;
	block.tick = tick;
	block.used = 0;
	block.open = -1;
	block.pairs.clear();
	block.in_flight.store(true, std::memory_order_release);
	buffer.EndRendering();
	buffer.Handle().resetQueryPool(state.pool,
	                               static_cast<uint32_t>(&block - state.blocks.data()) * BlockQueries,
	                               BlockQueries);
	// Runs once this tick has executed, so every query of the block is written.
	scheduler.DeferPriorityOperation([&state, &block] { ReadBlock(state, block); });
	state.current = &block;
	return &block;
}

} // namespace

namespace Detail {

void BeforeSlow(RenderContext& context, CommandBuffer& buffer) {
	auto* block = CurrentBlock(context, buffer);
	if (block == nullptr) {
		return;
	}
	auto&      state = GetState();
	const auto base  = static_cast<uint32_t>(block - state.blocks.data()) * BlockQueries;
	buffer.Handle().writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, state.pool,
	                               base + block->used);
	block->open = block->used++;
}

void AfterSlow(RenderContext& context, CommandBuffer& buffer, uint64_t key, uint32_t kind) {
	auto& state = GetState();
	auto* block = state.current;
	if (block == nullptr || block->open < 0 ||
	    block->tick != context.GetCommandScheduler().CurrentTick() ||
	    block->used >= BlockQueries) {
		return;
	}
	const auto base = static_cast<uint32_t>(block - state.blocks.data()) * BlockQueries;
	buffer.Handle().writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, state.pool,
	                               base + block->used);
	block->pairs.push_back({static_cast<uint32_t>(block->open), block->used, key, kind});
	block->used++;
	block->open = -1;
}

} // namespace Detail

void Present(uint64_t frame) {
	const auto& window = GetWindow();
	if (window.count == 0) {
		return;
	}
	auto& state = GetState();
	if (Detail::g_active.load(std::memory_order_relaxed)) {
		std::vector<std::pair<uint64_t, Stat>> sorted;
		double                                 timed = 0.0;
		double                                 total = 0.0;
		{
			std::lock_guard lock(state.stats_mutex);
			sorted.assign(state.stats.begin(), state.stats.end());
			state.stats.clear();
			total          = state.total_ms;
			state.total_ms = 0.0;
		}
		std::sort(sorted.begin(), sorted.end(),
		          [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
		for (const auto& [key, stat]: sorted) {
			if (key != 0) {
				timed += stat.ms;
			}
		}
		LOGF("GpuTiming frame %" PRIu64 ": span=%.1fms commands=%.1fms\n", frame, total, timed);
		for (size_t i = 0; i < sorted.size() && i < 20; i++) {
			const auto& [key, stat] = sorted[i];
			LOGF("GpuTiming   %s 0x%016" PRIx64 " %7.2fms x%" PRIu64 "\n",
			     key == 0 ? "between " : (stat.kind == Draw ? "draw    " : "dispatch"), key,
			     stat.ms, stat.count);
		}
	}
	const bool active = frame >= window.first && frame - window.first < window.count;
	Detail::g_active.store(active, std::memory_order_relaxed);
}

} // namespace Libs::Graphics::GpuTiming
