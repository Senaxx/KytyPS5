#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "common/timer.h"
#include "graphics/host_gpu/graphicContext.h"

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

// Every blocking wait on the host GPU goes through here. Per-thread totals every 256 waits,
// and a host stack for the slow ones, name the paths that serialize the CPU on the GPU.
namespace {
struct SemaphoreWaitStats {
	uint64_t count   = 0;
	double   total_s = 0.0;
	double   max_s   = 0.0;
};
thread_local SemaphoreWaitStats g_semaphore_wait_stats;
std::atomic<uint32_t>           g_slow_wait_traces = 0;

void RecordSemaphoreWait(double seconds, uint64_t tick, uint64_t gpu_tick) {
	auto& stats = g_semaphore_wait_stats;
	stats.count++;
	stats.total_s += seconds;
	stats.max_s = std::max(stats.max_s, seconds);
	if (seconds >= 0.020 && g_slow_wait_traces.fetch_add(1) < 24) {
		LOGF("SemaphoreWait slow: %.1fms tick=%" PRIu64 " gpu_tick=%" PRIu64 " thread=%d\n%s",
		     seconds * 1000.0, tick, gpu_tick, Common::Thread::GetThreadIdUnique(),
		     Common::HostBacktrace().c_str());
	}
	if (stats.count % 256 == 0) {
		LOGF("SemaphoreWaits: thread=%d count=%" PRIu64 " total=%.1fms max=%.1fms\n",
		     Common::Thread::GetThreadIdUnique(), stats.count, stats.total_s * 1000.0,
		     stats.max_s * 1000.0);
		stats = {};
	}
}
} // namespace

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	// Profiling-only zone: this is the actual CPU-blocks-on-GPU stall. Everything above this
	// point is a fast non-blocking check; only reaching here means the GPU genuinely hasn't
	// caught up yet.
	KYTY_PROFILER_BLOCK("MasterSemaphore::Wait (blocked on GPU)");

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	Common::Timer timer;
	timer.Start();
	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	RecordSemaphoreWait(timer.GetTimeS(), tick, m_gpu_tick.load(std::memory_order_acquire));
	if (result != vk::Result::eSuccess) {
		if (result == vk::Result::eErrorDeviceLost) {
			DumpDeviceLossDiagnostics(m_graphics);
		}
		EXIT("MasterSemaphore: wait for tick %" PRIu64 " failed: %s (gpu tick %" PRIu64 ")\n", tick,
		     vk::to_string(result).c_str(), m_gpu_tick.load(std::memory_order_acquire));
	}
	Refresh();
}

} // namespace Libs::Graphics
