#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/timer.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/timeline.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <optional>

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

namespace {

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
}

// KYTY_DRAW_FLUSH_INTERVAL=N overrides CompleteDraw()'s periodic non-blocking flush interval.
// Defaults to 16 -- validated against real gameplay, where it cut the fraction of the main thread
// spent in MasterSemaphore::Wait from dominating the frame to under 10%. Explicitly setting it to
// 0 disables the flush entirely, same as before this had a default.
uint32_t DrawFlushInterval() {
	static const uint32_t interval = [] {
		const char* v = std::getenv("KYTY_DRAW_FLUSH_INTERVAL");
		if (v == nullptr) {
			return 16u;
		}
		return static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
	}();
	return interval;
}

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {}

CommandScheduler::~CommandScheduler() {
	Shutdown();
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	// Every tick was waited on above, so the queue thread has nothing left to submit.
	StopSubmitThread();
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering() {
	if (Active() && !m_command.IsInvalid()) {
		Current().EndRendering();
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::CompleteReleaseMemWrite() {
	constexpr uint32_t WritesPerSubmission = 32;
	if (++m_recorded_release_mem_writes < WritesPerSubmission) {
		return;
	}
	CheckActive();
	Flush();
}

void CommandScheduler::CompleteReleaseMemInterrupt() {
	// Deliberately smaller than CompleteReleaseMemWrite's 32: this event has already been queued
	// for guest delivery once its tick completes (see Sync::TriggerEopEventAtEndOfPipe ->
	// DeferPriorityOperation), and a guest thread may be blocked waiting on it via an event queue.
	// Batching still defers only the vkQueueSubmit -- the event fires once that (now slightly
	// larger) submission's tick completes, same as before, just a handful of RELEASE_MEM events
	// later instead of immediately.
	constexpr uint32_t InterruptsPerSubmission = 8;
	if (++m_recorded_release_mem_interrupts < InterruptsPerSubmission) {
		return;
	}
	CheckActive();
	Flush();
}

void CommandScheduler::CompleteDraw() {
	const auto interval = DrawFlushInterval();
	if (interval == 0u || ++m_recorded_draws < interval) {
		return;
	}
	CheckActive();
	Flush();
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::FlushAndWait() {
	KYTY_PROFILER_FUNCTION();
	const auto tick = Submit();
	m_master.Wait(tick);
	BeginNext();
}

void CommandScheduler::Finish() {
	KYTY_PROFILER_FUNCTION();
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

// How long the GPU thread spends drained on the host GPU, and how often. A drain that
// submits the current tick first (a CPU read of GPU-owned memory mid-frame) is counted apart
// from one that waits on already submitted work.
namespace {
struct WaitStats {
	uint64_t count       = 0;
	uint64_t drain_count = 0;
	double   total_s     = 0.0;
	double   drain_s     = 0.0;
	double   max_s       = 0.0;
};
WaitStats g_wait_stats;

void RecordWait(double seconds, bool drained) {
	if (!GuestGpu::IsGpuThread()) {
		return;
	}
	auto& stats = g_wait_stats;
	stats.count++;
	stats.total_s += seconds;
	stats.max_s = std::max(stats.max_s, seconds);
	if (drained) {
		stats.drain_count++;
		stats.drain_s += seconds;
	}
	if (stats.count % 256 == 0) {
		LOGF("GpuWaits: count=%" PRIu64 " total=%.1fms max=%.1fms drains=%" PRIu64
		     " drain_total=%.1fms\n",
		     stats.count, stats.total_s * 1000.0, stats.max_s * 1000.0, stats.drain_count,
		     stats.drain_s * 1000.0);
		stats = {};
	}
}
} // namespace

void CommandScheduler::Wait(uint64_t tick) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(tick > CurrentTick());
	Common::Timer timer;
	timer.Start();
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		{
			KYTY_PROFILER_BLOCK("CommandScheduler::Wait (forced submit-then-wait)");
			const auto submitted_tick = Submit();
			EXIT_IF(submitted_tick != tick);
			Timeline::Mark("drain-begin", tick);
			m_master.Wait(tick);
			Timeline::Mark("drain-end", tick);
			BeginNext();
		}
		RecordWait(timer.GetTimeS(), true);
	} else {
		Timeline::Mark("wait-begin", tick);
		m_master.Wait(tick);
		Timeline::Mark("wait-end", tick);
		RecordWait(timer.GetTimeS(), false);
	}
}

void CommandScheduler::PopPendingOperations() {
	m_master.Refresh();
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	DeferRelease(std::move(operation));
}

void CommandScheduler::DeferRelease(Common::UniqueFunction<void>&& operation) {
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_pending_operations.push({std::move(operation), CurrentTick()});
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_priority_operations.push({std::move(operation), CurrentTick()});
		lock.unlock();
		m_operation_available.notify_one();
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active      = true;
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		Timeline::Mark("tick-done", operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active      = false;
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	CheckActive();
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	m_command.m_buffer = m_command_pool.Commit();
	m_command.Begin();
	return m_command;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(m_command.IsInvalid());
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);

	m_command.End();
	EXIT_IF(m_graphics.queue == nullptr);

	SubmitJob job {
	    .buffer       = m_command.m_buffer,
	    .submit       = submit,
	    .debug_op     = m_command.m_debug_op,
	    .debug_submit = m_command.m_debug_submit_id,
	    .debug_arg0   = m_command.m_debug_arg0,
	    .debug_arg1   = m_command.m_debug_arg1,
	    .debug_arg2   = m_command.m_debug_arg2,
	    .debug_arg3   = m_command.m_debug_arg3,
	    .debug_arg4   = m_command.m_debug_arg4,
	};
	m_command.m_buffer                = nullptr;
	m_recorded_release_mem_writes     = 0;
	m_recorded_release_mem_interrupts = 0;
	m_recorded_draws                  = 0;

	if (!m_async_submit) {
		QueueSubmit(job);
		return job.tick;
	}
	{
		// Ticks are allocated in queue order and the queue thread submits in that order, so the
		// timeline is signaled in order.
		std::lock_guard lock(m_submit_mutex);
		job.tick = m_master.NextTick();
		job.submit.AddSignal(m_master.Handle(), job.tick);
		m_submit_jobs.push_back(job);
	}
	m_submit_available.notify_one();
	return job.tick;
}

void CommandScheduler::QueueSubmit(SubmitJob& job) {
	auto&      graphics = m_graphics;
	vk::Result result;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		if (!m_async_submit) {
			job.tick = m_master.NextTick();
			job.submit.AddSignal(m_master.Handle(), job.tick);
		}
		const auto& submit = job.submit;

		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &job.buffer;
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

		result = graphics.queue.submit(1, &submit_info, nullptr);
	}
	Timeline::Mark("vk-submit", job.tick);

	if (result == vk::Result::eErrorDeviceLost) {
		DumpDeviceLossDiagnostics(graphics);
	}
	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, job.tick, job.debug_op, job.debug_submit,
		                  job.debug_arg0, job.debug_arg1, job.debug_arg2, job.debug_arg3,
		                  job.debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandScheduler::EnableAsyncSubmit() {
	// The owner enables it while constructing, before any other thread can submit.
	EXIT_IF(m_async_submit);
	m_async_submit  = true;
	m_submit_thread = std::jthread([this](std::stop_token stop) { SubmitThread(stop); });
}

void CommandScheduler::SubmitThread(std::stop_token stop) {
	KYTY_PROFILER_THREAD("GpuQueueSubmit");
	for (;;) {
		SubmitJob job;
		{
			std::unique_lock lock(m_submit_mutex);
			// A stop request still submits the queued jobs: their ticks may already be waited on.
			if (!m_submit_available.wait(lock, stop, [this] { return !m_submit_jobs.empty(); })) {
				return;
			}
			job = m_submit_jobs.front();
			m_submit_jobs.pop_front();
		}
		QueueSubmit(job);
	}
}

void CommandScheduler::StopSubmitThread() {
	if (m_submit_thread.joinable()) {
		m_submit_thread.request_stop();
		m_submit_thread.join();
	}
	std::lock_guard lock(m_submit_mutex);
	EXIT_IF(!m_submit_jobs.empty());
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
