#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <condition_variable>
#include <mutex>

#include <queue>

#include <thread>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler {
public:
	CommandScheduler(RenderContext& context, GraphicContext& graphics);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering();
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           FlushAndWait();
	void           Finish();
	// Bounds how many no-interrupt RELEASE_MEM fence writes accumulate in one command buffer
	// before it is submitted, so consecutive fence updates that nothing is blocked waiting on
	// don't each pay a host vkQueueSubmit. Only safe for a RELEASE_MEM whose guest-visible write
	// already happened synchronously and that scheduled no interrupt callback -- see the call
	// site in pm4Handlers.cpp CpOpReleaseMem for the exact condition. Never waits itself.
	void           CompleteReleaseMemWrite();
	// Same idea, but for a RELEASE_MEM that DOES request a guest interrupt/event (a guest thread
	// may be waiting on it via an event queue) -- deferring the flush delays real event delivery,
	// so this uses a much smaller batch bound than CompleteReleaseMemWrite as a hedge, trading
	// some of the possible win for less added latency. Never waits itself.
	void           CompleteReleaseMemInterrupt();
	// Called after every DrawIndex/DrawAuto. Long chains of draws with no intervening
	// RELEASE_MEM/wait can otherwise sit fully recorded but unsubmitted for a long time, leaving
	// the GPU idle until something else finally forces a flush -- this periodically calls Flush()
	// (non-blocking: Submit() + BeginNext(), same as CompleteReleaseMemWrite) every
	// KYTY_DRAW_FLUSH_INTERVAL draws to keep the queue fed instead. Submitting while the GPU is
	// still executing earlier work is always legal (vkQueueSubmit never waits on prior submissions
	// completing), so this never blocks the CPU. Defaults to 16, validated against real gameplay --
	// override via KYTY_DRAW_FLUSH_INTERVAL (0 disables) if a different workload needs retuning:
	// too small reintroduces per-submit overhead, too large leaves the same idle bubbles this
	// exists to remove.
	void           CompleteDraw();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

private:
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	void BeginNext();
	void PriorityOperationsThread(std::stop_token stop);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	uint32_t                     m_recorded_release_mem_writes     = 0;
	uint32_t                     m_recorded_release_mem_interrupts = 0;
	uint32_t                     m_recorded_draws                  = 0;
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	OperationState               m_operation_state      = OperationState::Open;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
