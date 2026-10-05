#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>

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
	// As DeferOperation, also between guest command buffers (no active command buffer): runs
	// once the GPU has finished the current tick, so work submitted earlier, still running, can
	// keep using what the operation releases.
	void                      DeferRelease(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] static bool InDeferredOperation() noexcept;
	// Hands later submissions to a dedicated queue thread, which submits them in tick order, so
	// vkQueueSubmit (and the queue lock that present also takes) leaves the recording thread.
	// Submit() then returns once the tick is allocated; host waits on a timeline value may precede
	// its signal operation. Enable before the first Submit().
	void EnableAsyncSubmit();
	[[nodiscard]] bool AsyncSubmit() const noexcept { return m_async_submit; }
	// A Vulkan recording thread (KYTY_RECORD_THREAD=0 turns it off; the pattern and the
	// CommandRecorder are IDXTRI's, d9958e28). The GPU thread queues the current command
	// buffer's Vulkan commands (CommandBuffer::Recorder(), Record()); the recording thread begins,
	// records, ends and submits the command buffers in order, so the driver's recording cost
	// leaves the GPU thread. Ticks are still allocated here, in order, at Submit(). Code that
	// records through the raw CommandBuffer::Handle() first waits for the thread to catch up
	// (DrainRecording), so it stays correct. Replaces the queue thread (EnableAsyncSubmit).
	// Enable before the first command buffer.
	void EnableRecordingThread();
	// The current command buffer's commands go through the recording thread.
	[[nodiscard]] bool Threaded() const noexcept { return m_threaded; }
	// Queues fn(vk::CommandBuffer) for the recording thread (threaded only). Captures must be
	// values; arrays go through Stash().
	template <typename F>
	void Record(F&& fn) {
		using Command = TypedCommand<std::decay_t<F>>;
		void* memory  = Allocate(sizeof(Command), alignof(Command));
		auto* command = new (memory) Command(std::forward<F>(fn));
		if (m_chunk->last != nullptr) {
			m_chunk->last->next = command;
		} else {
			m_chunk->first = command;
		}
		m_chunk->last = command;
	}
	// Copies count elements into memory that stays valid until the recording thread has run the
	// next command recorded after this call (threaded only).
	template <typename T>
	[[nodiscard]] const T* Stash(const T* data, size_t count) {
		if (data == nullptr || count == 0) {
			return nullptr;
		}
		auto* copy = static_cast<T*>(Allocate(sizeof(T) * count, alignof(T)));
		std::memcpy(static_cast<void*>(copy), data, sizeof(T) * count);
		return copy;
	}
	// Waits until the recording thread has run every queued command, and returns the command
	// buffer it records into, for code that records directly.
	vk::CommandBuffer DrainRecording();

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

		// tick: the tick the command buffer will be submitted with.
		vk::CommandBuffer Commit(uint64_t tick);

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

	// A finished command buffer and everything its vkQueueSubmit needs, including the recording
	// state that a fatal submit report prints.
	struct SubmitJob {
		vk::CommandBuffer buffer = nullptr;
		SubmitInfo        submit;
		uint64_t          tick         = 0;
		uint32_t          debug_op     = 0;
		uint64_t          debug_submit = 0;
		uint32_t          debug_arg0   = 0;
		uint32_t          debug_arg1   = 0;
		uint32_t          debug_arg2   = 0;
		uint32_t          debug_arg3   = 0;
		uint64_t          debug_arg4   = 0;
	};

	struct RecordedCommand {
		virtual ~RecordedCommand()                  = default;
		virtual void     Execute(vk::CommandBuffer) = 0;
		RecordedCommand* next                       = nullptr;
	};
	template <typename F>
	struct TypedCommand final: RecordedCommand {
		explicit TypedCommand(F&& function): fn(std::move(function)) {}
		explicit TypedCommand(const F& function): fn(function) {}
		void Execute(vk::CommandBuffer command) override { fn(command); }
		F    fn;
	};
	struct Chunk {
		static constexpr size_t Size = 256 * 1024;
		alignas(64) std::byte storage[Size];
		size_t           used  = 0;
		RecordedCommand* first = nullptr;
		RecordedCommand* last  = nullptr;
	};
	void* Allocate(size_t size, size_t alignment);
	void  DispatchChunk();
	void  RecordingThread(std::stop_token stop);
	void  StopRecordingThread();
	void  ReportRecording();

	void BeginNext();
	void QueueSubmit(SubmitJob& job);
	void SubmitThread(std::stop_token stop);
	void StopSubmitThread();
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
	bool                         m_async_submit         = false;
	std::deque<SubmitJob>        m_submit_jobs;
	std::mutex                   m_submit_mutex;
	std::condition_variable_any  m_submit_available;
	std::jthread                 m_submit_thread;
	// Recording thread. m_threaded and m_chunk belong to the producer (the GPU thread); the queue,
	// free list and counters to m_record_mutex; m_worker_buffer to the recording thread (read by
	// the producer only after a drain).
	bool                                m_threaded = false;
	std::unique_ptr<Chunk>              m_chunk;
	std::mutex                          m_record_mutex;
	std::condition_variable_any         m_record_available;
	std::condition_variable             m_record_executed;
	std::deque<std::unique_ptr<Chunk>>  m_record_queue;
	std::vector<std::unique_ptr<Chunk>> m_free_chunks;
	uint64_t                            m_chunks_dispatched = 0;
	uint64_t                            m_chunks_executed   = 0;
	uint64_t                            m_drains            = 0;
	vk::CommandBuffer                   m_worker_buffer     = nullptr;
	uint64_t                            m_record_report_buffers = 0;
	uint64_t                            m_record_report_chunks  = 0;
	uint64_t                            m_record_report_drains  = 0;
	std::chrono::steady_clock::time_point m_record_report_time {};
	std::jthread                        m_record_thread;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
