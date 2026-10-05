#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/SrtNative.h"

#include <array>
#include <memory>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Value;

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	SrtMemoryReader           read_memory                = nullptr;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	// Accept image atomics on k32Float descriptors: upstream's float atomics, and integer atomics
	// run as uint atomics on the raw bits. On by default, as in the emulator; the emulator passes
	// --no-float-image-atomics through here.
	bool                      float_image_atomics        = true;
	// Compute: the dispatch's workgroup count (zero when the host does not know it, as for an
	// indirect dispatch) and workgroup size bound the invocation IDs in buffer write extents.
	std::array<uint32_t, 3>   workgroup_count            = {};
	std::array<uint32_t, 3>   workgroup_size             = {};
	// Optional fast path for raw reads: the host bytes of a guest page whose contents are current
	// (no GPU-written byte), or null; such a page is read directly instead of through
	// read_memory. Valid for one refresh. A reader that cannot read (CleanRuntime without a
	// strict reader) has none.
	const uint8_t* (*map_clean_page)(void* userdata, uint64_t page) = nullptr;
	void*          page_userdata                                     = nullptr;
	uint32_t       page_shift                                        = 12;
	// When the refresh captures its reads (specialization_reads), reads taken through the fast
	// path are recorded here, as read_memory's capture records them.
	std::vector<std::pair<uint64_t, uint64_t>>* capture_ranges = nullptr;
};

enum class RuntimeValueType { Any, Integer };

bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
// Uses the strict reader for values that affect shader specialization.
SrtRuntime CleanRuntime(SrtRuntime runtime);

class SrtWalker;

// Replay traces (KYTY_SRT_TRACE=0 turns them off). A resource refresh evaluates one program's
// values through two walkers (clean and ordinary) with memoized, recursive, lazy evaluation; the
// walk was a third of the GPU thread in the jungle (DEBUGGING.md, 2026-10-05). A trace records
// one refresh as a flat list of the instructions it evaluated, in order, with the operands each
// used, and the top-level requests (calls) it served. The next refresh of the program replays
// the list in a loop: no memo, no recursion, no operand resolution.
//
// Every data-dependent choice the recording made is a guard: an instruction that succeeded or
// failed must do the same, a select must pick the same arm, a logical AND/OR must take the same
// short-circuit path, and the refresh must request the same values in the same order. A guard
// that does not hold abandons the trace: that request and the rest of the refresh are evaluated
// as before (the memo is untouched by replay), and the program records again.
// KYTY_SRT_TRACE_VERIFY=1 evaluates every replayed refresh again without the trace and stops on
// a difference; KYTY_SRT_TRACE_STATS=1 reports how refreshes went.
struct SrtTraceOp {
	enum Kind : uint8_t {
		Pure,            // ApplyPureOp over the operands
		UserData,        // user data word `imm`
		ShaderBase,
		Pass,            // the operand (Phi, bit casts, ReadConst, ConditionRef, extract of a pair)
		Extract64,       // a 64-bit operand's dword `flags`
		AddCarryExtract, // IAddCarry32 of two operands, dword `flags`
		Select,          // predicate, then the arm `flags` (1 or 2) the recording took
		LogicalAnd,      // the interpreter's short-circuit rules
		LogicalOr,
		RawRead,         // EvaluateRawRead; `flags` 1 for a constant buffer, `imm` the offset
	};
	Kind        kind   = Pure;
	uint8_t     walker = 0;     // 0 the clean walker, 1 the ordinary one
	uint8_t     count  = 0;     // operands
	uint8_t     flags  = 0;
	bool        ok     = false; // what the recording got
	ValueOpcode opcode = ValueOpcode::Void;
	int32_t     imm    = 0;
	int32_t     args[5] {};     // SrtTraceSession::Event refs
	const Inst* inst = nullptr;
};

struct SrtTraceCall {
	Value       raw;                 // the requested value as given (matched without resolving)
	const Inst* inst      = nullptr; // null: an immediate
	uint64_t    immediate = 0;
	uint32_t    end       = 0;       // ops up to here are evaluated for it
	int32_t     result    = 0;       // an Event ref
	uint8_t     walker    = 0;
	bool        ok        = false;
};

struct SrtTrace {
	std::vector<SrtTraceOp>   ops;
	std::vector<uint64_t>     immediates;
	std::vector<SrtTraceCall> calls;
	uint8_t                   key = 0;
};


class SrtTraceSession {
public:
	SrtTraceSession(const ResourcePlan& program, SrtWalker& clean, SrtWalker& walker,
	                const SrtRuntime& runtime);
	~SrtTraceSession();
	SrtTraceSession(const SrtTraceSession&)            = delete;
	SrtTraceSession& operator=(const SrtTraceSession&) = delete;

	// The refresh succeeded: a recording becomes the program's trace.
	void Succeeded() { m_succeeded = true; }
	// The refresh was served from the trace from start to end.
	[[nodiscard]] bool FullyServed() const { return m_mode == Mode::Serve; }

	// Turns tracing off on this thread (verification re-runs).
	static bool& Suppressed();

private:
	friend class SrtWalker;
	enum class Mode : uint8_t { Off, Record, Serve, Passthrough };

	struct Event {
		int32_t  ref   = 0; // >= 0: an op; < 0: immediate -1 - ref; Failed: an operand that failed
		bool     ok    = false;
		uint64_t value = 0; // while recording
	};
	static constexpr int32_t Failed = INT32_MIN;
	struct Frame {
		std::array<Event, 6> events {};
		uint32_t             count  = 0;
		Event                result {Failed, false, 0};
		bool                 have   = false;
	};

	// A request the walkers make: answered inline from the trace when it is the next one and
	// needs no more ops (the common case), else through EvaluateSlow.
	inline bool Evaluate(SrtWalker& walker, Value value, uint64_t& result);
	bool EvaluateSlow(SrtWalker& walker, Value value, uint64_t& result);
	bool Serve(SrtWalker& walker, Value value, uint64_t& result);
	bool RunOps(uint32_t end);
	void Abandon(const char* reason);
	void Abort(const char* reason);
	// Recording hooks, called from SrtWalker::EvaluateWideImpl.
	void OnImmediate(bool ok, uint64_t bits);
	void OnMemoHit(const SrtWalker& walker, uint32_t index, uint64_t value);
	void OnInst(const SrtWalker& walker, const Inst& inst, uint32_t index, bool ok, uint64_t value);
	void RestoreNative();

	const ResourcePlan&    m_program;
	SrtWalker*             m_walkers[2];
	const SrtNativeCode*   m_native[2] {};
	Mode                   m_mode      = Mode::Off;
	bool                   m_succeeded = false;
	// Recording.
	std::unique_ptr<SrtTrace> m_recording;
	std::vector<Frame>        m_frames;
	std::vector<int32_t>      m_slots[2]; // evaluation index -> op, per walker
	const char*               m_abort = nullptr;
	// Serving.
	std::shared_ptr<SrtTrace> m_trace;
	uint64_t*                 m_values = nullptr; // thread-local scratch, as large as the trace
	uint8_t*                  m_status = nullptr;
	uint32_t                  m_cursor  = 0;
	uint32_t                  m_next_op = 0;
};

// One memoized evaluation session shared by the entire shader resource refresh.
class SrtWalker {
public:
	SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	          std::span<const uint8_t> clean_flat_slots = {}, SrtWalker* clean_evaluator = nullptr,
	          Value active_mask = {});
	~SrtWalker();
	SrtWalker(const SrtWalker&)            = delete;
	SrtWalker& operator=(const SrtWalker&) = delete;

	bool Evaluate(Value value, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// Refreshes reachable scalar reads and active descriptor sources in one walk.
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);

private:
	// RefreshFlatBuffer's full walk over the control flow (records the walk for replay).
	template <typename Refresh, typename Outcome>
	bool WalkBlocks(std::vector<uint32_t>& flat, const Refresh& refresh, const Outcome& outcome,
	                uint8_t key);
	friend struct SrtNativeHelpers;
	friend class SrtTraceSession;

	// Binds this walker to the plan's native code when the configuration is one it was compiled
	// for (SrtNative.h), compiling it once the plan is refreshed often enough.
	void BindNative();
	bool VerifyNative(Value value, bool native_ok, uint64_t native_result);
	// A value from the native code's tables, evaluated with this walker's frame and mode.
	bool EvaluateNative(const SrtNativeValue& value, uint64_t& result);
	[[nodiscard]] bool UseNativeTables() const;

	static ResourcePlan::EvaluationContext& AcquireContext(const ResourcePlan& program);
	static float Float32(uint64_t bits);
	bool EvaluateWide(Value value, uint64_t& result) {
		if (m_trace != nullptr) [[unlikely]] {
			return m_trace->Evaluate(*this, value, result);
		}
		return EvaluateWideImpl(value, result);
	}
	bool EvaluateWideImpl(Value value, uint64_t& result);
	bool Arg(const Inst& inst, size_t index, uint64_t& result);
	bool EvaluatePhi(const Inst& inst, uint64_t& result);
	bool EvaluateExtract(const Inst& inst, uint64_t& result);
	bool EvaluateRawRead(const Inst& inst, uint64_t& result);
	// EvaluateRawRead's read once the address is known.
	bool ReadRawWord(uint64_t address, uint64_t base, uint64_t& result);
	bool EvaluateBufferRead(const Inst& inst, uint64_t& result);
	bool EvaluateInst(const Inst& inst, uint64_t& result);

	const ResourcePlan&              m_program;
	SrtRuntime                      m_runtime;
	std::span<const uint8_t>         m_clean_flat_slots;
	SrtWalker*                      m_clean_evaluator = nullptr;
	Value                           m_active_mask;
	const Inst*                     m_failed_value = nullptr;
	ResourcePlan::EvaluationContext& m_context;
	const SrtNativeCode*            m_native      = nullptr;
	SrtNativeMode                   m_native_mode = SrtNativeMode::Self;
	SrtNativeFrame                  m_native_frame;
	SrtTraceSession*                m_trace    = nullptr;
	uint8_t                         m_trace_id = 0;
	// ReadRawWord's fast path: the last page mapped (SrtRuntime::map_clean_page).
	uint64_t       m_mapped_page  = UINT64_MAX;
	const uint8_t* m_mapped_bytes = nullptr;
	// The last raw read that failed, for RefreshFlatBuffer's report.
	const char* m_read_failure         = nullptr;
	uint64_t    m_read_failure_address = 0;
	uint64_t    m_read_failure_offset  = 0;
	uint64_t    m_read_failure_size    = 0;
};

inline bool SrtTraceSession::Evaluate(SrtWalker& walker, Value value, uint64_t& result) {
	if (m_mode == Mode::Serve && m_frames.empty() && m_cursor < m_trace->calls.size()) {
		const auto& call = m_trace->calls[m_cursor];
		if (call.walker == walker.m_trace_id && call.raw.SameInstruction(value) &&
		    (call.end <= m_next_op || RunOps(call.end))) {
			m_cursor++;
			if (call.result == Failed) {
				return false;
			}
			if (call.result < 0) {
				result = m_trace->immediates[static_cast<size_t>(-1 - call.result)];
				return true;
			}
			result = m_values[static_cast<size_t>(call.result)];
			return m_status[static_cast<size_t>(call.result)] != 0u;
		}
	}
	// Not the expected request, or RunOps failed a guard (Serve then runs those ops again, fails
	// the same guard and abandons the trace).
	return EvaluateSlow(walker, value, result);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
