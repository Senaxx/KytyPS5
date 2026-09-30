#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTNATIVE_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTNATIVE_H_

// The native SRT walker: a resource plan's evaluation graph compiled into x86-64 machine code.
//
// SrtWalker (SrtWalker.cpp) interprets a plan's IR on every draw and dispatch: resolve the value,
// check the memo, dispatch on the opcode, recurse into the operands. That interpretation was most
// of the GPU thread's resource walk (DEBUGGING.md, 2026-09-30). SrtNativeCode emits one routine
// per plan instruction and walker mode that does the same work with the dispatch compiled away:
//
// - Each routine checks and fills the same memo entry the interpreter uses
//   (ResourcePlan::EvaluationContext, {value, generation}, the low generation bit marking a value
//   still being evaluated), so native and interpreted evaluation mix freely.
// - Operands are evaluated lazily, in the interpreter's order: a select evaluates only the arm its
//   predicate chooses, so no memory is read that the interpreter would not read.
// - Guest memory reads go through the walker's reader, like the interpreter's.
// - An instruction the compiler does not handle calls the interpreter for that one instruction.
//
// Two modes, matching MaterializeResources' two walkers:
// - Self: the walker has no clean evaluator. Select predicates and every SRT slot use it.
// - Split: the walker has a clean evaluator (itself in Self mode) and the plan's clean flat slots;
//   select predicates and clean slots are evaluated in the clean walker, as the interpreter does.
//
// Routine convention (private to the generated code): R12 = the current walker's frame,
// R13 = its memo base, R14 = its generation. A routine returns the value in RAX with CF clear, or
// CF set on failure. It may clobber RAX, RCX, RDX, R8-R11 and XMM0-XMM1, and preserves the rest.

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

class SrtWalker;

// One walker's state as the generated code sees it. The field offsets are part of the code.
struct SrtNativeFrame {
	ResourcePlan::EvaluationContext::Entry* memo           = nullptr; // 0
	uint64_t                                generation     = 0;       // 8
	const uint32_t*                         user_data      = nullptr; // 16
	uint64_t                                user_data_size = 0;       // 24
	uint64_t                                shader_base    = 0;       // 32
	SrtWalker*                              walker         = nullptr; // 40
	SrtNativeFrame*                         clean          = nullptr; // 48
	const Inst**                            failed         = nullptr; // 56
};

enum class SrtNativeMode : uint32_t { Self = 0, Split = 1 };

// Totals of the plans compiled so far (SrtWalker.cpp).
struct SrtNativeStats {
	uint32_t plans        = 0;
	uint32_t failed       = 0;
	uint64_t bytes        = 0;
	uint64_t instructions = 0;
	uint64_t interpreted  = 0;
};

// True, with the totals, once after each doubling of the number of plans compiled.
bool TakeSrtNativeReport(SrtNativeStats& stats);

class SrtNativeCode {
public:
	~SrtNativeCode();
	SrtNativeCode(const SrtNativeCode&)            = delete;
	SrtNativeCode& operator=(const SrtNativeCode&) = delete;

	// Compiles every instruction of the plan in both modes. Assigns the plan's evaluation indices,
	// so evaluation_value_count is final afterwards. Returns null when the host cannot run it.
	static std::unique_ptr<SrtNativeCode> Compile(const ResourcePlan& program);

	// Whether this build can generate and run native code at all.
	static bool Supported();

	// Whether the instruction with this evaluation index has a routine in this mode.
	[[nodiscard]] bool Has(SrtNativeMode mode, uint32_t index) const;
	// Evaluates the instruction with this evaluation index (Has() must be true): the same result,
	// memo updates and failure as SrtWalker::EvaluateWide on that instruction.
	bool Evaluate(SrtNativeFrame& frame, SrtNativeMode mode, uint32_t index,
	              uint64_t& result) const;

	[[nodiscard]] size_t CodeSize() const { return m_size; }
	[[nodiscard]] uint32_t Instructions() const { return m_instructions; }
	[[nodiscard]] uint32_t Interpreted() const { return m_interpreted; }

private:
	SrtNativeCode() = default;

	using Entry = bool (*)(SrtNativeFrame* frame, const void* routine, uint64_t* result);

	uint8_t*              m_memory = nullptr;
	size_t                m_size   = 0;
	Entry                 m_entry  = nullptr;
	// Routine offsets by evaluation index, per mode; UINT32_MAX when the index has no routine.
	std::vector<uint32_t> m_routines[2];
	uint32_t              m_instructions = 0;
	uint32_t              m_interpreted  = 0;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTNATIVE_H_
