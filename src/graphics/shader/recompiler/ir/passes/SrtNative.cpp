#include "graphics/shader/recompiler/ir/passes/SrtNative.h"

#include "common/assert.h"
#include "common/config.h"
#include "graphics/shader/recompiler/ir/passes/SrtNativeAsm.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <bit>
#include <cstddef>
#include <cstring>


namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {

using namespace X64;

// SrtNativeFrame field offsets, as the generated code addresses them.
constexpr int32_t FrameMemo         = 0;
constexpr int32_t FrameGeneration   = 8;
constexpr int32_t FrameUserData     = 16;
constexpr int32_t FrameUserDataSize = 24;
constexpr int32_t FrameShaderBase   = 32;
constexpr int32_t FrameClean        = 48;
constexpr int32_t FrameFailed       = 56;
static_assert(offsetof(SrtNativeFrame, memo) == FrameMemo);
static_assert(offsetof(SrtNativeFrame, generation) == FrameGeneration);
static_assert(offsetof(SrtNativeFrame, user_data) == FrameUserData);
static_assert(offsetof(SrtNativeFrame, user_data_size) == FrameUserDataSize);
static_assert(offsetof(SrtNativeFrame, shader_base) == FrameShaderBase);
static_assert(offsetof(SrtNativeFrame, clean) == FrameClean);
static_assert(offsetof(SrtNativeFrame, failed) == FrameFailed);
static_assert(sizeof(ResourcePlan::EvaluationContext::Entry) == 16);

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

// Integer argument registers and the shadow space of the host C ABI, for helper calls.
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
constexpr Reg     Arg0   = RCX;
constexpr Reg     Arg1   = RDX;
constexpr Reg     Arg2   = R8;
constexpr Reg     Arg3   = R9;
constexpr int32_t Shadow = 32;
#else
constexpr Reg     Arg0   = RDI;
constexpr Reg     Arg1   = RSI;
constexpr Reg     Arg2   = RDX;
constexpr Reg     Arg3   = RCX;
constexpr int32_t Shadow = 0;
#endif

bool ImmediateBits(Value value, uint64_t& bits) {
	switch (value.GetType()) {
		case Type::U1: bits = value.U1(); return true;
		case Type::U8: bits = value.U8(); return true;
		case Type::U16: bits = value.U16(); return true;
		case Type::U32: bits = value.U32(); return true;
		case Type::U64: bits = value.U64(); return true;
		case Type::F32: bits = std::bit_cast<uint32_t>(value.F32Value()); return true;
		default: return false;
	}
}

bool IsRawRead(const ResourcePlan& program, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= program.memory_info.size()) {
		return false;
	}
	const auto kind = program.memory_info[index].kind;
	return (op == ValueOpcode::LoadAddressU32 && kind == ResourceKind::ScalarAddress) ||
	       (op == ValueOpcode::ReadConstBuffer && kind == ResourceKind::ScalarBuffer);
}

} // namespace

// Helpers the generated code calls. They are members so that they can reach the walker.
struct SrtNativeHelpers {
	// One instruction through the interpreter; its own memo entry is the routine's to fill.
	static bool Interpret(SrtNativeFrame* frame, const Inst* inst, uint64_t* result) {
		return frame->walker->EvaluateInst(*inst, *result);
	}
	// A value outside the compiled set (not in the plan's storage), memo and all.
	static bool EvaluateValue(SrtNativeFrame* frame, const Inst* inst, uint64_t* result) {
		return frame->walker->EvaluateWide(Value(const_cast<Inst*>(inst)), *result);
	}
	// SrtWalker::EvaluateRawRead's read, after the address computation.
	static bool Read(SrtNativeFrame* frame, uint64_t address, uint64_t base, uint64_t* result) {
		auto&    walker = *frame->walker;
		uint32_t word   = 0;
		if (walker.m_runtime.read_memory != nullptr) {
			if (!walker.m_runtime.read_memory(walker.m_runtime.userdata, address, {&word, 1})) {
				if (base == 0) {
					*result = 0;
					return true;
				}
				walker.m_read_failure         = "guest memory unreadable";
				walker.m_read_failure_address = address;
				walker.m_read_failure_offset  = 0;
				walker.m_read_failure_size    = 0;
				return false;
			}
		} else {
			std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
		}
		*result = word;
		return true;
	}
};

namespace {

class Compiler {
public:
	explicit Compiler(const ResourcePlan& program): m_program(program) {}

	bool Run(std::vector<uint32_t> (&routines)[2], uint32_t& instructions, uint32_t& interpreted) {
		// Every instruction gets its evaluation index now, so the count is final.
		for (const auto& inst: m_program.value_storage) {
			const auto index = inst.EvaluationIndex(m_program.evaluation_value_count);
			if (index >= m_insts.size()) {
				m_insts.resize(index + 1, nullptr);
			}
			m_insts[index] = &inst;
		}
		for (auto& labels: m_labels) {
			labels.resize(m_insts.size());
			for (auto& label: labels) {
				label = m_asm.NewLabel();
			}
		}
		EmitEntry();
		for (uint32_t mode = 0; mode < 2; mode++) {
			routines[mode].assign(m_insts.size(), UINT32_MAX);
			for (size_t index = 0; index < m_insts.size(); index++) {
				if (m_insts[index] == nullptr) continue;
				routines[mode][index] = static_cast<uint32_t>(m_asm.Size());
				EmitRoutine(static_cast<SrtNativeMode>(mode), *m_insts[index]);
			}
		}
		instructions = static_cast<uint32_t>(m_insts.size());
		interpreted  = m_interpreted / 2;
		return m_asm.Finish();
	}

	[[nodiscard]] const Assembler& Asm() const { return m_asm; }

private:
	// entry(frame, routine, result) -> bool, in the host C ABI.
	void EmitEntry() {
		auto& a = m_asm;
		a.Push(RBX), a.Push(RBP), a.Push(R12), a.Push(R13), a.Push(R14), a.Push(R15);
		a.MovRR(R15, Arg2);
		a.MovRR(R11, Arg1);
		a.MovRR(R12, Arg0);
		a.Load(R13, R12, FrameMemo);
		a.Load(R14, R12, FrameGeneration);
		a.CallReg(R11);
		const auto failed = a.NewLabel();
		const auto done   = a.NewLabel();
		a.Jcc(CondB, failed);
		a.Store(R15, 0, RAX);
		a.MovRI(RAX, 1);
		a.Jmp(done);
		a.Bind(failed);
		a.MovRI(RAX, 0);
		a.Bind(done);
		a.Pop(R15), a.Pop(R14), a.Pop(R13), a.Pop(R12), a.Pop(RBP), a.Pop(RBX);
		a.Ret();
	}

	[[nodiscard]] bool Compiled(const Inst* inst) const {
		const auto index = inst->EvaluationIndex(m_program.evaluation_value_count);
		return index < m_insts.size() && m_insts[index] == inst;
	}

	Label RoutineLabel(SrtNativeMode mode, const Inst* inst) const {
		return m_labels[static_cast<uint32_t>(mode)]
		               [inst->EvaluationIndex(m_program.evaluation_value_count)];
	}

	// Calls a C helper: aligns the stack, keeps RBX (callee-saved) as the saved stack pointer.
	// `setup` loads the arguments; the result is the helper's bool in RCX (0/1) and its uint64_t
	// output in RAX. Reloads the memo base, which an interpreter call may have moved.
	template <typename Setup>
	void EmitHelperCall(const void* helper, Setup&& setup) {
		auto& a = m_asm;
		a.Push(RBX);
		a.MovRR(RBX, RSP);
		a.AndI(RSP, -16);
		a.SubI(RSP, Shadow + 16);
		setup();
		a.CallAbs(helper);
		a.MovzxB(RCX, RAX);
		a.Load(RAX, RSP, Shadow);
		a.MovRR(RSP, RBX);
		a.Pop(RBX);
		a.Load(R13, R12, FrameMemo);
	}

	// Value -> RAX; jumps to `fail` when it cannot be evaluated.
	void EmitValue(SrtNativeMode mode, Value value, Label fail) {
		auto& a = m_asm;
		value   = value.Resolve();
		if (value.IsImmediate()) {
			uint64_t bits = 0;
			if (ImmediateBits(value, bits)) {
				a.MovRI(RAX, bits);
			} else {
				a.Jmp(fail);
			}
			return;
		}
		const auto* inst = value.TryInstruction();
		if (!Compiled(inst)) {
			EmitHelperCall(reinterpret_cast<const void*>(&SrtNativeHelpers::EvaluateValue), [&] {
				a.MovRR(Arg0, R12);
				a.MovRI(Arg1, reinterpret_cast<uint64_t>(inst));
				a.Lea(Arg2, RSP, Shadow);
			});
			a.Test(RCX, RCX);
			a.Jcc(CondE, fail);
			return;
		}
		a.Call(RoutineLabel(mode, inst));
		a.Jcc(CondB, fail);
	}

	// A value the interpreter evaluates in the clean walker when there is one: select predicates
	// and clean SRT slots. In Split mode that is the clean frame, which runs Self code.
	void EmitCleanValue(SrtNativeMode mode, Value value, Label fail) {
		if (mode == SrtNativeMode::Self) {
			EmitValue(mode, value, fail);
			return;
		}
		auto& a        = m_asm;
		const auto res = value.Resolve();
		if (res.IsImmediate()) {
			EmitValue(SrtNativeMode::Self, res, fail);
			return;
		}
		if (!Compiled(res.TryInstruction())) {
			// An uncompiled value goes through the clean walker's interpreter.
			a.Push(R12), a.Push(R13), a.Push(R14);
			a.Load(R12, R12, FrameClean);
			a.Load(R13, R12, FrameMemo);
			a.Load(R14, R12, FrameGeneration);
			const auto failed = a.NewLabel();
			const auto done   = a.NewLabel();
			EmitValue(SrtNativeMode::Self, res, failed);
			a.Pop(R14), a.Pop(R13), a.Pop(R12);
			a.Jmp(done);
			a.Bind(failed);
			a.Pop(R14), a.Pop(R13), a.Pop(R12);
			a.Jmp(fail);
			a.Bind(done);
			return;
		}
		a.Push(R12), a.Push(R13), a.Push(R14);
		a.Load(R12, R12, FrameClean);
		a.Load(R13, R12, FrameMemo);
		a.Load(R14, R12, FrameGeneration);
		a.Call(RoutineLabel(SrtNativeMode::Self, res.TryInstruction()));
		// POP leaves the flags alone, so the callee's CF survives.
		a.Pop(R14), a.Pop(R13), a.Pop(R12);
		a.Jcc(CondB, fail);
	}

	// Evaluates `count` operands in order into RAX, RCX, RDX, R8 (the interpreter's order, which
	// stops at the first failure).
	void EmitOperands(SrtNativeMode mode, const Inst& inst, size_t count, Label fail) {
		static constexpr Reg Targets[4] = {RAX, RCX, RDX, R8};
		auto&                a          = m_asm;
		uint64_t             immediates[4] {};
		bool                 immediate[4] {};
		for (size_t i = 0; i < count; i++) {
			const auto value = inst.Arg(i).Resolve();
			if (value.IsImmediate()) {
				if (!ImmediateBits(value, immediates[i])) {
					a.Jmp(fail);
					return;
				}
				immediate[i] = true;
				continue;
			}
			EmitValue(mode, value, fail);
			a.Push(RAX);
		}
		for (size_t i = count; i-- > 0;) {
			if (!immediate[i]) a.Pop(Targets[i]);
		}
		for (size_t i = 0; i < count; i++) {
			if (immediate[i]) a.MovRI(Targets[i], immediates[i]);
		}
	}

	void EmitInterpret(const Inst& inst, Label fail) {
		auto& a = m_asm;
		m_interpreted++;
		EmitHelperCall(reinterpret_cast<const void*>(&SrtNativeHelpers::Interpret), [&] {
			a.MovRR(Arg0, R12);
			a.MovRI(Arg1, reinterpret_cast<uint64_t>(&inst));
			a.Lea(Arg2, RSP, Shadow);
		});
		a.Test(RCX, RCX);
		a.Jcc(CondE, fail);
	}

	void EmitCompare(SrtNativeMode mode, const Inst& inst, Cond cond, Label fail) {
		auto& a = m_asm;
		EmitOperands(mode, inst, 2, fail);
		a.Cmp32(RAX, RCX);
		a.SetCC(cond, RAX);
		a.MovzxB(RAX, RAX);
	}

	void EmitBool(Reg reg) {
		m_asm.Test(reg, reg);
		m_asm.SetCC(CondNE, reg);
		m_asm.MovzxB(reg, reg);
	}

	// SrtWalker::EvaluateRawRead.
	void EmitRawRead(SrtNativeMode mode, const Inst& inst, Label fail) {
		auto&      a     = m_asm;
		const auto flags = inst.Flags<MemoryFlags>();
		if (flags.index >= m_program.memory_info.size()) {
			a.Jmp(fail);
			return;
		}
		if (!inst.Arg(0).Resolve().TryInstruction()) {
			// The interpreter would stop the emulator here; leave that to it.
			EmitInterpret(inst, fail);
			return;
		}
		const auto& mem           = m_program.memory_info[flags.index];
		const auto& handle        = *inst.Arg(0).ResolveInstruction();
		const bool  const_buffer  = inst.GetOpcode() == ValueOpcode::ReadConstBuffer;
		const auto  immediate     = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
		// Operands onto the stack in the interpreter's order: low, high, offset[, records, word3].
		// [RBP-8] low, [RBP-16] high, [RBP-24] offset, [RBP-32] records.
		if (handle.NumArgs() < 2u) {
			EmitInterpret(inst, fail);
			return;
		}
		EmitValue(mode, handle.Arg(0), fail);
		a.Push(RAX);
		EmitValue(mode, handle.Arg(1), fail);
		a.Push(RAX);
		EmitValue(mode, inst.Arg(1), fail);
		a.Push(RAX);
		if (const_buffer) {
			if (handle.NumArgs() != 4u) {
				a.Jmp(fail);
				return;
			}
			EmitValue(mode, handle.Arg(2), fail);
			a.Push(RAX);
			EmitValue(mode, handle.Arg(3), fail);
			if (immediate < 0) {
				a.Jmp(fail);
				return;
			}
		}
		// R11 = base = ((high << 32) | low32) & AddressMask.
		a.Load(R11, RBP, -16);
		a.ShlI(R11, 32);
		a.Load32(RCX, RBP, -8);
		a.Or(R11, RCX);
		a.MovRI(RCX, AddressMask);
		a.And(R11, RCX);
		const auto read = a.NewLabel();
		if (const_buffer) {
			const auto zero = a.NewLabel();
			// RCX = byte offset = (immediate & ~3) + (offset32 & ~3).
			a.Load32(RCX, RBP, -24);
			a.AndI32(RCX, ~3u);
			a.AddI(RCX, static_cast<int32_t>(immediate & ~int64_t {3}));
			// RDX = stride = (high32 >> 16) & 0x3fff; R8 = size.
			a.Load32(RDX, RBP, -16);
			a.ShrI(RDX, 16);
			a.AndI(RDX, 0x3fff);
			a.Load32(R8, RBP, -32);
			const auto sized = a.NewLabel();
			a.Test(RDX, RDX);
			a.Jcc(CondE, sized);
			a.Imul(R8, RDX);
			a.Bind(sized);
			// Past the end reads zero (PS5 ISA, scalar buffer addressing).
			a.Cmp(RCX, R8);
			a.Jcc(CondA, zero);
			a.MovRR(RDX, R8);
			a.Sub(RDX, RCX);
			a.CmpI(RDX, 4);
			a.Jcc(CondB, zero);
			a.MovRR(RAX, R11);
			a.AndI(RAX, -4);
			a.Add(RAX, RCX);
			a.Jmp(read);
			a.Bind(zero);
			a.MovRI(RAX, 0);
			const auto done = a.NewLabel();
			a.Jmp(done);
			a.Bind(read);
			EmitRead(fail);
			a.Bind(done);
			return;
		}
		// RCX = relative = (immediate & ~3) + (offset32 & ~3), signed.
		a.Load32(RCX, RBP, -24);
		a.AndI32(RCX, ~3u);
		a.MovRI(RDX, static_cast<uint64_t>(immediate & ~int64_t {3}));
		a.Add(RCX, RDX);
		// RAX = (base & ~3) + relative, within the 48-bit address space (AddSignedAddress).
		a.MovRR(RAX, R11);
		a.AndI(RAX, -4);
		const auto negative = a.NewLabel();
		a.Test(RCX, RCX);
		a.Jcc(CondL, negative);
		a.MovRI(RDX, AddressMask);
		a.Sub(RDX, RAX);
		a.Cmp(RCX, RDX);
		a.Jcc(CondA, fail);
		a.Add(RAX, RCX);
		a.Jmp(read);
		a.Bind(negative);
		a.Neg(RCX);
		a.Cmp(RCX, RAX);
		a.Jcc(CondA, fail);
		a.Sub(RAX, RCX);
		a.Bind(read);
		EmitRead(fail);
	}

	// RAX = address, R11 = base -> RAX = the dword read.
	void EmitRead(Label fail) {
		auto& a = m_asm;
		EmitHelperCall(reinterpret_cast<const void*>(&SrtNativeHelpers::Read), [&] {
			a.MovRR(Arg1, RAX);
			a.MovRR(Arg2, R11);
			a.MovRR(Arg0, R12);
			a.Lea(Arg3, RSP, Shadow);
		});
		a.Test(RCX, RCX);
		a.Jcc(CondE, fail);
	}

	// SrtWalker::EvaluateExtract.
	void EmitExtract(SrtNativeMode mode, const Inst& inst, Label fail) {
		auto&      a     = m_asm;
		const auto index = inst.Arg(1).Resolve();
		if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
			a.Jmp(fail);
			return;
		}
		const auto component = index.U32();
		if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
			EmitValue(mode, inst.Arg(0), fail);
			if (component != 0u) a.ShrI(RAX, 32);
			a.MovRR32(RAX, RAX);
			return;
		}
		if (!inst.Arg(0).Resolve().TryInstruction()) {
			EmitInterpret(inst, fail);
			return;
		}
		const auto& source = *inst.Arg(0).ResolveInstruction();
		if (source.GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
			EmitValue(mode, source.Arg(component), fail);
			return;
		}
		if (source.GetOpcode() == ValueOpcode::IAddCarry32) {
			EmitOperands(mode, source, 2, fail);
			a.MovRR32(RAX, RAX);
			a.MovRR32(RCX, RCX);
			a.Add(RAX, RCX);
			if (component == 0u) {
				a.MovRR32(RAX, RAX);
			} else {
				a.ShrI(RAX, 32);
			}
			return;
		}
		a.Jmp(fail);
	}

	// BitFieldUExtract / BitFieldSExtract: RAX = base, RCX = offset, RDX = width.
	void EmitBitFieldExtract(bool sign, Label fail) {
		auto&      a    = m_asm;
		const auto zero = a.NewLabel();
		const auto done = a.NewLabel();
		// offset > 32 || width > 32 - offset fails.
		a.MovRR32(RCX, RCX);
		a.MovRR32(RDX, RDX);
		a.CmpI(RCX, 32);
		a.Jcc(CondA, fail);
		a.MovRI(R8, 32);
		a.Sub(R8, RCX);
		a.Cmp(RDX, R8);
		a.Jcc(CondA, fail);
		a.Test(RDX, RDX);
		a.Jcc(CondE, zero);
		// RAX = (base32 >> offset) & ((1 << width) - 1); offset < 32 here since width > 0.
		a.MovRR32(RAX, RAX);
		a.ShrCl(RAX);
		a.MovRR(R8, RDX);
		a.MovRR(RCX, R8);
		a.MovRI(R9, 1);
		a.ShlCl(R9);
		a.SubI(R9, 1);
		a.And(RAX, R9);
		if (sign) {
			// Sign-extend from `width` bits, then keep 32.
			a.MovRI(RCX, 64);
			a.Sub(RCX, R8);
			a.ShlCl(RAX);
			a.SarCl(RAX);
			a.MovRR32(RAX, RAX);
		}
		a.Jmp(done);
		a.Bind(zero);
		a.MovRI(RAX, 0);
		a.Bind(done);
	}

	// BitFieldInsert: RAX = base, RCX = insert, RDX = offset, R8 = width.
	void EmitBitFieldInsert(Label fail) {
		auto&      a    = m_asm;
		const auto keep = a.NewLabel();
		const auto done = a.NewLabel();
		a.MovRR32(RDX, RDX);
		a.MovRR32(R8, R8);
		a.CmpI(RDX, 32);
		a.Jcc(CondA, fail);
		a.MovRI(R9, 32);
		a.Sub(R9, RDX);
		a.Cmp(R8, R9);
		a.Jcc(CondA, fail);
		a.Test(R8, R8);
		a.Jcc(CondE, keep);
		// R9 = mask = ((1 << width) - 1) << offset, in 32 bits.
		a.MovRR(R10, RCX);
		a.MovRR(RCX, R8);
		a.MovRI(R9, 1);
		a.ShlCl(R9);
		a.SubI(R9, 1);
		a.MovRR(RCX, RDX);
		a.ShlCl(R9);
		a.MovRR32(R9, R9);
		// RAX = (base & ~mask) | ((insert << offset) & mask).
		a.ShlCl(R10);
		a.And(R10, R9);
		a.Not(R9);
		a.And(RAX, R9);
		a.Or(RAX, R10);
		a.Bind(keep);
		a.MovRR32(RAX, RAX);
		a.Bind(done);
	}

	// Flushes a float operand's denormal input to a signed zero (FPCompareFlags).
	void EmitFlushDenorm(Reg reg) {
		auto&      a    = m_asm;
		const auto keep = a.NewLabel();
		a.MovRR32(R9, reg);
		a.AndI32(R9, 0x7fffffffu);
		a.CmpI(R9, 0x00800000);
		a.Jcc(CondAE, keep);
		a.AndI32(reg, 0x80000000u);
		a.Bind(keep);
	}

	// The instruction's value -> RAX, as SrtWalker::EvaluateInst.
	void EmitBody(SrtNativeMode mode, const Inst& inst, Label fail) {
		auto& a = m_asm;
		switch (inst.GetOpcode()) {
			case ValueOpcode::GetUserData: {
				const auto reg = inst.Arg(0);
				if (reg.GetType() != Type::ScalarReg) {
					EmitInterpret(inst, fail);
					return;
				}
				const auto index = RegIndex(reg.ScalarRegister());
				if (index < m_program.user_data_base) {
					a.Jmp(fail);
					return;
				}
				const auto slot = index - m_program.user_data_base;
				a.Load(RCX, R12, FrameUserDataSize);
				a.CmpI(RCX, static_cast<int32_t>(slot));
				a.Jcc(CondBE, fail);
				a.Load(RAX, R12, FrameUserData);
				a.Load32(RAX, RAX, static_cast<int32_t>(slot * 4u));
				return;
			}
			case ValueOpcode::GetShaderBase: a.Load(RAX, R12, FrameShaderBase); return;
			case ValueOpcode::Phi: {
				const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
				if (value.IsEmpty()) {
					a.Jmp(fail);
					return;
				}
				EmitValue(mode, value, fail);
				return;
			}
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::ConditionRef: EmitValue(mode, inst.Arg(0), fail); return;
			case ValueOpcode::CompositeExtractU64:
			case ValueOpcode::CompositeExtractU32x2: EmitExtract(mode, inst, fail); return;
			case ValueOpcode::CompositeConstructU64:
				EmitOperands(mode, inst, 2, fail);
				a.MovRR32(RAX, RAX);
				a.ShlI(RCX, 32);
				a.Or(RAX, RCX);
				return;
			case ValueOpcode::ReadConst: {
				const auto slot = inst.Arg(1).Resolve();
				if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
				    slot.U32() >= m_program.srt_reads.size()) {
					a.Jmp(fail);
					return;
				}
				const auto index = slot.U32();
				const auto value = m_program.srt_reads[index].value;
				// Split mode: the walker's clean slots are the plan's (checked when binding).
				if (mode == SrtNativeMode::Split && index < m_program.clean_flat_slots.size() &&
				    m_program.clean_flat_slots[index] != 0u) {
					EmitCleanValue(mode, value, fail);
				} else {
					EmitValue(mode, value, fail);
				}
				return;
			}
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer:
				if (IsRawRead(m_program, inst)) {
					EmitRawRead(mode, inst, fail);
				} else {
					a.Jmp(fail);
				}
				return;
			case ValueOpcode::IAdd32:
				EmitOperands(mode, inst, 2, fail), a.Add(RAX, RCX), a.MovRR32(RAX, RAX);
				return;
			case ValueOpcode::IAdd64: EmitOperands(mode, inst, 2, fail), a.Add(RAX, RCX); return;
			case ValueOpcode::ISub32:
				EmitOperands(mode, inst, 2, fail), a.Sub(RAX, RCX), a.MovRR32(RAX, RAX);
				return;
			case ValueOpcode::ISub64: EmitOperands(mode, inst, 2, fail), a.Sub(RAX, RCX); return;
			case ValueOpcode::IMul32:
				EmitOperands(mode, inst, 2, fail), a.Imul(RAX, RCX), a.MovRR32(RAX, RAX);
				return;
			case ValueOpcode::IMul64: EmitOperands(mode, inst, 2, fail), a.Imul(RAX, RCX); return;
			case ValueOpcode::UMulHi:
				EmitOperands(mode, inst, 2, fail);
				a.MovRR32(RAX, RAX), a.MovRR32(RCX, RCX), a.Imul(RAX, RCX), a.ShrI(RAX, 32);
				return;
			case ValueOpcode::UMin32: {
				EmitOperands(mode, inst, 2, fail);
				const auto done = a.NewLabel();
				a.MovRR32(RAX, RAX), a.MovRR32(RCX, RCX);
				a.Cmp(RAX, RCX);
				a.Jcc(CondBE, done);
				a.MovRR(RAX, RCX);
				a.Bind(done);
				return;
			}
			case ValueOpcode::ConvertF32U32:
				EmitValue(mode, inst.Arg(0), fail);
				a.MovRR32(RAX, RAX), a.Cvtsi2ss64(XMM0, RAX), a.MovdRX(RAX, XMM0);
				return;
			case ValueOpcode::ConvertU32F32:
				// Finite, not below zero (-0 converts to 0) and within uint32_t, else fail.
				EmitValue(mode, inst.Arg(0), fail);
				a.MovRR32(RAX, RAX);
				a.MovRR(RCX, RAX);
				a.AndI32(RCX, 0x7f800000u);
				a.CmpI(RCX, 0x7f800000);
				a.Jcc(CondE, fail);
				{
					const auto positive = a.NewLabel();
					a.MovRR(RCX, RAX);
					a.ShrI(RCX, 31);
					a.Test(RCX, RCX);
					a.Jcc(CondE, positive);
					a.MovRR(RCX, RAX);
					a.AndI32(RCX, 0x7fffffffu);
					a.Test(RCX, RCX);
					a.Jcc(CondNE, fail);
					a.Bind(positive);
				}
				a.MovdXR(XMM0, RAX);
				a.Cvttss2si64(RAX, XMM0);
				a.MovRI(RCX, UINT32_MAX);
				a.Cmp(RAX, RCX);
				a.Jcc(CondA, fail);
				return;
			case ValueOpcode::FPMul32:
				EmitOperands(mode, inst, 2, fail);
				a.MovdXR(XMM0, RAX), a.MovdXR(XMM1, RCX), a.Mulss(XMM0, XMM1), a.MovdRX(RAX, XMM0);
				return;
			case ValueOpcode::FPTrunc32:
				EmitValue(mode, inst.Arg(0), fail);
				a.MovdXR(XMM0, RAX), a.Roundss(XMM0, XMM0, 0x0b), a.MovdRX(RAX, XMM0);
				return;
			case ValueOpcode::FPRecipIFlag32:
				// Exact reciprocals of normal powers of two only.
				EmitValue(mode, inst.Arg(0), fail);
				a.MovRR32(RAX, RAX);
				a.MovRR(RCX, RAX);
				a.AndI32(RCX, 0x807fffffu);
				a.Test(RCX, RCX);
				a.Jcc(CondNE, fail);
				a.ShrI(RAX, 23);
				a.AndI(RAX, 255);
				a.Test(RAX, RAX);
				a.Jcc(CondE, fail);
				a.CmpI(RAX, 254);
				a.Jcc(CondAE, fail);
				a.MovRI(RCX, 254);
				a.Sub(RCX, RAX);
				a.ShlI(RCX, 23);
				a.MovRR(RAX, RCX);
				return;
			case ValueOpcode::FPIsNan32:
				EmitValue(mode, inst.Arg(0), fail);
				a.AndI32(RAX, 0x7fffffffu);
				a.CmpI(RAX, 0x7f800000);
				a.SetCC(CondA, RAX);
				a.MovzxB(RAX, RAX);
				return;
			case ValueOpcode::FPOrdLessThanEqual32:
			case ValueOpcode::FPOrdGreaterThanEqual32:
				EmitOperands(mode, inst, 2, fail);
				if (inst.Flags<FPCompareFlags>().flush_input_denorms) {
					EmitFlushDenorm(RAX);
					EmitFlushDenorm(RCX);
				}
				a.MovdXR(XMM0, RAX), a.MovdXR(XMM1, RCX);
				// Ordered: an unordered compare sets CF, so AE is false for NaN.
				if (inst.GetOpcode() == ValueOpcode::FPOrdLessThanEqual32) {
					a.Ucomiss(XMM1, XMM0);
				} else {
					a.Ucomiss(XMM0, XMM1);
				}
				a.SetCC(CondAE, RAX);
				a.MovzxB(RAX, RAX);
				return;
			case ValueOpcode::BitwiseAnd32:
				EmitOperands(mode, inst, 2, fail), a.And(RAX, RCX), a.MovRR32(RAX, RAX);
				return;
			case ValueOpcode::BitwiseAnd64: EmitOperands(mode, inst, 2, fail), a.And(RAX, RCX); return;
			case ValueOpcode::BitwiseOr32:
				EmitOperands(mode, inst, 2, fail), a.Or(RAX, RCX), a.MovRR32(RAX, RAX);
				return;
			case ValueOpcode::BitwiseXor32:
				EmitOperands(mode, inst, 2, fail), a.Xor(RAX, RCX), a.MovRR32(RAX, RAX);
				return;
			case ValueOpcode::BitwiseNot32:
				EmitValue(mode, inst.Arg(0), fail), a.Not(RAX), a.MovRR32(RAX, RAX);
				return;
			case ValueOpcode::ShiftLeftLogical32:
				EmitOperands(mode, inst, 2, fail), a.ShlCl32(RAX);
				return;
			case ValueOpcode::ShiftLeftLogical64: EmitOperands(mode, inst, 2, fail), a.ShlCl(RAX); return;
			case ValueOpcode::ShiftRightLogical32:
				EmitOperands(mode, inst, 2, fail), a.ShrCl32(RAX);
				return;
			case ValueOpcode::ShiftRightLogical64: EmitOperands(mode, inst, 2, fail), a.ShrCl(RAX); return;
			case ValueOpcode::ShiftRightArithmetic32:
				EmitOperands(mode, inst, 2, fail), a.SarCl32(RAX);
				return;
			case ValueOpcode::ShiftRightArithmetic64:
				EmitOperands(mode, inst, 2, fail), a.SarCl(RAX);
				return;
			case ValueOpcode::BitFieldUExtract:
				EmitOperands(mode, inst, 3, fail), EmitBitFieldExtract(false, fail);
				return;
			case ValueOpcode::BitFieldSExtract:
				EmitOperands(mode, inst, 3, fail), EmitBitFieldExtract(true, fail);
				return;
			case ValueOpcode::BitFieldInsert:
				EmitOperands(mode, inst, 4, fail), EmitBitFieldInsert(fail);
				return;
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectF32: {
				const auto other = a.NewLabel();
				const auto done  = a.NewLabel();
				EmitCleanValue(mode, inst.Arg(0), fail);
				a.Test(RAX, RAX);
				a.Jcc(CondE, other);
				EmitValue(mode, inst.Arg(1), fail);
				a.Jmp(done);
				a.Bind(other);
				EmitValue(mode, inst.Arg(2), fail);
				a.Bind(done);
				return;
			}
			case ValueOpcode::IEqual32: EmitCompare(mode, inst, CondE, fail); return;
			case ValueOpcode::INotEqual32: EmitCompare(mode, inst, CondNE, fail); return;
			case ValueOpcode::ULessThan32: EmitCompare(mode, inst, CondB, fail); return;
			case ValueOpcode::UGreaterThan32: EmitCompare(mode, inst, CondA, fail); return;
			case ValueOpcode::ULessThanEqual32: EmitCompare(mode, inst, CondBE, fail); return;
			case ValueOpcode::SGreaterThanEqual32: EmitCompare(mode, inst, CondGE, fail); return;
			case ValueOpcode::LogicalAnd:
				EmitOperands(mode, inst, 2, fail), EmitBool(RAX), EmitBool(RCX), a.And(RAX, RCX);
				return;
			case ValueOpcode::LogicalOr:
				EmitOperands(mode, inst, 2, fail), EmitBool(RAX), EmitBool(RCX), a.Or(RAX, RCX);
				return;
			case ValueOpcode::LogicalXor:
				EmitOperands(mode, inst, 2, fail), EmitBool(RAX), EmitBool(RCX), a.Xor(RAX, RCX);
				return;
			case ValueOpcode::LogicalNot:
				EmitValue(mode, inst.Arg(0), fail);
				a.Test(RAX, RAX), a.SetCC(CondE, RAX), a.MovzxB(RAX, RAX);
				return;
			case ValueOpcode::ReadFirstLane:
			case ValueOpcode::LoadBufferU32: EmitInterpret(inst, fail); return;
			default: a.Jmp(fail); return;
		}
	}

	void EmitRoutine(SrtNativeMode mode, const Inst& inst) {
		auto&      a       = m_asm;
		const auto index   = inst.EvaluationIndex(m_program.evaluation_value_count);
		const auto value   = static_cast<int32_t>(index * 16u);
		const auto state   = value + 8;
		const auto compute = a.NewLabel();
		const auto cycle   = a.NewLabel();
		const auto fail    = a.NewLabel();
		a.Bind(RoutineLabel(mode, &inst));
		// Memo hit: this generation's value. CF is clear after an equal compare.
		a.Load(RAX, R13, state);
		a.Cmp(RAX, R14);
		a.Jcc(CondNE, compute);
		a.Load(RAX, R13, value);
		a.Ret();
		a.Bind(compute);
		// generation | 1 marks a value being evaluated: a cycle fails.
		a.Lea(RCX, R14, 1);
		a.Cmp(RAX, RCX);
		a.Jcc(CondE, cycle);
		a.Store(R13, state, RCX);
		a.Push(RBP);
		a.MovRR(RBP, RSP);
		EmitBody(mode, inst, fail);
		a.MovRR(RSP, RBP);
		a.Pop(RBP);
		a.Load(R13, R12, FrameMemo);
		a.Store(R13, value, RAX);
		a.Store(R13, state, R14);
		a.Clc();
		a.Ret();
		// Failure: clear the memo entry and remember the first failing instruction.
		a.Bind(fail);
		a.MovRR(RSP, RBP);
		a.Pop(RBP);
		a.Load(R13, R12, FrameMemo);
		a.StoreImm32(R13, state, 0);
		const auto recorded = a.NewLabel();
		a.Load(RAX, R12, FrameFailed);
		a.CmpMemI(RAX, 0, 0);
		a.Jcc(CondNE, recorded);
		a.MovRI(RCX, reinterpret_cast<uint64_t>(&inst));
		a.Store(RAX, 0, RCX);
		a.Bind(recorded);
		a.Stc();
		a.Ret();
		a.Bind(cycle);
		a.Stc();
		a.Ret();
	}

	const ResourcePlan&      m_program;
	Assembler                m_asm;
	std::vector<const Inst*> m_insts;
	std::vector<Label>       m_labels[2];
	uint32_t                 m_interpreted = 0;
};

} // namespace

// SrtNativeMemory.cpp
void* SrtNativeAllocateCode(const void* code, size_t size);
void  SrtNativeFreeCode(void* memory, size_t size);

SrtNativeCode::~SrtNativeCode() {
	SrtNativeFreeCode(m_memory, m_size);
}

bool SrtNativeCode::Supported() {
#if defined(__x86_64__) || defined(_M_X64)
	return true;
#else
	return false;
#endif
}

std::unique_ptr<SrtNativeCode> SrtNativeCode::Compile(const ResourcePlan& program) {
	if (!Supported()) {
		return nullptr;
	}
	std::unique_ptr<SrtNativeCode> code(new SrtNativeCode());
	Compiler                       compiler(program);
	if (!compiler.Run(code->m_routines, code->m_instructions, code->m_interpreted)) {
		return nullptr;
	}
	const auto& bytes = compiler.Asm().Code();
	code->m_size      = bytes.size();
	auto* memory = static_cast<uint8_t*>(SrtNativeAllocateCode(bytes.data(), code->m_size));
	if (memory == nullptr) {
		return nullptr;
	}
	code->m_memory = memory;
	// The entry thunk is emitted first.
	code->m_entry = reinterpret_cast<Entry>(memory);
	const auto resolve = [&](Value value) {
		SrtNativeValue out;
		value = value.Resolve();
		if (value.IsImmediate()) {
			out.kind = ImmediateBits(value, out.immediate) ? SrtNativeValue::Immediate
			                                                : SrtNativeValue::Fail;
			return out;
		}
		const auto* inst  = value.TryInstruction();
		const auto  index = inst->EvaluationIndex(program.evaluation_value_count);
		if (code->Has(SrtNativeMode::Self, index) && code->Has(SrtNativeMode::Split, index)) {
			out.kind  = SrtNativeValue::Routine;
			out.index = index;
			out.inst  = inst;
		} else {
			out.kind = SrtNativeValue::Interpret;
			out.inst = inst;
		}
		return out;
	};
	for (const auto& read: program.srt_reads) {
		code->m_flat_reads.push_back(resolve(read.value));
	}
	for (const auto& block: program.control_flow) {
		code->m_conditions.push_back(block.condition.IsEmpty() ? SrtNativeValue {}
		                                                       : resolve(block.condition));
	}
	code->m_descriptor_dwords.resize(program.descriptor_sources.size() * 8u);
	for (size_t source = 0; source < program.descriptor_sources.size(); source++) {
		const auto& descriptor = program.descriptor_sources[source];
		for (uint32_t dword = 0; dword < descriptor.dword_count && dword < 8u; dword++) {
			code->m_descriptor_dwords[source * 8u + dword] = resolve(descriptor.dwords[dword]);
		}
	}
	return code;
}

bool SrtNativeCode::Has(SrtNativeMode mode, uint32_t index) const {
	const auto& routines = m_routines[static_cast<uint32_t>(mode)];
	return index < routines.size() && routines[index] != UINT32_MAX;
}

bool SrtNativeCode::Evaluate(SrtNativeFrame& frame, SrtNativeMode mode, uint32_t index,
                             uint64_t& result) const {
	const auto* routine = m_memory + m_routines[static_cast<uint32_t>(mode)][index];
	return m_entry(&frame, routine, &result);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
