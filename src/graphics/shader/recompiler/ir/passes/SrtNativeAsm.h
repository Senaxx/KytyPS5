#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTNATIVEASM_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTNATIVEASM_H_

// A minimal x86-64 assembler for the native SRT walker (SrtNative.cpp): only the encodings the
// walker emits. Every memory operand is [base + disp32], so no base register needs a special
// ModRM form except RSP/R12, which take a SIB byte.

#include <cstdint>
#include <cstring>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR::X64 {

enum Reg : uint8_t {
	RAX = 0,
	RCX = 1,
	RDX = 2,
	RBX = 3,
	RSP = 4,
	RBP = 5,
	RSI = 6,
	RDI = 7,
	R8  = 8,
	R9  = 9,
	R10 = 10,
	R11 = 11,
	R12 = 12,
	R13 = 13,
	R14 = 14,
	R15 = 15,
};

enum Xmm : uint8_t { XMM0 = 0, XMM1 = 1 };

// Condition codes (the low nibble of Jcc/SETcc).
enum Cond : uint8_t {
	CondB  = 0x2, // below / carry
	CondAE = 0x3, // above or equal / no carry
	CondE  = 0x4,
	CondNE = 0x5,
	CondBE = 0x6,
	CondA  = 0x7,
	CondL  = 0xc,
	CondGE = 0xd,
	CondLE = 0xe,
	CondG  = 0xf,
};

// A forward or backward jump target inside one Assembler.
struct Label {
	int32_t id = -1;
};

class Assembler {
public:
	[[nodiscard]] const std::vector<uint8_t>& Code() const { return m_code; }
	[[nodiscard]] size_t                      Size() const { return m_code.size(); }

	Label NewLabel() {
		m_labels.push_back(-1);
		return Label {static_cast<int32_t>(m_labels.size() - 1)};
	}
	void Bind(Label label) { m_labels[label.id] = static_cast<int64_t>(m_code.size()); }
	[[nodiscard]] int64_t Position(Label label) const { return m_labels[label.id]; }

	// Resolves label jumps; returns false if a label was never bound.
	bool Finish() {
		for (const auto& fixup: m_fixups) {
			const auto target = m_labels[fixup.label];
			if (target < 0) {
				return false;
			}
			Patch32(fixup.position, static_cast<int32_t>(target - (fixup.position + 4)));
		}
		m_fixups.clear();
		return true;
	}

	// --- moves ---
	void MovRR(Reg dst, Reg src) { Rex(true, src, dst), Byte(0x89), ModRR(src, dst); }
	void MovRR32(Reg dst, Reg src) { RexOpt(false, src, dst), Byte(0x89), ModRR(src, dst); }
	void MovRI(Reg dst, uint64_t imm) {
		if (imm <= UINT32_MAX) {
			RexOpt(false, 0, dst);
			Byte(static_cast<uint8_t>(0xb8 + (dst & 7)));
			Imm32(static_cast<uint32_t>(imm));
		} else {
			Rex(true, 0, dst);
			Byte(static_cast<uint8_t>(0xb8 + (dst & 7)));
			Imm64(imm);
		}
	}
	void Load(Reg dst, Reg base, int32_t disp) { Rex(true, dst, base), Byte(0x8b), ModMem(dst, base, disp); }
	void Load32(Reg dst, Reg base, int32_t disp) {
		RexOpt(false, dst, base), Byte(0x8b), ModMem(dst, base, disp);
	}
	void Store(Reg base, int32_t disp, Reg src) { Rex(true, src, base), Byte(0x89), ModMem(src, base, disp); }
	void StoreImm32(Reg base, int32_t disp, int32_t imm) {
		Rex(true, 0, base), Byte(0xc7), ModMem(0, base, disp), Imm32(static_cast<uint32_t>(imm));
	}
	void Lea(Reg dst, Reg base, int32_t disp) { Rex(true, dst, base), Byte(0x8d), ModMem(dst, base, disp); }
	void Push(Reg reg) {
		if (reg >= 8) Byte(0x41);
		Byte(static_cast<uint8_t>(0x50 + (reg & 7)));
	}
	void Pop(Reg reg) {
		if (reg >= 8) Byte(0x41);
		Byte(static_cast<uint8_t>(0x58 + (reg & 7)));
	}

	// --- 64-bit ALU, register/register ---
	void Add(Reg dst, Reg src) { Alu(0x01, dst, src); }
	void Sub(Reg dst, Reg src) { Alu(0x29, dst, src); }
	void And(Reg dst, Reg src) { Alu(0x21, dst, src); }
	void Or(Reg dst, Reg src) { Alu(0x09, dst, src); }
	void Xor(Reg dst, Reg src) { Alu(0x31, dst, src); }
	void Cmp(Reg lhs, Reg rhs) { Alu(0x39, lhs, rhs); }
	void Cmp32(Reg lhs, Reg rhs) { RexOpt(false, rhs, lhs), Byte(0x39), ModRR(rhs, lhs); }
	void Test(Reg lhs, Reg rhs) { Alu(0x85, lhs, rhs); }
	void Imul(Reg dst, Reg src) { Rex(true, dst, src), Byte(0x0f), Byte(0xaf), ModRR(dst, src); }
	// rdx:rax = rax * src (unsigned)
	void Mul(Reg src) { Rex(true, 0, src), Byte(0xf7), ModRR(4, src); }
	void Not(Reg reg) { Rex(true, 0, reg), Byte(0xf7), ModRR(2, reg); }
	void Neg(Reg reg) { Rex(true, 0, reg), Byte(0xf7), ModRR(3, reg); }
	// 32-bit AND with an immediate; zero-extends into the full register.
	void AndI32(Reg reg, uint32_t imm) { RexOpt(false, 0, reg), Byte(0x81), ModRR(4, reg), Imm32(imm); }
	void ShlCl(Reg reg) { Rex(true, 0, reg), Byte(0xd3), ModRR(4, reg); }
	void ShrCl(Reg reg) { Rex(true, 0, reg), Byte(0xd3), ModRR(5, reg); }
	void SarCl(Reg reg) { Rex(true, 0, reg), Byte(0xd3), ModRR(7, reg); }
	void ShrCl32(Reg reg) { RexOpt(false, 0, reg), Byte(0xd3), ModRR(5, reg); }
	void ShlCl32(Reg reg) { RexOpt(false, 0, reg), Byte(0xd3), ModRR(4, reg); }
	void SarCl32(Reg reg) { RexOpt(false, 0, reg), Byte(0xd3), ModRR(7, reg); }
	void ShlI(Reg reg, uint8_t count) { Rex(true, 0, reg), Byte(0xc1), ModRR(4, reg), Byte(count); }
	void ShrI(Reg reg, uint8_t count) { Rex(true, 0, reg), Byte(0xc1), ModRR(5, reg), Byte(count); }
	// 64-bit ALU with a sign-extended 32-bit immediate: /0 add, /4 and, /5 sub, /7 cmp.
	void AddI(Reg reg, int32_t imm) { AluI(0, reg, imm); }
	void AndI(Reg reg, int32_t imm) { AluI(4, reg, imm); }
	void SubI(Reg reg, int32_t imm) { AluI(5, reg, imm); }
	void CmpI(Reg reg, int32_t imm) { AluI(7, reg, imm); }
	void CmpMemI(Reg base, int32_t disp, int32_t imm) {
		Rex(true, 0, base), Byte(0x81), ModMem(7, base, disp), Imm32(static_cast<uint32_t>(imm));
	}
	void CmpMem(Reg reg, Reg base, int32_t disp) { Rex(true, reg, base), Byte(0x3b), ModMem(reg, base, disp); }
	void SetCC(Cond cond, Reg reg) {
		// REX so that SIL/DIL and R8B-R15B are addressable.
		Byte(static_cast<uint8_t>(0x40 | (reg >= 8 ? 1 : 0)));
		Byte(0x0f), Byte(static_cast<uint8_t>(0x90 | cond)), ModRR(0, reg);
	}
	void MovzxB(Reg dst, Reg src) {
		Byte(static_cast<uint8_t>(0x48 | ((dst >= 8) ? 4 : 0) | ((src >= 8) ? 1 : 0)));
		Byte(0x0f), Byte(0xb6), ModRR(dst, src);
	}
	void Clc() { Byte(0xf8); }
	void Stc() { Byte(0xf9); }
	void Ret() { Byte(0xc3); }

	// --- control flow ---
	void Jmp(Label label) { Byte(0xe9), Fixup(label); }
	void Jcc(Cond cond, Label label) { Byte(0x0f), Byte(static_cast<uint8_t>(0x80 | cond)), Fixup(label); }
	void Call(Label label) { Byte(0xe8), Fixup(label); }
	void CallAbs(const void* target) {
		MovRI(RAX, reinterpret_cast<uint64_t>(target));
		Byte(0xff), ModRR(2, RAX);
	}
	void CallReg(Reg reg) { RexOpt(false, 0, reg), Byte(0xff), ModRR(2, reg); }

	// --- SSE (scalar single) ---
	void MovdXR(Xmm dst, Reg src) { Byte(0x66), RexOpt(false, dst, src), Byte(0x0f), Byte(0x6e), ModRR(dst, src); }
	void MovdRX(Reg dst, Xmm src) { Byte(0x66), RexOpt(false, src, dst), Byte(0x0f), Byte(0x7e), ModRR(src, dst); }
	void Mulss(Xmm dst, Xmm src) { Byte(0xf3), Byte(0x0f), Byte(0x59), ModRR(dst, src); }
	void Ucomiss(Xmm lhs, Xmm rhs) { Byte(0x0f), Byte(0x2e), ModRR(lhs, rhs); }
	void Cvtsi2ss64(Xmm dst, Reg src) {
		Byte(0xf3), Rex(true, dst, src), Byte(0x0f), Byte(0x2a), ModRR(dst, src);
	}
	void Cvttss2si64(Reg dst, Xmm src) {
		Byte(0xf3), Rex(true, dst, src), Byte(0x0f), Byte(0x2c), ModRR(dst, src);
	}
	void Xorps(Xmm dst, Xmm src) { Byte(0x0f), Byte(0x57), ModRR(dst, src); }
	// roundss dst, src, imm (SSE4.1); imm 3 = truncate toward zero.
	void Roundss(Xmm dst, Xmm src, uint8_t mode) {
		Byte(0x66), Byte(0x0f), Byte(0x3a), Byte(0x0a), ModRR(dst, src), Byte(mode);
	}

private:
	struct FixupEntry {
		size_t  position;
		int32_t label;
	};

	void Byte(uint8_t value) { m_code.push_back(value); }
	void Imm32(uint32_t value) {
		for (int i = 0; i < 4; i++) Byte(static_cast<uint8_t>(value >> (8 * i)));
	}
	void Imm64(uint64_t value) {
		for (int i = 0; i < 8; i++) Byte(static_cast<uint8_t>(value >> (8 * i)));
	}
	void Patch32(size_t position, int32_t value) { std::memcpy(&m_code[position], &value, 4); }
	void Fixup(Label label) {
		m_fixups.push_back({m_code.size(), label.id});
		Imm32(0);
	}
	// REX.W with R (ModRM.reg) and B (ModRM.rm or base) extensions.
	void Rex(bool w, uint8_t reg, uint8_t rm) {
		Byte(static_cast<uint8_t>(0x40 | (w ? 8 : 0) | ((reg & 8) ? 4 : 0) | ((rm & 8) ? 1 : 0)));
	}
	// A REX prefix only when an extended register needs one.
	void RexOpt(bool w, uint8_t reg, uint8_t rm) {
		if (w || (reg & 8) || (rm & 8)) Rex(w, reg, rm);
	}
	void ModRR(uint8_t reg, uint8_t rm) {
		Byte(static_cast<uint8_t>(0xc0 | ((reg & 7) << 3) | (rm & 7)));
	}
	// [base + disp32]; RSP and R12 as a base need a SIB byte.
	void ModMem(uint8_t reg, uint8_t base, int32_t disp) {
		Byte(static_cast<uint8_t>(0x80 | ((reg & 7) << 3) | (base & 7)));
		if ((base & 7) == RSP) Byte(0x24);
		Imm32(static_cast<uint32_t>(disp));
	}
	void Alu(uint8_t opcode, Reg dst, Reg src) { Rex(true, src, dst), Byte(opcode), ModRR(src, dst); }
	void AluI(uint8_t ext, Reg reg, int32_t imm) {
		Rex(true, 0, reg), Byte(0x81), ModRR(ext, reg), Imm32(static_cast<uint32_t>(imm));
	}

	std::vector<uint8_t>    m_code;
	std::vector<int64_t>    m_labels;
	std::vector<FixupEntry> m_fixups;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR::X64

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTNATIVEASM_H_
