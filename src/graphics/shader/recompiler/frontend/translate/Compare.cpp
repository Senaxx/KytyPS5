#include "graphics/shader/recompiler/frontend/translate/Translator.h"

#include <bit>

namespace Libs::Graphics::ShaderRecompiler::Frontend {

void Translator::EmitCompareResult(const Decoder::Instruction& inst, IR::U1 value, bool scalar,
                                   bool cmpx) {
	if (scalar) {
		ir.SetScc(value);
		return;
	}
	const auto masked = ir.LogicalAnd(ir.GetExec(), value);
	if (cmpx) {
		const auto mask = BallotMask(masked);
		ir.SetExec(masked);
		ir.SetExecLo(mask[0]);
		ir.SetExecHi(mask[1]);
		return;
	}
	WriteMask(inst.dst, masked);
}

void Translator::EmitCompareConstant(const Decoder::Instruction& inst, bool value, bool scalar,
                                     bool cmpx) {
	EmitCompareResult(inst, IR::U1(IR::Value(value)), scalar, cmpx);
}

void Translator::EmitIntegerCompare(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                                    IR::Type type, bool scalar, bool cmpx) {
	const bool signed_64 = opcode == IR::ValueOpcode::SLessThan64 ||
	                       opcode == IR::ValueOpcode::SLessThanEqual64 ||
	                       inst.opcode == Decoder::Opcode::V_CMP_EQ_I64 ||
	                       inst.opcode == Decoder::Opcode::V_CMPX_NE_I64;
	const auto read = [&](const Decoder::Operand& operand) {
		// RDNA2 expands signed 64-bit integer literals by sign extension.
		if (signed_64 && operand.kind == Decoder::OperandKind::LiteralConstant) {
			return IR::Value(static_cast<uint64_t>(
			    static_cast<int64_t>(std::bit_cast<int32_t>(operand.value))));
		}
		return ReadOperand(operand, type);
	};
	EmitCompareResult(inst, IR::U1(ir.Emit(opcode, {read(inst.src0), read(inst.src1)})), scalar,
	                  cmpx);
}

void Translator::EmitInteger16Compare(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                                      bool signed_value, bool cmpx) {
	const auto lhs = ReadU16AsU32(inst.src0, signed_value);
	const auto rhs = ReadU16AsU32(inst.src1, signed_value);
	EmitCompareResult(inst, IR::U1(ir.Emit(opcode, {lhs, rhs})), false, cmpx);
}

void Translator::EmitFloatCompare(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                                  bool half, bool cmpx) {
	const auto type = IR::ArgTypeOf(opcode, 0);
	const auto lhs =
	    half ? IR::Value(ReadF16AsF32(inst.src0)) : ReadOperand(inst.src0, type);
	const auto rhs =
	    half ? IR::Value(ReadF16AsF32(inst.src1)) : ReadOperand(inst.src1, type);
	const IR::FPCompareFlags flags{
	    .flush_input_denorms = !half && type == IR::Type::F32 && flush_f32_inputs};
	EmitCompareResult(inst, IR::U1(ir.Emit(opcode, {lhs, rhs}, flags)), false, cmpx);
}

// A double as its {low, high} dwords. Inline float constants are f64 values in f64 instructions;
// 1/(2*pi) has its own f64 encoding. ABS and NEG act on the sign bit, bit 63.
std::array<IR::U32, 2> Translator::ReadF64Halves(const Decoder::Operand& operand) {
	if (operand.kind == Decoder::OperandKind::FloatInlineConstant) {
		const uint64_t bits =
		    operand.value == 0x3e22f983u
		        ? 0x3fc45f306dc9c882ull
		        : std::bit_cast<uint64_t>(static_cast<double>(std::bit_cast<float>(operand.value)));
		return {IR::U32(IR::Value(static_cast<uint32_t>(bits))),
		        IR::U32(IR::Value(static_cast<uint32_t>(bits >> 32u)))};
	}
	auto halves = ReadU32Pair(PlainOperand(operand));
	if (operand.absolute) {
		halves[1] = ir.BitwiseAnd(halves[1], IR::U32(IR::Value(0x7fffffffu)));
	}
	if (operand.negate) {
		halves[1] = ir.BitwiseXor(halves[1], IR::U32(IR::Value(0x80000000u)));
	}
	return halves;
}

// V_CMP_EQ_F64 on the bit patterns: equal unless either side is NaN, and +0 equals -0.
void Translator::EmitFloat64EqualCompare(const Decoder::Instruction& inst) {
	const auto lhs       = ReadF64Halves(inst.src0);
	const auto rhs       = ReadF64Halves(inst.src1);
	const auto magnitude = [&](const std::array<IR::U32, 2>& value) {
		return ir.BitwiseAnd(value[1], IR::U32(IR::Value(0x7fffffffu)));
	};
	const auto is_nan = [&](const std::array<IR::U32, 2>& value) {
		const auto high     = magnitude(value);
		const auto infinity = IR::U32(IR::Value(0x7ff00000u));
		return ir.LogicalOr(ir.UGreaterThan(high, infinity),
		                    ir.LogicalAnd(ir.IEqual(high, infinity),
		                                  ir.INotEqual(value[0], IR::U32(IR::Value(0u)))));
	};
	const auto is_zero = [&](const std::array<IR::U32, 2>& value) {
		return ir.IEqual(ir.BitwiseOr(magnitude(value), value[0]), IR::U32(IR::Value(0u)));
	};
	const auto same_bits =
	    ir.LogicalAnd(ir.IEqual(lhs[0], rhs[0]), ir.IEqual(lhs[1], rhs[1]));
	const auto equal =
	    ir.LogicalAnd(ir.LogicalNot(ir.LogicalOr(is_nan(lhs), is_nan(rhs))),
	                  ir.LogicalOr(same_bits, ir.LogicalAnd(is_zero(lhs), is_zero(rhs))));
	EmitCompareResult(inst, equal, false, false);
}

void Translator::EmitFloatOrderedCompare(const Decoder::Instruction& inst, bool ordered, bool cmpx) {
	const auto lhs       = IR::F32(ReadOperand(inst.src0, IR::Type::F32));
	const auto rhs       = IR::F32(ReadOperand(inst.src1, IR::Type::F32));
	const auto unordered = ir.LogicalOr(IR::U1(ir.Emit(IR::ValueOpcode::FPIsNan32, {lhs})),
	                                    IR::U1(ir.Emit(IR::ValueOpcode::FPIsNan32, {rhs})));
	EmitCompareResult(inst, ordered ? ir.LogicalNot(unordered) : unordered, false, cmpx);
}

void Translator::EmitFloatClassCompare(const Decoder::Instruction& inst, bool cmpx) {
	const auto value = ReadOperand(inst.src0, IR::Type::F32);
	const auto mask  = ReadOperand(inst.src1, IR::Type::U32);
	EmitCompareResult(inst, IR::U1(ir.Emit(IR::ValueOpcode::FPCmpClass32, {value, mask})), false,
	                  cmpx);
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
