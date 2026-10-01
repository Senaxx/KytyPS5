#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

Value Arg(const Inst& inst, size_t index) {
	return inst.Arg(index).Resolve();
}

bool IsImmediate(Value value, Type type) {
	return value.IsImmediate() && value.GetType() == type;
}

void Replace(Inst& inst, Value value) {
	inst.ReplaceUsesWith(value.Resolve());
}

template <typename Function>
bool FoldU32(Inst& inst, Function function) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (!IsImmediate(lhs, Type::U32) || !IsImmediate(rhs, Type::U32)) {
		return false;
	}
	Replace(inst, Value(static_cast<uint32_t>(function(lhs.U32(), rhs.U32()))));
	return true;
}

template <typename Function>
bool FoldU64(Inst& inst, Function function) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (!IsImmediate(lhs, Type::U64) || !IsImmediate(rhs, Type::U64)) {
		return false;
	}
	Replace(inst, Value(static_cast<uint64_t>(function(lhs.U64(), rhs.U64()))));
	return true;
}

template <typename Function>
bool FoldU64Shift(Inst& inst, Function function) {
	const auto value = Arg(inst, 0);
	const auto shift = Arg(inst, 1);
	if (!IsImmediate(value, Type::U64) || !IsImmediate(shift, Type::U32)) {
		return false;
	}
	Replace(inst, Value(static_cast<uint64_t>(function(value.U64(), shift.U32()))));
	return true;
}

template <typename Function>
bool FoldU32Compare(Inst& inst, Function function) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (lhs == rhs) {
		Replace(inst, Value(function(0u, 0u)));
		return true;
	}
	if (!IsImmediate(lhs, Type::U32) || !IsImmediate(rhs, Type::U32)) {
		return false;
	}
	Replace(inst, Value(function(lhs.U32(), rhs.U32())));
	return true;
}

template <typename Function>
bool FoldU64Compare(Inst& inst, Function function) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (lhs == rhs) {
		Replace(inst, Value(function(uint64_t {0}, uint64_t {0})));
		return true;
	}
	if (!IsImmediate(lhs, Type::U64) || !IsImmediate(rhs, Type::U64)) {
		return false;
	}
	Replace(inst, Value(function(lhs.U64(), rhs.U64())));
	return true;
}

template <typename Function>
bool FoldLogical(Inst& inst, Function function) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (!IsImmediate(lhs, Type::U1) || !IsImmediate(rhs, Type::U1)) {
		return false;
	}
	Replace(inst, Value(function(lhs.U1(), rhs.U1())));
	return true;
}

bool ReplaceBinaryIdentity(Inst& inst, Type type, uint64_t identity) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (IsImmediate(lhs, type) &&
	    (type == Type::U32 ? lhs.U32() == identity : lhs.U64() == identity)) {
		Replace(inst, rhs);
		return true;
	}
	if (IsImmediate(rhs, type) &&
	    (type == Type::U32 ? rhs.U32() == identity : rhs.U64() == identity)) {
		Replace(inst, lhs);
		return true;
	}
	return false;
}

// Whether one boolean is the LogicalNot of the other.
bool IsComplement(Value lhs, Value rhs) {
	const auto negates = [](Value negation, Value value) {
		const auto* inst = negation.TryInstruction();
		return inst != nullptr && inst->GetOpcode() == ValueOpcode::LogicalNot &&
		       Arg(*inst, 0) == value;
	};
	return negates(lhs, rhs) || negates(rhs, lhs);
}

bool FoldSelect(Inst& inst) {
	const auto condition   = Arg(inst, 0);
	const auto true_value  = Arg(inst, 1);
	const auto false_value = Arg(inst, 2);
	if (IsImmediate(condition, Type::U1)) {
		Replace(inst, condition.U1() ? true_value : false_value);
		return true;
	}
	if (true_value == false_value) {
		Replace(inst, true_value);
		return true;
	}
	return false;
}

bool FoldPhi(Inst& inst) {
	Value same;
	for (size_t index = 0; index < inst.NumArgs(); index++) {
		const auto value = Arg(inst, index);
		if (value.TryInstruction() == &inst) {
			continue;
		}
		if (same.IsEmpty()) {
			same = value;
		} else if (same != value) {
			return false;
		}
	}
	if (same.IsEmpty()) {
		return false;
	}
	Replace(inst, same);
	return true;
}

bool FoldBitCast(Inst& inst, ValueOpcode reverse) {
	const auto value = Arg(inst, 0);
	if (IsImmediate(value, Type::F32) && inst.GetOpcode() == ValueOpcode::BitCastU32F32) {
		Replace(inst, Value(std::bit_cast<uint32_t>(value.F32Value())));
		return true;
	}
	if (IsImmediate(value, Type::U32) && inst.GetOpcode() == ValueOpcode::BitCastF32U32) {
		Replace(inst, Value::F32(std::bit_cast<float>(value.U32())));
		return true;
	}
	if (IsImmediate(value, Type::F16) && inst.GetOpcode() == ValueOpcode::BitCastU16F16) {
		Replace(inst, Value(value.F16Bits()));
		return true;
	}
	if (IsImmediate(value, Type::U16) && inst.GetOpcode() == ValueOpcode::BitCastF16U16) {
		Replace(inst, Value::F16(value.U16()));
		return true;
	}
	if (auto* producer = value.TryInstruction();
	    producer != nullptr && producer->GetOpcode() == reverse) {
		Replace(inst, producer->Arg(0));
		return true;
	}
	return false;
}

bool FoldCompositeExtract(Inst& inst, ValueOpcode construct, size_t components) {
	const auto composite = Arg(inst, 0);
	const auto index     = Arg(inst, 1);
	if (!IsImmediate(index, Type::U32) || index.U32() >= components) {
		return false;
	}
	const auto  component = index.U32();
	const auto* producer  = composite.TryInstruction();
	if (IsImmediate(composite, Type::U64)) {
		Replace(inst, Value(static_cast<uint32_t>(composite.U64() >> (component * 32u))));
		return true;
	}
	if (producer != nullptr && producer->GetOpcode() == construct) {
		Replace(inst, producer->Arg(component));
		return true;
	}
	if (component < 2u && producer != nullptr &&
	    producer->GetOpcode() == ValueOpcode::IAddCarry32) {
		const auto lhs = producer->Arg(0).Resolve();
		const auto rhs = producer->Arg(1).Resolve();
		if (IsImmediate(lhs, Type::U32) && IsImmediate(rhs, Type::U32)) {
			const auto sum = static_cast<uint64_t>(lhs.U32()) + rhs.U32();
			Replace(inst, Value(component == 0u ? static_cast<uint32_t>(sum)
			                                    : static_cast<uint32_t>(sum >> 32u)));
			return true;
		}
	}
	return false;
}

void FoldInstruction(Block& block, Block::iterator instruction,
                     std::unordered_set<Inst*>& lowered_ancillary) {
	auto& inst = *instruction;
	switch (inst.GetOpcode()) {
		case ValueOpcode::BitReverse32: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U32)) {
				auto source = value.U32();
				uint32_t reversed = 0;
				for (uint32_t bit = 0; bit < 32; ++bit) {
					reversed = (reversed << 1u) | (source & 1u);
					source >>= 1u;
				}
				Replace(inst, Value(reversed));
			}
			return;
		}
		case ValueOpcode::Phi: FoldPhi(inst); return;
		case ValueOpcode::SelectU1:
			if (!FoldSelect(inst) && IsImmediate(Arg(inst, 2), Type::U1) && !Arg(inst, 2).U1()) {
				auto result = block.PrependNewInst(instruction, ValueOpcode::LogicalAnd,
				                                   {Arg(inst, 0), Arg(inst, 1)});
				Replace(inst, Value(&*result));
			}
			return;
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32: FoldSelect(inst); return;
		case ValueOpcode::BitFieldInsert: {
			const auto base   = Arg(inst, 0);
			const auto insert = Arg(inst, 1);
			const auto offset = Arg(inst, 2);
			const auto count  = Arg(inst, 3);
			if (IsImmediate(base, Type::U32) && IsImmediate(insert, Type::U32) &&
			    IsImmediate(offset, Type::U32) && IsImmediate(count, Type::U32) &&
			    offset.U32() <= 32u && count.U32() <= 32u - offset.U32()) {
				if (count.U32() == 0u) {
					Replace(inst, base);
					return;
				}
				const auto mask = count.U32() == 32u
				                      ? UINT32_MAX
				                      : ((uint32_t {1} << count.U32()) - 1u) << offset.U32();
				Replace(inst,
				        Value((base.U32() & ~mask) | ((insert.U32() << offset.U32()) & mask)));
			}
			return;
		}
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract: {
			const auto value  = Arg(inst, 0);
			const auto offset = Arg(inst, 1);
			const auto count  = Arg(inst, 2);
			auto*      source = value.TryInstruction();
			if (source != nullptr && source->GetOpcode() == ValueOpcode::ShiftLeftLogical32 &&
			    IsImmediate(offset, Type::U32) && IsImmediate(count, Type::U32)) {
				const auto shift = Arg(*source, 1);
				if (IsImmediate(shift, Type::U32) && shift.U32() < 32u &&
				    offset.U32() <= shift.U32() && count.U32() <= shift.U32() - offset.U32()) {
					Replace(inst, Value(0u));
					return;
				}
			}
			if (source != nullptr && source->GetOpcode() == ValueOpcode::GetBuiltin &&
			    source->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::PackedAncillary)) &&
			    IsImmediate(offset, Type::U32) && IsImmediate(count, Type::U32) &&
			    count.U32() != 0u) {
				constexpr struct {
					uint32_t       start;
					uint32_t       end;
					StageInputKind kind;
				} fields[] = {{8u, 12u, StageInputKind::SampleId},
				              {16u, 27u, StageInputKind::Layer}};
				for (const auto& field: fields) {
					if (offset.U32() >= field.start && offset.U32() < field.end &&
					    count.U32() <= field.end - offset.U32()) {
						// Preserve extraction and sign extension while exposing only the used
						// field.
						const auto input = block.PrependNewInst(
						    instruction, ValueOpcode::GetBuiltin,
						    {Value(static_cast<uint32_t>(field.kind)), Value(0u)});
						inst.SetArg(0, Value(&*input));
						inst.SetArg(1, Value(offset.U32() - field.start));
						lowered_ancillary.insert(source);
						return;
					}
				}
			}
			if (!IsImmediate(value, Type::U32) || !IsImmediate(offset, Type::U32) ||
			    !IsImmediate(count, Type::U32) || offset.U32() > 32u ||
			    count.U32() > 32u - offset.U32()) {
				return;
			}
			if (count.U32() == 0u) {
				Replace(inst, Value(0u));
			} else if (inst.GetOpcode() == ValueOpcode::BitFieldUExtract) {
				const auto mask =
				    count.U32() == 32u ? UINT32_MAX : (uint32_t {1} << count.U32()) - 1u;
				Replace(inst, Value((value.U32() >> offset.U32()) & mask));
			} else {
				const auto left = 32u - offset.U32() - count.U32();
				const auto bits = value.U32() << left;
				Replace(inst, Value(static_cast<uint32_t>(std::bit_cast<int32_t>(bits) >>
				                                          (left + offset.U32()))));
			}
			return;
		}
		case ValueOpcode::BitCastU16F16: FoldBitCast(inst, ValueOpcode::BitCastF16U16); return;
		case ValueOpcode::BitCastF16U16: FoldBitCast(inst, ValueOpcode::BitCastU16F16); return;
		case ValueOpcode::BitCastU32F32: FoldBitCast(inst, ValueOpcode::BitCastF32U32); return;
		case ValueOpcode::BitCastF32U32: FoldBitCast(inst, ValueOpcode::BitCastU32F32); return;
		case ValueOpcode::ConvertU16U32: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U32)) {
				Replace(inst, Value(static_cast<uint16_t>(value.U32())));
			}
			return;
		}
		case ValueOpcode::ConvertU32U16: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U16)) {
				Replace(inst, Value(static_cast<uint32_t>(value.U16())));
			} else if (auto* producer = value.TryInstruction();
			           producer != nullptr && producer->GetOpcode() == ValueOpcode::ConvertU16U32) {
				Replace(inst, producer->Arg(0));
			}
			return;
		}
		case ValueOpcode::ConvertU8U32: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U32)) {
				Replace(inst, Value(static_cast<uint8_t>(value.U32())));
			}
			return;
		}
		case ValueOpcode::ConvertU32U8: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U8)) {
				Replace(inst, Value(static_cast<uint32_t>(value.U8())));
			} else if (auto* producer = value.TryInstruction();
			           producer != nullptr && producer->GetOpcode() == ValueOpcode::ConvertU8U32) {
				Replace(inst, producer->Arg(0));
			}
			return;
		}
		case ValueOpcode::CompositeExtractU64:
			FoldCompositeExtract(inst, ValueOpcode::CompositeConstructU64, 2);
			return;
		case ValueOpcode::CompositeExtractU32x2:
			FoldCompositeExtract(inst, ValueOpcode::CompositeConstructU32x2, 2);
			return;
		case ValueOpcode::CompositeExtractU32x3:
			FoldCompositeExtract(inst, ValueOpcode::CompositeConstructU32x3, 3);
			return;
		case ValueOpcode::CompositeExtractU32x4:
			FoldCompositeExtract(inst, ValueOpcode::CompositeConstructU32x4, 4);
			return;
		case ValueOpcode::CompositeConstructU64: {
			const auto low  = Arg(inst, 0);
			const auto high = Arg(inst, 1);
			if (IsImmediate(low, Type::U32) && IsImmediate(high, Type::U32)) {
				Replace(inst, Value(static_cast<uint64_t>(low.U32()) |
				                    (static_cast<uint64_t>(high.U32()) << 32u)));
			}
			return;
		}
		case ValueOpcode::IAdd32:
			if (!FoldU32(inst, [](uint32_t a, uint32_t b) { return a + b; })) {
				ReplaceBinaryIdentity(inst, Type::U32, 0u);
			}
			return;
		case ValueOpcode::IAdd64:
			if (!FoldU64(inst, [](uint64_t a, uint64_t b) { return a + b; })) {
				ReplaceBinaryIdentity(inst, Type::U64, 0u);
			}
			return;
		case ValueOpcode::ISub32: {
			if (FoldU32(inst, [](uint32_t a, uint32_t b) { return a - b; })) {
				return;
			}
			const auto rhs = Arg(inst, 1);
			if (IsImmediate(rhs, Type::U32) && rhs.U32() == 0u) {
				Replace(inst, Arg(inst, 0));
			}
			return;
		}
		case ValueOpcode::ISub64: {
			if (FoldU64(inst, [](uint64_t a, uint64_t b) { return a - b; })) {
				return;
			}
			const auto rhs = Arg(inst, 1);
			if (IsImmediate(rhs, Type::U64) && rhs.U64() == 0u) {
				Replace(inst, Arg(inst, 0));
			}
			return;
		}
		case ValueOpcode::IMul32:
			if (!FoldU32(inst, [](uint32_t a, uint32_t b) { return a * b; })) {
				ReplaceBinaryIdentity(inst, Type::U32, 1u);
			}
			return;
		case ValueOpcode::IMul64:
			if (!FoldU64(inst, [](uint64_t a, uint64_t b) { return a * b; })) {
				ReplaceBinaryIdentity(inst, Type::U64, 1u);
			}
			return;
		case ValueOpcode::WqmU64: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U64)) {
				const auto expand = [](uint32_t word) {
					auto quads = word | (word >> 1u);
					quads |= quads >> 2u;
					return (quads & 0x11111111u) * 0x0fu;
				};
				const auto low  = expand(static_cast<uint32_t>(value.U64()));
				const auto high = expand(static_cast<uint32_t>(value.U64() >> 32u));
				Replace(inst,
				        Value(static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32u)));
			}
			return;
		}
		case ValueOpcode::UDiv32: {
			const auto rhs = Arg(inst, 1);
			if (IsImmediate(rhs, Type::U32) && rhs.U32() == 1u) {
				Replace(inst, Arg(inst, 0));
			} else if (IsImmediate(rhs, Type::U32) && rhs.U32() != 0u) {
				FoldU32(inst, [](uint32_t a, uint32_t b) { return a / b; });
			}
			return;
		}
		case ValueOpcode::SMulHi:
			FoldU32(inst, [](uint32_t a, uint32_t b) {
				const auto product = static_cast<int64_t>(std::bit_cast<int32_t>(a)) *
				                     static_cast<int64_t>(std::bit_cast<int32_t>(b));
				return static_cast<uint32_t>(static_cast<uint64_t>(product) >> 32u);
			});
			return;
		case ValueOpcode::UMulHi:
			FoldU32(inst, [](uint32_t a, uint32_t b) {
				return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32u);
			});
			return;
		case ValueOpcode::IAbs32: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U32)) {
				Replace(inst,
				        Value((value.U32() & 0x80000000u) != 0u ? 0u - value.U32() : value.U32()));
			}
			return;
		}
		case ValueOpcode::ShiftLeftLogical32:
			if (!FoldU32(inst, [](uint32_t a, uint32_t b) { return a << (b & 31u); })) {
				const auto shift = Arg(inst, 1);
				if (IsImmediate(shift, Type::U32) && (shift.U32() & 31u) == 0u) {
					Replace(inst, Arg(inst, 0));
				}
			}
			return;
		case ValueOpcode::ShiftRightLogical32:
			if (!FoldU32(inst, [](uint32_t a, uint32_t b) { return a >> (b & 31u); })) {
				if (IsImmediate(Arg(inst, 0), Type::U32) && Arg(inst, 0).U32() == 0u) {
					Replace(inst, Value(0u));
					return;
				}
				const auto shift = Arg(inst, 1);
				if (IsImmediate(shift, Type::U32) && (shift.U32() & 31u) == 0u) {
					Replace(inst, Arg(inst, 0));
				}
			}
			return;
		case ValueOpcode::ShiftRightArithmetic32:
			FoldU32(inst, [](uint32_t a, uint32_t b) {
				return static_cast<uint32_t>(std::bit_cast<int32_t>(a) >> (b & 31u));
			});
			return;
		case ValueOpcode::ShiftLeftLogical64:
			if (!FoldU64Shift(inst, [](uint64_t a, uint32_t b) { return a << (b & 63u); })) {
				const auto shift = Arg(inst, 1);
				if (IsImmediate(shift, Type::U32) && (shift.U32() & 63u) == 0u) {
					Replace(inst, Arg(inst, 0));
				}
			}
			return;
		case ValueOpcode::ShiftRightLogical64:
			if (!FoldU64Shift(inst, [](uint64_t a, uint32_t b) { return a >> (b & 63u); })) {
				const auto shift = Arg(inst, 1);
				if (IsImmediate(shift, Type::U32) && (shift.U32() & 63u) == 0u) {
					Replace(inst, Arg(inst, 0));
				}
			}
			return;
		case ValueOpcode::ShiftRightArithmetic64:
			FoldU64Shift(inst, [](uint64_t a, uint32_t b) {
				return static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
			});
			return;
		case ValueOpcode::BitwiseAnd32:
			if (!FoldU32(inst, [](uint32_t a, uint32_t b) { return a & b; })) {
				ReplaceBinaryIdentity(inst, Type::U32, 0xffffffffu);
			}
			return;
		case ValueOpcode::BitwiseAnd64:
			if (!FoldU64(inst, [](uint64_t a, uint64_t b) { return a & b; })) {
				ReplaceBinaryIdentity(inst, Type::U64, UINT64_MAX);
			}
			return;
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
			if (!FoldU32(inst, [opcode = inst.GetOpcode()](uint32_t a, uint32_t b) {
				    return opcode == ValueOpcode::BitwiseOr32 ? a | b : a ^ b;
			    })) {
				ReplaceBinaryIdentity(inst, Type::U32, 0u);
			}
			return;
		case ValueOpcode::BitwiseNot32: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U32)) {
				Replace(inst, Value(~value.U32()));
			} else if (auto* producer = value.TryInstruction();
			           producer != nullptr && producer->GetOpcode() == ValueOpcode::BitwiseNot32) {
				Replace(inst, producer->Arg(0));
			}
			return;
		}
		case ValueOpcode::BitCount32: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U32)) {
				Replace(inst, Value(static_cast<uint32_t>(std::popcount(value.U32()))));
			}
			return;
		}
		case ValueOpcode::BitCount64: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U64)) {
				Replace(inst, Value(static_cast<uint32_t>(std::popcount(value.U64()))));
			}
			return;
		}
		case ValueOpcode::SMin32:
		case ValueOpcode::SMax32:
			FoldU32(inst, [opcode = inst.GetOpcode()](uint32_t a, uint32_t b) {
				const auto lhs = std::bit_cast<int32_t>(a);
				const auto rhs = std::bit_cast<int32_t>(b);
				return opcode == ValueOpcode::SMin32 ? (lhs < rhs ? a : b) : (lhs > rhs ? a : b);
			});
			return;
		case ValueOpcode::UMin32:
			FoldU32(inst, [](uint32_t a, uint32_t b) { return std::min(a, b); });
			return;
		case ValueOpcode::UMax32:
			FoldU32(inst, [](uint32_t a, uint32_t b) { return std::max(a, b); });
			return;
		case ValueOpcode::IEqual32:
			FoldU32Compare(inst, [](uint32_t a, uint32_t b) { return a == b; });
			return;
		case ValueOpcode::INotEqual32:
			FoldU32Compare(inst, [](uint32_t a, uint32_t b) { return a != b; });
			return;
		case ValueOpcode::ULessThan32:
			FoldU32Compare(inst, [](uint32_t a, uint32_t b) { return a < b; });
			return;
		case ValueOpcode::ULessThanEqual32:
			FoldU32Compare(inst, [](uint32_t a, uint32_t b) { return a <= b; });
			return;
		case ValueOpcode::UGreaterThan32:
			FoldU32Compare(inst, [](uint32_t a, uint32_t b) { return a > b; });
			return;
		case ValueOpcode::UGreaterThanEqual32:
			FoldU32Compare(inst, [](uint32_t a, uint32_t b) { return a >= b; });
			return;
		case ValueOpcode::SLessThan32:
		case ValueOpcode::SLessThanEqual32:
		case ValueOpcode::SGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
			FoldU32Compare(inst, [opcode = inst.GetOpcode()](uint32_t a, uint32_t b) {
				const auto lhs = std::bit_cast<int32_t>(a);
				const auto rhs = std::bit_cast<int32_t>(b);
				switch (opcode) {
					case ValueOpcode::SLessThan32: return lhs < rhs;
					case ValueOpcode::SLessThanEqual32: return lhs <= rhs;
					case ValueOpcode::SGreaterThan32: return lhs > rhs;
					default: return lhs >= rhs;
				}
			});
			return;
		case ValueOpcode::IEqual64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) { return a == b; });
			return;
		case ValueOpcode::INotEqual64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) { return a != b; });
			return;
		case ValueOpcode::ULessThan64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) { return a < b; });
			return;
		case ValueOpcode::UGreaterThan64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) { return a > b; });
			return;
		case ValueOpcode::ULessThanEqual64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) { return a <= b; });
			return;
		case ValueOpcode::UGreaterThanEqual64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) { return a >= b; });
			return;
		case ValueOpcode::SLessThanEqual64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) {
				return std::bit_cast<int64_t>(a) <= std::bit_cast<int64_t>(b);
			});
			return;
		case ValueOpcode::SLessThan64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) {
				return std::bit_cast<int64_t>(a) < std::bit_cast<int64_t>(b);
			});
			return;
		case ValueOpcode::LogicalAnd:
			if (!FoldLogical(inst, [](bool a, bool b) { return a && b; })) {
				const auto lhs      = Arg(inst, 0);
				const auto rhs      = Arg(inst, 1);
				if (IsComplement(lhs, rhs)) {
					Replace(inst, Value(false));
					return;
				}
				const auto simplify = [&](Value assumption, Value expression) {
					const auto* disjunction = expression.TryInstruction();
					if (disjunction == nullptr ||
					    disjunction->GetOpcode() != ValueOpcode::LogicalOr) {
						return false;
					}
					for (uint32_t i = 0; i < 2u; ++i) {
						const auto* inverse = disjunction->Arg(i).Resolve().TryInstruction();
						if (inverse != nullptr && inverse->GetOpcode() == ValueOpcode::LogicalNot &&
						    inverse->Arg(0).Resolve() == assumption) {
							inst.SetArg(assumption == lhs ? 1u : 0u, disjunction->Arg(i ^ 1u));
							return true;
						}
					}
					return false;
				};
				if (simplify(lhs, rhs) || simplify(rhs, lhs)) return;
				if (IsImmediate(lhs, Type::U1)) {
					Replace(inst, lhs.U1() ? rhs : lhs);
				} else if (IsImmediate(rhs, Type::U1)) {
					Replace(inst, rhs.U1() ? lhs : rhs);
				}
			}
			return;
		case ValueOpcode::LogicalOr:
			if (!FoldLogical(inst, [](bool a, bool b) { return a || b; })) {
				const auto lhs = Arg(inst, 0);
				const auto rhs = Arg(inst, 1);
				if (IsComplement(lhs, rhs)) {
					Replace(inst, Value(true));
				} else if (IsImmediate(lhs, Type::U1)) {
					Replace(inst, lhs.U1() ? lhs : rhs);
				} else if (IsImmediate(rhs, Type::U1)) {
					Replace(inst, rhs.U1() ? rhs : lhs);
				}
			}
			return;
		case ValueOpcode::LogicalXor:
			if (!FoldLogical(inst, [](bool a, bool b) { return a != b; })) {
				const auto lhs = Arg(inst, 0);
				const auto rhs = Arg(inst, 1);
				if (IsImmediate(lhs, Type::U1) && !lhs.U1()) {
					Replace(inst, rhs);
				} else if (IsImmediate(rhs, Type::U1) && !rhs.U1()) {
					Replace(inst, lhs);
				}
			}
			return;
		case ValueOpcode::ConditionRef: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U1)) Replace(inst, value);
			return;
		}
		case ValueOpcode::LogicalNot: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U1)) {
				Replace(inst, Value(!value.U1()));
			} else if (auto* producer = value.TryInstruction();
			           producer != nullptr && producer->GetOpcode() == ValueOpcode::LogicalNot) {
				Replace(inst, producer->Arg(0));
			}
			return;
		}
		default: return;
	}
}

// In wave32, extracting this invocation's bit from a ballot recovers its predicate.
// Keep that identity through scalar EXEC operations and their loop-carried mask Phis.
class LaneMaskProjection {
public:
	explicit LaneMaskProjection(std::unordered_set<Inst*>& lowered_ancillary)
	    : m_lowered_ancillary(lowered_ancillary) {}

	void Fold(Inst& inst) {
		if (inst.GetOpcode() != ValueOpcode::INotEqual32 || !Immediate(Arg(inst, 1), 0u)) return;
		const auto* bit = Arg(inst, 0).TryInstruction();
		if (bit == nullptr || bit->GetOpcode() != ValueOpcode::BitwiseAnd32 ||
		    !Immediate(Arg(*bit, 1), 1u)) return;
		const auto* shift = Arg(*bit, 0).TryInstruction();
		if (shift == nullptr || shift->GetOpcode() != ValueOpcode::ShiftRightLogical32) return;
		const auto* index = Arg(*shift, 1).TryInstruction();
		if (index == nullptr || index->GetOpcode() != ValueOpcode::BitwiseAnd32 ||
		    !Immediate(Arg(*index, 1), 31u)) return;
		const auto* lane = Arg(*index, 0).TryInstruction();
		if (lane == nullptr || lane->GetOpcode() != ValueOpcode::LaneId) return;
		m_visited.clear();
		m_grounded = false;
		if (CanProject(Arg(*shift, 0)) && m_grounded) Replace(inst, Project(Arg(*shift, 0)));
	}

private:
	static bool Immediate(Value value, uint32_t expected) {
		return IsImmediate(value, Type::U32) && value.U32() == expected;
	}

	static Value BallotPredicate(const Inst& inst) {
		if (inst.GetOpcode() != ValueOpcode::CompositeExtractU32x4 ||
		    !Immediate(Arg(inst, 1), 0u)) return {};
		const auto* source = Arg(inst, 0).TryInstruction();
		return source != nullptr && source->GetOpcode() == ValueOpcode::Ballot
		           ? Arg(*source, 0) : Value {};
	}

	bool CanProject(Value value) {
		if (value.IsImmediate()) {
			const bool supported = Immediate(value, 0u) || Immediate(value, UINT32_MAX);
			m_grounded |= supported;
			return supported;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return false;
		if (m_values.contains(inst) || !BallotPredicate(*inst).IsEmpty()) {
			m_grounded = true;
			return true;
		}
		if (!m_visited.insert(inst).second) return true;
		const auto op = inst->GetOpcode();
		if (op != ValueOpcode::BitwiseAnd32 && op != ValueOpcode::BitwiseOr32 &&
		    op != ValueOpcode::BitwiseNot32 && op != ValueOpcode::SelectU32 && op != ValueOpcode::Phi)
			return false;
		for (size_t arg = op == ValueOpcode::SelectU32 ? 1u : 0u; arg < inst->NumArgs(); ++arg) {
			if (!CanProject(Arg(*inst, arg))) return false;
		}
		return inst->NumArgs() != 0;
	}

	Value Project(Value value) {
		if (value.IsImmediate()) return Value(value.U32() != 0u);
		auto* source = value.TryInstruction();
		if (const auto found = m_values.find(source); found != m_values.end()) return found->second.Resolve();
		if (const auto predicate = BallotPredicate(*source); !predicate.IsEmpty()) return predicate;
		auto* block = source->Parent();
		const auto where = std::find_if(block->begin(), block->end(),
		                               [source](const Inst& candidate) { return &candidate == source; });
		const auto op = source->GetOpcode() == ValueOpcode::BitwiseAnd32 ? ValueOpcode::LogicalAnd
		              : source->GetOpcode() == ValueOpcode::BitwiseOr32 ? ValueOpcode::LogicalOr
		              : source->GetOpcode() == ValueOpcode::BitwiseNot32 ? ValueOpcode::LogicalNot
		              : source->GetOpcode() == ValueOpcode::SelectU32 ? ValueOpcode::SelectU1
		                                                               : ValueOpcode::Phi;
		auto result = op == ValueOpcode::Phi ? block->PrependNewInst(where, op)
		            : op == ValueOpcode::LogicalNot
		                ? block->PrependNewInst(where, op, {Value(false)})
		            : op == ValueOpcode::SelectU1
		                ? block->PrependNewInst(where, op, {Arg(*source, 0), Value(false), Value(false)})
		                : block->PrependNewInst(where, op, {Value(false), Value(false)});
		m_values.emplace(source, Value(&*result));
		if (op == ValueOpcode::Phi) result->SetFlags(Type::U1);
		for (size_t arg = op == ValueOpcode::SelectU1 ? 1u : 0u; arg < source->NumArgs(); ++arg) {
			const auto projected = Project(Arg(*source, arg));
			if (op == ValueOpcode::Phi) result->AddPhiOperand(source->PhiBlock(arg), projected);
			else result->SetArg(arg, projected);
		}
		FoldInstruction(*block, result, m_lowered_ancillary);
		return Value(&*result).Resolve();
	}

	std::unordered_set<Inst*>& m_lowered_ancillary;
	std::unordered_set<const Inst*> m_visited;
	std::unordered_map<const Inst*, Value> m_values;
	bool m_grounded = false;
};

} // namespace

uint32_t SimplifyBoundedLoopRegisters(Program& program) {
	// Prove a simple loop inductively before narrowing its relative-register indices.
	// The scalar counter starts at zero and advances unconditionally by one. A
	// possibly lane-varying limit is safe if every lane preserves its upper bound.
	struct Range {
		uint32_t lo = 0, hi = UINT32_MAX;
	};
	struct Evaluator {
		const Inst*                     counter = nullptr;
		const Inst*                     limit   = nullptr;
		uint32_t                        bound   = 0;
		std::unordered_set<const Inst*> visiting;
		Range                           Get(Value value) {
			value = value.Resolve();
			if (value.IsImmediate()) {
				if (value.GetType() == Type::U32) return {value.U32(), value.U32()};
				if (value.GetType() == Type::U1) return {value.U1(), value.U1()};
				return {};
			}
			const auto* inst = value.TryInstruction();
			if (inst == nullptr) return {};
			if (inst == counter) return {0, bound - 1u};
			if (inst == limit) return {0, bound};
			if (!visiting.insert(inst).second) return {};
			const auto result = Compute(*inst);
			visiting.erase(inst);
			return result;
		}
		Range Compute(const Inst& inst) {
			const auto op = inst.GetOpcode();
			if (op == ValueOpcode::SelectU32 || op == ValueOpcode::SelectU1) {
				const auto condition = Get(inst.Arg(0));
				if (condition.hi == 0) return Get(inst.Arg(2));
				if (condition.lo != 0) return Get(inst.Arg(1));
				const auto a = Get(inst.Arg(1)), b = Get(inst.Arg(2));
				return {std::min(a.lo, b.lo), std::max(a.hi, b.hi)};
			}
			if (op == ValueOpcode::ReadFirstLane) return Get(inst.Arg(0));
			if (inst.NumArgs() != 2) return {};
			// Unknown operations are not traversed; in particular, no memory is read.
			if (op != ValueOpcode::IAdd32 && op != ValueOpcode::BitwiseAnd32 &&
			    op != ValueOpcode::ShiftLeftLogical32 && op != ValueOpcode::ShiftRightLogical32 &&
			    op != ValueOpcode::IEqual32 && op != ValueOpcode::ULessThan32 &&
			    op != ValueOpcode::LogicalAnd)
				return {};
			const auto a = Get(inst.Arg(0)), b = Get(inst.Arg(1));
			switch (op) {
				case ValueOpcode::IAdd32:
					if (uint64_t {a.hi} + b.hi <= UINT32_MAX) return {a.lo + b.lo, a.hi + b.hi};
					break;
				case ValueOpcode::BitwiseAnd32: return {0, std::min(a.hi, b.hi)};
				case ValueOpcode::ShiftLeftLogical32:
					if (b.lo == b.hi && (uint64_t {a.hi} << (b.lo & 31u)) <= UINT32_MAX)
						return {a.lo << (b.lo & 31u), a.hi << (b.lo & 31u)};
					break;
				case ValueOpcode::ShiftRightLogical32:
					if (b.lo == b.hi) return {a.lo >> (b.lo & 31u), a.hi >> (b.lo & 31u)};
					break;
				case ValueOpcode::IEqual32:
					if (a.hi < b.lo || b.hi < a.lo) return {0, 0};
					if (a.lo == a.hi && a.lo == b.lo && b.lo == b.hi) return {1, 1};
					return {0, 1};
				case ValueOpcode::ULessThan32:
					if (a.hi < b.lo) return {1, 1};
					if (a.lo >= b.hi) return {0, 0};
					return {0, 1};
				case ValueOpcode::LogicalAnd:
					if (a.hi == 0 || b.hi == 0) return {0, 0};
					return {a.lo != 0 && b.lo != 0 ? 1u : 0u, 1};
				default: break;
			}
			return {};
		}
	};
	std::unordered_map<uint32_t, Block*>               by_id;
	std::unordered_map<const Block*, const BlockInfo*> info;
	for (size_t i = 0; i < program.blocks.size(); i++) {
		by_id.emplace(program.block_info[i].id, program.blocks[i]);
		info.emplace(program.blocks[i], &program.block_info[i]);
	}
	uint32_t folded = 0;
	for (auto* guard: program.blocks) {
		const auto& control = *info.at(guard);
		if (control.terminator.kind != CFG::TerminatorKind::ConditionalBranch) continue;
		auto condition = control.condition.Resolve();
		bool inverse   = false;
		// A branch condition is wrapped in ConditionRef, which marks its kind and passes the lane
		// value through: the proof only needs the counter below some lane's limit in the body.
		while (auto* node = condition.TryInstruction()) {
			if (node->GetOpcode() == ValueOpcode::LogicalNot) {
				inverse = !inverse;
			} else if (node->GetOpcode() != ValueOpcode::ConditionRef) {
				break;
			}
			condition = node->Arg(0).Resolve();
		}
		auto* compare = condition.TryInstruction();
		if (compare && compare->GetOpcode() == ValueOpcode::LogicalAnd) {
			for (size_t i = 0; i < 2; i++) {
				auto* candidate = compare->Arg(i).Resolve().TryInstruction();
				if (candidate && candidate->GetOpcode() == ValueOpcode::ULessThan32) {
					compare = candidate;
					break;
				}
			}
		}
		if (!compare || compare->GetOpcode() != ValueOpcode::ULessThan32) continue;
		auto* counter = compare->Arg(0).Resolve().TryInstruction();
		auto* limit   = compare->Arg(1).Resolve().TryInstruction();
		if (!counter || !limit || counter->GetOpcode() != ValueOpcode::Phi ||
		    limit->GetOpcode() != ValueOpcode::Phi || counter->NumArgs() != 2 ||
		    limit->NumArgs() != 2 || counter->Parent() != limit->Parent())
			continue;
		auto*      header = counter->Parent();
		const auto body_id =
		    inverse ? control.terminator.false_block : control.terminator.true_block;
		if (!by_id.contains(body_id)) continue;
		auto* body = by_id.at(body_id);
		if (body == header || body == guard || body->ImmPredecessors().size() != 1 ||
		    body->ImmPredecessors()[0] != guard || header->ImmPredecessors().size() != 2)
			continue;
		const auto& latch = info.at(body)->terminator;
		if (latch.kind != CFG::TerminatorKind::Branch || latch.true_block != info.at(header)->id)
			continue;
		if (header != guard) {
			const auto& entry = info.at(header)->terminator;
			if (entry.kind != CFG::TerminatorKind::Branch || entry.true_block != control.id ||
			    guard->ImmPredecessors().size() != 1 || guard->ImmPredecessors()[0] != header)
				continue;
		}
		const auto back = counter->PhiBlock(0) == body ? 0u : 1u;
		if (counter->PhiBlock(back) != body) continue;
		const auto start     = counter->Arg(back ^ 1u).Resolve();
		auto*      increment = counter->Arg(back).Resolve().TryInstruction();
		if (!IsImmediate(start, Type::U32) || start.U32() != 0 || !increment ||
		    increment->Parent() != body || increment->GetOpcode() != ValueOpcode::IAdd32)
			continue;
		const auto one = increment->Arg(1).Resolve();
		if (increment->Arg(0).Resolve().TryInstruction() != counter || !IsImmediate(one, Type::U32) ||
		    one.U32() != 1)
			continue;
		const auto limit_back = limit->PhiBlock(0) == body ? 0u : 1u;
		if (limit->PhiBlock(limit_back) != body ||
		    limit->PhiBlock(limit_back ^ 1u) != counter->PhiBlock(back ^ 1u))
			continue;
		Evaluator  initial;
		const auto bound = initial.Get(limit->Arg(limit_back ^ 1u)).hi;
		if (bound == 0 || bound == UINT32_MAX) continue;
		Evaluator ranges {counter, limit, bound};
		// This is the induction step: indirect writes must preserve the proposed
		// limit bound. If they can reach that register, the proof fails unchanged.
		if (ranges.Get(limit->Arg(limit_back)).hi > bound) continue;
		for (auto& inst: *body) {
			if (inst.GetOpcode() != ValueOpcode::IEqual32) continue;
			const auto range = ranges.Get(Value(&inst));
			if (range.lo == range.hi && range.hi <= 1) {
				Replace(inst, Value(range.lo != 0));
				folded++;
			}
		}
	}
	return folded;
}

namespace {

// Translator::ThreadBit reads the current lane's bit of a wave mask as
//   INotEqual32(BitwiseAnd32(ShiftRightLogical32(word, BitwiseAnd32(LaneId, 31)), 1), 0)
// where a wave64 word is SelectU32(ULessThan32(LaneId, 32), low, high).
struct LaneBitRead {
	Value low;
	Value high;
	Value high_half; // the ULessThan32 of a wave64 read; empty for wave32
};

bool IsOpcode(Value value, ValueOpcode opcode) {
	const auto* inst = value.TryInstruction();
	return inst != nullptr && inst->GetOpcode() == opcode;
}

// The operand of a commutative instruction that is not the immediate `constant`.
std::optional<Value> OtherThan(const Inst& inst, uint32_t constant) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (IsImmediate(rhs, Type::U32) && rhs.U32() == constant) {
		return lhs;
	}
	if (IsImmediate(lhs, Type::U32) && lhs.U32() == constant) {
		return rhs;
	}
	return std::nullopt;
}

bool IsLaneBitIndex(Value value) {
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || inst->GetOpcode() != ValueOpcode::BitwiseAnd32) {
		return false;
	}
	const auto lane = OtherThan(*inst, 31u);
	return lane && IsOpcode(*lane, ValueOpcode::LaneId);
}

std::optional<LaneBitRead> MatchLaneBit(const Inst& inst) {
	const auto masked = OtherThan(inst, 0u);
	if (!masked || !IsOpcode(*masked, ValueOpcode::BitwiseAnd32)) {
		return std::nullopt;
	}
	const auto shifted = OtherThan(*masked->TryInstruction(), 1u);
	if (!shifted || !IsOpcode(*shifted, ValueOpcode::ShiftRightLogical32)) {
		return std::nullopt;
	}
	const auto& shift = *shifted->TryInstruction();
	if (!IsLaneBitIndex(Arg(shift, 1))) {
		return std::nullopt;
	}
	const auto word = Arg(shift, 0);
	if (const auto* select = word.TryInstruction();
	    select != nullptr && select->GetOpcode() == ValueOpcode::SelectU32) {
		const auto condition = Arg(*select, 0);
		const auto* compare  = condition.TryInstruction();
		if (compare != nullptr && compare->GetOpcode() == ValueOpcode::ULessThan32 &&
		    IsOpcode(Arg(*compare, 0), ValueOpcode::LaneId) &&
		    IsImmediate(Arg(*compare, 1), Type::U32) && Arg(*compare, 1).U32() == 32u) {
			return LaneBitRead {Arg(*select, 1), Arg(*select, 2), condition};
		}
	}
	return LaneBitRead {word, Value {}, Value {}};
}

// Word `part` of a U64 built by CompositeConstructU64, or of a U64 constant.
std::optional<Value> U64Word(Value value, uint32_t part) {
	if (IsImmediate(value, Type::U64)) {
		return Value(static_cast<uint32_t>(value.U64() >> (part * 32u)));
	}
	const auto* inst = value.TryInstruction();
	if (inst != nullptr && inst->GetOpcode() == ValueOpcode::CompositeConstructU64) {
		return Arg(*inst, part);
	}
	return std::nullopt;
}

// What the fold needs to know about the program. In a pixel shader, helper invocations may be
// left out of ballots (NVIDIA leaves them out), so there bit LaneId of Ballot(c) is c only for a
// lane that takes part in ballots: c AND Participating(). Lanes keep that status for the whole
// shader (a discard ends the invocation, OpKill). Elsewhere every running lane takes part.
struct LaneFold {
	Program& program;
	bool     pixel = false;
	Value    participating; // made on first use, at the top of the entry block
	// Lane bits already emitted in the current block, by (word, part): the same word gives the
	// same value, so x || !x stays recognisable (IsEveryLane compares instructions).
	std::unordered_map<const Inst*, std::array<Value, 2>> emitted;
	const Block*                                           emitted_block = nullptr;

	Value Participating() {
		if (!participating.IsEmpty()) {
			return participating;
		}
		auto&      entry = *program.blocks.front();
		const auto at    = entry.begin();
		const auto add   = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
            return Value(&*entry.PrependNewInst(at, opcode, args));
		};
		const auto ballot = add(ValueOpcode::Ballot, {Value(true)});
		const auto lane   = add(ValueOpcode::LaneId, {});
		auto       word   = add(ValueOpcode::CompositeExtractU32x4, {ballot, Value(0u)});
		if (program.wave_size == 64u) {
			const auto high     = add(ValueOpcode::CompositeExtractU32x4, {ballot, Value(1u)});
			const auto low_half = add(ValueOpcode::ULessThan32, {lane, Value(32u)});
			word                = add(ValueOpcode::SelectU32, {low_half, word, high});
		}
		const auto bit     = add(ValueOpcode::BitwiseAnd32, {lane, Value(31u)});
		const auto shifted = add(ValueOpcode::ShiftRightLogical32, {word, bit});
		const auto masked  = add(ValueOpcode::BitwiseAnd32, {shifted, Value(1u)});
		// IEqual32(.., 1), not ThreadBit's INotEqual32(.., 0): FoldLaneMasks must not fold it.
		participating = add(ValueOpcode::IEqual32, {masked, Value(1u)});
		return participating;
	}
};

// Whether the current lane's bit of `word` is known to be set: an all-one constant, a ballot of
// true (outside pixel shaders; or under WQM, as a helper shares its quad with a lane that takes
// part), or logic or WQM of those. WQM only adds lanes, so it keeps a set bit set.
bool LaneBitSet(const LaneFold& fold, Value word, uint32_t part, int depth, bool under_wqm) {
	if (depth > 16) {
		return false;
	}
	if (IsImmediate(word, Type::U32)) {
		return word.U32() == UINT32_MAX;
	}
	const auto* inst = word.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	switch (inst->GetOpcode()) {
		case ValueOpcode::CompositeExtractU32x4: {
			const auto  index  = Arg(*inst, 1);
			const auto* ballot = Arg(*inst, 0).TryInstruction();
			return (!fold.pixel || under_wqm) && IsImmediate(index, Type::U32) &&
			       index.U32() == part && ballot != nullptr &&
			       ballot->GetOpcode() == ValueOpcode::Ballot &&
			       IsImmediate(Arg(*ballot, 0), Type::U1) && Arg(*ballot, 0).U1();
		}
		case ValueOpcode::CompositeExtractU64: {
			const auto index = Arg(*inst, 1);
			if (!IsImmediate(index, Type::U32) || index.U32() != part) {
				return false;
			}
			auto source = Arg(*inst, 0);
			bool wqm    = under_wqm;
			if (const auto* producer = source.TryInstruction();
			    producer != nullptr && producer->GetOpcode() == ValueOpcode::WqmU64) {
				source = Arg(*producer, 0);
				wqm    = true;
			}
			const auto component = U64Word(source, part);
			return component && LaneBitSet(fold, *component, part, depth + 1, wqm);
		}
		case ValueOpcode::BitwiseAnd32:
			return LaneBitSet(fold, Arg(*inst, 0), part, depth + 1, under_wqm) &&
			       LaneBitSet(fold, Arg(*inst, 1), part, depth + 1, under_wqm);
		case ValueOpcode::BitwiseOr32:
			return LaneBitSet(fold, Arg(*inst, 0), part, depth + 1, under_wqm) ||
			       LaneBitSet(fold, Arg(*inst, 1), part, depth + 1, under_wqm);
		default: return false;
	}
}

// Whether the current lane's bit of `word` (part 0: lanes 0-31, part 1: lanes 32-63) is known
// without the word: a ballot of a predicate, an all-zero or all-one constant, a bit known to be
// set (LaneBitSet), or bitwise and/or/xor/not of those.
bool LaneBitKnown(const LaneFold& fold, Value word, uint32_t part, int depth) {
	if (depth > 16) {
		return false;
	}
	if (IsImmediate(word, Type::U32)) {
		return word.U32() == 0u || word.U32() == UINT32_MAX;
	}
	if (LaneBitSet(fold, word, part, depth, false)) {
		return true;
	}
	const auto* inst = word.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	switch (inst->GetOpcode()) {
		case ValueOpcode::CompositeExtractU64: {
			const auto index = Arg(*inst, 1);
			if (!IsImmediate(index, Type::U32) || index.U32() != part) {
				return false;
			}
			const auto component = U64Word(Arg(*inst, 0), part);
			return component && LaneBitKnown(fold, *component, part, depth + 1);
		}
		case ValueOpcode::CompositeExtractU32x4: {
			const auto index = Arg(*inst, 1);
			return IsImmediate(index, Type::U32) && index.U32() == part &&
			       IsOpcode(Arg(*inst, 0), ValueOpcode::Ballot);
		}
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
			return LaneBitKnown(fold, Arg(*inst, 0), part, depth + 1) &&
			       LaneBitKnown(fold, Arg(*inst, 1), part, depth + 1);
		case ValueOpcode::BitwiseNot32: return LaneBitKnown(fold, Arg(*inst, 0), part, depth + 1);
		default: return false;
	}
}

Value EmitLaneBitUncached(LaneFold& fold, Value word, uint32_t part, Block& block,
                          Block::iterator at);

// Emits the current lane's bit of `word` before `at`; LaneBitKnown(word, part) must hold.
Value EmitLaneBit(LaneFold& fold, Value word, uint32_t part, Block& block, Block::iterator at) {
	if (IsImmediate(word, Type::U32)) {
		return Value(word.U32() != 0u);
	}
	if (fold.emitted_block != &block) {
		fold.emitted.clear();
		fold.emitted_block = &block;
	}
	const auto* key = word.TryInstruction();
	if (key != nullptr) {
		if (const auto found = fold.emitted.find(key);
		    found != fold.emitted.end() && !found->second[part].IsEmpty()) {
			return found->second[part];
		}
	}
	const auto value = EmitLaneBitUncached(fold, word, part, block, at);
	if (key != nullptr) {
		fold.emitted[key][part] = value;
	}
	return value;
}

Value EmitLaneBitUncached(LaneFold& fold, Value word, uint32_t part, Block& block,
                          Block::iterator at) {
	if (LaneBitSet(fold, word, part, 0, false)) {
		return Value(true);
	}
	const auto* inst   = word.TryInstruction();
	const auto  binary = [&](ValueOpcode opcode) {
		const auto lhs = EmitLaneBit(fold, Arg(*inst, 0), part, block, at);
		const auto rhs = EmitLaneBit(fold, Arg(*inst, 1), part, block, at);
		return Value(&*block.PrependNewInst(at, opcode, {lhs, rhs}));
	};
	switch (inst->GetOpcode()) {
		case ValueOpcode::CompositeExtractU64:
			return EmitLaneBit(fold, *U64Word(Arg(*inst, 0), part), part, block, at);
		case ValueOpcode::CompositeExtractU32x4: {
			// Bit LaneId of a ballot is the current lane's own predicate, if the lane takes part.
			const auto predicate = Arg(*Arg(*inst, 0).TryInstruction(), 0);
			if (!fold.pixel) {
				return predicate;
			}
			return Value(&*block.PrependNewInst(at, ValueOpcode::LogicalAnd,
			                                    {predicate, fold.Participating()}));
		}
		case ValueOpcode::BitwiseAnd32: return binary(ValueOpcode::LogicalAnd);
		case ValueOpcode::BitwiseOr32: return binary(ValueOpcode::LogicalOr);
		case ValueOpcode::BitwiseXor32: return binary(ValueOpcode::LogicalXor);
		default: {
			const auto value = EmitLaneBit(fold, Arg(*inst, 0), part, block, at);
			return Value(&*block.PrependNewInst(at, ValueOpcode::LogicalNot, {value}));
		}
	}
}

} // namespace

// KYTY_FOLD_LANE_MASKS=0 turns this off, for comparisons.
uint32_t FoldLaneMasks(Program& program) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FOLD_LANE_MASKS");
		return value == nullptr || value[0] != '0';
	}();
	// A hull shader's LaneId is its invocation id, not its subgroup lane.
	if (!enabled || program.stage == ShaderType::TessellationControl || program.blocks.empty()) {
		return 0;
	}
	LaneFold fold {program, program.stage == ShaderType::Pixel, Value {}};
	uint32_t folded = 0;
	for (auto* block: program.blocks) {
		for (auto inst = block->begin(); inst != block->end(); ++inst) {
			if (inst->GetOpcode() != ValueOpcode::INotEqual32) {
				continue;
			}
			const auto read = MatchLaneBit(*inst);
			if (!read || !LaneBitKnown(fold, read->low, 0u, 0) ||
			    (!read->high_half.IsEmpty() && !LaneBitKnown(fold, read->high, 1u, 0))) {
				continue;
			}
			auto value = EmitLaneBit(fold, read->low, 0u, *block, inst);
			if (!read->high_half.IsEmpty()) {
				const auto high = EmitLaneBit(fold, read->high, 1u, *block, inst);
				if (high != value) {
					value = Value(&*block->PrependNewInst(inst, ValueOpcode::SelectU1,
					                                      {read->high_half, value, high}));
				}
			}
			Replace(*inst, value);
			folded++;
		}
	}
	return folded;
}

void ConstantPropagationPass(const BlockList& blocks, uint32_t wave_size) {
	std::unordered_set<Inst*> lowered_ancillary;
	LaneMaskProjection mask_projection(lowered_ancillary);
	for (auto* block: blocks) {
		for (auto inst = block->begin(); inst != block->end(); ++inst) {
			if (wave_size == 32u) mask_projection.Fold(*inst);
			FoldInstruction(*block, inst, lowered_ancillary);
		}
	}
	// Normalize retained PHI/select values only after every supported field read has
	// been lowered; direct raw consumers remain unsupported.
	for (auto* source: lowered_ancillary) {
		const bool retained_only = std::ranges::all_of(source->Uses(), [](const Use& use) {
			const auto& user = *use.user;
			if (!user.HasUses() && !user.MayHaveSideEffects()) {
				return true;
			}
			return user.GetOpcode() == ValueOpcode::Phi ||
			       (user.GetOpcode() == ValueOpcode::SelectU32 && use.operand == 2u);
		});
		if (retained_only) {
			Replace(*source, Value(0u));
		}
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
