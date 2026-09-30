// The native SRT walker (SrtNative.cpp) against the interpreter (SrtWalker.cpp): the same plan is
// built twice, one copy stays on the interpreter and the other is compiled, and every expression
// is evaluated by both for many inputs. Results, including failures, must be identical.

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtNative.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "SrtNativeTests: failed: %s\n", text);
		std::abort();
	}
}

constexpr uint32_t UserDataCount = 12;
constexpr uint64_t MemoryBase    = 0x10000;
constexpr uint32_t MemoryDwords  = 64;

// Guest memory as the tests' readers see it: MemoryDwords dwords at MemoryBase, with a different
// pattern for the clean (specialization) reader so that routing mistakes show.
struct TestMemory {
	std::array<uint32_t, MemoryDwords> ordinary {};
	std::array<uint32_t, MemoryDwords> clean {};
	uint32_t                           reads = 0;
};

bool ReadFrom(const std::array<uint32_t, MemoryDwords>& words, uint64_t address,
              std::span<uint32_t> values) {
	if (address < MemoryBase || (address & 3u) != 0u ||
	    address + values.size() * 4u > MemoryBase + MemoryDwords * 4u) {
		return false;
	}
	std::memcpy(values.data(), &words[(address - MemoryBase) / 4u], values.size_bytes());
	return true;
}

bool ReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& memory = *static_cast<TestMemory*>(userdata);
	memory.reads++;
	return ReadFrom(memory.ordinary, address, values);
}

bool ReadClean(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& memory = *static_cast<TestMemory*>(userdata);
	memory.reads++;
	return ReadFrom(memory.clean, address, values);
}

Block& AddValueBlock(Program& program) {
	auto  block  = std::make_unique<Block>();
	auto* result = block.get();
	program.blocks.push_back(result);
	program.block_info.push_back({.id = 0});
	program.block_storage.push_back(std::move(block));
	return *result;
}

// Every expression the native compiler handles, over user data s0..s11:
// s0..s3 arbitrary values, s4..s7 small counts (offsets, widths), s8/s9 an address, s10/s11 more.
ResourcePlan BuildPlan() {
	Program program;
	program.stage                      = Libs::Graphics::ShaderType::Compute;
	program.srt_plan_complete          = true;
	program.resource_tracking_complete = true;
	auto& b                            = AddValueBlock(program);

	std::vector<Value> exprs;
	auto               u = [&](uint32_t reg) {
        return Value(&b.AppendNewInst(ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(reg))}));
	};
	auto op = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
		return Value(&b.AppendNewInst(opcode, args));
	};
	auto add = [&](Value value) { exprs.push_back(value); };
	auto lo  = [&](Value wide) { return op(ValueOpcode::CompositeExtractU64, {wide, Value(0u)}); };
	auto hi  = [&](Value wide) { return op(ValueOpcode::CompositeExtractU64, {wide, Value(1u)}); };

	const auto s0 = u(0), s1 = u(1), s2 = u(2), s3 = u(3);
	const auto s4 = u(4), s5 = u(5), s6 = u(6), s7 = u(7);
	const auto s8 = u(8), s9 = u(9), s10 = u(10), s11 = u(11);
	const auto w0 = op(ValueOpcode::CompositeConstructU64, {s0, s1});
	const auto w1 = op(ValueOpcode::CompositeConstructU64, {s2, s3});

	add(s0), add(s11);
	add(op(ValueOpcode::IAdd32, {s0, s1}));
	add(op(ValueOpcode::ISub32, {s0, s1}));
	add(op(ValueOpcode::IMul32, {s0, s1}));
	add(op(ValueOpcode::UMulHi, {s0, s1}));
	add(op(ValueOpcode::UMin32, {s0, s1}));
	add(op(ValueOpcode::IAdd32, {s0, Value(0xfffffff0u)}));
	for (auto opcode: {ValueOpcode::IAdd64, ValueOpcode::ISub64, ValueOpcode::IMul64,
	                   ValueOpcode::BitwiseAnd64}) {
		const auto r = op(opcode, {w0, w1});
		add(lo(r)), add(hi(r));
	}
	for (auto opcode: {ValueOpcode::ShiftLeftLogical64, ValueOpcode::ShiftRightLogical64,
	                   ValueOpcode::ShiftRightArithmetic64}) {
		const auto r = op(opcode, {w0, s4});
		add(lo(r)), add(hi(r));
	}
	for (auto opcode: {ValueOpcode::BitwiseAnd32, ValueOpcode::BitwiseOr32, ValueOpcode::BitwiseXor32,
	                   ValueOpcode::ShiftLeftLogical32, ValueOpcode::ShiftRightLogical32,
	                   ValueOpcode::ShiftRightArithmetic32, ValueOpcode::IEqual32,
	                   ValueOpcode::INotEqual32, ValueOpcode::ULessThan32, ValueOpcode::UGreaterThan32,
	                   ValueOpcode::ULessThanEqual32, ValueOpcode::SGreaterThanEqual32}) {
		add(op(opcode, {s0, s1}));
		add(op(opcode, {s2, s4}));
	}
	add(op(ValueOpcode::BitwiseNot32, {s0}));
	for (auto opcode: {ValueOpcode::LogicalAnd, ValueOpcode::LogicalOr, ValueOpcode::LogicalXor}) {
		add(op(opcode, {op(ValueOpcode::ULessThan32, {s0, s1}), op(ValueOpcode::IEqual32, {s2, s3})}));
	}
	add(op(ValueOpcode::LogicalNot, {op(ValueOpcode::ULessThan32, {s0, s1})}));
	add(op(ValueOpcode::BitFieldUExtract, {s0, s4, s5}));
	add(op(ValueOpcode::BitFieldSExtract, {s0, s4, s5}));
	add(op(ValueOpcode::BitFieldUExtract, {s1, Value(8u), Value(8u)}));
	add(op(ValueOpcode::BitFieldInsert, {s0, s1, s6, s7}));
	add(op(ValueOpcode::BitFieldInsert, {s0, s1, Value(0u), Value(32u)}));
	// Floats: s0..s3 carry float bit patterns too.
	add(op(ValueOpcode::ConvertF32U32, {s0}));
	add(op(ValueOpcode::ConvertU32F32, {s2}));
	add(op(ValueOpcode::FPMul32, {s2, s3}));
	add(op(ValueOpcode::FPTrunc32, {s2}));
	add(op(ValueOpcode::FPRecipIFlag32, {s3}));
	add(op(ValueOpcode::FPIsNan32, {s2}));
	for (auto opcode: {ValueOpcode::FPOrdLessThanEqual32, ValueOpcode::FPOrdGreaterThanEqual32}) {
		add(op(opcode, {s2, s3}));
		auto& flushed = b.AppendNewInst(opcode, {s2, s3});
		flushed.SetFlags(FPCompareFlags {.flush_input_denorms = true});
		add(Value(&flushed));
	}
	add(op(ValueOpcode::BitCastF32U32, {op(ValueOpcode::BitCastU32F32, {s1})}));
	// Selects, including nested ones and a select between two memory reads.
	const auto less = op(ValueOpcode::ULessThan32, {s0, s1});
	add(op(ValueOpcode::SelectU32, {less, s2, s3}));
	add(op(ValueOpcode::SelectU32, {op(ValueOpcode::IEqual32, {s4, Value(3u)}),
	                                op(ValueOpcode::SelectU32, {less, s0, s10}), s11}));
	// Raw reads: a scalar address at s8/s9 plus s10 (and a negative immediate), and a constant
	// buffer with records s6 and stride from s9's high bits.
	{
		MemoryInfo address;
		address.kind          = ResourceKind::ScalarAddress;
		address.planning_only = true;
		program.memory_info.push_back(address);
		address.offset = static_cast<uint32_t>(-8);
		program.memory_info.push_back(address);
		MemoryInfo buffer;
		buffer.kind          = ResourceKind::ScalarBuffer;
		buffer.planning_only = true;
		buffer.offset        = 12;
		program.memory_info.push_back(buffer);

		auto& handle = b.AppendNewInst(ValueOpcode::GetAddressResource, {s8, s9});
		for (uint32_t index = 0; index < 2; index++) {
			auto& raw = b.AppendNewInst(ValueOpcode::LoadAddressU32,
			                            {Value(&handle), s10, Value(0u), Value(true)});
			raw.SetFlags(MemoryFlags {.index = index, .pc = 0x40});
			add(Value(&raw));
			add(op(ValueOpcode::SelectU32, {less, Value(&raw), s0}));
		}
		auto& buffer_handle = b.AppendNewInst(ValueOpcode::GetBufferResource, {s8, s9, s6, s1});
		auto& read          = b.AppendNewInst(ValueOpcode::ReadConstBuffer, {Value(&buffer_handle), s10});
		read.SetFlags(MemoryFlags {.index = 2, .pc = 0x44});
		add(Value(&read));
		// A null base reads zero.
		auto& null_handle = b.AppendNewInst(ValueOpcode::GetAddressResource, {Value(0u), Value(0u)});
		auto& null_read   = b.AppendNewInst(ValueOpcode::LoadAddressU32,
		                                    {Value(&null_handle), s10, Value(0u), Value(true)});
		null_read.SetFlags(MemoryFlags {.index = 0, .pc = 0x48});
		add(Value(&null_read));
	}

	for (const auto value: exprs) {
		DescriptorSource source;
		source.dwords[0]   = value;
		source.dword_count = 1;
		program.descriptor_sources.push_back(source);
	}
	return ExtractResourcePlan(program);
}

// Interesting 32-bit values: edges of integers and floats.
constexpr uint32_t Specials[] = {0u,          1u,          2u,          31u,         32u,
                                 33u,         0x7fffffffu, 0x80000000u, 0xffffffffu, 0x3f800000u,
                                 0xbf800000u, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0x00400000u,
                                 0x80400000u, 0x4f800000u, 0x4f7fffffu, 0x40000000u, 0x00800000u};

void RunCompare(const ResourcePlan& reference, const ResourcePlan& native, bool split) {
	std::mt19937 random(split ? 7u : 3u);
	TestMemory   memory;
	for (uint32_t i = 0; i < MemoryDwords; i++) {
		memory.ordinary[i] = 0x1000u + i * 0x01010101u;
		memory.clean[i]    = 0x2000u + i * 0x00100001u;
	}
	uint32_t mismatches = 0;
	for (uint32_t trial = 0; trial < 4000; trial++) {
		std::array<uint32_t, UserDataCount> user {};
		for (uint32_t i = 0; i < 4; i++) {
			user[i] = (random() & 1u) != 0 ? Specials[random() % std::size(Specials)] : random();
		}
		for (uint32_t i = 4; i < 8; i++) {
			user[i] = random() % 40u;
		}
		// Addresses: mostly inside the test memory, sometimes outside or null.
		const auto pick = random() % 8u;
		const auto base = pick == 0 ? 0ull : pick == 1 ? 0x7000ull : MemoryBase + (random() % 16u) * 4u;
		user[8]         = static_cast<uint32_t>(base);
		// The high dword doubles as the constant buffer's stride (bits 16-29).
		user[9]  = static_cast<uint32_t>(base >> 32u) | ((random() % 3u) << 18u);
		user[10] = (random() % 48u) * ((random() & 3u) == 0 ? 3u : 4u);
		user[11] = random();
		SrtRuntime ordinary {.user_data = user, .read_memory = ReadOrdinary, .userdata = &memory,
		                     .read_specialization_memory = ReadClean};
		SrtRuntime clean    = CleanRuntime(ordinary);
		for (uint32_t source = 0; source < reference.descriptor_sources.size(); source++) {
			DescriptorValue expected {};
			DescriptorValue actual {};
			bool            expected_ok = false;
			bool            actual_ok   = false;
			if (split) {
				SrtWalker ref_clean(reference, clean);
				SrtWalker ref(reference, ordinary, {}, &ref_clean);
				expected_ok = ref.EvaluateDescriptor(source, expected);
				SrtWalker nat_clean(native, clean);
				SrtWalker nat(native, ordinary, {}, &nat_clean);
				actual_ok = nat.EvaluateDescriptor(source, actual);
			} else {
				SrtWalker ref(reference, ordinary);
				expected_ok = ref.EvaluateDescriptor(source, expected);
				SrtWalker nat(native, ordinary);
				actual_ok = nat.EvaluateDescriptor(source, actual);
			}
			if (expected_ok != actual_ok || (expected_ok && expected.dwords[0] != actual.dwords[0])) {
				if (mismatches++ < 20) {
					std::fprintf(stderr,
					             "SrtNativeTests: %s source %u: interpreter %s 0x%08x, native %s "
					             "0x%08x (s0..s3 %08x %08x %08x %08x, s4..s7 %u %u %u %u, "
					             "s8 %08x s9 %08x s10 %u)\n",
					             split ? "split" : "self", source, expected_ok ? "ok" : "failed",
					             expected.dwords[0], actual_ok ? "ok" : "failed", actual.dwords[0],
					             user[0], user[1], user[2], user[3], user[4], user[5], user[6],
					             user[7], user[8], user[9], user[10]);
				}
			}
		}
	}
	Check(mismatches == 0, "native and interpreted results differ");
}

void TestNativeMatchesInterpreter() {
	Check(SrtNativeCode::Supported(), "native SRT code is not supported on this host");
	const auto reference = BuildPlan();
	// Keep the reference plan on the interpreter.
	reference.native_attempted = true;
	const auto native = BuildPlan();
	native.native_code = SrtNativeCode::Compile(native);
	Check(native.native_code != nullptr, "the plan did not compile");
	Check(native.native_code->Interpreted() == 0, "an expression went through the interpreter");
	RunCompare(reference, native, false);
	RunCompare(reference, native, true);
	std::printf("SrtNativeTests: %zu expressions, %u instructions, %zu bytes of code\n",
	            reference.descriptor_sources.size(), native.native_code->Instructions(),
	            native.native_code->CodeSize());
}

} // namespace

namespace Common {

int DbgExitHandler(const char* file, int line, std::string_view text) {
	std::fprintf(stderr, "SrtNativeTests: EXIT at %s:%d: %.*s\n", file, line,
	             static_cast<int>(text.size()), text.data());
	std::abort();
}

int DbgExitHandler(const char* file, int line, fmt::text_style, std::string_view text) {
	return DbgExitHandler(file, line, text);
}

int DbgExitIfHandler(const char* expression, const char* file, int line) {
	std::fprintf(stderr, "SrtNativeTests: EXIT_IF(%s) at %s:%d\n", expression, file, line);
	return 1;
}

void DbgExit(int) {
	std::abort();
}

} // namespace Common

int main() {
	TestNativeMatchesInterpreter();
	std::puts("SrtNativeTests: all cases passed");
	return 0;
}

#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
