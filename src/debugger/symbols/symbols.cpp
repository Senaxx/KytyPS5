#include "debugger/symbols/symbols.h"

#include "common/singleton.h"
#include "common/stringUtils.h"
#include "debugger/core/session.h"
#include "debugger/target/memory.h"
#include "loader/elf.h"
#include "loader/runtimeLinker.h"
#include "loader/symbolDatabase.h"

#include <Zydis/Zydis.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstring>

namespace Debugger::Symbols {

namespace {

std::string ModuleName(const Loader::Program* program) {
	if (program == nullptr) {
		return {};
	}
	return Common::FilenameWithoutDirectory(Common::PathToGenericString(program->file_name));
}

// The linker's databases are keyed by name, not sorted by address, so a nearest-preceding
// lookup is a linear scan. The symbol count is in the low thousands and this only runs for
// addresses the user is actually looking at, so a scan is cheaper than maintaining an index
// that would have to be invalidated on every module load.
bool NearestSymbol(const Loader::SymbolDatabase* db, uint64_t address, uint64_t low, uint64_t high,
                   std::string& name_out, uint64_t& offset_out) {
	if (db == nullptr) {
		return false;
	}

	const Loader::SymbolRecord* best = nullptr;

	for (const auto& record: db->Records()) {
		if (record.vaddr > address || record.vaddr < low || record.vaddr >= high) {
			continue;
		}
		if (best == nullptr || record.vaddr > best->vaddr) {
			best = &record;
		}
	}

	if (best == nullptr) {
		return false;
	}

	name_out   = best->dbg_name.empty() ? best->name : best->dbg_name;
	offset_out = address - best->vaddr;
	return true;
}

bool ParseHex(const std::string& text, uint64_t& out) {
	std::string_view view = text;
	if (view.size() > 2 && (view.substr(0, 2) == "0x" || view.substr(0, 2) == "0X")) {
		view.remove_prefix(2);
	}
	if (view.empty()) {
		return false;
	}

	uint64_t   value = 0;
	const auto end   = view.data() + view.size();
	const auto res   = std::from_chars(view.data(), end, value, 16);
	if (res.ec != std::errc {} || res.ptr != end) {
		return false;
	}

	out = value;
	return true;
}

} // namespace

Location Describe(uint64_t address) {
	Location location {};
	location.address = address;

	if (address == 0) {
		return location;
	}

	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();
	if (rt == nullptr) {
		return location;
	}

	auto* program = rt->FindProgramByAddr(address);
	if (program == nullptr) {
		return location;
	}

	location.module        = ModuleName(program);
	location.module_offset = address - program->base_vaddr;
	location.resolved      = true;

	const uint64_t low  = program->base_vaddr;
	const uint64_t high = program->base_vaddr + program->base_size_aligned;

	// A module's own exports win over the global table, which also holds HLE entry points that
	// live outside any guest module.
	if (!NearestSymbol(program->export_symbols.get(), address, low, high, location.symbol,
	                   location.symbol_offset)) {
		NearestSymbol(rt->Symbols(), address, low, high, location.symbol, location.symbol_offset);
	}

	return location;
}

std::string Format(uint64_t address) {
	const auto location = Describe(address);

	std::array<char, 256> buffer {};

	if (!location.resolved) {
		std::snprintf(buffer.data(), buffer.size(), "0x%016llx",
		              static_cast<unsigned long long>(address));
		return buffer.data();
	}

	if (location.symbol.empty()) {
		std::snprintf(buffer.data(), buffer.size(), "%s+0x%llx", location.module.c_str(),
		              static_cast<unsigned long long>(location.module_offset));
		return buffer.data();
	}

	if (location.symbol_offset == 0) {
		std::snprintf(buffer.data(), buffer.size(), "%s!%s", location.module.c_str(),
		              location.symbol.c_str());
		return buffer.data();
	}

	std::snprintf(buffer.data(), buffer.size(), "%s!%s+0x%llx", location.module.c_str(),
	              location.symbol.c_str(), static_cast<unsigned long long>(location.symbol_offset));
	return buffer.data();
}

bool Resolve(const std::string& text, uint64_t& address_out) {
	if (text.empty()) {
		return false;
	}

	// Plain address.
	if (ParseHex(text, address_out)) {
		return true;
	}

	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();
	if (rt == nullptr) {
		return false;
	}

	// module+offset
	if (const auto plus = text.find('+'); plus != std::string::npos) {
		const auto module_part = text.substr(0, plus);
		const auto offset_part = text.substr(plus + 1);

		uint64_t offset = 0;
		if (!ParseHex(offset_part, offset)) {
			return false;
		}

		for (auto* program: rt->Programs()) {
			if (Common::EqualNoCase(ModuleName(program), module_part)) {
				address_out = program->base_vaddr + offset;
				return true;
			}
		}
		return false;
	}

	// Symbol name, searched in every module's exports and then the global table.
	for (auto* program: rt->Programs()) {
		if (program->export_symbols == nullptr) {
			continue;
		}
		for (const auto& record: program->export_symbols->Records()) {
			if (record.name == text || record.dbg_name == text) {
				address_out = record.vaddr;
				return true;
			}
		}
	}

	if (auto* symbols = rt->Symbols(); symbols != nullptr) {
		for (const auto& record: symbols->Records()) {
			if (record.name == text || record.dbg_name == text) {
				address_out = record.vaddr;
				return true;
			}
		}
	}

	return false;
}

std::vector<ModuleInfo> Modules() {
	std::vector<ModuleInfo> out;

	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();
	if (rt == nullptr) {
		return out;
	}

	for (auto* program: rt->Programs()) {
		if (program == nullptr) {
			continue;
		}

		ModuleInfo info {};
		info.id         = program->unique_id;
		info.base_vaddr = program->base_vaddr;
		info.size       = program->base_size_aligned;
		info.name       = ModuleName(program);
		out.push_back(std::move(info));
	}

	std::sort(out.begin(), out.end(),
	          [](const ModuleInfo& a, const ModuleInfo& b) { return a.base_vaddr < b.base_vaddr; });

	return out;
}

std::vector<SymbolMatch> Search(const std::string& filter, size_t limit,
                                const std::string& module_filter) {
	std::vector<SymbolMatch> out;

	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();
	if (rt == nullptr) {
		return out;
	}

	const auto matches = [&filter](const std::string& name) {
		if (filter.empty()) {
			return true;
		}
		const auto haystack = Common::ToLower(name);
		return haystack.find(Common::ToLower(filter)) != std::string::npos;
	};

	const auto collect = [&](const Loader::SymbolDatabase* db, const std::string& module) {
		if (db == nullptr) {
			return;
		}
		if (!module_filter.empty() && !Common::EqualNoCase(module, module_filter)) {
			return;
		}
		for (const auto& record: db->Records()) {
			if (out.size() >= limit) {
				return;
			}
			const auto& name = record.dbg_name.empty() ? record.name : record.dbg_name;
			if (!matches(name)) {
				continue;
			}
			out.push_back(SymbolMatch {record.vaddr, name, module});
		}
	};

	for (auto* program: rt->Programs()) {
		if (program == nullptr) {
			continue;
		}
		collect(program->export_symbols.get(), ModuleName(program));
	}
	collect(rt->Symbols(), "hle");

	return out;
}

} // namespace Debugger::Symbols

namespace Debugger::Disasm {

namespace {

constexpr uint32_t MAX_INSTRUCTION_BYTES = 16;

const ZydisDecoder& Decoder() {
	static const ZydisDecoder decoder = [] {
		ZydisDecoder value {};
		ZydisDecoderInit(&value, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
		return value;
	}();
	return decoder;
}

const ZydisFormatter& Formatter() {
	static const ZydisFormatter formatter = [] {
		ZydisFormatter value {};
		ZydisFormatterInit(&value, ZYDIS_FORMATTER_STYLE_INTEL);
		return value;
	}();
	return formatter;
}

// Read up to MAX_INSTRUCTION_BYTES, shrinking the request until it fits inside mapped memory so
// an instruction at the very end of a mapping still decodes.
//
// Goes through Session::ReadMemory, not the raw target, so software-breakpoint bytes are
// replaced by the originals. Decoding the 0xCC instead would not just show "int3" in the
// listing — step-over derives its instruction length from here, so it would plant a one-shot
// breakpoint in the middle of the real instruction and corrupt the code stream.
uint32_t ReadCodeBytes(uint64_t address, uint8_t* buffer) {
	for (uint32_t size = MAX_INSTRUCTION_BYTES; size > 0; size--) {
		if (Session::ReadMemory(address, buffer, size)) {
			return size;
		}
	}
	return 0;
}

} // namespace

std::vector<DisassembledInstruction> Decode(uint64_t address, uint32_t count) {
	std::vector<DisassembledInstruction> out;
	out.reserve(count);

	uint64_t cursor = address;

	for (uint32_t i = 0; i < count; i++) {
		std::array<uint8_t, MAX_INSTRUCTION_BYTES> bytes {};

		const uint32_t available = ReadCodeBytes(cursor, bytes.data());
		if (available == 0) {
			break;
		}

		ZydisDecodedInstruction                                  instruction {};
		std::array<ZydisDecodedOperand, ZYDIS_MAX_OPERAND_COUNT> operands {};

		DisassembledInstruction decoded {};
		decoded.address = cursor;

		if (ZYAN_SUCCESS(ZydisDecoderDecodeFull(&Decoder(), bytes.data(), available, &instruction,
		                                        operands.data()))) {
			decoded.length = instruction.length;

			std::array<char, 256> text {};
			if (ZYAN_SUCCESS(ZydisFormatterFormatInstruction(
			        &Formatter(), &instruction, operands.data(), instruction.operand_count_visible,
			        text.data(), text.size(), cursor, ZYAN_NULL))) {
				decoded.text = text.data();
			} else {
				decoded.text = "(unformattable)";
			}
		} else {
			// Undecodable byte: emit it as data and resynchronise on the next one rather than
			// abandoning the whole listing.
			decoded.length = 1;
			decoded.text   = "(bad)";
		}

		std::array<char, 3 * MAX_INSTRUCTION_BYTES + 1> hex {};
		for (uint32_t b = 0; b < decoded.length && b < available; b++) {
			std::snprintf(hex.data() + b * 3, 4, "%02x ", bytes[b]);
		}
		decoded.bytes = hex.data();

		if (const auto location = Symbols::Describe(cursor);
		    location.resolved && !location.symbol.empty() && location.symbol_offset == 0) {
			decoded.symbol = location.symbol;
		}

		cursor += decoded.length;
		out.push_back(std::move(decoded));
	}

	return out;
}

uint32_t InstructionLength(uint64_t address) {
	std::array<uint8_t, MAX_INSTRUCTION_BYTES> bytes {};

	const uint32_t available = ReadCodeBytes(address, bytes.data());
	if (available == 0) {
		return 0;
	}

	ZydisDecodedInstruction instruction {};
	if (!ZYAN_SUCCESS(ZydisDecoderDecodeInstruction(&Decoder(), nullptr, bytes.data(), available,
	                                                &instruction))) {
		return 0;
	}

	return instruction.length;
}

bool IsCall(uint64_t address, uint32_t& length_out) {
	length_out = 0;

	std::array<uint8_t, MAX_INSTRUCTION_BYTES> bytes {};

	const uint32_t available = ReadCodeBytes(address, bytes.data());
	if (available == 0) {
		return false;
	}

	ZydisDecodedInstruction instruction {};
	if (!ZYAN_SUCCESS(ZydisDecoderDecodeInstruction(&Decoder(), nullptr, bytes.data(), available,
	                                                &instruction))) {
		return false;
	}

	length_out = instruction.length;
	return instruction.mnemonic == ZYDIS_MNEMONIC_CALL;
}

} // namespace Debugger::Disasm
