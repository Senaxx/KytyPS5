#ifndef EMULATOR_SRC_DEBUGGER_SYMBOLS_SYMBOLS_H_
#define EMULATOR_SRC_DEBUGGER_SYMBOLS_SYMBOLS_H_

#include "common/common.h"
#include "debugger/core/types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Debugger::Symbols {

struct Location {
	uint64_t    address = 0;
	std::string module; // e.g. "eboot.bin"
	uint64_t    module_offset = 0;
	std::string symbol; // nearest preceding symbol, empty if none
	uint64_t    symbol_offset = 0;
	bool        resolved      = false;
};

// Best-effort description of a code address: owning guest module, module-relative offset, and
// the nearest known symbol at or before it.
Location Describe(uint64_t address);

// Human-readable form: "eboot.bin+0x1234 (sceFoo+0x10)" — what the disassembly and call-stack
// panels render.
std::string Format(uint64_t address);

// Resolve a user-typed breakpoint location. Accepts:
//   0x00401000            absolute address
//   eboot.bin+0x1234      module-relative
//   sceKernelFoo          exported or HLE symbol name
// Returns false when nothing matches (the caller keeps the breakpoint pending).
bool Resolve(const std::string& text, uint64_t& address_out);

std::vector<ModuleInfo> Modules();

// Every symbol whose name contains `filter` (case-insensitive), capped at `limit`.
struct SymbolMatch {
	uint64_t    address = 0;
	std::string name;
	std::string module;
};
std::vector<SymbolMatch> Search(const std::string& filter, size_t limit,
                                const std::string& module_filter = {});

} // namespace Debugger::Symbols

namespace Debugger::Disasm {

// Decode `count` instructions starting at `address`. Stops early on an undecodable byte or an
// unreadable page. Breakpoint bytes are already overlaid by the caller's memory read, so the
// original instruction stream is what gets decoded.
std::vector<DisassembledInstruction> Decode(uint64_t address, uint32_t count);

// Length of the single instruction at `address`, or 0 if it cannot be decoded. Used to place
// the temporary breakpoint for step-over.
uint32_t InstructionLength(uint64_t address);

// True when the instruction at `address` is a call, i.e. step-over needs a temporary
// breakpoint rather than a single step.
bool IsCall(uint64_t address, uint32_t& length_out);

} // namespace Debugger::Disasm

#endif // EMULATOR_SRC_DEBUGGER_SYMBOLS_SYMBOLS_H_
