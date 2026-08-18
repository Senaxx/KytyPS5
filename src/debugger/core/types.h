#ifndef EMULATOR_SRC_DEBUGGER_CORE_TYPES_H_
#define EMULATOR_SRC_DEBUGGER_CORE_TYPES_H_

#include "common/common.h"

#include <cstdint>
#include <string>

namespace Debugger {

// General-purpose register file as the debugger presents it. Filled either from a trap context
// (a thread stopped inside the exception handler) or from a suspended thread's OS context.
struct Registers {
	uint64_t rax    = 0;
	uint64_t rbx    = 0;
	uint64_t rcx    = 0;
	uint64_t rdx    = 0;
	uint64_t rsi    = 0;
	uint64_t rdi    = 0;
	uint64_t rbp    = 0;
	uint64_t rsp    = 0;
	uint64_t r8     = 0;
	uint64_t r9     = 0;
	uint64_t r10    = 0;
	uint64_t r11    = 0;
	uint64_t r12    = 0;
	uint64_t r13    = 0;
	uint64_t r14    = 0;
	uint64_t r15    = 0;
	uint64_t rip    = 0;
	uint64_t rflags = 0;
	bool     valid  = false;
};

enum class StopReason : uint8_t {
	None,
	Breakpoint,
	Step,
	Pause,
	Entry,
	// A failing assert or an unhandled fault. The process is on its way out, so these stops can
	// be inspected but not stepped or continued past — resuming lets the exit proceed.
	Fatal,
};

// A guest thread parked inside the trap handler. `native_context` stays valid only while the
// thread is parked; the session applies register edits through it just before resuming.
struct StoppedThread {
	int        unique_id     = 0;
	uint64_t   host_tid      = 0;
	uint64_t   address       = 0;
	StopReason reason        = StopReason::None;
	uint32_t   breakpoint_id = 0;
	Registers  regs;
	// Set for StopReason::Fatal: the report that caused the halt.
	std::string message;
};

struct Breakpoint {
	uint32_t id            = 0;
	uint64_t address       = 0;
	bool     enabled       = true;
	bool     one_shot      = false;
	bool     armed         = false;
	bool     pending       = false; // requested by symbol/address that is not mapped yet
	uint64_t hit_count     = 0;
	uint8_t  original_byte = 0;
	// Non-zero for the one-shots that step-over and step-out plant: only that thread should stop
	// there. Another thread reaching the same address steps over the patch and keeps running,
	// which matters because a game's worker threads share code with the one being stepped.
	int         owner_unique_id = 0;
	std::string label;
};

// One decoded instruction for the disassembly panel.
struct DisassembledInstruction {
	uint64_t    address = 0;
	uint32_t    length  = 0;
	std::string bytes;
	std::string text;
	std::string symbol; // non-empty when `address` is a known symbol's entry point
};

struct ModuleInfo {
	int32_t     id         = -1;
	uint64_t    base_vaddr = 0;
	uint64_t    size       = 0;
	std::string name;
};

} // namespace Debugger

#endif // EMULATOR_SRC_DEBUGGER_CORE_TYPES_H_
