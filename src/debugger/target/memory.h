#ifndef EMULATOR_SRC_DEBUGGER_TARGET_MEMORY_H_
#define EMULATOR_SRC_DEBUGGER_TARGET_MEMORY_H_

#include "common/common.h"

#include <cstddef>
#include <cstdint>

namespace Debugger::Target {

// Fault-free access to the emulator's own address space (guest memory included). These never
// raise: an unmapped or unreadable range returns false instead of taking the fault path, so the
// memory panel can be scrolled anywhere without killing the process.
//
// Neither function applies the software-breakpoint byte overlay — use Session::ReadMemory for
// anything the user sees, or the 0xCC patches show through.
bool SafeRead(uint64_t address, void* dst, size_t size);
bool SafeWrite(uint64_t address, const void* src, size_t size);

// Write through a read-only mapping by flipping protection around the store. Used to plant and
// lift breakpoint bytes in guest code.
bool WriteCode(uint64_t address, const void* src, size_t size);

} // namespace Debugger::Target

#endif // EMULATOR_SRC_DEBUGGER_TARGET_MEMORY_H_
