#ifndef EMULATOR_SRC_DEBUGGER_TARGET_THREADS_H_
#define EMULATOR_SRC_DEBUGGER_TARGET_THREADS_H_

#include "common/common.h"
#include "debugger/core/types.h"

#include <cstdint>

namespace Debugger::Target {

// Opaque per-platform handle to a suspended thread. Only meaningful between Suspend() and the
// matching Resume().
struct SuspendedHandle {
	void*    native   = nullptr;
	uint64_t host_tid = 0;
	bool     valid    = false;
};

// Suspend a running guest thread and read its register file.
//
// Only implemented where the OS can stop a thread from inside the same process (Windows). On
// POSIX the debugger stops threads by signalling them into the trap handler instead, so this
// returns false and Session falls back to that path.
bool CanSuspend();

bool Suspend(uint64_t host_tid, SuspendedHandle& out);
bool ReadRegisters(const SuspendedHandle& handle, Registers& out);
bool WriteRegisters(const SuspendedHandle& handle, const Registers& regs);
void Resume(SuspendedHandle& handle);

// Ask a running guest thread to trap into the debugger at the next opportunity. POSIX only;
// returns false on Windows, where Suspend() is used instead.
bool RequestTrap(uint64_t host_tid);

// Install the POSIX stop-signal handler. No-op on Windows.
bool InstallStopSignal();

uint64_t CurrentHostThreadId();

// Snapshot the calling thread's own registers, for a stop that did not come through a trap (a
// failing assert, say) and so has no exception context to read them from. Windows only for now;
// elsewhere the fatal stop simply reports no registers.
bool CaptureRegisters(Registers& out);

} // namespace Debugger::Target

#endif // EMULATOR_SRC_DEBUGGER_TARGET_THREADS_H_
