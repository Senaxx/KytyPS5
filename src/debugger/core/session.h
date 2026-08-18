#ifndef EMULATOR_SRC_DEBUGGER_CORE_SESSION_H_
#define EMULATOR_SRC_DEBUGGER_CORE_SESSION_H_

#include "common/common.h"
#include "debugger/core/types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Debugger::Session {

// Maximum simultaneously armed software breakpoints. The trap handler scans this table on every
// breakpoint exception, so it is a fixed array of atomics rather than a container: no lock, no
// allocation, nothing that could deadlock against a parked thread.
constexpr uint32_t MAX_BREAKPOINTS = 256;

enum class ResumeMode : uint8_t {
	Continue,
	StepInto,
	StepOver,
	StepOut,
};

// Arm the trap handler. Returns false if the handler chain rejected the registration, in which
// case the debugger stays inert.
bool Initialize();
void Shutdown();

bool IsEnabled();

// True while at least one guest thread is stopped.
bool IsPaused();

// ---- Breakpoints -------------------------------------------------------------------------

// `location` is a symbol, module+offset or hex address (see Symbols::Resolve). A location that
// does not resolve yet is kept pending and retried when a module is loaded.
uint32_t AddBreakpoint(const std::string& location, bool one_shot = false);
uint32_t AddBreakpointAt(uint64_t address, bool one_shot = false);
bool     RemoveBreakpoint(uint32_t id);
bool     SetBreakpointEnabled(uint32_t id, bool enabled);
void     ClearBreakpoints();

std::vector<Breakpoint> Breakpoints();

// Retry every pending breakpoint. Called after a module load.
void ResolvePendingBreakpoints();

// ---- Execution ---------------------------------------------------------------------------

// Stop every guest thread. On Windows this suspends them; on POSIX it signals them into the
// trap handler. Either way the threads end up in Stopped().
void Pause();

// Resume one stopped thread, or every stopped thread when `unique_id` is 0.
void Resume(int unique_id, ResumeMode mode);
void ResumeAll();

std::vector<StoppedThread> Stopped();

// Overwrite a stopped thread's registers; applied when it resumes.
bool WriteRegisters(int unique_id, const Registers& regs);

// ---- Memory ------------------------------------------------------------------------------

// Read guest memory with software-breakpoint bytes replaced by the originals, so what the user
// sees is the real instruction stream.
bool ReadMemory(uint64_t address, void* dst, size_t size);
bool WriteMemory(uint64_t address, const void* src, size_t size);

// ---- Call stack --------------------------------------------------------------------------

struct Frame {
	uint64_t    address = 0;
	uint64_t    frame   = 0;
	std::string description;
};

// Frame-pointer chain walk. Guest code is frequently compiled without frame pointers, so this
// truncates rather than inventing frames; `.eh_frame` unwinding is the accurate follow-up.
std::vector<Frame> Backtrace(int unique_id, uint32_t max_frames = 32);

// ---- Lifecycle hooks ---------------------------------------------------------------------

// Called once the guest entry point is known, before it runs.
void OnGuestEntry(uint64_t entry_address);

// ---- Fatal errors ------------------------------------------------------------------------

// Halt the calling thread on a failing assert so its state can be read before the process dies.
// Returns once the halt is released, and the caller then exits as it would have.
//
// Only guest threads are halted: parking the thread that drives presentation or the GPU would
// freeze the very overlay meant to show the failure, and possibly deadlock it. Fatal errors on
// those threads are reported and left to terminate.
void ReportFatal(const char* report);

bool BreakOnFatalEnabled();
void SetBreakOnFatal(bool enabled);

// Called by the overlay every time it draws. A fatal halt uses this to tell whether presentation
// survived parking the failing thread; if the overlay never draws, the halt releases itself so
// the process can exit instead of hanging.
void NotifyOverlayDrawn();

} // namespace Debugger::Session

#endif // EMULATOR_SRC_DEBUGGER_CORE_SESSION_H_
