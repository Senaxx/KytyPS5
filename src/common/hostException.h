#ifndef KYTY_COMMON_HOST_EXCEPTION_H_
#define KYTY_COMMON_HOST_EXCEPTION_H_

#include "common/common.h"

namespace Common::HostException {

enum class ExceptionType { Unknown, AccessViolation, IllegalInstruction, Breakpoint, SingleStep };

enum class AccessViolationType { Unknown, Read, Write, Execute };

struct ExceptionInfo {
	ExceptionType       type                   = ExceptionType::Unknown;
	AccessViolationType access_violation_type  = AccessViolationType::Unknown;
	uint64_t            access_violation_vaddr = 0;
	uint64_t            exception_address      = 0;
	uint64_t            rax                    = 0;
	uint64_t            rbx                    = 0;
	uint64_t            rcx                    = 0;
	uint64_t            rdx                    = 0;
	uint64_t            rsi                    = 0;
	uint64_t            rdi                    = 0;
	uint64_t            rbp                    = 0;
	uint64_t            rsp                    = 0;
	uint64_t            r8                     = 0;
	uint64_t            r9                     = 0;
	uint64_t            r10                    = 0;
	uint64_t            r11                    = 0;
	uint64_t            r12                    = 0;
	uint64_t            r13                    = 0;
	uint64_t            r14                    = 0;
	uint64_t            r15                    = 0;
	// Instruction pointer, normalised across platforms.
	//
	// For ExceptionType::Breakpoint this is always the address of the trap instruction itself.
	// The platforms disagree natively: the Windows kernel backs the reported Rip up onto the
	// int3, while a POSIX SIGTRAP reports the address after it. Normalising here keeps that
	// difference out of every handler — a handler that claims the fault should point the
	// context's instruction pointer at this address to re-execute the original instruction.
	//
	// For ExceptionType::SingleStep it is the next instruction to execute, on both platforms.
	uint64_t rip         = 0;
	uint64_t rflags      = 0;
	uint32_t native_code = 0;
	// Platform-specific mutable context, valid only for the duration of the handler call.
	void* native_context = nullptr;
};

using Handler = bool (*)(const ExceptionInfo&);

// Handlers are dispatched in ascending priority order; the first one to return true owns the
// fault and execution resumes. A handler must claim only the faults it understands: swallowing
// another subsystem's fault (the GPU page tracker's, in particular) breaks it silently.
constexpr int32_t PRIORITY_DEBUGGER = 0;
constexpr int32_t PRIORITY_DEFAULT  = 100;

constexpr uint32_t MAX_HANDLERS = 8;

bool AddHandler(Handler handler, int32_t priority);
bool RemoveHandler(Handler handler);

// Equivalent to AddHandler(handler, PRIORITY_DEFAULT).
bool InstallHandler(Handler handler);

// Mutate the faulting thread's context from inside a handler. `native_context` must be the
// pointer from the ExceptionInfo currently being handled; the write takes effect when the
// handler returns true.
void SetInstructionPointer(void* native_context, uint64_t rip);
void SetFlagsRegister(void* native_context, uint64_t rflags);

// General-purpose registers, in the order the debugger presents them.
enum class Gpr : uint8_t {
	Rax,
	Rbx,
	Rcx,
	Rdx,
	Rsi,
	Rdi,
	Rbp,
	Rsp,
	R8,
	R9,
	R10,
	R11,
	R12,
	R13,
	R14,
	R15,
};

void SetGpr(void* native_context, Gpr reg, uint64_t value);

// x86-64 RFLAGS.TF — set to have the CPU raise a single-step trap after the next instruction.
constexpr uint64_t FLAG_TRAP = 0x100;

} // namespace Common::HostException

#endif /* KYTY_COMMON_HOST_EXCEPTION_H_ */
