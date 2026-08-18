#include "debugger/target/threads.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#else
#include "kernel/pthread.h"

#include <csignal>
#include <pthread.h>
#include <unistd.h>
#if !defined(__APPLE__)
#include <sys/syscall.h>
#endif
#endif

namespace Debugger::Target {

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

bool CanSuspend() {
	return true;
}

bool Suspend(uint64_t host_tid, SuspendedHandle& out) {
	out = {};

	HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT,
	                           FALSE, static_cast<DWORD>(host_tid));
	if (handle == nullptr) {
		return false;
	}

	if (SuspendThread(handle) == static_cast<DWORD>(-1)) {
		CloseHandle(handle);
		return false;
	}

	out.native   = handle;
	out.host_tid = host_tid;
	out.valid    = true;
	return true;
}

bool ReadRegisters(const SuspendedHandle& handle, Registers& out) {
	if (!handle.valid || handle.native == nullptr) {
		return false;
	}

	CONTEXT context {};
	context.ContextFlags = CONTEXT_INTEGER | CONTEXT_CONTROL;

	if (GetThreadContext(static_cast<HANDLE>(handle.native), &context) == 0) {
		return false;
	}

	out.rax    = context.Rax;
	out.rbx    = context.Rbx;
	out.rcx    = context.Rcx;
	out.rdx    = context.Rdx;
	out.rsi    = context.Rsi;
	out.rdi    = context.Rdi;
	out.rbp    = context.Rbp;
	out.rsp    = context.Rsp;
	out.r8     = context.R8;
	out.r9     = context.R9;
	out.r10    = context.R10;
	out.r11    = context.R11;
	out.r12    = context.R12;
	out.r13    = context.R13;
	out.r14    = context.R14;
	out.r15    = context.R15;
	out.rip    = context.Rip;
	out.rflags = context.EFlags;
	out.valid  = true;
	return true;
}

bool WriteRegisters(const SuspendedHandle& handle, const Registers& regs) {
	if (!handle.valid || handle.native == nullptr) {
		return false;
	}

	CONTEXT context {};
	context.ContextFlags = CONTEXT_INTEGER | CONTEXT_CONTROL;

	if (GetThreadContext(static_cast<HANDLE>(handle.native), &context) == 0) {
		return false;
	}

	context.Rax    = regs.rax;
	context.Rbx    = regs.rbx;
	context.Rcx    = regs.rcx;
	context.Rdx    = regs.rdx;
	context.Rsi    = regs.rsi;
	context.Rdi    = regs.rdi;
	context.Rbp    = regs.rbp;
	context.Rsp    = regs.rsp;
	context.R8     = regs.r8;
	context.R9     = regs.r9;
	context.R10    = regs.r10;
	context.R11    = regs.r11;
	context.R12    = regs.r12;
	context.R13    = regs.r13;
	context.R14    = regs.r14;
	context.R15    = regs.r15;
	context.Rip    = regs.rip;
	context.EFlags = static_cast<DWORD>(regs.rflags);

	return SetThreadContext(static_cast<HANDLE>(handle.native), &context) != 0;
}

void Resume(SuspendedHandle& handle) {
	if (!handle.valid || handle.native == nullptr) {
		return;
	}

	ResumeThread(static_cast<HANDLE>(handle.native));
	CloseHandle(static_cast<HANDLE>(handle.native));
	handle = {};
}

bool RequestTrap(uint64_t /*host_tid*/) {
	return false;
}

bool InstallStopSignal() {
	return true;
}

uint64_t CurrentHostThreadId() {
	return static_cast<uint64_t>(GetCurrentThreadId());
}

bool CaptureRegisters(Registers& out) {
	CONTEXT context {};
	context.ContextFlags = CONTEXT_INTEGER | CONTEXT_CONTROL;
	RtlCaptureContext(&context);

	out.rax    = context.Rax;
	out.rbx    = context.Rbx;
	out.rcx    = context.Rcx;
	out.rdx    = context.Rdx;
	out.rsi    = context.Rsi;
	out.rdi    = context.Rdi;
	out.rbp    = context.Rbp;
	out.rsp    = context.Rsp;
	out.r8     = context.R8;
	out.r9     = context.R9;
	out.r10    = context.R10;
	out.r11    = context.R11;
	out.r12    = context.R12;
	out.r13    = context.R13;
	out.r14    = context.R14;
	out.r15    = context.R15;
	out.rip    = context.Rip;
	out.rflags = context.EFlags;
	out.valid  = true;
	return true;
}

#else

// POSIX cannot suspend a thread of the current process from the outside, so the debugger drives
// threads into its own trap handler with a signal instead: the handler raises a breakpoint trap
// on the target, which parks it exactly like a real breakpoint hit.
constexpr int STOP_SIGNAL = SIGUSR2;

namespace {

void StopSignalHandler(int /*sig*/) {
	// Raising the trap from inside the signal handler makes the thread re-enter through the
	// normal SIGTRAP path with a valid ucontext, so the debugger has one park mechanism, not
	// two. Async-signal-safe: raise() is on the permitted list.
	::raise(SIGTRAP);
}

} // namespace

bool CanSuspend() {
	return false;
}

bool Suspend(uint64_t /*host_tid*/, SuspendedHandle& out) {
	out = {};
	return false;
}

bool ReadRegisters(const SuspendedHandle& /*handle*/, Registers& /*out*/) {
	return false;
}

bool WriteRegisters(const SuspendedHandle& /*handle*/, const Registers& /*regs*/) {
	return false;
}

void Resume(SuspendedHandle& handle) {
	handle = {};
}

bool RequestTrap(uint64_t host_tid) {
	return Libs::LibKernel::PthreadKillHostByOsId(host_tid, STOP_SIGNAL);
}

bool InstallStopSignal() {
	struct sigaction action {};
	action.sa_handler = StopSignalHandler;
	sigemptyset(&action.sa_mask);
	action.sa_flags = SA_RESTART;

	return ::sigaction(STOP_SIGNAL, &action, nullptr) == 0;
}

uint64_t CurrentHostThreadId() {
#if defined(__APPLE__)
	uint64_t tid = 0;
	pthread_threadid_np(nullptr, &tid);
	return tid;
#else
	return static_cast<uint64_t>(::syscall(SYS_gettid));
#endif
}

bool CaptureRegisters(Registers& /*out*/) {
	return false;
}

#endif

} // namespace Debugger::Target
