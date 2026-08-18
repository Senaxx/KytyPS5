#include "common/hostException.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#elif defined(__APPLE__)
#include <csignal>
#include <sys/ucontext.h>
#else
#include <csignal>
#include <initializer_list>
#include <ucontext.h> // IWYU pragma: keep
#include <unistd.h>
#endif

// IWYU pragma: no_include <errhandlingapi.h>
// IWYU pragma: no_include <excpt.h>
// IWYU pragma: no_include <minwinbase.h>
// IWYU pragma: no_include <minwindef.h>
// IWYU pragma: no_include <wtypes.h>

namespace Common::HostException {

#if !defined(__APPLE__)

static std::atomic_uint32_t g_install_state {0};
static thread_local bool    g_in_exception_filter = false;

static_assert(decltype(g_install_state)::is_always_lock_free);

[[noreturn]] static void FailFast(const char* reason) noexcept {
	std::fputs("HostException fail-fast: ", stderr);
	std::fputs(reason != nullptr ? reason : "unspecified", stderr);
	std::fputc('\n', stderr);
	std::fflush(stderr);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TerminateProcess(GetCurrentProcess(), static_cast<UINT>(EXCEPTION_NONCONTINUABLE_EXCEPTION));
#endif
	std::_Exit(321);
}

class FilterScope final {
public:
	FilterScope() noexcept {
		if (g_in_exception_filter) {
			FailFast("nested exception while resolving a host fault");
		}
		g_in_exception_filter = true;
	}

	~FilterScope() { g_in_exception_filter = false; }

	KYTY_CLASS_NO_COPY(FilterScope);
};
#endif

namespace {

struct HandlerEntry {
	Handler fn       = nullptr;
	int32_t priority = 0;
};

// Immutable snapshot of the handler chain, sorted by ascending priority. Published by an
// atomic pointer swap so the fault path never takes a lock; superseded tables are leaked
// rather than freed, since a fault on another thread may still be walking one. Registration
// happens a handful of times per process, so the leak is bounded at a few hundred bytes.
struct HandlerTable {
	uint32_t     count = 0;
	HandlerEntry entries[MAX_HANDLERS] {};
};

std::atomic<const HandlerTable*> g_table {nullptr};
std::mutex                       g_register_mutex;

static_assert(decltype(g_table)::is_always_lock_free);

// Replace the published chain with `next`. Caller holds g_register_mutex.
void PublishTable(const HandlerTable& next) {
	auto* published = new HandlerTable(next); // NOLINT(cppcoreguidelines-owning-memory)
	g_table.store(published, std::memory_order_release);
}

// Walk the chain in priority order. Returns true when a handler claims the fault.
bool DispatchToChain(const ExceptionInfo& info) noexcept {
	const auto* table = g_table.load(std::memory_order_acquire);
	if (table == nullptr) {
		return false;
	}

	for (uint32_t i = 0; i < table->count; i++) {
		if (table->entries[i].fn != nullptr && table->entries[i].fn(info)) {
			return true;
		}
	}
	return false;
}

} // namespace

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

static LONG WINAPI ExceptionFilter(PEXCEPTION_POINTERS exception) noexcept {
	FilterScope filter_scope;

	auto* exception_record = exception->ExceptionRecord;

	if (exception_record->ExceptionCode == DBG_PRINTEXCEPTION_C ||
	    exception_record->ExceptionCode == DBG_PRINTEXCEPTION_WIDE_C) {
		return EXCEPTION_CONTINUE_SEARCH;
	}

	if (exception_record->ExceptionCode == 0x406D1388) {
		// Set a thread name.
		return EXCEPTION_CONTINUE_EXECUTION;
	}

	ExceptionInfo info {};
	info.exception_address = reinterpret_cast<uint64_t>(exception_record->ExceptionAddress);
	info.native_code       = exception_record->ExceptionCode;
	info.native_context    = exception->ContextRecord;

	if (exception_record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
		info.type = ExceptionType::AccessViolation;
		switch (exception_record->ExceptionInformation[0]) {
			case 0: info.access_violation_type = AccessViolationType::Read; break;
			case 1: info.access_violation_type = AccessViolationType::Write; break;
			case 8: info.access_violation_type = AccessViolationType::Execute; break;
			default: info.access_violation_type = AccessViolationType::Unknown; break;
		}
		info.access_violation_vaddr = exception_record->ExceptionInformation[1];
	} else if (exception_record->ExceptionCode == EXCEPTION_ILLEGAL_INSTRUCTION) {
		info.type = ExceptionType::IllegalInstruction;
	} else if (exception_record->ExceptionCode == EXCEPTION_BREAKPOINT) {
		info.type = ExceptionType::Breakpoint;
	} else if (exception_record->ExceptionCode == EXCEPTION_SINGLE_STEP) {
		info.type = ExceptionType::SingleStep;
	} else {
		return EXCEPTION_CONTINUE_SEARCH;
	}

	info.rax = exception->ContextRecord->Rax;
	info.rbx = exception->ContextRecord->Rbx;
	info.rcx = exception->ContextRecord->Rcx;
	info.rdx = exception->ContextRecord->Rdx;
	info.rsi = exception->ContextRecord->Rsi;
	info.rdi = exception->ContextRecord->Rdi;
	info.rbp = exception->ContextRecord->Rbp;
	info.rsp = exception->ContextRecord->Rsp;
	info.r8  = exception->ContextRecord->R8;
	info.r9  = exception->ContextRecord->R9;
	info.r10 = exception->ContextRecord->R10;
	info.r11 = exception->ContextRecord->R11;
	info.r12 = exception->ContextRecord->R12;
	info.r13 = exception->ContextRecord->R13;
	info.r14 = exception->ContextRecord->R14;
	info.r15 = exception->ContextRecord->R15;
	// Windows already reports Rip on the int3 for EXCEPTION_BREAKPOINT, so it needs no
	// adjustment to satisfy the normalisation described in the header.
	info.rip    = exception->ContextRecord->Rip;
	info.rflags = exception->ContextRecord->EFlags;

	// Nothing claimed it: CONTINUE_SEARCH, so a native debugger attached to the emulator still
	// receives its own breakpoints and the OS still terminates on a genuine crash.
	return DispatchToChain(info) ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}

#elif defined(__APPLE__)

static std::atomic_uint32_t g_install_state {0};
static thread_local bool    g_in_exception_filter = false;

static_assert(decltype(g_install_state)::is_always_lock_free);

[[noreturn]] static void FailFast(const char* reason) noexcept {
	std::fputs("HostException fail-fast: ", stderr);
	std::fputs(reason != nullptr ? reason : "unspecified", stderr);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	std::_Exit(321);
}

// Translate the x86-64 page-fault error code (mcontext __es.__err) into an access type.
// bit 1 (0x2) = write, bit 4 (0x10) = instruction fetch, otherwise a read.
static AccessViolationType DecodeAccess(uint64_t err) {
	if ((err & 0x10u) != 0) {
		return AccessViolationType::Execute;
	}
	if ((err & 0x2u) != 0) {
		return AccessViolationType::Write;
	}
	return AccessViolationType::Read;
}

// POSIX signal handler that mirrors the Windows vectored handler: build an ExceptionInfo
// from the mcontext and dispatch. A resolved fault (handler returns true) simply returns,
// re-executing the faulting instruction against the now-fixed protection. An unresolved
// fault restores the default disposition so the retry terminates the process.
static void SignalHandler(int sig, siginfo_t* si, void* uctx) {
	if (g_in_exception_filter) {
		FailFast("nested exception while resolving a host fault");
	}
	g_in_exception_filter = true;

	auto*       uc = static_cast<ucontext_t*>(uctx);
	const auto* mc = uc->uc_mcontext;
	const auto& ss = mc->__ss;

	ExceptionInfo info {};
	info.exception_address = ss.__rip;
	info.native_code       = static_cast<uint32_t>(si->si_code);
	info.native_context    = uctx;

	if (sig == SIGILL) {
		info.type = ExceptionType::IllegalInstruction;
	} else if (sig == SIGTRAP) {
		info.type =
		    (si->si_code == TRAP_TRACE) ? ExceptionType::SingleStep : ExceptionType::Breakpoint;
	} else {
		info.type                   = ExceptionType::AccessViolation;
		info.access_violation_type  = DecodeAccess(mc->__es.__err);
		info.access_violation_vaddr = reinterpret_cast<uint64_t>(si->si_addr);
	}

	info.rax = ss.__rax;
	info.rbx = ss.__rbx;
	info.rcx = ss.__rcx;
	info.rdx = ss.__rdx;
	info.rsi = ss.__rsi;
	info.rdi = ss.__rdi;
	info.rbp = ss.__rbp;
	info.rsp = ss.__rsp;
	info.r8  = ss.__r8;
	info.r9  = ss.__r9;
	info.r10 = ss.__r10;
	info.r11 = ss.__r11;
	info.r12 = ss.__r12;
	info.r13 = ss.__r13;
	info.r14 = ss.__r14;
	info.r15 = ss.__r15;
	// A SIGTRAP from int3 reports the address after the trap byte; back it onto the instruction
	// so ExceptionInfo::rip means the same thing here as it does on Windows.
	info.rip    = (info.type == ExceptionType::Breakpoint) ? ss.__rip - 1 : ss.__rip;
	info.rflags = ss.__rflags;

	const bool resolved   = DispatchToChain(info);
	g_in_exception_filter = false;

	if (resolved) {
		return; // retry the faulting instruction against the fixed mapping
	}

	// An unclaimed trap is not fatal and must not disarm the handler: unlike a fault, it does
	// not re-execute on return, so restoring SIG_DFL here would only kill the process on some
	// later, unrelated trap.
	if (sig == SIGTRAP) {
		return;
	}

	// Unresolved: restore the default action so the re-executed instruction terminates.
	struct sigaction dfl {};
	dfl.sa_handler = SIG_DFL;
	sigemptyset(&dfl.sa_mask);
	sigaction(sig, &dfl, nullptr);
}

#else

// x86-64 page-fault error bits.
constexpr uint64_t PAGE_FAULT_ERROR_WRITE       = 0x02;
constexpr uint64_t PAGE_FAULT_ERROR_INSTRUCTION = 0x10;

// Let the kernel handle an unresolved fault on retry.
static void ChainToDefault(int signal_number) noexcept {
	struct sigaction restore {};
	restore.sa_handler = SIG_DFL;
	sigemptyset(&restore.sa_mask);
	restore.sa_flags = 0;
	::sigaction(signal_number, &restore, nullptr);
}

static void SignalHandler(int signal_number, siginfo_t* signal_info, void* native_context) {
	FilterScope filter_scope;

	auto* context = static_cast<ucontext_t*>(native_context);
	auto* gregs   = context->uc_mcontext.gregs;

	ExceptionInfo info {};
	info.exception_address = static_cast<uint64_t>(gregs[REG_RIP]);
	info.native_code       = static_cast<uint32_t>(signal_number);
	info.native_context    = context;

	if (signal_number == SIGSEGV || signal_number == SIGBUS) {
		info.type             = ExceptionType::AccessViolation;
		const auto error_code = static_cast<uint64_t>(gregs[REG_ERR]);
		if ((error_code & PAGE_FAULT_ERROR_INSTRUCTION) != 0) {
			info.access_violation_type = AccessViolationType::Execute;
		} else if ((error_code & PAGE_FAULT_ERROR_WRITE) != 0) {
			info.access_violation_type = AccessViolationType::Write;
		} else {
			info.access_violation_type = AccessViolationType::Read;
		}
		info.access_violation_vaddr = reinterpret_cast<uint64_t>(signal_info->si_addr);
	} else if (signal_number == SIGILL) {
		info.type = ExceptionType::IllegalInstruction;
	} else if (signal_number == SIGTRAP) {
		info.type = (signal_info->si_code == TRAP_TRACE) ? ExceptionType::SingleStep
		                                                 : ExceptionType::Breakpoint;
	} else {
		ChainToDefault(signal_number);
		return;
	}

	info.rax = static_cast<uint64_t>(gregs[REG_RAX]);
	info.rbx = static_cast<uint64_t>(gregs[REG_RBX]);
	info.rcx = static_cast<uint64_t>(gregs[REG_RCX]);
	info.rdx = static_cast<uint64_t>(gregs[REG_RDX]);
	info.rsi = static_cast<uint64_t>(gregs[REG_RSI]);
	info.rdi = static_cast<uint64_t>(gregs[REG_RDI]);
	info.rbp = static_cast<uint64_t>(gregs[REG_RBP]);
	info.rsp = static_cast<uint64_t>(gregs[REG_RSP]);
	info.r8  = static_cast<uint64_t>(gregs[REG_R8]);
	info.r9  = static_cast<uint64_t>(gregs[REG_R9]);
	info.r10 = static_cast<uint64_t>(gregs[REG_R10]);
	info.r11 = static_cast<uint64_t>(gregs[REG_R11]);
	info.r12 = static_cast<uint64_t>(gregs[REG_R12]);
	info.r13 = static_cast<uint64_t>(gregs[REG_R13]);
	info.r14 = static_cast<uint64_t>(gregs[REG_R14]);
	info.r15 = static_cast<uint64_t>(gregs[REG_R15]);
	// A SIGTRAP from int3 reports the address after the trap byte; back it onto the instruction
	// so ExceptionInfo::rip means the same thing here as it does on Windows.
	info.rip =
	    static_cast<uint64_t>(gregs[REG_RIP]) - (info.type == ExceptionType::Breakpoint ? 1 : 0);
	info.rflags = static_cast<uint64_t>(gregs[REG_EFL]);

	if (DispatchToChain(info)) {
		return;
	}

	// An unclaimed trap is not fatal and must not disarm the handler (see the macOS path).
	if (signal_number == SIGTRAP) {
		return;
	}

	ChainToDefault(signal_number);
}

#endif

// Install the OS-level trap hook. Idempotent: only the first successful call arms it.
static bool EnsureInstalled() {
	uint32_t expected_state = 0;
	if (!g_install_state.compare_exchange_strong(expected_state, 1, std::memory_order_acq_rel)) {
		// Another registration already armed it, or is in the middle of doing so.
		return expected_state != 0;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (AddVectoredExceptionHandler(1, ExceptionFilter) == nullptr) {
		g_install_state.store(0, std::memory_order_release);
		printf("AddVectoredExceptionHandler() failed\n");
		return false;
	}
#elif defined(__APPLE__)
	struct sigaction sa {};
	sa.sa_sigaction = SignalHandler;
	sa.sa_flags     = SA_SIGINFO;
	sigemptyset(&sa.sa_mask);
	// The guest signal-dispatch path (KernelRaiseException) interrupts threads with
	// SIGUSR1; block it while a fault is being resolved so a stop-the-world request
	// cannot preempt the handler between the protection fix and the retry.
	sigaddset(&sa.sa_mask, SIGUSR1);

	// macOS raises SIGBUS for protection faults on some paths and SIGSEGV on others;
	// SIGILL covers instructions the host cannot execute (routed to the x64 emulator);
	// SIGTRAP carries debugger breakpoints and single-step traps.
	bool ok = sigaction(SIGSEGV, &sa, nullptr) == 0 && sigaction(SIGBUS, &sa, nullptr) == 0 &&
	          sigaction(SIGILL, &sa, nullptr) == 0 && sigaction(SIGTRAP, &sa, nullptr) == 0;
	if (!ok) {
		g_install_state.store(0, std::memory_order_release);
		printf("sigaction() failed to install the host fault handler\n");
		return false;
	}
#else
	struct sigaction action {};
	action.sa_sigaction = SignalHandler;
	sigemptyset(&action.sa_mask);
	// Fault resolution needs the normal thread stack.
	action.sa_flags = SA_SIGINFO | SA_RESTART;

	for (const int signal_number: {SIGSEGV, SIGBUS, SIGILL, SIGTRAP}) {
		if (::sigaction(signal_number, &action, nullptr) != 0) {
			g_install_state.store(0, std::memory_order_release);
			printf("sigaction(%d) failed\n", signal_number);
			return false;
		}
	}
#endif

	g_install_state.store(2, std::memory_order_release);
	return true;
}

bool AddHandler(Handler handler, int32_t priority) {
	if (handler == nullptr) {
		return false;
	}

	const std::lock_guard lock(g_register_mutex);

	HandlerTable next {};
	if (const auto* current = g_table.load(std::memory_order_acquire); current != nullptr) {
		next = *current;
	}

	for (uint32_t i = 0; i < next.count; i++) {
		if (next.entries[i].fn == handler) {
			return true; // already registered
		}
	}

	if (next.count >= MAX_HANDLERS) {
		return false;
	}

	next.entries[next.count++] = HandlerEntry {handler, priority};
	std::stable_sort(
	    next.entries, next.entries + next.count,
	    [](const HandlerEntry& a, const HandlerEntry& b) { return a.priority < b.priority; });

	if (!EnsureInstalled()) {
		return false;
	}

	PublishTable(next);
	return true;
}

bool RemoveHandler(Handler handler) {
	if (handler == nullptr) {
		return false;
	}

	const std::lock_guard lock(g_register_mutex);

	const auto* current = g_table.load(std::memory_order_acquire);
	if (current == nullptr) {
		return false;
	}

	HandlerTable next {};
	bool         removed = false;
	for (uint32_t i = 0; i < current->count; i++) {
		if (current->entries[i].fn == handler) {
			removed = true;
			continue;
		}
		next.entries[next.count++] = current->entries[i];
	}

	if (!removed) {
		return false;
	}

	PublishTable(next);
	return true;
}

bool InstallHandler(Handler handler) {
	return AddHandler(handler, PRIORITY_DEFAULT);
}

void SetInstructionPointer([[maybe_unused]] void* native_context, [[maybe_unused]] uint64_t rip) {
	if (native_context == nullptr) {
		return;
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static_cast<PCONTEXT>(native_context)->Rip = rip;
#elif defined(__APPLE__)
	static_cast<ucontext_t*>(native_context)->uc_mcontext->__ss.__rip = rip;
#else
	static_cast<ucontext_t*>(native_context)->uc_mcontext.gregs[REG_RIP] = static_cast<greg_t>(rip);
#endif
}

void SetFlagsRegister([[maybe_unused]] void* native_context, [[maybe_unused]] uint64_t rflags) {
	if (native_context == nullptr) {
		return;
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static_cast<PCONTEXT>(native_context)->EFlags = static_cast<DWORD>(rflags);
#elif defined(__APPLE__)
	static_cast<ucontext_t*>(native_context)->uc_mcontext->__ss.__rflags = rflags;
#else
	static_cast<ucontext_t*>(native_context)->uc_mcontext.gregs[REG_EFL] =
	    static_cast<greg_t>(rflags);
#endif
}

void SetGpr([[maybe_unused]] void* native_context, [[maybe_unused]] Gpr reg,
            [[maybe_unused]] uint64_t value) {
	if (native_context == nullptr) {
		return;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	auto* context = static_cast<PCONTEXT>(native_context);
	switch (reg) {
		case Gpr::Rax: context->Rax = value; break;
		case Gpr::Rbx: context->Rbx = value; break;
		case Gpr::Rcx: context->Rcx = value; break;
		case Gpr::Rdx: context->Rdx = value; break;
		case Gpr::Rsi: context->Rsi = value; break;
		case Gpr::Rdi: context->Rdi = value; break;
		case Gpr::Rbp: context->Rbp = value; break;
		case Gpr::Rsp: context->Rsp = value; break;
		case Gpr::R8: context->R8 = value; break;
		case Gpr::R9: context->R9 = value; break;
		case Gpr::R10: context->R10 = value; break;
		case Gpr::R11: context->R11 = value; break;
		case Gpr::R12: context->R12 = value; break;
		case Gpr::R13: context->R13 = value; break;
		case Gpr::R14: context->R14 = value; break;
		case Gpr::R15: context->R15 = value; break;
	}
#elif defined(__APPLE__)
	auto& ss = static_cast<ucontext_t*>(native_context)->uc_mcontext->__ss;
	switch (reg) {
		case Gpr::Rax: ss.__rax = value; break;
		case Gpr::Rbx: ss.__rbx = value; break;
		case Gpr::Rcx: ss.__rcx = value; break;
		case Gpr::Rdx: ss.__rdx = value; break;
		case Gpr::Rsi: ss.__rsi = value; break;
		case Gpr::Rdi: ss.__rdi = value; break;
		case Gpr::Rbp: ss.__rbp = value; break;
		case Gpr::Rsp: ss.__rsp = value; break;
		case Gpr::R8: ss.__r8 = value; break;
		case Gpr::R9: ss.__r9 = value; break;
		case Gpr::R10: ss.__r10 = value; break;
		case Gpr::R11: ss.__r11 = value; break;
		case Gpr::R12: ss.__r12 = value; break;
		case Gpr::R13: ss.__r13 = value; break;
		case Gpr::R14: ss.__r14 = value; break;
		case Gpr::R15: ss.__r15 = value; break;
	}
#else
	auto*      gregs = static_cast<ucontext_t*>(native_context)->uc_mcontext.gregs;
	const auto set   = [&](int index) { gregs[index] = static_cast<greg_t>(value); };
	switch (reg) {
		case Gpr::Rax: set(REG_RAX); break;
		case Gpr::Rbx: set(REG_RBX); break;
		case Gpr::Rcx: set(REG_RCX); break;
		case Gpr::Rdx: set(REG_RDX); break;
		case Gpr::Rsi: set(REG_RSI); break;
		case Gpr::Rdi: set(REG_RDI); break;
		case Gpr::Rbp: set(REG_RBP); break;
		case Gpr::Rsp: set(REG_RSP); break;
		case Gpr::R8: set(REG_R8); break;
		case Gpr::R9: set(REG_R9); break;
		case Gpr::R10: set(REG_R10); break;
		case Gpr::R11: set(REG_R11); break;
		case Gpr::R12: set(REG_R12); break;
		case Gpr::R13: set(REG_R13); break;
		case Gpr::R14: set(REG_R14); break;
		case Gpr::R15: set(REG_R15); break;
	}
#endif
}

} // namespace Common::HostException
