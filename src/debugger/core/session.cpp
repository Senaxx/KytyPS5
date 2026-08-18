#include "debugger/core/session.h"

#include "common/assert.h"
#include "common/hostException.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "debugger/core/dossier.h"
#include "debugger/symbols/symbols.h"
#include "debugger/target/memory.h"
#include "debugger/target/threads.h"
#include "debugger/ui/overlay.h"
#include "kernel/pthread.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace Debugger::Session {

namespace {

using Common::HostException::ExceptionInfo;
using Common::HostException::ExceptionType;

// ---- Armed-breakpoint table ---------------------------------------------------------------
//
// Read by the trap handler on every breakpoint exception, so it must be lock-free: a parked
// guest thread must never be able to hold a lock the handler needs. `address` is the publish
// gate — it is stored last when arming and cleared first when disarming, so a half-written slot
// is never observable.
struct ArmedSlot {
	std::atomic<uint64_t> address {0};
	std::atomic<uint32_t> id {0};
	std::atomic<uint8_t>  original_byte {0};
};

ArmedSlot g_armed[MAX_BREAKPOINTS];

bool FindArmed(uint64_t address, uint32_t& id_out, uint8_t& byte_out) {
	for (auto& slot: g_armed) {
		if (slot.address.load(std::memory_order_acquire) == address) {
			id_out   = slot.id.load(std::memory_order_relaxed);
			byte_out = slot.original_byte.load(std::memory_order_relaxed);
			return true;
		}
	}
	return false;
}

bool PublishArmed(uint64_t address, uint32_t id, uint8_t original_byte) {
	for (auto& slot: g_armed) {
		if (slot.address.load(std::memory_order_relaxed) != 0) {
			continue;
		}
		slot.id.store(id, std::memory_order_relaxed);
		slot.original_byte.store(original_byte, std::memory_order_relaxed);
		slot.address.store(address, std::memory_order_release);
		return true;
	}
	return false;
}

void UnpublishArmed(uint64_t address) {
	for (auto& slot: g_armed) {
		if (slot.address.load(std::memory_order_relaxed) == address) {
			slot.address.store(0, std::memory_order_release);
			return;
		}
	}
}

// ---- Per-thread stop state ----------------------------------------------------------------

struct ThreadState {
	int               unique_id = 0;
	uint64_t          host_tid  = 0;
	std::atomic<bool> parked {false};
	std::atomic<bool> in_use {false};
	ResumeMode        resume_mode   = ResumeMode::Continue;
	StopReason        reason        = StopReason::None;
	uint32_t          breakpoint_id = 0;
	uint64_t          address       = 0;
	Registers         regs;
	std::atomic<bool> regs_dirty {false};
	Registers         pending_regs;
	void*             native_context = nullptr;

	// Set while the thread is mid single-step, so an unrelated single-step trap (from an
	// attached native debugger, say) is not mistaken for ours.
	bool expecting_single_step = false;
	// Breakpoint temporarily lifted so the thread can execute the real instruction under it;
	// re-armed on the following single-step trap.
	uint64_t step_over_bp_address = 0;
	// Stop again once that single step completes (step-into), as opposed to just resuming.
	bool stop_after_step = false;
	// Set for StopReason::Fatal.
	std::string message;
};

constexpr uint32_t MAX_THREAD_STATES = 128;

ThreadState               g_thread_states[MAX_THREAD_STATES];
thread_local ThreadState* t_state = nullptr;

// Claim a state slot for `unique_id`, reusing one already registered for that thread. Resume()
// pre-registers a slot for an OS-suspended thread before setting its trap flag, so the trap that
// follows has to find that slot rather than allocating a second one.
ThreadState* AcquireThreadStateFor(int unique_id, uint64_t host_tid) {
	for (auto& state: g_thread_states) {
		if (state.in_use.load(std::memory_order_acquire) && state.unique_id == unique_id) {
			return &state;
		}
	}

	for (auto& state: g_thread_states) {
		bool expected = false;
		if (state.in_use.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
			state.unique_id = unique_id;
			state.host_tid  = host_tid;
			return &state;
		}
	}

	return nullptr;
}

ThreadState* AcquireThreadState() {
	if (t_state != nullptr) {
		return t_state;
	}

	t_state =
	    AcquireThreadStateFor(Common::Thread::GetThreadIdUnique(), Target::CurrentHostThreadId());
	return t_state;
}

// ---- Session state ------------------------------------------------------------------------

std::atomic<bool> g_enabled {false};
std::atomic<bool> g_break_on_fatal {true};
// Bumped by the overlay every time it draws, so a fatal halt can tell whether presentation is
// still alive or whether it has just deadlocked itself.
std::atomic<uint64_t> g_overlay_frames {0};
std::atomic<uint32_t> g_stopped_count {0};
std::atomic<uint32_t> g_next_breakpoint_id {1};

std::mutex              g_breakpoints_mutex;
std::vector<Breakpoint> g_breakpoints;

// Suspended (Windows) rather than parked-in-handler threads.
struct SuspendedThread {
	Target::SuspendedHandle handle;
	int                     unique_id = 0;
	Registers               regs;
};

std::mutex                   g_suspended_mutex;
std::vector<SuspendedThread> g_suspended;

Registers FromExceptionInfo(const ExceptionInfo& info) {
	Registers regs {};
	regs.rax    = info.rax;
	regs.rbx    = info.rbx;
	regs.rcx    = info.rcx;
	regs.rdx    = info.rdx;
	regs.rsi    = info.rsi;
	regs.rdi    = info.rdi;
	regs.rbp    = info.rbp;
	regs.rsp    = info.rsp;
	regs.r8     = info.r8;
	regs.r9     = info.r9;
	regs.r10    = info.r10;
	regs.r11    = info.r11;
	regs.r12    = info.r12;
	regs.r13    = info.r13;
	regs.r14    = info.r14;
	regs.r15    = info.r15;
	regs.rip    = info.rip;
	regs.rflags = info.rflags;
	regs.valid  = true;
	return regs;
}

void ApplyRegistersToContext(void* context, const Registers& regs) {
	using Common::HostException::Gpr;

	Common::HostException::SetGpr(context, Gpr::Rax, regs.rax);
	Common::HostException::SetGpr(context, Gpr::Rbx, regs.rbx);
	Common::HostException::SetGpr(context, Gpr::Rcx, regs.rcx);
	Common::HostException::SetGpr(context, Gpr::Rdx, regs.rdx);
	Common::HostException::SetGpr(context, Gpr::Rsi, regs.rsi);
	Common::HostException::SetGpr(context, Gpr::Rdi, regs.rdi);
	Common::HostException::SetGpr(context, Gpr::Rbp, regs.rbp);
	Common::HostException::SetGpr(context, Gpr::Rsp, regs.rsp);
	Common::HostException::SetGpr(context, Gpr::R8, regs.r8);
	Common::HostException::SetGpr(context, Gpr::R9, regs.r9);
	Common::HostException::SetGpr(context, Gpr::R10, regs.r10);
	Common::HostException::SetGpr(context, Gpr::R11, regs.r11);
	Common::HostException::SetGpr(context, Gpr::R12, regs.r12);
	Common::HostException::SetGpr(context, Gpr::R13, regs.r13);
	Common::HostException::SetGpr(context, Gpr::R14, regs.r14);
	Common::HostException::SetGpr(context, Gpr::R15, regs.r15);
	Common::HostException::SetInstructionPointer(context, regs.rip);
}

// ---- Breakpoint arming --------------------------------------------------------------------

constexpr uint8_t INT3 = 0xCC;

bool ArmLocked(Breakpoint& bp) {
	if (bp.armed || !bp.enabled || bp.address == 0) {
		return bp.armed;
	}

	uint8_t original = 0;
	if (!Target::SafeRead(bp.address, &original, 1)) {
		return false;
	}

	// Re-arming over our own patch would capture 0xCC as the "original" byte and turn the
	// breakpoint into a permanent trap.
	if (original == INT3) {
		uint32_t existing_id   = 0;
		uint8_t  existing_byte = 0;
		if (FindArmed(bp.address, existing_id, existing_byte)) {
			original = existing_byte;
		} else {
			return false;
		}
	}

	if (!PublishArmed(bp.address, bp.id, original)) {
		return false;
	}

	if (!Target::WriteCode(bp.address, &INT3, 1)) {
		UnpublishArmed(bp.address);
		return false;
	}

	bp.original_byte = original;
	bp.armed         = true;
	return true;
}

void DisarmLocked(Breakpoint& bp) {
	if (!bp.armed) {
		return;
	}

	Target::WriteCode(bp.address, &bp.original_byte, 1);
	UnpublishArmed(bp.address);
	bp.armed = false;
}

// Lift a breakpoint byte without touching bookkeeping, so a thread can step over it.
void LiftForStep(uint64_t address) {
	uint32_t id            = 0;
	uint8_t  original_byte = 0;
	if (FindArmed(address, id, original_byte)) {
		Target::WriteCode(address, &original_byte, 1);
	}
}

void RestoreAfterStep(uint64_t address) {
	uint32_t id            = 0;
	uint8_t  original_byte = 0;
	if (FindArmed(address, id, original_byte)) {
		Target::WriteCode(address, &INT3, 1);
	}
}

void BumpHitCount(uint32_t id) {
	const std::lock_guard lock(g_breakpoints_mutex);
	for (auto& bp: g_breakpoints) {
		if (bp.id == id) {
			bp.hit_count++;
			return;
		}
	}
}

bool IsOneShot(uint32_t id) {
	const std::lock_guard lock(g_breakpoints_mutex);
	for (const auto& bp: g_breakpoints) {
		if (bp.id == id) {
			return bp.one_shot;
		}
	}
	return false;
}

// Whether this thread is the one that should halt at this breakpoint. A step-over one-shot
// belongs to the thread being stepped; anything else stops whoever arrives.
bool ShouldStopHere(uint32_t id, int unique_id) {
	const std::lock_guard lock(g_breakpoints_mutex);
	for (const auto& bp: g_breakpoints) {
		if (bp.id == id) {
			return bp.enabled && (bp.owner_unique_id == 0 || bp.owner_unique_id == unique_id);
		}
	}
	// Unknown id: the breakpoint was removed between the trap and this lookup. Stepping over it
	// transparently is the safe answer.
	return false;
}

// One-shot breakpoint used internally by step-over and step-out. Deliberately not
// AddBreakpointAt: that symbolises the address, which takes the runtime linker's lock and
// allocates, and this runs inside the trap handler on a parked thread.
uint32_t AddInternalBreakpoint(uint64_t address, int owner_unique_id);

// True when a plausible return address can be found for `regs`, i.e. a stack slot holding an
// address that decodes as code and is immediately preceded by a call.
//
// Guest code is frequently compiled without frame pointers, so [rbp+8] is usually not a return
// address at all. Planting a breakpoint on whatever it happens to hold writes 0xCC into
// unrelated memory, which is how step-out used to take the process down.
bool LooksLikeReturnAddress(uint64_t address) {
	if (address == 0 || Disasm::InstructionLength(address) == 0) {
		return false;
	}

	// x86-64 call encodings run from 2 to 7 bytes; requiring the decode to end exactly on
	// `address` is what makes this a real check rather than a guess.
	for (uint32_t back = 2; back <= 7; back++) {
		if (back > address) {
			break;
		}
		uint32_t length = 0;
		if (Disasm::IsCall(address - back, length) && length == back) {
			return true;
		}
	}
	return false;
}

bool FindReturnAddress(const Registers& regs, uint64_t& address_out) {
	// A frame-pointer-based frame is the cheap case, so try it before scanning.
	uint64_t candidate = 0;
	if (regs.rbp != 0 && Target::SafeRead(regs.rbp + 8, &candidate, sizeof(candidate)) &&
	    LooksLikeReturnAddress(candidate)) {
		address_out = candidate;
		return true;
	}

	constexpr int MAX_STACK_SLOTS = 256;
	for (int slot = 0; slot < MAX_STACK_SLOTS; slot++) {
		const uint64_t at = regs.rsp + static_cast<uint64_t>(slot) * sizeof(uint64_t);
		if (!Target::SafeRead(at, &candidate, sizeof(candidate))) {
			break;
		}
		if (LooksLikeReturnAddress(candidate)) {
			address_out = candidate;
			return true;
		}
	}

	return false;
}

// Work out how to advance a thread for `mode`, planting any one-shot breakpoint the mode needs.
// Returns true when the thread should be single-stepped (the caller sets the trap flag).
bool PrepareStep(ResumeMode mode, const Registers& regs, int owner_unique_id) {
	switch (mode) {
		case ResumeMode::Continue: return false;

		case ResumeMode::StepInto: return true;

		case ResumeMode::StepOver: {
			uint32_t length = 0;
			if (Disasm::IsCall(regs.rip, length) && length != 0) {
				// Run the call at full speed and catch it coming back.
				AddInternalBreakpoint(regs.rip + length, owner_unique_id);
				return false;
			}
			// Anything else advances by one instruction anyway.
			return true;
		}

		case ResumeMode::StepOut: {
			uint64_t return_address = 0;
			if (FindReturnAddress(regs, return_address)) {
				AddInternalBreakpoint(return_address, owner_unique_id);
			}
			// No credible return address: behave as continue rather than guessing.
			return false;
		}
	}

	return false;
}

// ---- Parking --------------------------------------------------------------------------------

// Park the calling thread inside the trap handler until the UI resumes it. Deliberately a
// sleep loop rather than a condition variable: this runs inside a POSIX signal handler and a
// Windows vectored handler, where taking a lock risks deadlocking against whatever the thread
// was doing when it trapped.
void ParkCurrentThread(ThreadState* state, bool fatal = false) {
	state->parked.store(true, std::memory_order_release);
	g_stopped_count.fetch_add(1, std::memory_order_acq_rel);

	// A fatal error can land on any thread, including the ones that draw the overlay or feed the
	// GPU. Parking those may block presentation, and a debugger that hangs the process is worse
	// than one that misses a stop — so watch for the overlay actually drawing. If it never comes
	// up, let go and allow the exit to proceed as it did before.
	const auto start           = std::chrono::steady_clock::now();
	const auto frames_at_start = g_overlay_frames.load(std::memory_order_relaxed);
	bool       overlay_seen    = false;

	constexpr auto OVERLAY_GRACE = std::chrono::seconds(5);

	while (state->parked.load(std::memory_order_acquire)) {
		Common::Thread::Sleep(2);

		if (!fatal || overlay_seen) {
			continue;
		}

		if (g_overlay_frames.load(std::memory_order_relaxed) != frames_at_start) {
			overlay_seen = true; // it is alive, so wait as long as the user wants
			continue;
		}

		if (std::chrono::steady_clock::now() - start > OVERLAY_GRACE) {
			LOGF_COLOR(Log::Color::BrightYellow,
			           "Debugger: the overlay did not draw within 5s of the fatal halt, so this "
			           "thread is probably the one that presents it; releasing the halt and "
			           "letting the process exit\n");
			state->parked.store(false, std::memory_order_release);
			break;
		}
	}

	g_stopped_count.fetch_sub(1, std::memory_order_acq_rel);
}

// Prepare the native context for resumption and return. Runs on the parked thread after the UI
// has chosen a resume mode.
void ApplyResume(ThreadState* state, void* context) {
	if (state->regs_dirty.exchange(false, std::memory_order_acq_rel)) {
		ApplyRegistersToContext(context, state->pending_regs);
		state->regs = state->pending_regs;
	}

	const uint64_t resume_ip = state->regs.rip;
	uint64_t       flags     = state->regs.rflags;

	state->step_over_bp_address  = 0;
	state->stop_after_step       = false;
	state->expecting_single_step = false;

	// If the thread is sitting on an armed breakpoint it cannot simply continue: the 0xCC is
	// still there. Lift it, single-step the real instruction, and re-arm on the resulting trap.
	uint32_t   id            = 0;
	uint8_t    original_byte = 0;
	const bool on_breakpoint = FindArmed(resume_ip, id, original_byte);

	const bool want_step = PrepareStep(state->resume_mode, state->regs, state->unique_id);

	if (on_breakpoint) {
		LiftForStep(resume_ip);
		state->step_over_bp_address  = resume_ip;
		state->expecting_single_step = true;
		state->stop_after_step       = want_step;
		flags |= Common::HostException::FLAG_TRAP;
	} else if (want_step) {
		state->expecting_single_step = true;
		state->stop_after_step       = true;
		flags |= Common::HostException::FLAG_TRAP;
	}

	Common::HostException::SetFlagsRegister(context, flags);
}

void StopAndPark(ThreadState* state, uint64_t address, StopReason reason, uint32_t breakpoint_id,
                 const Registers& regs, void* context, bool fatal = false) {
	state->address        = address;
	state->reason         = reason;
	state->breakpoint_id  = breakpoint_id;
	state->regs           = regs;
	state->native_context = context;
	state->resume_mode    = ResumeMode::Continue;

	ParkCurrentThread(state, fatal);

	// A null context means the stop cannot be resumed from — a fatal error on its way to
	// terminating the process. There is nothing to prepare, and touching the breakpoint tables
	// or the trap flag on the way out would be pointless at best.
	if (context != nullptr) {
		ApplyResume(state, context);
	}

	state->native_context = nullptr;
	state->reason         = StopReason::None;
	state->message.clear();
}

// ---- Trap handler ---------------------------------------------------------------------------

bool OnTrap(const ExceptionInfo& info) {
	if (!g_enabled.load(std::memory_order_acquire)) {
		return false;
	}

	if (info.type == ExceptionType::Breakpoint) {
		// ExceptionInfo::rip is normalised to the trap instruction itself, so no adjustment is
		// needed here — the platforms disagree natively and hostException hides that.
		const uint64_t address = info.rip;

		uint32_t id            = 0;
		uint8_t  original_byte = 0;
		if (!FindArmed(address, id, original_byte)) {
			return false; // not one of ours
		}

		auto* state = AcquireThreadState();
		if (state == nullptr) {
			return false;
		}

		Common::HostException::SetInstructionPointer(info.native_context, address);

		// Not this thread's breakpoint (a step-over one-shot belonging to another thread, or one
		// disabled since the trap). It still has to get past the 0xCC, so lift it, single-step
		// the real instruction and re-arm — the same dance as resuming, without halting.
		if (!ShouldStopHere(id, state->unique_id)) {
			LiftForStep(address);
			state->step_over_bp_address  = address;
			state->expecting_single_step = true;
			state->stop_after_step       = false;
			Common::HostException::SetFlagsRegister(info.native_context,
			                                        info.rflags | Common::HostException::FLAG_TRAP);
			return true;
		}

		Registers regs = FromExceptionInfo(info);
		regs.rip       = address;

		BumpHitCount(id);

		const bool one_shot = IsOneShot(id);

		StopAndPark(state, address, StopReason::Breakpoint, id, regs, info.native_context);

		if (one_shot) {
			RemoveBreakpoint(id);
		}

		return true;
	}

	if (info.type == ExceptionType::SingleStep) { // NOLINT(readability-misleading-indentation)
		auto* state = AcquireThreadState();
		if (state == nullptr || !state->expecting_single_step) {
			return false; // somebody else's single step
		}

		state->expecting_single_step = false;

		// Clear TF so the thread does not keep trapping.
		Common::HostException::SetFlagsRegister(info.native_context,
		                                        info.rflags & ~Common::HostException::FLAG_TRAP);

		if (state->step_over_bp_address != 0) {
			RestoreAfterStep(state->step_over_bp_address);
			state->step_over_bp_address = 0;
		}

		if (state->stop_after_step) {
			state->stop_after_step = false;
			StopAndPark(state, info.rip, StopReason::Step, 0, FromExceptionInfo(info),
			            info.native_context);
		}

		return true;
	}

	return false;
}

// Last chance before the process dies: this runs after every other fault handler has declined,
// so anything reaching it is a crash nobody resolved. Registered behind PRIORITY_DEFAULT rather
// than at the debugger's usual slot precisely so it cannot swallow the GPU tracker's page
// faults, which are resolved earlier in the chain.
bool OnUnhandledFault(const ExceptionInfo& info) {
	if (!g_enabled.load(std::memory_order_acquire) ||
	    !g_break_on_fatal.load(std::memory_order_acquire)) {
		return false;
	}

	if (info.type != ExceptionType::AccessViolation &&
	    info.type != ExceptionType::IllegalInstruction) {
		return false;
	}

	auto* state = AcquireThreadState();
	if (state == nullptr) {
		return false;
	}

	char message[256] {};
	std::snprintf(
	    message, sizeof(message), "Unhandled %s at 0x%016llx (faulting address 0x%016llx)",
	    info.type == ExceptionType::AccessViolation ? "access violation" : "illegal instruction",
	    static_cast<unsigned long long>(info.rip),
	    static_cast<unsigned long long>(info.access_violation_vaddr));
	state->message = message;

	Ui::SetVisible(true);

	// Null context: resuming would only re-run the faulting instruction and trap again, so the
	// halt is for inspection and then the process terminates as it would have.
	StopAndPark(state, info.rip, StopReason::Fatal, 0, FromExceptionInfo(info), nullptr, true);

	return false;
}

} // namespace

// ---- Public API -------------------------------------------------------------------------

bool Initialize() {
	if (g_enabled.load(std::memory_order_acquire)) {
		return true;
	}

	if (!Common::HostException::AddHandler(OnTrap, Common::HostException::PRIORITY_DEBUGGER)) {
		LOGF_COLOR(Log::Color::BrightRed, "Debugger: could not register the trap handler\n");
		return false;
	}

	// Behind everything else, so it only sees faults no other handler resolved.
	Common::HostException::AddHandler(OnUnhandledFault,
	                                  Common::HostException::PRIORITY_DEFAULT + 100);

	Common::SetFatalHandler(&ReportFatal);

	Target::InstallStopSignal();

	g_enabled.store(true, std::memory_order_release);
	return true;
}

void Shutdown() {
	if (!g_enabled.exchange(false, std::memory_order_acq_rel)) {
		return;
	}

	ResumeAll();
	ClearBreakpoints();

	Common::SetFatalHandler(nullptr);
	Common::HostException::RemoveHandler(OnUnhandledFault);
	Common::HostException::RemoveHandler(OnTrap);
}

bool IsEnabled() {
	return g_enabled.load(std::memory_order_acquire);
}

bool IsPaused() {
	if (g_stopped_count.load(std::memory_order_acquire) != 0) {
		return true;
	}
	const std::lock_guard lock(g_suspended_mutex);
	return !g_suspended.empty();
}

uint32_t AddBreakpointAt(uint64_t address, bool one_shot) {
	if (address == 0) {
		return 0;
	}

	const std::lock_guard lock(g_breakpoints_mutex);

	for (const auto& existing: g_breakpoints) {
		if (existing.address == address && existing.one_shot == one_shot) {
			return existing.id;
		}
	}

	Breakpoint bp {};
	bp.id       = g_next_breakpoint_id.fetch_add(1, std::memory_order_relaxed);
	bp.address  = address;
	bp.one_shot = one_shot;
	bp.enabled  = true;
	bp.label    = Symbols::Format(address);

	if (!ArmLocked(bp)) {
		LOGF_COLOR(Log::Color::BrightYellow,
		           "Debugger: could not arm a breakpoint at 0x%016" PRIx64 "\n", address);
		return 0;
	}

	g_breakpoints.push_back(bp);
	return bp.id;
}

namespace {

uint32_t AddInternalBreakpoint(uint64_t address, int owner_unique_id) {
	if (address == 0) {
		return 0;
	}

	const std::lock_guard lock(g_breakpoints_mutex);

	for (const auto& existing: g_breakpoints) {
		if (existing.address == address && existing.owner_unique_id == owner_unique_id) {
			return existing.id;
		}
	}

	Breakpoint bp {};
	bp.id              = g_next_breakpoint_id.fetch_add(1, std::memory_order_relaxed);
	bp.address         = address;
	bp.one_shot        = true;
	bp.enabled         = true;
	bp.owner_unique_id = owner_unique_id;
	// No label: symbolising takes the runtime linker's lock and allocates, and this can run
	// inside the trap handler.

	if (!ArmLocked(bp)) {
		return 0;
	}

	g_breakpoints.push_back(bp);
	return bp.id;
}

} // namespace

uint32_t AddBreakpoint(const std::string& location, bool one_shot) {
	uint64_t address = 0;
	if (Symbols::Resolve(location, address)) {
		return AddBreakpointAt(address, one_shot);
	}

	// Keep it for later: the target module may not be mapped yet.
	const std::lock_guard lock(g_breakpoints_mutex);

	Breakpoint bp {};
	bp.id       = g_next_breakpoint_id.fetch_add(1, std::memory_order_relaxed);
	bp.one_shot = one_shot;
	bp.enabled  = true;
	bp.pending  = true;
	bp.label    = location;

	g_breakpoints.push_back(bp);
	return bp.id;
}

bool RemoveBreakpoint(uint32_t id) {
	const std::lock_guard lock(g_breakpoints_mutex);

	const auto it = std::find_if(g_breakpoints.begin(), g_breakpoints.end(),
	                             [id](const Breakpoint& bp) { return bp.id == id; });
	if (it == g_breakpoints.end()) {
		return false;
	}

	DisarmLocked(*it);
	g_breakpoints.erase(it);
	return true;
}

bool SetBreakpointEnabled(uint32_t id, bool enabled) {
	const std::lock_guard lock(g_breakpoints_mutex);

	for (auto& bp: g_breakpoints) {
		if (bp.id != id) {
			continue;
		}

		if (enabled == bp.enabled) {
			return true;
		}

		bp.enabled = enabled;
		if (enabled) {
			return ArmLocked(bp);
		}
		DisarmLocked(bp);
		return true;
	}

	return false;
}

void ClearBreakpoints() {
	const std::lock_guard lock(g_breakpoints_mutex);

	for (auto& bp: g_breakpoints) {
		DisarmLocked(bp);
	}
	g_breakpoints.clear();
}

std::vector<Breakpoint> Breakpoints() {
	const std::lock_guard lock(g_breakpoints_mutex);
	return g_breakpoints;
}

void ResolvePendingBreakpoints() {
	const std::lock_guard lock(g_breakpoints_mutex);

	for (auto& bp: g_breakpoints) {
		if (!bp.pending) {
			continue;
		}

		uint64_t address = 0;
		if (!Symbols::Resolve(bp.label, address)) {
			continue;
		}

		bp.address = address;
		if (ArmLocked(bp)) {
			bp.pending = false;
			LOGF("Debugger: breakpoint %u resolved to %s\n", bp.id,
			     Symbols::Format(address).c_str());
		}
	}
}

void Pause() {
	if (!g_enabled.load(std::memory_order_acquire)) {
		return;
	}

	std::vector<Libs::LibKernel::GuestThreadInfo> threads;
	Libs::LibKernel::PthreadEnumerate(threads);

	const uint64_t self = Target::CurrentHostThreadId();

	for (const auto& thread: threads) {
		if (!thread.alive || thread.host_thread_id == 0 || thread.host_thread_id == self) {
			continue;
		}

		if (Target::CanSuspend()) {
			Target::SuspendedHandle handle {};
			if (!Target::Suspend(thread.host_thread_id, handle)) {
				continue;
			}

			SuspendedThread entry {};
			entry.handle    = handle;
			entry.unique_id = thread.unique_id;
			if (!Target::ReadRegisters(handle, entry.regs)) {
				Target::Resume(handle);
				continue;
			}

			const std::lock_guard lock(g_suspended_mutex);
			g_suspended.push_back(entry);
		} else {
			Target::RequestTrap(thread.host_thread_id);
		}
	}
}

void Resume(int unique_id, ResumeMode mode) {
	// Parked threads first.
	for (auto& state: g_thread_states) {
		if (!state.in_use.load(std::memory_order_acquire) ||
		    !state.parked.load(std::memory_order_acquire)) {
			continue;
		}
		if (unique_id != 0 && state.unique_id != unique_id) {
			continue;
		}

		state.resume_mode = mode;
		state.parked.store(false, std::memory_order_release);
	}

	// Then OS-suspended ones.
	std::vector<SuspendedThread> resumed;
	{
		const std::lock_guard lock(g_suspended_mutex);

		for (auto it = g_suspended.begin(); it != g_suspended.end();) {
			if (unique_id != 0 && it->unique_id != unique_id) {
				++it;
				continue;
			}
			resumed.push_back(*it);
			it = g_suspended.erase(it);
		}
	}

	for (auto& entry: resumed) {
		// A step on an OS-suspended thread is served by planting whatever the mode needs and, if
		// it is a single step, setting the trap flag so the thread traps straight back into the
		// handler and parks there with the full stop machinery.
		//
		// The state slot must be claimed and marked *before* the thread runs: without it the
		// resulting single-step trap belongs to nobody, OnTrap declines it, and an unhandled
		// EXCEPTION_SINGLE_STEP takes the process down.
		if (mode != ResumeMode::Continue) {
			if (auto* state = AcquireThreadStateFor(entry.unique_id, entry.handle.host_tid);
			    state != nullptr) {
				const bool single_step = PrepareStep(mode, entry.regs, entry.unique_id);

				state->step_over_bp_address  = 0;
				state->expecting_single_step = single_step;
				state->stop_after_step       = single_step;

				if (single_step) {
					Registers regs = entry.regs;
					regs.rflags |= Common::HostException::FLAG_TRAP;
					Target::WriteRegisters(entry.handle, regs);
				}
			}
		}
		Target::Resume(entry.handle);
	}
}

void ResumeAll() {
	Resume(0, ResumeMode::Continue);
}

std::vector<StoppedThread> Stopped() {
	std::vector<StoppedThread> out;

	for (auto& state: g_thread_states) {
		if (!state.in_use.load(std::memory_order_acquire) ||
		    !state.parked.load(std::memory_order_acquire)) {
			continue;
		}

		StoppedThread stopped {};
		stopped.unique_id     = state.unique_id;
		stopped.host_tid      = state.host_tid;
		stopped.address       = state.address;
		stopped.reason        = state.reason;
		stopped.breakpoint_id = state.breakpoint_id;
		stopped.regs          = state.regs;
		stopped.message       = state.message;
		out.push_back(stopped);
	}

	{
		const std::lock_guard lock(g_suspended_mutex);
		for (const auto& entry: g_suspended) {
			StoppedThread stopped {};
			stopped.unique_id = entry.unique_id;
			stopped.host_tid  = entry.handle.host_tid;
			stopped.address   = entry.regs.rip;
			stopped.reason    = StopReason::Pause;
			stopped.regs      = entry.regs;
			out.push_back(stopped);
		}
	}

	return out;
}

bool WriteRegisters(int unique_id, const Registers& regs) {
	for (auto& state: g_thread_states) {
		if (!state.in_use.load(std::memory_order_acquire) ||
		    !state.parked.load(std::memory_order_acquire) || state.unique_id != unique_id) {
			continue;
		}

		state.pending_regs = regs;
		state.regs_dirty.store(true, std::memory_order_release);
		return true;
	}

	const std::lock_guard lock(g_suspended_mutex);
	for (auto& entry: g_suspended) {
		if (entry.unique_id == unique_id) {
			entry.regs = regs;
			return Target::WriteRegisters(entry.handle, regs);
		}
	}

	return false;
}

bool ReadMemory(uint64_t address, void* dst, size_t size) {
	if (!Target::SafeRead(address, dst, size)) {
		return false;
	}

	// Show the instruction stream the guest would have executed, not our 0xCC patches.
	auto* bytes = static_cast<uint8_t*>(dst);
	for (auto& slot: g_armed) {
		const uint64_t patched = slot.address.load(std::memory_order_acquire);
		if (patched == 0 || patched < address || patched >= address + size) {
			continue;
		}
		bytes[patched - address] = slot.original_byte.load(std::memory_order_relaxed);
	}

	return true;
}

bool WriteMemory(uint64_t address, const void* src, size_t size) {
	if (src == nullptr || size == 0) {
		return false;
	}

	const auto* bytes = static_cast<const uint8_t*>(src);

	// A byte with a breakpoint planted on it holds 0xCC, not the guest's instruction. Writing
	// straight through would overwrite the patch — the breakpoint would silently stop firing,
	// and lifting it later would restore the pre-edit byte and lose the write. So the edit goes
	// into the breakpoint's saved original, and the patch is put back afterwards; the read
	// overlay then shows exactly what was written.
	{
		const std::lock_guard lock(g_breakpoints_mutex);

		for (auto& slot: g_armed) {
			const uint64_t patched = slot.address.load(std::memory_order_acquire);
			if (patched == 0 || patched < address || patched >= address + size) {
				continue;
			}

			const uint8_t replacement = bytes[patched - address];
			slot.original_byte.store(replacement, std::memory_order_relaxed);

			const uint32_t id = slot.id.load(std::memory_order_relaxed);
			for (auto& bp: g_breakpoints) {
				if (bp.id == id) {
					bp.original_byte = replacement;
					break;
				}
			}
		}
	}

	if (!Target::SafeWrite(address, src, size)) {
		return false;
	}

	for (auto& slot: g_armed) {
		const uint64_t patched = slot.address.load(std::memory_order_acquire);
		if (patched == 0 || patched < address || patched >= address + size) {
			continue;
		}
		Target::WriteCode(patched, &INT3, 1);
	}

	return true;
}

std::vector<Frame> Backtrace(int unique_id, uint32_t max_frames) {
	std::vector<Frame> frames;

	Registers regs {};
	for (const auto& stopped: Stopped()) {
		if (stopped.unique_id == unique_id) {
			regs = stopped.regs;
			break;
		}
	}

	if (!regs.valid) {
		return frames;
	}

	frames.push_back(Frame {regs.rip, regs.rbp, Symbols::Format(regs.rip)});

	uint64_t frame_ptr = regs.rbp;

	for (uint32_t i = 1; i < max_frames && frame_ptr != 0; i++) {
		uint64_t next_frame     = 0;
		uint64_t return_address = 0;

		if (!Target::SafeRead(frame_ptr, &next_frame, sizeof(next_frame)) ||
		    !Target::SafeRead(frame_ptr + 8, &return_address, sizeof(return_address))) {
			break;
		}

		if (return_address == 0 || next_frame <= frame_ptr) {
			break; // chain is not walkable any further
		}

		frames.push_back(Frame {return_address, next_frame, Symbols::Format(return_address)});
		frame_ptr = next_frame;
	}

	return frames;
}

void NotifyOverlayDrawn() {
	g_overlay_frames.fetch_add(1, std::memory_order_relaxed);
}

bool BreakOnFatalEnabled() {
	return g_break_on_fatal.load(std::memory_order_acquire);
}

void SetBreakOnFatal(bool enabled) {
	g_break_on_fatal.store(enabled, std::memory_order_release);
}

void ReportFatal(const char* report) {
	if (!g_enabled.load(std::memory_order_acquire) || !BreakOnFatalEnabled()) {
		return;
	}

	// Any thread, not just guest threads: the most useful stops are emulator asserts, and those
	// land on the GPU and renderer threads. Whether the overlay survives parking this particular
	// thread is not knowable here, so ParkCurrentThread watches for it and gives up if it never
	// draws.
	auto* state = AcquireThreadState();
	if (state == nullptr) {
		return;
	}

	Registers regs {};
	Target::CaptureRegisters(regs);

	state->message = report != nullptr ? report : "";

	// Written before parking, because a fatal on the presenting thread cannot show the overlay at
	// all — the renderer holds its own mutex across the frame it died in, so nothing else can
	// draw. The dossier is what makes that case still useful.
	if (const auto path = Dossier::Write(state->message, regs); !path.empty()) {
		LOGF_COLOR(Log::Color::Cyan, "Debugger: crash dossier written to %s\n", path.c_str());
	}

	LOGF_COLOR(Log::Color::Cyan,
	           "Debugger: halted on a fatal error; continue to let the process exit\n");

	// Nothing to see the halt on if the overlay is hidden.
	Ui::SetVisible(true);

	// No context: this stop cannot be stepped or resumed past, only inspected.
	StopAndPark(state, regs.rip, StopReason::Fatal, 0, regs, nullptr, true);
}

void OnGuestEntry(uint64_t entry_address) {
	if (!g_enabled.load(std::memory_order_acquire)) {
		return;
	}

	ResolvePendingBreakpoints();

	if (entry_address != 0) {
		LOGF("Debugger: guest entry at %s\n", Symbols::Format(entry_address).c_str());
	}
}

} // namespace Debugger::Session
