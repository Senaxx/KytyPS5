#ifndef EMULATOR_SRC_DEBUGGER_DEBUGGER_H_
#define EMULATOR_SRC_DEBUGGER_DEBUGGER_H_

#include "common/common.h"

#include <cstdint>

union SDL_Event;

namespace Debugger {

// Subsystem entry points. Initialize() checks Config::DebuggerEnabled() first and returns
// immediately when --debugger was not passed, so the cost of building the debugger in is a
// single branch at startup plus one relaxed atomic load on each hot-path hook.
void Initialize();
void Shutdown();

struct Lifecycle {
	static constexpr const char* name       = "Debugger";
	static constexpr auto        initialize = Debugger::Initialize;
	static constexpr auto        shutdown   = Debugger::Shutdown;
};

// True when --debugger was passed and the trap handler registered successfully.
bool IsEnabled();

bool IsOverlayVisible();
void SetOverlayVisible(bool visible);
void ToggleOverlay();

// Consume an SDL event for the overlay. Returns true when the overlay took it, in which case
// the caller must not forward it to the guest (see the input arbiter in window.cpp).
bool HandleEvent(const SDL_Event& event);

// Called once the guest entry point is resolved, before guest code runs.
void OnGuestEntry(uint64_t entry_address);

} // namespace Debugger

#endif // EMULATOR_SRC_DEBUGGER_DEBUGGER_H_
