#include "debugger/debugger.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "debugger/core/session.h"
#include "debugger/remote/server.h"
#include "debugger/target/graphics.h"
#include "debugger/target/io.h"
#include "debugger/ui/overlay.h"

namespace Debugger {

void Initialize() {
	if (!Config::DebuggerEnabled()) {
		return;
	}

	if (!Session::Initialize()) {
		return;
	}
	Graphics::Reset();
	Io::Reset();

	Ui::SetVisible(Config::DebuggerUiVisible());
	if (Config::DebuggerServerEnabled() && !Remote::Start()) {
		LOGF_COLOR(Log::Color::BrightYellow, "Debugger: external endpoint failed to start\n");
	}

	for (const auto& location: Config::GetDebuggerBreakpoints()) {
		const auto id = Session::AddBreakpoint(location);
		if (id == 0) {
			LOGF_COLOR(Log::Color::BrightYellow, "Debugger: could not add breakpoint '%s'\n",
			           location.c_str());
		} else {
			LOGF("Debugger: breakpoint %u at '%s'\n", id, location.c_str());
		}
	}

	LOGF_COLOR(Log::Color::Cyan, "Debugger: enabled (F1 toggles the overlay)\n");
}

void Shutdown() {
	if (!Session::IsEnabled()) {
		return;
	}

	Ui::SetVisible(false);
	Remote::Stop();
	Session::Shutdown();
}

bool IsEnabled() {
	return Session::IsEnabled();
}

bool IsOverlayVisible() {
	return Session::IsEnabled() && Ui::IsVisible();
}

void SetOverlayVisible(bool visible) {
	if (Session::IsEnabled()) {
		Ui::SetVisible(visible);
	}
}

void ToggleOverlay() {
	if (Session::IsEnabled()) {
		Ui::Toggle();
	}
}

bool HandleEvent(const SDL_Event& event) {
	if (!Session::IsEnabled()) {
		return false;
	}
	return Ui::ProcessEvent(event);
}

void OnGuestEntry(uint64_t entry_address) {
	if (!Session::IsEnabled()) {
		return;
	}

	Session::OnGuestEntry(entry_address);

	if (Config::DebuggerBreakOnEntry() && entry_address != 0) {
		Session::AddBreakpointAt(entry_address, true);
		Ui::SetVisible(true);
		LOGF("Debugger: will halt at the guest entry point\n");
	}
}

} // namespace Debugger
