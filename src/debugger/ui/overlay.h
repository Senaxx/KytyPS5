#ifndef EMULATOR_SRC_DEBUGGER_UI_OVERLAY_H_
#define EMULATOR_SRC_DEBUGGER_UI_OVERLAY_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <memory>

union SDL_Event;

namespace Libs::Graphics {
struct GraphicContext;
} // namespace Libs::Graphics

namespace Debugger::Ui {

// Queue an SDL event for the overlay. Called on the window thread; the events are drained on
// the presentation thread inside PrepareFrame, because ImGui's input queue is not thread-safe
// and the two live on different threads (same approach as ImeOverlay).
//
// Returns true when the overlay consumed the event, in which case the caller must not forward
// it to the guest.
bool ProcessEvent(const SDL_Event& event);

// True when the overlay is drawing and therefore wants the presenter to keep producing frames
// even though the guest has stopped flipping.
bool IsVisible();
void SetVisible(bool visible);
void Toggle();

class DebuggerOverlay final {
public:
	explicit DebuggerOverlay(Libs::Graphics::GraphicContext& graphics);
	~DebuggerOverlay();
	KYTY_CLASS_NO_COPY(DebuggerOverlay);

	[[nodiscard]] bool PrepareFrame(vk::Extent2D extent, vk::Format format, uint32_t image_count);
	void               Record(vk::CommandBuffer command, vk::ImageView target);
	void               ReleaseVulkan();

	// Test seam: walk the whole panel tree against the caller's ImGui context, with no Vulkan
	// device involved. ImGui asserts on mismatched Begin/End pairs and table column overruns,
	// so exercising every pane headlessly catches layout mistakes that would otherwise only
	// show up with a game running. `force_tab` selects a bottom tab by index (-1 leaves the
	// user's choice alone) so all four can be covered in one run.
	static void DrawPanelsForTesting(int force_tab = -1);

	// Number of times the disassembly has asked ImGui to scroll to the halted instruction.
	// Following rip must issue one request per new halt address, not one per frame — doing it
	// every frame pins the view and undoes the user's scrolling.
	static uint64_t ScrollRequestCountForTesting();

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Debugger::Ui

#endif // EMULATOR_SRC_DEBUGGER_UI_OVERLAY_H_
