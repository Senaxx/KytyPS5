#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_RENDERDOC_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_RENDERDOC_H_

#include <cstdint>
#include <string>

namespace Libs::Graphics {

enum class RenderDocCaptureState : uint32_t { Idle, Requested, Capturing };

struct RenderDocStatus {
	bool                  available = false;
	RenderDocCaptureState state = RenderDocCaptureState::Idle;
	std::string           capture_path;
	uint64_t              completed_captures = 0;
	bool                  has_result = false;
	bool                  last_succeeded = false;
};

void RenderDocInit();
[[nodiscard]] bool RenderDocRequestCapture();
void RenderDocStartCapture();
void RenderDocEndCapture();

[[nodiscard]] RenderDocStatus RenderDocGetStatus();
[[nodiscard]] const char* RenderDocCaptureStateName(RenderDocCaptureState state);
[[nodiscard]] bool RenderDocCaptureRequested();
[[nodiscard]] bool RenderDocCaptureInProgress();

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_RENDERDOC_H_ */
