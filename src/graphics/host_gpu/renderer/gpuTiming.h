#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMING_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMING_H_

#include <atomic>
#include <cstdint>

// Diagnostics: KYTY_GPU_TIMING=<first frame>:<frames> times every draw and dispatch on the host
// GPU (timestamps before and after, which the per-command barriers make exact) and logs, per
// presented frame in that window, the GPU time of the costliest shaders and of everything
// between commands. Off by default.
namespace Libs::Graphics {

class CommandBuffer;
class RenderContext;

namespace GpuTiming {

namespace Detail {
inline std::atomic<bool> g_active {false};
void BeforeSlow(RenderContext& context, CommandBuffer& buffer);
void AfterSlow(RenderContext& context, CommandBuffer& buffer, uint64_t key, uint32_t kind);
} // namespace Detail

enum Kind : uint32_t { Draw = 1, Dispatch = 2 };

inline void Before(RenderContext& context, CommandBuffer& buffer) {
	if (Detail::g_active.load(std::memory_order_relaxed)) {
		Detail::BeforeSlow(context, buffer);
	}
}

inline void After(RenderContext& context, CommandBuffer& buffer, uint64_t key, uint32_t kind) {
	if (Detail::g_active.load(std::memory_order_relaxed)) {
		Detail::AfterSlow(context, buffer, key, kind);
	}
}

// Called once per presented frame.
void Present(uint64_t frame);

} // namespace GpuTiming
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMING_H_
