#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_

#include "graphics/guest_gpu/gpu_defs.h"

#include <array>
#include <cstdint>
#include <utility>

namespace Libs::Graphics {

struct ShaderVertexInputInfo;

namespace HW {
class Context;
class Shader;
} // namespace HW

[[nodiscard]] std::pair<int32_t, uint32_t>
ResolveDrawOffsets(uint32_t index_offset, const ShaderVertexInputInfo& vs_input_info);

// Draw prep scanner (drawPrep.h): whether a draw with these registers has a vertex shader, and
// whether it runs its pixel stage and with which target export mapping, decided as
// PrepareDrawRenderState and RefreshShaders do.
[[nodiscard]] bool DrawPrepHasVertexShader(const HW::Shader& sh_ctx);
[[nodiscard]] bool DrawPrepPixelState(
    const HW::Context& ctx, const HW::Shader& sh_ctx,
    std::array<Prospero::ColorComponentMapping, 8>& target_export_mapping);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_
