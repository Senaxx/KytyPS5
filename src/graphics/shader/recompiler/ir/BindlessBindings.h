#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_

#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Descriptor set 1 of a pipeline that samples bindless images (the host side is BindlessTable).
inline constexpr uint32_t BindlessDescriptorSet  = 1;
inline constexpr uint32_t BindlessImages2D       = 0;
inline constexpr uint32_t BindlessImages2DArray  = 1;
inline constexpr uint32_t BindlessImagesCube     = 2;
inline constexpr uint32_t BindlessImages3D       = 3;
inline constexpr uint32_t BindlessTranslation    = 4;
inline constexpr uint32_t BindlessFeedback       = 5;
// A translation entry for a key whose texture is not resident yet; the shader samples slot 0.
inline constexpr uint32_t BindlessPending        = 0xffffffffu;

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_
