#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERFUNCTIONS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERFUNCTIONS_H_

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Decoder {

using ShaderCodeReader = std::function<bool(uint64_t, std::span<uint32_t>)>;

// Expands scalar calls through immutable pointers or bounded scalar-buffer tables.
// Table calls retain runtime selection and trap on targets outside the captured set.
// Empty output means no calls; unresolved calls never discard guest instructions.
bool InlineShaderFunctions(std::span<const uint32_t> code, uint64_t base,
                           std::span<const uint32_t> user_data, const ShaderCodeReader& read,
                           std::vector<uint32_t>& expanded, std::string& reason,
                           uint32_t wave_size = 64);

} // namespace Libs::Graphics::ShaderRecompiler::Decoder

#endif
