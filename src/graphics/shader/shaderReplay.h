#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERREPLAY_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERREPLAY_H_

#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::ShaderReplay {

struct ComputeCapture {
	uint64_t shader_hash          = 0;
	uint64_t shader_base          = 0;
	uint32_t user_data_base       = 0;
	uint32_t user_data_count      = 0;
	uint32_t scratch_dwords       = 0;
	uint32_t push_constant_offset = 0;

	ShaderComputeInputInfo                 input;
	std::vector<uint32_t>                  code;
	std::vector<uint32_t>                  user_data;
	ShaderRecompiler::IR::ResourceSnapshot resources;
};

struct PixelCapture {
	uint64_t shader_hash          = 0;
	uint64_t shader_base          = 0;
	uint32_t wave_size            = 64;
	uint32_t user_data_base       = 0;
	uint32_t user_data_count      = 0;
	uint32_t scratch_dwords       = 0;
	uint32_t push_constant_offset = 0;

	ShaderPixelInputInfo                   input;
	std::vector<uint32_t>                  code;
	std::vector<uint32_t>                  user_data;
	ShaderRecompiler::IR::ResourceSnapshot resources;
};

bool WriteComputeCapture(const std::filesystem::path& file, std::span<const uint32_t> code,
                         const ShaderRecompiler::CompileOptions&       options,
                         const ShaderComputeInputInfo&                 input,
                         const ShaderRecompiler::IR::ResourceSnapshot& resources,
                         std::string*                                  error);

bool ReadComputeCapture(const std::filesystem::path& file, ComputeCapture& capture,
                        std::string* error);

bool WritePixelCapture(const std::filesystem::path& file, std::span<const uint32_t> code,
                       const ShaderRecompiler::CompileOptions&       options,
                       const ShaderPixelInputInfo&                   input,
                       const ShaderRecompiler::IR::ResourceSnapshot& resources, std::string* error);

bool ReadPixelCapture(const std::filesystem::path& file, PixelCapture& capture, std::string* error);

} // namespace Libs::Graphics::ShaderReplay

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERREPLAY_H_ */
