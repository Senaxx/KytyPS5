#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

std::vector<uint32_t> EmitProgram(const IR::Program& program,
                                  ShaderStageInputInfo input_info);

// Optional image features of the device, set once by the device layer.
struct HostImageFeatures {
	// shaderResourceMinLod: the MinLod image operand, used for IMAGE_SAMPLE*_CL.
	bool min_lod = false;
};

void              SetHostImageFeatures(const HostImageFeatures& features);
HostImageFeatures GetHostImageFeatures();

// S_MEMREALTIME counts at 100 MHz. The emitter reads the device clock (OpReadClockKHR, device
// scope) and shifts it right by this many bits to approach that rate. The device layer sets it
// from the device's clock rate: 3 for a 1 GHz clock (NVIDIA, the default), 0 for a 100 MHz one
// (AMD).
void     SetDeviceClockShift(uint32_t shift);
uint32_t GetDeviceClockShift();

// A coherent (GLC) buffer load is followed by a device-scope acquire barrier, so that a load
// polling for another workgroup's write sees it. Without one, AMD's drivers keep serving the
// cached value (BufferWorkgroupPublication, ISSUES.md #16); NVIDIA reloads anyway. The device
// layer enables it on every vendor but NVIDIA; KYTY_COHERENT_LOAD_ACQUIRE=0/1 overrides that.
void SetCoherentLoadAcquire(bool enabled);
bool GetCoherentLoadAcquire();

// Why a mesh program cannot run in passes (ShaderMeshInputInfo::passes), or nullptr.
[[nodiscard]] const char* MeshPassesUnsupported(const IR::Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
