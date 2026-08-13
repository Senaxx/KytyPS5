#include "graphics/host_gpu/renderer/pipeline/shaderSubgroup.h"

#include "graphics/shader/recompiler/emitter/SpirvEmitter.h"

namespace Libs::Graphics {

ShaderSubgroupCapabilities::ShaderSubgroupCapabilities(const GraphicContext& graphics)
    : subgroup_size(graphics.subgroup_size), min_subgroup_size(graphics.min_subgroup_size),
      max_subgroup_size(graphics.max_subgroup_size),
      required_subgroup_size_stages(graphics.required_subgroup_size_stages),
      subgroup_size_control_enabled(graphics.subgroup_size_control_enabled) {}

ShaderLaneMaskMode SelectGraphicsLaneMaskMode(uint32_t guest_wave_size) {
	// A 64-wide guest wave can't assume the host GPU's lane layout matches
	// GCN's. AMD RDNA also reports a 64-wide subgroup but uses a different
	// lane/pixel layout, which corrupted rendering under "native wave" mode.
	// Per-invocation emulation works regardless of the host's lane layout.
	if (guest_wave_size == 64u) {
		return ShaderLaneMaskMode::PerInvocation;
	}
	return ShaderLaneMaskMode::NativeWave;
}

ShaderLaneMaskMode SelectComputeLaneMaskMode(const ShaderSubgroupCapabilities& capabilities,
                                             uint32_t guest_wave_size,
                                             uint32_t local_threads) {
	if (guest_wave_size != 64u || local_threads < guest_wave_size) {
		return ShaderLaneMaskMode::NativeWave;
	}
	if (capabilities.subgroup_size == guest_wave_size) {
		return ShaderLaneMaskMode::NativeWave;
	}
	const auto compute_stage = vk::ShaderStageFlagBits::eCompute;
	const bool controlled64 = capabilities.subgroup_size_control_enabled &&
	                          (capabilities.required_subgroup_size_stages & compute_stage) &&
	                          guest_wave_size >= capabilities.min_subgroup_size &&
	                          guest_wave_size <= capabilities.max_subgroup_size;
	return controlled64 ? ShaderLaneMaskMode::NativeWave : ShaderLaneMaskMode::PerInvocation;
}

ShaderLaneMaskMode SelectComputeProgramLaneMaskMode(
    const ShaderSubgroupCapabilities& capabilities, uint32_t guest_wave_size,
    uint32_t local_threads, const ShaderRecompiler::IR::Program& native_program) {
	// Compile and analyze native IR first. Most compute programs do not consume
	// whole-wave masks or subgroup ordering, and remain correct in the existing
	// FlattenedMasks path even when one guest wave spans two host subgroups.
	// Per-invocation lowering changes the representation of EXEC/VCC, so only
	// select it for programs which actually require exact guest-wave semantics.
	const bool cached = native_program.compute_subgroup.complete &&
	                    native_program.compute_subgroup.local_threads == local_threads;
	const bool requires_exact =
	    cached ? native_program.compute_subgroup.requires_exact_subgroup
	           : ShaderRecompiler::Spirv::ProgramRequiresExactSubgroupSize(native_program);
	if (SelectComputeLaneMaskMode(capabilities, guest_wave_size, local_threads) !=
	        ShaderLaneMaskMode::PerInvocation ||
	    !requires_exact) {
		return ShaderLaneMaskMode::NativeWave;
	}
	if (guest_wave_size == 64u && capabilities.subgroup_size == 32u &&
	    local_threads == guest_wave_size &&
	    (cached ? native_program.compute_subgroup.half_wave_separable
	            : ShaderRecompiler::Spirv::ProgramSupportsHalfWaveSeparable(native_program))) {
		return ShaderLaneMaskMode::NativeWave;
	}
	if (cached) {
		const bool supported = local_threads == guest_wave_size
		                           ? native_program.compute_subgroup.logical_single_wave_supported
		                           : native_program.compute_subgroup.logical_multi_wave_supported;
		return supported ? ShaderLaneMaskMode::PerInvocation : ShaderLaneMaskMode::NativeWave;
	}
	auto logical_program           = native_program;
	logical_program.lane_mask_mode = ShaderLaneMaskMode::PerInvocation;
	const bool supported =
	    local_threads == guest_wave_size
	        ? ShaderRecompiler::Spirv::ProgramSupportsLogicalSingleWaveWorkgroup(logical_program)
	        : local_threads > guest_wave_size && local_threads % guest_wave_size == 0u &&
	              ShaderRecompiler::Spirv::ProgramSupportsLogicalMultiWaveWorkgroup(logical_program);
	if (!supported) {
		return ShaderLaneMaskMode::NativeWave;
	}
	return ShaderLaneMaskMode::PerInvocation;
}

ShaderSubgroupConfiguration ConfigureShaderSubgroup(const ShaderSubgroupCapabilities& capabilities,
                                                    vk::ShaderStageFlagBits           stage,
                                                    const ShaderRecompiler::IR::Program& program,
                                                    uint32_t local_threads) {
	const auto guest_wave_size = program.wave_size;
	if (guest_wave_size != 32u && guest_wave_size != 64u) {
		return {};
	}
	if (stage != vk::ShaderStageFlagBits::eCompute) {
		const auto expected = SelectGraphicsLaneMaskMode(guest_wave_size);
		if (program.lane_mask_mode != expected || (capabilities.subgroup_size != guest_wave_size &&
		                                           expected != ShaderLaneMaskMode::PerInvocation)) {
			return {};
		}
		return {expected == ShaderLaneMaskMode::NativeWave
		            ? ShaderSubgroupMode::Natural
		            : ShaderSubgroupMode::PerInvocationGraphics,
		        0};
	}
	const bool cached = program.compute_subgroup.complete &&
	                    program.compute_subgroup.local_threads == local_threads;
	if (program.lane_mask_mode == ShaderLaneMaskMode::PerInvocation) {
		const bool single_supported =
		    cached ? program.compute_subgroup.logical_single_wave_supported
		           : ShaderRecompiler::Spirv::ProgramSupportsLogicalSingleWaveWorkgroup(program);
		if (local_threads == guest_wave_size && single_supported) {
			return {ShaderSubgroupMode::LogicalSingleWaveWorkgroup, 0};
		}
		const bool multi_supported =
		    cached ? program.compute_subgroup.logical_multi_wave_supported
		           : ShaderRecompiler::Spirv::ProgramSupportsLogicalMultiWaveWorkgroup(program);
		if (local_threads > guest_wave_size && local_threads % guest_wave_size == 0u &&
		    multi_supported) {
			return {ShaderSubgroupMode::LogicalMultiWaveWorkgroup, 0};
		}
		return {};
	}
	if (program.lane_mask_mode != ShaderLaneMaskMode::NativeWave) {
		return {};
	}
	if (capabilities.subgroup_size == guest_wave_size) {
		return {ShaderSubgroupMode::Natural, 0};
	}
	// A workgroup smaller than one guest wave is a single partial wave. The
	// inactive guest lanes have no invocations, so native subgroup operations
	// are exact when every live lane fits in one host subgroup.
	if (local_threads != 0u && local_threads < guest_wave_size &&
	    local_threads <= capabilities.subgroup_size) {
		return {ShaderSubgroupMode::PartialWave, 0};
	}
	if (capabilities.subgroup_size_control_enabled &&
	    (capabilities.required_subgroup_size_stages & stage) &&
	    guest_wave_size >= capabilities.min_subgroup_size &&
	    guest_wave_size <= capabilities.max_subgroup_size) {
		return {ShaderSubgroupMode::Controlled, guest_wave_size};
	}
	if (guest_wave_size == 64u && capabilities.subgroup_size == 32u &&
	    local_threads == guest_wave_size &&
	    (cached ? program.compute_subgroup.half_wave_separable
	            : ShaderRecompiler::Spirv::ProgramSupportsHalfWaveSeparable(program))) {
		return {ShaderSubgroupMode::HalfWaveIndependent, 0};
	}
	const bool requires_exact =
	    cached ? program.compute_subgroup.requires_exact_subgroup
	           : ShaderRecompiler::Spirv::ProgramRequiresExactSubgroupSize(program);
	if (!requires_exact) {
		return {ShaderSubgroupMode::FlattenedMasks, 0};
	}
	return {};
}

bool ComputeProgramWave64Supported(const ShaderSubgroupCapabilities& capabilities,
                                   uint32_t local_threads,
                                   const ShaderRecompiler::IR::Program& native_program) {
	if (native_program.wave_size != 64u ||
	    native_program.lane_mask_mode != ShaderLaneMaskMode::NativeWave) {
		return false;
	}
	const auto selected = SelectComputeProgramLaneMaskMode(
	    capabilities, native_program.wave_size, local_threads, native_program);
	if (selected == ShaderLaneMaskMode::PerInvocation) {
		// Selection already cloned and certified the logical representation. The
		// caller will perform the actual PerInvocation recompile afterwards.
		return true;
	}
	return ConfigureShaderSubgroup(capabilities, vk::ShaderStageFlagBits::eCompute,
	                               native_program, local_threads)
	           .mode != ShaderSubgroupMode::Unsupported;
}

uint32_t SelectComputeExecutionWaveSize(const ShaderSubgroupCapabilities& capabilities,
                                        uint32_t local_threads,
                                        const ShaderRecompiler::IR::Program& native_program) {
	if (native_program.wave_size != 64u ||
	    ComputeProgramWave64Supported(capabilities, local_threads, native_program)) {
		return native_program.wave_size;
	}
	// Older hosts cannot always expose a 64-lane subgroup, and not every shader
	// can be represented by the exact logical-wave lowering yet. Keep such
	// programs runnable with the established wave32 compatibility lowering.
	return 32u;
}

} // namespace Libs::Graphics
