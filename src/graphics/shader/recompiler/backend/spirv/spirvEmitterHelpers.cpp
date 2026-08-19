#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

uint32_t EmitTrueBool(EmitterState& state) {
	const auto ret = state.builder.AllocateId();
	state.builder.AddFunction(
	    {OpINotEqual, state.bool_type, ret, ConstantU32(state, 1), ConstantU32(state, 0)});
	return ret;
}

DppTargetLane EmitDppQuadPermTargetLane(EmitterState& state, uint32_t subid, uint32_t control) {
	const auto quad_base = state.builder.AllocateId();
	const auto lane      = state.builder.AllocateId();
	const auto shift     = state.builder.AllocateId();
	const auto selected0 = state.builder.AllocateId();
	const auto selected  = state.builder.AllocateId();
	const auto target    = state.builder.AllocateId();
	state.builder.AddFunction(
	    {OpBitwiseAnd, state.uint_type, quad_base, subid, ConstantU32(state, 0xfffffffcu)});
	state.builder.AddFunction({OpBitwiseAnd, state.uint_type, lane, subid, ConstantU32(state, 3)});
	state.builder.AddFunction(
	    {OpShiftLeftLogical, state.uint_type, shift, lane, ConstantU32(state, 1)});
	state.builder.AddFunction(
	    {OpShiftRightLogical, state.uint_type, selected0, ConstantU32(state, control), shift});
	state.builder.AddFunction(
	    {OpBitwiseAnd, state.uint_type, selected, selected0, ConstantU32(state, 3)});
	state.builder.AddFunction({OpBitwiseOr, state.uint_type, target, quad_base, selected});
	return {target, EmitTrueBool(state)};
}

DppTargetLane EmitDppRowShiftTargetLane(EmitterState& state, uint32_t subid, uint32_t amount,
                                        bool left) {
	const auto row          = state.builder.AllocateId();
	const auto lane         = state.builder.AllocateId();
	const auto lane_shifted = state.builder.AllocateId();
	const auto target       = state.builder.AllocateId();
	const auto valid        = state.builder.AllocateId();
	state.builder.AddFunction(
	    {OpBitwiseAnd, state.uint_type, row, subid, ConstantU32(state, 0xfffffff0u)});
	state.builder.AddFunction({OpBitwiseAnd, state.uint_type, lane, subid, ConstantU32(state, 15)});
	if (left) {
		state.builder.AddFunction(
		    {OpIAdd, state.uint_type, lane_shifted, lane, ConstantU32(state, amount)});
		state.builder.AddFunction(
		    {OpULessThan, state.bool_type, valid, lane, ConstantU32(state, 16u - amount)});
	} else {
		state.builder.AddFunction(
		    {OpISub, state.uint_type, lane_shifted, lane, ConstantU32(state, amount)});
		state.builder.AddFunction(
		    {OpUGreaterThanEqual, state.bool_type, valid, lane, ConstantU32(state, amount)});
	}
	state.builder.AddFunction({OpBitwiseOr, state.uint_type, target, row, lane_shifted});
	return {target, valid};
}

DppTargetLane EmitDppRowRotateRightTargetLane(EmitterState& state, uint32_t subid,
                                              uint32_t amount) {
	const auto row      = state.builder.AllocateId();
	const auto lane     = state.builder.AllocateId();
	const auto in_high  = state.builder.AllocateId();
	const auto minus    = state.builder.AllocateId();
	const auto plus     = state.builder.AllocateId();
	const auto selected = state.builder.AllocateId();
	const auto target   = state.builder.AllocateId();
	state.builder.AddFunction(
	    {OpBitwiseAnd, state.uint_type, row, subid, ConstantU32(state, 0xfffffff0u)});
	state.builder.AddFunction({OpBitwiseAnd, state.uint_type, lane, subid, ConstantU32(state, 15)});
	state.builder.AddFunction(
	    {OpUGreaterThanEqual, state.bool_type, in_high, lane, ConstantU32(state, amount)});
	state.builder.AddFunction({OpISub, state.uint_type, minus, lane, ConstantU32(state, amount)});
	state.builder.AddFunction(
	    {OpIAdd, state.uint_type, plus, lane, ConstantU32(state, 16u - amount)});
	state.builder.AddFunction({OpSelect, state.uint_type, selected, in_high, minus, plus});
	state.builder.AddFunction({OpBitwiseOr, state.uint_type, target, row, selected});
	return {target, EmitTrueBool(state)};
}

DppTargetLane EmitDppMirrorTargetLane(EmitterState& state, uint32_t subid, bool half_row) {
	const auto base_mask = half_row ? 0xfffffff8u : 0xfffffff0u;
	const auto lane_mask = half_row ? 7u : 15u;
	const auto base      = state.builder.AllocateId();
	const auto lane      = state.builder.AllocateId();
	const auto mirrored  = state.builder.AllocateId();
	const auto target    = state.builder.AllocateId();
	state.builder.AddFunction(
	    {OpBitwiseAnd, state.uint_type, base, subid, ConstantU32(state, base_mask)});
	state.builder.AddFunction(
	    {OpBitwiseAnd, state.uint_type, lane, subid, ConstantU32(state, lane_mask)});
	state.builder.AddFunction(
	    {OpISub, state.uint_type, mirrored, ConstantU32(state, lane_mask), lane});
	state.builder.AddFunction({OpBitwiseOr, state.uint_type, target, base, mirrored});
	return {target, EmitTrueBool(state)};
}

DppTargetLane EmitDppTargetLane(EmitterState& state, uint32_t control) {
	const auto subid = EmitSubgroupLocalInvocationId(state);
	if (control <= 0xffu) {
		return EmitDppQuadPermTargetLane(state, subid, control);
	}
	if (control >= 0x101u && control <= 0x10fu) {
		return EmitDppRowShiftTargetLane(state, subid, control & 0xfu, true);
	}
	if (control >= 0x111u && control <= 0x11fu) {
		return EmitDppRowShiftTargetLane(state, subid, control & 0xfu, false);
	}
	if (control >= 0x121u && control <= 0x12fu) {
		return EmitDppRowRotateRightTargetLane(state, subid, control & 0xfu);
	}
	if (control == 0x140u) {
		return EmitDppMirrorTargetLane(state, subid, false);
	}
	if (control == 0x141u) {
		return EmitDppMirrorTargetLane(state, subid, true);
	}
	return {subid, EmitTrueBool(state)};
}

uint32_t EmitSubgroupLocalInvocationId(EmitterState& state) {
	if (state.subgroup_local_invocation_id_variable == 0) {
		return ConstantU32(state, 0);
	}
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(
	    {OpLoad, state.uint_type, value, state.subgroup_local_invocation_id_variable});
	return value;
}

uint32_t EmitGuestLaneIndex(EmitterState& state) {
	return state.logical_single_wave_workgroup ? EmitLocalInvocationIndex(state)
	                                           : EmitSubgroupLocalInvocationId(state);
}

uint32_t EmitLogicalWaveAnyBool(EmitterState& state, uint32_t value_bool) {
	if (!state.logical_single_wave_workgroup) {
		return value_bool;
	}

	// A guest wave64 spans two subgroup32 waves on NVIDIA. Publish one Boolean per
	// guest lane in private LDS scratch and reduce all 64 values at a workgroup barrier.
	const auto semantics = MemorySemanticsAcquireRelease | MemorySemanticsWorkgroupMemory;
	const auto lane      = EmitLocalInvocationIndex(state);
	const auto own_index = state.builder.AllocateId();
	state.builder.AddFunction({OpIAdd, state.uint_type, own_index,
	                           ConstantU32(state, state.logical_wave_scratch_base), lane});
	const auto own_ptr = EmitLdsElementPointer(state, own_index);
	const auto word    = state.builder.AllocateId();
	state.builder.AddFunction({OpSelect, state.uint_type, word, value_bool,
	                           ConstantU32(state, 1), ConstantU32(state, 0)});
	state.builder.AddFunction({OpStore, own_ptr, word});
	state.builder.AddFunction({OpControlBarrier, ConstantU32(state, ScopeWorkgroup),
	                           ConstantU32(state, ScopeWorkgroup),
	                           ConstantU32(state, semantics)});

	const auto lane_zero = state.builder.AllocateId();
	state.builder.AddFunction(
	    {OpIEqual, state.bool_type, lane_zero, lane, ConstantU32(state, 0)});
	const auto summary_index = ConstantU32(state, state.logical_wave_scratch_base + 64u);
	const auto summary_ptr   = EmitLdsElementPointer(state, summary_index);
	EmitIfCondition(state, lane_zero, [&]() {
		uint32_t aggregate = ConstantU32(state, 0);
		for (uint32_t source_lane = 0; source_lane < 64u; source_lane++) {
			const auto source_index =
			    ConstantU32(state, state.logical_wave_scratch_base + source_lane);
			const auto source_ptr = EmitLdsElementPointer(state, source_index);
			const auto source     = state.builder.AllocateId();
			state.builder.AddFunction({OpLoad, state.uint_type, source, source_ptr});
			const auto combined = state.builder.AllocateId();
			state.builder.AddFunction(
			    {OpBitwiseOr, state.uint_type, combined, aggregate, source});
			aggregate = combined;
		}
		state.builder.AddFunction({OpStore, summary_ptr, aggregate});
	});
	state.builder.AddFunction({OpControlBarrier, ConstantU32(state, ScopeWorkgroup),
	                           ConstantU32(state, ScopeWorkgroup),
	                           ConstantU32(state, semantics)});

	const auto summary = state.builder.AllocateId();
	state.builder.AddFunction({OpLoad, state.uint_type, summary, summary_ptr});
	const auto active = state.builder.AllocateId();
	state.builder.AddFunction(
	    {OpINotEqual, state.bool_type, active, summary, ConstantU32(state, 0)});
	return active;
}

uint32_t InputVariableForKind(const EmitterState& state, IR::StageInputKind kind) {
	for (const auto& input: state.inputs) {
		if (input.kind == kind) {
			return input.variable_id;
		}
	}
	return 0;
}

const InputBinding* InputBindingForParameter(const EmitterState& state, uint32_t location) {
	for (const auto& input: state.inputs) {
		if (input.kind == IR::StageInputKind::Parameter && input.location == location) {
			return &input;
		}
	}
	return nullptr;
}

uint32_t EmitInputComponentU32(EmitterState& state, IR::StageInputKind kind, uint32_t component) {
	const auto variable = InputVariableForKind(state, kind);
	if (variable == 0) {
		return ConstantU32(state, 0);
	}
	const auto pointer = state.builder.AllocateId();
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(
	    {OpAccessChain, state.ptr_input_uint, pointer, variable, ConstantU32(state, component)});
	state.builder.AddFunction({OpLoad, state.uint_type, value, pointer});
	return value;
}

uint32_t EmitLocalInvocationIndex(EmitterState& state) {
	const auto variable = InputVariableForKind(state, IR::StageInputKind::LocalInvocationIndex);
	if (variable == 0) {
		return ConstantU32(state, 0);
	}
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction({OpLoad, state.uint_type, value, variable});
	return value;
}

uint32_t VertexInputDefaultComponentU32(EmitterState& state, VertexInputScalarKind kind,
                                        uint32_t component) {
	if ((component & 3u) != 3u) {
		return ConstantU32(state, 0);
	}
	return ConstantU32(state, kind == VertexInputScalarKind::Float ? 0x3f800000u : 1u);
}

uint32_t EmitVertexParameterComponentU32(EmitterState& state, const InputBinding& input,
                                         uint32_t component) {
	const auto count = VertexParameterComponentCount(state, input);
	const auto kind  = VertexParameterScalarKind(state, input.location);
	if (component >= count) {
		return VertexInputDefaultComponentU32(state, kind, component);
	}

	const auto scalar_type = VertexParameterScalarType(state, kind);
	uint32_t   raw         = state.builder.AllocateId();
	if (count == 1u) {
		state.builder.AddFunction({OpLoad, scalar_type, raw, input.variable_id});
	} else {
		const auto pointer_type = VertexParameterScalarPointerType(state, kind);
		const auto pointer      = state.builder.AllocateId();
		state.builder.AddFunction({OpAccessChain, pointer_type, pointer, input.variable_id,
		                           ConstantU32(state, component)});
		state.builder.AddFunction({OpLoad, scalar_type, raw, pointer});
	}

	if (kind == VertexInputScalarKind::Uint) {
		return raw;
	}

	const auto bits = state.builder.AllocateId();
	state.builder.AddFunction({OpBitcast, state.uint_type, bits, raw});
	return bits;
}

uint32_t EmitSubgroupLaneActiveBool(EmitterState& state, uint32_t lane) {
	const auto active_ballot = state.builder.AllocateId();
	state.builder.AddFunction({OpGroupNonUniformBallot, state.vec4_uint_type, active_ballot,
	                           ConstantU32(state, ScopeSubgroup), EmitTrueBool(state)});
	return EmitBallotLaneActiveBool(state, active_ballot, lane);
}
uint32_t EmitBallotLaneActiveBool(EmitterState& state, uint32_t active_ballot, uint32_t lane) {
	const auto low = state.builder.AllocateId();
	state.builder.AddFunction({OpCompositeExtract, state.uint_type, low, active_ballot, 0});
	uint32_t mask = low;
	if (state.wave_size == 64u) {
		const auto high     = state.builder.AllocateId();
		const auto in_high  = state.builder.AllocateId();
		const auto selected = state.builder.AllocateId();
		state.builder.AddFunction({OpCompositeExtract, state.uint_type, high, active_ballot, 1});
		state.builder.AddFunction(
		    {OpUGreaterThanEqual, state.bool_type, in_high, lane, ConstantU32(state, 32)});
		state.builder.AddFunction({OpSelect, state.uint_type, selected, in_high, high, low});
		mask = selected;
	}

	const auto lane_low = state.builder.AllocateId();
	const auto bit      = state.builder.AllocateId();
	state.builder.AddFunction(
	    {OpBitwiseAnd, state.uint_type, lane_low, lane, ConstantU32(state, 31)});
	state.builder.AddFunction(
	    {OpShiftLeftLogical, state.uint_type, bit, ConstantU32(state, 1), lane_low});

	const auto hit      = state.builder.AllocateId();
	const auto active   = state.builder.AllocateId();
	const auto in_range = state.builder.AllocateId();
	const auto ret      = state.builder.AllocateId();
	state.builder.AddFunction({OpBitwiseAnd, state.uint_type, hit, mask, bit});
	state.builder.AddFunction({OpINotEqual, state.bool_type, active, hit, ConstantU32(state, 0)});
	state.builder.AddFunction(
	    {OpULessThan, state.bool_type, in_range, lane, ConstantU32(state, state.wave_size)});
	state.builder.AddFunction({OpLogicalAnd, state.bool_type, ret, active, in_range});
	return ret;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
