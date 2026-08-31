#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <array>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {

bool DependsOnInvocation(IR::Value value) {
	std::queue<IR::Value>         queue;
	std::unordered_set<IR::Inst*> visited;
	queue.push(value);
	while (!queue.empty()) {
		value = queue.front().Resolve();
		queue.pop();
		auto* inst = value.TryInstruction();
		if (inst == nullptr || !visited.insert(inst).second) {
			continue;
		}
		if (inst->GetOpcode() == IR::ValueOpcode::ReadLane ||
		    inst->GetOpcode() == IR::ValueOpcode::ReadFirstLane) {
			continue;
		}
		if (inst->GetOpcode() == IR::ValueOpcode::LaneId) {
			return true;
		}
		if (inst->GetOpcode() == IR::ValueOpcode::GetBuiltin) {
			const auto kind = static_cast<IR::StageInputKind>(inst->Arg(0).U32());
			if (kind == IR::StageInputKind::LocalInvocationId ||
			    kind == IR::StageInputKind::LocalInvocationIndex ||
			    kind == IR::StageInputKind::GlobalInvocationId) {
				return true;
			}
		}
		for (size_t index = 0; index < inst->NumArgs(); index++) {
			queue.push(inst->Arg(index));
		}
	}
	return false;
}

bool CanReachBlock(const IR::Program& program, uint32_t source_id, uint32_t target_id) {
	const auto FindBlock = [&](uint32_t id) -> const IR::Block* {
		const auto info = std::ranges::find_if(
		    program.block_info, [&](const IR::BlockInfo& block) { return block.id == id; });
		return info == program.block_info.end()
		           ? nullptr
		           : program.blocks[static_cast<size_t>(info - program.block_info.begin())];
	};
	const auto* source = FindBlock(source_id);
	const auto* target = FindBlock(target_id);
	if (source == nullptr || target == nullptr) {
		return false;
	}
	std::queue<const IR::Block*>         queue;
	std::unordered_set<const IR::Block*> visited;
	queue.push(source);
	while (!queue.empty()) {
		const auto* block = queue.front();
		queue.pop();
		if (block == target) {
			return true;
		}
		if (!visited.insert(block).second) {
			continue;
		}
		for (const auto* successor: block->ImmSuccessors()) {
			queue.push(successor);
		}
	}
	return false;
}

void CollectConvergedWave64ReadLanes(Emitter::EmitterState& state,
	                                  const IR::Program& program) {
	uint32_t                               divergence_depth = 0;
	std::unordered_map<uint32_t, uint32_t> divergence_ends;
	bool                                   permanently_divergent = false;
	for (size_t index = 0; index < program.blocks.size(); index++) {
		const auto& info = program.block_info[index];
		if (const auto end = divergence_ends.find(info.id); end != divergence_ends.end()) {
			if (end->second > divergence_depth) {
				permanently_divergent = true;
				divergence_depth      = 0;
			} else {
				divergence_depth -= end->second;
			}
		}
		if (!permanently_divergent && divergence_depth == 0) {
			for (const auto& inst: *program.blocks[index]) {
				if (inst.GetOpcode() == IR::ValueOpcode::ReadLane) {
					state.wave64_read_lane_scratch_banks.emplace(
					    &inst,
					    static_cast<uint32_t>(state.wave64_read_lane_scratch_banks.size()));
				}
			}
		}

		const auto& term = info.terminator;
		if (term.kind != CFG::TerminatorKind::ConditionalBranch ||
		    !DependsOnInvocation(info.condition)) {
			continue;
		}
		if (term.merge_block == UINT32_MAX ||
		    !CanReachBlock(program, term.true_block, term.merge_block) ||
		    !CanReachBlock(program, term.false_block, term.merge_block)) {
			permanently_divergent = true;
			continue;
		}
		divergence_depth++;
		divergence_ends[term.merge_block]++;
	}
}

[[noreturn]] void Fail(const IR::Program& program, const char* reason) {
	EXIT("SPIR-V validation failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     program.shader_hash, static_cast<unsigned>(program.stage), reason);
	std::abort();
}

void ValidateNativeProgram(const IR::Program& program) {
	using Kind                                             = IR::DescriptorBindingKind;
	constexpr auto                               KindCount = static_cast<size_t>(Kind::Count);
	std::array<std::vector<uint32_t>, KindCount> expected;
	std::array<bool, KindCount>                  present {};
	const auto                                   Dense = [](size_t size) {
		std::vector<uint32_t> values(size);
		for (uint32_t i = 0; i < values.size(); i++) {
			values[i] = i;
		}
		return values;
	};
	auto Expect = [&](Kind kind, std::vector<uint32_t> resources = {}) {
		const auto index = static_cast<size_t>(kind);
		present[index]   = true;
		expected[index]  = std::move(resources);
	};
	if (!program.info.buffers.empty()) {
		Expect(Kind::Buffers, Dense(program.info.buffers.size()));
	}
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto kind = IR::DescriptorBindingForImage(program.info.images[i]);
		if (!kind.has_value()) {
			Fail(program, "native shader plan has an invalid image class");
		}
		present[static_cast<size_t>(*kind)] = true;
		const auto dynamic = program.info.images[i].mip_mode == IR::ImageMipMode::DynamicStorage;
		const auto count   = dynamic ? program.info.images[i].mip_count : 1u;
		if (count == 0u || (!dynamic && program.info.images[i].mip_count != 1u)) {
			Fail(program, "native shader plan has an invalid image mip descriptor count");
		}
		expected[static_cast<size_t>(*kind)].insert(expected[static_cast<size_t>(*kind)].end(),
		                                            count, i);
	}
	if (!program.info.samplers.empty()) {
		Expect(Kind::Samplers, Dense(program.info.samplers.size()));
	}
	bool uses_gds = false;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (IR::SharedAccessOf(inst.GetOpcode()) == IR::SharedAccess::None) {
				continue;
			}
			const auto index = inst.Flags<IR::MemoryFlags>().index;
			if (index >= program.memory_info.size()) {
				Fail(program, "shared operation has invalid memory metadata");
			}
			const auto kind = program.memory_info[index].kind;
			if (kind != IR::ResourceKind::Lds && kind != IR::ResourceKind::Gds) {
				Fail(program, "shared operation has invalid resource kind");
			}
			uses_gds |= kind == IR::ResourceKind::Gds;
		}
	}
	if (uses_gds) {
		Expect(Kind::Gds);
	}
	if (program.info.uses_dma) {
		Expect(Kind::BdaPagetable);
		Expect(Kind::FaultBuffer);
	}
	const bool uses_flattened_runtime =
	    !program.srt_reads.empty() ||
	     std::ranges::any_of(program.info.images, [](const IR::ImageResource& image) {
		     return image.indirect_mapping_capacity != 0u;
	     });
	if (uses_flattened_runtime) {
		Expect(Kind::FlattenedSrt);
	}
	if (program.bindings.ShaderDataDwords() != 0 && program.bindings.push_constant_size == 0) {
		Expect(Kind::UserData);
	}

	std::array<bool, KindCount> seen {};
	for (const auto& binding: program.bindings.descriptors) {
		const auto kind = static_cast<size_t>(binding.kind);
		if (kind >= KindCount || seen[kind] || !present[kind] ||
		    binding.resources != expected[kind]) {
			Fail(program, "native descriptor groups do not match shader topology");
		}
		seen[kind] = true;
	}
	for (size_t i = 0; i < KindCount; i++) {
		if (present[i] != seen[i]) {
			Fail(program, "native shader plan is missing a required descriptor group");
		}
	}
	const auto has_user_storage = present[static_cast<size_t>(Kind::UserData)];
	if (program.bindings.push_constant_offset % sizeof(uint32_t) != 0 ||
	    program.bindings.push_constant_offset + program.bindings.push_constant_size >
	        IR::NativePushConstantSize ||
	    program.bindings.memory_offset_dword != program.bindings.user_data_registers.size() ||
	    program.bindings.memory_offset_count != program.info.buffers.size() ||
	    (!has_user_storage &&
	     program.bindings.push_constant_size != program.bindings.ShaderDataDwords() * 4u) ||
	    (has_user_storage && program.bindings.push_constant_size != 0) ||
	    !std::is_sorted(program.bindings.user_data_registers.begin(),
	                    program.bindings.user_data_registers.end()) ||
	    std::adjacent_find(program.bindings.user_data_registers.begin(),
	                       program.bindings.user_data_registers.end()) !=
	        program.bindings.user_data_registers.end()) {
		Fail(program, "native user-data layout is inconsistent");
	}

	const auto planning_only_handle = [&](const IR::Inst& handle) {
		return !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       const auto op = use.user->GetOpcode();
			       if (op != IR::ValueOpcode::LoadAddressU32 &&
			           op != IR::ValueOpcode::ReadConstBuffer) {
				       return false;
			       }
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].planning_only;
		       });
	};
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto dense = inst.Flags<uint32_t>();
			switch (inst.GetOpcode()) {
				case IR::ValueOpcode::GetBufferResource:
					if (planning_only_handle(inst)) {
						break;
					}
					if (dense >= program.info.buffers.size()) {
						Fail(program, "typed buffer handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetAddressResource:
					if (planning_only_handle(inst)) {
						break;
					}
					if (inst.NumArgs() != 2 || !program.info.uses_dma) {
						Fail(program, "typed address handle has invalid DMA metadata");
					}
					break;
				case IR::ValueOpcode::GetScratchResource:
					if (inst.NumArgs() != 0 || program.scratch_dwords == 0) {
						Fail(program, "typed scratch handle has invalid shader metadata");
					}
					break;
				case IR::ValueOpcode::GetImageResource:
					if (dense >= program.info.images.size()) {
						Fail(program, "typed image handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetSamplerResource:
					if (dense >= program.info.samplers.size()) {
						Fail(program, "typed sampler handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::ReadConst: {
					const auto slot = inst.Arg(1).Resolve();
					if (!slot.IsImmediate() || slot.GetType() != IR::Type::U32 ||
					    slot.U32() >= program.srt_reads.size()) {
						Fail(program, "flattened SRT read has an invalid dense slot");
					}
					break;
				}
				default: break;
			}
		}
	}
}

} // namespace

void AnalyzeProgramRequirements(IR::Program& program) {
	program.spirv_requirements.reset();
	IR::SpirvRequirements requirements {};
	const auto MarkBallot = [&] { requirements.subgroup_ballot = true; };
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto address_access = IR::AddressOpcodeInfoOf(inst.GetOpcode()).access;
			if (address_access != IR::AddressAccess::None) {
				const auto memory_index = inst.Flags<IR::MemoryFlags>().index;
				if (memory_index >= program.memory_info.size()) {
					Fail(program, "address operation has invalid memory metadata");
				}
				if (program.memory_info[memory_index].kind == IR::ResourceKind::Scratch) {
					if (program.scratch_dwords == 0) {
						Fail(program, "scratch operation has no per-thread storage");
					}
					requirements.function_scratch = true;
				} else if (address_access == IR::AddressAccess::Write) {
					Fail(program, "writable FLAT/GLOBAL addresses require GPU ownership tracking");
				}
			}
			if (IR::BufferAccessOf(inst.GetOpcode()) != IR::BufferAccess::None) {
				const auto memory_index = inst.Flags<IR::MemoryFlags>().index;
				if (memory_index >= program.memory_info.size()) {
					Fail(program, "buffer operation has invalid memory metadata");
				}
				const auto& memory = program.memory_info[memory_index];
				if (memory.kind == IR::ResourceKind::Buffer) {
					if (memory.resource >= program.info.buffers.size()) {
						Fail(program, "buffer operation has invalid resource metadata");
					}
					if ((program.info.buffers[memory.resource].packed_stride & (1u << 20u)) != 0u) {
						if (program.stage != ShaderType::Compute) {
							Fail(program, "buffer ADD_TID is only valid for compute shaders");
						}
						requirements.subgroup_local_invocation_id = true;
					}
				}
			}
			const auto shared_access = IR::SharedAccessOf(inst.GetOpcode());
			if (shared_access != IR::SharedAccess::None) {
				const auto index = inst.Flags<IR::MemoryFlags>().index;
				if (index >= program.memory_info.size()) {
					Fail(program, "shared operation has invalid memory metadata");
				}
				const auto kind = program.memory_info[index].kind;
				if (kind != IR::ResourceKind::Lds && kind != IR::ResourceKind::Gds) {
					Fail(program, "shared operation has invalid resource kind");
				}
				if (program.stage != ShaderType::Compute && kind == IR::ResourceKind::Lds) {
					requirements.function_lds = true;
				}
				if (shared_access == IR::SharedAccess::Append ||
				    shared_access == IR::SharedAccess::Consume) {
					MarkBallot();
					requirements.subgroup_shuffle             = true;
					requirements.subgroup_local_invocation_id = true;
				}
			}
			switch (inst.GetOpcode()) {
				case IR::ValueOpcode::Ballot: MarkBallot(); break;
				case IR::ValueOpcode::DppMoveU32:
				case IR::ValueOpcode::ReadFirstLane:
				case IR::ValueOpcode::ReadLane: {
					MarkBallot();
					requirements.subgroup_shuffle = true;
					if (inst.GetOpcode() == IR::ValueOpcode::DppMoveU32) {
						requirements.subgroup_local_invocation_id = true;
					}
					break;
				}
				case IR::ValueOpcode::DppUpdateU32:
				case IR::ValueOpcode::WqmMask:
				case IR::ValueOpcode::WriteLane: {
					MarkBallot();
					requirements.subgroup_local_invocation_id = true;
					break;
				}
				case IR::ValueOpcode::Permlane16U32: {
					MarkBallot();
					requirements.subgroup_shuffle             = true;
					requirements.subgroup_local_invocation_id = true;
					break;
				}
				case IR::ValueOpcode::SwizzleU32:
				case IR::ValueOpcode::BpermuteU32: {
					MarkBallot();
					requirements.subgroup_shuffle             = true;
					requirements.subgroup_local_invocation_id = true;
					break;
				}
				case IR::ValueOpcode::LaneId:
					requirements.subgroup_local_invocation_id = true;
					break;
				case IR::ValueOpcode::ImageQueryLod: requirements.compute_derivatives = true; break;
				case IR::ValueOpcode::ImageGatherRaw:
					requirements.image_gather_extended = true;
					break;
				case IR::ValueOpcode::SetAttribute: {
					const auto index = inst.Flags<IR::ExportFlags>().index;
					if (index >= program.export_info.size()) {
						Fail(program, "attribute export has invalid metadata");
					}
					if (program.stage == ShaderType::Pixel &&
					    program.export_info[index].vm) {
						requirements.pixel_valid_mask = true;
					}
					break;
				}
				default: break;
			}
		}
	}
	program.spirv_requirements.emplace(requirements);
}

std::vector<uint32_t> EmitProgram(const IR::Program& program,
                                  const IR::ResourceSnapshot& resources,
                                  ShaderStageInputInfo input_info) {
	using namespace Emitter;

	if (program.stage != ShaderType::Compute && program.stage != ShaderType::Vertex &&
	    program.stage != ShaderType::Pixel) {
		Fail(program, "binary SPIR-V emitter supports compute, vertex, and pixel shaders");
	}
	if (!program.srt_plan_complete || !program.resource_tracking_complete ||
	    !program.shader_info_complete || !program.binding_layout_complete ||
	    !program.spirv_requirements.has_value()) {
		Fail(program, "SPIR-V emitter requires a fully planned native shader program");
	}
	if (!IR::ValidateResourceSnapshot(program, resources)) {
		Fail(program, "resource snapshot is inconsistent with the native shader plan");
	}
	if (!IR::ValidateResourceSpecialization(program, resources)) {
		Fail(program, "resource specialization is inconsistent with the native shader plan");
	}
	ValidateNativeProgram(program);
	IR::ValidateProgram(program, true);
	EmitterState state(program, resources, input_info);
	state.stage     = program.stage;
	state.wave_size = program.wave_size;
	state.inputs.reserve(program.info.inputs.size());
	state.outputs.reserve(program.info.outputs.size());
	state.interface_variables.reserve(program.info.inputs.size() + program.info.outputs.size());
	CopyProgramInputsAndOutputs(state, program);
	if (program.stage == ShaderType::Compute && program.wave_size == 64u &&
	    input_info.compute != nullptr && input_info.compute->needs_lds_barriers &&
	    input_info.compute->threads_num[0] * input_info.compute->threads_num[1] *
	            input_info.compute->threads_num[2] ==
	        64u) {
		CollectConvergedWave64ReadLanes(state, program);
	}
	if (!state.wave64_read_lane_scratch_banks.empty() &&
	    std::ranges::none_of(state.inputs, [](const Emitter::InputBinding& input) {
		    return input.kind == IR::StageInputKind::LocalInvocationIndex;
	    })) {
		state.inputs.push_back({IR::StageInputKind::LocalInvocationIndex, 0, 1, 0,
		                        "gl_LocalInvocationIndex", false});
	}
	AllocateInputVariables(state);
	AllocateOutputVariables(state);
	DefineModule(state);
	EmitProgram(state, program);

	auto binary = state.builder.Build();
	if (binary.empty()) {
		Fail(program, "SPIR-V builder returned an empty module");
	}
	return binary;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
