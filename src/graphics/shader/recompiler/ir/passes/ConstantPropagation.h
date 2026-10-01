#pragma once

#include "graphics/shader/recompiler/ir/Block.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

void ConstantPropagationPass(const BlockList& blocks, uint32_t wave_size = 64);
struct Program;
uint32_t SimplifyBoundedLoopRegisters(Program& program);
uint32_t SimplifyLocalAddressStores(Program& program);
// Reads of the current lane's bit of a wave mask that is known per lane (a ballot of a
// predicate, a constant, or bitwise logic of those) become that predicate. Returns how many.
uint32_t FoldLaneMasks(Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::IR
