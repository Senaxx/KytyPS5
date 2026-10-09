#include "graphics/guest_gpu/drawPrepScanner.h"

#include "common/profiler.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/drawPrep.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderDraw.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <vector>

namespace Libs::Graphics {

namespace {

bool IsGraphicsDraw(uint32_t opcode) {
	return DrawPrep::CountsAsDrawOrDispatch(opcode) && opcode != Pm4::IT_DISPATCH_DIRECT &&
	       opcode != Pm4::IT_DISPATCH_INDIRECT;
}

} // namespace

DrawPrepScanner::DrawPrepScanner(RenderContext& renderer): m_renderer(renderer) {
	m_thread = std::thread([this] { Run(); });
}

DrawPrepScanner::QueueState& DrawPrepScanner::State(uint32_t queue) {
	auto& state = m_queues[queue % m_queues.size()];
	if (state == nullptr) {
		state            = std::make_unique<QueueState>();
		state->cp        = std::make_unique<CommandProcessor>(m_renderer, queue == 0 ? 0 : 0x1f + static_cast<int>(queue));
		state->saved_ctx = std::make_unique<HW::Context>();
		// As GuestGpu: the graphics processor starts reset, a compute one as constructed.
		if (queue == 0) {
			state->cp->Reset();
		}
	}
	return *state;
}

DrawPrepScanner::~DrawPrepScanner() {
	{
		std::lock_guard lock(m_mutex);
		m_stop = true;
	}
	m_work.notify_one();
	if (m_thread.joinable()) {
		m_thread.join();
	}
}

void DrawPrepScanner::EnqueueSubmission(uint32_t queue, uint64_t seq,
                                        std::span<const uint32_t> commands) {
	{
		std::lock_guard lock(m_mutex);
		m_items.push_back({queue, seq, commands});
	}
	m_work.notify_one();
}

void DrawPrepScanner::EnqueueReset(uint32_t queue) {
	{
		std::lock_guard lock(m_mutex);
		m_items.push_back({queue, 0, {}});
	}
	m_work.notify_one();
}

void DrawPrepScanner::Run() {
	KYTY_PROFILER_THREAD("Thread_DrawPrep");
	// The clean-backing reads (TryReadGpuCleanBacking) answer "maybe GPU-written" on any other
	// thread; a read delegate checks the buffer cache's GPU-dirty ranges under their shared lock,
	// which the GPU thread holds exclusively whenever it changes them.
	LibKernel::Memory::SetGpuReadDelegate(true);
	for (;;) {
		Item item;
		{
			std::unique_lock lock(m_mutex);
			m_work.wait(lock, [this] { return m_stop || !m_items.empty(); });
			if (m_stop) {
				return;
			}
			item = m_items.front();
			m_items.pop_front();
		}
		m_queue = item.queue;
		m_state = &State(item.queue);
		if (item.seq == 0) {
			// The GPU thread resets its command processor at a suspend point.
			m_state->cp->Reset();
			m_state->pushed  = false;
			m_state->tainted = false;
		} else {
			Scan(item);
		}
		DrawPrep::LogCountersIfDue();
	}
}

// As CommandProcessor::ApplyContextStateOperation, without its exits: a sequence the scanner
// cannot follow only makes its register copy unreliable.
void DrawPrepScanner::ApplyContextState(uint32_t operation) {
	auto&      ctx        = m_state->cp->GetCtx();
	const bool was_tainted = m_state->tainted;
	struct Count {
		const bool& was;
		const bool& now;
		~Count() {
			if (!was && now) {
				DrawPrep::Add(DrawPrep::Counter::TaintContext);
			}
		}
	} count {was_tainted, m_state->tainted};
	switch (static_cast<ContextStateOperation>(operation)) {
		case ContextStateOperation::Clear: ctx.Reset(); break;
		case ContextStateOperation::Push:
			m_state->tainted |= m_state->pushed;
			*m_state->saved_ctx = ctx;
			m_state->pushed     = true;
			break;
		case ContextStateOperation::Pop:
			m_state->tainted |= !m_state->pushed;
			ctx      = *m_state->saved_ctx;
			m_state->pushed = false;
			break;
		case ContextStateOperation::PushClear:
			m_state->tainted |= m_state->pushed;
			*m_state->saved_ctx = ctx;
			m_state->pushed     = true;
			ctx.Reset();
			break;
		default: m_state->tainted = true; break;
	}
}

// SET_*_REG_INDIRECT: the register pairs are read with the clean backing read (no fault, no wait)
// and handed to the command processor's own handler from a local copy. False when the pairs
// cannot be read that way (the GPU wrote them): the register copy is then unreliable.
bool DrawPrepScanner::ApplyRegisterIndirect(uint32_t header, const uint32_t* body) {
	if (KYTY_PM4_LEN(header) != 5u) {
		return false;
	}
	const uint64_t address =
	    (static_cast<uint64_t>(body[0]) & 0xfffffffcu) | (static_cast<uint64_t>(body[1]) << 32u);
	const uint32_t count = body[3] & 0x3fffu;
	if (count == 0) {
		return true;
	}
	if (address == 0) {
		return false;
	}
	thread_local std::vector<uint32_t> pairs;
	pairs.resize(static_cast<size_t>(count) * 2u);
	if (!LibKernel::Memory::TryReadGpuCleanBacking(address, pairs.data(),
	                                               pairs.size() * sizeof(uint32_t))) {
		return false;
	}
	const auto                    local = reinterpret_cast<uint64_t>(pairs.data());
	const std::array<uint32_t, 4> copy {static_cast<uint32_t>(local), static_cast<uint32_t>(local >> 32u),
	                                    body[2], body[3]};
	const auto opcode = (header >> 8u) & 0xffu;
	(void)g_cp_op_func[opcode](*m_state->cp, header, copy.data(), 5u, 5u);
	return true;
}

void DrawPrepScanner::ScanDraw(uint64_t seq, uint32_t ordinal, const uint32_t* packet) {
	if (m_state->tainted) {
		DrawPrep::Add(DrawPrep::Counter::Tainted);
		return;
	}
	const DrawPrep::Key key {.queue = m_queue, .seq = seq, .ordinal = ordinal, .packet = packet};
	if (!DrawPrep::GpuProgress(m_queue).Before(key)) {
		DrawPrep::Add(DrawPrep::Counter::SkipBehind);
		return;
	}
	const auto& ctx  = m_state->cp->GetCtx();
	const auto& sh   = m_state->cp->GetShCtx();
	const auto& ucfg = m_state->cp->GetUcfg();
	if (!DrawPrepHasVertexShader(sh)) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
		return;
	}
	std::array<Prospero::ColorComponentMapping, 8> mapping {};
	const bool ps_active = DrawPrepPixelState(ctx, sh, mapping);
	DrawPrep::PacketResult result;
	DrawPrep::Recycle(result);
	result.key = key;
	if (m_renderer.GetPipelineCache().DrawPrepGraphics(sh.GetVs(), sh.GetPs(),
	                                                   ctx.GetShaderRegisters(), ctx, ucfg, mapping,
	                                                   ps_active, result)) {
		DrawPrep::Publish(std::move(result));
	}
}

// A dispatch: the compute inputs RenderExecutor::DispatchDirect / DispatchIndirect build.
void DrawPrepScanner::ScanDispatch(uint64_t seq, uint32_t ordinal, const uint32_t* packet,
                                   uint32_t opcode) {
	if (m_state->tainted) {
		DrawPrep::Add(DrawPrep::Counter::Tainted);
		return;
	}
	const DrawPrep::Key key {.queue = m_queue, .seq = seq, .ordinal = ordinal, .packet = packet};
	if (!DrawPrep::GpuProgress(m_queue).Before(key)) {
		DrawPrep::Add(DrawPrep::Counter::SkipBehind);
		return;
	}
	const auto& cs = m_state->cp->GetShCtx().GetCs();
	if (cs.cs_regs.data_addr == 0 || PipelineCache::IsSkipListed(cs.cs_regs.data_addr)) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
		return;
	}
	ShaderComputeInputInfo input_info {};
	if (opcode == Pm4::IT_DISPATCH_DIRECT) {
		if (KYTY_PM4_LEN(packet[0]) != 5u) {
			DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
			return;
		}
		const uint32_t groups[3] = {packet[1], packet[2], packet[3]};
		const uint32_t mode      = packet[4];
		if (groups[0] == 0 || groups[1] == 0 || groups[2] == 0) {
			DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
			return;
		}
		const bool use_thread_dimensions =
		    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
		input_info.dispatch_thread_dimensions = use_thread_dimensions;
		const uint32_t group_sizes[3] = {cs.cs_regs.num_thread_x, cs.cs_regs.num_thread_y,
		                                 cs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			input_info.workgroup_counts[axis] = groups[axis];
			if (use_thread_dimensions) {
				const auto size = std::max(group_sizes[axis], 1u);
				input_info.workgroup_counts[axis] = groups[axis] / size + (groups[axis] % size != 0u);
			}
		}
	} else {
		const uint32_t mode = packet[0] == 0xc0021600u ? packet[3] : packet[2];
		const bool     use_thread_dimensions =
		    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
		input_info.dispatch_thread_dimensions   = use_thread_dimensions;
		input_info.dispatch_dimensions_indirect = use_thread_dimensions;
	}
	DrawPrep::PacketResult result;
	DrawPrep::Recycle(result);
	result.key = key;
	if (m_renderer.GetPipelineCache().DrawPrepCompute(cs, m_state->cp->GetCtx().GetShaderRegisters(),
	                                                  input_info, result)) {
		DrawPrep::Publish(std::move(result));
	}
}

// The packet walk of CommandProcessor::ProcessPm4, for registers only: register writes go to the
// scanner's command processor, draws are prepared, everything else is stepped over.
void DrawPrepScanner::Scan(const Item& item) {
	struct Cursor {
		std::span<const uint32_t> commands;
		uint32_t                  offset = 0;
	};
	std::vector<Cursor> stack {{item.commands}};
	uint32_t            ordinal = 0;
	while (!stack.empty()) {
		auto& cursor = stack.back();
		if (cursor.offset >= cursor.commands.size()) {
			stack.pop_back();
			continue;
		}
		const auto* const packet    = cursor.commands.data() + cursor.offset;
		const auto        total     = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining = total - cursor.offset;
		const auto        header    = packet[0];
		if (header == 0x80000000u) {
			cursor.offset++;
			continue;
		}
		if (remaining < 2) {
			return;
		}
		if ((header >> 30u) == 0u) {
			const auto length = ((header >> 16u) & 0x3fffu) + 2u;
			if (length > remaining) {
				return;
			}
			cursor.offset += length;
			continue;
		}
		const auto opcode = (header >> 8u) & 0xffu;
		const auto length = KYTY_PM4_LEN(header);
		if (length > remaining) {
			return;
		}
		if (DrawPrep::CountsAsDrawOrDispatch(opcode)) {
			const auto this_ordinal = ordinal++;
			DrawPrep::Add(DrawPrep::Counter::Scanned);
			if (IsGraphicsDraw(opcode)) {
				ScanDraw(item.seq, this_ordinal, packet);
			} else {
				ScanDispatch(item.seq, this_ordinal, packet, opcode);
			}
			cursor.offset += length;
			continue;
		}
		const auto command = header & ~1u;
		switch (opcode) {
			case Pm4::IT_SET_CONTEXT_REG:
			case Pm4::IT_SET_SH_REG:
			case Pm4::IT_SET_UCONFIG_REG:
			case Pm4::IT_SET_UCONFIG_REG_INDEX: {
				const auto consumed = g_cp_op_func[opcode](*m_state->cp, command, packet + 1, remaining, total) + 1u;
				cursor.offset += consumed;
				continue;
			}
			case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
			case Pm4::IT_SET_SH_REG_INDIRECT:
			case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
				if (!ApplyRegisterIndirect(command, packet + 1)) {
					m_state->tainted = true;
					DrawPrep::Add(DrawPrep::Counter::TaintIndirect);
				}
				break;
			case Pm4::IT_CLEAR_STATE: m_state->cp->GetCtx().Reset(); break;
			case Pm4::IT_INDIRECT_BUFFER: {
				if (length != 4u) {
					// A conditional branch on guest memory: the scanner cannot follow it.
					return;
				}
				const auto* sub = reinterpret_cast<const uint32_t*>(
				    packet[1] | (static_cast<uint64_t>(packet[2]) << 32u));
				const uint32_t control = packet[3];
				const uint32_t count   = control & 0xfffffu;
				cursor.offset += length;
				if (count != 0 && sub != nullptr) {
					if ((control & (1u << 20u)) != 0) {
						cursor = {std::span(sub, count)};
					} else {
						stack.push_back({std::span(sub, count)});
					}
				}
				continue;
			}
			case Pm4::IT_NOP: {
				const auto r = KYTY_PM4_R(header);
				if (r == Pm4::R_ZERO) {
					if ((packet[1] & 0xffff0000u) == 0x68750000u) {
						const auto id = packet[1] & 0xfffu;
						if (id == 0x4u || id == 0xdu) {
							(void)g_cp_op_func[Pm4::IT_NOP](*m_state->cp, command, packet + 1, remaining,
							                                total);
						}
					}
				} else if (r == Pm4::R_CONTEXT_STATE) {
					ApplyContextState(packet[1]);
				} else if (r == Pm4::R_DISPATCH_RESET) {
					m_state->cp->Reset();
					m_state->pushed = false;
				}
				break;
			}
			default: break;
		}
		cursor.offset += length;
	}
}

} // namespace Libs::Graphics
