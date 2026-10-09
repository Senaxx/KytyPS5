#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_H_

#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <span>
#include <vector>

// Draw preparation ahead of the GPU thread (on by default; KYTY_DRAW_PREP=0 turns it off;
// drawprep-plan.md).
//
// A scanner thread (guest_gpu/drawPrepScanner.cpp) reads every graphics submission as soon as it
// is queued, follows its register writes on a command processor of its own, and for each draw
// runs the descriptor walk (the SRT refresh of ProgramCache::Get) with a reader that only takes
// bytes no GPU work has written and records them. The GPU thread executes exactly as before; where
// it would walk, it takes the prepared result instead when the program, user data and shader base
// are the same and every recorded range still holds the same bytes and is still clean. Otherwise
// it walks as before, so correctness never depends on what happened in between.
//
// KYTY_DRAW_PREP_VERIFY=1|exit: every result taken is compared with a walk on the GPU thread.

namespace Libs::Graphics::DrawPrep {

[[nodiscard]] bool Enabled();
[[nodiscard]] int  VerifyMode(); // 0 off, 1 count and log, 2 exit on a difference

// GPU queues (GuestGpu: graphics 0, compute 1..56).
constexpr uint32_t MaxQueues = 64;

// A draw or dispatch packet: its queue, its submission (numbered at enqueue), its position among
// the draw and dispatch packets of that submission's command stream, and its address (a check that
// both sides counted the same packets).
struct Key {
	uint32_t        queue   = 0;
	uint64_t        seq     = 0;
	uint32_t        ordinal = 0;
	const uint32_t* packet  = nullptr;

	[[nodiscard]] bool operator==(const Key&) const = default;
	[[nodiscard]] bool Before(const Key& other) const {
		return seq < other.seq || (seq == other.seq && ordinal < other.ordinal);
	}
};

// Whether a packet opcode counts as a draw or dispatch for Key::ordinal (both sides use this).
[[nodiscard]] bool CountsAsDrawOrDispatch(uint32_t opcode);

// The guest bytes a preparation read, in read order, merged where contiguous.
class ReadSet {
public:
	void Clear() {
		m_ranges.clear();
		m_bytes.clear();
	}
	void Add(uint64_t address, const void* data, uint32_t size);
	// GPU thread: every range is still clean for a backing read and holds the recorded bytes.
	[[nodiscard]] bool Validate() const;
	[[nodiscard]] size_t RangeCount() const { return m_ranges.size(); }

private:
	struct Range {
		uint64_t address;
		uint32_t offset; // into m_bytes
		uint32_t size;
	};
	std::vector<Range>   m_ranges;
	std::vector<uint8_t> m_bytes;
};

// One stage's prepared refresh.
struct StageResult {
	bool                                         ok       = false;
	bool                                         consumed = false;
	uint64_t                                     serial   = 0; // unique per preparation
	uint32_t                                     stage    = 0; // ShaderType
	const void*                                  entry    = nullptr; // ProgramCache source entry
	uint64_t                                     shader_base = 0;
	std::vector<uint32_t>                        user_data;
	// Compute: the dispatch size the walk saw, and whether GPU-written slots were left to the GPU
	// (SrtGpuFill; their addresses are in resources.gpu_fills).
	std::array<uint32_t, 3>                      workgroup_counts {};
	std::array<uint32_t, 3>                      workgroup_size {};
	bool                                         gpu_fills = false;
	ShaderRecompiler::IR::ResourceSnapshot       resources;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	ReadSet                                      reads;
};

struct PacketResult {
	Key                        key;
	std::array<StageResult, 2> stages;
	uint32_t                   stage_count = 0;
};

// GPU thread: the draw or dispatch packet being executed, and its progress per queue (which the
// scanner reads to skip packets the GPU thread has already passed).
void        SetCurrent(const Key& key);
void        ClearCurrent();
[[nodiscard]] const Key* Current();
[[nodiscard]] Key        GpuProgress(uint32_t queue);

// Scanner: a prepared packet, in stream order. Waits a little while too many results wait.
void Publish(PacketResult&& result);
// Scanner: a result to fill, reusing the buffers of one the GPU thread is done with.
void Recycle(PacketResult& result);
// GPU thread: the prepared refresh of `stage` for the current packet, or null; consumed ones too
// (a later draw of the same packet may use it again). The result stays owned by the store.
[[nodiscard]] StageResult* Take(uint32_t stage);

// Counters (always counted; logged every 10 s while enabled).
enum class Counter : uint32_t {
	Scanned,          // draw/dispatch packets the scanner saw
	Prepared,         // stage refreshes prepared
	SkipBehind,       // packets the GPU thread had already passed
	SkipUnknownPair,  // shader pair the GPU thread has not prepared yet
	SkipUnsupported,  // tessellation, no vertex shader, a skip-listed compute shader, ...
	SkipNoEntry,      // program not translated yet
	FailWalk,         // walk failed (unclean or unreadable bytes)
	Tainted,          // packets scanned while the register copy was unreliable
	TaintIndirect,    // register-indirect pairs that could not be read cleanly
	TaintContext,     // context-state pushes and pops the scanner could not follow
	Takes,            // GPU thread asked
	Used,             // results used
	Reused,           // results used again by a later draw of the same packet (multi-draw, retry)
	MissNone,         // no result for the current packet
	MissEntry,        // different program
	MissInputs,       // user data or shader base differ
	MissReads,        // recorded bytes changed or are no longer clean
	VerifyChecks,
	VerifyMismatches,
	Count,
};
void Add(Counter counter, uint64_t value = 1);
void LogCountersIfDue();

// GPU thread: a vertex/pixel shader pair whose programs it prepared successfully; the scanner
// prepares only such pairs (it never meets a shader state the draw path has not handled).
void NoteGoodPair(uint64_t vs_addr, uint64_t ps_addr);
[[nodiscard]] bool IsGoodPair(uint64_t vs_addr, uint64_t ps_addr);

} // namespace Libs::Graphics::DrawPrep

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_H_
