#ifndef EMULATOR_SRC_DEBUGGER_TARGET_GRAPHICS_H_
#define EMULATOR_SRC_DEBUGGER_TARGET_GRAPHICS_H_

#include "common/common.h"

#include <cstdint>
#include <string>
#include <vector>

// Observation layer for the GPU side of the debugger.
//
// Everything here is read-only: the emulator's graphics code reports what it did, and the
// overlay renders it. Nothing halts or reorders the GPU thread — a PM4 breakpoint has to stop
// that thread between submits, never while it holds the command scheduler, and getting that
// wrong deadlocks presentation along with the debugger that is meant to be showing it. That is
// a separate piece of work; see docs/debugger-design.md.
namespace Debugger::Graphics {

enum class ShaderStage : uint8_t { Unknown, Vertex, Pixel, Compute, Fetch };

const char* StageName(ShaderStage stage);

struct ShaderSummary {
	uint64_t    hash         = 0;
	uint64_t    base_address = 0;
	ShaderStage stage        = ShaderStage::Unknown;
	uint32_t    gcn_bytes    = 0;
	uint32_t    spirv_words  = 0;
	uint32_t    resource_count = 0;
	uint32_t    sequence     = 0; // order in which the recompiler first produced it
};

struct ShaderCode {
	struct Resource {
		std::string           kind;
		uint32_t              index        = 0;
		uint32_t              source       = 0;
		uint32_t              first_use_pc = 0;
		bool                  read         = false;
		bool                  written      = false;
		bool                  atomic       = false;
		std::vector<uint32_t> descriptor;
		uint64_t              address = 0; // decoded guest address for the captured specialization
		uint64_t              size    = 0; // exact for buffers; images are resolved by address
		uint32_t              width   = 0;
		uint32_t              height  = 0;
		uint32_t              depth   = 0;
		uint32_t              format  = 0;
		uint32_t              tile    = 0;
	};

	std::string isa;   // decoded RDNA2
	std::string ir;    // recompiler IR
	std::string spirv; // SPIR-V disassembly, produced on demand
	std::vector<Resource> resources; // specialization snapshot captured with this compilation
};

// True while the debugger is enabled, i.e. while any of this should be captured at all.
bool IsCapturing();

// The recompiler only builds its ISA and IR text when asked, because the strings are large.
// The debugger wants them so the shader views have something to show.
bool WantsShaderText();

// Reported by the shader recompiler once per shader it produces.
void RecordShader(ShaderStage stage, uint64_t hash, uint64_t base_address, uint32_t gcn_bytes,
                  const uint32_t* spirv, size_t spirv_words, const std::string& isa,
                  const std::string& ir, const std::vector<ShaderCode::Resource>& resources = {});

std::vector<ShaderSummary> Shaders();

// SPIR-V is disassembled here rather than at compile time, so the cost is only paid for shaders
// somebody actually opens.
bool GetShaderCode(uint64_t hash, ShaderCode& out);

enum class DrawKind : uint8_t { Draw, DrawIndexed, DrawIndirect, Dispatch };

const char* KindName(DrawKind kind);

struct DrawRecord {
	uint32_t frame      = 0;
	uint32_t index      = 0; // position within the frame
	uint64_t submit_id  = 0;
	DrawKind kind       = DrawKind::Draw;
	uint32_t count      = 0; // vertices or indices
	uint32_t instances  = 0;
	uint32_t groups[3]  = {}; // compute thread groups
	uint64_t vs_address = 0;
	uint64_t ps_address = 0;
	uint64_t cs_address = 0;
};

void RecordDraw(const DrawRecord& record);

// Exact top-level PM4 streams for a submitted graphics/compute batch. These bytes are enough to
// inspect and validate the command stream offline. They are deliberately not called executable
// replay yet: indirect command buffers and the guest resources referenced by descriptors must be
// snapshotted before executing the stream against an isolated state can be deterministic.
struct SubmissionRecord {
	uint64_t              submit_id = 0;
	uint32_t              frame = 0;
	uint32_t              queue_id = 0;
	bool                  compute = false;
	bool                  interrupt_on_done = false;
	bool                  reset_processor = false;
	bool                  truncated = false;
	std::vector<uint32_t> commands;
	std::vector<uint32_t> constant_commands;
};

void RecordSubmission(uint64_t submit_id, uint32_t queue_id, bool compute,
                      const uint32_t* commands, uint32_t command_words,
                      const uint32_t* constant_commands, uint32_t constant_words,
                      bool interrupt_on_done, bool reset_processor);
bool GetSubmission(uint64_t submit_id, SubmissionRecord& out);

// Called when the command processor flips, which is what closes a frame.
void RecordFlip();

// Draws captured for the frame most recently completed, so the list does not churn while it is
// being read.
std::vector<DrawRecord> LastFrame();

struct Stats {
	uint32_t frame                 = 0;
	uint32_t draws_last_frame      = 0;
	uint32_t dispatches_last_frame = 0;
	uint32_t draws_this_frame      = 0;
	uint64_t total_draws           = 0;
	uint64_t total_dispatches      = 0;
	uint64_t total_flips           = 0;
	uint32_t shader_count          = 0;
	bool     truncated             = false; // the per-frame list hit its cap
};

Stats GetStats();

struct ResourceEvent {
	uint64_t    sequence     = 0;
	uint64_t    timestamp_us = 0;
	uint32_t    frame        = 0;
	uint32_t    command_index = UINT32_MAX;
	uint64_t    submit_id     = 0;
	DrawKind    command_kind  = DrawKind::Draw;
	uint64_t    vs_address    = 0;
	uint64_t    ps_address    = 0;
	uint64_t    cs_address    = 0;
	std::string action;
	std::string note;
	uint32_t    image_index      = UINT32_MAX;
	uint32_t    image_generation = 0;
	uint64_t    host_image       = 0;
	bool        active           = true;
	uint64_t    address      = 0;
	uint64_t    size         = 0;
	uint64_t    stencil_address  = 0;
	uint64_t    stencil_size     = 0;
	uint64_t    metadata_address = 0;
	uint64_t    metadata_size    = 0;
	uint32_t    width        = 0;
	uint32_t    height       = 0;
	uint32_t    depth        = 0;
	uint32_t    pitch        = 0;
	uint32_t    bytes_per_block = 0;
	uint32_t    guest_format = 0;
	uint32_t    host_format  = 0;
	uint32_t    tile_mode    = 0;
	uint32_t    image_type   = 0;
	uint32_t    samples      = 1;
	uint32_t    levels       = 1;
	uint32_t    layers       = 1;
	uint32_t    metadata_kind = 0;
	bool        registered      = false;
	bool        cpu_dirty       = false;
	bool        maybe_cpu_dirty = false;
	bool        buffer_modified = false;
	bool        gpu_modified    = false;
	bool        usage_texture       = false;
	bool        usage_storage       = false;
	bool        usage_render_target = false;
	bool        usage_depth_target  = false;
	bool        usage_video_out     = false;
	bool        bound              = false;
	bool        target             = false;
	bool        needs_rebind       = false;
	bool        force_general      = false;
	bool        shader_write       = false;
};

// Lightweight image ownership/materialization history. This deliberately records metadata,
// not image bytes; actual previews need a scheduler-safe GPU readback request. Rich events are
// supplied by TextureCache, while the legacy overload keeps focused tests and simple producers
// concise.
void RecordImageEvent(ResourceEvent event);
void RecordImageEvent(const char* action, uint64_t address, uint64_t size, uint32_t width,
                      uint32_t height, uint32_t depth, uint32_t bytes_per_block,
                      uint32_t guest_format, uint32_t host_format, uint32_t tile_mode);
std::vector<ResourceEvent> ResourceHistory(uint64_t address = 0, size_t limit = 1024);
std::vector<ResourceEvent> ResourceTrace(uint64_t address, uint64_t size, size_t limit = 4096);
std::vector<ResourceEvent> ResourceAliases(uint64_t address, uint64_t size);

// A pixel watchpoint resolves a logical mip-0 coordinate to its exact guest backing byte when
// the captured image layout is supported. Resource events are image-granular today, so a hit
// means "this command changed or reinterpreted the image containing this byte"; it does not
// claim that a partial draw necessarily covered the pixel.
struct PixelWatch {
	uint64_t    id            = 0;
	uint64_t    address       = 0;
	uint32_t    x             = 0;
	uint32_t    y             = 0;
	uint64_t    pixel_address = 0;
	uint32_t    pixel_size    = 0;
	bool        exact         = false;
	uint64_t    hits          = 0;
	std::string status;
};

struct PixelWatchHit {
	uint64_t      sequence      = 0;
	uint64_t      watch_id      = 0;
	uint32_t      x             = 0;
	uint32_t      y             = 0;
	uint64_t      pixel_address = 0;
	uint32_t      pixel_size    = 0;
	bool          exact         = false;
	std::string   status;
	ResourceEvent event;
};

uint64_t                   AddPixelWatch(uint64_t address, uint32_t x, uint32_t y);
bool                       RemovePixelWatch(uint64_t id);
void                       ClearPixelWatches();
std::vector<PixelWatch>    PixelWatches();
std::vector<PixelWatchHit> PixelWatchHits(size_t limit = 4096);

enum class BreakConditionKind : uint8_t { Shader, Resource, PixelWatch, NonFinite };
const char* BreakConditionKindName(BreakConditionKind kind);
bool ParseBreakConditionKind(const std::string& name, BreakConditionKind& out);

struct BreakCondition {
	uint64_t           id = 0;
	BreakConditionKind kind = BreakConditionKind::Shader;
	uint64_t           value = 0; // shader hash/base, resource address, watch id, or optional image base
	std::string        action;    // optional resource-action substring
	bool               enabled = true;
	bool               one_shot = false;
	uint64_t           hits = 0;
	std::string        last_reason;
};

uint64_t AddBreakCondition(BreakConditionKind kind, uint64_t value, const std::string& action,
                           bool one_shot);
bool RemoveBreakCondition(uint64_t id);
void ClearBreakConditions();
std::vector<BreakCondition> BreakConditions();

// Called by the GPU worker only after a submission reached its renderer safe point. Returns true
// when a matching condition requested a guest-thread pause and writes the human-readable reason.
bool TakeBreakRequest(std::string& reason);

struct PreviewRequest {
	uint64_t id      = 0;
	uint64_t address = 0;
	uint64_t size    = 0;
};

struct ResourcePreview {
	uint64_t             request_id    = 0;
	uint64_t             address       = 0;
	uint64_t             size          = 0;
	uint32_t             width         = 0;
	uint32_t             height        = 0;
	uint32_t             source_width  = 0;
	uint32_t             source_height = 0;
	uint32_t             host_format   = 0;
	std::string          status;
	std::string          error;
	uint64_t             content_hash          = 0;
	uint64_t             total_pixels          = 0;
	uint64_t             zero_pixels           = 0;
	uint64_t             non_finite_pixels     = 0;
	uint64_t             non_finite_components = 0;
	std::vector<uint8_t> rgba;
};

struct PreviewDiagnostics {
	uint64_t content_hash          = 0;
	uint64_t total_pixels          = 0;
	uint64_t zero_pixels           = 0;
	uint64_t non_finite_pixels     = 0;
	uint64_t non_finite_components = 0;
};

// The UI requests a preview; the GPU thread consumes it at a scheduler-safe point and publishes
// a small RGBA thumbnail asynchronously after the copy completes.
uint64_t RequestResourcePreview(uint64_t address, uint64_t size, bool refresh);
bool     TakePreviewRequest(PreviewRequest& out);
void     PublishResourcePreview(uint64_t request_id, uint64_t address, uint32_t source_width,
                                uint32_t source_height, uint32_t host_format, uint32_t width,
                                uint32_t height, std::vector<uint8_t> rgba,
	                            const std::string& error = {},
	                            PreviewDiagnostics diagnostics = {});
ResourcePreview GetResourcePreview(uint64_t address, uint64_t size);

// Write a shader's ISA, IR and SPIR-V next to the other shader dumps. Returns the folder used.
bool DumpShader(uint64_t hash, std::string& path_out);

void Reset();

} // namespace Debugger::Graphics

#endif // EMULATOR_SRC_DEBUGGER_TARGET_GRAPHICS_H_
