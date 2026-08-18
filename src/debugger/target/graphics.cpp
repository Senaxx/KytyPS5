#include "debugger/target/graphics.h"

#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "debugger/core/session.h"
#include "graphics/guest_gpu/tile.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <deque>
#include <mutex>
#include <spirv-tools/libspirv.hpp>
#include <unordered_map>

namespace Debugger::Graphics {

namespace {

// The GPU thread writes these and the presentation thread reads them, so both take the lock.
// Neither runs inside a trap handler, which is what makes an ordinary mutex fine here — unlike
// the breakpoint table, which cannot use one.
std::mutex g_mutex;

struct ShaderEntry {
	ShaderSummary         summary;
	std::string           isa;
	std::string           ir;
	std::vector<uint32_t> spirv;
	std::string           spirv_text; // filled the first time it is asked for
	std::vector<ShaderCode::Resource> resources;
};

std::unordered_map<uint64_t, ShaderEntry> g_shaders;
uint32_t                                  g_shader_sequence = 0;

// One frame's worth of draws, capped: a heavy frame can issue tens of thousands, and the list
// exists to be read by a human.
constexpr size_t MAX_DRAWS_PER_FRAME = 4096;

std::vector<DrawRecord> g_current_frame;
std::vector<DrawRecord> g_last_frame;
bool                    g_current_truncated = false;
bool                    g_last_truncated    = false;

constexpr size_t MAX_SUBMISSION_RECORDS = 128;
constexpr size_t MAX_SUBMISSION_DWORDS = 4u * 1024u * 1024u;
constexpr size_t MAX_SUBMISSION_TOTAL_DWORDS = 16u * 1024u * 1024u;
std::deque<SubmissionRecord> g_submissions;
size_t                       g_submission_dwords = 0;

uint32_t g_frame                 = 0;
uint32_t g_draws_last_frame      = 0;
uint32_t g_dispatches_last_frame = 0;
uint64_t g_total_draws           = 0;
uint64_t g_total_dispatches      = 0;
uint64_t g_total_flips           = 0;

constexpr size_t MAX_RESOURCE_EVENTS = 32768;
std::vector<ResourceEvent> g_resource_events;
uint64_t                   g_resource_sequence = 0;
const auto                 g_resource_start = std::chrono::steady_clock::now();

constexpr size_t MAX_PIXEL_WATCHES = 64;
constexpr size_t MAX_PIXEL_WATCH_HITS = 8192;
std::vector<PixelWatch>    g_pixel_watches;
std::vector<PixelWatchHit> g_pixel_watch_hits;
uint64_t                   g_pixel_watch_sequence = 1;
uint64_t                   g_pixel_hit_sequence = 1;

constexpr size_t          MAX_BREAK_CONDITIONS = 64;
std::vector<BreakCondition> g_break_conditions;
uint64_t                    g_break_condition_sequence = 1;
std::string                 g_pending_break_reason;

uint64_t        g_preview_sequence = 0;
PreviewRequest  g_preview_request;
ResourcePreview g_preview;

bool g_logged_first_shader = false;
bool g_logged_first_frame  = false;

bool Disassemble(const std::vector<uint32_t>& spirv, std::string& out) {
	if (spirv.empty()) {
		return false;
	}

	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	return tools.Disassemble(spirv, &out,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT));
}

} // namespace

const char* StageName(ShaderStage stage) {
	switch (stage) {
		case ShaderStage::Vertex: return "vertex";
		case ShaderStage::Pixel: return "pixel";
		case ShaderStage::Compute: return "compute";
		case ShaderStage::Fetch: return "fetch";
		case ShaderStage::Unknown: break;
	}
	return "unknown";
}

const char* KindName(DrawKind kind) {
	switch (kind) {
		case DrawKind::Draw: return "draw";
		case DrawKind::DrawIndexed: return "draw indexed";
		case DrawKind::DrawIndirect: return "draw indirect";
		case DrawKind::Dispatch: return "dispatch";
	}
	return "?";
}

bool IsCapturing() {
	return Session::IsEnabled();
}

bool WantsShaderText() {
	return Session::IsEnabled();
}

void RecordShader(ShaderStage stage, uint64_t hash, uint64_t base_address, uint32_t gcn_bytes,
                  const uint32_t* spirv, size_t spirv_words, const std::string& isa,
                  const std::string& ir, const std::vector<ShaderCode::Resource>& resources) {
	if (!IsCapturing()) {
		return;
	}

	const std::lock_guard lock(g_mutex);

	auto& entry = g_shaders[hash];

	// The same shader is recompiled under different specialisations; keep the first record and
	// only fill in text that was missing, so opening one always shows something.
	if (entry.summary.hash == 0) {
		entry.summary.hash         = hash;
		entry.summary.base_address = base_address;
		entry.summary.stage        = stage;
		entry.summary.gcn_bytes    = gcn_bytes;
		entry.summary.sequence     = g_shader_sequence++;
	}

	entry.summary.spirv_words = static_cast<uint32_t>(spirv_words);

	// One line, the first time only: enough to tell from a log that GPU capture is actually
	// wired up, without narrating every shader in a session that compiles thousands.
	if (!g_logged_first_shader) {
		g_logged_first_shader = true;
		LOGF("Debugger: GPU capture live, first shader is %s hash=0x%016" PRIx64 "\n",
		     StageName(stage), hash);
	}

	if (entry.isa.empty() && !isa.empty()) {
		entry.isa = isa;
	}
	if (entry.ir.empty() && !ir.empty()) {
		entry.ir = ir;
	}
	if (entry.spirv.empty() && spirv != nullptr && spirv_words != 0) {
		entry.spirv.assign(spirv, spirv + spirv_words);
		entry.spirv_text.clear();
	}
	if (!resources.empty()) {
		entry.summary.resource_count = static_cast<uint32_t>(resources.size());
		entry.resources = resources;
	}
}

const char* BreakConditionKindName(BreakConditionKind kind) {
	switch (kind) {
		case BreakConditionKind::Shader: return "shader";
		case BreakConditionKind::Resource: return "resource";
		case BreakConditionKind::PixelWatch: return "pixel";
		case BreakConditionKind::NonFinite: return "nonfinite";
	}
	return "?";
}

bool ParseBreakConditionKind(const std::string& name, BreakConditionKind& out) {
	if (name == "shader") out = BreakConditionKind::Shader;
	else if (name == "resource") out = BreakConditionKind::Resource;
	else if (name == "pixel") out = BreakConditionKind::PixelWatch;
	else if (name == "nonfinite") out = BreakConditionKind::NonFinite;
	else return false;
	return true;
}

bool EventContainsWatch(const ResourceEvent& event, const PixelWatch& watch) {
	if (event.address == 0 || event.size == 0) return false;
	return watch.address == event.address ||
	       (watch.address >= event.address && watch.address - event.address < event.size);
}

PixelWatch ResolvePixelWatch(PixelWatch watch, const ResourceEvent& event) {
	watch.pixel_address = 0;
	watch.pixel_size = 0;
	watch.exact = false;
	if (watch.x >= event.width || watch.y >= event.height) {
		watch.status = "coordinate is outside the image extent";
		return watch;
	}
	if (event.samples != 1) {
		watch.status = "multisampled image: per-sample address is not resolved yet";
		return watch;
	}
	if (event.levels == 0 || event.layers == 0 || event.depth == 0 || event.bytes_per_block == 0) {
		watch.status = "incomplete image layout metadata";
		return watch;
	}

	uint64_t offset = 0;
	uint32_t pixel_size = event.bytes_per_block;
	const auto tile = static_cast<Libs::Graphics::Prospero::TileMode>(event.tile_mode);
	if (tile == Libs::Graphics::Prospero::TileMode::kLinear) {
		const uint64_t pitch = event.pitch != 0 ? event.pitch : event.width;
		offset = (static_cast<uint64_t>(watch.y) * pitch + watch.x) * pixel_size;
	} else {
		Libs::Graphics::TileSurfaceDescription description {};
		description.format = static_cast<Libs::Graphics::Prospero::BufferFormat>(event.guest_format);
		description.tile_mode = tile;
		description.dimension = Libs::Graphics::TileSurfaceDimension::Dim2D;
		description.width = std::max(event.width, event.pitch);
		description.height = event.height;
		description.depth = 1;
		description.levels = event.levels;
		description.layers = event.layers;
		Libs::Graphics::TileSurfaceLayout layout {};
		if (!Libs::Graphics::TileGetTiledTextureLayout(description, layout)) {
			watch.status = "unsupported PS5 tile/format combination";
			return watch;
		}
		const auto& block = layout.texture.block;
		const auto& mip = layout.mips[0];
		const uint32_t element_x = watch.x / layout.texture.texel_width;
		const uint32_t element_y = watch.y / layout.texture.texel_height;
		const uint32_t block_x = element_x / block.block_width;
		const uint32_t block_y = element_y / block.block_height;
		const uint32_t local_x = element_x % block.block_width;
		const uint32_t local_y = element_y % block.block_height;
		uint32_t local = 0;
		uint32_t block_xor = 0;
		if (!Libs::Graphics::TileGetBlockOffset(block, local_x, local_y, 0, local) ||
		    !Libs::Graphics::TileGetBlockXor(block, block_x, block_y, 0, block_xor)) {
			watch.status = "PS5 tile address calculation failed";
			return watch;
		}
		const uint64_t columns =
		    (mip.padded_width + block.block_width - 1u) / block.block_width;
		const uint64_t block_index = static_cast<uint64_t>(block_y) * columns + block_x;
		offset = mip.offset + block_index * block.block_size + (local ^ block_xor);
		pixel_size = block.bytes_per_element;
	}
	if (offset > event.size || pixel_size > event.size - offset ||
	    event.address > UINT64_MAX - offset) {
		watch.status = "resolved pixel escaped the guest image allocation";
		return watch;
	}
	watch.pixel_address = event.address + offset;
	watch.pixel_size = pixel_size;
	watch.exact = true;
	watch.status = "exact mip-0 backing byte; events are image-granular";
	return watch;
}

void AppendPixelHit(PixelWatch& watch, const ResourceEvent& event) {
	if (!EventContainsWatch(event, watch)) return;
	const auto resolved = ResolvePixelWatch(watch, event);
	watch.pixel_address = resolved.pixel_address;
	watch.pixel_size = resolved.pixel_size;
	watch.exact = resolved.exact;
	watch.status = resolved.status;
	watch.hits++;
	if (g_pixel_watch_hits.size() >= MAX_PIXEL_WATCH_HITS) {
		g_pixel_watch_hits.erase(g_pixel_watch_hits.begin(),
		                         g_pixel_watch_hits.begin() + MAX_PIXEL_WATCH_HITS / 4);
	}
	g_pixel_watch_hits.push_back({g_pixel_hit_sequence++, watch.id, watch.x, watch.y,
	                              resolved.pixel_address, resolved.pixel_size, resolved.exact,
	                              resolved.status, event});
}

uint64_t RangeEnd(uint64_t address, uint64_t size) {
	return size > UINT64_MAX - address ? UINT64_MAX : address + size;
}

bool RangesOverlap(uint64_t left_address, uint64_t left_size, uint64_t right_address,
                   uint64_t right_size) {
	if (left_size == 0 || right_size == 0) return false;
	return left_address < RangeEnd(right_address, right_size) &&
	       right_address < RangeEnd(left_address, left_size);
}

bool ContainsIgnoreCase(const std::string& text, const std::string& part) {
	if (part.empty()) return true;
	return std::search(text.begin(), text.end(), part.begin(), part.end(),
	                   [](char left, char right) {
		                   return std::tolower(static_cast<unsigned char>(left)) ==
		                          std::tolower(static_cast<unsigned char>(right));
	                   }) != text.end();
}

void TriggerCondition(BreakCondition& condition, const std::string& reason) {
	condition.hits++;
	condition.last_reason = reason;
	if (condition.one_shot) condition.enabled = false;
	if (g_pending_break_reason.empty()) g_pending_break_reason = reason;
}

bool ShaderValueMatches(uint64_t value, uint64_t address) {
	if (value == address) return true;
	const auto shader = g_shaders.find(value);
	return shader != g_shaders.end() && shader->second.summary.base_address == address;
}

void MatchDrawConditions(const DrawRecord& draw) {
	for (auto& condition: g_break_conditions) {
		if (!condition.enabled || condition.kind != BreakConditionKind::Shader) continue;
		const bool match = ShaderValueMatches(condition.value, draw.vs_address) ||
		                   ShaderValueMatches(condition.value, draw.ps_address) ||
		                   ShaderValueMatches(condition.value, draw.cs_address);
		if (!match) continue;
		TriggerCondition(condition,
		                 std::string("GPU shader condition matched ") + KindName(draw.kind) +
		                     " submit " + std::to_string(draw.submit_id));
	}
}

void MatchEventConditions(const ResourceEvent& event) {
	for (auto& condition: g_break_conditions) {
		if (!condition.enabled) continue;
		bool match = false;
		if (condition.kind == BreakConditionKind::Resource) {
			const bool address_match =
			    condition.value == 0 || RangesOverlap(condition.value, 1, event.address, event.size) ||
			    RangesOverlap(condition.value, 1, event.stencil_address, event.stencil_size) ||
			    RangesOverlap(condition.value, 1, event.metadata_address, event.metadata_size);
			match = address_match && ContainsIgnoreCase(event.action, condition.action);
		} else if (condition.kind == BreakConditionKind::PixelWatch) {
			const auto watch = std::find_if(g_pixel_watches.begin(), g_pixel_watches.end(),
			                                [&condition](const PixelWatch& value) {
				                                return value.id == condition.value;
			                                });
			match = watch != g_pixel_watches.end() && EventContainsWatch(event, *watch);
		}
		if (!match) continue;
		TriggerCondition(condition,
		                 std::string(BreakConditionKindName(condition.kind)) +
		                     " condition matched " + event.action + " at 0x" +
		                     [&event] {
			                     char text[24] {};
			                     std::snprintf(text, sizeof(text), "%llx",
			                                   static_cast<unsigned long long>(event.address));
			                     return std::string(text);
		                     }());
	}
}

void MatchNonFiniteConditions(uint64_t address, const PreviewDiagnostics& diagnostics) {
	if (diagnostics.non_finite_pixels == 0) return;
	for (auto& condition: g_break_conditions) {
		if (!condition.enabled || condition.kind != BreakConditionKind::NonFinite ||
		    (condition.value != 0 && condition.value != address))
			continue;
		TriggerCondition(condition,
		                 "non-finite preview detected " +
		                     std::to_string(diagnostics.non_finite_pixels) + " pixels");
	}
}

std::vector<ShaderSummary> Shaders() {
	const std::lock_guard lock(g_mutex);

	std::vector<ShaderSummary> out;
	out.reserve(g_shaders.size());
	for (const auto& [hash, entry]: g_shaders) {
		out.push_back(entry.summary);
	}

	std::sort(out.begin(), out.end(), [](const ShaderSummary& a, const ShaderSummary& b) {
		return a.sequence < b.sequence;
	});
	return out;
}

bool GetShaderCode(uint64_t hash, ShaderCode& out) {
	const std::lock_guard lock(g_mutex);

	const auto it = g_shaders.find(hash);
	if (it == g_shaders.end()) {
		return false;
	}

	if (it->second.spirv_text.empty() && !it->second.spirv.empty()) {
		Disassemble(it->second.spirv, it->second.spirv_text);
	}

	out.isa   = it->second.isa;
	out.ir    = it->second.ir;
	out.spirv = it->second.spirv_text;
	out.resources = it->second.resources;
	return true;
}

void RecordDraw(const DrawRecord& record) {
	if (!IsCapturing()) {
		return;
	}

	const std::lock_guard lock(g_mutex);

	if (record.kind == DrawKind::Dispatch) {
		g_total_dispatches++;
	} else {
		g_total_draws++;
	}

	if (g_current_frame.size() >= MAX_DRAWS_PER_FRAME) {
		g_current_truncated = true;
		return;
	}

	auto stored  = record;
	stored.frame = g_frame;
	stored.index = static_cast<uint32_t>(g_current_frame.size());
	g_current_frame.push_back(stored);
	MatchDrawConditions(stored);
}

void RecordSubmission(uint64_t submit_id, uint32_t queue_id, bool compute,
                      const uint32_t* commands, uint32_t command_words,
                      const uint32_t* constant_commands, uint32_t constant_words,
                      bool interrupt_on_done, bool reset_processor) {
	if (!IsCapturing() || submit_id == 0 || commands == nullptr || command_words == 0) return;

	SubmissionRecord record {};
	record.submit_id = submit_id;
	record.queue_id = queue_id;
	record.compute = compute;
	record.interrupt_on_done = interrupt_on_done;
	record.reset_processor = reset_processor;
	const size_t requested = static_cast<size_t>(command_words) + constant_words;
	record.truncated = requested > MAX_SUBMISSION_DWORDS;
	const size_t draw_words = std::min<size_t>(command_words, MAX_SUBMISSION_DWORDS);
	record.commands.assign(commands, commands + draw_words);
	const size_t remaining = MAX_SUBMISSION_DWORDS - draw_words;
	const size_t ce_words = std::min<size_t>(constant_words, remaining);
	if (ce_words != 0 && constant_commands != nullptr) {
		record.constant_commands.assign(constant_commands, constant_commands + ce_words);
	}
	if (ce_words != constant_words || (ce_words != 0 && constant_commands == nullptr))
		record.truncated = true;

	const std::lock_guard lock(g_mutex);
	record.frame = g_frame;
	const size_t stored_words = record.commands.size() + record.constant_commands.size();
	while (!g_submissions.empty() &&
	       (g_submissions.size() >= MAX_SUBMISSION_RECORDS ||
	        g_submission_dwords + stored_words > MAX_SUBMISSION_TOTAL_DWORDS)) {
		g_submission_dwords -= g_submissions.front().commands.size() +
		                       g_submissions.front().constant_commands.size();
		g_submissions.pop_front();
	}
	if (stored_words > MAX_SUBMISSION_TOTAL_DWORDS) return;
	g_submission_dwords += stored_words;
	g_submissions.push_back(std::move(record));
}

bool GetSubmission(uint64_t submit_id, SubmissionRecord& out) {
	const std::lock_guard lock(g_mutex);
	const auto it = std::find_if(g_submissions.rbegin(), g_submissions.rend(),
	                             [submit_id](const SubmissionRecord& value) {
		                             return value.submit_id == submit_id;
	                             });
	if (it == g_submissions.rend()) return false;
	out = *it;
	return true;
}

void RecordFlip() {
	if (!IsCapturing()) {
		return;
	}

	const std::lock_guard lock(g_mutex);

	g_total_flips++;
	g_frame++;

	g_draws_last_frame      = 0;
	g_dispatches_last_frame = 0;
	for (const auto& draw: g_current_frame) {
		if (draw.kind == DrawKind::Dispatch) {
			g_dispatches_last_frame++;
		} else {
			g_draws_last_frame++;
		}
	}

	// Only swap in a frame that had something in it, so a flip with no draws does not blank a
	// list somebody is reading.
	if (!g_current_frame.empty()) {
		g_last_frame.swap(g_current_frame);
		g_last_truncated = g_current_truncated;

		if (!g_logged_first_frame) {
			g_logged_first_frame = true;
			LOGF("Debugger: first captured frame has %u draws and %u dispatches\n",
			     g_draws_last_frame, g_dispatches_last_frame);
		}
	}

	g_current_frame.clear();
	g_current_truncated = false;
}

std::vector<DrawRecord> LastFrame() {
	const std::lock_guard lock(g_mutex);
	return g_last_frame;
}

Stats GetStats() {
	const std::lock_guard lock(g_mutex);

	Stats stats {};
	stats.frame                 = g_frame;
	stats.draws_last_frame      = g_draws_last_frame;
	stats.dispatches_last_frame = g_dispatches_last_frame;
	stats.draws_this_frame      = static_cast<uint32_t>(g_current_frame.size());
	stats.total_draws           = g_total_draws;
	stats.total_dispatches      = g_total_dispatches;
	stats.total_flips           = g_total_flips;
	stats.shader_count          = static_cast<uint32_t>(g_shaders.size());
	stats.truncated             = g_last_truncated;
	return stats;
}

void RecordImageEvent(ResourceEvent event) {
	if (!IsCapturing() || event.action.empty()) return;
	const std::lock_guard lock(g_mutex);
	if (g_resource_events.size() >= MAX_RESOURCE_EVENTS) {
		g_resource_events.erase(g_resource_events.begin(),
		                        g_resource_events.begin() + MAX_RESOURCE_EVENTS / 4);
	}
	event.sequence = g_resource_sequence++;
	event.frame = g_frame;
	event.timestamp_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	    std::chrono::steady_clock::now() - g_resource_start).count());
	if (!g_current_frame.empty()) {
		const auto& command = g_current_frame.back();
		event.command_index = command.index;
		event.submit_id = command.submit_id;
		event.command_kind = command.kind;
		event.vs_address = command.vs_address;
		event.ps_address = command.ps_address;
		event.cs_address = command.cs_address;
	}
	for (auto& watch: g_pixel_watches) AppendPixelHit(watch, event);
	MatchEventConditions(event);
	g_resource_events.push_back(std::move(event));
}

void RecordImageEvent(const char* action, uint64_t address, uint64_t size, uint32_t width,
                      uint32_t height, uint32_t depth, uint32_t bytes_per_block,
                      uint32_t guest_format, uint32_t host_format, uint32_t tile_mode) {
	if (action == nullptr) return;
	ResourceEvent event {};
	event.action          = action;
	event.address         = address;
	event.size            = size;
	event.width           = width;
	event.height          = height;
	event.depth           = depth;
	event.bytes_per_block = bytes_per_block;
	event.guest_format    = guest_format;
	event.host_format     = host_format;
	event.tile_mode       = tile_mode;
	RecordImageEvent(std::move(event));
}

std::vector<ResourceEvent> ResourceHistory(uint64_t address, size_t limit) {
	const std::lock_guard lock(g_mutex);
	std::vector<ResourceEvent> out;
	limit = std::min(limit, g_resource_events.size());
	for (auto it = g_resource_events.rbegin(); it != g_resource_events.rend() && out.size() < limit;
	     ++it) {
		if (address == 0 || (address >= it->address && address < it->address + it->size)) {
			out.push_back(*it);
		}
	}
	std::reverse(out.begin(), out.end());
	return out;
}

std::vector<ResourceEvent> ResourceTrace(uint64_t address, uint64_t size, size_t limit) {
	const std::lock_guard lock(g_mutex);
	std::vector<ResourceEvent> out;
	if (size == 0) size = 1;
	limit = std::min(limit, g_resource_events.size());
	for (auto it = g_resource_events.rbegin(); it != g_resource_events.rend() && out.size() < limit;
	     ++it) {
		const bool data_overlap = RangesOverlap(address, size, it->address, it->size);
		const bool stencil_overlap = RangesOverlap(address, size, it->stencil_address,
		                                           it->stencil_size);
		const bool metadata_overlap = RangesOverlap(address, size, it->metadata_address,
		                                            it->metadata_size);
		if (data_overlap || stencil_overlap || metadata_overlap) out.push_back(*it);
	}
	std::reverse(out.begin(), out.end());
	return out;
}

std::vector<ResourceEvent> ResourceAliases(uint64_t address, uint64_t size) {
	const std::lock_guard lock(g_mutex);
	if (size == 0) size = 1;
	std::unordered_map<uint64_t, ResourceEvent> latest;
	for (const auto& event: g_resource_events) {
		if (event.image_index == UINT32_MAX) continue;
		const bool data_overlap = RangesOverlap(address, size, event.address, event.size);
		const bool stencil_overlap = RangesOverlap(address, size, event.stencil_address,
		                                           event.stencil_size);
		const bool metadata_overlap = RangesOverlap(address, size, event.metadata_address,
		                                            event.metadata_size);
		if (!data_overlap && !stencil_overlap && !metadata_overlap) continue;
		const auto key = (static_cast<uint64_t>(event.image_generation) << 32u) |
		                 event.image_index;
		latest[key] = event;
	}
	std::vector<ResourceEvent> out;
	out.reserve(latest.size());
	for (auto& [key, event]: latest) {
		(void)key;
		out.push_back(std::move(event));
	}
	std::sort(out.begin(), out.end(), [](const ResourceEvent& left, const ResourceEvent& right) {
		if (left.active != right.active) return left.active > right.active;
		if (left.address != right.address) return left.address < right.address;
		return left.sequence > right.sequence;
	});
	return out;
}

uint64_t AddPixelWatch(uint64_t address, uint32_t x, uint32_t y) {
	if (!IsCapturing() || address == 0) return 0;
	const std::lock_guard lock(g_mutex);
	if (g_pixel_watches.size() >= MAX_PIXEL_WATCHES) return 0;
	PixelWatch watch {.id = g_pixel_watch_sequence++, .address = address, .x = x, .y = y,
	                  .status = "waiting for a matching image event"};
	for (const auto& event: g_resource_events) AppendPixelHit(watch, event);
	const auto id = watch.id;
	g_pixel_watches.push_back(std::move(watch));
	return id;
}

bool RemovePixelWatch(uint64_t id) {
	const std::lock_guard lock(g_mutex);
	const auto it = std::find_if(g_pixel_watches.begin(), g_pixel_watches.end(),
	                             [id](const PixelWatch& watch) { return watch.id == id; });
	if (it == g_pixel_watches.end()) return false;
	g_pixel_watches.erase(it);
	g_pixel_watch_hits.erase(
	    std::remove_if(g_pixel_watch_hits.begin(), g_pixel_watch_hits.end(),
	                   [id](const PixelWatchHit& hit) { return hit.watch_id == id; }),
	    g_pixel_watch_hits.end());
	return true;
}

void ClearPixelWatches() {
	const std::lock_guard lock(g_mutex);
	g_pixel_watches.clear();
	g_pixel_watch_hits.clear();
}

std::vector<PixelWatch> PixelWatches() {
	const std::lock_guard lock(g_mutex);
	return g_pixel_watches;
}

std::vector<PixelWatchHit> PixelWatchHits(size_t limit) {
	const std::lock_guard lock(g_mutex);
	limit = std::min(limit, g_pixel_watch_hits.size());
	return {g_pixel_watch_hits.end() - static_cast<std::ptrdiff_t>(limit),
	        g_pixel_watch_hits.end()};
}

uint64_t AddBreakCondition(BreakConditionKind kind, uint64_t value, const std::string& action,
	                       bool one_shot) {
	if (!IsCapturing()) return 0;
	const std::lock_guard lock(g_mutex);
	if (g_break_conditions.size() >= MAX_BREAK_CONDITIONS) return 0;
	const auto id = g_break_condition_sequence++;
	g_break_conditions.push_back({id, kind, value, action, true, one_shot, 0, {}});
	return id;
}

bool RemoveBreakCondition(uint64_t id) {
	const std::lock_guard lock(g_mutex);
	const auto before = g_break_conditions.size();
	g_break_conditions.erase(
	    std::remove_if(g_break_conditions.begin(), g_break_conditions.end(),
	                   [id](const BreakCondition& condition) { return condition.id == id; }),
	    g_break_conditions.end());
	return g_break_conditions.size() != before;
}

void ClearBreakConditions() {
	const std::lock_guard lock(g_mutex);
	g_break_conditions.clear();
	g_pending_break_reason.clear();
}

std::vector<BreakCondition> BreakConditions() {
	const std::lock_guard lock(g_mutex);
	return g_break_conditions;
}

bool TakeBreakRequest(std::string& reason) {
	const std::lock_guard lock(g_mutex);
	if (g_pending_break_reason.empty()) return false;
	reason = std::move(g_pending_break_reason);
	g_pending_break_reason.clear();
	return true;
}

uint64_t RequestResourcePreview(uint64_t address, uint64_t size, bool refresh) {
	if (!IsCapturing() || address == 0) return 0;
	const std::lock_guard lock(g_mutex);
	if (!refresh && g_preview.address == address && g_preview.size == size &&
	    !g_preview.status.empty()) {
		return g_preview.request_id;
	}
	const auto id = ++g_preview_sequence;
	g_preview = {.request_id = id, .address = address, .size = size, .status = "pending"};
	g_preview_request = {id, address, size};
	return id;
}

bool TakePreviewRequest(PreviewRequest& out) {
	const std::lock_guard lock(g_mutex);
	if (g_preview_request.id == 0) return false;
	out = g_preview_request;
	g_preview_request = {};
	return true;
}

void PublishResourcePreview(uint64_t request_id, uint64_t address, uint32_t source_width,
                            uint32_t source_height, uint32_t host_format, uint32_t width,
                            uint32_t height, std::vector<uint8_t> rgba,
	                        const std::string& error, PreviewDiagnostics diagnostics) {
	const std::lock_guard lock(g_mutex);
	if (g_preview.request_id != request_id || g_preview.address != address) return;
	g_preview.source_width  = source_width;
	g_preview.source_height = source_height;
	g_preview.host_format   = host_format;
	g_preview.content_hash = diagnostics.content_hash;
	g_preview.total_pixels = diagnostics.total_pixels;
	g_preview.zero_pixels = diagnostics.zero_pixels;
	g_preview.non_finite_pixels = diagnostics.non_finite_pixels;
	g_preview.non_finite_components = diagnostics.non_finite_components;
	MatchNonFiniteConditions(address, diagnostics);
	g_preview.width         = width;
	g_preview.height        = height;
	g_preview.rgba          = std::move(rgba);
	g_preview.error         = error;
	g_preview.status        = error.empty() ? "ready" : "error";
}

ResourcePreview GetResourcePreview(uint64_t address, uint64_t size) {
	const std::lock_guard lock(g_mutex);
	if (g_preview.address != address || g_preview.size != size) return {};
	return g_preview;
}

bool DumpShader(uint64_t hash, std::string& path_out) {
	ShaderCode code {};
	if (!GetShaderCode(hash, code)) {
		return false;
	}

	const auto folder = Config::GetShaderLogFolder() / "debugger";
	Common::File::CreateDirectories(folder);

	char name[64] {};
	std::snprintf(name, sizeof(name), "%016llx", static_cast<unsigned long long>(hash));

	const auto write = [&folder, &name](const char* extension, const std::string& text) {
		if (text.empty()) {
			return;
		}
		auto path = folder / (std::string(name) + extension);

		Common::File file;
		file.Create(path);
		if (!file.IsInvalid()) {
			file.Printf("%s", text.c_str());
			file.Close();
		}
	};

	write(".rdna2", code.isa);
	write(".ir", code.ir);
	write(".spvasm", code.spirv);

	path_out = folder.string();
	return true;
}

void Reset() {
	const std::lock_guard lock(g_mutex);

	g_shaders.clear();
	g_shader_sequence = 0;
	g_current_frame.clear();
	g_last_frame.clear();
	g_submissions.clear();
	g_submission_dwords = 0;
	g_current_truncated     = false;
	g_last_truncated        = false;
	g_frame                 = 0;
	g_draws_last_frame      = 0;
	g_dispatches_last_frame = 0;
	g_total_draws           = 0;
	g_total_dispatches      = 0;
	g_total_flips           = 0;
	g_resource_events.clear();
	g_resource_sequence = 0;
	g_pixel_watches.clear();
	g_pixel_watch_hits.clear();
	g_pixel_watch_sequence = 1;
	g_pixel_hit_sequence = 1;
	g_break_conditions.clear();
	g_break_condition_sequence = 1;
	g_pending_break_reason.clear();
	g_preview_request = {};
	g_preview = {};
	g_preview_sequence = 0;
	g_logged_first_shader   = false;
	g_logged_first_frame    = false;
}

} // namespace Debugger::Graphics
