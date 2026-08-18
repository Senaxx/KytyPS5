#include "debugger/remote/server.h"

#include "common/logging/log.h"
#include "debugger/core/session.h"
#include "debugger/symbols/symbols.h"
#include "debugger/target/graphics.h"
#include "debugger/target/io.h"
#include "graphics/presentation/renderDoc.h"
#include "kernel/pthread.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <thread>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#endif

namespace Debugger::Remote {
namespace {

using Json = nlohmann::json;

constexpr uint32_t PROTOCOL_VERSION = 1;

std::atomic_bool      g_running = false;
std::thread           g_thread;
std::filesystem::path g_descriptor_path;
std::string           g_endpoint;
std::string           g_token;

const char* StopReasonName(StopReason reason) {
	switch (reason) {
		case StopReason::Breakpoint: return "breakpoint";
		case StopReason::Step: return "step";
		case StopReason::Pause: return "pause";
		case StopReason::Entry: return "entry";
		case StopReason::Fatal: return "fatal";
		default: return "none";
	}
}

std::string MakeToken() {
	std::random_device random;
	std::ostringstream out;
	out << std::hex << std::setfill('0');
	for (int i = 0; i < 4; i++) out << std::setw(8) << random();
	return out.str();
}

std::filesystem::path SessionDirectory() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (const auto* local = std::getenv("LOCALAPPDATA"); local != nullptr && *local != '\0') {
		return std::filesystem::path(local) / "KytyPS5" / "DebuggerSessions";
	}
#endif
	return std::filesystem::temp_directory_path() / "KytyPS5" / "DebuggerSessions";
}

Json Error(const std::string& message) {
	return {{"ok", false}, {"error", message}, {"protocol", PROTOCOL_VERSION}};
}

bool RequestU64(const Json& request, const char* key, uint64_t& value) {
	const auto it = request.find(key);
	if (it == request.end()) return false;
	if (it->is_number_unsigned()) {
		value = it->get<uint64_t>();
		return true;
	}
	if (it->is_number_integer()) {
		const auto signed_value = it->get<int64_t>();
		if (signed_value < 0) return false;
		value = static_cast<uint64_t>(signed_value);
		return true;
	}
	if (!it->is_string()) return false;
	auto text = it->get<std::string>();
	int  base = 10;
	if (text.starts_with("0x") || text.starts_with("0X")) {
		text.erase(0, 2);
		base = 16;
	}
	const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value, base);
	return ec == std::errc() && end == text.data() + text.size();
}

std::string HexBytes(const uint8_t* bytes, size_t size) {
	static constexpr char DIGITS[] = "0123456789abcdef";
	std::string out(size * 2, '0');
	for (size_t i = 0; i < size; i++) {
		out[i * 2]     = DIGITS[bytes[i] >> 4u];
		out[i * 2 + 1] = DIGITS[bytes[i] & 0x0fu];
	}
	return out;
}

std::string Base64(const std::vector<uint8_t>& bytes) {
	static constexpr char TABLE[] =
	    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	std::string out;
	out.reserve((bytes.size() + 2) / 3 * 4);
	for (size_t i = 0; i < bytes.size(); i += 3) {
		const uint32_t value = static_cast<uint32_t>(bytes[i]) << 16u |
		                       (i + 1 < bytes.size() ? static_cast<uint32_t>(bytes[i + 1]) << 8u : 0) |
		                       (i + 2 < bytes.size() ? bytes[i + 2] : 0);
		out.push_back(TABLE[(value >> 18u) & 63u]);
		out.push_back(TABLE[(value >> 12u) & 63u]);
		out.push_back(i + 1 < bytes.size() ? TABLE[(value >> 6u) & 63u] : '=');
		out.push_back(i + 2 < bytes.size() ? TABLE[value & 63u] : '=');
	}
	return out;
}

bool ParseHexBytes(const std::string& text, std::vector<uint8_t>& out) {
	if (text.size() % 2 != 0 || text.size() > 8192) return false;
	const auto digit = [](char c) -> int {
		if (c >= '0' && c <= '9') return c - '0';
		if (c >= 'a' && c <= 'f') return c - 'a' + 10;
		if (c >= 'A' && c <= 'F') return c - 'A' + 10;
		return -1;
	};
	out.resize(text.size() / 2);
	for (size_t i = 0; i < out.size(); i++) {
		const int hi = digit(text[i * 2]);
		const int lo = digit(text[i * 2 + 1]);
		if (hi < 0 || lo < 0) return false;
		out[i] = static_cast<uint8_t>((hi << 4) | lo);
	}
	return true;
}

Json RegistersJson(const Registers& regs) {
	return {{"valid", regs.valid}, {"rax", regs.rax}, {"rbx", regs.rbx}, {"rcx", regs.rcx},
	        {"rdx", regs.rdx}, {"rsi", regs.rsi}, {"rdi", regs.rdi}, {"rbp", regs.rbp},
	        {"rsp", regs.rsp}, {"r8", regs.r8}, {"r9", regs.r9}, {"r10", regs.r10},
	        {"r11", regs.r11}, {"r12", regs.r12}, {"r13", regs.r13}, {"r14", regs.r14},
	        {"r15", regs.r15}, {"rip", regs.rip}, {"rflags", regs.rflags}};
}

Json Summary() {
	const auto stats = Graphics::GetStats();
	return {{"ok", true},
	        {"protocol", PROTOCOL_VERSION},
	        {"paused", Session::IsPaused()},
	        {"graphics",
	         {{"frame", stats.frame},
	          {"draws_last_frame", stats.draws_last_frame},
	          {"dispatches_last_frame", stats.dispatches_last_frame},
	          {"total_draws", stats.total_draws},
	          {"total_dispatches", stats.total_dispatches},
	          {"shader_count", stats.shader_count},
	          {"truncated", stats.truncated}}}};
}

Json Threads() {
	std::vector<Libs::LibKernel::GuestThreadInfo> threads;
	Libs::LibKernel::PthreadEnumerate(threads);
	const auto stopped = Session::Stopped();

	Json values = Json::array();
	for (const auto& thread: threads) {
		Json value = {{"id", thread.unique_id},
		              {"guest_id", thread.guest_thread_id},
		              {"host_id", thread.host_thread_id},
		              {"name", thread.name},
		              {"alive", thread.alive},
		              {"stack_address", thread.stack_addr},
		              {"stack_size", thread.stack_size},
		              {"stopped", false}};
		for (const auto& stop: stopped) {
			if (stop.unique_id == thread.unique_id) {
				value["stopped"]     = true;
				value["reason"]      = StopReasonName(stop.reason);
				value["address"]     = stop.address;
				value["description"] = Symbols::Format(stop.address);
				break;
			}
		}
		values.push_back(std::move(value));
	}
	return {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"threads", std::move(values)}};
}

Json Modules() {
	Json values = Json::array();
	for (const auto& module: Symbols::Modules()) {
		values.push_back({{"id", module.id}, {"name", module.name},
		                  {"base", module.base_vaddr}, {"size", module.size}});
	}
	return {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"modules", std::move(values)}};
}

Json Shaders() {
	Json values = Json::array();
	for (const auto& shader: Graphics::Shaders()) {
		values.push_back({{"hash", shader.hash}, {"base", shader.base_address},
		                  {"stage", Graphics::StageName(shader.stage)},
		                  {"gcn_bytes", shader.gcn_bytes}, {"spirv_words", shader.spirv_words},
		                  {"resource_count", shader.resource_count},
		                  {"sequence", shader.sequence}});
	}
	return {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"shaders", std::move(values)}};
}

Json Frame() {
	Json values = Json::array();
	for (const auto& draw: Graphics::LastFrame()) {
		values.push_back({{"frame", draw.frame}, {"index", draw.index}, {"submit", draw.submit_id},
		                  {"kind", Graphics::KindName(draw.kind)}, {"count", draw.count},
		                  {"instances", draw.instances},
		                  {"groups", {draw.groups[0], draw.groups[1], draw.groups[2]}},
		                  {"vs", draw.vs_address}, {"ps", draw.ps_address}, {"cs", draw.cs_address}});
	}
	return {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"commands", std::move(values)}};
}

Json Symbols(const Json& request) {
	const auto filter = request.value("filter", std::string());
	const auto module = request.value("module", std::string());
	const auto limit = std::clamp(request.value("limit", 400u), 1u, 2000u);
	Json values = Json::array();
	for (const auto& symbol: Debugger::Symbols::Search(filter, limit, module)) {
		values.push_back({{"name", symbol.name}, {"module", symbol.module},
		                  {"address", symbol.address}});
	}
	return {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"symbols", std::move(values)}};
}

Json ShaderDetails(const Json& request) {
	uint64_t hash = 0;
	if (!RequestU64(request, "hash", hash)) return Error("invalid shader hash");
	Graphics::ShaderCode code;
	if (!Graphics::GetShaderCode(hash, code)) return Error("shader not found");
	Json resources = Json::array();
	for (const auto& resource: code.resources) {
		resources.push_back({{"kind", resource.kind}, {"index", resource.index},
		                     {"source", resource.source}, {"first_use_pc", resource.first_use_pc},
		                     {"read", resource.read}, {"written", resource.written},
		                     {"atomic", resource.atomic}, {"descriptor", resource.descriptor},
		                     {"address", resource.address}, {"size", resource.size},
		                     {"width", resource.width}, {"height", resource.height},
		                     {"depth", resource.depth}, {"format", resource.format},
		                     {"tile", resource.tile}});
	}
	return {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"hash", hash},
	        {"isa", code.isa}, {"ir", code.ir}, {"spirv", code.spirv},
	        {"resources", std::move(resources)}};
}

Json DumpShader(const Json& request) {
	uint64_t hash = 0;
	if (!RequestU64(request, "hash", hash)) return Error("invalid shader hash");
	std::string path;
	if (!Graphics::DumpShader(hash, path)) return Error("shader dump failed");
	return {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"path", path}};
}

Json StoppedDetails(const Json& request) {
	const int id = request.value("thread", 0);
	const auto stopped = Session::Stopped();
	const auto it = std::find_if(stopped.begin(), stopped.end(),
	                             [id](const StoppedThread& value) { return value.unique_id == id; });
	if (it == stopped.end()) return Error("thread is not stopped");
	Json frames = Json::array();
	for (const auto& frame: Session::Backtrace(id)) {
		frames.push_back({{"address", frame.address}, {"frame", frame.frame},
		                  {"description", frame.description}});
	}
	Json instructions = Json::array();
	for (const auto& instruction: Disasm::Decode(it->address, 128)) {
		instructions.push_back({{"address", instruction.address}, {"length", instruction.length},
		                        {"bytes", instruction.bytes}, {"text", instruction.text},
		                        {"symbol", instruction.symbol}});
	}
	return {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"registers", RegistersJson(it->regs)},
	        {"frames", std::move(frames)}, {"instructions", std::move(instructions)}};
}

Json Breakpoints() {
	Json values = Json::array();
	for (const auto& breakpoint: Session::Breakpoints()) {
		values.push_back({{"id", breakpoint.id}, {"address", breakpoint.address},
		                  {"label", breakpoint.label}, {"enabled", breakpoint.enabled},
		                  {"armed", breakpoint.armed}, {"pending", breakpoint.pending},
		                  {"one_shot", breakpoint.one_shot}, {"hits", breakpoint.hit_count}});
	}
	return {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"breakpoints", std::move(values)}};
}

Json MemoryRead(const Json& request) {
	uint64_t address = 0;
	if (!RequestU64(request, "address", address)) return Error("invalid memory address");
	const auto size = std::clamp(request.value("size", 256u), 1u, 4096u);
	std::vector<uint8_t> bytes(size);
	if (!Session::ReadMemory(address, bytes.data(), bytes.size())) return Error("memory is unreadable");
	return {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"address", address},
	        {"bytes", HexBytes(bytes.data(), bytes.size())}};
}

Json ResourceEventJson(const Graphics::ResourceEvent& event) {
	return {{"sequence", event.sequence}, {"timestamp_us", event.timestamp_us},
	        {"frame", event.frame}, {"has_command", event.command_index != UINT32_MAX},
	        {"command_index", event.command_index}, {"submit", event.submit_id},
	        {"command_kind", Graphics::KindName(event.command_kind)},
	        {"vs", event.vs_address}, {"ps", event.ps_address}, {"cs", event.cs_address},
	        {"action", event.action}, {"note", event.note},
	        {"has_image", event.image_index != UINT32_MAX}, {"image_index", event.image_index},
	        {"image_generation", event.image_generation}, {"host_image", event.host_image},
	        {"active", event.active}, {"address", event.address}, {"size", event.size},
	        {"stencil_address", event.stencil_address}, {"stencil_size", event.stencil_size},
	        {"metadata_address", event.metadata_address}, {"metadata_size", event.metadata_size},
	        {"width", event.width}, {"height", event.height}, {"depth", event.depth},
	        {"pitch", event.pitch}, {"bpb", event.bytes_per_block},
	        {"guest_format", event.guest_format}, {"host_format", event.host_format},
	        {"tile", event.tile_mode}, {"image_type", event.image_type},
	        {"samples", event.samples}, {"levels", event.levels}, {"layers", event.layers},
	        {"metadata_kind", event.metadata_kind}, {"registered", event.registered},
	        {"cpu_dirty", event.cpu_dirty}, {"maybe_cpu_dirty", event.maybe_cpu_dirty},
	        {"buffer_modified", event.buffer_modified}, {"gpu_modified", event.gpu_modified},
	        {"usage_texture", event.usage_texture}, {"usage_storage", event.usage_storage},
	        {"usage_render_target", event.usage_render_target},
	        {"usage_depth_target", event.usage_depth_target},
	        {"usage_video_out", event.usage_video_out}, {"bound", event.bound},
	        {"target", event.target}, {"needs_rebind", event.needs_rebind},
	        {"force_general", event.force_general}, {"shader_write", event.shader_write}};
}

bool WriteWords(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	if (!out) return false;
	if (!words.empty()) {
		out.write(reinterpret_cast<const char*>(words.data()),
		          static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
	}
	return out.good();
}

Json CaptureCommand(const Json& request) {
	uint64_t submit = 0;
	if (!RequestU64(request, "submit", submit) || submit == 0)
		return Error("invalid GPU submit id");
	Graphics::SubmissionRecord submission;
	if (!Graphics::GetSubmission(submit, submission))
		return Error("submission is no longer in the debugger capture ring");

	Json commands = Json::array();
	std::vector<uint64_t> shader_addresses;
	for (const auto& draw: Graphics::LastFrame()) {
		if (draw.submit_id != submit) continue;
		commands.push_back({{"frame", draw.frame}, {"index", draw.index},
		                    {"kind", Graphics::KindName(draw.kind)}, {"count", draw.count},
		                    {"instances", draw.instances},
		                    {"groups", {draw.groups[0], draw.groups[1], draw.groups[2]}},
		                    {"vs", draw.vs_address}, {"ps", draw.ps_address},
		                    {"cs", draw.cs_address}});
		for (const auto address: {draw.vs_address, draw.ps_address, draw.cs_address})
			if (address != 0 && std::find(shader_addresses.begin(), shader_addresses.end(), address) ==
			                        shader_addresses.end())
				shader_addresses.push_back(address);
	}

	Json shaders = Json::array();
	for (const auto& shader: Graphics::Shaders()) {
		if (std::find(shader_addresses.begin(), shader_addresses.end(), shader.base_address) ==
		    shader_addresses.end())
			continue;
		shaders.push_back({{"stage", Graphics::StageName(shader.stage)}, {"hash", shader.hash},
		                   {"base", shader.base_address}, {"gcn_bytes", shader.gcn_bytes},
		                   {"spirv_words", shader.spirv_words},
		                   {"resource_count", shader.resource_count}});
	}

	Json resources = Json::array();
	for (const auto& event: Graphics::ResourceHistory(0, 32768))
		if (event.submit_id == submit) resources.push_back(ResourceEventJson(event));

	const auto root = std::filesystem::current_path() / "_CommandReplays";
	std::ostringstream name;
	name << "frame-" << submission.frame << "-submit-" << submit;
	const auto folder = root / name.str();
	std::error_code ec;
	std::filesystem::create_directories(folder, ec);
	if (ec) return Error("could not create command-capture directory: " + ec.message());
	if (!WriteWords(folder / "commands.pm4", submission.commands) ||
	    !WriteWords(folder / "constant.pm4", submission.constant_commands))
		return Error("could not write command stream");

	Json missing = Json::array({"indirect_pm4_buffers", "resource_pre_contents",
	                           "resource_post_contents", "descriptor_address_remapping"});
	Json manifest = {
	    {"schema", "kyty_gpu_command_capture"},
	    {"version", 1},
	    {"capture_kind", "top_level_pm4"},
	    {"submit", submit},
	    {"frame", submission.frame},
	    {"queue", submission.queue_id},
	    {"queue_kind", submission.compute ? "compute" : "graphics"},
	    {"interrupt_on_done", submission.interrupt_on_done},
	    {"reset_processor", submission.reset_processor},
	    {"command_words", submission.commands.size()},
	    {"constant_words", submission.constant_commands.size()},
	    {"top_level_stream_exact", !submission.truncated},
	    {"commands", std::move(commands)},
	    {"shaders", std::move(shaders)},
	    {"resource_events", std::move(resources)},
	    {"replay", {{"executable", false},
	                {"status", submission.truncated ? "top-level PM4 was truncated"
	                                                : "capture complete; resource snapshots required"},
	                {"missing", missing}}}};
	std::ofstream manifest_file(folder / "manifest.json", std::ios::binary | std::ios::trunc);
	if (!manifest_file) return Error("could not write command-capture manifest");
	manifest_file << manifest.dump(2) << '\n';
	if (!manifest_file.good()) return Error("could not finish command-capture manifest");

	return {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"path", folder.string()},
	        {"submit", submit}, {"command_words", submission.commands.size()},
	        {"constant_words", submission.constant_commands.size()},
	        {"executable", false},
	        {"status", manifest["replay"]["status"]}, {"missing", std::move(missing)}};
}

Json ResourceHistory(const Json& request) {
	uint64_t address = 0;
	if (request.contains("address") && !RequestU64(request, "address", address))
		return Error("invalid resource address");
	const auto limit = std::clamp(request.value("limit", 512u), 1u, 2048u);
	Json values = Json::array();
	for (const auto& event: Graphics::ResourceHistory(address, limit)) {
		values.push_back({{"sequence", event.sequence}, {"frame", event.frame},
		                  {"action", event.action}, {"address", event.address}, {"size", event.size},
		                  {"width", event.width}, {"height", event.height}, {"depth", event.depth},
		                  {"bpb", event.bytes_per_block}, {"guest_format", event.guest_format},
		                  {"host_format", event.host_format}, {"tile", event.tile_mode}});
	}
	return {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"events", std::move(values)}};
}

Json ResourceTrace(const Json& request) {
	uint64_t address = 0;
	uint64_t size = 1;
	if (!RequestU64(request, "address", address) || address == 0)
		return Error("invalid resource trace address");
	if (request.contains("size") && !RequestU64(request, "size", size))
		return Error("invalid resource trace size");
	if (size == 0) size = 1;
	const auto limit = std::clamp(request.value("limit", 4096u), 1u, 8192u);
	Json events = Json::array();
	for (const auto& event: Graphics::ResourceTrace(address, size, limit))
		events.push_back(ResourceEventJson(event));
	Json aliases = Json::array();
	for (const auto& alias: Graphics::ResourceAliases(address, size))
		aliases.push_back(ResourceEventJson(alias));
	return {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"address", address}, {"size", size},
	        {"events", std::move(events)}, {"aliases", std::move(aliases)}};
}

Json PixelWatches() {
	Json watches = Json::array();
	for (const auto& watch: Graphics::PixelWatches()) {
		watches.push_back({{"id", watch.id}, {"address", watch.address}, {"x", watch.x},
		                   {"y", watch.y}, {"pixel_address", watch.pixel_address},
		                   {"pixel_size", watch.pixel_size}, {"exact", watch.exact},
		                   {"hits", watch.hits}, {"status", watch.status}});
	}
	Json hits = Json::array();
	for (const auto& hit: Graphics::PixelWatchHits()) {
		hits.push_back({{"sequence", hit.sequence}, {"watch_id", hit.watch_id}, {"x", hit.x},
		                {"y", hit.y}, {"pixel_address", hit.pixel_address},
		                {"pixel_size", hit.pixel_size}, {"exact", hit.exact},
		                {"status", hit.status}, {"event", ResourceEventJson(hit.event)}});
	}
	return {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"watches", std::move(watches)},
	        {"hits", std::move(hits)}};
}

Json AddPixelWatch(const Json& request) {
	uint64_t address = 0;
	if (!RequestU64(request, "address", address) || address == 0)
		return Error("invalid pixel-watch image address");
	const auto x_value = request.value("x", -1ll);
	const auto y_value = request.value("y", -1ll);
	if (x_value < 0 || y_value < 0 || x_value > UINT32_MAX || y_value > UINT32_MAX)
		return Error("invalid pixel coordinate");
	const auto id = Graphics::AddPixelWatch(address, static_cast<uint32_t>(x_value),
	                                        static_cast<uint32_t>(y_value));
	if (id == 0) return Error("pixel watchpoint limit reached or debugger capture is disabled");
	auto result = PixelWatches();
	result["added"] = id;
	return result;
}

Json GpuBreakConditions() {
	Json conditions = Json::array();
	for (const auto& condition: Graphics::BreakConditions()) {
		conditions.push_back({{"id", condition.id},
		                      {"kind", Graphics::BreakConditionKindName(condition.kind)},
		                      {"value", condition.value}, {"action", condition.action},
		                      {"enabled", condition.enabled}, {"one_shot", condition.one_shot},
		                      {"hits", condition.hits}, {"last_reason", condition.last_reason}});
	}
	return {{"ok", true}, {"protocol", PROTOCOL_VERSION},
	        {"conditions", std::move(conditions)}};
}

Json AddGpuBreakCondition(const Json& request) {
	const auto kind_name = request.value("kind", std::string());
	Graphics::BreakConditionKind kind {};
	if (!Graphics::ParseBreakConditionKind(kind_name, kind))
		return Error("invalid GPU break-condition kind");
	uint64_t value = 0;
	if (!RequestU64(request, "value", value)) return Error("invalid GPU break-condition value");
	if (kind != Graphics::BreakConditionKind::NonFinite && value == 0)
		return Error("this GPU break-condition requires a nonzero match value");
	const auto id = Graphics::AddBreakCondition(kind, value,
	                                             request.value("action", std::string()),
	                                             request.value("one_shot", false));
	if (id == 0) return Error("GPU break-condition limit reached or debugger capture is disabled");
	auto result = GpuBreakConditions();
	result["added"] = id;
	return result;
}

Json RenderDocStatus(bool accepted = false, bool include_accepted = false) {
	const auto status = Libs::Graphics::RenderDocGetStatus();
	Json result = {{"ok", true}, {"protocol", PROTOCOL_VERSION},
	               {"available", status.available},
	               {"state", Libs::Graphics::RenderDocCaptureStateName(status.state)},
	               {"capture_path", status.capture_path},
	               {"completed_captures", status.completed_captures},
	               {"has_result", status.has_result},
	               {"last_succeeded", status.last_succeeded}};
	if (include_accepted) result["accepted"] = accepted;
	return result;
}

Json ResourcePreview(const Json& request) {
	uint64_t address = 0;
	uint64_t size = 0;
	if (!RequestU64(request, "address", address)) return Error("invalid resource address");
	if (request.contains("size") && !RequestU64(request, "size", size))
		return Error("invalid resource size");
	Graphics::RequestResourcePreview(address, size, request.value("refresh", false));
	const auto preview = Graphics::GetResourcePreview(address, size);
	Json result = {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"status", preview.status},
	               {"address", address}, {"size", size}, {"error", preview.error},
	               {"width", preview.width}, {"height", preview.height},
	               {"source_width", preview.source_width},
	               {"source_height", preview.source_height}, {"host_format", preview.host_format},
	               {"content_hash", preview.content_hash}, {"total_pixels", preview.total_pixels},
	               {"zero_pixels", preview.zero_pixels},
	               {"non_finite_pixels", preview.non_finite_pixels},
	               {"non_finite_components", preview.non_finite_components}};
	if (!preview.rgba.empty()) result["rgba_base64"] = Base64(preview.rgba);
	return result;
}

Json IoHistory(const Json& request) {
	const auto filter = request.value("filter", std::string());
	const auto limit = std::clamp(request.value("limit", 1000u), 1u, 4096u);
	Json values = Json::array();
	for (const auto& event: Io::History(filter, limit)) {
		values.push_back({{"sequence", event.sequence}, {"timestamp_us", event.timestamp_us},
		                  {"thread", event.thread_id}, {"caller", event.caller},
		                  {"module", event.module}, {"operation", event.operation},
		                  {"descriptor", event.descriptor}, {"guest_path", event.guest_path},
		                  {"host_path", event.host_path}, {"offset", event.offset},
		                  {"requested", event.requested}, {"result", event.result}});
	}
	return {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"events", std::move(values)}};
}

Json IoFiles(const Json& request) {
	const auto filter = request.value("filter", std::string());
	const auto limit = std::clamp(request.value("limit", 4096u), 1u, 4096u);
	Json values = Json::array();
	for (const auto& file: Io::Files(filter, limit)) {
		values.push_back({{"guest_path", file.guest_path}, {"host_path", file.host_path},
		                  {"module", file.last_module}, {"first_sequence", file.first_sequence},
		                  {"last_sequence", file.last_sequence},
		                  {"last_timestamp_us", file.last_timestamp_us}, {"opens", file.opens},
		                  {"closes", file.closes}, {"reads", file.reads}, {"writes", file.writes},
		                  {"seeks", file.seeks}, {"stats", file.stats},
		                  {"bytes_read", file.bytes_read}, {"bytes_written", file.bytes_written}});
	}
	return {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"files", std::move(values)}};
}

Json Handle(const std::string& text) {
	auto request = Json::parse(text, nullptr, false);
	if (request.is_discarded() || !request.is_object()) return Error("invalid JSON request");
	if (request.value("token", std::string()) != g_token) return Error("authentication failed");

	const auto command = request.value("command", std::string());
	if (command == "summary") return Summary();
	if (command == "threads") return Threads();
	if (command == "modules") return Modules();
	if (command == "shaders") return Shaders();
	if (command == "frame") return Frame();
	if (command == "capture_command") return CaptureCommand(request);
	if (command == "symbols") return Symbols(request);
	if (command == "shader_code") return ShaderDetails(request);
	if (command == "dump_shader") return DumpShader(request);
	if (command == "stopped_details") return StoppedDetails(request);
	if (command == "breakpoints") return Breakpoints();
	if (command == "memory_read") return MemoryRead(request);
	if (command == "resource_history") return ResourceHistory(request);
	if (command == "resource_trace") return ResourceTrace(request);
	if (command == "resource_preview") return ResourcePreview(request);
	if (command == "pixel_watches") return PixelWatches();
	if (command == "pixel_watch_add") return AddPixelWatch(request);
	if (command == "pixel_watch_remove") {
		uint64_t id = 0;
		if (!RequestU64(request, "id", id) || !Graphics::RemovePixelWatch(id))
			return Error("pixel watchpoint was not found");
		return PixelWatches();
	}
	if (command == "pixel_watch_clear") {
		Graphics::ClearPixelWatches();
		return PixelWatches();
	}
	if (command == "gpu_break_conditions") return GpuBreakConditions();
	if (command == "gpu_break_condition_add") return AddGpuBreakCondition(request);
	if (command == "gpu_break_condition_remove") {
		uint64_t id = 0;
		if (!RequestU64(request, "id", id) || !Graphics::RemoveBreakCondition(id))
			return Error("GPU break condition was not found");
		return GpuBreakConditions();
	}
	if (command == "gpu_break_condition_clear") {
		Graphics::ClearBreakConditions();
		return GpuBreakConditions();
	}
	if (command == "renderdoc_status") return RenderDocStatus();
	if (command == "renderdoc_capture")
		return RenderDocStatus(Libs::Graphics::RenderDocRequestCapture(), true);
	if (command == "io_history") return IoHistory(request);
	if (command == "io_files") return IoFiles(request);
	if (command == "pause") {
		Session::Pause();
		return Summary();
	}
	if (command == "continue") {
		Session::ResumeAll();
		return Summary();
	}
	if (command == "step") {
		const int id = request.value("thread", 0);
		const auto mode = request.value("mode", std::string("into"));
		auto resume = Session::ResumeMode::StepInto;
		if (mode == "over") resume = Session::ResumeMode::StepOver;
		else if (mode == "out") resume = Session::ResumeMode::StepOut;
		else if (mode != "into") return Error("invalid step mode");
		Session::Resume(id, resume);
		return Summary();
	}
	if (command == "breakpoint_add") {
		const auto location = request.value("location", std::string());
		if (location.empty()) return Error("breakpoint location is empty");
		const auto id = Session::AddBreakpoint(location);
		if (id == 0) return Error("breakpoint could not be added");
		return {{"ok", true}, {"protocol", PROTOCOL_VERSION}, {"id", id}};
	}
	if (command == "breakpoint_remove") {
		if (!Session::RemoveBreakpoint(request.value("id", 0u))) return Error("breakpoint not found");
		return Breakpoints();
	}
	if (command == "memory_write") {
		uint64_t address = 0;
		if (!RequestU64(request, "address", address)) return Error("invalid memory address");
		std::vector<uint8_t> bytes;
		if (!ParseHexBytes(request.value("bytes", std::string()), bytes) || bytes.empty())
			return Error("invalid memory bytes");
		if (!Session::WriteMemory(address, bytes.data(), bytes.size())) return Error("memory write failed");
		return {{"ok", true}, {"protocol", PROTOCOL_VERSION}};
	}
	return Error("unknown command");
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

bool WriteAll(HANDLE pipe, const std::string& value) {
	size_t offset = 0;
	while (offset < value.size()) {
		DWORD written = 0;
		if (!WriteFile(pipe, value.data() + offset, static_cast<DWORD>(value.size() - offset),
		               &written, nullptr)) return false;
		offset += written;
	}
	return true;
}

void ServeClient(HANDLE pipe) {
	std::string pending;
	char        bytes[4096];
	while (g_running.load(std::memory_order_acquire)) {
		DWORD read = 0;
		if (!ReadFile(pipe, bytes, sizeof(bytes), &read, nullptr) || read == 0) break;
		pending.append(bytes, read);
		for (;;) {
			const auto newline = pending.find('\n');
			if (newline == std::string::npos) break;
			const auto request = pending.substr(0, newline);
			pending.erase(0, newline + 1);
			auto response = Handle(request).dump();
			response.push_back('\n');
			if (!WriteAll(pipe, response)) return;
		}
		if (pending.size() > 1024 * 1024) {
			WriteAll(pipe, Error("request too large").dump() + "\n");
			return;
		}
	}
}

void ServerThread() {
	const auto full_name = L"\\\\.\\pipe\\" + std::wstring(g_endpoint.begin(), g_endpoint.end());
	while (g_running.load(std::memory_order_acquire)) {
		const auto pipe = CreateNamedPipeW(full_name.c_str(), PIPE_ACCESS_DUPLEX,
		                                   PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1,
		                                   1024 * 1024, 1024 * 1024, 0, nullptr);
		if (pipe == INVALID_HANDLE_VALUE) break;
		const bool connected = ConnectNamedPipe(pipe, nullptr) != FALSE ||
		                       GetLastError() == ERROR_PIPE_CONNECTED;
		if (connected && g_running.load(std::memory_order_acquire)) ServeClient(pipe);
		FlushFileBuffers(pipe);
		DisconnectNamedPipe(pipe);
		CloseHandle(pipe);
	}
}

#endif

} // namespace

bool Start() {
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
	LOGF_COLOR(Log::Color::BrightYellow,
	           "Debugger remote: external attach is not implemented on this platform yet\n");
	return false;
#else
	if (g_running.exchange(true, std::memory_order_acq_rel)) return true;
	const auto pid = GetCurrentProcessId();
	g_endpoint     = "KytyDebugger-" + std::to_string(pid);
	g_token        = MakeToken();

	std::error_code ec;
	const auto      folder = SessionDirectory();
	std::filesystem::create_directories(folder, ec);
	if (ec) {
		g_running.store(false, std::memory_order_release);
		return false;
	}
	g_descriptor_path = folder / (std::to_string(pid) + ".json");
	std::ofstream file(g_descriptor_path, std::ios::binary | std::ios::trunc);
	if (!file) {
		g_running.store(false, std::memory_order_release);
		return false;
	}
	file << Json({{"schema", "kyty_debugger_session"}, {"protocol", PROTOCOL_VERSION},
	              {"pid", pid}, {"endpoint", g_endpoint}, {"token", g_token}}).dump(2);
	file.close();

	g_thread = std::thread(ServerThread);
	LOGF_COLOR(Log::Color::Cyan, "Debugger remote: listening as %s\n", g_endpoint.c_str());
	return true;
#endif
}

void Stop() {
	if (!g_running.exchange(false, std::memory_order_acq_rel)) return;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (g_thread.joinable()) CancelSynchronousIo(g_thread.native_handle());
	const auto full_name = L"\\\\.\\pipe\\" + std::wstring(g_endpoint.begin(), g_endpoint.end());
	if (const auto wake = CreateFileW(full_name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
	                                  OPEN_EXISTING, 0, nullptr);
	    wake != INVALID_HANDLE_VALUE) CloseHandle(wake);
#endif
	if (g_thread.joinable()) g_thread.join();
	std::error_code ec;
	std::filesystem::remove(g_descriptor_path, ec);
	g_descriptor_path.clear();
	g_endpoint.clear();
	g_token.clear();
}

bool IsRunning() {
	return g_running.load(std::memory_order_acquire);
}

} // namespace Debugger::Remote
