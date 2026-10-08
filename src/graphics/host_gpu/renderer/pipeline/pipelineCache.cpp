#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/cpuAffinity.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/shaderDiskCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderFunctions.h"
#include "graphics/shader/recompiler/ir/passes/SrtNative.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace Libs::Graphics {

bool ShaderFailureNonFatal() {
	static const bool value = true;
	return value;
}

namespace {

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// No emulator revision: the driver keys every pipeline by its whole create info, SPIR-V
	// included, so a new build reuses the pipelines whose shaders did not change. Keying on the
	// revision threw the whole cache away at every rebuild and release, and the first start then
	// compiled everything behind a black screen.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n", properties.vendorID,
	                   properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

// _ShaderCache/<title id>, next to _PipelineCache; empty without a title id.
std::filesystem::path ShaderDiskCacheDirectory() {
	const auto title_id = PipelineCacheTitleId();
	return title_id.empty() ? std::filesystem::path() : std::filesystem::path("_ShaderCache") / title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

// The backing bytes of the guest pages one resource evaluation (MaterializeResources, a function
// expansion) reads, found once per page: every SRT dword read otherwise took the backing store's
// lock, two map lookups and the GPU-dirty range queries (a third of the GPU thread in the
// jungle). Only pages with no GPU-written byte are cached; within one synchronous evaluation on
// the GPU thread nothing marks them GPU-written. A page that is not clean stays on the exact path.
struct GuestPageReadCache {
	static constexpr size_t Capacity = 16;
	struct Entry {
		uint64_t       page    = 0;
		const uint8_t* backing = nullptr;
	};
	std::array<Entry, Capacity> entries {};
	size_t                      count = 0;
	size_t                      next  = 0;

	// The page's backing bytes when it is GPU-clean, else null (SrtRuntime::map_clean_page).
	const uint8_t* Map(uint64_t page) {
		for (size_t i = 0; i < count; i++) {
			if (entries[i].page == page) {
				return entries[i].backing;
			}
		}
		auto& slot = count < Capacity ? entries[count++] : entries[next++ % Capacity];
		slot       = {.page    = page,
		              .backing = Libs::LibKernel::Memory::FindGpuCleanBacking(page, TRACKER_PAGE_SIZE)};
		return slot.backing;
	}

	// True when the page is GPU-clean and the bytes were copied from its backing.
	bool Read(uint64_t address, std::span<uint32_t> values) {
		const auto page = Common::AlignDown(address, TRACKER_PAGE_SIZE);
		if (Common::AlignDown(address + values.size_bytes() - 1, TRACKER_PAGE_SIZE) != page) {
			return false;
		}
		const Entry* entry = nullptr;
		for (size_t i = 0; i < count; i++) {
			if (entries[i].page == page) {
				entry = &entries[i];
				break;
			}
		}
		if (entry == nullptr) {
			auto& slot = count < Capacity ? entries[count++] : entries[next++ % Capacity];
			slot       = {.page    = page,
			              .backing = Libs::LibKernel::Memory::FindGpuCleanBacking(page,
			                                                                     TRACKER_PAGE_SIZE)};
			entry      = &slot;
		}
		if (entry->backing == nullptr) {
			return false;
		}
		std::memcpy(values.data(), entry->backing + (address - page), values.size_bytes());
		return true;
	}
};

bool UsesShaderClock(const ShaderRecompiler::IR::Program& program) {
	for (auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() == ShaderRecompiler::IR::ValueOpcode::ReadClockRealtime64) {
				return true;
			}
		}
	}
	return false;
}

// Diagnostics: the shader whose resources are being evaluated, and how often its SRT reads
// hit memory the GPU owns (each such read drains the GPU).
thread_local uint64_t t_srt_shader_hash = 0;
void CountGpuOwnedSrtRead(uint64_t address, uint32_t bytes) {
	struct Entry {
		uint64_t count   = 0;
		uint64_t address = 0;
		uint32_t bytes   = 0;
	};
	static std::unordered_map<uint64_t, Entry> counts;
	static uint64_t                            total = 0;
	auto& entry   = counts[t_srt_shader_hash];
	entry.count++;
	entry.address = address;
	entry.bytes   = bytes;
	if (++total % 4096 == 0) {
		LOGF("GpuOwnedSrtReads: total=%" PRIu64 " shaders=%zu\n", total, counts.size());
		for (const auto& [hash, e]: counts) {
			LOGF("\t hash=0x%016" PRIx64 " count=%" PRIu64 " last=0x%016" PRIx64 " bytes=%u\n",
			     hash, e.count, e.address, e.bytes);
		}
	}
}

// Ordinary (raw) SRT reads: only memory the guest has committed, read through the guest
// mapping so a GPU-owned page is refreshed first. Without a reader the walker dereferenced
// whatever address a descriptor chain produced, including 0 on a path the shader never takes.
bool ReadShaderGuestMemoryRaw(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (values.empty()) {
		return false;
	}
	if (auto* cache = static_cast<GuestPageReadCache*>(userdata);
	    cache != nullptr && cache->Read(address, values)) {
		return true;
	}
	// Bytes the GPU has not written are current in the backing store. Reading them there skips
	// the tracked-page fault, which drains the GPU to refresh whatever else on the page the GPU
	// wrote: constants that share a page with GPU-written arguments cost a drain per dispatch.
	if (Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(),
	                                                    values.size_bytes())) {
		return true;
	}
	if (!Libs::LibKernel::Memory::TryReadBacking(address, values.data(), values.size_bytes())) {
		return false;
	}
	if (t_srt_shader_hash != 0 && Libs::Graphics::GuestGpu::IsGpuThread()) {
		CountGpuOwnedSrtRead(address, static_cast<uint32_t>(values.size_bytes()));
	}
	std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	return true;
}

// KYTY_SRT_GPU_FILL=0: read every flat SRT slot on the host, waiting for the GPU when it wrote
// the bytes, instead of leaving shader-only slots to the GPU (SrtGpuFill).
bool SrtGpuFillEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SRT_GPU_FILL");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

bool SrtBytesGpuWritten(uint64_t address, uint64_t size) {
	return Libs::LibKernel::Memory::IsGpuBufferWritten(address, size);
}

bool ReadShaderGuestMemory(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (values.empty()) {
		return false;
	}
	if (auto* cache = static_cast<GuestPageReadCache*>(userdata);
	    cache != nullptr && cache->Read(address, values)) {
		return true;
	}
	if (Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes())) {
		return true;
	}
	// The bytes are mapped but GPU-owned. Reading them through the guest mapping takes the
	// tracked-page fault, which drains the GPU and refreshes the page, so the value read is
	// the one the shader would see. Before, this only ever succeeded because the aggressive
	// garbage collector happened to have downloaded the range first.
	if (!Libs::LibKernel::Memory::TryReadBacking(address, values.data(), values.size_bytes())) {
		return false;
	}
	std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	return true;
}

// The reader of a stage walked in the parallel window (StagePrepWorker): bytes no GPU work has
// written, from the backing store, and nothing else: no download, no fault, no wait for the GPU.
// A read it cannot serve fails the walk, which the GPU thread then repeats on the exact path.
bool ReadShaderGuestMemoryProbe(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (values.empty()) {
		return false;
	}
	if (auto* cache = static_cast<GuestPageReadCache*>(userdata);
	    cache != nullptr && cache->Read(address, values)) {
		return true;
	}
	return Libs::LibKernel::Memory::TryReadGpuCleanBacking(address, values.data(),
	                                                       values.size_bytes());
}

// "VS", "PS", ... for zone texts.
const char* StageLabel(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "VS";
		case ShaderType::Mesh: return "MS";
		case ShaderType::Local: return "LS";
		case ShaderType::TessellationControl: return "HS";
		case ShaderType::TessellationEvaluation: return "DS";
		case ShaderType::Pixel: return "PS";
		case ShaderType::Compute: return "CS";
		default: return "??";
	}
}

// --skip-shaders and KYTY_SKIP_SHADER_HASHES="hash,hash,...": skip the draws and dispatches of
// these guest shaders, the same way a shader that fails to compile is skipped. To see what one
// shader contributes, to step past one that loses the device, or to leave out work nothing can
// use (the ray-tracing BVH updates whose only consumer gives up).
bool SkipShaderRequested(uint64_t shader_hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> result;
		const auto parse = [&result](std::string_view list) {
			size_t start = 0;
			while (start < list.size()) {
				const auto end = std::min(list.find(',', start), list.size());
				if (end > start) {
					result.push_back(
					    std::strtoull(std::string(list.substr(start, end - start)).c_str(), nullptr, 16));
				}
				start = end + 1;
			}
		};
		parse(Config::GetSkipShaderHashes());
		if (const char* value = std::getenv("KYTY_SKIP_SHADER_HASHES"); value != nullptr) {
			parse(value);
		}
		for (const auto hash: result) {
			LOGF("ProgramCache: skipping the draws and dispatches of shader 0x%016" PRIx64 "\n",
			     hash);
		}
		return result;
	}();
	return std::ranges::find(hashes, shader_hash) != hashes.end();
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	// KYTY_DUMP_SPIRV_HASHES="hash,hash,...": dump these shaders' SPIR-V without the full
	// graphics debug dump.
	static const std::string requested = [] {
		const char* value = std::getenv("KYTY_DUMP_SPIRV_HASHES");
		return value != nullptr ? std::string(value) : std::string();
	}();
	if (!Config::GraphicsDebugDumpEnabled() &&
	    (requested.empty() ||
	     requested.find(fmt::format("{:016x}", shader_hash)) == std::string::npos)) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	if (tools.Validate(spirv)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

// A/B switch for background pipeline compiles (KYTY_ASYNC_PIPELINES=0 compiles every graphics
// pipeline on the GPU thread, as before) and how long a draw waits for a new one
// (KYTY_PIPELINE_WAIT_MS, default 20) before it is skipped.
static const bool g_async_pipelines = [] {
	const char* value = std::getenv("KYTY_ASYNC_PIPELINES");
	return value == nullptr || value[0] != '0';
}();
// Pipelines whose stages are this small (SPIR-V words, all stages; KYTY_PIPELINE_SYNC_WORDS)
// compile on the GPU thread: a few milliseconds, and their draws are never skipped. A skipped
// draw is lost for good when the game draws it once into a cached layer: the title's UI drew its
// WOLVERINE logo once, during the cold-start compile backlog, and the logo stayed missing.
static const uint64_t g_pipeline_sync_words = [] {
	const char* value = std::getenv("KYTY_PIPELINE_SYNC_WORDS");
	return value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {12000};
}();
static const bool g_dynamic_raster_state = [] {
	const char* value = std::getenv("KYTY_DYNAMIC_RASTER_STATE");
	return value == nullptr || value[0] != '0';
}();

bool PipelineDynamicRasterStateEnabled() {
	return g_dynamic_raster_state;
}

namespace {

// Zeroes key fields that cannot change the pipeline PrepareGraphicsPipeline creates, so draws
// that differ only in them share one pipeline instead of compiling identical copies (idea from
// Jetsku/KytyPS5 ef5e8fd9).
// - Cull mode and front face are dynamic state, recorded per draw from the same registers with
//   the same rect-list rule (SetGraphicsDynamicParams). The polygon mode stays static; it is
//   resolved from the real cull bits before they leave the key.
// - The depth-bounds test enable and bounds are dynamic state as well, so each DB_DEPTH_BOUNDS
//   value no longer creates its own pipeline (before, even with the test off). On macOS the
//   pipeline always disables the test (MoltenVK has no depthBounds), so they never mattered.
// Blend factors and ops of an attachment with blending off, and the alpha ones without separate
// alpha blending, are never written into the key (TryGetGraphicsPipeline), so they need no rule.
// KYTY_DYNAMIC_RASTER_STATE=0 keeps all of these in the key and bakes them into the pipeline.
void NormalizeGraphicsPipelineKey(PipelineStaticParameters& params) {
	if (!g_dynamic_raster_state) {
		return;
	}
	params.cull_front               = false;
	params.cull_back                = false;
	params.face                     = false;
	params.depth_bounds_test_enable = false;
	params.depth_min_bounds         = 0.0f;
	params.depth_max_bounds         = 0.0f;
}

} // namespace

// Draws without depth or stencil whose vertex shaders total at most KYTY_PIPELINE_SYNC_UI_VS_WORDS
// SPIR-V words (default 8000; 0 = off) also compile on the GPU thread, whatever the pixel
// shader's size: those are UI and 2D layer draws, which a game may also draw only once. On cold
// starts the title's frosted panels lost their blur (PS 0x09161b28ae4039bb, 80k words) and
// showed as dark boxes.
static const uint64_t g_pipeline_sync_ui_vs_words = [] {
	const char* value = std::getenv("KYTY_PIPELINE_SYNC_UI_VS_WORDS");
	return value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {8000};
}();
static const std::chrono::milliseconds g_pipeline_wait = [] {
	const char* value = std::getenv("KYTY_PIPELINE_WAIT_MS");
	return std::chrono::milliseconds(value != nullptr ? std::strtoul(value, nullptr, 10) : 20u);
}();

struct PendingGraphicsPipeline {
	std::unique_ptr<GraphicsPipelineBuild> build;
	// For the profiler: the guest shaders (pixel 0 without one).
	uint64_t                               vs_hash      = 0;
	uint64_t                               ps_hash      = 0;
	vk::PipelineCache                      driver_cache = nullptr;
	vk::Pipeline                           pipeline     = nullptr;
	vk::Result                             result       = vk::Result::eSuccess;
	std::atomic<bool>                      done {false};
};

// Counts a finished pipeline build and puts it on the timeline (slow ones as a warning).
static void ReportPipelineBuilt(Profiler::Counter counter, const char* kind, uint64_t first_hash,
                                uint64_t second_hash, std::chrono::steady_clock::time_point begin) {
	const auto us = static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                         std::chrono::steady_clock::now() - begin)
	                                         .count());
	Profiler::Add(counter);
	Profiler::Add(Profiler::Counter::PipelineBuildTime, us);
	Profiler::Message(us >= 100000 ? Profiler::MessageWarning : Profiler::MessageInfo,
	                  "%s pipeline built in %.1f ms: 0x%016" PRIx64 " 0x%016" PRIx64, kind,
	                  static_cast<double>(us) / 1000.0, first_hash, second_hash);
}

static void ReportGraphicsPipelineBuilt(uint64_t vs_hash, uint64_t ps_hash,
                                        std::chrono::steady_clock::time_point begin) {
	ReportPipelineBuilt(Profiler::Counter::GraphicsPipelinesBuilt, "Graphics", vs_hash, ps_hash,
	                    begin);
}

// Compiles graphics pipelines on worker threads. The driver compiled on the GPU thread, which
// stopped the whole emulated GPU for seconds per new pipeline (103 slow pipelines, 170 s of a
// 600 s first pass through the prologue).
class PipelineCompiler {
public:
	explicit PipelineCompiler(uint32_t threads) {
		for (uint32_t i = 0; i < threads; i++) {
			m_threads.emplace_back([this] { Run(); });
		}
	}
	~PipelineCompiler() { Stop(); }
	KYTY_CLASS_NO_COPY(PipelineCompiler);

	void Submit(std::shared_ptr<PendingGraphicsPipeline> job) {
		{
			std::lock_guard lock(m_mutex);
			m_queue.push_back(std::move(job));
		}
		m_work.notify_one();
	}

	// Waits up to the budget; true when the job is done.
	bool Wait(const PendingGraphicsPipeline& job, std::chrono::milliseconds budget) {
		KYTY_PROFILER_BLOCK("PipelineCompiler::Wait");
		std::unique_lock lock(m_mutex);
		return m_done.wait_for(lock, budget,
		                       [&job] { return job.done.load(std::memory_order_acquire); });
	}

	// Finishes the compiles in progress; queued ones are left to their owner.
	void Stop() {
		{
			std::lock_guard lock(m_mutex);
			if (m_stopped) {
				return;
			}
			m_stopped = true;
			m_queue.clear();
		}
		m_work.notify_all();
		for (auto& thread: m_threads) {
			thread.join();
		}
		m_threads.clear();
	}

	[[nodiscard]] bool Stopped() {
		std::lock_guard lock(m_mutex);
		return m_stopped;
	}

private:
	void Run() {
		KYTY_PROFILER_THREAD("PipelineCompiler");
		for (;;) {
			std::shared_ptr<PendingGraphicsPipeline> job;
			{
				std::unique_lock lock(m_mutex);
				m_work.wait(lock, [this] { return m_stopped || !m_queue.empty(); });
				if (m_queue.empty()) {
					return;
				}
				job = std::move(m_queue.front());
				m_queue.pop_front();
			}
			{
				Profiler::Phases zone;
				zone.SetText("VS 0x%016" PRIx64 " PS 0x%016" PRIx64, job->vs_hash, job->ps_hash);
				KYTY_PROFILER_PHASE(zone, "PipelineCompiler::Compile", profiler::colors::DeepOrangeA200);
				const auto begin = std::chrono::steady_clock::now();
				job->result =
				    CreateGraphicsPipeline(*job->build, job->driver_cache, &job->pipeline);
				ReportGraphicsPipelineBuilt(job->vs_hash, job->ps_hash, begin);
				Profiler::Add(Profiler::Counter::PipelinesPending, -1);
			}
			{
				std::lock_guard lock(m_mutex);
				job->done.store(true, std::memory_order_release);
			}
			m_done.notify_all();
		}
	}

	std::mutex                                           m_mutex;
	std::condition_variable                              m_work;
	std::condition_variable                              m_done;
	std::deque<std::shared_ptr<PendingGraphicsPipeline>> m_queue;
	std::vector<std::thread>                             m_threads;
	bool                                                 m_stopped = false;
};

// Graphics shader translation on worker threads (plan.md P1; KYTY_ASYNC_TRANSLATION=0 translates
// every shader on the GPU thread, as before). A new shader's translation (decode, CFG, IR,
// resource tracking) took ~85 ms on the GPU thread, and an effect brings dozens: the first
// Square slash in the jungle stopped the emulated GPU for 2-3 s, a new area for minutes. The
// translation depends only on the code and the stage inputs, never on guest memory or the draw's
// user data values (the disk cache's source key holds the same), so a worker can do it while the
// draw is skipped; the materialization and the SPIR-V that need guest memory stay on the GPU
// thread. Only draws with the depth test on are deferred (GetGraphicsPrograms): UI and layer
// draws, which a game may draw once, translate at once, as their pipelines compile at once.
// Compute always translates at once: a skipped dispatch can leave data unwritten.
static const bool g_async_translation = [] {
	const char* value = std::getenv("KYTY_ASYNC_TRANSLATION");
	return value == nullptr || value[0] != '0';
}();

struct TranslationJob {
	std::vector<uint32_t>                code;
	std::vector<uint32_t>                back_code;
	std::vector<uint32_t>                user_data;
	ShaderVertexInputInfo                vertex {};
	ShaderPixelInputInfo                 pixel {};
	ShaderRecompiler::CompileOptions     options;
	ShaderRecompiler::TranslateResult    result;
	std::chrono::steady_clock::time_point queued;
	std::atomic<bool>                    done {false};
};

class TranslationWorkers {
public:
	explicit TranslationWorkers(uint32_t threads) {
		for (uint32_t i = 0; i < threads; i++) {
			m_threads.emplace_back([this] { Run(); });
		}
	}
	~TranslationWorkers() {
		{
			std::lock_guard lock(m_mutex);
			m_stopped = true;
			m_queue.clear();
		}
		m_work.notify_all();
		for (auto& thread: m_threads) {
			thread.join();
		}
	}
	KYTY_CLASS_NO_COPY(TranslationWorkers);

	void Submit(std::shared_ptr<TranslationJob> job) {
		{
			std::lock_guard lock(m_mutex);
			m_queue.push_back(std::move(job));
		}
		m_work.notify_one();
	}

private:
	void Run() {
		KYTY_PROFILER_THREAD("ShaderTranslator");
		for (;;) {
			std::shared_ptr<TranslationJob> job;
			{
				std::unique_lock lock(m_mutex);
				m_work.wait(lock, [this] { return m_stopped || !m_queue.empty(); });
				if (m_queue.empty()) {
					return;
				}
				job = std::move(m_queue.front());
				m_queue.pop_front();
			}
			job->result = ShaderRecompiler::TranslateProgram(job->code, job->options);
			job->done.store(true, std::memory_order_release);
		}
	}

	std::mutex                                  m_mutex;
	std::condition_variable                     m_work;
	std::deque<std::shared_ptr<TranslationJob>> m_queue;
	std::vector<std::thread>                    m_threads;
	bool                                        m_stopped = false;
};

// Vertex and pixel SRT refresh in parallel (plan.md P10, after Jetsku's draw-prep S3, c7c4da45).
// The SRT refresh is a third of the GPU thread in the jungle, and a draw's two stages walk
// independent plans into independent snapshots. In a short window per draw, this helper walks the
// vertex stage while the GPU thread walks the pixel stage; both only read bytes no GPU work has
// written (ReadShaderGuestMemoryProbe), so neither downloads, faults or waits, and nothing changes
// the GPU-written state the other reads. A walk that fails there (a byte the GPU wrote, memory
// not mapped) is repeated by the GPU thread on the exact path, as before. The helper spins
// KYTY_STAGE_PREP_SPIN_US (default 250) after its last job, then sleeps; a draw never waits for a
// sleeping helper, it wakes it for the next. Off by default: in the jungle a window took ~7 us,
// because the helper's walk runs on cold caches and the GPU thread waits for it, which cost more
// than the walk it saved (Tracy, 2026-10-07). KYTY_STAGE_PREP_PARALLEL=1 turns it on;
// KYTY_STAGE_PREP_VERIFY=1 repeats each parallel walk on the exact path and logs differences.
static const bool g_stage_prep_parallel = [] {
	const char* value = std::getenv("KYTY_STAGE_PREP_PARALLEL");
	return value != nullptr && value[0] == '1';
}();

inline void StagePrepPause() {
#if defined(__x86_64__) || defined(_M_X64)
	_mm_pause();
#else
	std::this_thread::yield();
#endif
}

class StagePrepWorker {
public:
	using Job = void (*)(void* context);

	StagePrepWorker() {
		const char* spin = std::getenv("KYTY_STAGE_PREP_SPIN_US");
		m_spin           = std::chrono::microseconds(spin != nullptr ? std::atoi(spin) : 250);
		m_thread         = std::thread([this] { Run(); });
	}
	~StagePrepWorker() {
		{
			std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		m_stop_flag.store(true, std::memory_order_release);
		m_wake.notify_one();
		m_thread.join();
	}
	KYTY_CLASS_NO_COPY(StagePrepWorker);

	// Hands the job to the helper; false when it sleeps (it is woken for the next draw) or is
	// still busy, and the caller does the work itself.
	bool Post(Job job, void* context) {
		if (m_sleeping.load(std::memory_order_acquire)) {
			{
				std::lock_guard lock(m_mutex);
				m_wake_requested = true;
			}
			m_wake.notify_one();
			return false;
		}
		if (m_state.load(std::memory_order_acquire) != Idle) {
			return false;
		}
		m_job     = job;
		m_context = context;
		m_state.store(Posted, std::memory_order_release);
		return true;
	}

	// Waits for the posted job; false when the helper never started it (taken back).
	bool Join() {
		uint32_t expected = Posted;
		if (m_state.compare_exchange_strong(expected, Idle, std::memory_order_acq_rel)) {
			return false;
		}
		while (m_state.load(std::memory_order_acquire) != Done) {
			StagePrepPause();
		}
		m_state.store(Idle, std::memory_order_release);
		return true;
	}

private:
	enum : uint32_t { Idle, Posted, Running, Done };

	void Run() {
		KYTY_PROFILER_THREAD("StagePrep");
		Common::PreferPerformanceCores("stage preparation helper");
		Libs::LibKernel::Memory::SetGpuReadDelegate(true);
		auto last_job = std::chrono::steady_clock::now();
		while (!m_stop_flag.load(std::memory_order_acquire)) {
			uint32_t expected = Posted;
			if (m_state.load(std::memory_order_acquire) == Posted &&
			    m_state.compare_exchange_strong(expected, Running, std::memory_order_acq_rel)) {
				m_job(m_context);
				m_state.store(Done, std::memory_order_release);
				last_job = std::chrono::steady_clock::now();
				continue;
			}
			if (std::chrono::steady_clock::now() - last_job < m_spin) {
				StagePrepPause();
				continue;
			}
			std::unique_lock lock(m_mutex);
			m_sleeping.store(true, std::memory_order_release);
			m_wake.wait(lock, [this] { return m_wake_requested || m_stop; });
			m_wake_requested = false;
			m_sleeping.store(false, std::memory_order_release);
			last_job = std::chrono::steady_clock::now();
		}
	}

	std::atomic<uint32_t>     m_state {Idle};
	Job                       m_job     = nullptr;
	void*                     m_context = nullptr;
	std::atomic<bool>         m_sleeping {false};
	std::atomic<bool>         m_stop_flag {false};
	std::mutex                m_mutex;
	std::condition_variable   m_wake;
	bool                      m_wake_requested = false;
	bool                      m_stop           = false;
	std::chrono::microseconds m_spin {250};
	std::thread               m_thread;
};

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;
		// Exact expanded code includes callees and their original return-PC constants.
		// A different target/body must not reuse a program compiled for an earlier call.
		std::vector<uint32_t> function_code;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SourceEntry {
		explicit SourceEntry(ShaderRecompiler::IR::ResourcePlan plan)
		    : resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<Permutation>                    permutations;
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			PipelineKeyHash::Mix(hash, key.function_code.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	struct Compiled {
		std::vector<uint32_t>                    spirv;
		ShaderRecompiler::IR::CompiledShaderInfo program;
	};

	Compiled CompileTranslated(const ShaderRecompiler::CompileOptions&             options,
	                           ShaderRecompiler::TranslateResult                   translated,
	                           const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                           uint32_t push_data_start_dword) {
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		last_spirv_words = result.spirv.size();
		return {.spirv   = std::move(result.spirv),
		        .program = std::move(result.program).TakeCompiledInfo()};
	}

	// A compiled (or cached) permutation of the source: its module, bound for this stage.
	template <typename InputInfo>
	ShaderProgram AddPermutation(SourceEntry& source, const char* stage_name,
	                             const ShaderRecompiler::CompileOptions&  options,
	                             const std::vector<uint32_t>&             spirv,
	                             ShaderRecompiler::IR::CompiledShaderInfo program,
	                             InputInfo& input_info, uint32_t& push_data_cursor) {
		Profiler::Phases phases;
		phases.SetText("%s 0x%016" PRIx64, stage_name, options.shader_hash);
		KYTY_PROFILER_PHASE(phases, "Shader: SPIR-V validate", profiler::colors::Blue300);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, spirv);

		KYTY_PROFILER_PHASE(phases, "Shader: module create", profiler::colors::Blue300);
		const auto module = CompileSPV(spirv, device);
		EXIT_IF(module == nullptr);
		phases.End();
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(spirv.size()), options.wave_size);
		}
		source.permutations.push_back({
		    .specialization = source.specialization,
		    .program        = std::move(program),
		    .handle         = {.id          = ++next_shader_id,
		                       .module      = module,
		                       .spirv_words = static_cast<uint32_t>(spirv.size()),
		                       .hash        = options.shader_hash},
		});
		const auto& permutation = source.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &source.resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, entry]: programs) {
			counts[static_cast<size_t>(key.stage)] += entry.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	// The first materialization of a new source; a failure skips this draw (see Get).
	static bool MaterializeFirstUse(SourceEntry&                           source,
	                                const ShaderRecompiler::IR::SrtRuntime& runtime,
	                                ShaderType stage, const char* stage_name, uint64_t hash) {
		if (ShaderRecompiler::IR::MaterializeResources(source.resource_plan, runtime,
		                                               source.resources, source.specialization)) {
			return true;
		}
		if (!ShaderFailureNonFatal()) {
			EXIT("shader resource materialization failed\n");
		}
		Profiler::Add(Profiler::Counter::SkippedResources);
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 16) {
			LOGF("ProgramCache: skipping stage %u hash=0x%016" PRIx64
			     ": resource materialization failed on first use (materialization line %d)\n",
			     static_cast<uint32_t>(stage), hash,
			     ShaderRecompiler::IR::LastIndirectImageFailureLine());
			Profiler::Message(Profiler::MessageWarning,
			                  "Resources unreadable on first use, draw skipped: %s 0x%016" PRIx64
			                  " (materialization line %d; first 16 reported)",
			                  stage_name, hash, ShaderRecompiler::IR::LastIndirectImageFailureLine());
		}
		return false;
	}

	static void ReportShaderClock(ShaderType stage, const char* stage_name, uint64_t hash) {
		if (!ShaderFailureNonFatal()) {
			EXIT("S_MEMREALTIME needs shaderDeviceClock\n");
		}
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 16) {
			LOGF("ProgramCache: skipping stage %u hash=0x%016" PRIx64
			     ": S_MEMREALTIME needs shaderDeviceClock\n",
			     static_cast<uint32_t>(stage), hash);
		}
		Profiler::Add(Profiler::Counter::ShadersGaveUp);
		Profiler::Message(Profiler::MessageFailure,
		                  "Shader gave up: %s 0x%016" PRIx64 ": S_MEMREALTIME needs shaderDeviceClock",
		                  stage_name, hash);
	}

	// The disk cache key of a source: the session key and everything TranslateProgram reads.
	template <typename InputInfo>
	void BuildSourceKey(std::vector<uint8_t>& key, std::span<const uint32_t> code,
	                    const ShaderParams& params, const ShaderRecompiler::CompileOptions& options,
	                    const InputInfo& input_info) {
		if (disk_session_key.empty()) {
			// Built at first use: the device layer has set the emitter's switches by then.
			disk_session_key = ShaderDiskCache::SessionKey(disk_session);
		}
		ShaderDiskCache::ByteWriter writer;
		writer.bytes.reserve(disk_session_key.size() + code.size_bytes() +
		                     params.back_code.size_bytes() + 512u);
		writer.PutBytes(disk_session_key);
		writer.Put<uint8_t>('S');
		writer.Put(static_cast<uint32_t>(options.stage));
		writer.Put(options.shader_hash);
		writer.Put(options.wave_size);
		writer.Put(options.user_data_base);
		writer.Put(static_cast<uint32_t>(options.user_data.size()));
		writer.Put(options.bindless_images);
		writer.Put(options.non_fatal);
		writer.PutWords(code);
		writer.PutWords(params.back_code);
		ShaderDiskCache::AppendInputKey(writer, input_info);
		key = std::move(writer.bytes);
	}

	static void BuildPermutationKey(std::vector<uint8_t>& key, std::span<const uint8_t> source_key,
	                                const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                                uint32_t push_data_cursor) {
		ShaderDiskCache::ByteWriter writer;
		writer.bytes.reserve(source_key.size() + 256u);
		writer.PutBytes(source_key);
		writer.Put<uint8_t>('P');
		ShaderDiskCache::AppendSpecialization(writer, specialization);
		writer.Put(push_data_cursor);
		key = std::move(writer.bytes);
	}

	// Stores a fresh entry; in verify mode compares it with the cached one first (`cached`, empty
	// when there was none) and stores it only when they differ.
	void StoreEntry(bool source, uint64_t hash, std::span<const uint8_t> key,
	                std::span<const uint8_t> cached, std::span<const uint8_t> payload) {
		const char* kind = source ? "source" : "permutation";
		if (!(source ? ShaderDiskCache::SourceRecordRoundTrips(payload)
		             : ShaderDiskCache::PermutationRecordRoundTrips(payload))) {
			if (disk->counters.round_trip_failures++ < 16) {
				LOGF("Shader disk cache: the %s entry of hash=0x%016" PRIx64
				     " does not decode to itself; not cached\n",
				     kind, hash);
			}
			return;
		}
		if (!cached.empty()) {
			disk->counters.verified++;
			if (std::ranges::equal(cached, payload)) {
				return;
			}
			if (disk->counters.verify_mismatches++ < 32) {
				LOGF("Shader disk cache: verify mismatch: %s entry of hash=0x%016" PRIx64
				     " (%zu bytes cached, %zu fresh)\n",
				     kind, hash, cached.size(), payload.size());
				Profiler::Message(Profiler::MessageFailure,
				                  "Shader disk cache: verify mismatch: %s entry of 0x%016" PRIx64,
				                  kind, hash);
			}
		}
		disk->Save(key, payload);
	}

	void CountDiskLookup() {
		if (++disk_lookups % 4096u == 0) {
			disk->LogTotals("so far");
		}
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}

		if (SkipShaderRequested(params.hash)) {
			Profiler::Add(Profiler::Counter::SkippedSkipList);
			Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": skip list", StageLabel(stage),
			                                params.hash);
			return ShaderProgram {};
		}
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		lookup_key.function_code.clear();
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			ExpandShaderFunctions(params, user_data, input_info.wave_size,
			                      lookup_key.function_code);
		}
		if (unsupported.contains(lookup_key)) {
			Profiler::Add(Profiler::Counter::SkippedGaveUp);
			Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": shader gave up",
			                                StageLabel(stage), params.hash);
			return ShaderProgram {};
		}
		KYTY_PROFILER_BLOCK("ProgramCache::Get");
		auto                                         entry = programs.find(lookup_key);
		// Scoped to this call only: see GuestPageReadCache.
		GuestPageReadCache                           clean_read_cache;
		ShaderRecompiler::IR::SrtRuntime             runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderGuestMemoryRaw,
		    .userdata                   = &clean_read_cache,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .float_image_atomics        = Config::FloatImageAtomicsEnabled(),
		};
		static_assert(TRACKER_PAGE_SIZE == 4096u);
		runtime.map_clean_page = +[](void* userdata, uint64_t page) {
			return static_cast<GuestPageReadCache*>(userdata)->Map(page);
		};
		runtime.page_userdata = &clean_read_cache;
		runtime.page_shift    = 12;
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			runtime.workgroup_counts = input_info.workgroup_counts;
			for (uint32_t axis = 0; axis < 3u; axis++) {
				runtime.workgroup_size[axis] = input_info.threads_num[axis];
			}
		}
		if (entry != programs.end()) {
			{
				// One zone per refresh, named by the program (kept with KYTY_PROFILE_ZONES=0).
				Profiler::Phases refresh_zone;
				refresh_zone.SetText("%s %016" PRIx64, StageLabel(stage), params.hash);
				KYTY_PROFILER_PHASE(refresh_zone, "SRT refresh", profiler::colors::Amber300);
				// Compute: flat slots only the shader uses are left to the GPU when it wrote their
				// bytes; the renderer copies them on the GPU before the dispatch (SrtGpuFill).
				entry->second.resources.gpu_fills.clear();
				if (stage == ShaderType::Compute && SrtGpuFillEnabled()) {
					runtime.gpu_written = SrtBytesGpuWritten;
					runtime.gpu_fills   = &entry->second.resources.gpu_fills;
				}
				t_srt_shader_hash = params.hash;
				bool materialized = false;
				if (TakePrepared(&entry->second)) {
					// Refreshed in this draw's parallel window (PrepareStagesInParallel).
					static const bool verify = std::getenv("KYTY_STAGE_PREP_VERIFY") != nullptr;
					materialized             = true;
					if (verify) {
						VerifyPrepared(entry->second, runtime, stage, params.hash);
					}
				} else {
					materialized = ShaderRecompiler::IR::MaterializeResources(
					    entry->second.resource_plan, runtime, entry->second.resources,
					    entry->second.specialization);
				}
				t_srt_shader_hash = 0;
				runtime.gpu_written = nullptr;
				runtime.gpu_fills   = nullptr;
				if (ShaderRecompiler::IR::SrtNativeStats stats;
				    ShaderRecompiler::IR::TakeSrtNativeReport(stats)) {
					LOGF("SRT native: %u plans compiled, %u failed, %" PRIu64 " KB of code, %" PRIu64
					     " instructions, %" PRIu64 " through the interpreter\n",
					     stats.plans, stats.failed, stats.bytes / 1024u, stats.instructions,
					     stats.interpreted);
				}
				if (!materialized) {
					// A descriptor source that cannot be read right now (memory the guest has
					// not mapped or filled yet) skips this draw rather than the session; the
					// next one re-evaluates from scratch.
					if (!ShaderFailureNonFatal()) {
						EXIT("shader resource materialization failed\n");
					}
					Profiler::Add(Profiler::Counter::SkippedResources);
					Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": resources unreadable",
					                                StageLabel(stage), params.hash);
					static std::atomic<uint32_t> reported = 0;
					if (reported.fetch_add(1) < 16) {
						LOGF("ProgramCache: skipping stage %u hash=0x%016" PRIx64
						     ": resource materialization failed (materialization line %d)\n",
						     static_cast<uint32_t>(stage), params.hash,
						     ShaderRecompiler::IR::LastIndirectImageFailureLine());
						Profiler::Message(Profiler::MessageWarning,
						                  "Resources unreadable, draw skipped: stage %u 0x%016" PRIx64
						                  " (materialization line %d; first 16 reported)",
						                  static_cast<uint32_t>(stage), params.hash,
						                  ShaderRecompiler::IR::LastIndirectImageFailureLine());
					}
					return ShaderProgram {};
				}
			}
			if (const auto permutation = std::ranges::find_if(
			        entry->second.permutations, [&](const Permutation& candidate) {
				        const auto& layout = candidate.program.bindings;
				        return layout.push_data_start_dword ==
				                   ShaderRecompiler::IR::PushData::StartFor(
				                       push_data_cursor, layout.ShaderDataDwords()) &&
				               candidate.specialization == entry->second.specialization;
			        });
			    permutation != entry->second.permutations.end()) {
				input_info.stage = {.program   = &permutation->program,
				                    .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

		// A translation on the workers (TranslationWorkers): the draw waits for it by being skipped.
		if (!translating.empty()) {
			if (const auto job = translating.find(lookup_key);
			    job != translating.end() && !job->second->done.load(std::memory_order_acquire)) {
				translation_skips++;
				Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": translating",
				                                StageLabel(stage), params.hash);
				return ShaderProgram {};
			}
		}

		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		const char* stage_name = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; stage_name = "vs"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; stage_name = "ms"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; stage_name = "ls"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; stage_name = "ds"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; stage_name = "ps"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.shader_hash = params.hash;
		options.user_data   = user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.dump_label  = label;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		options.non_fatal = ShaderFailureNonFatal();
		options.bindless_images = bindless_images;
		const auto compile_code = lookup_key.function_code.empty()
		                              ? params.code
		                              : std::span<const uint32_t>(lookup_key.function_code);
		// The shader disk cache (shaderDiskCache.h): a source entry stands in for the translation
		// of a source new to this session, a permutation entry for the compile. In verify mode
		// both are only compared with a fresh translation. A hit is not a compile: it is not
		// counted in ShadersCompiled or ShaderCompileTime.
		std::vector<uint8_t> source_key;
		std::vector<uint8_t> cached_source;
		std::vector<uint8_t> permutation_key;
		std::vector<uint8_t> cached_permutation;
		if (disk != nullptr) {
			Profiler::Phases disk_zone;
			disk_zone.SetText("%s 0x%016" PRIx64, stage_name, params.hash);
			KYTY_PROFILER_PHASE(disk_zone, "Shader disk cache", profiler::colors::CyanA700);
			BuildSourceKey(source_key, compile_code, params, options, input_info);
			if (entry == programs.end()) {
				CountDiskLookup();
				std::vector<uint8_t>          payload;
				ShaderDiskCache::SourceRecord record;
				if (!disk->Load(source_key, payload)) {
					disk->counters.source_misses++;
				} else if (disk_verify) {
					disk->counters.source_hits++;
					cached_source = std::move(payload);
				} else if (!ShaderDiskCache::DecodeSourceRecord(payload, record)) {
					disk->counters.rejected++;
					disk->counters.source_misses++;
				} else {
					disk->counters.source_hits++;
					if (record.unsupported) {
						Profiler::Add(Profiler::Counter::ShadersGaveUp);
						Profiler::Message(Profiler::MessageFailure,
						                  "Shader gave up: %s 0x%016" PRIx64 ": in an earlier run (disk cache)",
						                  stage_name, params.hash);
						unsupported.insert(lookup_key);
						return ShaderProgram {};
					}
					if (record.uses_clock && !shader_clock) {
						ReportShaderClock(stage, stage_name, params.hash);
						unsupported.insert(lookup_key);
						return ShaderProgram {};
					}
					entry = programs.try_emplace(lookup_key, std::move(record.plan)).first;
					if (!MaterializeFirstUse(entry->second, runtime, stage, stage_name, params.hash)) {
						return ShaderProgram {};
					}
				}
			}
			if (entry != programs.end()) {
				BuildPermutationKey(permutation_key, source_key, entry->second.specialization,
				                    push_data_cursor);
				std::vector<uint8_t>               payload;
				ShaderDiskCache::PermutationRecord record;
				if (!disk->Load(permutation_key, payload)) {
					disk->counters.permutation_misses++;
				} else if (disk_verify) {
					disk->counters.permutation_hits++;
					cached_permutation = std::move(payload);
				} else if (!ShaderDiskCache::DecodePermutationRecord(payload, record)) {
					disk->counters.rejected++;
					disk->counters.permutation_misses++;
				} else {
					disk->counters.permutation_hits++;
					Profiler::Message(Profiler::MessageInfo,
					                  "Shader from the disk cache: %s 0x%016" PRIx64 ", %zu SPIR-V words",
					                  stage_name, params.hash, record.spirv.size());
					return AddPermutation(entry->second, stage_name, options, record.spirv,
					                      std::move(record.program), input_info, push_data_cursor);
				}
			}
		}

		// The whole compile as one zone on the timeline, its phases below it.
		Profiler::Phases compile_zone;
		compile_zone.SetText("%s 0x%016" PRIx64, stage_name, params.hash);
		KYTY_PROFILER_PHASE(compile_zone, "Shader compile", profiler::colors::DeepOrangeA200);
		const auto compile_begin = std::chrono::steady_clock::now();
		ShaderRecompiler::TranslateResult translated;
		if (const auto job = translating.find(lookup_key); job != translating.end()) {
			// Done (the check above returned while it was not): take the worker's translation.
			const auto waited = std::chrono::duration<double, std::milli>(
			                        std::chrono::steady_clock::now() - job->second->queued)
			                        .count();
			if (translations_done++ < 32) {
				LOGF("ShaderTranslator: %s 0x%016" PRIx64 " ready after %.0f ms, %" PRIu64
				     " draws skipped so far while translating\n",
				     stage_name, params.hash, waited, translation_skips);
			}
			translated = std::move(job->second->result);
			translating.erase(job);
		} else if (defer_translation && stage != ShaderType::Compute && translators != nullptr) {
			auto job = std::make_shared<TranslationJob>();
			job->code.assign(compile_code.begin(), compile_code.end());
			job->back_code.assign(params.back_code.begin(), params.back_code.end());
			job->user_data.assign(user_data.begin(), user_data.end());
			job->options           = options;
			job->options.user_data = job->user_data;
			job->options.back_code = job->back_code;
			if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
				job->vertex                   = input_info;
				job->options.input_info.vertex = &job->vertex;
			} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
				job->pixel                   = input_info;
				job->options.input_info.pixel = &job->pixel;
			}
			job->queued = std::chrono::steady_clock::now();
			translating.emplace(lookup_key, job);
			translators->Submit(std::move(job));
			translation_skips++;
			Profiler::CommandZone::Annotate("skipped %s %016" PRIx64 ": translating",
			                                StageLabel(stage), params.hash);
			return ShaderProgram {};
		} else {
			translated = ShaderRecompiler::TranslateProgram(compile_code, options);
		}
		if (translated.unsupported) {
			// Remember the refusal: a skipped shader is dispatched again every frame, and
			// re-deriving the same answer costs as much as a compile each time.
			if (disk != nullptr && entry == programs.end()) {
				std::vector<uint8_t> payload;
				if (ShaderDiskCache::EncodeSourceRecord(true, false, nullptr, payload)) {
					StoreEntry(true, params.hash, source_key, cached_source, payload);
				}
			}
			unsupported.insert(lookup_key);
			return ShaderProgram {};
		}
		const bool uses_clock = UsesShaderClock(translated.program);
		std::optional<ShaderRecompiler::IR::ResourcePlan> plan;
		if (entry == programs.end()) {
			plan.emplace(ShaderRecompiler::IR::ExtractResourcePlan(translated.program));
			if (disk != nullptr) {
				std::vector<uint8_t> payload;
				if (ShaderDiskCache::EncodeSourceRecord(false, uses_clock, &*plan, payload)) {
					StoreEntry(true, params.hash, source_key, cached_source, payload);
				} else if (disk->counters.round_trip_failures++ < 16) {
					LOGF("Shader disk cache: the resource plan of hash=0x%016" PRIx64
					     " refers outside itself; not cached\n",
					     params.hash);
				}
			}
		}
		if (!shader_clock && uses_clock) {
			ReportShaderClock(stage, stage_name, params.hash);
			unsupported.insert(lookup_key);
			return ShaderProgram {};
		}
		if (entry == programs.end()) {
			entry = programs.try_emplace(lookup_key, std::move(*plan)).first;
			if (!MaterializeFirstUse(entry->second, runtime, stage, stage_name, params.hash)) {
				return ShaderProgram {};
			}
			if (disk != nullptr) {
				BuildPermutationKey(permutation_key, source_key, entry->second.specialization,
				                    push_data_cursor);
				if (disk_verify) {
					std::vector<uint8_t> payload;
					if (disk->Load(permutation_key, payload)) {
						cached_permutation = std::move(payload);
					}
				}
			}
		}
		auto compiled = CompileTranslated(options, std::move(translated),
		                                  entry->second.specialization, push_data_cursor);
		std::vector<uint8_t> permutation_payload;
		if (disk != nullptr) {
			ShaderDiskCache::EncodePermutationRecord(compiled.spirv, compiled.program,
			                                         permutation_payload);
		}
		const auto handle = AddPermutation(entry->second, stage_name, options, compiled.spirv,
		                                   std::move(compiled.program), input_info,
		                                   push_data_cursor);
		{
			const auto us = static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                         std::chrono::steady_clock::now() - compile_begin)
			                                         .count());
			Profiler::Add(Profiler::Counter::ShadersCompiled);
			Profiler::Add(Profiler::Counter::ShaderCompileTime, us);
			Profiler::Message(Profiler::MessageInfo,
			                  "Shader compiled: %s 0x%016" PRIx64 " in %.1f ms, %zu SPIR-V words",
			                  stage_name, params.hash, static_cast<double>(us) / 1000.0,
			                  last_spirv_words);
		}
		compile_zone.End();
		if (disk != nullptr) {
			// After AddPermutation: SPIR-V that fails validation is never stored.
			StoreEntry(false, params.hash, permutation_key, cached_permutation,
			           permutation_payload);
		}
		return handle;
	}

	ProgramCache(vk::Device device, bool shader_clock, bool bindless_images,
	             ShaderDiskCache::SessionInfo session, std::filesystem::path disk_directory)
	    : device(device), shader_clock(shader_clock), bindless_images(bindless_images),
	      disk_session(session), disk_verify(ShaderDiskCache::VerifyEnabled()) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
		if (Config::GraphicsDebugDumpEnabled()) {
			// The dumps are written while translating: every shader has to be translated.
			PipelineCacheLog("Shader disk cache: off (graphics debug dump)");
		} else {
			disk = ShaderDiskCache::Store::Open(std::move(disk_directory));
		}
		if (g_async_translation) {
			// KYTY_TRANSLATION_THREADS=<n> (default 2).
			const char* value   = std::getenv("KYTY_TRANSLATION_THREADS");
			const auto  threads = std::max<uint32_t>(
			    value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 2u, 1u);
			translators = std::make_unique<TranslationWorkers>(threads);
			PipelineCacheLog("Shader translation: graphics shaders of 3D draws on {} worker threads",
			                 threads);
		}
		if (g_stage_prep_parallel) {
			stage_prep = std::make_unique<StagePrepWorker>();
			PipelineCacheLog("Stage preparation: vertex and pixel SRT refresh in parallel");
		}
	}

	void FlushDisk() {
		if (disk != nullptr) {
			disk->Flush();
			disk->LogTotals("saved");
		}
	}
	~ProgramCache() {
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	// Shader function expansion decodes the shader and every callee, and a shader with table
	// calls can be dispatched hundreds of times a frame; the expander keeps what it can reuse.
	void ExpandShaderFunctions(const ShaderParams& params, std::span<const uint32_t> user_data,
	                           uint32_t wave_size, std::vector<uint32_t>& expanded) {
		std::string reason;
		if (!function_expander.Expand(
		        params.code, params.Base(), user_data,
		        [](uint64_t address, std::span<uint32_t> words) {
			        return ReadShaderGuestMemoryRaw(nullptr, address, words);
		        },
		        expanded, reason, wave_size)) {
			static std::atomic<uint32_t> reports {0};
			if (reports.fetch_add(1) < 16) {
				::printf("Shader function expansion hash=%016" PRIx64 ": %s\n", params.hash,
				         reason.c_str());
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash>    programs;
	std::unordered_set<ProgramKey, ProgramKeyHash>                 unsupported;
	ShaderRecompiler::Decoder::ShaderFunctionExpander             function_expander;
	ProgramKey                                                     lookup_key;
	vk::Device                                                  device;
	bool                                                        shader_clock = false;
	bool                                                        bindless_images = false;
	uint64_t                                                    next_shader_id = 0;
	// SPIR-V size of the last permutation compiled (for the profiler).
	size_t                                                      last_spirv_words = 0;
	// The shader disk cache (null when off) and its session key, built at first use.
	ShaderDiskCache::SessionInfo                                disk_session;
	std::vector<uint8_t>                                        disk_session_key;
	bool                                                        disk_verify  = false;
	uint64_t                                                    disk_lookups = 0;
	std::unique_ptr<ShaderDiskCache::Store>                     disk;
	// Graphics translations in progress on the workers (KYTY_ASYNC_TRANSLATION), by program.
	std::unordered_map<ProgramKey, std::shared_ptr<TranslationJob>, ProgramKeyHash> translating;
	uint64_t                                                    translation_skips = 0;
	uint64_t                                                    translations_done = 0;
	// Set by GetGraphicsPrograms for the draw at hand: may its shaders translate in the background?
	bool                                                        defer_translation = false;
	// The entries whose snapshot the parallel window refreshed for the draw at hand
	// (PrepareStagesInParallel); Get takes them instead of walking again.
	std::array<SourceEntry*, 2>                                 prepared {};
	ProgramKey                                                  prep_key;
	uint64_t                                                    prep_windows    = 0;
	uint64_t                                                    prep_vs_ok      = 0;
	uint64_t                                                    prep_ps_ok      = 0;
	uint64_t                                                    prep_declined   = 0;
	uint64_t                                                    prep_mismatches = 0;
	std::unique_ptr<StagePrepWorker>                            stage_prep;
	// Last: destroyed first, so its threads have stopped before anything else goes.
	std::unique_ptr<TranslationWorkers>                         translators;

	// A stage's program entry as Get would find it, or null (not translated yet, skipped).
	template <typename InputInfo>
	SourceEntry* FindForPrep(const ShaderParams& params, const InputInfo& input_info,
	                         ShaderType stage) {
		if (SkipShaderRequested(params.hash)) {
			return nullptr;
		}
		prep_key.stage           = stage;
		prep_key.hash            = params.hash;
		prep_key.user_data_count = params.user_data_count;
		prep_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, prep_key.static_state);
		prep_key.function_code.clear();
		const auto entry = programs.find(prep_key);
		return entry != programs.end() ? &entry->second : nullptr;
	}

	// The SRT refresh of Get with the probe reader (ReadShaderGuestMemoryProbe).
	static bool WalkProbe(SourceEntry& entry, const ShaderParams& params) {
		GuestPageReadCache               cache;
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = std::span(params.user_data).first(params.user_data_count),
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderGuestMemoryProbe,
		    .userdata                   = &cache,
		    .read_specialization_memory = ReadShaderGuestMemoryProbe,
		    .float_image_atomics        = Config::FloatImageAtomicsEnabled(),
		};
		runtime.map_clean_page = +[](void* userdata, uint64_t page) {
			return static_cast<GuestPageReadCache*>(userdata)->Map(page);
		};
		runtime.page_userdata = &cache;
		runtime.page_shift    = 12;
		return ShaderRecompiler::IR::MaterializeResources(entry.resource_plan, runtime,
		                                                  entry.resources, entry.specialization);
	}

	// The parallel window of a draw (StagePrepWorker): the helper refreshes the vertex stage while
	// this thread refreshes the pixel stage. Both entries must exist already.
	void PrepareStagesInParallel(const ShaderParams& ps_params, const ShaderPixelInputInfo& ps_info,
	                             const ShaderParams&          vs_params,
	                             const ShaderVertexInputInfo& vs_info) {
		prepared = {};
		if (stage_prep == nullptr) {
			return;
		}
		auto* ps = FindForPrep(ps_params, ps_info, ShaderType::Pixel);
		if (ps == nullptr) {
			return;
		}
		auto* vs = FindForPrep(vs_params, vs_info, vs_info.logical_stage);
		if (vs == nullptr || vs == ps) {
			return;
		}
		struct Context {
			SourceEntry*        entry;
			const ShaderParams* params;
			bool                ok;
		} job {vs, &vs_params, false};
		KYTY_PROFILER_BLOCK("StagePrep::Window");
		if (!stage_prep->Post(
		        [](void* context) {
			        auto& job = *static_cast<Context*>(context);
			        job.ok    = WalkProbe(*job.entry, *job.params);
		        },
		        &job)) {
			prep_declined++;
			return;
		}
		const bool ps_ok = WalkProbe(*ps, ps_params);
		const bool vs_ok = stage_prep->Join() && job.ok;
		prep_windows++;
		prep_ps_ok += ps_ok ? 1u : 0u;
		prep_vs_ok += vs_ok ? 1u : 0u;
		prepared = {ps_ok ? ps : nullptr, vs_ok ? vs : nullptr};
		if ((prep_windows & (prep_windows - 1u)) == 0u && prep_windows >= 1024u) {
			LOGF("Stage preparation: %" PRIu64 " parallel windows, pixel walks served %" PRIu64
			     ", vertex walks served %" PRIu64 ", %" PRIu64 " declined (helper asleep or "
			     "busy), %" PRIu64 " verify mismatches\n",
			     prep_windows, prep_ps_ok, prep_vs_ok, prep_declined, prep_mismatches);
		}
	}

	bool TakePrepared(const SourceEntry* entry) {
		for (auto& slot: prepared) {
			if (slot == entry && entry != nullptr) {
				slot = nullptr;
				return true;
			}
		}
		return false;
	}

	// KYTY_STAGE_PREP_VERIFY=1: the exact walk again after a parallel one, compared.
	void VerifyPrepared(SourceEntry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                    ShaderType stage, uint64_t hash) {
		const auto resources      = entry.resources;
		const auto specialization = entry.specialization;
		if (!ShaderRecompiler::IR::MaterializeResources(entry.resource_plan, runtime,
		                                                entry.resources, entry.specialization)) {
			return;
		}
		const auto& fresh = entry.resources;
		if (resources.flattened_srt == fresh.flattened_srt && resources.buffers == fresh.buffers &&
		    resources.images == fresh.images && resources.samplers == fresh.samplers &&
		    specialization == entry.specialization) {
			return;
		}
		if (prep_mismatches++ < 32) {
			LOGF("Stage preparation: verify mismatch %s 0x%016" PRIx64 " (flat %d, buffers %d, "
			     "images %d, samplers %d, specialization %d)\n",
			     StageLabel(stage), hash,
			     static_cast<int>(resources.flattened_srt != fresh.flattened_srt),
			     static_cast<int>(resources.buffers != fresh.buffers),
			     static_cast<int>(resources.images != fresh.images),
			     static_cast<int>(resources.samplers != fresh.samplers),
			     static_cast<int>(!(specialization == entry.specialization)));
		}
	}

	void SetDeferTranslation(bool value) { defer_translation = value; }
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics),
      m_program_cache(std::make_unique<ProgramCache>(
          graphics.device, graphics.shader_device_clock_enabled,
          graphics.bindless_enabled && Config::BindlessImagesEnabled(),
          ShaderDiskCache::SessionInfo {
              .vendor_id           = graphics.GetPhysicalDeviceProperties().vendorID,
              .device_id           = graphics.GetPhysicalDeviceProperties().deviceID,
              .bindless_images     = graphics.bindless_enabled && Config::BindlessImagesEnabled(),
              .float_image_atomics = Config::FloatImageAtomicsEnabled()},
          ShaderDiskCacheDirectory())) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeDriverCache();
	if (g_async_pipelines) {
		const auto threads = std::clamp(std::thread::hardware_concurrency() / 4u, 1u, 4u);
		m_compiler         = std::make_unique<PipelineCompiler>(threads);
		PipelineCacheLog("Vulkan pipelines: compiled on {} worker threads (wait {} ms)", threads,
		                 g_pipeline_wait.count());
	}
}

PipelineCache::~PipelineCache() {
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			if (pipeline->pending && pipeline->pending->done.load(std::memory_order_acquire)) {
				m_graphics.device.destroyPipeline(pipeline->pending->pipeline, nullptr);
			}
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}

	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		// Pipelines of older builds stay in the file; past 1 GiB it starts over.
		constexpr uint64_t max_file_size = uint64_t {1} << 30u;
		if (file_size >= signature.size() + sizeof(uint64_t) && file_size <= max_file_size) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::Save() {
	m_program_cache->FlushDisk();
	// No worker may use the driver cache once it is destroyed below; later pipelines are
	// compiled on the GPU thread (FinishPending).
	if (m_compiler != nullptr) {
		m_compiler->Stop();
	}
	if (m_driver_cache == nullptr) {
		return;
	}

	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
		                 vk::to_string(result), size);
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    !Common::File::RenameFile(temp_path, m_driver_cache_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}", payload.size(),
	                 Common::PathToString(m_driver_cache_path));
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	m_driver_cache = nullptr;
}

// Glyphs, icons and other cached layers are drawn once into small offscreen targets, and a draw
// skipped while its shaders translate or its pipeline compiles is lost for good: the Leap Attack
// prompt lost letters ("L p Att k") in every new game whose session translated shaders
// (2026-10-08, ISSUES #27), whatever the depth state of those draws. They translate and compile
// at once instead; a one-time cost per program, none once the disk caches hold it.
bool SmallColorTargetDraw(const HW::Context& context) {
	static const uint32_t limit = [] {
		const char* value = std::getenv("KYTY_SMALL_TARGET_SYNC");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 1024u;
	}();
	if (limit == 0) {
		return false;
	}
	const auto mask  = context.GetRenderTargetMask();
	bool       found = false;
	for (uint32_t slot = 0; slot < 8; slot++) {
		const auto& target = context.GetRenderTarget(slot);
		if (((mask >> (slot * 4u)) & 0x0fu) == 0 || target.base.addr == 0) {
			continue;
		}
		if (target.attrib2.width + 1u > limit || target.attrib2.height + 1u > limit) {
			return false;
		}
		found = true;
	}
	return found;
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info) {
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		if (!PrepareTessellationPrograms(vertex_regs, context, vertex_info, vertex_params)) {
			if (!ShaderFailureNonFatal()) {
				EXIT("unsupported tessellation programs\n");
			}
			return {};
		}
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		// The pipeline requires the wave size where it can (PrepareGraphicsPipeline).
		const auto required =
		    m_graphics.GraphicsSubgroupSize(vk::ShaderStageFlagBits::eMeshEXT, mesh.wave_size);
		mesh.host_subgroup_size = required != 0 ? required : m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		// A subgroup with more threads than a mesh workgroup may have (NVIDIA: 128) runs its
		// waves in passes; see EmitMeshEntryPoint.
		mesh.passes = mesh.PassesFor(
		    std::min(limits.maxMeshWorkGroupInvocations, limits.maxMeshWorkGroupSize[0]));
		if (mesh.passes == 0 || mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			// Skipped like the draws of a shader that gives up, and reported once per shader.
			static std::mutex                   logged_mutex;
			static std::unordered_set<uint64_t> logged;
			std::lock_guard                     lock(logged_mutex);
			if (logged.insert(vertex_params[0].hash).second) {
				LOGF("mesh shader 0x%016" PRIx64 " exceeds host limits, draws skipped: wave%u "
				     "threads=%u vertices=%u primitives=%u LDS=%u\n",
				     vertex_params[0].hash, mesh.wave_size, mesh.HostThreads(), mesh.max_vertices,
				     mesh.max_primitives, mesh.lds_size_dwords);
			}
			return {};
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		// SPI_SHADER_COL_FORMAT describes export packing, not the attachment numeric type.
		// In particular, 32-bit exports can carry raw integer material data.
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			const auto& rt = context.GetRenderTarget(slot);
			if (rt.base.addr == 0 ||
			    render_target_mask_slot(context.GetRenderTargetMask(), slot) == 0) {
				continue;
			}
			if (rt.info.channel_type == Prospero::ChannelType::kUInt) {
				pixel_info.target_uint_mask |= 1u << slot;
			} else if (rt.info.channel_type == Prospero::ChannelType::kSInt) {
				pixel_info.target_sint_mask |= 1u << slot;
			}
		}
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           ((pixel_info.target_uint_mask | pixel_info.target_sint_mask) & 1u) == 0 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
		if (pixel_info.dual_source_blending) {
			// The second source takes the numeric type of target 0.
			pixel_info.target_uint_mask =
			    (pixel_info.target_uint_mask & ~2u) | ((pixel_info.target_uint_mask & 1u) << 1u);
			pixel_info.target_sint_mask =
			    (pixel_info.target_sint_mask & ~2u) | ((pixel_info.target_sint_mask & 1u) << 1u);
		}
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	uint32_t          push_data_cursor =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount : 0;
	GraphicsPrograms  result;
	// New shaders of 3D draws (depth test on) translate on the workers while the draw is skipped;
	// UI and layer draws translate at once (TranslationWorkers).
	struct DeferScope {
		ProgramCache& cache;
		~DeferScope() { cache.SetDeferTranslation(false); }
	} defer_scope {*m_program_cache};
	m_program_cache->SetDeferTranslation(context.GetDepthControl().z_enable &&
	                                     !SmallColorTargetDraw(context));
	struct PreparedScope {
		ProgramCache& cache;
		~PreparedScope() { cache.prepared = {}; }
	} prepared_scope {*m_program_cache};
	if (pixel_active && !tess_active) {
		m_program_cache->PrepareStagesInParallel(pixel_params, pixel_info, vertex_params[0],
		                                         vertex_info[0]);
	}
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
		if (!result.vertex[i]) {
			return {};
		}
	}
	return result;
}

bool PipelineCache::IsSkipListed(uint64_t shader_addr) {
	return SkipShaderRequested(ShaderDeclaredHash(shader_addr));
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t          push_data_cursor = 0;
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	auto* pipeline = TryGetGraphicsPipeline(colors, depth, vertex_info, command, ps_input_info,
	                                        topology, primitive_restart_enable, programs, false);
	EXIT_IF(pipeline == nullptr);
	return *pipeline;
}

bool PipelineCache::FinishPending(Pipeline& pipeline) {
	auto& job = *pipeline.pending;
	if (!job.done.load(std::memory_order_acquire)) {
		if (m_compiler != nullptr && !m_compiler->Stopped()) {
			return false;
		}
		// The workers stopped before this job started: compile it here.
		job.result = CreateGraphicsPipeline(*job.build, m_driver_cache, &job.pipeline);
		Profiler::Add(Profiler::Counter::PipelinesPending, -1);
		job.done.store(true, std::memory_order_release);
	}
	EXIT_NOT_IMPLEMENTED(job.result != vk::Result::eSuccess || job.pipeline == nullptr);
	pipeline.pipeline = job.pipeline;
	pipeline.pending.reset();
	return true;
}

PipelineCache::Pipeline* PipelineCache::TryGetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs, bool may_defer) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_input_info != nullptr) {
			alpha_source = ps_input_info->alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	static_params.depth_bounds_test_enable = depth.depth_bounds_test_enable;
	static_params.depth_min_bounds         = depth.depth_min_bounds;
	static_params.depth_max_bounds         = depth.depth_max_bounds;
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);
	NormalizeGraphicsPipelineKey(static_params);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	const auto defer = [&] {
		Profiler::Add(Profiler::Counter::SkippedPipelinePending);
		Profiler::CommandZone::Annotate("skipped: pipeline compiling");
		// Diagnostics: each pipeline whose draws are skipped, once (the first 128).
		static std::unordered_set<uint64_t> logged;
		const auto pipeline_key = (vs_id << 32u) ^ ps_id;
		if (logged.size() < 128 && logged.insert(pipeline_key).second) {
			uint32_t vs_words = 0;
			for (const auto& program: programs.vertex) {
				vs_words += program.spirv_words;
			}
			LOGF("PipelineCache: draw skipped, pipeline compiling: VS 0x%016" PRIx64
			     " (%u words) PS 0x%016" PRIx64
			     " (%u words) colors=%u depth=%d/%d stencil=%d\n",
			     vertex_program.hash, vs_words, ps_active ? pixel_program.hash : 0,
			     ps_active ? pixel_program.spirv_words : 0, color_count,
			     depth.depth_test_enable ? 1 : 0, depth.depth_write_enable ? 1 : 0,
			     depth.stencil_test_enable ? 1 : 0);
		}
		if (++m_deferred_draws % 256 == 1) {
			LOGF("PipelineCache: %" PRIu64 " draws skipped while their pipeline compiled in the "
			     "background\n",
			     m_deferred_draws);
		}
		return nullptr;
	};
	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		auto& found = *iter->second;
		if (found.pending && !FinishPending(found)) {
			return defer();
		}
		return &found;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	auto build = PrepareGraphicsPipeline(m_graphics, *cached, rendering, key.vertex_input,
	                                     vertex_info, ps_input_info, programs, static_params);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);
	uint64_t vertex_words = 0;
	for (const auto& program: programs.vertex) {
		vertex_words += program.spirv_words;
	}
	const uint64_t spirv_words = vertex_words + pixel_program.spirv_words;
	// Stencil does not count: UI libraries clip their panels with stencil masks.
	const bool layer_draw = !depth.depth_test_enable && !depth.depth_write_enable &&
	                        vertex_words <= g_pipeline_sync_ui_vs_words;
	if (may_defer && spirv_words > g_pipeline_sync_words && !layer_draw && m_compiler != nullptr &&
	    !m_compiler->Stopped()) {
		// A pipeline the driver has cached finishes within the wait; a new one is compiled in
		// the background, and its draws are skipped until it is ready.
		auto job          = std::make_shared<PendingGraphicsPipeline>();
		job->build        = std::move(build);
		job->driver_cache = m_driver_cache;
		job->vs_hash      = vs_input_info.stage.program->shader_hash;
		job->ps_hash      = ps_active ? ps_input_info->stage.program->shader_hash : 0;
		cached->pending   = job;
		Profiler::Add(Profiler::Counter::PipelinesPending);
		m_compiler->Submit(job);
		(void)m_compiler->Wait(*job, g_pipeline_wait);
		auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
		EXIT_IF(!inserted);
		auto& pipeline = *iter->second;
		if (!FinishPending(pipeline)) {
			return defer();
		}
		LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);
		return &pipeline;
	}
	const auto build_begin = std::chrono::steady_clock::now();
	const auto result = CreateGraphicsPipeline(*build, m_driver_cache, &cached->pipeline);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	ReportGraphicsPipelineBuilt(vs_input_info.stage.program->shader_hash,
	                            ps_active ? ps_input_info->stage.program->shader_hash : 0,
	                            build_begin);
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);

	return iter->second.get();
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	{
		const auto hash = input_info.stage.program != nullptr ? input_info.stage.program->shader_hash : 0;
		Profiler::Phases zone;
		zone.SetText("CS 0x%016" PRIx64, hash);
		KYTY_PROFILER_PHASE(zone, "Compute pipeline build", profiler::colors::RedA100);
		const auto begin = std::chrono::steady_clock::now();
		CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache);
		ReportPipelineBuilt(Profiler::Counter::ComputePipelinesBuilt, "Compute", hash, 0, begin);
	}

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);

	return *iter->second;
}
} // namespace Libs::Graphics
