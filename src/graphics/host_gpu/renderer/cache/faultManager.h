#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_

#include "common/abi.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics {

class BufferCache;

// Appended after the page-fault bitset. The first invocation claiming the record
// writes its payload; the host reads it only after the submission completes.
struct ShaderTrapRecord {
	uint32_t claimed          = 0;
	uint32_t shader_hash_low  = 0;
	uint32_t shader_hash_high = 0;
	uint32_t pc               = 0;
	uint32_t code             = 0;
	uint32_t reserved[3] {};
};
static_assert(sizeof(ShaderTrapRecord) == 32);

struct ShaderFaultReport {
	uint64_t         page_count = 0;
	ShaderTrapRecord trap;
};

class FaultManager {
	static constexpr size_t MaxPendingFaults = 8;

public:
	FaultManager(GraphicContext& graphics, CommandScheduler& scheduler, BufferCache& buffer_cache);
	~FaultManager();
	KYTY_CLASS_NO_COPY(FaultManager);

	[[nodiscard]] Buffer*           GetFaultBuffer() noexcept;
	void                            ProcessFaultBuffer();
	[[nodiscard]] ShaderFaultReport CollectFaults();

private:
	void                                   ProcessFaultBufferImpl(ShaderFaultReport* report);
	GraphicContext&                        m_graphics;
	CommandScheduler&                      m_scheduler;
	BufferCache&                           m_buffer_cache;
	size_t                                 m_download_area_size;
	Buffer                                 m_fault_buffer;
	Buffer                                 m_download_buffer;
	std::array<uint64_t, MaxPendingFaults> m_fault_areas {};
	uint32_t                               m_current_area                  = 0;
	bool                                   m_initialized                   = false;
	vk::DescriptorSetLayout                m_fault_process_desc_layout     = nullptr;
	vk::Pipeline                           m_fault_process_pipeline        = nullptr;
	vk::PipelineLayout                     m_fault_process_pipeline_layout = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_
