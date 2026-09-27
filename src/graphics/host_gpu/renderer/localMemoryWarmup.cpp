#include "common/assert.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "gpu_tiler_shaders/local_memory_warmup_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics {

// KYTY_LOCAL_MEMORY_WARMUP=<bytes per thread>: before any guest work, run one dispatch that needs
// that much local memory per thread, so the driver sizes its local-memory area up front instead of
// growing it while guest command buffers are in flight. A diagnostic for device losses that fault
// on writes above every allocation Kyty made.
void WarmUpLocalMemory(GraphicContext& graphics, CommandScheduler& /*scheduler*/) {
	const char* value = std::getenv("KYTY_LOCAL_MEMORY_WARMUP");
	if (value == nullptr) {
		return;
	}
	const auto bytes = std::strtoul(value, nullptr, 0);
	if (bytes < 4) {
		return;
	}
	const uint32_t array_size = static_cast<uint32_t>(bytes / 4u);

	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eStorageBuffer, 1,
	                                              vk::ShaderStageFlagBits::eCompute, nullptr};
	vk::DescriptorSetLayoutCreateInfo    layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = 1;
	layout_info.pBindings    = &binding;
	vk::DescriptorSetLayout set_layout {};
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &set_layout),
	                     "create warm-up descriptor layout");

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount = 1;
	pipeline_layout_info.pSetLayouts    = &set_layout;
	vk::PipelineLayout pipeline_layout {};
	RequireVulkanSuccess(
	    graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr, &pipeline_layout),
	    "create warm-up pipeline layout");

	const vk::SpecializationMapEntry entry {0, 0, sizeof(uint32_t)};
	vk::SpecializationInfo           specialization {};
	specialization.mapEntryCount = 1;
	specialization.pMapEntries   = &entry;
	specialization.dataSize      = sizeof(array_size);
	specialization.pData         = &array_size;

	const auto                        module = CompileSPV(LOCAL_MEMORY_WARMUP_SPV, graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage               = vk::ShaderStageFlagBits::eCompute;
	stage.module              = module;
	stage.pName               = "main";
	stage.pSpecializationInfo = &specialization;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = pipeline_layout;
	vk::Pipeline pipeline {};
	const auto   result =
	    graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline);
	graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create warm-up pipeline");

	// This runs before the guest GPU thread exists, so it records and submits on its own instead of
	// going through the command scheduler.
	constexpr vk::DeviceSize OutputSize = 64 * sizeof(uint32_t);
	vk::BufferCreateInfo     buffer_info {};
	buffer_info.size  = OutputSize;
	buffer_info.usage = vk::BufferUsageFlagBits::eStorageBuffer;
	vk::Buffer output {};
	RequireVulkanSuccess(graphics.device.createBuffer(&buffer_info, nullptr, &output),
	                     "create warm-up buffer");
	vk::MemoryRequirements requirements {};
	graphics.device.getBufferMemoryRequirements(output, &requirements);
	uint32_t memory_type = UINT32_MAX;
	for (uint32_t i = 0; i < graphics.physical_device_memory_properties.memoryTypeCount; i++) {
		if ((requirements.memoryTypeBits & (1u << i)) != 0 &&
		    (graphics.physical_device_memory_properties.memoryTypes[i].propertyFlags &
		     vk::MemoryPropertyFlagBits::eDeviceLocal)) {
			memory_type = i;
			break;
		}
	}
	EXIT_IF(memory_type == UINT32_MAX);
	vk::MemoryAllocateInfo allocate_info {};
	allocate_info.allocationSize  = requirements.size;
	allocate_info.memoryTypeIndex = memory_type;
	vk::DeviceMemory memory {};
	RequireVulkanSuccess(graphics.device.allocateMemory(&allocate_info, nullptr, &memory),
	                     "allocate warm-up memory");
	graphics.device.bindBufferMemory(output, memory, 0);

	vk::CommandPoolCreateInfo pool_info {};
	pool_info.queueFamilyIndex = graphics.queue_family;
	vk::CommandPool pool {};
	RequireVulkanSuccess(graphics.device.createCommandPool(&pool_info, nullptr, &pool),
	                     "create warm-up command pool");
	vk::CommandBufferAllocateInfo command_info {};
	command_info.commandPool        = pool;
	command_info.level              = vk::CommandBufferLevel::ePrimary;
	command_info.commandBufferCount = 1;
	vk::CommandBuffer command {};
	RequireVulkanSuccess(graphics.device.allocateCommandBuffers(&command_info, &command),
	                     "allocate warm-up command buffer");

	const vk::DescriptorBufferInfo info {output, 0, OutputSize};
	vk::WriteDescriptorSet         write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &info;

	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	RequireVulkanSuccess(command.begin(&begin), "begin warm-up command buffer");
	command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, 1, &write);
	command.dispatch(4096, 1, 1);
	command.end();

	vk::FenceCreateInfo fence_info {};
	vk::Fence           fence {};
	RequireVulkanSuccess(graphics.device.createFence(&fence_info, nullptr, &fence),
	                     "create warm-up fence");
	vk::SubmitInfo submit {};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		RequireVulkanSuccess(graphics.queue.submit(1, &submit, fence), "submit warm-up");
	}
	RequireVulkanSuccess(graphics.device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX),
	                     "wait for warm-up");

	graphics.device.destroyFence(fence, nullptr);
	graphics.device.destroyCommandPool(pool, nullptr);
	graphics.device.destroyBuffer(output, nullptr);
	graphics.device.freeMemory(memory, nullptr);
	graphics.device.destroyPipeline(pipeline, nullptr);
	graphics.device.destroyPipelineLayout(pipeline_layout, nullptr);
	graphics.device.destroyDescriptorSetLayout(set_layout, nullptr);
	LOGF("Local memory warm-up: %u bytes per thread\n", array_size * 4u);
}

} // namespace Libs::Graphics
