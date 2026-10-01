// Checks on the GPU that the float min/max the SPIR-V emitter produces with float controls on
// (NMin/NMax plus the signed-zero rule, spirvEmitterAluHelpers.cpp) and its NaN test (OpIsNan)
// give the guest's results: IEEE minNum/maxNum as V_MIN/V_MAX_F32, -0 below +0.
//
// usage: float_minmax_check <float_minmax_check.spv>
//   (spirv-as --target-env vulkan1.3 tests/data/float_minmax_check.spvasm -o <file>.spv)
//
// Every pair of special values (zeros, ones, infinities, NaNs, denormals, extremes) and random
// pairs are compared bit for bit with the reference below, which is what the emitter's bitwise
// form computes. A tool, not a test: it needs a GPU.

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

#define VK_FUNCTIONS(X)                                                                            \
	X(vkCreateInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties)             \
	X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties)             \
	X(vkCreateDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkGetBufferMemoryRequirements)       \
	X(vkAllocateMemory) X(vkBindBufferMemory) X(vkMapMemory) X(vkCreateShaderModule)               \
	X(vkCreateDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkCreateComputePipelines)           \
	X(vkCreateDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets)                \
	X(vkCreateCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer)                     \
	X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdDispatch) X(vkEndCommandBuffer)         \
	X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkDeviceWaitIdle) X(vkDestroyDevice) X(vkDestroyInstance)

#define DECLARE(name) PFN_##name name##_ = nullptr;
VK_FUNCTIONS(DECLARE)
#undef DECLARE
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_ = nullptr;

bool Check(VkResult result, const char* what) {
	if (result != VK_SUCCESS) {
		std::fprintf(stderr, "%s failed: %d\n", what, static_cast<int>(result));
		return false;
	}
	return true;
}

// The guest's V_MAX/V_MIN_F32 on bits, as the emitter's bitwise form computes them.
bool IsNan(uint32_t bits) {
	return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u;
}

uint32_t Reference(uint32_t a, uint32_t b, bool max_value) {
	if (IsNan(a)) {
		return b;
	}
	if (IsNan(b)) {
		return a;
	}
	if (((a | b) & 0x7fffffffu) == 0u) {
		return max_value ? (a & b) : (a | b);
	}
	const float fa = std::bit_cast<float>(a);
	const float fb = std::bit_cast<float>(b);
	return max_value ? (fa >= fb ? a : b) : (fa < fb ? a : b);
}

} // namespace

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: float_minmax_check <float_minmax_check.spv>\n");
		return 1;
	}
	std::ifstream file(argv[1], std::ios::binary);
	std::vector<char> code((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	if (code.empty() || code.size() % 4 != 0) {
		std::fprintf(stderr, "cannot read %s\n", argv[1]);
		return 1;
	}

#if defined(_WIN32)
	HMODULE library = LoadLibraryA("vulkan-1.dll");
	vkGetInstanceProcAddr_ =
	    library ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(library, "vkGetInstanceProcAddr"))
	            : nullptr;
#else
	void* library = dlopen("libvulkan.so.1", RTLD_NOW);
	vkGetInstanceProcAddr_ =
	    library ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"))
	            : nullptr;
#endif
	if (vkGetInstanceProcAddr_ == nullptr) {
		std::fprintf(stderr, "no Vulkan loader\n");
		return 1;
	}
	vkCreateInstance_ =
	    reinterpret_cast<PFN_vkCreateInstance>(vkGetInstanceProcAddr_(nullptr, "vkCreateInstance"));
	VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.apiVersion = VK_API_VERSION_1_3;
	VkInstanceCreateInfo instance_info {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	instance_info.pApplicationInfo = &app;
	VkInstance instance = VK_NULL_HANDLE;
	if (!Check(vkCreateInstance_(&instance_info, nullptr, &instance), "vkCreateInstance")) {
		return 1;
	}
#define LOAD(name) name##_ = reinterpret_cast<PFN_##name>(vkGetInstanceProcAddr_(instance, #name));
	VK_FUNCTIONS(LOAD)
#undef LOAD

	uint32_t count = 0;
	vkEnumeratePhysicalDevices_(instance, &count, nullptr);
	std::vector<VkPhysicalDevice> devices(count);
	vkEnumeratePhysicalDevices_(instance, &count, devices.data());
	VkPhysicalDevice physical = VK_NULL_HANDLE;
	VkPhysicalDeviceProperties properties {};
	for (auto device: devices) {
		vkGetPhysicalDeviceProperties_(device, &properties);
		physical = device;
		if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			break;
		}
	}
	std::printf("%s\n", properties.deviceName);

	uint32_t family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties_(physical, &family_count, nullptr);
	std::vector<VkQueueFamilyProperties> families(family_count);
	vkGetPhysicalDeviceQueueFamilyProperties_(physical, &family_count, families.data());
	uint32_t family = 0;
	while (family < family_count && (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
		family++;
	}
	float priority = 1.0f;
	VkDeviceQueueCreateInfo queue_info {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	queue_info.queueFamilyIndex = family;
	queue_info.queueCount       = 1;
	queue_info.pQueuePriorities = &priority;
	VkDeviceCreateInfo device_info {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	device_info.queueCreateInfoCount = 1;
	device_info.pQueueCreateInfos    = &queue_info;
	VkDevice device = VK_NULL_HANDLE;
	if (!Check(vkCreateDevice_(physical, &device_info, nullptr, &device), "vkCreateDevice")) {
		return 1;
	}
	VkQueue queue = VK_NULL_HANDLE;
	vkGetDeviceQueue_(device, family, 0, &queue);

	// Inputs: every pair of special values, then random bit patterns.
	const std::vector<uint32_t> specials = {
	    0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x7f800000u, 0xff800000u,
	    0x7fc00000u, 0xffc00000u, 0x7f800001u, 0xff800001u, 0x00000001u, 0x80000001u,
	    0x007fffffu, 0x00800000u, 0x7f7fffffu, 0xff7fffffu};
	std::vector<std::pair<uint32_t, uint32_t>> pairs;
	for (auto a: specials) {
		for (auto b: specials) {
			pairs.emplace_back(a, b);
		}
	}
	std::mt19937 random(20261001u);
	for (int i = 0; i < 4000; i++) {
		pairs.emplace_back(random(), random());
	}
	while (pairs.size() % 64 != 0) {
		pairs.emplace_back(0u, 0u);
	}

	const VkDeviceSize size = pairs.size() * 6 * sizeof(uint32_t);
	VkBufferCreateInfo buffer_info {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer_info.size  = size;
	buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	VkBuffer buffer   = VK_NULL_HANDLE;
	Check(vkCreateBuffer_(device, &buffer_info, nullptr, &buffer), "vkCreateBuffer");
	VkMemoryRequirements requirements {};
	vkGetBufferMemoryRequirements_(device, buffer, &requirements);
	VkPhysicalDeviceMemoryProperties memory {};
	vkGetPhysicalDeviceMemoryProperties_(physical, &memory);
	uint32_t type = 0;
	const auto wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	while (type < memory.memoryTypeCount &&
	       (!(requirements.memoryTypeBits & (1u << type)) ||
	        (memory.memoryTypes[type].propertyFlags & wanted) != wanted)) {
		type++;
	}
	VkMemoryAllocateInfo allocate {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	allocate.allocationSize  = requirements.size;
	allocate.memoryTypeIndex = type;
	VkDeviceMemory device_memory = VK_NULL_HANDLE;
	Check(vkAllocateMemory_(device, &allocate, nullptr, &device_memory), "vkAllocateMemory");
	Check(vkBindBufferMemory_(device, buffer, device_memory, 0), "vkBindBufferMemory");
	uint32_t* words = nullptr;
	Check(vkMapMemory_(device, device_memory, 0, size, 0, reinterpret_cast<void**>(&words)), "vkMapMemory");
	for (size_t i = 0; i < pairs.size(); i++) {
		words[i * 6 + 0] = pairs[i].first;
		words[i * 6 + 1] = pairs[i].second;
		words[i * 6 + 2] = words[i * 6 + 3] = words[i * 6 + 4] = 0xdeadbeefu;
	}

	VkShaderModuleCreateInfo module_info {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	module_info.codeSize = code.size();
	module_info.pCode    = reinterpret_cast<const uint32_t*>(code.data());
	VkShaderModule module = VK_NULL_HANDLE;
	Check(vkCreateShaderModule_(device, &module_info, nullptr, &module), "vkCreateShaderModule");
	VkDescriptorSetLayoutBinding binding {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
	                                      VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
	VkDescriptorSetLayoutCreateInfo set_info {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	set_info.bindingCount = 1;
	set_info.pBindings    = &binding;
	VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
	Check(vkCreateDescriptorSetLayout_(device, &set_info, nullptr, &set_layout), "set layout");
	VkPipelineLayoutCreateInfo layout_info {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	layout_info.setLayoutCount = 1;
	layout_info.pSetLayouts    = &set_layout;
	VkPipelineLayout layout    = VK_NULL_HANDLE;
	Check(vkCreatePipelineLayout_(device, &layout_info, nullptr, &layout), "pipeline layout");
	VkComputePipelineCreateInfo pipeline_info {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
	pipeline_info.stage  = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
	pipeline_info.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = layout;
	VkPipeline pipeline        = VK_NULL_HANDLE;
	if (!Check(vkCreateComputePipelines_(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline),
	           "vkCreateComputePipelines")) {
		return 1;
	}
	VkDescriptorPoolSize pool_size {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
	VkDescriptorPoolCreateInfo pool_info {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
	pool_info.maxSets       = 1;
	pool_info.poolSizeCount = 1;
	pool_info.pPoolSizes    = &pool_size;
	VkDescriptorPool pool   = VK_NULL_HANDLE;
	Check(vkCreateDescriptorPool_(device, &pool_info, nullptr, &pool), "descriptor pool");
	VkDescriptorSetAllocateInfo set_allocate {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
	set_allocate.descriptorPool     = pool;
	set_allocate.descriptorSetCount = 1;
	set_allocate.pSetLayouts        = &set_layout;
	VkDescriptorSet set             = VK_NULL_HANDLE;
	Check(vkAllocateDescriptorSets_(device, &set_allocate, &set), "descriptor set");
	VkDescriptorBufferInfo described {buffer, 0, VK_WHOLE_SIZE};
	VkWriteDescriptorSet write {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
	write.dstSet          = set;
	write.descriptorCount = 1;
	write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	write.pBufferInfo     = &described;
	vkUpdateDescriptorSets_(device, 1, &write, 0, nullptr);

	VkCommandPoolCreateInfo command_pool_info {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	command_pool_info.queueFamilyIndex = family;
	VkCommandPool command_pool         = VK_NULL_HANDLE;
	Check(vkCreateCommandPool_(device, &command_pool_info, nullptr, &command_pool), "command pool");
	VkCommandBufferAllocateInfo command_allocate {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	command_allocate.commandPool        = command_pool;
	command_allocate.commandBufferCount = 1;
	VkCommandBuffer command             = VK_NULL_HANDLE;
	Check(vkAllocateCommandBuffers_(device, &command_allocate, &command), "command buffer");
	VkCommandBufferBeginInfo begin {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	vkBeginCommandBuffer_(command, &begin);
	vkCmdBindPipeline_(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets_(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
	vkCmdDispatch_(command, static_cast<uint32_t>(pairs.size() / 64), 1, 1);
	vkEndCommandBuffer_(command);
	VkSubmitInfo submit {VK_STRUCTURE_TYPE_SUBMIT_INFO};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	Check(vkQueueSubmit_(queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
	vkQueueWaitIdle_(queue);

	// Results. Two NaN operands may give any NaN; everything else must match bit for bit.
	int failures = 0;
	int checked  = 0;
	for (size_t i = 0; i < pairs.size(); i++) {
		const auto [a, b] = pairs[i];
		const uint32_t got[3]  = {words[i * 6 + 2], words[i * 6 + 3], words[i * 6 + 4]};
		const uint32_t want[3] = {Reference(a, b, true), Reference(a, b, false), IsNan(a) ? 1u : 0u};
		for (int k = 0; k < 3; k++) {
			checked++;
			const bool both_nan = k < 2 && IsNan(a) && IsNan(b);
			const bool ok       = both_nan ? IsNan(got[k]) : got[k] == want[k];
			if (!ok) {
				if (failures++ < 30) {
					std::printf("  %s(0x%08x, 0x%08x) = 0x%08x, expected 0x%08x\n",
					            k == 0 ? "max" : (k == 1 ? "min" : "isnan"), a, b, got[k], want[k]);
				}
			}
		}
	}
	std::printf("%d of %d results differ\n", failures, checked);
	vkDeviceWaitIdle_(device);
	vkDestroyDevice_(device, nullptr);
	vkDestroyInstance_(instance, nullptr);
	return failures == 0 ? 0 : 1;
}
