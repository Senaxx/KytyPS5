#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdlib>

namespace Libs::Graphics {

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}

	// Testing aid: KYTY_VRAM_LIMIT_MB=<n> caps every device-local heap at n MB, so a large card
	// behaves like a 4, 6 or 8 GB one: VMA reports the capped budget, allocations past it fail
	// or spill as they would there, and the caches' collection thresholds follow the budget.
	std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> heap_limits {};
	if (const char* value = std::getenv("KYTY_VRAM_LIMIT_MB"); value != nullptr) {
		const auto  limit      = static_cast<VkDeviceSize>(std::strtoull(value, nullptr, 10)) << 20u;
		const auto& properties = GetPhysicalDeviceMemoryProperties();
		if (limit != 0) {
			for (uint32_t heap = 0; heap < properties.memoryHeapCount; heap++) {
				const bool device_local = static_cast<bool>(properties.memoryHeaps[heap].flags &
				                                            vk::MemoryHeapFlagBits::eDeviceLocal);
				heap_limits[heap] = device_local ? std::min<VkDeviceSize>(
				                                       limit, properties.memoryHeaps[heap].size)
				                                 : VK_WHOLE_SIZE;
				if (device_local) {
					LOGF("KYTY_VRAM_LIMIT_MB: device heap %u capped at %" PRIu64 " MB (of %" PRIu64
					     " MB)\n",
					     heap, static_cast<uint64_t>(heap_limits[heap] >> 20u),
					     static_cast<uint64_t>(properties.memoryHeaps[heap].size >> 20u));
				}
			}
			info.pHeapSizeLimit = heap_limits.data();
		}
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}

	// A system whose commit limit is nearly used up stops the session later with "Out of memory"
	// (ISSUES #21); say so at start. KYTY_COMMIT_WARN_MB sets the threshold (default 16 GB).
	uint64_t commit_limit = 0;
	uint64_t commit_free  = 0;
	if (Common::VirtualMemory::QueryCommit(commit_limit, commit_free)) {
		const char*    value     = std::getenv("KYTY_COMMIT_WARN_MB");
		const uint64_t warn_mb   = value != nullptr ? std::strtoull(value, nullptr, 10) : 16384u;
		LOGF("Commit limit (start): %" PRIu64 " MB free of %" PRIu64 " MB\n", commit_free >> 20u,
		     commit_limit >> 20u);
		if ((commit_free >> 20u) < warn_mb) {
			std::fprintf(stderr,
			             "Warning: only %" PRIu64 " MB of the system's commit limit (%" PRIu64
			             " MB, RAM plus page file) are free; a game can need 16-24 GB. If the "
			             "emulator stops with \"Out of memory\", close other programs or enlarge "
			             "the page file.\n",
			             commit_free >> 20u, commit_limit >> 20u);
			LOGF("Warning: only %" PRIu64 " MB of the commit limit are free (threshold %" PRIu64
			     " MB)\n",
			     commit_free >> 20u, warn_mb);
		}
	}
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

// The system's commit limit (Windows: RAM + page file): an allocation fails when it is used up,
// whatever the VMA budget says (ISSUES #21: one Wolverine session commits ~23 GB).
static void LogCommit(const char* when) {
	uint64_t limit     = 0;
	uint64_t available = 0;
	if (Common::VirtualMemory::QueryCommit(limit, available)) {
		LOGF("Commit limit (%s): %" PRIu64 " MB free of %" PRIu64 " MB\n", when, available >> 20u,
		     limit >> 20u);
	}
}

// The video memory the emulator plans for, whatever the card has (KYTY_VRAM_TARGET_MB, default
// 14336 = what a 16 GB card leaves an application; 0 = the card's own budget). Senaxx, 2026-10-09:
// it must always fit on a 16 GB GPU. On cards with less the card's budget is the smaller one.
static uint64_t VramTargetBytes() {
	static const uint64_t target = [] {
		const char* value = std::getenv("KYTY_VRAM_TARGET_MB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {14336}) << 20u;
	}();
	return target;
}

uint64_t GraphicContext::MemoryShortfall() const {
	// KYTY_COMMIT_RESERVE_MB (default 0 = off): collecting video memory for a commit shortfall
	// was tried on 2026-10-09 and withdrawn. On a PC whose commit limit is short for other reasons
	// (other programs, guest memory, host heaps) it never catches up: the collectors wrote back
	// and freed ~100 images per interval from 7 GB of device use on, the frame rate fell to 7 fps
	// and glyph atlases lost letters. The limit is also dynamic on a system-managed page file. The
	// start-up warning and the log lines remain; the video memory target is the pressure that counts.
	static const uint64_t reserve = [] {
		const char* value = std::getenv("KYTY_COMMIT_RESERVE_MB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {0}) << 20u;
	}();
	static std::atomic<int64_t>  next_ms {0};
	static std::atomic<uint64_t> shortfall {0};
	const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
	                     std::chrono::steady_clock::now().time_since_epoch())
	                     .count();
	auto next = next_ms.load(std::memory_order_relaxed);
	if (now >= next && next_ms.compare_exchange_strong(next, now + 250, std::memory_order_relaxed)) {
		// The system's commit limit below the reserve.
		uint64_t limit     = 0;
		uint64_t available = 0;
		const uint64_t commit =
		    reserve != 0 && Common::VirtualMemory::QueryCommit(limit, available) &&
		            available < reserve
		        ? reserve - available
		        : 0;
		// Everything the device holds (images, buffers, the driver's own) past 90 % of the
		// budget, which the video memory target caps: the caches' own thresholds leave out the
		// bindless heap textures and render targets, so they alone cannot keep the total in it.
		uint64_t device = 0;
		if (CanReportMemoryUsage()) {
			const auto budget = GetTotalMemoryBudget();
			const auto usage  = GetDeviceMemoryUsage();
			const auto mark   = budget / 10 * 9;
			device            = budget != 0 && usage > mark ? usage - mark : 0;
		}
		const uint64_t value    = std::max(commit, device);
		const auto     previous = shortfall.exchange(value, std::memory_order_relaxed);
		static std::atomic<uint32_t> reported {0};
		if ((previous == 0) != (value == 0) && reported.fetch_add(1) < 64) {
			LOGF("Memory: %s (commit %" PRIu64 " MB free of %" PRIu64 " MB, reserve %" PRIu64
			     " MB; device %" PRIu64 " MB of %" PRIu64 " MB budget)\n",
			     value != 0 ? "short, the caches collect" : "room again", available >> 20u,
			     limit >> 20u, reserve >> 20u, GetDeviceMemoryUsage() >> 20u,
			     GetTotalMemoryBudget() >> 20u);
		}
	}
	return shortfall.load(std::memory_order_relaxed);
}

void GraphicContext::LogMemoryBudget() const {
	LogCommit("now");
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		const auto own = budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
		const auto target = VramTargetBytes();
		return target != 0 ? std::min(own, target) : own;
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	vk::Image::CType native_image = VK_NULL_HANDLE;
	auto             result       = static_cast<vk::Result>(
	    vmaCreateImage(allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
	                   &alloc_info, &native_image, &image.allocation, nullptr));
	if (result != vk::Result::eSuccess) {
		// The device heap is full. The guest has a unified 16 GB pool, so the image is
		// real; place it in host memory and pay in bandwidth rather than fail.
		alloc_info.requiredFlags  = 0;
		alloc_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
		result                    = static_cast<vk::Result>(vmaCreateImage(
            allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
            &alloc_info, &native_image, &image.allocation, nullptr));
		static std::atomic<uint32_t> spill_count {0};
		if (const auto seen = spill_count.fetch_add(1, std::memory_order_relaxed); seen < 32) {
			LOGF("Image spilled to host memory (%u): %ux%ux%u layers=%u levels=%u format=%d -> %s\n",
			     seen + 1, image_info.extent.width, image_info.extent.height,
			     image_info.extent.depth, image_info.arrayLayers, image_info.mipLevels,
			     static_cast<int>(image_info.format), vk::to_string(result).c_str());
		}
	}
	image.image = native_image;
	if (result != vk::Result::eSuccess) {
		LogMemoryBudget();
		return false;
	}

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	vmaDestroyImage(allocator, image.image, image.allocation);
	image.image      = nullptr;
	image.allocation = nullptr;
}

} // namespace Libs::Graphics
