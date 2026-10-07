#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/timer.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/timeline.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "kernel/memory.h"

#include <algorithm>
#include <fmt/format.h>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		buffer.CopyFrom(m_scheduler.Current(), m_staging_buffer, offset, buffer.Offset(address),
		                chunk, vk::AccessFlagBits::eHostWrite);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		g_cpu_dirty_epoch.fetch_add(1, std::memory_order_release);
		buffer.lru_id = m_lru_cache.Insert(id, LruClock());
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, LruClock());
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	Unregister(id);
	// As TextureCache::DeleteImage: earlier command buffers may still use the buffer on the GPU.
	m_scheduler.DeferRelease([this, id] { ReleaseBuffer(id); });
}

void BufferCache::ReleaseBuffer(BufferId id) {
	const auto* buffer = m_slot_buffers.try_get(id);
	if (buffer != nullptr && !m_scheduler.IsFree(buffer->last_use_tick)) {
		// Bound after it was dropped, in a tick the GPU has not finished: keep it until then.
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 16) {
			LOGF("BufferCache: buffer guest=0x%016" PRIx64 "+0x%" PRIx64 " used in tick %" PRIu64
			     " after its release was queued; destruction waits for that tick\n",
			     buffer->CpuAddress(), buffer->Size(), buffer->last_use_tick);
		}
		m_scheduler.DeferRelease([this, id] { ReleaseBuffer(id); });
		return;
	}
	m_slot_buffers.erase(id);
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	// One reservation cannot exceed the download ring, so a larger range goes in ring-sized
	// windows; a window that does not fit drains the ring before it is mapped.
	const auto capacity = m_download_buffer.Size();
	bool       any      = false;
	for (uint64_t offset = 0; offset < size; offset += capacity) {
		any |= DownloadBufferWindow<async>(buffer, vaddr + offset, std::min(capacity, size - offset));
	}
	return any;
}

template <bool async>
bool BufferCache::DownloadBufferWindow(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    std::unique_lock lock(m_dirty_ranges_mutex);
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
			    m_downloading_ranges.Add(start, end - start);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}
	const auto capacity = m_download_buffer.Size();
	for (size_t first = 0; first < copies.size();) {
		const auto base       = copies[first].dstOffset;
		auto       last       = first;
		uint64_t   batch_size = 0;
		while (last < copies.size()) {
			const auto end = copies[last].dstOffset - base + Common::AlignUp(copies[last].size, 64);
			if (end > capacity) {
				break;
			}
			batch_size = end;
			last++;
		}
		EXIT_IF(last == first);
		std::vector<vk::BufferCopy> batch(copies.begin() + static_cast<std::ptrdiff_t>(first),
		                                  copies.begin() + static_cast<std::ptrdiff_t>(last));
		for (auto& copy: batch) {
			copy.dstOffset -= base;
		}
		DownloadBufferCopies<async>(buffer, std::move(batch), batch_size);
		first = last;
	}
	return true;
}

template <bool async>
void BufferCache::DownloadBufferCopies(Buffer& buffer, std::vector<vk::BufferCopy> copies,
                                       uint64_t total_size) {
	Profiler::Add(Profiler::Counter::Readbacks);
	Profiler::Add(Profiler::Counter::ReadbackBytes, static_cast<int64_t>(total_size));
	const auto buffer_address = buffer.CpuAddress();
	auto [mapped, offset]     = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	const auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	// Recorded through the recording thread: a raw Handle() made every readback wait for it to
	// drain first (4.5 % of the GPU thread at the Leap Attack prompt, 2026-10-05).
	const auto              native = command.Recorder();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	auto publish = [this, mapped, offset, total_size, buffer_address,
	                copies = std::move(copies), owner = std::move(temporary)] {
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
		std::unique_lock lock(m_dirty_ranges_mutex);
		for (const auto& copy: copies) {
			m_downloading_ranges.Subtract(buffer_address + copy.srcOffset, copy.size);
		}
	};
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish));
	} else {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		publish();
	}
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
}

bool BufferCache::WriteClean(uint64_t vaddr, const void* data, uint64_t size) {
	if (!GuestGpu::IsGpuThread() || size == 0 || !GuestRange {vaddr, size}.Valid() ||
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	{
		// A download publishes the GPU's older value of these bytes when it lands.
		std::shared_lock lock(m_dirty_ranges_mutex);
		if (m_downloading_ranges.Intersects(vaddr, size)) {
			return false;
		}
	}

	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner || !m_slot_buffers[*owner].IsInBounds(vaddr, size) ||
	    !Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		return false;
	}
	WriteDataBuffer(m_slot_buffers[*owner], vaddr, data, size);
	if (HasGpuDirtyBytes(vaddr, size)) {
		// Overwritten in full: guest memory holds the value the GPU copy will have.
		std::unique_lock lock(m_dirty_ranges_mutex);
		m_gpu_modified_ranges.Subtract(vaddr, size);
	}
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

// A game thread's fault on GPU-written memory no longer makes the GPU thread wait for the GPU.
// The GPU thread records the download, submits it and goes on; the faulting thread waits for
// that download's publication itself, then asks again (the window's pages then have no
// GPU-written bytes left and their protection is lifted). KYTY_ASYNC_READBACK=0 waits on the GPU
// thread as before.
static const bool g_async_readback = [] {
	const char* value = std::getenv("KYTY_ASYNC_READBACK");
	return value == nullptr || value[0] != '0';
}();

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	KYTY_PROFILER_FUNCTION();
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const bool from_gpu_thread = GuestGpu::IsGpuThread();
	const bool async           = g_async_readback && !from_gpu_thread;
	for (;;) {
		uint64_t wait_tick = 0;
		m_scheduler.Context().GetGpu().SendCommandSync(
		    [this, vaddr, size, is_write, from_gpu_thread, async, &wait_tick] {
			    wait_tick = ReadMemoryStep(vaddr, size, is_write, from_gpu_thread, async);
		    });
		if (wait_tick == 0) {
			return;
		}
		m_scheduler.WaitPriorityOperations(wait_tick);
	}
}

// Runs on the GPU thread. Returns 0 when the access may proceed, or (asynchronous readback only)
// the tick whose download publication the caller must wait for before asking again.
// The readback window (KYTY_READBACK_WINDOW_KB, default 512; 0 reads only the requested pages,
// as upstream does).
static uint64_t ReadbackWindow() {
	static const uint64_t window = [] {
		const char* value = std::getenv("KYTY_READBACK_WINDOW_KB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {512}) * 1024u;
	}();
	return window;
}

static bool ReadbackRetick() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_READBACK_RETICK");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

uint64_t BufferCache::ReadMemoryStep(uint64_t vaddr, uint64_t size, bool is_write,
                                     bool from_gpu_thread, bool async) {
	if (is_write && !IsRegionRegistered(vaddr, size)) {
		return 0;
	}
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];

	// Diagnostics: whether the bytes the faulting access touches were written by the GPU
	// (true sharing) or only share the page with bytes that were (false sharing).
	{
		static uint64_t exact = 0;
		static uint64_t shared = 0;
		const bool      dirty  = HasGpuDirtyBytes(vaddr, std::max<uint64_t>(size, 4));
		(dirty ? exact : shared)++;
		static uint32_t logged = 0;
		if (!dirty && logged < 32) {
			logged++;
			// Which bytes of the 4 KiB page the GPU did write, in 16-byte steps.
			std::string dirty_ranges;
			const auto  page  = vaddr & ~uint64_t {0xfff};
			uint64_t    begin = 0;
			bool        open  = false;
			for (uint64_t at = page; at <= page + 0x1000; at += 16) {
				const bool here = at < page + 0x1000 && HasGpuDirtyBytes(at, 16);
				if (here && !open) {
					begin = at;
					open  = true;
				} else if (!here && open) {
					dirty_ranges += fmt::format(" {:x}-{:x}", begin - page, at - page);
					open = false;
				}
			}
			LOGF("ReadMemory false sharing: vaddr=0x%016" PRIx64 " write=%d page dirty:%s\n",
			     vaddr, is_write ? 1 : 0, dirty_ranges.c_str());
		}
		if ((exact + shared) % 1024 == 0) {
			LOGF("ReadMemory faults: gpu_written=%" PRIu64 " false_sharing=%" PRIu64 "\n", exact,
			     shared);
		}
	}

	// The page is protected as GPU-written, but none of its bytes are waiting for a download:
	// the bytes the GPU wrote are elsewhere on the page, or were downloaded with another
	// page. The page is current, so lift its protection. Downloading the window instead drained
	// the GPU (~30 ms a fault) for guest reads of structs next to the command processor's
	// labels.
	const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	if (m_memory_tracker.IsRegionGpuModified(page_begin, page_end - page_begin) &&
	    !HasGpuDirtyBytes(page_begin, page_end - page_begin)) {
		// An asynchronous readback of these bytes may not be in guest memory yet: the access
		// must not see the older bytes, so it waits for that publication first.
		bool downloading = false;
		{
			std::shared_lock lock(m_dirty_ranges_mutex);
			downloading = m_downloading_ranges.Intersects(page_begin, page_end - page_begin);
		}
		if (downloading && m_last_async_download_tick != 0) {
			if (async) {
				return m_last_async_download_tick;
			}
			m_scheduler.WaitPriorityOperations(m_last_async_download_tick);
		}
		if (async && ReadbackWindow() != 0) {
			// The whole download window, as the synchronous path lifts it: page by page, each
			// page of the window faulted on its own and went through the GPU thread again.
			const auto window = ReadbackWindow();
			const auto begin  = std::max(Common::AlignDown(vaddr, window), buffer.CpuAddress());
			const auto end    = std::min(begin + window, buffer.CpuAddress() + buffer.Size());
			std::vector<uint64_t> current;
			{
				std::shared_lock lock(m_dirty_ranges_mutex);
				for (auto page = Common::AlignDown(begin, TRACKER_PAGE_SIZE); page < end;
				     page += TRACKER_PAGE_SIZE) {
					if (!m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE) &&
					    !m_downloading_ranges.Intersects(page, TRACKER_PAGE_SIZE)) {
						current.push_back(page);
					}
				}
			}
			// One lift per run of neighbouring pages: each lift is a protection change, and
			// page by page they were most of the GPU thread's VirtualProtect calls.
			for (size_t first = 0; first < current.size();) {
				size_t last = first + 1;
				while (last < current.size() &&
				       current[last] == current[last - 1] + TRACKER_PAGE_SIZE) {
					last++;
				}
				const auto run_begin = current[first];
				const auto run_size  = current[last - 1] + TRACKER_PAGE_SIZE - run_begin;
				if (m_memory_tracker.IsRegionGpuModified(run_begin, run_size)) {
					m_memory_tracker.UnmarkRegionAsGpuModified(run_begin, run_size);
				}
				first = last;
			}
		}
		m_memory_tracker.UnmarkRegionAsGpuModified(page_begin, page_end - page_begin);
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
		static uint64_t lifted = 0;
		if (++lifted % 1024 == 1) {
			LOGF("ReadMemory: lifted %" PRIu64 " protections of pages without GPU-written "
			     "bytes\n",
			     lifted);
		}
		return 0;
	}

	// Diagnostics: the GPU-written ranges (whole) on the faulting page of the first downloads.
	{
		static uint32_t logged = 0;
		if (logged < 96) {
			logged++;
			std::string ranges;
			uint32_t    count = 0;
			m_gpu_modified_ranges.ForEachOverlapping(
			    page_begin, page_end - page_begin, [&](uint64_t begin, uint64_t end) {
				    if (count++ < 4) {
					    ranges += fmt::format(" {:x}+{:x}", begin, end - begin);
				    }
			    });
			LOGF("ReadMemory download: vaddr=0x%016" PRIx64
			     " write=%d gpu_thread=%d ranges=%u:%s\n",
			     vaddr, is_write ? 1 : 0, from_gpu_thread ? 1 : 0, count, ranges.c_str());
		}
	}

	// Widen nearby CPU reads so they share one GPU drain. Upstream (0ce84e59) reads only the
	// requested pages; Wolverine's game threads read ~2,500 GPU-written pages a frame at the Leap
	// Attack prompt, each its own fault and download that way (7.5 fps against 20.3).
	auto window_begin = vaddr;
	auto window_end   = vaddr + size;
	if (const auto window = ReadbackWindow(); window != 0) {
		const auto buffer_begin = buffer.CpuAddress();
		const auto buffer_end   = buffer_begin + buffer.Size();
		window_begin = std::max(Common::AlignDown(vaddr, window), buffer_begin);
		window_end   = std::min(std::max(window_begin + window, vaddr + size), buffer_end);
	}

	Timeline::Mark("readmem", vaddr, from_gpu_thread ? 1u : 0u);
	if (async) {
		// The download publishes when its tick completes; the page stays protected until the
		// caller's next step finds its bytes downloaded and lifts it.
		auto tick = m_scheduler.CurrentTick();
		if (DownloadBufferMemory<true>(buffer, window_begin, window_end - window_begin)) {
			// The tick of the last copy, not of the first: a download ring that wraps within this
			// recording submits it (StreamBuffer::WaitPendingOperations), and the later copies
			// and their publication then land in the next tick. A caller waiting for the earlier
			// tick went on before its bytes were published, and the GPU thread then lifted the
			// page's protection over the older bytes (Jetsku bf8d6148, same idea for writers).
			// KYTY_READBACK_RETICK=0 keeps the tick from before the download.
			if (ReadbackRetick()) {
				tick = m_scheduler.CurrentTick();
			}
			m_scheduler.Flush();
			m_last_async_download_tick = tick;
			return tick;
		}
	} else if (DownloadBufferMemory<false>(buffer, window_begin, window_end - window_begin)) {
		m_memory_tracker.UnmarkRegionAsGpuModified(window_begin, window_end - window_begin);
	}
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
	return 0;
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	KYTY_PROFILER_FUNCTION();
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	if (vaddr < CACHING_PAGESIZE) {
		// Guest page 0 is never mapped; a request here is a garbage or null descriptor that
		// slipped past the null checks. The memory tracker cannot hold address 0, so name the
		// caller now rather than in the garbage collector later.
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 8) {
			LOGF("BufferCache: buffer requested in guest page 0: vaddr=0x%016" PRIx64
			     " size=0x%016" PRIx64 "\n%s",
			     vaddr, size, Common::HostBacktrace().c_str());
		}
	}
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	// A mirror far larger than the request means the merge, not a descriptor, chose this
	// size: buffers only ever grow here, so one that swallows its neighbours as a title
	// streams across its heap ends up spanning it. Name the request that did it.
	if (overlap.end - overlap.begin >= 256ull * 1024 * 1024) {
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 12) {
			size_t absorbed = 0;
			for (auto it = overlap.first; it != overlap.last; ++it) {
				absorbed++;
			}
			LOGF("BufferCache: large mirror %" PRIu64 "MB at 0x%016" PRIx64
			     " for request 0x%016" PRIx64 "+%" PRIu64 "KB, absorbing %zu buffers, leap=%d\n",
			     (overlap.end - overlap.begin) >> 20u, overlap.begin, vaddr, size >> 10u, absorbed,
			     overlap.has_stream_leap ? 1 : 0);
		}
	}
	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

namespace {
// What the per-draw synchronization is asked to cover. A giant raw descriptor makes every
// draw walk the tracker for hundreds of regions; the histogram says whether that is the case.
struct SyncStats {
	uint64_t calls      = 0;
	uint64_t uploads    = 0;
	uint64_t bytes      = 0;
	uint64_t buckets[6] = {}; // <=4K, <=64K, <=1M, <=16M, <=256M, larger
	uint64_t written_buckets[6] = {};
	uint64_t written    = 0;
	double   walk_s     = 0.0;
};
SyncStats g_sync_stats;

void RecordSync(uint64_t size, bool is_written, bool uploaded, double walk_s) {
	auto& st = g_sync_stats;
	st.calls++;
	st.uploads += uploaded ? 1 : 0;
	st.bytes += size;
	st.written += is_written ? 1 : 0;
	st.walk_s += walk_s;
	const int bucket = size <= 0x1000 ? 0 : size <= 0x10000 ? 1 : size <= 0x100000 ? 2
	                 : size <= 0x1000000 ? 3 : size <= 0x10000000 ? 4 : 5;
	st.buckets[bucket]++;
	if (is_written) {
		st.written_buckets[bucket]++;
	}
	if (st.calls % 200000 == 0) {
		LOGF("SyncStats: calls=%" PRIu64 " uploads=%" PRIu64 " written=%" PRIu64
		     " avg_bytes=%" PRIu64 " walk_total=%.1fms sizes(<=4K,64K,1M,16M,256M,>)=%" PRIu64
		     ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 " written_sizes=%" PRIu64
		     ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
		     st.calls, st.uploads, st.written, st.bytes / st.calls, st.walk_s * 1000.0,
		     st.buckets[0], st.buckets[1], st.buckets[2], st.buckets[3], st.buckets[4],
		     st.buckets[5], st.written_buckets[0], st.written_buckets[1], st.written_buckets[2],
		     st.written_buckets[3], st.written_buckets[4], st.written_buckets[5]);
		st = {};
	}
}
} // namespace

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	KYTY_PROFILER_FUNCTION();
	// Timing every call cost ~2 % of the GPU thread (two performance-counter reads); SyncStats
	// times the tracker walk only with KYTY_SYNC_STATS=1.
	static const bool timed = std::getenv("KYTY_SYNC_STATS") != nullptr;
	Common::Timer               walk_timer;
	if (timed) {
		walk_timer.Start();
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	{
		KYTY_PROFILER_BLOCK("Sync::Tracker");
		m_memory_tracker.ForEachUploadRange(
		    vaddr, size, is_written,
		    [&](uint64_t address, uint64_t bytes) noexcept {
			    copies.emplace_back(total_size, buffer.Offset(address), bytes);
			    total_size += bytes;
		    },
		    [&]() noexcept {
			    KYTY_PROFILER_BLOCK("Sync::Upload");
			    source = UploadCopies(buffer, copies, total_size);
		    });
	}
	RecordSync(size, is_written, static_cast<bool>(source), timed ? walk_timer.GetTimeS() : 0.0);
	if (source) {
		KYTY_PROFILER_BLOCK("Sync::Copy");
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Recorder();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		KYTY_PROFILER_BLOCK("Sync::TexelImage");
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

// Reads guest memory for an upload. The guest can unmap memory a cached buffer still covers: AMM
// maps streaming textures in blocks, and a texture upload once read one byte past the end of the
// block it had mapped. Such bytes read as zero, like ObtainBufferForImage's direct path.
static void ReadGuestForUpload(uint8_t* destination, uint64_t address, uint64_t size) {
	if (!Libs::LibKernel::Memory::TryReadBacking(address, destination, size) &&
	    !Libs::LibKernel::Memory::TryReadSparseBacking(address, destination, size)) {
		static std::atomic<uint32_t> reported {0};
		if (reported.fetch_add(1) < 16) {
			LOGF("BufferCache: upload of unmapped guest memory 0x%016" PRIx64 " size=0x%" PRIx64
			     " reads as zero\n",
			     address, size);
		}
		std::memset(destination, 0, static_cast<size_t>(size));
	}
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}
	Profiler::Add(Profiler::Counter::UploadBytes, static_cast<int64_t>(total_size));

	auto [mapped, base_offset] = m_staging_buffer.Map(total_size, 4);
	if (mapped != nullptr) {
		// Reading guest memory into staging was ~28 MB a frame of memcpy on the GPU thread in
		// the jungle (~8,500 uploads). With a recording thread the reads run there instead, in
		// the recorded command stream right before the copies that use them, which the caller
		// records next; the staging memory must be host-coherent, since nothing flushes it after.
		// A read sees the guest bytes of slightly later than the upload decision: newer CPU
		// writes re-dirty their pages and are uploaded again, as before. Off by default: no
		// measurable gain in the jungle (16.3 fps from 140 to 240 s with and without, 2026-10-07),
		// and writes that bypass tracking land in the upload (two strict cache tests fail).
		// KYTY_UPLOAD_READS_ON_RECORD_THREAD=1 turns it on.
		static const bool deferred = [] {
			const char* value = std::getenv("KYTY_UPLOAD_READS_ON_RECORD_THREAD");
			return value != nullptr && value[0] == '1';
		}();
		if (deferred && m_staging_buffer.IsCoherent() && m_scheduler.Current().Recorder().Threaded()) {
			m_upload_reads.clear();
			for (auto& copy: copies) {
				m_upload_reads.push_back(
				    {mapped + copy.srcOffset, buffer.CpuAddress() + copy.dstOffset, copy.size});
				copy.srcOffset += base_offset;
			}
			const auto* reads = m_scheduler.Stash(m_upload_reads.data(), m_upload_reads.size());
			const auto  count = m_upload_reads.size();
			m_scheduler.Record([reads, count](vk::CommandBuffer) {
				for (size_t i = 0; i < count; i++) {
					ReadGuestForUpload(reads[i].destination, reads[i].address, reads[i].size);
				}
			});
			m_staging_buffer.Commit();
			return m_staging_buffer.Handle();
		}
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			ReadGuestForUpload(mapped + copy.srcOffset, address, copy.size);
			copy.srcOffset += base_offset;
		}
		m_staging_buffer.Commit();
		return m_staging_buffer.Handle();
	}

	KYTY_PROFILER_BLOCK("BufferCache::UploadCopies(temporary)");
	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         vk::BufferUsageFlagBits::eTransferSrc, total_size);
	for (const auto& copy: copies) {
		const auto address = buffer.CpuAddress() + copy.dstOffset;
		ReadGuestForUpload(temporary->Mapped().data() + copy.srcOffset, address, copy.size);
	}
	temporary->Flush(0, total_size);
	const auto handle = temporary->Handle();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return handle;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id, bool needs_device_address) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	if (!is_written && size <= CACHING_PAGESIZE &&
	    (!needs_device_address || m_stream_buffer.HasDeviceAddress()) &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			std::memcpy(mapped, reinterpret_cast<const void*>(vaddr), size);
			m_stream_buffer.Commit();
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	buffer.last_use_tick = m_scheduler.CurrentTick();
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		MarkGpuWritten(vaddr, size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferWritten(uint64_t vaddr, uint64_t size,
                                                              uint64_t written_vaddr,
                                                              uint64_t written_size, BufferId id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid() ||
	    written_vaddr < vaddr || written_size > size || written_vaddr - vaddr > size - written_size) {
		EXIT("BufferCache: invalid written-range buffer request\n");
	}
	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	buffer.last_use_tick = m_scheduler.CurrentTick();
	TouchBuffer(buffer);
	// The whole binding is current on the GPU, as for a written buffer; only the bytes the
	// stores can reach become GPU-written.
	(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
	if (written_size != 0) {
		(void)SynchronizeBuffer(buffer, written_vaddr, written_size, true, false);
		MarkGpuWritten(written_vaddr, written_size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

void BufferCache::MarkGpuWritten(uint64_t vaddr, uint64_t size) {
	KYTY_PROFILER_BLOCK("Obtain::MarkWritten");
	{
		std::unique_lock lock(m_dirty_ranges_mutex);
		m_gpu_modified_ranges.Add(vaddr, size);
	}
	// Diagnostics: KYTY_WATCH_GPU_WRITE=<address>[+<size>][,<address>[+<size>]...] (hex) names
	// the bindings that mark bytes of those ranges GPU-written: the first write of each
	// (range, shader), with a host stack when no shader is being bound (copies, fills).
	static const std::vector<std::pair<uint64_t, uint64_t>> watches = [] {
		std::vector<std::pair<uint64_t, uint64_t>> result;
		const char*                                value = std::getenv("KYTY_WATCH_GPU_WRITE");
		while (value != nullptr && *value != '\0') {
			char*      end   = nullptr;
			const auto begin = std::strtoull(value, &end, 16);
			uint64_t   bytes = 1;
			if (end != nullptr && *end == '+') {
				bytes = std::strtoull(end + 1, &end, 16);
			}
			result.emplace_back(begin, bytes);
			value = end != nullptr && *end == ',' ? end + 1 : nullptr;
		}
		return result;
	}();
	static std::atomic<uint32_t> watched {0};
	for (size_t index = 0; index < watches.size(); index++) {
		const auto& [watch_begin, watch_size] = watches[index];
		if (watch_size == 0 || vaddr >= watch_begin + watch_size ||
		    watch_begin >= vaddr + size) {
			continue;
		}
		static std::mutex                             seen_mutex;
		static std::set<std::pair<size_t, uint64_t>> seen;
		bool                                          first = false;
		{
			std::lock_guard lock(seen_mutex);
			first = seen.emplace(index, s_diag_shader_hash).second;
		}
		if (first && watched.fetch_add(1) < 128) {
			LOGF("GPU write covers watched #%zu 0x%016" PRIx64 "+0x%" PRIx64
			     ": range=0x%016" PRIx64 " size=0x%" PRIx64 " shader=0x%016" PRIx64 "\n%s",
			     index, watch_begin, watch_size, vaddr, size, s_diag_shader_hash,
			     s_diag_shader_hash == 0 ? Common::HostBacktrace().c_str() : "");
		}
	}
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	auto [staging, stage_offset] = m_staging_buffer.Map(size, 16);
	if (staging == nullptr) {
		EXIT("BufferCache: staging reservation failed for guest image\n");
	}
	if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, staging, size)) {
		std::memset(staging, 0, static_cast<size_t>(size));
	}
	m_staging_buffer.Commit();
	return {&m_staging_buffer, stage_offset};
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

void BufferCache::DownloadRangeForDiagnostics(uint64_t vaddr, uint64_t size) {
	std::vector<std::pair<uint64_t, uint64_t>> downloaded;
	auto                                       it = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < vaddr + size; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto begin  = std::max(vaddr, buffer.CpuAddress());
		const auto end    = std::min(vaddr + size, buffer.CpuAddress() + buffer.Size());
		if (begin < end && DownloadBufferMemory<true>(buffer, begin, end - begin)) {
			downloaded.emplace_back(begin, end - begin);
		}
	}
	if (downloaded.empty()) {
		return;
	}
	const auto tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(tick);
	m_scheduler.WaitPriorityOperations(tick);
	for (const auto& [begin, bytes]: downloaded) {
		m_memory_tracker.UnmarkRegionAsGpuModified(begin, bytes);
	}
}

bool BufferCache::IsCleanForConcurrentRead(uint64_t vaddr, uint64_t size) const {
	std::shared_lock lock(m_dirty_ranges_mutex);
	return !m_gpu_modified_ranges.Intersects(vaddr, size) &&
	       !m_downloading_ranges.Intersects(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

uint64_t BufferCache::LruClock() const noexcept {
	return m_graphics.presented_frames.load(std::memory_order_relaxed) + m_gc_tick / 512;
}

void BufferCache::RunGarbageCollector() {
	KYTY_PROFILER_FUNCTION();
	m_gc_tick++;
	const auto clock = LruClock();
	// Pressure is judged by this cache's own bytes. Device-wide usage also counts the images
	// spilled to host memory, which on a 6 GB card keeps it above the critical mark forever
	// and has the collector destroy and recreate every buffer twice a second (a third of
	// the GPU thread's time in the world). KYTY_GC_DEVICE_BYTES=1 restores that policy.
	static const bool device_bytes = std::getenv("KYTY_GC_DEVICE_BYTES") != nullptr;
	if (device_bytes && m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	// Ages in frames, as in the texture cache: a buffer used this frame or the last is
	// never a candidate, whatever the submission count.
	const uint64_t age        = std::min<uint64_t>(aggressive ? 2 : 4, clock);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	static uint64_t       gc_runs = 0, gc_visited = 0, gc_retained = 0, gc_deleted = 0;
	static double         gc_seconds = 0.0;
	Common::Timer         gc_timer;
	gc_timer.Start();
	gc_runs++;
	m_lru_cache.ForEachItemBelow(clock - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		gc_visited++;
		if (buffer.CpuAddress() == 0) {
			// See CreateBuffer: the tracker rejects address 0, so this one is never collected.
			return false;
		}
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		// An aged GPU-dirty buffer is downloaded and dropped under either policy. Retaining it
		// (the non-aggressive rule before) leaves its GPU-written bytes owning the pages
		// forever, and the images that share those pages are then re-sourced from the
		// buffer's stale contents: the world renders black. Measured 2026-09-21.
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size())) {
			// Its pages are still marked GPU-written, but no byte waits for a download: they were
			// downloaded with another range, or a game thread's asynchronous readback of them is
			// still in flight (ReadMemoryStep's "page is current" case). This ended the session on
			// cards small enough for the collector to run (a 4 GB laptop, 2026-10-06). In flight:
			// leave the buffer to a later pass, after the publication. Otherwise the bytes are
			// current in guest memory and the buffer goes like a clean one.
			bool downloading = false;
			{
				std::shared_lock lock(m_dirty_ranges_mutex);
				downloading = m_downloading_ranges.Intersects(buffer.CpuAddress(), buffer.Size());
			}
			if (downloading) {
				return false;
			}
			m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
		} else if (dirty) {
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
		}
		gc_deleted++;
		return ++retire_count == limit;
	});
	gc_seconds += gc_timer.GetTimeS();
	if (gc_runs % 500 == 0) {
		LOGF("GcStats: runs=%" PRIu64 " visited=%" PRIu64 " retained=%" PRIu64 " deleted=%" PRIu64
		     " scan_total=%.1fms buffers=%zu used=%" PRIu64 "MB trigger=%" PRIu64 "MB aggressive=%d\n",
		     gc_runs, gc_visited, gc_retained, gc_deleted, gc_seconds * 1000.0, m_buffers.size(),
		     m_total_used_memory >> 20u, m_trigger_gc_memory >> 20u, static_cast<int>(aggressive));
		gc_visited = gc_retained = gc_deleted = 0;
		gc_seconds = 0.0;
		// The largest buffers: the whole-heap descriptors of a title mirror as whole device
		// buffers, and a handful of them can be most of the cache.
		std::vector<std::pair<uint64_t, uint64_t>> largest;
		for (const auto& [address, id]: m_buffers) {
			const auto* buffer = m_slot_buffers.try_get(id);
			if (buffer != nullptr && !buffer->is_deleted) {
				largest.emplace_back(buffer->Size(), address);
			}
		}
		std::partial_sort(largest.begin(), largest.begin() + std::min<size_t>(largest.size(), 6),
		                  largest.end(), std::greater<>());
		for (size_t i = 0; i < std::min<size_t>(largest.size(), 6); i++) {
			LOGF("\t largest[%zu] = %" PRIu64 "MB at 0x%016" PRIx64 "\n", i, largest[i].first >> 20u,
			     largest[i].second);
		}
	}
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners. Must be
	// CurrentTick(): DownloadBufferMemory above queued copy-out commands into the currently open
	// recording, so that recording has to actually submit and complete -- see ReadMemory's wait
	// for why waiting on an older per-buffer tick here would skip that entirely.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::SynchronizeCpuDirtyBuffersInRange(uint64_t vaddr, uint64_t size) {
	m_memory_tracker.ForEachMaybeCpuDirtyRegion(
	    vaddr, size, [this](uint64_t address, uint64_t bytes) {
		    SynchronizeBuffersInRange(address, bytes);
	    });
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

} // namespace Libs::Graphics
