#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_

#include "common/common.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <deque>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class SamplerCache;

// Descriptor set 1 of every pipeline that samples bindless images: typed arrays of sampled
// images a shader indexes with a slot it looks up per pixel, the key -> slot translation the
// host fills per guest descriptor heap, and a bitset where shaders flag keys whose texture is
// not resident yet. Created only when the device supports descriptor indexing.
class BindlessTable {
public:
	enum Binding : uint32_t {
		Images2D      = 0,
		Images2DArray = 1,
		ImagesCube    = 2,
		Images3D      = 3,
		Translation   = 4,
		Feedback      = 5,
		Samplers      = 6,
	};
	static_assert(Images2D == ShaderRecompiler::IR::BindlessImages2D &&
	              Images2DArray == ShaderRecompiler::IR::BindlessImages2DArray &&
	              ImagesCube == ShaderRecompiler::IR::BindlessImagesCube &&
	              Images3D == ShaderRecompiler::IR::BindlessImages3D &&
	              Translation == ShaderRecompiler::IR::BindlessTranslation &&
	              Feedback == ShaderRecompiler::IR::BindlessFeedback &&
	              Samplers == ShaderRecompiler::IR::BindlessSamplers);
	static constexpr uint32_t ImageArrays        = 4;
	static constexpr uint32_t PlaceholderColors  = 3;
	static constexpr uint32_t Placeholders       = ImageArrays * PlaceholderColors;
	static constexpr uint32_t MaxImagesPerArray  = 16384;
	static constexpr uint32_t TranslationEntries = 1u << 20u;
	static constexpr uint32_t MaxSamplers        = 4096;

	BindlessTable(GraphicContext& graphics, CommandScheduler& scheduler);
	~BindlessTable();
	KYTY_CLASS_NO_COPY(BindlessTable);

	[[nodiscard]] bool                    Enabled() const noexcept { return m_set != nullptr; }
	[[nodiscard]] vk::DescriptorSetLayout Layout() const noexcept { return m_layout; }
	[[nodiscard]] vk::DescriptorSet       Set() const noexcept { return m_set; }
	[[nodiscard]] uint32_t                ImagesPerArray() const noexcept { return m_images_per_array; }

	// A guest descriptor heap as one view type samples it: a region of the translation (and
	// feedback) buffer, one entry per key, and the textures resolved so far.
	struct Heap {
		uint64_t                            base         = 0;
		uint32_t                            table_offset = 0;
		uint32_t                            binding      = 0;
		uint32_t                            region       = 0;
		// Keys the region holds. The guest appends textures to a heap as it streams them in and
		// its descriptor grows; each draw patches its own descriptor's entry count.
		uint32_t                            entries      = 0;
		ShaderRecompiler::IR::ImageResource resource;
		std::vector<uint32_t>               slots;    // per key; 0 = not resolved
		std::vector<uint8_t>                settled;  // per key; resolved, or known placeholder
		std::vector<ImageId>                resolved; // images to keep readable for draws
		// Per settled key, the T# it was settled from and the image it resolved to (none for a
		// placeholder). The guest rewrites entries as it streams textures out and others in;
		// RenderExecutor::RevalidateBindlessKeys settles a key whose T# changed again.
		std::vector<std::array<uint32_t, 8>> descriptors;
		std::vector<ImageId>                 images;
		uint32_t                             revalidate_cursor = 0; // next key to compare
		// What RenderExecutor::CommitBindings last checked: g_bindless_state_generation then, and
		// how many resolved images. Unchanged since, only the images resolved after need a look.
		uint64_t checked_generation = 0;
		size_t   checked_count      = 0;
	};

	// The heap for (base, table offset, view binding), created with every entry pending and
	// room to grow, or moved to a larger region when a descriptor covers more keys; null when
	// the translation buffer is full.
	[[nodiscard]] Heap* FindOrCreateHeap(uint64_t base, uint32_t table_offset, uint32_t binding,
	                                     uint32_t entries,
	                                     const ShaderRecompiler::IR::ImageResource& resource);
	[[nodiscard]] std::deque<Heap>& Heaps() noexcept { return m_heaps; }
	// 0 when the array is full. A released slot is reused once the GPU has finished the work
	// recorded before its release.
	[[nodiscard]] uint32_t AllocateSlot(uint32_t binding);
	void WriteSlot(uint32_t binding, uint32_t slot, vk::ImageView view, vk::ImageLayout layout);
	// Resolve a key to a slot (0 = placeholder), or back to pending.
	void SetTranslation(const Heap& heap, uint32_t key, uint32_t slot);
	// The host reads the feedback flags from a copy in host-cached memory. The feedback buffer
	// itself is device memory, and reading every key of three heaps from it uncached took a
	// fifth of the GPU thread. Records that copy at the current point of the command stream.
	void RecordFeedbackSnapshot(CommandScheduler& scheduler);
	[[nodiscard]] bool SnapshotRecorded() const noexcept { return m_snapshot_tick != 0; }
	// Whether the GPU has executed the recorded copy; never waits.
	[[nodiscard]] bool SnapshotReady(CommandScheduler& scheduler);
	void               ConsumeSnapshot() noexcept { m_snapshot_tick = 0; }
	// Pending keys flagged in the snapshot; their flags are cleared.
	void TakeRequests(const Heap& heap, std::vector<uint32_t>& keys);
	// Diagnostics: feedback word 0 (the last out-of-range key, top bit set) in the snapshot,
	// cleared.
	[[nodiscard]] uint32_t TakeWordZero();
	// Usage probe. Every draw that samples a key stores its feedback word (0 for a resident
	// texture, 1 for a pending one), so marking every resident key's word with ProbeMark and
	// reading a snapshot some frames later shows which keys no draw sampled in between.
	// ArmUsageProbe marks them; CollectUnusedImages lists the images all of whose keys still
	// carry the mark in the current snapshot.
	static constexpr uint32_t ProbeMark = 2;
	void                      ArmUsageProbe();
	void                      CollectUnusedImages(std::vector<ImageId>& unused);
	void AddImageReference(ImageId id, Heap& heap, uint32_t key);
	// The key no longer samples what it was settled to: it is pending again, and its image loses
	// the key's reference. Its slot keeps the old view, which frames in flight may still sample;
	// a new resolution takes a new slot. True when the image has no reference left, so it need
	// not stay pinned in the texture cache.
	[[nodiscard]] bool ReleaseKey(Heap& heap, uint32_t key);

	// A guest sampler heap as one kind of use samples it: a region of the sampler array that
	// mirrors its S# records, key for key. Uses differ in whether the depth-compare function is
	// kept, whether point filtering is forced and whether border colours are integer, so each
	// kind gets its own region.
	static constexpr uint32_t SamplerDepthCompare   = 1u;
	static constexpr uint32_t SamplerPointFiltering = 2u;
	static constexpr uint32_t SamplerIntegerBorder  = 4u;
	struct SamplerHeap {
		uint64_t                             base          = 0;
		uint32_t                             table_offset  = 0;
		uint32_t                             flags         = 0;
		uint32_t                             region        = 0; // first slot; 0 = none yet
		uint32_t                             capacity      = 0;
		std::vector<std::array<uint32_t, 4>> records;           // mirrored S# records
		uint64_t                             checked_frame = UINT64_MAX;
	};
	[[nodiscard]] SamplerHeap* FindOrCreateSamplerHeap(uint64_t base, uint32_t table_offset,
	                                                   uint32_t flags);
	// Mirrors the heap's records into its region. New records are appended in place (their slots
	// were never used); a changed record, or more records than the region holds, moves the heap
	// to a new region, because slots the GPU may be reading are never rewritten. False when the
	// sampler array is full; the heap then keeps its previous records.
	bool MirrorSamplerHeap(SamplerHeap& heap, std::span<const std::array<uint32_t, 4>> records,
	                       SamplerCache& cache);
	// Slot 0 of the sampler array, used for keys outside their heap; written once.
	void WriteDefaultSampler(vk::Sampler sampler);
	[[nodiscard]] bool SamplersEnabled() const noexcept { return m_samplers_per_array != 0; }
	// The texture cache dropped a resolved image: point its slots back at the placeholder and
	// make its keys pending again, so a draw that still needs it asks for it anew.
	void OnImageUnregistered(ImageId id);
	// Before an image is destroyed: a slot that still holds one of its views is repointed to the
	// placeholder and the keys that use it are pending again. True when one did, so the caller
	// keeps the image until the GPU has finished the work recorded with that slot.
	[[nodiscard]] bool ReleaseViews(std::span<const vk::ImageView> views);

private:
	[[nodiscard]] bool AllocateRegion(Heap& heap, uint32_t entries);
public:

private:
	// Slot 0 of every image array: a 1x1 grey texture of that view type, sampled for keys
	// outside a heap and while a texture is pending. Created, cleared and made read-only once.
	void CreatePlaceholders(CommandScheduler& scheduler);

	[[nodiscard]] static uint64_t ImageKey(ImageId id) {
		return static_cast<uint64_t>(id.index) | (static_cast<uint64_t>(id.generation) << 32u);
	}

	// A key gave up its slot: free it for reuse after the GPU passes the current tick.
	void ReleaseSlot(uint32_t binding, uint32_t slot);

	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	std::deque<Heap>        m_heaps;
	std::array<std::deque<std::pair<uint32_t, uint64_t>>, ImageArrays> m_free_slots; // slot, tick
	std::unordered_map<uint64_t, std::vector<std::pair<Heap*, uint32_t>>> m_image_refs;
	uint32_t                m_next_region = 1; // translation[0] is the out-of-range entry
	std::array<uint32_t, ImageArrays> m_next_slot {PlaceholderColors, PlaceholderColors,
	                                               PlaceholderColors, PlaceholderColors};
	std::array<VulkanImage, Placeholders>   m_placeholders;
	std::array<vk::ImageView, Placeholders> m_placeholder_views {};
	vk::DescriptorPool      m_pool   = nullptr;
	vk::DescriptorSetLayout m_layout = nullptr;
	vk::DescriptorSet       m_set    = nullptr;
	uint32_t                m_images_per_array = 0;
	uint32_t                m_samplers_per_array = 0;
	uint32_t                m_next_sampler_slot  = 1; // slot 0 is the default sampler
	bool                    m_default_sampler_written = false;
	std::deque<SamplerHeap> m_sampler_heaps;
	std::unique_ptr<Buffer> m_translation;
	std::unique_ptr<Buffer> m_feedback;
	std::unique_ptr<Buffer> m_feedback_snapshot;
	uint64_t                m_snapshot_tick = 0; // 0 = none recorded
	// The view each image slot holds, and the slots each view is in (placeholders excluded).
	std::array<std::vector<vk::ImageView>, ImageArrays>                        m_slot_views;
	std::unordered_map<VkImageView, std::vector<std::pair<uint32_t, uint32_t>>> m_view_slots;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_
