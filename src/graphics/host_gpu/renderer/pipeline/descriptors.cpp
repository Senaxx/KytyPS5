#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <bit>
#include <fmt/format.h>
#include <limits>
#include <span>
#include <vector>
#include <xxhash.h>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace Libs::Graphics {

namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;

struct WatchedReread {
	uint64_t frame   = 0;
	uint64_t address = 0;
	uint64_t size    = 0;
};

std::mutex                 g_watched_reread_mutex;
std::vector<WatchedReread> g_watched_rereads;

void LogWords(const uint32_t* words, size_t count) {
	for (size_t row = 0; row < count; row += 8) {
		std::string line;
		for (size_t k = row; k < std::min(count, row + 8); k++) {
			line += fmt::format(" {:08x}", words[k]);
		}
		LOGF("    +0x%03zx:%s\n", row * 4, line.c_str());
	}
}

} // namespace

static uint64_t WatchedShaderHash() {
	static const uint64_t hash = [] {
		const char* value = std::getenv("KYTY_WATCH_SHADER");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	}();
	return hash;
}

static std::atomic<uint64_t> g_watched_first_frame {UINT64_MAX};

uint64_t WatchedShaderFirstFrame() {
	return g_watched_first_frame.load();
}

bool InWatchedShaderFrame(uint64_t hash, uint64_t presented_frame, std::atomic<uint32_t>& logged) {
	static const uint64_t delta = [] {
		const char* value = std::getenv("KYTY_WATCH_SHADER_FRAME");
		return value != nullptr ? std::strtoull(value, nullptr, 10) : UINT64_MAX;
	}();
	if (hash == 0 || hash != WatchedShaderHash()) {
		return false;
	}
	uint64_t unset = UINT64_MAX;
	g_watched_first_frame.compare_exchange_strong(unset, presented_frame);
	return delta != UINT64_MAX && presented_frame == g_watched_first_frame.load() + delta &&
	       logged.fetch_add(1) < 64u;
}

void LogWatchedRereads(BufferCache& cache, uint64_t presented_frames) {
	std::vector<WatchedReread> due;
	{
		std::lock_guard lock(g_watched_reread_mutex);
		for (auto it = g_watched_rereads.begin(); it != g_watched_rereads.end();) {
			if (it->frame <= presented_frames) {
				due.push_back(*it);
				it = g_watched_rereads.erase(it);
			} else {
				++it;
			}
		}
	}
	for (const auto& reread: due) {
		std::vector<uint32_t> words(reread.size / 4u);
		const bool gpu_written = cache.HasGpuDirtyBytes(reread.address, reread.size);
		const bool read =
		    Libs::LibKernel::Memory::TryReadBacking(reread.address, words.data(), reread.size);
		LOGF("WatchShader reread at frame %" PRIu64 ": 0x%012" PRIx64 "+0x%" PRIx64
		     " gpu_written=%d%s\n",
		     presented_frames, reread.address, reread.size, gpu_written ? 1 : 0,
		     read ? "" : " (unreadable)");
		if (read) {
			LogWords(words.data(), words.size());
		}
	}
}

vk::DescriptorType NativeDescriptorType(BindingKind kind) {
	const auto image_class = ShaderRecompiler::IR::ImageBindingResourceClass(kind);
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Sampled) {
		return vk::DescriptorType::eSampledImage;
	}
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Storage) {
		return vk::DescriptorType::eStorageImage;
	}
	switch (kind) {
		case BindingKind::Samplers: return vk::DescriptorType::eSampler;
		case BindingKind::Buffers:
		case BindingKind::Gds:
		case BindingKind::BdaPagetable:
		case BindingKind::FaultBuffer:
		case BindingKind::FlattenedSrt:
		case BindingKind::ShaderData:
		case BindingKind::SharedMemory: return vk::DescriptorType::eStorageBuffer;
		case BindingKind::Count: EXIT("invalid native descriptor binding kind");
	}
	EXIT("invalid native descriptor binding kind");
}

uint32_t NativeDescriptorCount(const ShaderRecompiler::IR::DescriptorBinding& binding) {
	return binding.resources.empty() ? 1u : static_cast<uint32_t>(binding.resources.size());
}

vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture, uint32_t element) {
	vk::ImageView view = nullptr;
	if (texture.mip_views.empty()) {
		if (element == 0u) {
			view = texture.image_view;
		}
	} else if (element < texture.mip_views.size()) {
		view = texture.mip_views[element];
	}
	EXIT_IF(!texture.image_id || view == nullptr || texture.layout == vk::ImageLayout::eUndefined);
	return {nullptr, view, texture.layout};
}

static const char* ShaderStageResourceName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "Vertex";
		case ShaderType::Mesh: return "Mesh";
		case ShaderType::Local: return "Local";
		case ShaderType::TessellationControl: return "Hull";
		case ShaderType::TessellationEvaluation: return "Domain";
		case ShaderType::Pixel: return "Pixel";
		case ShaderType::Compute: return "Compute";
		default: return "Unknown";
	}
}

static Prospero::ImageType TextureType(const ShaderTextureResource& descriptor) {
	const auto type = descriptor.Type();
	return type == Prospero::ImageType::kCube ? Prospero::ImageType::kColor2DArray : type;
}

static Prospero::ImageType TextureBaseType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1DArray: return Prospero::ImageType::kColor1D;
		case Prospero::ImageType::kColor2DArray:
		case Prospero::ImageType::kColor2DMsaa:
		case Prospero::ImageType::kColor2DMsaaArray: return Prospero::ImageType::kColor2D;
		default: return type;
	}
}

static bool IsMultisampledTexture(Prospero::ImageType type) {
	return type == Prospero::ImageType::kColor2DMsaa ||
	       type == Prospero::ImageType::kColor2DMsaaArray;
}

// A/B switch for bounded buffer writes (KYTY_BOUNDED_BUFFER_WRITES=0 marks every written buffer's
// whole range GPU-written, as before).
static const bool g_bounded_buffer_writes = [] {
	const char* value = std::getenv("KYTY_BOUNDED_BUFFER_WRITES");
	return value == nullptr || value[0] != '0';
}();

// Diagnostics: names the shader whose bindings are prepared, for KYTY_WATCH_GPU_WRITE.
struct DiagShaderScope {
	explicit DiagShaderScope(uint64_t hash) { BufferCache::s_diag_shader_hash = hash; }
	~DiagShaderScope() { BufferCache::s_diag_shader_hash = 0; }
	KYTY_CLASS_NO_COPY(DiagShaderScope);
};

static vk::DescriptorBufferInfo
NativeStorageBuffer(RenderContext& context, const PreparedBindings::BufferSource& source,
                    const ShaderRecompiler::IR::BufferResource&    resource,
                    const ShaderRecompiler::IR::BufferWriteExtent* extent, uint32_t& buffer_offset) {
	buffer_offset = 0;

	const auto& [address, size, id] = source;
	if (address < BufferCache::CACHING_PAGESIZE || size == 0) {
		return {context.GetBufferCache().GetBuffer(NULL_BUFFER_ID).Handle(), 0, 16};
	}
	const auto& graphics  = context.GetGraphics();
	const auto  alignment = graphics.StorageMinAlignment();
	if (size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range is unsupported\n");
	}
	// The stores' addresses bound the bytes this draw or dispatch writes: only those become
	// GPU-written, so the guest's accesses elsewhere in the range do not drain the GPU.
	const bool bounded_write = resource.written && extent != nullptr && extent->valid &&
	                           g_bounded_buffer_writes;
	const auto written_begin = bounded_write ? std::min(extent->begin, size) : 0u;
	const auto written_end   = bounded_write ? std::min(extent->end, size) : size;
	auto [buffer, offset] =
	    bounded_write
	        ? context.GetBufferCache().ObtainBufferWritten(address, size, address + written_begin,
	                                                       written_end - std::min(written_begin,
	                                                                              written_end),
	                                                       id)
	        : context.GetBufferCache().ObtainBuffer(address, size, resource.written,
	                                                resource.formatted, id);
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	const auto adjustment     = offset - aligned_offset;
	const auto max_range      = graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	if (adjustment >= 256 || size > max_range - adjustment) {
		EXIT("storage buffer offset adjustment is unsupported\n");
	}
	buffer_offset = static_cast<uint32_t>(adjustment);
	const vk::DescriptorBufferInfo result {buffer->Handle(), aligned_offset, size + adjustment};
	if (resource.written && written_end > written_begin) {
		context.GetTextureCache().InvalidateMemoryFromGPU(address + written_begin,
		                                                  written_end - written_begin);
	}
	return result;
}

bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	const uint32_t     field3_expected = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     field4_expected = descriptor.Depth() | (descriptor.BaseArray5() << 16u);
	const uint32_t     field5_expected = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	const bool         common          = (descriptor.fields[1] & field1_reserved_mask) == 0 &&
	                                     (descriptor.fields[2] & field2_reserved_mask) == 0 &&
	                                     descriptor.fields[3] == field3_expected;
	if (r128) {
		return common && descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
		       descriptor.fields[6] == 0 && descriptor.fields[7] == 0;
	}
	const bool full = common && descriptor.fields[4] == field4_expected &&
	                  descriptor.fields[5] == field5_expected;
	if (!full ||
	    (descriptor.MsaaDepth() && !IsMultisampledTexture(descriptor.Type()))) {
		return false;
	}
	const auto metadata_control = descriptor.fields[6] & 0x00ffffffu;
	if (metadata_control == 0) {
		return true;
	}
	constexpr uint32_t htile_control = 0x00280000u;
	const uint32_t expected_control  = htile_control | (descriptor.MsaaDepth() ? (1u << 10u) : 0u);
	const auto     metadata_addr     = descriptor.MetaAddr() << 8u;
	return metadata_control == expected_control && GuestRange {metadata_addr, 1}.Valid() &&
	       (metadata_addr & 0x7fffu) == 0 &&
	       descriptor.TileMode() == Prospero::TileMode::kDepth;
}

static void ValidateSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                        const ShaderTextureResource& descriptor, const Image& image,
                                        vk::Format view_format, uint64_t size) {
	const bool resource_ok = IsSupportedSampledDepthResource(resource);
	const bool encoding_ok = IsSupportedDepthTextureEncoding(descriptor, resource.r128);
	const bool view_ok =
	    IsSupportedSampledDepthView(image.info.pixel_format, view_format, descriptor.DstSelXYZW());
	if (resource_ok && encoding_ok && view_ok) {
		return;
	}
	const auto descriptor_pitch =
	    TileGetTexturePitch(descriptor.Format(), static_cast<uint32_t>(descriptor.Width5()) + 1u,
	                        descriptor.TileMode());
	EXIT("unsupported sampled depth image: resource=%d encoding=%d view=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d atomic=%d compare=%d "
	     "guest_format=%u swizzle=0x%03x image_format=%d view_format=%d image_layers=%u "
	     "descriptor_type=%u base_array=%u depth=%u descriptor_pitch=%u target_pitch=%u "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, encoding_ok, view_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.read, resource.written, resource.atomic,
	     resource.depth_compare, static_cast<uint32_t>(descriptor.Format()),
	     descriptor.DstSelXYZW(), static_cast<int>(image.info.pixel_format),
	     static_cast<int>(view_format), image.info.resources.layers,
	     static_cast<uint32_t>(descriptor.Type()), descriptor.BaseArray5(), descriptor.Depth(),
	     descriptor_pitch, image.info.pitch, descriptor.Base40(), size, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static bool IsSupportedStorageTextureDescriptor(const ShaderRecompiler::IR::ImageResource& resource,
                                                const ShaderTextureResource& descriptor) {
	const auto tile              = descriptor.TileMode();
	const bool is_color_1d       = descriptor.Type() == Prospero::ImageType::kColor1D;
	const bool is_color_1d_array = descriptor.Type() == Prospero::ImageType::kColor1DArray;
	const bool valid_1d_slice =
	    (is_color_1d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_1d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_1d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D &&
	                   descriptor.Height5() == 0 && valid_1d_slice;
	const bool is_1d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1DArray &&
	    is_color_1d_array && descriptor.Height5() == 0 &&
	    descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_color_2d       = descriptor.Type() == Prospero::ImageType::kColor2D;
	const bool is_color_2d_array = descriptor.Type() == Prospero::ImageType::kColor2DArray;
	const bool valid_2d_slice =
	    (is_color_2d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_2d =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D && valid_2d_slice;
	// Storage cube coordinates address individual faces, including partial cube views.
	const bool is_cube = resource.cube && descriptor.Type() == Prospero::ImageType::kCube &&
	                     descriptor.Width5() == descriptor.Height5() &&
	                     descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_2d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray &&
	    ((!resource.cube && is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth()) ||
	     is_cube);
	const bool is_3d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
	                   descriptor.Type() == Prospero::ImageType::kColor3D &&
	                   descriptor.BaseArray5() == 0;
	TileTextureBlockLayout tile_layout {};
	bool                   supported_tile = false;
	switch (tile) {
		case Prospero::TileMode::kLinear: supported_tile = true; break;
		case Prospero::TileMode::kDepth:
			supported_tile =
			    !resource.read && !Prospero::IsFmaskTextureFormat(descriptor.Format()) &&
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard256B:
			supported_tile =
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, is_3d, tile_layout);
			break;
		case Prospero::TileMode::kRenderTarget:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		default: break;
	}
	const auto swizzle = descriptor.DstSelXYZW();
	const bool supported_swizzle =
	    IsValidImageSwizzle(swizzle) &&
	    (swizzle == DstSel(4, 5, 6, 7) || !resource.read || resource.atomic);
	return (is_1d || is_1d_array || is_2d || is_2d_array || is_3d) && supported_tile &&
	       descriptor.BaseLevel() <= descriptor.LastLevel() &&
	       descriptor.MinLod() == 0 && supported_swizzle && descriptor.BCSwizzle() == 0 &&
	       !descriptor.MsaaDepth();
}

static bool IsSupportedStorageTextureEncoding(const ShaderRecompiler::IR::ImageResource& resource,
                                              const ShaderTextureResource& descriptor) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	constexpr uint32_t field5_expected      = 0x00700000u;
	constexpr uint32_t field5_max_mip_mask  = 0x000000f0u;
	const uint32_t     expected_field3 = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     expected_field4 =
	    descriptor.Depth() | (static_cast<uint32_t>(descriptor.BaseArray5()) << 16u);
	const bool common = (descriptor.fields[1] & field1_reserved_mask) == 0 &&
	                    (descriptor.fields[2] & field2_reserved_mask) == 0 &&
	                    descriptor.fields[3] == expected_field3;
	if (resource.r128) {
		return common && descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
		       descriptor.fields[6] == 0 && descriptor.fields[7] == 0;
	}
	return common && descriptor.fields[4] == expected_field4 &&
	       (descriptor.fields[5] & ~field5_max_mip_mask) == field5_expected;
}

void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size) {
	const auto format        = descriptor.Format();
	const bool resource_ok   = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok   = IsSupportedStorageTextureEncoding(resource, descriptor);
	const bool uint_resource    = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const bool raw_sint_storage = format == Prospero::BufferFormat::k32SInt && uint_resource &&
	                              resource.written && !resource.read && !resource.atomic;
	const bool raw_atomic_storage = resource.atomic && uint_resource &&
	                                (format == Prospero::BufferFormat::k32UInt ||
	                                 format == Prospero::BufferFormat::k32SInt ||
	                                 format == Prospero::BufferFormat::k32Float);
	const auto numeric_class = Prospero::SampledTextureNumericClass(format);
	const bool raw_float_atomic = format == Prospero::BufferFormat::k32Float && uint_resource &&
	                              resource.atomic && !resource.atomic64;
	const bool format_ok =
	    raw_sint_storage || raw_float_atomic || raw_atomic_storage ||
	    (numeric_class != Prospero::TextureNumericClass::Unsupported &&
	     numeric_class != Prospero::TextureNumericClass::Sint &&
	     uint_resource == (numeric_class == Prospero::TextureNumericClass::Uint) &&
	     (!resource.atomic || format == (resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
	                                                       : Prospero::BufferFormat::k32UInt)));
	if (resource_ok && descriptor_ok && encoding_ok && format_ok && size != 0) {
		return;
	}
	EXIT("unsupported storage texture: resource=%d descriptor=%d encoding=%d format=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u atomic=%d compare=%d "
	     "base_level=%u last_level=%u max_mip=%u min_lod=%u base_array=%u bc=%u msaa=%d "
	     "depth_tile_bpe=%u swizzle_ok=%d "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " extent=%ux%ux%u type=%u format=%u tile=%u swizzle=0x%03x read=%d written=%d "
	     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, descriptor_ok, encoding_ok, format_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.atomic, resource.depth_compare,
	     descriptor.BaseLevel(), descriptor.LastLevel(), descriptor.MaxMip(), descriptor.MinLod(),
	     descriptor.BaseArray5(), descriptor.BCSwizzle(), descriptor.MsaaDepth(),
	     Prospero::RenderTargetBytesPerElement(format),
	     IsValidImageSwizzle(descriptor.DstSelXYZW()), descriptor.Base40(), size,
	     static_cast<uint32_t>(descriptor.Width5()) + 1u,
	     static_cast<uint32_t>(descriptor.Height5()) + 1u,
	     static_cast<uint32_t>(descriptor.Depth()) + 1u, static_cast<uint32_t>(descriptor.Type()),
	     static_cast<uint32_t>(format), static_cast<uint32_t>(descriptor.TileMode()),
	     descriptor.DstSelXYZW(), resource.read, resource.written, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static TextureCache::ImageDesc NullTextureDesc(const ShaderRecompiler::IR::ImageResource& resource,
                                               TextureCache::BindingType                  binding) {
	TextureCache::ImageDesc desc {};
	switch (resource.numeric_class) {
		case Prospero::TextureNumericClass::Float:
			desc.info.guest_format = Prospero::BufferFormat::k32Float;
			break;
		case Prospero::TextureNumericClass::Uint:
			desc.info.guest_format = resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
			                                         : Prospero::BufferFormat::k32UInt;
			break;
		case Prospero::TextureNumericClass::Sint:
			desc.info.guest_format = Prospero::BufferFormat::k32SInt;
			break;
		default: EXIT("null image has unsupported numeric class\n");
	}
	// The view must match the dimension the shader declares (Vulkan requires the view type to
	// match the OpTypeImage), so 1D and 3D slots get 1D and 3D null images.
	auto image_type = Prospero::ImageType::kColor2D;
	auto view_type  = vk::ImageViewType::e2D;
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
			image_type = Prospero::ImageType::kColor1D;
			view_type  = vk::ImageViewType::e1D;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
			image_type = Prospero::ImageType::kColor1D;
			view_type  = vk::ImageViewType::e1DArray;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view_type = vk::ImageViewType::e2DArray;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			image_type = Prospero::ImageType::kColor3D;
			view_type  = vk::ImageViewType::e3D;
			break;
		default: break;
	}
	desc.info.pixel_format    = VulkanFormat(desc.info.guest_format);
	desc.info.type            = image_type;
	desc.info.extent          = {1, 1, 1};
	desc.info.resources       = {1, 1};
	desc.info.bytes_per_block = Prospero::NumBytesPerElement(desc.info.guest_format);
	desc.info.samples         = 1;
	desc.info.mip_layout[0]   = {0, 0, 1, 1};
	desc.view_info.format     = resource.atomic64 ? vk::Format::eR64Uint : desc.info.pixel_format;
	desc.view_info.type       = view_type;
	desc.view_info.aspect     = vk::ImageAspectFlagBits::eColor;
	desc.view_info.usage      = binding == TextureCache::BindingType::Storage
	                                ? vk::ImageUsageFlagBits::eStorage
	                                : vk::ImageUsageFlagBits::eSampled;
	desc.type                 = binding;
	return desc;
}

static void PopulateTextureMipLayout(ImageInfo& info) {
	if (info.IsVolume() && info.tile_mode != Prospero::TileMode::kLinear) {
		TileSurfaceLayout            surface {};
		const TileSurfaceDescription description {
		    info.guest_format,  info.tile_mode,    TileSurfaceDimension::Dim3D, info.extent.width,
		    info.extent.height, info.extent.depth, info.resources.levels,       1};
		if (!TileGetTiledTextureLayout(description, surface)) {
			EXIT("unsupported normalized volume texture layout\n");
		}
		for (uint32_t level = 0; level < info.resources.levels; level++) {
			const auto& mip        = surface.mips[level];
			info.mip_layout[level] = {
			    mip.offset,
			    mip.size,
			    mip.padded_width,
			    mip.padded_height,
			};
		}
		return;
	}

	TileSizeOffset levels[16] {};
	TilePaddedSize padded[16] {};
	TileGetTextureSize(info.guest_format, info.extent.width, info.extent.height,
	                   info.resources.levels, info.tile_mode, nullptr, levels, padded);
	const auto texel_shift = info.IsBlock() ? 2u : 0u;
	for (uint32_t level = 0; level < info.resources.levels; level++) {
		const auto offset =
		    levels[level].src_size != 0 ? levels[level].src_offset : levels[level].offset;
		auto size = static_cast<uint64_t>(levels[level].src_size != 0 ? levels[level].src_size
		                                                              : levels[level].size);
		if (info.IsVolume()) {
			size *= std::max(info.extent.depth >> level, 1u);
		} else {
			size *= info.resources.layers;
		}
		info.mip_layout[level] = {
		    offset,
		    size,
		    padded[level].width >> texel_shift,
		    padded[level].height >> texel_shift,
		};
	}
}

static ImageViewInfo TextureViewInfo(const ShaderRecompiler::IR::ImageResource& resource,
                                     const ShaderTextureResource& descriptor, vk::Format format,
                                     const SurfaceFormatInfo& surface_format, bool storage,
                                     uint32_t view_levels, uint32_t image_layers) {
	ImageViewInfo view {};
	view.format      = format;
	view.aspect      = vk::ImageAspectFlagBits::eColor;
	view.base_level  = descriptor.BaseLevel();
	view.level_count = view_levels;
	if (descriptor.MinLod() > descriptor.LastLevel() * 256u) {
		EXIT("texture minimum LOD exceeds last mip level: min_lod=%u last_level=%u\n",
		     descriptor.MinLod(), descriptor.LastLevel());
	}
	const auto base_lod = view.base_level * 256u;
	if (descriptor.MinLod() > base_lod) {
		view.min_lod = descriptor.MinLod() - base_lod;
	}
	view.usage = storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled;
	view.mapping =
	    storage || surface_format.conversion_format != Prospero::BufferFormat::kInvalid
	        ? vk::ComponentMapping {}
	        : TextureGetComponentMapping(descriptor.DstSelXYZW(), surface_format.host_to_storage);
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
			view.type       = vk::ImageViewType::e1D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
			view.type       = vk::ImageViewType::e1DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			view.type        = vk::ImageViewType::e3D;
			view.base_layer  = 0;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view.type       = vk::ImageViewType::e2DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
			view.type       = vk::ImageViewType::e2D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		default: EXIT("unsupported texture view dimension\n");
	}
	return view;
}

static bool ResolveTextureMipView(const TileSurfaceDescription& description, bool metadata,
                                   uint32_t view_levels, uint32_t& levels, uint32_t& base_level) {
	TileSurfaceLayout physical {};
	TileSurfaceLayout view {};
	auto              view_description = description;
	view_description.levels            = levels;
	if (!TileGetTiledTextureLayout(description, physical) ||
	    !TileGetTiledTextureLayout(view_description, view)) {
		return false;
	}
	if (physical.first_tail_level == view.first_tail_level &&
	    physical.block_slice_size == view.block_slice_size &&
	    physical.total_size == view.total_size &&
	    std::equal(std::begin(physical.mips), std::begin(physical.mips) + description.levels,
	               std::begin(view.mips))) {
		return true;
	}
	if (metadata || ((description.layers > 1 || description.depth > 1) &&
	                 physical.block_slice_size != view.block_slice_size)) {
		return false;
	}
	// T# addresses the last mip. A view can select the same stored subresources
	// with different mip indices; inaccessible mips need no host representation.
	for (uint32_t base = 0; base + view_levels <= description.levels; ++base) {
		bool matches = true;
		for (uint32_t i = 0; i < view_levels; ++i) {
			const auto source = base_level + i;
			const auto target = base + i;
			if (physical.mips[target] != view.mips[source] ||
			    (target >= physical.first_tail_level) != (source >= view.first_tail_level) ||
			    std::max(description.width >> target, 1u) != std::max(description.width >> source, 1u) ||
			    std::max(description.height >> target, 1u) != std::max(description.height >> source, 1u) ||
			    std::max(description.depth >> target, 1u) != std::max(description.depth >> source, 1u)) {
				matches = false;
				break;
			}
		}
		if (matches) {
			levels     = description.levels;
			base_level = base;
			return true;
		}
	}
	return false;
}

namespace {

// ResolveTexture's description of a texture (tiling, sizes, mip layout, view) depends only on the
// T# and on how the shader uses the image: it was computed again for every bound image of every
// draw, 2.4 % of the GPU thread at the jungle prompt (DEBUGGING.md, 2026-10-05). The texture
// cache lookup that follows still runs every time. KYTY_TEXTURE_DESC_CACHE=0 turns this off.
struct TextureDescKey {
	std::array<uint32_t, 8> dwords {};
	uint32_t                resource_class = 0;
	uint32_t                numeric_class  = 0;
	uint32_t                dimension      = 0;
	uint32_t                mip_mode       = 0;
	uint32_t                mip_count      = 0;
	uint32_t                conversion     = 0;
	uint32_t                swizzle        = 0;
	uint32_t                flags          = 0;

	bool operator==(const TextureDescKey&) const = default;
};
static_assert(sizeof(TextureDescKey) == 64);

struct TextureDescKeyHash {
	size_t operator()(const TextureDescKey& key) const noexcept {
		return static_cast<size_t>(XXH3_64bits(&key, sizeof(key)));
	}
};

struct TextureDescEntry {
	TextureCache::ImageDesc desc;
	vk::Format              pixel_format      = vk::Format::eUndefined;
	vk::Format              view_format       = vk::Format::eUndefined;
	uint32_t                size              = 0;
	bool                    shader_conversion = false;
};

TextureDescKey MakeTextureDescKey(const ShaderRecompiler::IR::ImageResource&   resource,
                                  const ShaderRecompiler::IR::DescriptorValue& value) {
	TextureDescKey key;
	std::copy_n(value.dwords.begin(), 8, key.dwords.begin());
	key.resource_class = static_cast<uint32_t>(resource.resource_class);
	key.numeric_class  = static_cast<uint32_t>(resource.numeric_class);
	key.dimension      = static_cast<uint32_t>(resource.dimension);
	key.mip_mode       = static_cast<uint32_t>(resource.mip_mode);
	key.mip_count      = resource.mip_count;
	key.conversion     = static_cast<uint32_t>(resource.conversion_format);
	key.swizzle        = resource.shader_swizzle;
	key.flags = (resource.read ? 1u : 0u) | (resource.written ? 2u : 0u) |
	            (resource.atomic ? 4u : 0u) | (resource.depth_compare ? 8u : 0u) |
	            (resource.cube ? 16u : 0u) | (resource.r128 ? 32u : 0u) |
	            (resource.bindless ? 64u : 0u) | (value.dword_count << 8u);
	return key;
}

bool TextureDescCacheEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_TEXTURE_DESC_CACHE");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

} // namespace

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value) {
	if (resource.atomic64 && !m_context.GetGraphics().shader_image_int64_atomics_enabled) {
		EXIT("64-bit image atomics require shaderImageInt64Atomics\n");
	}
	auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const bool storage = resource.written;
	if (storage) {
		ValidateStorageImageResource(resource);
	}

	auto& texture_cache = m_context.GetTextureCache();
	if (descriptor.IsNull()) {
		auto       desc = NullTextureDesc(resource, storage ? TextureCache::BindingType::Storage
		                                                    : TextureCache::BindingType::Texture);
		const auto id   = texture_cache.FindImage(desc);
		return {id, nullptr, std::move(desc)};
	}

	thread_local std::unordered_map<TextureDescKey, TextureDescEntry, TextureDescKeyHash> cache;
	const bool use_cache = TextureDescCacheEnabled();
	const auto key       = use_cache ? MakeTextureDescKey(resource, value) : TextureDescKey {};
	if (use_cache) {
		if (const auto it = cache.find(key); it != cache.end()) {
			return FindResolvedTexture(resource, descriptor, it->second.desc,
			                           it->second.shader_conversion, it->second.pixel_format,
			                           it->second.view_format, it->second.size);
		}
	}

	const auto address         = descriptor.Base40();
	const auto width           = static_cast<uint32_t>(descriptor.Width5()) + 1u;
	const auto height          = static_cast<uint32_t>(descriptor.Height5()) + 1u;
	const auto base_level      = descriptor.BaseLevel();
	const auto last_level      = descriptor.LastLevel();
	const auto type            = TextureType(descriptor);
	const bool multisampled    = IsMultisampledTexture(type);
	const auto max_mip         = resource.r128 ? last_level : descriptor.MaxMip();
	const auto physical_levels = multisampled ? 1u : static_cast<uint32_t>(max_mip) + 1u;
	// IMAGE_STORE addresses BASE_LEVEL; only IMAGE_STORE_MIP selects other view mips.
	const bool single_storage_mip =
	    storage && resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic;
	const auto view_levels = multisampled || single_storage_mip
	                             ? 1u
	                             : static_cast<uint32_t>(last_level - base_level) + 1u;
	auto levels =
	    multisampled ? 1u : std::max(physical_levels, base_level + view_levels);
	const auto tile       = descriptor.TileMode();
	const bool depth_tile = tile == Prospero::TileMode::kDepth;
	const bool msaa_tile  = depth_tile || tile == Prospero::TileMode::kRenderTarget;
	const bool msaa_array = type == Prospero::ImageType::kColor2DMsaaArray;
	if ((!multisampled && base_level > last_level) ||
	    (multisampled &&
	     (base_level != 0 || last_level == 0 || last_level > 3 || max_mip != last_level ||
	      !msaa_tile || (descriptor.MsaaDepth() && !depth_tile) ||
	      (!msaa_array && (descriptor.Depth() != 0 || descriptor.BaseArray5() != 0))))) {
		EXIT("unsupported texture mip view: base=%u last=%u levels=%u max=%u type=%u tile=%u "
		     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d "
		     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     base_level, last_level, levels, descriptor.MaxMip(),
		     static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(tile),
		     static_cast<uint32_t>(resource.resource_class),
		     static_cast<uint32_t>(resource.numeric_class),
		     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
		     resource.read, resource.written, descriptor.fields[0], descriptor.fields[1],
		     descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		     descriptor.fields[6], descriptor.fields[7]);
	}
	const auto samples = multisampled ? 1u << last_level : 1u;
	const auto depth          = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	const auto format         = descriptor.Format();
	const auto surface_format = TextureGetSurfaceFormatInfo(format);
	const bool shader_conversion =
	    surface_format.conversion_format != Prospero::BufferFormat::kInvalid;
	const bool sampled_numeric_class =
	    storage || resource.numeric_class == Prospero::SampledTextureNumericClass(format);
	if (!storage && resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	    !sampled_numeric_class) {
		EXIT("sampled image numeric class mismatch: numeric=%u format=%u addr=0x%016" PRIx64 "\n",
		     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(format), address);
	}

	const bool    volume       = type == Prospero::ImageType::kColor3D;
	const bool    layered      = type == Prospero::ImageType::kColor1DArray ||
	                             type == Prospero::ImageType::kColor2DArray ||
	                             type == Prospero::ImageType::kColor2DMsaaArray;
	const auto    image_layers = layered ? depth : 1u;
	auto          view_base    = static_cast<uint32_t>(base_level);
	if (levels > physical_levels) {
		const TileSurfaceDescription physical {
		    format, tile, volume ? TileSurfaceDimension::Dim3D : TileSurfaceDimension::Dim2D,
		    width, height, volume ? depth : 1u, physical_levels, image_layers};
		if (!ResolveTextureMipView(physical, !resource.r128 && descriptor.MetaCompress(),
		                           view_levels, levels, view_base)) {
			EXIT("unsupported texture mip view changes physical layout: base=%u last=%u max=%u "
			     "extent=%ux%ux%u tile=%u\n",
			     base_level, last_level, max_mip, width, height, depth,
			     static_cast<uint32_t>(tile));
		}
	}
	uint32_t      pitch = 0;
	TileSizeAlign size {};
	if (multisampled) {
		const auto bytes = Prospero::NumBytesPerElement(format);
		pitch            = depth_tile ? TileGetDepthPitch(width, bytes, last_level)
		                              : TileGetRenderTargetPitch(width, bytes, last_level);
		if (pitch == 0 || !TileGetRenderTargetSize(width, height, pitch, bytes, size, last_level) ||
		    size.size > UINT32_MAX / image_layers) {
			EXIT("unsupported multisample texture layout\n");
		}
		size.size *= image_layers;
	} else {
		pitch = TileGetTexturePitch(format, width, tile);
		TileGetTextureTotalSize(format, width, height, volume ? depth : image_layers,
		                        physical_levels, tile, volume, size);
	}
	EXIT_NOT_IMPLEMENTED(size.size == 0 || size.align == 0 ||
	                     (address & (static_cast<uint64_t>(size.align) - 1u)) != 0);
	if (storage) {
		ValidateStorageTexture(resource, descriptor, size.size);
	}

	auto pixel_format = surface_format.vk_format;
	if (resource.depth_compare) {
		if (const auto* depth_format = FindGuestDepthFormatPolicy(format)) {
			pixel_format = depth_format->depth_attachment_format;
		}
	}
	// Raw sint stores and all image atomics run as uint operations on the texel bits
	const auto storage_view_format = resource.atomic64 ? vk::Format::eR64Uint
	                                 : storage && (resource.atomic ||
	                                               format == Prospero::BufferFormat::k32SInt)
	                                     ? vk::Format::eR32Uint
	                                     : SrgbStorageViewFormat(pixel_format);
	const auto view_format         = storage && storage_view_format != vk::Format::eUndefined
	                                     ? storage_view_format
	                                     : pixel_format;
	const auto block_bytes         = Prospero::BlockCompressedBytesPerBlock(format);
	TextureCache::ImageDesc desc {};
	desc.info.data         = {address, size.size};
	desc.info.pixel_format = pixel_format;
	desc.info.guest_format = format;
	desc.info.type         = TextureBaseType(type);
	desc.info.extent       = {width, height, volume ? depth : 1u};
	desc.info.resources    = {levels, image_layers};
	desc.info.pitch        = pitch;
	desc.info.bytes_per_block =
	    block_bytes != 0 ? block_bytes : Prospero::NumBytesPerElement(format);
	desc.info.samples   = samples;
	desc.info.tile_mode = tile;
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth &&
	    !desc.info.IsDepth()) {
		TileSizeAlign metadata_size {};
		(void)TileGetDccSize(width, height, volume ? depth : image_layers,
		                     desc.info.bytes_per_block, physical_levels, tile, metadata_size,
		                     std::countr_zero(samples));
		desc.info.metadata.kind          = ImageMetadataKind::Dcc;
		desc.info.metadata.range         = {descriptor.MetaAddr() << 8u, metadata_size.size};
		desc.info.metadata.dcc_alpha_msb = descriptor.DccAlphaPos();
	}
	if (samples > 1) {
		desc.info.mip_layout[0] = {0, size.size, pitch, height};
	} else {
		PopulateTextureMipLayout(desc.info);
	}
	desc.view_info = TextureViewInfo(resource, descriptor, view_format, surface_format, storage,
	                                 view_levels, desc.info.resources.layers);
	desc.view_info.base_level = view_base;
	desc.type = storage ? TextureCache::BindingType::Storage : TextureCache::BindingType::Texture;

	if (use_cache) {
		if (cache.size() >= 65536) {
			cache.clear();
		}
		cache.emplace(key, TextureDescEntry {desc, pixel_format, view_format, size.size,
		                                     shader_conversion});
	}
	return FindResolvedTexture(resource, descriptor, std::move(desc), shader_conversion,
	                           pixel_format, view_format, size.size);
}

// ResolveTexture's texture cache lookup for a description (FindImage may adjust it).
TextureBinding RenderExecutor::FindResolvedTexture(const ShaderRecompiler::IR::ImageResource& resource,
                                                   const ShaderTextureResource&               descriptor,
                                                   TextureCache::ImageDesc desc,
                                                   bool shader_conversion, vk::Format pixel_format,
                                                   vk::Format view_format, uint32_t size) {
	auto&      texture_cache       = m_context.GetTextureCache();
	const bool storage             = resource.written;
	auto       id                  = texture_cache.FindImage(desc, shader_conversion);
	auto*      image               = &texture_cache.GetImage(id);
	const bool stencil_association = static_cast<bool>(image->depth_id);
	if (stencil_association) {
		id    = image->depth_id;
		image = &texture_cache.GetImage(id);
	} else if (image->info.IsDepth()) {
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		ValidateSampledDepthBinding(resource, descriptor, *image, pixel_format, size);
	} else if (storage) {
		ValidateStorageColorView(image->info.pixel_format, view_format, descriptor.DstSelXYZW());
	} else {
		(void)SelectSampledColorView(image->info.pixel_format, pixel_format,
		                             descriptor.DstSelXYZW());
	}
	return {id, nullptr, std::move(desc)};
}

static vk::Sampler NativeSampler(RenderContext&                       context,
                                 const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                 uint32_t index,
                                 const ShaderRecompiler::IR::DescriptorValue& value) {
	auto        descriptor = DecodeNativeDescriptor<ShaderSamplerResource>(value);
	const auto& sampler = program.info.samplers[index];
	if (!sampler.depth_compare) {
		descriptor.fields[0] &= ~(0x7u << 12u);
	}
	if (sampler.force_point_filtering) {
		descriptor.SetPointFiltering();
	}
	return context.GetSamplerCache().GetSampler(descriptor, sampler.integer_border);
}

static vk::DescriptorBufferInfo NativeUpload(RenderContext&            context,
                                             std::span<const uint32_t> data) {
	EXIT_IF(data.empty());
	auto& command_buffer = context.GetCommandScheduler().Current();
	EXIT_IF(command_buffer.IsInvalid());
	auto&      buffer = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset = buffer.Copy(data.data(), data.size_bytes(), 256);
	return {buffer.Handle(), offset, data.size_bytes()};
}

// The flat slots the refresh left to the GPU (SrtGpuFill): their guest bytes are copied, where
// the buffer cache holds them, into the uploaded flat buffer before the command, behind the
// writes that produced them. Reading them on the host instead waited for the GPU (2.8 ms a frame
// for one dispatch in the jungle).
static void FillFlatSlotsOnGpu(RenderContext& context, const vk::DescriptorBufferInfo& flat,
                               std::span<const ShaderRecompiler::IR::SrtGpuFill> fills) {
	if (fills.empty()) {
		return;
	}
	auto& cache   = context.GetBufferCache();
	auto& command = context.GetCommandScheduler().Current();
	command.EndRendering();
	const auto      recorder = command.Recorder();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	recorder.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                         vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
	                         nullptr);
	for (const auto& fill: fills) {
		auto [buffer, offset] = cache.ObtainBuffer(fill.address, sizeof(uint32_t), false);
		(void)offset;
		const vk::BufferCopy copy {buffer->Offset(fill.address),
		                           flat.offset + uint64_t {fill.flat_offset} * sizeof(uint32_t),
		                           sizeof(uint32_t)};
		recorder.copyBuffer(buffer->Handle(), flat.buffer, 1, &copy);
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eUniformRead;
	recorder.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                         vk::PipelineStageFlagBits::eAllCommands, {}, 1, &after, 0, nullptr, 0,
	                         nullptr);
}

void RenderExecutor::BindImage(ImageId id, bool storage) {
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (image.info.data.Empty()) {
		return;
	}
	if (image.binding.is_bound) {
		image.binding.force_general |= image.binding.shader_write != storage;
	}
	image.binding.is_bound = true;
	image.binding.shader_write |= storage;
	m_bound_images.push_back(id);
}

void RenderExecutor::BindRenderTarget(ImageId id) {
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (!image.binding.is_target) {
		NoteBindlessStateChange(image);
	}
	image.binding.is_target = true;
	m_bound_images.push_back(id);
}

void RenderExecutor::ResetBindings() {
	for (const auto id: m_bound_images) {
		if (auto* image = m_context.GetTextureCache().m_slot_images.try_get(id); image != nullptr) {
			if (image->binding.is_target) {
				NoteBindlessStateChange(*image);
			}
			image->binding = {};
		}
	}
	m_bound_images.clear();
}

// Diagnostics: the first time a draw uses a bindless heap,
// read its texture descriptors and log how many are valid, of which types, and roughly how
// much memory they would take as host textures. Nothing is created.
static void SurveyBindlessHeap(const ShaderRecompiler::IR::BindlessHeapUse& use) {
	constexpr uint64_t MaxEntries = 1u << 20u;
	const uint64_t     span       = use.size > use.table_offset ? use.size - use.table_offset : 0;
	const uint64_t     entries    = std::min<uint64_t>(span / 32u, MaxEntries);
	uint64_t valid = 0, bytes = 0, unreadable = 0;
	std::array<uint64_t, 16> types {};
	std::vector<uint32_t> chunk(8u * 1024u);
	for (uint64_t first = 0; first < entries; first += chunk.size() / 8u) {
		const auto count   = std::min<uint64_t>(chunk.size() / 8u, entries - first);
		const auto address = use.base + use.table_offset + first * 32u;
		if (!Libs::LibKernel::Memory::TryReadBacking(address, chunk.data(), count * 32u)) {
			unreadable += count;
			continue;
		}
		for (uint64_t i = 0; i < count; i++) {
			ShaderRecompiler::IR::DescriptorValue value {};
			value.dword_count = 8u;
			std::copy_n(chunk.begin() + static_cast<ptrdiff_t>(i * 8u), 8, value.dwords.begin());
			const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
			if (descriptor.IsNull()) {
				continue;
			}
			valid++;
			types[static_cast<uint32_t>(descriptor.Type()) & 15u]++;
			const uint64_t width  = static_cast<uint64_t>(descriptor.Width5()) + 1u;
			const uint64_t height = static_cast<uint64_t>(descriptor.Height5()) + 1u;
			const uint64_t bpe    = std::max<uint32_t>(Prospero::NumBytesPerElement(descriptor.Format()), 1u);
			bytes += width * height * bpe * 4u / 3u;
		}
	}
	std::string histogram;
	for (uint32_t t = 0; t < types.size(); t++) {
		if (types[t] != 0) {
			histogram += fmt::format(" t{}={}", t, types[t]);
		}
	}
	LOGF("Bindless heap: base=0x%016" PRIx64 " size=0x%" PRIx64 " offset=%u entries=%" PRIu64
	     " valid=%" PRIu64 " unreadable=%" PRIu64 " est=%" PRIu64 "MB types:%s\n",
	     use.base, use.size, use.table_offset, entries, valid, unreadable, bytes >> 20u,
	     histogram.c_str());
}

static uint32_t BindlessBindingFor(const ShaderRecompiler::IR::ImageResource& resource) {
	using ShaderRecompiler::Decoder::ImageDimension;
	if (resource.cube) {
		return ShaderRecompiler::IR::BindlessImagesCube;
	}
	if (resource.dimension == ImageDimension::Dim3D) {
		return ShaderRecompiler::IR::BindlessImages3D;
	}
	if (resource.dimension == ImageDimension::Dim2DArray ||
	    resource.dimension == ImageDimension::Dim2DMsaaArray) {
		return ShaderRecompiler::IR::BindlessImages2DArray;
	}
	return ShaderRecompiler::IR::BindlessImages2D;
}

// A heap entry the bindless array of this binding can hold: a float-sampled, single-sample
// texture of a matching shape. Anything else stays the placeholder rather than reaching
// ResolveTexture, which exits on descriptors a draw would never bind that way.
static bool BindlessCompatible(const ShaderTextureResource& descriptor, uint32_t binding) {
	if (descriptor.IsNull() ||
	    Prospero::SampledTextureNumericClass(descriptor.Format()) !=
	        Prospero::TextureNumericClass::Float) {
		return false;
	}
	switch (descriptor.Type()) {
		case Prospero::ImageType::kColor2D:
			return binding == ShaderRecompiler::IR::BindlessImages2D ||
			       binding == ShaderRecompiler::IR::BindlessImages2DArray;
		case Prospero::ImageType::kColor2DArray:
			return binding == ShaderRecompiler::IR::BindlessImages2D ||
			       binding == ShaderRecompiler::IR::BindlessImages2DArray;
		case Prospero::ImageType::kCube: return binding == ShaderRecompiler::IR::BindlessImagesCube;
		case Prospero::ImageType::kColor3D: return binding == ShaderRecompiler::IR::BindlessImages3D;
		default: return false;
	}
}

bool RenderExecutor::ResolveBindlessKey(BindlessTable::Heap& heap, uint32_t key) {
	auto&                                 table = m_context.GetBindlessTable();
	ShaderRecompiler::IR::DescriptorValue value {};
	value.dword_count  = 8u;
	const auto address = heap.base + heap.table_offset + static_cast<uint64_t>(key) * 32u;
	heap.settled[key]  = 1;
	heap.stale[key]    = 0;
	heap.descriptors[key] = {};
	if (!Libs::LibKernel::Memory::TryReadBacking(address, value.dwords.data(), 32u)) {
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	std::copy_n(value.dwords.data(), 8, heap.descriptors[key].begin());
	const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	if (!BindlessCompatible(descriptor, heap.binding)) {
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	auto binding = ResolveTexture(heap.resource, value);
	auto& texture_cache = m_context.GetTextureCache();
	auto* image = texture_cache.m_slot_images.try_get(binding.image_id);
	if (image == nullptr || image->info.data.Empty()) {
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	BindImage(binding.image_id, false);
	const auto view = texture_cache.FindTexture(binding.image_id, binding.desc);
	image           = texture_cache.m_slot_images.try_get(binding.image_id);
	if (image != nullptr && !image->registered) {
		// Dropped by the cache (its destruction waits for the GPU): a slot holding its view would
		// outlive it, since its unregistration is over. The key is asked for again next frame.
		heap.settled[key] = 0;
		table.SetTranslation(heap, key, ShaderRecompiler::IR::BindlessPending);
		return false;
	}
	const auto slot = table.AllocateSlot(heap.binding);
	if (image == nullptr || view == nullptr || slot == 0) {
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	const auto layout = image->info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
	                                          : vk::ImageLayout::eShaderReadOnlyOptimal;
	table.WriteSlot(heap.binding, slot, view, layout);
	heap.slots[key]         = slot;
	heap.images[key]        = binding.image_id;
	image->bindless_pinned  = true;
	image->usage.texture    = true;
	heap.resolved.push_back(binding.image_id);
	heap.unchecked.push_back(binding.image_id);
	table.AddImageReference(binding.image_id, heap, key);
	table.SetTranslation(heap, key, slot);
	return true;
}

// Engine-method counters (SyncBindlessHeaps), logged and reset with the heap statistics.
struct BindlessEngineStats {
	std::atomic<uint64_t> syncs {0};
	std::atomic<uint64_t> written_bytes {0};
	std::atomic<uint64_t> view_changes {0};
	std::atomic<uint64_t> resource_changes {0};
	std::atomic<uint64_t> resettled {0};
	std::atomic<uint64_t> eager_settled {0};
	std::atomic<uint64_t> eager_failed {0};
	std::atomic<uint64_t> eager_paused {0};
	std::atomic<uint64_t> requested {0}; // keys draws asked for through feedback
	std::atomic<uint64_t> released {0};  // rewritten keys no draw had sampled: left to E3
	std::atomic<uint64_t> eager_capped {0}; // keys E3 stopped settling (MaxEagerSettles)
	std::atomic<uint64_t> kept_stale {0};   // view changes of unsampled keys left as they were
	std::atomic<uint64_t> stale_settled {0}; // such keys settled again once a draw sampled them
};
constexpr uint8_t MaxEagerSettles = 2;
static BindlessEngineStats g_bindless_engine_stats;

// With the engine method, every entry the guest has written is settled before draws ask for it
// (E3), at most KYTY_BINDLESS_ENGINE_BUDGET (default 256) per presented frame. On by default;
// KYTY_BINDLESS_ENGINE_EAGER=0 settles keys only when draws ask for them.
// Eager settling (E3) is off by default since 2026-10-08: in a new game it paused under the
// texture cache's bound for nearly the whole session, yet kept the ~3 GB it settled early (heap
// textures 9.3 GB against 6.0 GB without it, peak video memory 20.2 against 17.1 GB), with as
// many late requests (544 against 620 in the first 5,500 frames) and the same or better frame
// rate (prompt 21.6 against 22.1 fps). Textures load when a draw samples them; rewrites of keys
// already settled are still followed (E2). KYTY_BINDLESS_ENGINE_EAGER=1 turns it on.
static bool BindlessEngineEager() {
	static const bool eager = [] {
		const char* value = std::getenv("KYTY_BINDLESS_ENGINE_EAGER");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return eager;
}

// KYTY_BINDLESS_ENGINE_BUDGET_MB: the image memory eager settling may touch per presented frame
// (default 32). A count alone let a new area's load settle ~9,800 textures in 300 frames, large
// ones uploaded and detiled together, and the frames stalled; engines stream with a bandwidth.
static uint64_t BindlessEngineBudgetBytes() {
	static const uint64_t bytes = [] {
		const char* value  = std::getenv("KYTY_BINDLESS_ENGINE_BUDGET_MB");
		const auto  parsed = value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
		return (parsed != 0 ? parsed : 32ull) << 20u;
	}();
	return bytes;
}

static uint32_t BindlessEngineBudget() {
	static const uint32_t budget = [] {
		const char* value = std::getenv("KYTY_BINDLESS_ENGINE_BUDGET");
		const auto  parsed = value != nullptr ? std::strtoul(value, nullptr, 10) : 0ul;
		return parsed != 0 ? static_cast<uint32_t>(parsed) : 256u;
	}();
	return budget;
}

// KYTY_BINDLESS_ENGINE_VERIFY=1 (with KYTY_BINDLESS_ENGINE=1): the rolling scan keeps running and
// logs every rewritten entry the write watch did not report.
static bool BindlessEngineVerify() {
	static const bool verify = [] {
		const char* value = std::getenv("KYTY_BINDLESS_ENGINE_VERIFY");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return verify;
}

void RenderExecutor::ReleaseBindlessKey(BindlessTable::Heap& heap, uint32_t key) {
	auto&      table         = m_context.GetBindlessTable();
	auto&      texture_cache = m_context.GetTextureCache();
	const auto old_image     = heap.images[key];
	if (table.ReleaseKey(heap, key)) {
		if (auto* image = texture_cache.m_slot_images.try_get(old_image);
		    image != nullptr && image->bindless_pinned) {
			NoteBindlessStateChange(*image);
			image->bindless_pinned = false;
			if (BindlessEngineEnabled()) {
				texture_cache.ReleaseDroppedByGuest(old_image);
			}
		}
	}
}

bool RenderExecutor::ResettleBindlessKey(BindlessTable::Heap& heap, uint32_t key) {
	ReleaseBindlessKey(heap, key);
	return ResolveBindlessKey(heap, key);
}

void RenderExecutor::SyncBindlessHeaps() {
	auto& table = m_context.GetBindlessTable();
	if (!BindlessEngineEnabled() || !table.Enabled()) {
		return;
	}
	if (table.HasWrittenRanges()) {
		SyncWrittenBindlessEntries();
	}
	if (BindlessEngineEager()) {
		SettleBindlessHeaps();
	}
}

void RenderExecutor::SettleBindlessHeaps() {
	auto& table = m_context.GetBindlessTable();
	auto& stats = g_bindless_engine_stats;
	const auto frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
	if (frame != m_bindless_eager_frame) {
		m_bindless_eager_frame = frame;
		m_bindless_eager_left  = BindlessEngineBudget();
		m_bindless_eager_bytes = BindlessEngineBudgetBytes();
		// Settling ahead stops while the device is at 80 % of its budget, below the 90 % at which
		// the texture cache starts evicting, so the two do not chase each other. A new game's
		// jungle load filled the 32 GB card when only the images' own threshold was checked.
		m_bindless_eager_device_full = m_context.GetGraphics().DeviceMemoryAtLeast(80);
		// Once a frame: stale keys a draw has sampled since are settled again (finer mips).
		uint32_t stale_settled = 0;
		for (auto& heap: table.Heaps()) {
			std::erase_if(heap.stale_keys, [&](uint32_t key) {
				if (heap.stale[key] == 0) {
					return true; // settled again meanwhile
				}
				if (m_bindless_eager_left == 0 || !table.WasSampled(heap, key)) {
					return false;
				}
				m_bindless_eager_left--;
				heap.stale[key] = 0;
				stale_settled += ResettleBindlessKey(heap, key) ? 1u : 0u;
				return true;
			});
		}
		stats.stale_settled.fetch_add(stale_settled, std::memory_order_relaxed);
	}
	// Engine method, E3: entries no draw has asked for yet are settled too, as an engine creates a
	// texture's descriptor when it loads the texture. A pass stops at the budget and goes on from
	// its cursor at the next submission; the request path covers what is not settled yet. Paused
	// while the texture cache is short of memory, so eviction and settling do not chase each other.
	if (std::ranges::all_of(table.Heaps(), [](const auto& heap) { return heap.eager_complete; })) {
		return;
	}
	if (m_context.GetTextureCache().UnderPressure() || m_bindless_eager_device_full) {
		stats.eager_paused.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	if (m_bindless_eager_left == 0) {
		return;
	}
	KYTY_PROFILER_FUNCTION();
	constexpr uint32_t Window   = 2048;
	uint32_t&          budget   = m_bindless_eager_left;
	uint32_t           settled  = 0;
	uint32_t           failed   = 0;
	uint32_t           capped   = 0;
	for (auto& heap: table.Heaps()) {
		if (heap.eager_complete || budget == 0) {
			continue;
		}
		const auto keys = std::min(heap.guest_entries, static_cast<uint32_t>(heap.settled.size()));
		const auto base = heap.base + heap.table_offset;
		uint32_t   key  = heap.eager_cursor < keys ? heap.eager_cursor : 0u;
		uint32_t   seen = 0;
		bool       marked = false;
		while (seen < keys && budget != 0) {
			const auto count = std::min(Window, keys - key);
			m_bindless_window.resize(count);
			uint32_t   looked   = count;
			const bool readable = Libs::LibKernel::Memory::TryReadBacking(
			    base + uint64_t {key} * 32u, m_bindless_window.data(), uint64_t {count} * 32u);
			for (uint32_t i = 0; readable && i < count; i++) {
				if (budget == 0) {
					looked = i;
					break;
				}
				if (heap.settled[key + i] != 0) {
					continue;
				}
				if (heap.eager_count[key + i] >= MaxEagerSettles) {
					continue; // its image keeps being replaced: the request path has it
				}
				ShaderTextureResource descriptor {};
				std::copy_n(m_bindless_window[i].data(), 8, descriptor.fields);
				if (descriptor.IsNull() || !BindlessCompatible(descriptor, heap.binding)) {
					continue;
				}
				budget--;
				if (++heap.eager_count[key + i] == MaxEagerSettles) {
					capped++;
				}
				if (ResolveBindlessKey(heap, key + i)) {
					if (const auto* image =
					        m_context.GetTextureCache().m_slot_images.try_get(heap.images[key + i]);
					    image != nullptr) {
						const auto size = image->AccountedSize();
						m_bindless_eager_bytes -= std::min(m_bindless_eager_bytes, size);
						if (m_bindless_eager_bytes == 0) {
							budget = 0; // this frame's bandwidth is spent
						}
					}
					// Not asked for by a draw: marked until one samples it (WasSampled), which also
					// lets the usage probe count it as unused under memory pressure.
					table.MarkUnsampled(heap, key + i);
					marked = true;
					settled++;
				} else {
					failed++;
				}
			}
			seen += looked;
			key = key + looked >= keys ? 0u : key + looked;
		}
		heap.eager_cursor = key;
		heap.eager_complete = seen >= keys && budget != 0;
		if (marked) {
			table.FlushFeedback(heap);
		}
	}
	stats.eager_settled.fetch_add(settled, std::memory_order_relaxed);
	stats.eager_failed.fetch_add(failed, std::memory_order_relaxed);
	stats.eager_capped.fetch_add(capped, std::memory_order_relaxed);
}

void RenderExecutor::SyncWrittenBindlessEntries() {
	auto& table = m_context.GetBindlessTable();
	KYTY_PROFILER_FUNCTION();
	// Engine method, E2: the pages the guest wrote since the last submission, already watched
	// again. Every entry in them is compared with the T# its key was settled from.
	table.TakeWrittenRanges(m_bindless_written);
	// T# bits that select a view of the same texture (RDNA 2 ISA 8.2.6); see LogBindlessHeapStats.
	constexpr std::array<uint32_t, 8> ViewMask {0u,          0x000fff00u, 0u,          0x0e0fffffu,
	                                            0x1fff0000u, 0x027fff00u, 0x000003ffu, 0u};
	uint32_t view_changes = 0, resource_changes = 0, resettled = 0, released = 0, kept = 0;
	for (auto& heap: table.Heaps()) {
		const auto heap_begin = heap.base + heap.table_offset;
		const auto keys       = std::min(heap.guest_entries, static_cast<uint32_t>(heap.settled.size()));
		const auto heap_end   = heap_begin + uint64_t {keys} * 32u;
		for (const auto& [begin, end]: m_bindless_written) {
			const auto from = std::max(begin, heap_begin);
			const auto to   = std::min(end, heap_end);
			if (from >= to) {
				continue;
			}
			heap.eager_complete = false; // entries may have become valid
			const auto first = static_cast<uint32_t>((from - heap_begin) / 32u);
			const auto last  = static_cast<uint32_t>((to - heap_begin + 31u) / 32u);
			m_bindless_window.resize(last - first);
			if (!Libs::LibKernel::Memory::TryReadBacking(heap_begin + uint64_t {first} * 32u,
			                                              m_bindless_window.data(),
			                                              uint64_t {last - first} * 32u)) {
				continue;
			}
			for (uint32_t key = first; key < last; key++) {
				const auto& words = m_bindless_window[key - first];
				if (heap.descriptors[key] != words) {
					heap.eager_count[key] = 0; // the guest wrote it: E3 may settle it again
				}
				if (heap.settled[key] == 0 || heap.descriptors[key] == words) {
					continue;
				}
				bool resource = false;
				for (uint32_t i = 0; i < 8; i++) {
					resource |= ((heap.descriptors[key][i] ^ words[i]) & ~ViewMask[i]) != 0;
				}
				(resource ? resource_changes : view_changes)++;
				if (BindlessEngineEager() && !table.WasSampled(heap, key)) {
					// No draw has sampled it since it was settled.
					if (!resource) {
						// Streaming: the same texture with finer mips. The old view stays valid;
						// it is settled again when a draw samples the key.
						if (heap.stale[key] == 0) {
							heap.stale[key] = 1;
							heap.stale_keys.push_back(key);
						}
						kept++;
						continue;
					}
					// Another texture: the settle pass takes it within its budget, instead of
					// every rewrite of a load at once.
					ReleaseBindlessKey(heap, key);
					released++;
					continue;
				}
				resettled += ResettleBindlessKey(heap, key) ? 1u : 0u;
			}
		}
	}
	uint64_t bytes = 0;
	for (const auto& [begin, end]: m_bindless_written) bytes += end - begin;
	auto& stats = g_bindless_engine_stats;
	stats.syncs.fetch_add(1, std::memory_order_relaxed);
	stats.written_bytes.fetch_add(bytes, std::memory_order_relaxed);
	stats.view_changes.fetch_add(view_changes, std::memory_order_relaxed);
	stats.resource_changes.fetch_add(resource_changes, std::memory_order_relaxed);
	stats.resettled.fetch_add(resettled, std::memory_order_relaxed);
	stats.released.fetch_add(released, std::memory_order_relaxed);
	stats.kept_stale.fetch_add(kept, std::memory_order_relaxed);
}

uint32_t RenderExecutor::RevalidateBindlessKeys(uint32_t budget) {
	KYTY_PROFILER_FUNCTION();
	if (BindlessEngineEnabled() && !BindlessEngineVerify()) {
		return 0; // the write watch reports rewritten entries (SyncBindlessHeaps)
	}
	// The guest rewrites heap entries as it streams textures out and others in. A key settled
	// from the old T# kept sampling the old texture, and kept it pinned in the texture cache for
	// the rest of the run (in the jungle, 14-21 keys and 44-65 MB at any time). Each frame a
	// window of every heap is read back in one go and compared; a changed key is settled again
	// into a new slot, and an image no key refers to any more is unpinned, so the cache can
	// collect it. A heap of 64 Ki keys is covered in 32 frames.
	constexpr uint32_t Window = 2048;
	auto&              table         = m_context.GetBindlessTable();
	uint32_t           resolved      = 0;
	for (auto& heap: table.Heaps()) {
		const auto keys = static_cast<uint32_t>(heap.settled.size());
		if (keys == 0) {
			continue;
		}
		const uint32_t begin = heap.revalidate_cursor < keys ? heap.revalidate_cursor : 0u;
		const uint32_t count = std::min(Window, keys - begin);
		m_bindless_window.resize(count);
		const auto address = heap.base + heap.table_offset + static_cast<uint64_t>(begin) * 32u;
		const bool readable = Libs::LibKernel::Memory::TryReadBacking(
		    address, m_bindless_window.data(), static_cast<uint64_t>(count) * 32u);
		uint32_t next = begin + count;
		for (uint32_t i = 0; readable && i < count; i++) {
			const auto key = begin + i;
			if (heap.settled[key] == 0 || heap.descriptors[key] == m_bindless_window[i]) {
				continue;
			}
			if (resolved >= budget) {
				next = key; // look at it again next frame
				break;
			}
			if (BindlessEngineEnabled()) {
				static std::atomic<uint32_t> missed = 0;
				if (missed.fetch_add(1) < 64) {
					LOGF("Bindless engine: write not reported, heap=0x%" PRIx64 "+0x%x binding=%u "
					     "key=%u\n",
					     heap.base, heap.table_offset, heap.binding, key);
				}
			}
			resolved += ResettleBindlessKey(heap, key) ? 1u : 0u;
		}
		heap.revalidate_cursor = next >= keys ? 0u : next;
	}
	return resolved;
}

// Bytes of a T#'s mip chain from first_level down, ignoring tiling padding: an estimate.
static uint64_t MipChainBytes(const ShaderTextureResource& descriptor, uint32_t first_level) {
	const auto     format = descriptor.Format();
	const auto     block  = Prospero::BlockCompressedBytesPerBlock(format);
	const auto     texel  = block != 0 ? 0u : Prospero::NumBytesPerElement(format);
	const uint64_t width  = descriptor.Width5() + 1u;
	const uint64_t height = descriptor.Height5() + 1u;
	uint64_t       layers = descriptor.Type() == Prospero::ImageType::kColor2D
	                            ? 1u
	                            : static_cast<uint64_t>(descriptor.Depth()) + 1u;
	if (descriptor.Type() == Prospero::ImageType::kCube) layers = 6u;
	uint64_t bytes = 0;
	for (uint32_t level = first_level; level <= descriptor.MaxMip(); level++) {
		const auto w = std::max<uint64_t>(width >> level, 1u);
		const auto h = std::max<uint64_t>(height >> level, 1u);
		bytes += layers * (block != 0 ? ((w + 3u) / 4u) * ((h + 3u) / 4u) * block : w * h * texel);
	}
	return bytes;
}

// Diagnostics for the engine method (engine-method.md, E0): KYTY_BINDLESS_HEAP_STATS=1 logs every
// 300 presented frames, per heap:
// - valid: entries holding a texture this heap's view type can sample;
// - textures: distinct texture addresses among them (format aliases of one texture count once);
// - all_mb: their full mip chains; resident_mb: only the mips at or above each T#'s min LOD (what
//   the game has streamed in); sampled_mb: resident_mb of the textures draws asked for so far,
//   what translation on demand holds;
// - rewrites since the last line: view fields only (min LOD: streaming) or resource fields;
// - mip_stats: entries enabling the PS5's mip statistics;
// then the write watch's totals since the last line.
static void LogBindlessHeapStats(BindlessTable& table, uint64_t frame) {
	static const bool enabled = std::getenv("KYTY_BINDLESS_HEAP_STATS") != nullptr;
	static uint64_t   next    = 0;
	if (!enabled || frame < next) {
		return;
	}
	next = frame + 300;
	// T# bits that select a view of the same texture (RDNA 2 ISA 8.2.6): min LOD; dst_sel, base
	// and last level, BC swizzle; base array; min LOD warning, perf mod, mip-stats enable;
	// mip-stats slot and cache policy. Everything else describes the texture itself.
	constexpr std::array<uint32_t, 8> ViewMask {0u,          0x000fff00u, 0u,          0x0e0fffffu,
	                                            0x1fff0000u, 0x027fff00u, 0x000003ffu, 0u};
	static std::unordered_map<const BindlessTable::Heap*, std::vector<std::array<uint32_t, 8>>>
	                                     previous;
	std::vector<std::array<uint32_t, 8>> words;
	for (const auto& heap: table.Heaps()) {
		const auto entries =
		    std::min(heap.guest_entries, static_cast<uint32_t>(heap.settled.size()));
		if (entries == 0) {
			continue;
		}
		words.resize(entries);
		if (!Libs::LibKernel::Memory::TryReadBacking(heap.base + heap.table_offset, words.data(),
		                                              static_cast<uint64_t>(entries) * 32u)) {
			continue;
		}
		uint32_t valid = 0, mip_stats = 0, settled = 0, view_changes = 0, resource_changes = 0;
		// Per texture address: the largest full chain and resident part seen, and whether a
		// sampled key refers to it.
		struct Texture {
			uint64_t all      = 0;
			uint64_t resident = 0;
			bool     sampled  = false;
		};
		std::unordered_map<uint64_t, Texture> textures;
		auto& last = previous[&heap];
		for (uint32_t key = 0; key < entries; key++) {
			settled += heap.settled[key] != 0 ? 1u : 0u;
			if (key < last.size() && last[key] != words[key]) {
				bool resource = false;
				for (uint32_t i = 0; i < 8; i++) {
					resource |= ((last[key][i] ^ words[key][i]) & ~ViewMask[i]) != 0;
				}
				(resource ? resource_changes : view_changes)++;
			}
			ShaderTextureResource descriptor {};
			std::copy_n(words[key].data(), 8, descriptor.fields);
			if (descriptor.IsNull() || !BindlessCompatible(descriptor, heap.binding)) {
				continue;
			}
			valid++;
			mip_stats += descriptor.MipStatsCntEn() ? 1u : 0u;
			const auto first =
			    std::max<uint32_t>(descriptor.BaseLevel(), descriptor.MinLod() >> 8u);
			auto& texture    = textures[descriptor.Base40()];
			texture.all      = std::max(texture.all, MipChainBytes(descriptor, 0));
			texture.resident = std::max(texture.resident, MipChainBytes(descriptor, first));
			texture.sampled |= heap.settled[key] != 0;
		}
		uint64_t all = 0, resident = 0, sampled = 0, sampled_textures = 0;
		for (const auto& [address, texture]: textures) {
			all += texture.all;
			resident += texture.resident;
			if (texture.sampled) {
				sampled += texture.resident;
				sampled_textures++;
			}
		}
		last = words;
		LOGF("Bindless heap stats: frame=%" PRIu64 " base=0x%" PRIx64 "+0x%x binding=%u entries=%u "
		     "valid=%u textures=%zu sampled_textures=%" PRIu64 " all_mb=%" PRIu64
		     " resident_mb=%" PRIu64 " sampled_mb=%" PRIu64 " settled=%u rewrites_view=%u "
		     "rewrites_resource=%u mip_stats=%u\n",
		     frame, heap.base, heap.table_offset, heap.binding, entries, valid, textures.size(),
		     sampled_textures, all >> 20u, resident >> 20u, sampled >> 20u, settled, view_changes,
		     resource_changes, mip_stats);
	}
	auto& stats = g_bindless_engine_stats;
	LOGF("Bindless engine stats: frame=%" PRIu64 " syncs=%" PRIu64 " written=%" PRIu64
	     " KiB view=%" PRIu64 " resource=%" PRIu64 " resettled=%" PRIu64 " eager_settled=%" PRIu64
	     " eager_failed=%" PRIu64 " eager_paused=%" PRIu64 " requested=%" PRIu64
	     " released=%" PRIu64 " eager_capped=%" PRIu64 " unregistered_keys=%" PRIu64
	     " kept_stale=%" PRIu64 " stale_settled=%" PRIu64 "\n",
	     frame, stats.syncs.exchange(0), stats.written_bytes.exchange(0) >> 10u,
	     stats.view_changes.exchange(0), stats.resource_changes.exchange(0),
	     stats.resettled.exchange(0), stats.eager_settled.exchange(0),
	     stats.eager_failed.exchange(0), stats.eager_paused.exchange(0),
	     stats.requested.exchange(0), stats.released.exchange(0), stats.eager_capped.exchange(0),
	     table.TakeUnregisteredKeys(), stats.kept_stale.exchange(0),
	     stats.stale_settled.exchange(0));
}

void RenderExecutor::ResolveBindlessRequests() {
	KYTY_PROFILER_FUNCTION();
	const auto frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
	if (frame == m_bindless_frame) {
		return;
	}
	auto& table     = m_context.GetBindlessTable();
	auto& scheduler = m_context.GetCommandScheduler();
	// The flags come from a snapshot recorded in an earlier frame. Until the GPU has executed
	// it, check again at the next bindless draw instead of waiting.
	if (table.SnapshotRecorded() && !table.SnapshotReady(scheduler)) {
		return;
	}
	m_bindless_frame = frame;
	if (!table.SnapshotRecorded()) {
		table.RecordFeedbackSnapshot(scheduler);
		m_bindless_snapshot_frame = frame;
		return;
	}
	table.ConsumeSnapshot();
	const auto snapshot_frame = m_bindless_snapshot_frame;
	// Each texture may upload and detile; spread first sight of a scene over a few frames.
	constexpr uint32_t Budget   = 128;
	uint32_t           resolved = RevalidateBindlessKeys(Budget);
	uint32_t           requested = 0;
	static std::atomic<uint32_t> frames_logged = 0;
	const bool log_frame = frames_logged.fetch_add(1) < 12;
	if (const auto word0 = m_context.GetBindlessTable().TakeWordZero(); word0 != 0) {
		static std::atomic<uint32_t> keys_logged = 0;
		if (keys_logged.fetch_add(1) < 64) {
			LOGF("Bindless out-of-range key: frame=%" PRIu64 " key=%u (0x%08x)\n", frame,
			     word0 & 0x7fffffffu, word0);
		}
	}
	for (auto& heap: m_context.GetBindlessTable().Heaps()) {
		m_bindless_requests.clear();
		m_context.GetBindlessTable().TakeRequests(heap, m_bindless_requests);
		requested += static_cast<uint32_t>(m_bindless_requests.size());
		if (log_frame) {
			LOGF("Bindless feedback: frame=%" PRIu64 " binding=%u region=%u flagged=%zu\n",
			     frame, heap.binding, heap.region, m_bindless_requests.size());
		}
		for (const auto key: m_bindless_requests) {
			if (heap.settled[key] != 0) {
				continue;
			}
			if (resolved >= Budget) {
				break; // still pending: the next frame flags it again
			}
			resolved += ResolveBindlessKey(heap, key) ? 1u : 0u;
		}
	}
	Profiler::Add(Profiler::Counter::BindlessResolved, resolved);
	g_bindless_engine_stats.requested.fetch_add(requested, std::memory_order_relaxed);
	static std::atomic<uint32_t> logged = 0;
	if (requested != 0 && logged.fetch_add(1) < 64) {
		LOGF("Bindless requests: frame=%" PRIu64 " requested=%u resolved=%u\n", frame,
		     requested, resolved);
	}
	UpdateBindlessUsageProbe(frame, snapshot_frame);
	LogBindlessHeapStats(table, frame);
	table.RecordFeedbackSnapshot(scheduler);
	m_bindless_snapshot_frame = frame;
}

// Bindless textures stay pinned in the texture cache while a heap key refers to them, and the
// guest's heaps refer to every texture it has streamed in: on 4-8 GB cards they grew to 2-5 GB
// and never shrank, until allocations failed (ISSUES #15). Under memory pressure a probe finds
// the ones no draw samples any more: it marks every resident key's feedback word, draws that
// sample a key overwrite its mark, and a snapshot taken ProbeFrames later lists the images whose
// keys all kept it. The texture cache may then free those like unpinned images; their keys go
// back to pending, and a draw that samples one again asks for it anew. KYTY_BINDLESS_EVICT=0
// disables the probe.
void RenderExecutor::UpdateBindlessUsageProbe(uint64_t frame, uint64_t snapshot_frame) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BINDLESS_EVICT");
		return value == nullptr || value[0] != '0';
	}();
	if (!enabled) {
		return;
	}
	constexpr uint64_t ProbeFrames   = 16; // frames a texture must go unsampled
	constexpr uint64_t ProbeInterval = 32; // frames between the end of one probe and the next
	auto&              table         = m_context.GetBindlessTable();
	auto&              texture_cache = m_context.GetTextureCache();
	if (m_bindless_probe_frame != 0) {
		// The snapshot read here must have been recorded ProbeFrames after the marks.
		if (snapshot_frame < m_bindless_probe_frame + ProbeFrames) {
			return;
		}
		m_bindless_unused.clear();
		table.CollectUnusedImages(m_bindless_unused);
		texture_cache.SetBindlessEvictable(m_bindless_unused);
		m_bindless_probe_frame = 0;
		m_bindless_next_probe  = frame + ProbeInterval;
		static std::atomic<uint32_t> logged = 0;
		if (logged.fetch_add(1) < 32) {
			uint64_t bytes = 0;
			for (const auto id: m_bindless_unused) {
				if (const auto* image = texture_cache.m_slot_images.try_get(id); image != nullptr) {
					bytes += image->AccountedSize();
				}
			}
			LOGF("Bindless probe: frame=%" PRIu64 " unused=%zu (%" PRIu64 " MB) may be evicted\n",
			     frame, m_bindless_unused.size(), bytes >> 20u);
		}
		return;
	}
	// The heap's textures follow the guest (settled when it writes an entry, freed when it drops
	// them), so the probe evicts them only when the device itself is short of memory (4-8 GB
	// cards), not at the texture cache's bound: under a 6 GiB bound it evicted 27,915 of them in
	// 14 minutes, each settled again when drawn.
	if (frame >= m_bindless_next_probe && m_context.GetGraphics().DeviceMemoryAtLeast(90)) {
		table.ArmUsageProbe();
		m_bindless_probe_frame = frame;
	}
}

// Bindless samplers: the S# records of each sampler heap a draw indexes are mirrored into the
// sampler array of set 1, once per frame per heap, and the shader gets the region base and record
// count. Keys outside it, and heaps that cannot be read, use the default sampler in slot 0.
void RenderExecutor::PrepareBindlessSamplers(const ShaderStageRuntime& runtime,
                                             PreparedBindings&         prepared) {
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	if (snapshot.bindless_sampler_heaps.empty()) {
		return;
	}
	auto& table = m_context.GetBindlessTable();
	auto& cache = m_context.GetSamplerCache();
	{
		ShaderSamplerResource default_sampler;
		std::ranges::copy(ShaderRecompiler::IR::BindlessDefaultSampler, default_sampler.fields);
		table.WriteDefaultSampler(cache.GetSampler(default_sampler, false));
	}
	const auto frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
	constexpr uint64_t MaxRecords = 1024;
	for (const auto& use: snapshot.bindless_sampler_heaps) {
		uint32_t region  = 0;
		uint32_t entries = 0;
		if (table.SamplersEnabled() && use.sampler < program.info.samplers.size()) {
			const auto&    sampler = program.info.samplers[use.sampler];
			const uint32_t flags =
			    (sampler.depth_compare ? BindlessTable::SamplerDepthCompare : 0u) |
			    (sampler.force_point_filtering ? BindlessTable::SamplerPointFiltering : 0u) |
			    (sampler.integer_border ? BindlessTable::SamplerIntegerBorder : 0u);
			auto* heap = table.FindOrCreateSamplerHeap(use.base, use.table_offset, flags);
			if (heap->checked_frame != frame) {
				heap->checked_frame = frame;
				const uint64_t available =
				    use.size > use.table_offset ? (use.size - use.table_offset) / 16u : 0u;
				std::vector<std::array<uint32_t, 4>> records(std::min(available, MaxRecords));
				const uint64_t address = use.base + use.table_offset;
				const uint64_t bytes   = records.size() * sizeof(records[0]);
				if (!records.empty() &&
				    !m_context.GetBufferCache().HasGpuDirtyBytes(address, bytes) &&
				    Libs::LibKernel::Memory::TryReadBacking(address, records.data(), bytes)) {
					(void)table.MirrorSamplerHeap(*heap, records, cache);
				}
			}
			region  = heap->region;
			entries = region != 0 ? static_cast<uint32_t>(heap->records.size()) : 0u;
		}
		prepared.bindless_patches.push_back({use.mapping_offset, region, entries});
	}
}

void RenderExecutor::PrepareBindlessHeaps(const ShaderStageRuntime& runtime,
                                          PreparedBindings& prepared) {
	prepared.bindless_patches.clear();
	prepared.bindless_heaps.clear();
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	auto&       table    = m_context.GetBindlessTable();
	if ((snapshot.bindless_heaps.empty() && snapshot.bindless_sampler_heaps.empty()) ||
	    !table.Enabled()) {
		return;
	}
	PrepareBindlessSamplers(runtime, prepared);
	if (snapshot.bindless_heaps.empty()) {
		return;
	}
	ResolveBindlessRequests();
	for (const auto& use: snapshot.bindless_heaps) {
		if (use.image >= program.info.images.size()) {
			continue;
		}
		const auto& resource = program.info.images[use.image];
		const uint64_t span  = use.size > use.table_offset ? use.size - use.table_offset : 0;
		const auto entries   = static_cast<uint32_t>(std::min<uint64_t>(span / 32u, 1u << 18u));
		auto* heap = table.FindOrCreateHeap(use.base, use.table_offset, BindlessBindingFor(resource),
		                                    entries, resource);
		if (heap == nullptr) {
			continue; // region 0, count 0: every key samples the placeholder
		}
		if (BindlessEngineEnabled() && table.NeedsWatch(*heap)) {
			table.WatchHeap(*heap);
		}
		prepared.bindless_patches.push_back(
		    {use.mapping_offset, heap->region, std::min(entries, heap->entries)});
		prepared.bindless_heaps.push_back(heap);
	}
}

void RenderExecutor::PrepareBindings(const ShaderStageRuntime& runtime,
                                     PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(!runtime);
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	const DiagShaderScope diag_shader_scope(program.shader_hash);
	// Diagnostics: KYTY_WATCH_SHADER=<hash> logs the buffer and texture descriptors a shader is
	// bound with (first few uses; KYTY_WATCH_SHADER_EVERY=<n> adds every n-th use, up to 8 more).
	const uint64_t        watch_shader = WatchedShaderHash();
	static const uint32_t watch_every = [] {
		const char* value = std::getenv("KYTY_WATCH_SHADER_EVERY");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	static std::atomic<uint32_t> watched_uses {0};
	const auto                   watched_use = watch_shader != 0 && program.shader_hash == watch_shader
	                                               ? watched_uses.fetch_add(1)
	                                               : UINT32_MAX;
	static std::atomic<uint32_t> frame_uses {0};
	const bool in_watched_frame =
	    watched_use != UINT32_MAX &&
	    InWatchedShaderFrame(program.shader_hash,
	                         m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed),
	                         frame_uses);
	if (watched_use < 4u ||
	    (watched_use != UINT32_MAX && watch_every != 0 && watched_use % watch_every == 0 &&
	     watched_use / watch_every <= 8u) ||
	    in_watched_frame) {
		LOGF("WatchShader 0x%016" PRIx64 " use=%u frame=%" PRIu64 " stage=%u buffers=%zu images=%zu\n",
		     program.shader_hash, watched_use,
		     m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed),
		     static_cast<uint32_t>(program.stage), program.info.buffers.size(),
		     program.info.images.size());
		for (size_t i = 0; i < program.info.buffers.size() && i < snapshot.buffers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
			// Whether the GPU has written bytes of the buffer that the CPU copy lacks.
			const uint64_t bytes =
			    std::max<uint64_t>(r.Stride(), 1u) * static_cast<uint64_t>(r.NumRecords());
			const bool gpu_written = r.Base48() != 0 && bytes != 0 &&
			                         m_context.GetBufferCache().HasGpuDirtyBytes(r.Base48(), bytes);
			LOGF("  buffer[%zu]: pc=0x%x formatted=%d written=%d base=0x%012" PRIx64
			     " stride=%u records=%u format=%u dst_sel=%u%u%u%u gpu_written=%d\n",
			     i, program.info.buffers[i].first_use_pc, program.info.buffers[i].formatted ? 1 : 0,
			     program.info.buffers[i].written ? 1 : 0, r.Base48(), r.Stride(), r.NumRecords(),
			     r.RawFormat(), r.DstSelX(), r.DstSelY(), r.DstSelZ(), r.DstSelW(),
			     gpu_written ? 1 : 0);
			// Small buffers (constants, tables) are listed in full, as guest memory holds them.
			if (r.Base48() != 0 && bytes != 0 && bytes <= 4096 && !gpu_written) {
				std::vector<uint32_t> words((bytes + 3) / 4);
				if (Libs::LibKernel::Memory::TryReadBacking(r.Base48(), words.data(), bytes)) {
					for (size_t row = 0; row < words.size(); row += 8) {
						std::string line;
						for (size_t k = row; k < std::min(words.size(), row + 8); k++) {
							line += fmt::format(" {:08x}", words[k]);
						}
						LOGF("    +0x%03zx:%s\n", row * 4, line.c_str());
					}
				}
			}
		}
		for (size_t i = 0; i < program.info.images.size() && i < snapshot.images.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderTextureResource>(snapshot.images[i]);
			LOGF("  image[%zu]: bindless=%d format=%u dst_sel=%u%u%u%u type=%u extent=%ux%u\n", i,
			     program.info.images[i].bindless ? 1 : 0, static_cast<uint32_t>(r.Format()),
			     r.DstSelX(), r.DstSelY(), r.DstSelZ(), r.DstSelW(),
			     static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Width5()) + 1u,
			     static_cast<uint32_t>(r.Height5()) + 1u);
		}
		// User data, and the first bytes of every buffer a user-data V# points at: the shader
		// reads constants and heap indices from these with scalar loads.
		const auto& user = snapshot.user_data;
		for (size_t row = 0; row < user.size(); row += 8) {
			std::string line;
			for (size_t k = row; k < std::min(user.size(), row + 8); k++) {
				line += fmt::format(" {:08x}", user[k]);
			}
			LOGF("  user_data[s%zu]:%s\n", program.user_data_base + row, line.c_str());
		}
		for (size_t reg = 0; reg + 4 <= user.size(); reg += 4) {
			ShaderBufferResource r;
			r.fields[0]         = user[reg];
			r.fields[1]         = user[reg + 1];
			r.fields[2]         = user[reg + 2];
			r.fields[3]         = user[reg + 3];
			const uint64_t size = r.Stride() != 0 ? static_cast<uint64_t>(r.Stride()) * r.NumRecords()
			                                      : r.NumRecords();
			if (r.Base48() == 0 || size == 0) {
				continue;
			}
			std::vector<uint32_t> words(std::min<uint64_t>(size, 0x100u) / 4u);
			if (words.empty() || m_context.GetBufferCache().HasGpuDirtyBytes(r.Base48(), words.size() * 4u) ||
			    !Libs::LibKernel::Memory::TryReadBacking(r.Base48(), words.data(), words.size() * 4u)) {
				continue;
			}
			LOGF("  user V# s[%zu:%zu]: base=0x%012" PRIx64 " stride=%u records=%u\n",
			     program.user_data_base + reg, program.user_data_base + reg + 3, r.Base48(), r.Stride(),
			     r.NumRecords());
			LogWords(words.data(), words.size());
			std::lock_guard lock(g_watched_reread_mutex);
			g_watched_rereads.push_back(
			    {m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed) + 2u,
			     r.Base48(), words.size() * 4u});
		}
	}
	prepared.runtime = &runtime;
	prepared.gds = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt = {};
	prepared.shader_data_buffer = {};
	prepared.shared_memory = {};
	prepared.images.resize(program.info.images.size());
	prepared.samplers.clear();
	prepared.shader_data.clear();
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		auto binding = ResolveTexture(program.info.images[i], snapshot.images[i]);
		BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
		binding.mip_views.swap(prepared.images[i].mip_views);
		binding.mip_views.clear();
		prepared.images[i] = std::move(binding);
	}
	prepared.samplers.reserve(program.info.samplers.size());
	for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
		prepared.samplers.push_back(NativeSampler(
		    m_context, program, i, snapshot.samplers[program.info.samplers[i].snapshot_index]));
	}
	for (const auto& use: snapshot.bindless_heaps) {
		if (m_bindless_surveyed.insert(use.base ^ (static_cast<uint64_t>(use.table_offset) << 48u))
		        .second) {
			SurveyBindlessHeap(use);
		}
	}
	PrepareBindlessHeaps(runtime, prepared);
	prepared.shader_data.reserve(program.bindings.ShaderDataDwords());
	for (const auto reg: program.bindings.user_data_registers) {
		prepared.shader_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
	prepared.shader_data.resize(program.bindings.ShaderDataDwords());
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::Gds) != nullptr) {
		prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
	}
}

void RenderExecutor::FindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       cache    = m_context.GetBufferCache();

	prepared.buffer_sources.clear();
	const auto& layout = program.bindings;
	if (layout.memory_offset_count == 0) {
		return;
	}
	const auto& resources = layout.descriptors.front().resources;
	prepared.buffer_sources.reserve(resources.size());
	for (const auto resource: resources) {
		auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[resource]);
		const auto address = descriptor.Base48();
		const auto requested_size = descriptor.GetSize();
		if (address != 0 && address < BufferCache::CACHING_PAGESIZE) {
			// A base in guest page 0 cannot be read on the hardware either; the shader must not
			// touch it, so it binds as null. Logged so the descriptor's origin can be traced.
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1) < 8) {
				LOGF("RenderExecutor: buffer descriptor %u in guest page 0: stage=%s words=[0x%08x "
				     "0x%08x 0x%08x 0x%08x] size=0x%016" PRIx64 "\n",
				     resource, ShaderStageResourceName(program.stage), descriptor.fields[0],
				     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3],
				     requested_size);
			}
		}
		if (address < BufferCache::CACHING_PAGESIZE || requested_size == 0) {
			prepared.buffer_sources.push_back({});
			continue;
		}
		uint64_t size = 0;
		{
			KYTY_PROFILER_BLOCK("FindBuffers::ClampRangeSize");
			size = Libs::LibKernel::Memory::TryClampRangeSize(address, requested_size);
		}
		if (size == 0) {
			// Nothing is mapped at the base yet. Streaming titles bind regions they back later (a
			// dispatch can precede the mapping by thousands of calls), and an unbacked region reads
			// zero and drops writes on the hardware; binding null gives the same.
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1) < 8) {
				LOGF("RenderExecutor: buffer descriptor %u outside mapped memory: stage=%s "
				     "base=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
				     resource, ShaderStageResourceName(program.stage), address, requested_size);
			}
			prepared.buffer_sources.push_back({});
			continue;
		}
		prepared.buffer_sources.push_back({address, size, cache.FindBuffer(address, size)});
	}
}

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program   = *prepared.runtime->program;
	const auto& snapshot  = *prepared.runtime->resources;
	const auto& layout    = program.bindings;
	EXIT_IF(prepared.buffer_sources.size() != layout.memory_offset_count);
	const DiagShaderScope diag_shader_scope(program.shader_hash);

	prepared.buffers.clear();
	prepared.buffers.reserve(layout.memory_offset_count);
	EXIT_IF(prepared.shader_data.size() != layout.ShaderDataDwords());
	std::fill(prepared.shader_data.begin() + layout.memory_offset_dword,
	          prepared.shader_data.end(), 0);
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.shader_data[dword] |= offset << shift;
	};
	for (uint32_t i = 0; i < layout.memory_offset_count; i++) {
		const auto resource = layout.descriptors.front().resources[i];
		uint32_t buffer_offset = 0;
		prepared.buffers.push_back(NativeStorageBuffer(
		    m_context, prepared.buffer_sources[i], program.info.buffers[resource],
		    resource < snapshot.buffer_write_extents.size() ? &snapshot.buffer_write_extents[resource]
		                                                    : nullptr,
		    buffer_offset));
		pack_memory_offset(i, buffer_offset);
		// The fallback allocation makes the Vulkan descriptor valid, but an empty guest
		// buffer still has no accessible elements. In particular, never let a null store
		// modify the shared fallback and affect subsequent null reads.
		prepared.shader_data[layout.BufferLengthDword() + i] =
		    prepared.buffer_sources[i].size == 0
		        ? 0u
		        : static_cast<uint32_t>(prepared.buffers.back().range / sizeof(uint32_t));
	}
	if (ShaderRecompiler::IR::FindBinding(
	        layout, ShaderRecompiler::IR::DescriptorBindingKind::FlattenedSrt) != nullptr) {
		if (prepared.bindless_patches.empty()) {
			prepared.flattened_srt = NativeUpload(m_context, snapshot.flattened_srt);
			FillFlatSlotsOnGpu(m_context, prepared.flattened_srt, snapshot.gpu_fills);
		} else {
			m_bindless_srt.assign(snapshot.flattened_srt.begin(), snapshot.flattened_srt.end());
			for (const auto& [offset, region, entries]: prepared.bindless_patches) {
				if (static_cast<size_t>(offset) + 1u < m_bindless_srt.size()) {
					m_bindless_srt[offset]      = region;
					m_bindless_srt[offset + 1u] = entries;
				}
				// The first 256 distinct patches are logged; after that no draw takes the lock.
				static std::atomic<bool>            logging_done = false;
				static std::mutex                   logged_mutex;
				static std::unordered_set<uint64_t> logged;
				if (logging_done.load(std::memory_order_relaxed)) {
					continue;
				}
				std::scoped_lock lock(logged_mutex);
				if (logged.size() >= 256) {
					logging_done.store(true, std::memory_order_relaxed);
				} else if (logged.insert(program.shader_hash ^ offset).second) {
					LOGF("Bindless patch: stage=%u hash=0x%016" PRIx64
					     " offset=%u region=%u entries=%u srt_words=%zu\n",
					     static_cast<uint32_t>(program.stage), program.shader_hash, offset, region,
					     entries, m_bindless_srt.size());
				}
			}
			prepared.flattened_srt = NativeUpload(m_context, m_bindless_srt);
			FillFlatSlotsOnGpu(m_context, prepared.flattened_srt, snapshot.gpu_fills);
		}
	}
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::ShaderData) != nullptr) {
		prepared.shader_data_buffer = NativeUpload(m_context, prepared.shader_data);
	}
}

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       images   = prepared.images;
	EXIT_IF(images.size() != program.info.images.size());
	auto& texture_cache = m_context.GetTextureCache();
	// Acquire each view before a later overlapping descriptor can replace its image.
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto old_image = texture_cache.m_slot_images.try_get(images[i].image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				if (old_image->binding.is_target) {
					NoteBindlessStateChange(*old_image);
				}
				old_image->binding = {};
			}
			images[i] = ResolveTexture(program.info.images[i], snapshot.images[i]);
			BindImage(images[i].image_id,
			          images[i].desc.type == TextureCache::BindingType::Storage);
		}
		auto& binding = images[i];
		binding.mip_views.clear();
		const auto& resource = program.info.images[i];
		if (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			EXIT_IF(resource.mip_count == 0u ||
			        resource.mip_count != binding.desc.view_info.level_count);
			binding.mip_views.reserve(resource.mip_count);
			for (uint32_t mip = 0; mip < resource.mip_count; mip++) {
				auto desc = binding.desc;
				desc.view_info.base_level += mip;
				desc.view_info.level_count = 1;
				// The shader selects the mip after applying the guest minimum LOD.
				desc.view_info.min_lod = 0;
				binding.mip_views.push_back(texture_cache.FindTexture(binding.image_id, desc));
			}
			binding.image_view = binding.mip_views.front();
		} else {
			binding.image_view = texture_cache.FindTexture(binding.image_id, binding.desc);
		}
		auto&      image   = texture_cache.GetImage(binding.image_id);
		const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
		image.usage.storage |= storage;
		image.usage.texture |= !storage;
	}
}

void RenderExecutor::PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
                                             std::span<RenderColorInfo> colors) {
	bool uses_dma = false;
	for (auto* stage: stages) {
		FindBuffers(*stage);
		uses_dma |= stage->runtime->program->info.uses_dma;
	}
	if (uses_dma) {
		m_context.PrepareBda();
	}
	for (auto* stage: stages) {
		RebindImages(*stage);
	}
	auto& cache = m_context.GetTextureCache();
	for (auto& target: colors) {
		EXIT_IF(!target.image_id);
		const auto old_image = cache.m_slot_images.try_get(target.image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				if (old_image->binding.is_target) {
					NoteBindlessStateChange(*old_image);
				}
				old_image->binding = {};
			}
			target.desc.view_info.base_level = target.guest_mip_level;
			target.desc.view_info.base_layer = target.guest_array_layer;
			target.image_id = cache.FindImage(target.desc);
			BindRenderTarget(target.image_id);
		}
	}
	// Discovery can read back PS5 metadata and submit the scheduler. Reserve draw buffers only
	// after image identities are final; attachment layout transitions follow buffer alias copies.
	for (auto* stage: stages) {
		RebindBuffers(*stage);
	}
}

void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings) {
	KYTY_PROFILER_FUNCTION();
	const auto vk_buffer    = buffer.Recorder();
	size_t descriptor_count = 0;
	size_t write_count      = 0;
	ShaderRecompiler::IR::PushData push_data;
	bool                           has_push_data = false;
	constexpr auto                 GraphicsStages =
	    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eMeshEXT |
	    vk::ShaderStageFlagBits::eTessellationControl |
	    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment;
	vk::ShaderStageFlags push_stages = pipeline_bind_point == vk::PipelineBindPoint::eGraphics
	                                       ? vk::ShaderStageFlagBits::eFragment
	                                       : vk::ShaderStageFlags {};
	for (const auto* prepared: prepared_bindings) {
		EXIT_IF(prepared == nullptr || prepared->runtime == nullptr || !*prepared->runtime);
		const auto& program = *prepared->runtime->program;
		write_count += program.bindings.descriptors.size();
		for (const auto& binding: program.bindings.descriptors) {
			descriptor_count += NativeDescriptorCount(binding);
		}
		const auto shader_stage = NativeShaderStage(program.stage);
		push_stages |= shader_stage;
		EXIT_IF((pipeline_bind_point == vk::PipelineBindPoint::eGraphics &&
		         (shader_stage & GraphicsStages) == vk::ShaderStageFlags {}) ||
		        (pipeline_bind_point == vk::PipelineBindPoint::eCompute &&
		         shader_stage != vk::ShaderStageFlagBits::eCompute));
	}
	// Scalar reads the host specialized on must not overlap what this draw or dispatch writes.
	// The written ranges are gathered once, and a reader's reads are compared one by one only
	// against a range that meets their span: a per-read scan of every bound image and written
	// buffer cost the jungle 1.6 fps (11.1 against 12.7, warm).
	struct WrittenRange {
		uint64_t                                address = 0;
		uint64_t                                size    = 0;
		const ShaderRecompiler::IR::CompiledShaderInfo* writer = nullptr; // null: an image or target
		uint32_t                                resource = 0;
		uint64_t                                descriptor_address = 0;
		uint64_t                                descriptor_size    = 0;
		bool                                    bounded  = false;
	};
	std::vector<WrittenRange> written_ranges;
	bool                      written_gathered = false;
	for (const auto* reader: prepared_bindings) {
		const auto& reads = reader->runtime->resources->specialization_reads;
		if (reads.empty()) continue;
		if (!written_gathered) {
			written_gathered = true;
			for (const auto* writer: prepared_bindings) {
				if (writer->runtime->program->has_address_writes) {
					EXIT("scalar resource reads cannot be proven disjoint from shader address writes\n");
				}
			}
			for (const auto id: m_bound_images) {
				const auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
				if (image == nullptr ||
				    (!image->binding.shader_write && !image->binding.is_target)) continue;
				for (const auto written: {image->info.data, image->info.stencil,
				                          image->info.metadata.range}) {
					if (written.size != 0) {
						written_ranges.push_back({.address = written.address, .size = written.size});
					}
				}
			}
			for (const auto* writer: prepared_bindings) {
				const auto& program = *writer->runtime->program;
				const auto& extents = writer->runtime->resources->buffer_write_extents;
				for (uint32_t i = 0; i < writer->buffer_sources.size(); ++i) {
					const auto resource = program.bindings.descriptors.front().resources[i];
					if (!program.info.buffers[resource].written) continue;
					const auto& written = writer->buffer_sources[i];
					// The bytes the dispatch can store to, when its stores are bounded
					// (BufferWriteExtent): a title's heap-wide V# otherwise covers the whole
					// heap and every descriptor read in it.
					auto       written_address = written.address;
					auto       written_size    = written.size;
					const bool bounded = resource < extents.size() && extents[resource].valid;
					if (bounded) {
						written_address += extents[resource].begin;
						written_size = extents[resource].end > extents[resource].begin
						                   ? extents[resource].end - extents[resource].begin
						                   : 0;
					}
					if (written_size != 0) {
						written_ranges.push_back({.address            = written_address,
						                          .size               = written_size,
						                          .writer             = &program,
						                          .resource           = resource,
						                          .descriptor_address = written.address,
						                          .descriptor_size    = written.size,
						                          .bounded            = bounded});
					}
				}
			}
		}
		if (written_ranges.empty()) break;
		uint64_t span_begin = UINT64_MAX;
		uint64_t span_end   = 0;
		for (const auto [address, size]: reads) {
			span_begin = std::min(span_begin, address);
			span_end   = std::max(span_end, address + size);
		}
		for (const auto& written: written_ranges) {
			if (span_end <= span_begin ||
			    !ImageRangeOverlaps(span_begin, span_end - span_begin, written.address, written.size)) {
				continue;
			}
			for (const auto [address, size]: reads) {
				if (!ImageRangeOverlaps(address, size, written.address, written.size)) continue;
				if (written.writer == nullptr) {
					EXIT("scalar resource reads overlap an image or attachment write\n");
				}
				// Upstream (6409be28) exits here. Wolverine's CS 0x85a58319a7a75e48 reads a
				// branch input from a 256-byte buffer the same dispatch updates; the host
				// evaluates it from the state before the dispatch, as the walk before this
				// check did (and the game ran correctly). Reported once per shader pair.
				static std::mutex                   reported_mutex;
				static std::unordered_set<uint64_t> reported;
				const auto reader_hash = reader->runtime->program->shader_hash;
				std::scoped_lock lock(reported_mutex);
				if (reported.insert(reader_hash ^ (written.writer->shader_hash << 1u)).second) {
					LOGF("Warning: scalar resource reads overlap a shader buffer write: "
					     "reader 0x%016" PRIx64 " reads 0x%" PRIx64 "+0x%" PRIx64
					     ", writer 0x%016" PRIx64 " buffer %u writes 0x%" PRIx64 "+0x%" PRIx64
					     " (descriptor 0x%" PRIx64 "+0x%" PRIx64 ", extent %s)\n",
					     reader_hash, address, size, written.writer->shader_hash, written.resource,
					     written.address, written.size, written.descriptor_address,
					     written.descriptor_size, written.bounded ? "bounded" : "none");
				}
				break;
			}
		}
	}
	m_descriptor_buffers.clear();
	m_descriptor_images.clear();
	m_descriptor_writes.clear();
	m_descriptor_buffers.reserve(descriptor_count);
	m_descriptor_images.reserve(descriptor_count);
	m_descriptor_writes.reserve(write_count);

	for (auto* prepared: prepared_bindings) {
		const auto& program       = *prepared->runtime->program;
		auto&       descriptors   = *prepared;
		const auto  shader_stage  = NativeShaderStage(program.stage);
		const auto  shader_stages = ShaderPipelineStages(shader_stage);
		if (descriptors.gds.buffer != nullptr) {
			buffer.EndRendering();
			const auto barrier = MakeGdsDependency(descriptors.gds.buffer);
			vk_buffer.pipelineBarrier(
			    vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
			        vk::PipelineStageFlagBits::eAllGraphics |
			        vk::PipelineStageFlagBits::eComputeShader,
			    shader_stages, vk::DependencyFlags {}, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		// Bindless textures are sampled through set 1 with a read-only layout. A heap holds
		// thousands of resolved images; one whose images' states did not change since it was
		// last checked (g_bindless_state_generation) needs a look only at the images resolved
		// since. KYTY_BINDLESS_VERIFY=1 checks every image and stops if a skipped one needed it.
		static const bool verify = std::getenv("KYTY_BINDLESS_VERIFY") != nullptr;
		if (BindlessEngineEnabled() && !verify && !descriptors.bindless_heaps.empty()) {
			// Engine method, E5: only images settled since the last bindless draw, and pinned
			// images whose state changed (NoteBindlessStateChange), are made readable again; a
			// heap of thousands of images is not walked whenever any one of them changed.
			auto&      cache         = m_context.GetTextureCache();
			const auto make_readable = [&](ImageId id) {
				auto* image = cache.m_slot_images.try_get(id);
				if (image == nullptr || !image->registered || !image->bindless_pinned ||
				    image->binding.is_target || image->info.data.Empty()) {
					return;
				}
				const auto layout = image->info.IsDepth()
				                        ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                        : vk::ImageLayout::eShaderReadOnlyOptimal;
				if (image->backing.state.layout != layout) {
					image->Transit(layout, vk::AccessFlagBits2::eShaderRead, {}, vk_buffer);
				}
			};
			for (auto* heap: descriptors.bindless_heaps) {
				for (const auto id: heap->unchecked) {
					make_readable(id);
				}
				heap->unchecked.clear();
			}
			{
				std::scoped_lock lock {g_bindless_state_changes.mutex};
				m_bindless_changed.swap(g_bindless_state_changes.ids);
			}
			for (const auto id: m_bindless_changed) {
				make_readable(id);
			}
			m_bindless_changed.clear();
		}
		using HeapSpan = std::span<BindlessTable::Heap* const>;
		for (auto* heap: BindlessEngineEnabled() && !verify ? HeapSpan {}
		                                                    : HeapSpan {descriptors.bindless_heaps}) {
			const auto& resolved = heap->resolved;
			size_t      begin =
			    heap->checked_generation ==
			            g_bindless_state_generation.load(std::memory_order_relaxed) &&
			            heap->checked_count <= resolved.size()
			         ? heap->checked_count
			         : 0;
			if (verify) {
				begin = 0;
			}
			const auto skipped =
			    heap->checked_generation ==
			            g_bindless_state_generation.load(std::memory_order_relaxed) &&
			            heap->checked_count <= resolved.size()
			        ? heap->checked_count
			        : 0;
			for (size_t i = begin; i < resolved.size(); i++) {
				auto* image = m_context.GetTextureCache().m_slot_images.try_get(resolved[i]);
				if (image == nullptr || !image->registered || image->binding.is_target ||
				    image->info.data.Empty()) {
					continue;
				}
				const auto layout = image->info.IsDepth()
				                        ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                        : vk::ImageLayout::eShaderReadOnlyOptimal;
				if (image->backing.state.layout != layout) {
					if (verify && i < skipped) {
						EXIT("bindless: image %u of heap 0x%016" PRIx64
						     " changed layout without a state generation change\n",
						     static_cast<uint32_t>(i), heap->base);
					}
					image->Transit(layout, vk::AccessFlagBits2::eShaderRead, {}, vk_buffer);
				}
			}
			// After the transitions above, which bump the generation themselves.
			heap->checked_generation = g_bindless_state_generation.load(std::memory_order_relaxed);
			heap->checked_count      = resolved.size();
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			auto& image   = m_context.GetTextureCache().GetImage(descriptors.images[i].image_id);
			auto& binding = descriptors.images[i];
			const auto&                 view = binding.desc.view_info;
			const ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
			                                   view.layer_count};
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (image.info.data.Empty()) {
				image.Transit(vk::ImageLayout::eGeneral,
				              storage ? vk::AccessFlagBits2::eShaderRead |
				                            vk::AccessFlagBits2::eShaderWrite
				                      : vk::AccessFlagBits2::eShaderRead,
				              range, vk_buffer);
			} else if (image.binding.is_target) {
				const auto layout = image.binding.attachment_layout;
				EXIT_IF(layout == vk::ImageLayout::eUndefined);
				if (image.info.IsDepth()) {
					const auto host_view =
					    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
					EXIT_IF(storage || host_view == image.views.end());
					const auto aspect = host_view->info.aspect;
					if (aspect & ~DepthReadableAspects(layout)) {
						EXIT("sampling a writable depth/stencil attachment aspect\n");
					}
				}
				image.Transit(layout,
				              image.binding.attachment_access | vk::AccessFlagBits2::eShaderRead |
				                  (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite
				                                              : vk::AccessFlags2 {}),
				              {}, vk_buffer);
			} else if (image.binding.force_general && !image.info.IsDepth()) {
				const vk::AccessFlags2 storage_access = image.binding.shader_write
				                                            ? vk::AccessFlagBits2::eShaderWrite
				                                            : vk::AccessFlags2 {};
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | storage_access, {}, vk_buffer);
			} else if (storage) {
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
				              range, vk_buffer);
			} else {
				image.Transit(image.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                                   : vk::ImageLayout::eShaderReadOnlyOptimal,
				              vk::AccessFlagBits2::eShaderRead, range, vk_buffer);
			}
			binding.layout = image.backing.state.layout;
		}

		m_image_occurrences.assign(descriptors.images.size(), 0);
		for (const auto& binding: program.bindings.descriptors) {
			vk::WriteDescriptorSet write {};
			write.dstBinding     = ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind);
			write.descriptorType = NativeDescriptorType(binding.kind);
			write.descriptorCount   = NativeDescriptorCount(binding);
			const auto buffer_start = m_descriptor_buffers.size();
			const auto image_start  = m_descriptor_images.size();
			if (ShaderRecompiler::IR::ImageBindingResourceClass(binding.kind) !=
			    ShaderRecompiler::IR::ImageResourceClass::None) {
				for (const auto resource: binding.resources) {
					m_descriptor_images.push_back(MakeImageInfo(
					    descriptors.images.at(resource), m_image_occurrences.at(resource)++));
				}
			} else {
				switch (binding.kind) {
					case BindingKind::Buffers:
						EXIT_IF(descriptors.buffers.size() != binding.resources.size());
						for (const auto& view: descriptors.buffers) {
							EXIT_IF(view.buffer == nullptr);
							m_descriptor_buffers.push_back(view);
						}
						break;
					case BindingKind::BdaPagetable:
					case BindingKind::FaultBuffer: {
						auto&       cache      = m_context.GetBufferCache();
						const auto* bda_buffer = binding.kind == BindingKind::BdaPagetable
						                             ? cache.GetBdaPageTableBuffer()
						                             : cache.GetFaultBuffer();
						m_descriptor_buffers.emplace_back(bda_buffer->Handle(), 0,
						                                  bda_buffer->Size());
						break;
					}
					case BindingKind::FlattenedSrt:
					case BindingKind::ShaderData:
					case BindingKind::SharedMemory:
					case BindingKind::Gds: {
						const vk::DescriptorBufferInfo* view = &descriptors.gds;
						if (binding.kind == BindingKind::FlattenedSrt) {
							view = &descriptors.flattened_srt;
						} else if (binding.kind == BindingKind::ShaderData) {
							view = &descriptors.shader_data_buffer;
						} else if (binding.kind == BindingKind::SharedMemory) {
							view = &descriptors.shared_memory;
						}
						EXIT_IF(view->buffer == nullptr);
						m_descriptor_buffers.push_back(*view);
						break;
					}
					case BindingKind::Samplers:
						for (const auto resource: binding.resources) {
							const auto sampler = descriptors.samplers.at(resource);
							EXIT_IF(sampler == nullptr);
							m_descriptor_images.emplace_back(sampler, nullptr,
							                                 vk::ImageLayout::eUndefined);
						}
						break;
					case BindingKind::Count: EXIT("invalid descriptor binding kind");
				}
			}
			if (m_descriptor_buffers.size() != buffer_start) {
				write.pBufferInfo = m_descriptor_buffers.data() + buffer_start;
			}
			if (m_descriptor_images.size() != image_start) {
				write.pImageInfo = m_descriptor_images.data() + image_start;
			}
			m_descriptor_writes.push_back(write);
		}
		for (uint32_t i = 0; i < descriptors.images.size(); i++) {
			const auto expected =
			    descriptors.images[i].mip_views.empty()
			        ? 1u
			        : static_cast<uint32_t>(descriptors.images[i].mip_views.size());
			EXIT_IF(m_image_occurrences[i] != expected);
		}

		const auto shader_data_dwords = program.bindings.ShaderDataDwords();
		EXIT_IF(prepared->shader_data.size() != shader_data_dwords);
		if (program.bindings.UsesPushData()) {
			std::ranges::copy(prepared->shader_data,
			                  push_data.dwords.begin() + program.bindings.push_data_start_dword);
			has_push_data = true;
		}
	}

	if (has_push_data) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, push_stages, 0, sizeof(push_data),
		                        push_data.dwords.data());
	}

	if (!m_descriptor_writes.empty()) {
		EXIT_IF(pipeline.descriptor_set_layout == nullptr);
		if (pipeline.uses_push_descriptors) {
			vk_buffer.pushDescriptorSetKHR(pipeline_bind_point, pipeline.pipeline_layout, 0,
			                               static_cast<uint32_t>(m_descriptor_writes.size()),
			                               m_descriptor_writes.data());
		} else {
			const auto set = m_context.GetDescriptorHeap().Commit(pipeline.descriptor_set_layout);
			for (auto& write: m_descriptor_writes) {
				write.dstSet = set;
			}
			// The update moves to the recording thread with the bind: the set is fresh.
			vk_buffer.updateAndBindDescriptorSet(
			    m_context.GetGraphics().device, pipeline_bind_point, pipeline.pipeline_layout, set,
			    static_cast<uint32_t>(m_descriptor_writes.size()), m_descriptor_writes.data());
		}
	}
	if (pipeline.uses_bindless) {
		const auto set = m_context.GetGraphics().bindless_set;
		EXIT_IF(set == nullptr);
		vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, 1, 1, &set, 0,
		                             nullptr);
	}
}

} // namespace Libs::Graphics
