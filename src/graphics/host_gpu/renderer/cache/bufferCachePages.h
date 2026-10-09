#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BUFFERCACHEPAGES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BUFFERCACHEPAGES_H_

#include <cstdint>

// The buffer cache's page layout as the shaders see it (BDA page table, fault records), and the
// shader trap record (faultManager.h), without
// the rest of bufferCache.h. The SPIR-V emitter includes this instead of the buffer cache: every
// header the recompiler includes is part of the shader disk cache key (shader_cache_version.cmake),
// and bufferCache.h brought in the memory tracker, the page manager and kernel/memory.h, so any
// change there threw away every shader cache. bufferCache.h checks these against its own values.
namespace Libs::Graphics::BufferCachePages {

inline constexpr uint32_t kPageBits           = 14;
inline constexpr uint64_t kPageSize           = uint64_t {1} << kPageBits;
// LOWER_ADDRESS_SIZE (regionDefinitions.h) and the extended memory window (kernel/memory.h).
inline constexpr uint64_t kLowerAddressSize   = uint64_t {1} << 40u;
inline constexpr uint64_t kExtendedMemoryBase = 0x080000000000ull;
inline constexpr uint64_t kExtendedMemorySize = 512ull * 1024 * 1024 * 1024;
inline constexpr uint64_t kNumPages           = (kLowerAddressSize + kExtendedMemorySize) >> kPageBits;

} // namespace Libs::Graphics::BufferCachePages

namespace Libs::Graphics {

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

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BUFFERCACHEPAGES_H_
