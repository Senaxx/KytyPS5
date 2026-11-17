#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_NULLPRTWORKAROUND_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_NULLPRTWORKAROUND_H_

#include <cstdint>

namespace Libs::Graphics {

// AMD GPUs can hang on a bogus page fault when a shader reads an unbound (NULL) page of a sparse buffer through the
// scalar (SMEM) unit, instead of reading 0. vkd3d-proton flags every AMD driver but RADV 26.2+ (which fixes it):
// https://github.com/HansKristian-Work/vkd3d-proton/blob/5b1089c89dbc5cb9f662650feb60d2d3df3e55ca/libs/vkd3d/device_workarounds.c#L1291-L1298
// We match its test; there the BDA page table's unbound pages go on a zero page (KYTY_ZERO_PAGE_TABLE: 1 forces it on,
// 0 off).
inline bool BrokenNullPrt(uint32_t vendor_id, uint32_t driver_id, uint32_t driver_version) {
	constexpr uint32_t VENDOR_AMD      = 0x1002;
	constexpr uint32_t DRIVER_ID_RADV  = 3;                       // VK_DRIVER_ID_MESA_RADV
	constexpr uint32_t RADV_FIXED      = (26u << 22u) | (2u << 12u); // VK_MAKE_VERSION(26, 2, 0)
	return vendor_id == VENDOR_AMD && (driver_id != DRIVER_ID_RADV || driver_version < RADV_FIXED);
}

// The switch's value (nullptr: unset) over the driver test.
inline bool WantZeroPageTable(const char* force, uint32_t vendor_id, uint32_t driver_id, uint32_t driver_version) {
	if (force != nullptr && force[0] == '1') return true;
	if (force != nullptr && force[0] == '0') return false;
	return BrokenNullPrt(vendor_id, driver_id, driver_version);
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_NULLPRTWORKAROUND_H_
