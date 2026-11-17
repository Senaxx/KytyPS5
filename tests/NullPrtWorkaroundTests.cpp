#include "graphics/host_gpu/nullPrtWorkaround.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

using Libs::Graphics::BrokenNullPrt;
using Libs::Graphics::WantZeroPageTable;

namespace {

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "NullPrtWorkaroundTests: failed: %s\n", message);
    std::abort();
  }
}

constexpr uint32_t AMD = 0x1002, NVIDIA = 0x10de, INTEL = 0x8086;
constexpr uint32_t RADV = 3, AMD_PROPRIETARY = 1, AMDVLK = 2, NVIDIA_PROPRIETARY = 4;
constexpr uint32_t Version(uint32_t major, uint32_t minor, uint32_t patch) {
  return (major << 22u) | (minor << 12u) | patch;
}

} // namespace

int main() {
  // vkd3d-proton's test: AMD, unless RADV 26.2 or newer.
  Check(BrokenNullPrt(AMD, AMD_PROPRIETARY, Version(2, 0, 300)), "AMD's Windows driver");
  Check(BrokenNullPrt(AMD, AMDVLK, Version(26, 9, 0)), "AMDVLK, any version");
  Check(BrokenNullPrt(AMD, RADV, Version(26, 1, 99)), "RADV before 26.2");
  Check(!BrokenNullPrt(AMD, RADV, Version(26, 2, 0)), "RADV 26.2");
  Check(!BrokenNullPrt(AMD, RADV, Version(27, 0, 0)), "RADV 27");
  Check(!BrokenNullPrt(NVIDIA, NVIDIA_PROPRIETARY, Version(580, 0, 0)), "NVIDIA");
  Check(!BrokenNullPrt(INTEL, AMD_PROPRIETARY, 0), "not AMD");
  // The switch: 1 on, 0 off, anything else (or unset) the driver test.
  Check(WantZeroPageTable("1", NVIDIA, NVIDIA_PROPRIETARY, 0), "forced on NVIDIA");
  Check(!WantZeroPageTable("0", AMD, AMD_PROPRIETARY, 0), "forced off on AMD");
  Check(!WantZeroPageTable(nullptr, NVIDIA, NVIDIA_PROPRIETARY, 0), "unset on NVIDIA");
  Check(WantZeroPageTable(nullptr, AMD, AMD_PROPRIETARY, 0), "unset on AMD");
  Check(WantZeroPageTable("", AMD, AMD_PROPRIETARY, 0), "empty on AMD");
  std::printf("NullPrtWorkaroundTests: all passed\n");
  return 0;
}
