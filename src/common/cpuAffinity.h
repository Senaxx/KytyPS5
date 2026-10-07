#ifndef EMULATOR_SRC_COMMON_CPUAFFINITY_H_
#define EMULATOR_SRC_COMMON_CPUAFFINITY_H_

namespace Common {

// On a CPU with cores of different speeds (Intel 12th-14th gen and later hybrid designs), keeps
// the calling thread on the fastest cores: the emulator's GPU thread is its bottleneck, and on an
// efficiency core it ran ~1.7x slower per operation (i7-13700F on Linux, 2026-10-06). Windows:
// CPU sets of the highest efficiency class (a scheduler preference). Linux: the cores with the
// highest maximum frequency. A no-op on CPUs whose cores are all alike, on macOS, and with
// KYTY_PCORE_AFFINITY=0. `name` is for the log line.
void PreferPerformanceCores(const char* name);

} // namespace Common

#endif // EMULATOR_SRC_COMMON_CPUAFFINITY_H_
