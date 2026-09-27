#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_AFTERMATH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_AFTERMATH_H_

namespace Libs::Graphics::Aftermath {

// Opt-in NVIDIA Aftermath GPU crash dumps: KYTY_AFTERMATH_DLL=<path to GFSDK_Aftermath_Lib.x64.dll>.
// Kyty already enables shader debug info through VK_NV_device_diagnostics_config; the Aftermath
// runtime receives that debug info, so the crash dump can be mapped back to SPIR-V instructions.
// Writes _aftermath.nv-gpudmp and _aftermath_<id>.nvdbg files next to the log.

// Must be called before the Vulkan device is created.
void Initialize();

// Waits (a few seconds at most) for Aftermath to finish writing the crash dump after a device loss.
void WaitForCrashDump();

} // namespace Libs::Graphics::Aftermath

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_AFTERMATH_H_
