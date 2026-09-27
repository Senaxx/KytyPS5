#include "graphics/host_gpu/aftermath.h"

#include "common/common.h"
#include "common/config.h"
#include "common/logging/log.h"

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Libs::Graphics::Aftermath {

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

namespace {

// Declared here from the Aftermath SDK headers (GFSDK_Aftermath_GpuCrashDump*.h), so Kyty builds
// without the SDK and only loads the DLL when asked to.
constexpr uint32_t ApiVersion                 = 0x0000021B;
constexpr uint32_t WatchedApiVulkan           = 0x2;
constexpr uint32_t FeatureDeferDebugInfo      = 0x1;
constexpr uint32_t ResultSuccess              = 0x1;
constexpr uint32_t StatusCollectingDataFailed = 2;
constexpr uint32_t StatusFinished             = 4;
constexpr uint32_t StatusUnknown              = 5;

struct ShaderDebugInfoIdentifier {
	uint64_t id[2];
};

using GpuCrashDumpCb    = void(__cdecl*)(const void*, uint32_t, void*);
using ShaderDebugInfoCb = void(__cdecl*)(const void*, uint32_t, void*);
using EnableFn = uint32_t(__cdecl*)(uint32_t, uint32_t, uint32_t, GpuCrashDumpCb, ShaderDebugInfoCb,
                                    void*, void*, void*);
using GetStatusFn = uint32_t(__cdecl*)(uint32_t*);
using GetShaderDebugInfoIdentifierFn =
    uint32_t(__cdecl*)(uint32_t, const void*, uint32_t, ShaderDebugInfoIdentifier*);

GetStatusFn                    g_get_status     = nullptr;
GetShaderDebugInfoIdentifierFn g_get_identifier = nullptr;

void WriteFile(const std::string& name, const void* data, uint32_t size) {
	if (FILE* file = std::fopen(name.c_str(), "wb"); file != nullptr) {
		std::fwrite(data, 1, size, file);
		std::fclose(file);
	}
}

void __cdecl OnGpuCrashDump(const void* dump, uint32_t size, void* /*user_data*/) {
	WriteFile("_aftermath.nv-gpudmp", dump, size);
	std::printf("Aftermath: GPU crash dump written to _aftermath.nv-gpudmp (%u bytes)\n", size);
	std::fflush(stdout);
}

void __cdecl OnShaderDebugInfo(const void* info, uint32_t size, void* /*user_data*/) {
	ShaderDebugInfoIdentifier identifier {};
	if (g_get_identifier == nullptr ||
	    g_get_identifier(ApiVersion, info, size, &identifier) != ResultSuccess) {
		return;
	}
	char name[96];
	std::snprintf(name, sizeof(name), "_aftermath_%016" PRIx64 "-%016" PRIx64 ".nvdbg",
	              identifier.id[0], identifier.id[1]);
	WriteFile(name, info, size);
}

} // namespace

void Initialize() {
	const char* path = std::getenv("KYTY_AFTERMATH_DLL");
	if (path == nullptr) {
		return;
	}
	HMODULE module = LoadLibraryA(path);
	if (module == nullptr) {
		LOGF("Aftermath: cannot load %s\n", path);
		return;
	}
	auto enable = reinterpret_cast<EnableFn>(
	    reinterpret_cast<void*>(GetProcAddress(module, "GFSDK_Aftermath_EnableGpuCrashDumps")));
	g_get_status = reinterpret_cast<GetStatusFn>(
	    reinterpret_cast<void*>(GetProcAddress(module, "GFSDK_Aftermath_GetCrashDumpStatus")));
	g_get_identifier = reinterpret_cast<GetShaderDebugInfoIdentifierFn>(reinterpret_cast<void*>(
	    GetProcAddress(module, "GFSDK_Aftermath_GetShaderDebugInfoIdentifier")));
	if (enable == nullptr) {
		LOGF("Aftermath: GFSDK_Aftermath_EnableGpuCrashDumps not found\n");
		return;
	}
	const auto result = enable(ApiVersion, WatchedApiVulkan, FeatureDeferDebugInfo, OnGpuCrashDump,
	                           OnShaderDebugInfo, nullptr, nullptr, nullptr);
	LOGF("Aftermath: GPU crash dumps %s (result 0x%08" PRIx32 ")\n",
	     result == ResultSuccess ? "enabled" : "NOT enabled", result);
}

void WaitForCrashDump() {
	if (g_get_status == nullptr) {
		return;
	}
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
	uint32_t   status   = StatusUnknown;
	while (std::chrono::steady_clock::now() < deadline) {
		g_get_status(&status);
		if (status == StatusFinished || status == StatusCollectingDataFailed) {
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}
	std::printf("Aftermath: crash dump status %u\n", status);
	std::fflush(stdout);
}

#else

void Initialize() {}
void WaitForCrashDump() {}

#endif

} // namespace Libs::Graphics::Aftermath
