#include "graphics/presentation/renderDoc.h"

#include "common/logging/log.h"

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <renderdoc_app.h>
#include <string>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <dlfcn.h>
#endif

namespace Libs::Graphics {

static RENDERDOC_API_1_6_0*               g_api             = nullptr;
static std::atomic<RenderDocCaptureState> g_state = RenderDocCaptureState::Idle;
static std::atomic_bool                   g_available = false;
static std::atomic_bool                   g_unavailable_log = false;
static std::atomic_uint64_t               g_completed_captures = 0;
static std::atomic_bool                   g_has_result = false;
static std::atomic_bool                   g_last_succeeded = false;
static std::mutex                         g_status_mutex;
static std::string                        g_capture_path;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

static bool BindRenderDocApi(HMODULE module) {
	auto* get_api = reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(module, "RENDERDOC_GetAPI"));
#else
static bool BindRenderDocApi(void* module) {
	auto* get_api = reinterpret_cast<pRENDERDOC_GetAPI>(::dlsym(module, "RENDERDOC_GetAPI"));
#endif
	if (get_api == nullptr) {
		return false;
	}

	void* api = nullptr;
	if (get_api(eRENDERDOC_API_Version_1_6_0, &api) != 1 || api == nullptr) {
		return false;
	}

	g_api = static_cast<RENDERDOC_API_1_6_0*>(api);
	g_api->SetCaptureKeys(nullptr, 0);
	g_api->UnloadCrashHandler();
	g_available.store(true, std::memory_order_release);
	LOGF("RenderDoc: API 1.6.0 bound\n");
	return true;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

void RenderDocInit() {
	if (g_api != nullptr) {
		return;
	}

	auto* module = GetModuleHandleA("renderdoc.dll");
	if (module == nullptr) {
		HKEY key = nullptr;
		if (RegOpenKeyExW(HKEY_LOCAL_MACHINE,
		                  L"SOFTWARE\\Classes\\RenderDoc.RDCCapture.1\\DefaultIcon\\", 0, KEY_READ,
		                  &key) != ERROR_SUCCESS) {
			return;
		}

		std::array<wchar_t, MAX_PATH> path_buffer {};
		DWORD      path_size = static_cast<DWORD>(path_buffer.size() * sizeof(wchar_t));
		const auto result    = RegQueryValueExW(
		    key, L"", nullptr, nullptr, reinterpret_cast<LPBYTE>(path_buffer.data()), &path_size);
		RegCloseKey(key);
		if (result != ERROR_SUCCESS) {
			return;
		}

		auto path = std::filesystem::path(path_buffer.data()).parent_path() / "renderdoc.dll";
		module    = LoadLibraryW(path.c_str());
		if (module == nullptr) {
			return;
		}
	}

	if (!BindRenderDocApi(module)) {
		LOGF("RenderDoc: API 1.6.0 is unavailable\n");
	}
}

#else

void RenderDocInit() {
	if (g_api != nullptr) {
		return;
	}

	auto* module = ::dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
	if (module == nullptr) {
		module = ::dlopen("librenderdoc.so", RTLD_NOW);
	}
	if (module == nullptr) {
		return;
	}

	if (!BindRenderDocApi(module)) {
		LOGF("RenderDoc: API 1.6.0 is unavailable\n");
		::dlclose(module);
	}
}

#endif

bool RenderDocRequestCapture() {
	if (!g_available.load(std::memory_order_acquire)) {
		if (!g_unavailable_log.exchange(true)) {
			LOGF("RenderDoc: capture requested, but RenderDoc is unavailable\n");
		}
		return false;
	}

	RenderDocCaptureState expected = RenderDocCaptureState::Idle;
	if (g_state.compare_exchange_strong(expected, RenderDocCaptureState::Requested)) {
		LOGF("RenderDoc: capture requested\n");
		return true;
	}
	return false;
}

const char* RenderDocCaptureStateName(RenderDocCaptureState state) {
	switch (state) {
		case RenderDocCaptureState::Requested: return "requested";
		case RenderDocCaptureState::Capturing: return "capturing";
		case RenderDocCaptureState::Idle: return "idle";
	}
	return "unknown";
}

RenderDocStatus RenderDocGetStatus() {
	RenderDocStatus status;
	status.available = g_available.load(std::memory_order_acquire);
	status.state = g_state.load(std::memory_order_acquire);
	status.completed_captures = g_completed_captures.load(std::memory_order_acquire);
	status.has_result = g_has_result.load(std::memory_order_acquire);
	status.last_succeeded = g_last_succeeded.load(std::memory_order_acquire);
	const std::lock_guard lock(g_status_mutex);
	status.capture_path = g_capture_path;
	return status;
}

bool RenderDocCaptureRequested() {
	return g_state.load(std::memory_order_acquire) == RenderDocCaptureState::Requested;
}

bool RenderDocCaptureInProgress() {
	return g_state.load(std::memory_order_acquire) == RenderDocCaptureState::Capturing;
}

void RenderDocStartCapture() {
	RenderDocCaptureState expected = RenderDocCaptureState::Requested;
	if (g_api == nullptr || !g_state.compare_exchange_strong(expected, RenderDocCaptureState::Capturing,
	                                                         std::memory_order_acq_rel)) {
		return;
	}

	if (g_api->IsFrameCapturing() != 0) {
		g_state.store(RenderDocCaptureState::Idle, std::memory_order_release);
		LOGF("RenderDoc: capture request ignored because a capture is already active\n");
		return;
	}

	const auto capture_id   = std::chrono::duration_cast<std::chrono::microseconds>(
	                              std::chrono::system_clock::now().time_since_epoch())
	                              .count();
	const auto capture_path = "_RenderDoc/kyty_" + std::to_string(capture_id);
	{
		const std::lock_guard lock(g_status_mutex);
		g_capture_path = capture_path;
	}
	g_api->SetCaptureFilePathTemplate(capture_path.c_str());
	g_api->StartFrameCapture(nullptr, nullptr);
	if (g_api->IsFrameCapturing() == 0) {
		g_state.store(RenderDocCaptureState::Idle, std::memory_order_release);
		LOGF("RenderDoc: capture failed to start\n");
		return;
	}
	LOGF("RenderDoc: capture started\n");
}

void RenderDocEndCapture() {
	if (g_api == nullptr || !RenderDocCaptureInProgress()) {
		return;
	}

	const auto ok = g_api->EndFrameCapture(nullptr, nullptr);
	g_last_succeeded.store(ok != 0, std::memory_order_release);
	g_has_result.store(true, std::memory_order_release);
	g_completed_captures.fetch_add(1, std::memory_order_acq_rel);
	g_state.store(RenderDocCaptureState::Idle, std::memory_order_release);
	LOGF(ok != 0 ? "RenderDoc: capture finished\n" : "RenderDoc: capture failed\n");
}

} // namespace Libs::Graphics
