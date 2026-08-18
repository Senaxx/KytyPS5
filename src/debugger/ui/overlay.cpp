#include "debugger/ui/overlay.h"

#include "SDL.h"
#include "common/assert.h"
#include "debugger/core/session.h"
#include "debugger/symbols/symbols.h"
#include "debugger/target/graphics.h"
#include "graphics/host_gpu/graphicContext.h"
#include "imgui.h"
#include "imgui_impl_vulkan.h"
// For BringWindowToDisplayFront: pinning the control bar above the panels has no public
// equivalent that does not also steal keyboard focus every frame.
#include "imgui_internal.h"
#include "kernel/pthread.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace Debugger::Ui {

namespace {

using Libs::Graphics::GraphicContext;

// Spelled out because plain `Graphics` here would sit next to Libs::Graphics.
namespace Gfx = Debugger::Graphics;

std::atomic<bool> g_visible {false};

// Counts scroll-to-current requests so the "follow rip without pinning the view" behaviour is
// observable from a test; see DebuggerOverlay::ScrollRequestCountForTesting.
uint64_t g_scroll_requests = 0;

// ---- Input queue --------------------------------------------------------------------------
//
// SDL events arrive on the window thread; ImGui's input queue is only safe to touch from the
// thread that calls NewFrame, which here is the presentation thread. Queue on one side, drain
// on the other.

enum class InputKind : uint8_t {
	Key,
	Text,
	MousePosition,
	MouseButton,
	MouseWheel,
};

struct InputEvent {
	InputKind kind = InputKind::Key;
	int       id   = 0;
	float     x    = 0.0f;
	float     y    = 0.0f;
	bool      down = false;
	char      text[8] {};
};

std::mutex             g_input_mutex;
std::deque<InputEvent> g_input_events;

void QueueInput(const InputEvent& event) {
	const std::scoped_lock lock(g_input_mutex);
	// Bound the queue: if the overlay is not drawing (minimised window, say) the events would
	// otherwise accumulate forever.
	if (g_input_events.size() > 4096) {
		g_input_events.pop_front();
	}
	g_input_events.push_back(event);
}

ImGuiKey SdlKeyToImGui(SDL_Keycode key) {
	switch (key) {
		case SDLK_TAB: return ImGuiKey_Tab;
		case SDLK_LEFT: return ImGuiKey_LeftArrow;
		case SDLK_RIGHT: return ImGuiKey_RightArrow;
		case SDLK_UP: return ImGuiKey_UpArrow;
		case SDLK_DOWN: return ImGuiKey_DownArrow;
		case SDLK_PAGEUP: return ImGuiKey_PageUp;
		case SDLK_PAGEDOWN: return ImGuiKey_PageDown;
		case SDLK_HOME: return ImGuiKey_Home;
		case SDLK_END: return ImGuiKey_End;
		case SDLK_INSERT: return ImGuiKey_Insert;
		case SDLK_DELETE: return ImGuiKey_Delete;
		case SDLK_BACKSPACE: return ImGuiKey_Backspace;
		case SDLK_SPACE: return ImGuiKey_Space;
		case SDLK_RETURN: return ImGuiKey_Enter;
		case SDLK_ESCAPE: return ImGuiKey_Escape;
		case SDLK_LCTRL: return ImGuiKey_LeftCtrl;
		case SDLK_LSHIFT: return ImGuiKey_LeftShift;
		case SDLK_LALT: return ImGuiKey_LeftAlt;
		case SDLK_RCTRL: return ImGuiKey_RightCtrl;
		case SDLK_RSHIFT: return ImGuiKey_RightShift;
		case SDLK_RALT: return ImGuiKey_RightAlt;
		case SDLK_F5: return ImGuiKey_F5;
		case SDLK_F9: return ImGuiKey_F9;
		case SDLK_F10: return ImGuiKey_F10;
		case SDLK_F11: return ImGuiKey_F11;
		default: break;
	}

	if (key >= SDLK_a && key <= SDLK_z) {
		return static_cast<ImGuiKey>(ImGuiKey_A + (key - SDLK_a));
	}
	if (key >= SDLK_0 && key <= SDLK_9) {
		return static_cast<ImGuiKey>(ImGuiKey_0 + (key - SDLK_0));
	}

	return ImGuiKey_None;
}

PFN_vkVoidFunction LoadVulkanFunction(const char* name, void* user_data) {
	auto& graphics = *static_cast<GraphicContext*>(user_data);
	return graphics.instance.getProcAddr(name);
}

void CheckVulkanResult(VkResult result) {
	EXIT_IF(result != VK_SUCCESS);
}

// ---- Theme ----------------------------------------------------------------------------------

namespace Palette {
constexpr ImVec4 Background {0.055f, 0.066f, 0.086f, 0.97f};
constexpr ImVec4 Panel {0.082f, 0.101f, 0.129f, 1.00f};
constexpr ImVec4 PanelRaised {0.106f, 0.133f, 0.169f, 1.00f};
constexpr ImVec4 Border {0.176f, 0.212f, 0.263f, 1.00f};
constexpr ImVec4 Text {0.788f, 0.831f, 0.878f, 1.00f};
constexpr ImVec4 TextDim {0.420f, 0.463f, 0.518f, 1.00f};
constexpr ImVec4 Accent {0.290f, 0.639f, 1.000f, 1.00f};
constexpr ImVec4 AccentDim {0.290f, 0.639f, 1.000f, 0.25f};
constexpr ImVec4 Amber {1.000f, 0.706f, 0.329f, 1.00f};
constexpr ImVec4 Green {0.420f, 0.851f, 0.408f, 1.00f};
constexpr ImVec4 Red {1.000f, 0.373f, 0.337f, 1.00f};
constexpr ImVec4 CurrentRow {0.129f, 0.239f, 0.161f, 1.00f};
} // namespace Palette

void ApplyTheme() {
	auto& style = ImGui::GetStyle();

	style.WindowRounding    = 7.0f;
	style.ChildRounding     = 6.0f;
	style.FrameRounding     = 4.0f;
	style.GrabRounding      = 4.0f;
	style.TabRounding       = 5.0f;
	style.ScrollbarRounding = 8.0f;
	style.PopupRounding     = 5.0f;

	style.WindowBorderSize = 1.0f;
	style.ChildBorderSize  = 1.0f;
	style.FrameBorderSize  = 0.0f;
	style.TabBorderSize    = 0.0f;

	style.WindowPadding    = {10.0f, 10.0f};
	style.FramePadding     = {9.0f, 4.0f};
	style.ItemSpacing      = {8.0f, 6.0f};
	style.ItemInnerSpacing = {6.0f, 4.0f};
	style.CellPadding      = {7.0f, 3.0f};
	style.ScrollbarSize    = 12.0f;
	style.GrabMinSize      = 11.0f;

	style.WindowTitleAlign = {0.0f, 0.5f};

	auto* colors = style.Colors;

	colors[ImGuiCol_WindowBg]             = Palette::Background;
	colors[ImGuiCol_ChildBg]              = Palette::Panel;
	colors[ImGuiCol_PopupBg]              = Palette::PanelRaised;
	colors[ImGuiCol_Border]               = Palette::Border;
	colors[ImGuiCol_BorderShadow]         = {0.0f, 0.0f, 0.0f, 0.0f};
	colors[ImGuiCol_Text]                 = Palette::Text;
	colors[ImGuiCol_TextDisabled]         = Palette::TextDim;
	colors[ImGuiCol_FrameBg]              = Palette::PanelRaised;
	colors[ImGuiCol_FrameBgHovered]       = {0.145f, 0.180f, 0.227f, 1.00f};
	colors[ImGuiCol_FrameBgActive]        = {0.176f, 0.220f, 0.278f, 1.00f};
	colors[ImGuiCol_TitleBg]              = Palette::Panel;
	colors[ImGuiCol_TitleBgActive]        = Palette::PanelRaised;
	colors[ImGuiCol_TitleBgCollapsed]     = Palette::Panel;
	colors[ImGuiCol_MenuBarBg]            = Palette::Panel;
	colors[ImGuiCol_ScrollbarBg]          = {0.0f, 0.0f, 0.0f, 0.0f};
	colors[ImGuiCol_ScrollbarGrab]        = Palette::Border;
	colors[ImGuiCol_ScrollbarGrabHovered] = {0.239f, 0.286f, 0.353f, 1.00f};
	colors[ImGuiCol_ScrollbarGrabActive]  = Palette::Accent;
	colors[ImGuiCol_CheckMark]            = Palette::Accent;
	colors[ImGuiCol_SliderGrab]           = Palette::Accent;
	colors[ImGuiCol_SliderGrabActive]     = Palette::Accent;
	colors[ImGuiCol_Button]               = Palette::PanelRaised;
	colors[ImGuiCol_ButtonHovered]        = {0.180f, 0.286f, 0.404f, 1.00f};
	colors[ImGuiCol_ButtonActive]         = Palette::AccentDim;
	colors[ImGuiCol_Header]               = Palette::AccentDim;
	colors[ImGuiCol_HeaderHovered]        = {0.290f, 0.639f, 1.000f, 0.35f};
	colors[ImGuiCol_HeaderActive]         = {0.290f, 0.639f, 1.000f, 0.45f};
	colors[ImGuiCol_Separator]            = Palette::Border;
	colors[ImGuiCol_SeparatorHovered]     = Palette::Accent;
	colors[ImGuiCol_SeparatorActive]      = Palette::Accent;
	colors[ImGuiCol_ResizeGrip]           = Palette::Border;
	colors[ImGuiCol_ResizeGripHovered]    = Palette::Accent;
	colors[ImGuiCol_ResizeGripActive]     = Palette::Accent;
	colors[ImGuiCol_Tab]                  = Palette::Panel;
	colors[ImGuiCol_TabHovered]           = Palette::AccentDim;
	colors[ImGuiCol_TabSelected]          = Palette::PanelRaised;
	colors[ImGuiCol_TabSelectedOverline]  = Palette::Accent;
	colors[ImGuiCol_TabDimmed]            = Palette::Panel;
	colors[ImGuiCol_TabDimmedSelected]    = Palette::PanelRaised;
	colors[ImGuiCol_TableHeaderBg]        = Palette::PanelRaised;
	colors[ImGuiCol_TableBorderStrong]    = Palette::Border;
	colors[ImGuiCol_TableBorderLight]     = {0.129f, 0.157f, 0.196f, 1.00f};
	colors[ImGuiCol_TableRowBg]           = {0.0f, 0.0f, 0.0f, 0.0f};
	colors[ImGuiCol_TableRowBgAlt]        = {1.0f, 1.0f, 1.0f, 0.018f};
	colors[ImGuiCol_TextSelectedBg]       = Palette::AccentDim;
	colors[ImGuiCol_NavCursor]            = Palette::Accent;
}

// ---- Small widgets ----------------------------------------------------------------------------

std::string Hex64(uint64_t value) {
	std::array<char, 32> buffer {};
	std::snprintf(buffer.data(), buffer.size(), "%016llx", static_cast<unsigned long long>(value));
	return buffer.data();
}

const char* StopReasonName(StopReason reason) {
	switch (reason) {
		case StopReason::Breakpoint: return "breakpoint";
		case StopReason::Step: return "step";
		case StopReason::Pause: return "paused";
		case StopReason::Entry: return "entry";
		case StopReason::Fatal: return "fatal error";
		case StopReason::None: break;
	}
	return "running";
}

// Uppercase caption with a rule under it, used to title each pane. Panes are plain regions of
// one window rather than separate ImGui windows, so they need their own visual separation.
void PaneTitle(const char* label, const ImVec4& color = Palette::TextDim) {
	ImGui::PushStyleColor(ImGuiCol_Text, color);
	ImGui::TextUnformatted(label);
	ImGui::PopStyleColor();

	const ImVec2 min = ImGui::GetItemRectMin();
	const ImVec2 max = ImGui::GetItemRectMax();
	ImGui::GetWindowDrawList()->AddLine({min.x, max.y + 2.0f},
	                                    {min.x + ImGui::GetContentRegionAvail().x, max.y + 2.0f},
	                                    ImGui::GetColorU32(ImGuiCol_Border));
	ImGui::Dummy({0.0f, 3.0f});
}

constexpr float SPLITTER_THICKNESS = 6.0f;

// Draggable dividers between panes. Implemented over the public API (an invisible button plus a
// drawn handle) so pane sizes stay owned by the overlay rather than by ImGui's internal
// child-resize state, which does not persist without an .ini file.
//
// `grows_with_drag` says which side of the divider the tracked size belongs to: true when the
// pane before the divider owns it (dragging away makes it bigger), false when the pane after it
// does (dragging away makes it smaller).
void VerticalSplitter(const char* id, float height, float* width, float min_width, float max_width,
                      bool grows_with_drag) {
	ImGui::SameLine(0.0f, 0.0f);

	const ImVec2 origin = ImGui::GetCursorScreenPos();
	ImGui::InvisibleButton(id, {SPLITTER_THICKNESS, height});

	const bool active  = ImGui::IsItemActive();
	const bool hovered = ImGui::IsItemHovered();

	if (active || hovered) {
		ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
	}
	if (active) {
		const float delta = ImGui::GetIO().MouseDelta.x * (grows_with_drag ? 1.0f : -1.0f);
		*width            = std::clamp(*width + delta, min_width, max_width);
	}

	const ImU32 color  = ImGui::GetColorU32(active    ? ImGuiCol_SeparatorActive
	                                        : hovered ? ImGuiCol_SeparatorHovered
	                                                  : ImGuiCol_Separator);
	const float centre = origin.x + SPLITTER_THICKNESS * 0.5f;
	ImGui::GetWindowDrawList()->AddRectFilled({centre - 0.5f, origin.y + 4.0f},
	                                          {centre + 0.5f, origin.y + height - 4.0f}, color);

	ImGui::SameLine(0.0f, 0.0f);
}

void HorizontalSplitter(const char* id, float width, float* height, float min_height,
                        float max_height, bool grows_with_drag) {
	const ImVec2 origin = ImGui::GetCursorScreenPos();
	ImGui::InvisibleButton(id, {width, SPLITTER_THICKNESS});

	const bool active  = ImGui::IsItemActive();
	const bool hovered = ImGui::IsItemHovered();

	if (active || hovered) {
		ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
	}
	if (active) {
		const float delta = ImGui::GetIO().MouseDelta.y * (grows_with_drag ? 1.0f : -1.0f);
		*height           = std::clamp(*height + delta, min_height, max_height);
	}

	const ImU32 color  = ImGui::GetColorU32(active    ? ImGuiCol_SeparatorActive
	                                        : hovered ? ImGuiCol_SeparatorHovered
	                                                  : ImGuiCol_Separator);
	const float centre = origin.y + SPLITTER_THICKNESS * 0.5f;
	ImGui::GetWindowDrawList()->AddRectFilled({origin.x + 4.0f, centre - 0.5f},
	                                          {origin.x + width - 4.0f, centre + 0.5f}, color);
}

// Pick an address to start a listing from so that `target` has some context above it.
//
// x86 cannot be decoded backwards, so this walks forward from successively closer candidates
// and keeps the first whose instruction stream lands exactly on `target`. A wrong starting
// point desynchronises and simply fails that check. Starting from the farthest candidate gives
// the most context. Only called when the listing re-anchors, not per frame.
uint64_t FindListingStart(uint64_t target, uint32_t max_back_bytes) {
	if (target <= max_back_bytes) {
		return target;
	}

	for (uint32_t back = max_back_bytes; back >= 1; back--) {
		uint64_t cursor = target - back;

		while (cursor < target) {
			const uint32_t length = Disasm::InstructionLength(cursor);
			if (length == 0) {
				break;
			}
			cursor += length;
		}

		if (cursor == target) {
			return target - back;
		}
	}

	return target;
}

// Zydis hands back one formatted string; splitting at the first space lets the mnemonic be
// tinted without re-implementing the formatter.
void DrawInstructionText(const std::string& text) {
	const auto space = text.find(' ');
	if (space == std::string::npos) {
		ImGui::TextColored(Palette::Accent, "%s", text.c_str());
		return;
	}

	ImGui::TextColored(Palette::Accent, "%s", text.substr(0, space).c_str());
	ImGui::SameLine(0.0f, ImGui::CalcTextSize(" ").x);
	ImGui::TextUnformatted(text.c_str() + space + 1);
}

} // namespace

bool ProcessEvent(const SDL_Event& event) {
	// The toggle key is handled whether or not the overlay is up; everything else only when it
	// is, so the guest keeps receiving input while the debugger is hidden.
	if (event.type == SDL_KEYDOWN && event.key.repeat == 0 &&
	    (event.key.keysym.sym == SDLK_F1 || event.key.keysym.sym == SDLK_BACKQUOTE)) {
		Toggle();
		return true;
	}

	if (!g_visible.load(std::memory_order_acquire)) {
		return false;
	}

	switch (event.type) {
		case SDL_KEYDOWN:
		case SDL_KEYUP: {
			InputEvent input {};
			input.kind = InputKind::Key;
			input.id   = static_cast<int>(SdlKeyToImGui(event.key.keysym.sym));
			input.down = event.type == SDL_KEYDOWN;
			if (input.id != ImGuiKey_None) {
				QueueInput(input);
			}
			return true;
		}
		case SDL_TEXTINPUT: {
			InputEvent input {};
			input.kind = InputKind::Text;
			std::strncpy(input.text, event.text.text, sizeof(input.text) - 1);
			QueueInput(input);
			return true;
		}
		case SDL_MOUSEMOTION: {
			InputEvent input {};
			input.kind = InputKind::MousePosition;
			input.x    = static_cast<float>(event.motion.x);
			input.y    = static_cast<float>(event.motion.y);
			QueueInput(input);
			return true;
		}
		case SDL_MOUSEBUTTONDOWN:
		case SDL_MOUSEBUTTONUP: {
			InputEvent input {};
			input.kind = InputKind::MouseButton;
			input.id   = event.button.button == SDL_BUTTON_RIGHT    ? 1
			             : event.button.button == SDL_BUTTON_MIDDLE ? 2
			                                                        : 0;
			input.down = event.type == SDL_MOUSEBUTTONDOWN;
			QueueInput(input);
			return true;
		}
		case SDL_MOUSEWHEEL: {
			InputEvent input {};
			input.kind = InputKind::MouseWheel;
			input.x    = static_cast<float>(event.wheel.x);
			input.y    = static_cast<float>(event.wheel.y);
			QueueInput(input);
			return true;
		}
		default: return false;
	}
}

bool IsVisible() {
	return g_visible.load(std::memory_order_acquire);
}

void SetVisible(bool visible) {
	g_visible.store(visible, std::memory_order_release);
}

void Toggle() {
	g_visible.store(!g_visible.load(std::memory_order_acquire), std::memory_order_release);
}

// ---- Overlay ------------------------------------------------------------------------------

struct DebuggerOverlay::Impl {
	explicit Impl(GraphicContext& context): graphics(context) {}

	~Impl() {
		ReleaseVulkan();
		if (imgui_context != nullptr) {
			ImGui::DestroyContext(imgui_context);
		}
	}

	KYTY_CLASS_NO_COPY(Impl);

	void EnsureContext() {
		if (imgui_context != nullptr) {
			ImGui::SetCurrentContext(imgui_context);
			return;
		}

		IMGUI_CHECKVERSION();
		imgui_context = ImGui::CreateContext();
		ImGui::SetCurrentContext(imgui_context);

		auto& io               = ImGui::GetIO();
		io.IniFilename         = nullptr;
		io.LogFilename         = nullptr;
		io.BackendPlatformName = "Kyty debugger";

		ApplyTheme();
	}

	void EnsureVulkan(vk::Format format, uint32_t image_count) {
		EnsureContext();
		if (vulkan_initialized) {
			return;
		}

		EXIT_IF(image_count < 2);
		EXIT_IF(!ImGui_ImplVulkan_LoadFunctions(VK_API_VERSION_1_3, LoadVulkanFunction, &graphics));

		const VkFormat            color_format = static_cast<VkFormat>(format);
		ImGui_ImplVulkan_InitInfo info {};
		info.ApiVersion                   = VK_API_VERSION_1_3;
		info.Instance                     = static_cast<VkInstance>(graphics.instance);
		info.PhysicalDevice               = static_cast<VkPhysicalDevice>(graphics.physical_device);
		info.Device                       = static_cast<VkDevice>(graphics.device);
		info.QueueFamily                  = graphics.queue_family;
		info.Queue                        = static_cast<VkQueue>(graphics.queue);
		info.DescriptorPoolSize           = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE;
		info.MinImageCount                = image_count;
		info.ImageCount                   = image_count;
		info.UseDynamicRendering          = true;
		info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
		info.PipelineInfoMain.PipelineRenderingCreateInfo.sType =
		    VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
		info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount    = 1;
		info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &color_format;
		info.CheckVkResultFn = CheckVulkanResult;

		EXIT_IF(!ImGui_ImplVulkan_Init(&info));
		vulkan_initialized = true;
	}

	void DrainInput() {
		std::deque<InputEvent> events;
		{
			const std::scoped_lock lock(g_input_mutex);
			events.swap(g_input_events);
		}

		auto& io = ImGui::GetIO();
		for (const auto& event: events) {
			switch (event.kind) {
				case InputKind::Key:
					io.AddKeyEvent(static_cast<ImGuiKey>(event.id), event.down);
					break;
				case InputKind::Text: io.AddInputCharactersUTF8(event.text); break;
				case InputKind::MousePosition: io.AddMousePosEvent(event.x, event.y); break;
				case InputKind::MouseButton: io.AddMouseButtonEvent(event.id, event.down); break;
				case InputKind::MouseWheel: io.AddMouseWheelEvent(event.x, event.y); break;
			}
		}
	}

	// ---- Panes ------------------------------------------------------------------------------

	// Point the disassembly somewhere explicitly, which also stops it following rip and puts the
	// target back at the top of the view.
	void GoTo(uint64_t address) {
		disasm_address = address;
		follow_rip     = false;
		scroll_to_top  = true;
	}

	// Re-follow the halted thread from scratch, re-anchoring and scrolling on the next frame.
	void FollowHaltedThread(uint64_t address) {
		disasm_address   = address;
		follow_rip       = true;
		followed_address = 0;
	}

	// A button that reads as pressed while its window is open.
	void WindowToggle(const char* label, bool* open) {
		ImGui::PushStyleColor(ImGuiCol_Button, *open ? Palette::AccentDim
		                                             : ImGui::GetStyle().Colors[ImGuiCol_Button]);
		ImGui::PushStyleColor(ImGuiCol_Text, *open ? Palette::Accent : Palette::TextDim);
		if (ImGui::Button(label)) {
			*open = !*open;
		}
		ImGui::PopStyleColor(2);
	}

	// The control bar is pinned to the top of the viewport and stays above the panels, so the
	// run controls are reachable whatever is open and wherever it has been dragged.
	void DrawControlBar(const StoppedThread* selected) {
		const auto* viewport = ImGui::GetMainViewport();

		ImGui::SetNextWindowPos({viewport->WorkPos.x, viewport->WorkPos.y});
		ImGui::SetNextWindowSize({viewport->WorkSize.x, 0.0f});

		constexpr auto flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
		                       ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
		                       ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNavFocus |
		                       ImGuiWindowFlags_AlwaysAutoResize;

		if (!ImGui::Begin("##controlbar", nullptr, flags)) {
			ImGui::End();
			return;
		}

		const bool paused = Session::IsPaused();

		// A fatal stop is on its way to terminating the process: it can be read, but not stepped,
		// and continuing only lets the exit proceed.
		const bool fatal = selected != nullptr && selected->reason == StopReason::Fatal;

		// Status pill.
		const ImVec4 pill_color = fatal ? Palette::Red : (paused ? Palette::Amber : Palette::Green);
		ImGui::PushStyleColor(ImGuiCol_Text, pill_color);
		ImGui::TextUnformatted(fatal ? "* FATAL" : (paused ? "* HALTED" : "* RUNNING"));
		ImGui::PopStyleColor();

		ImGui::SameLine(0.0f, 16.0f);

		ImGui::BeginDisabled(!paused);
		if (ImGui::Button(fatal ? "Exit" : "Continue")) {
			Session::ResumeAll();
		}
		ImGui::EndDisabled();
		ImGui::SameLine();

		ImGui::BeginDisabled(!paused || fatal);
		if (ImGui::Button("Step Into")) {
			Session::Resume(selected_thread, Session::ResumeMode::StepInto);
		}
		ImGui::SameLine();
		if (ImGui::Button("Step Over")) {
			Session::Resume(selected_thread, Session::ResumeMode::StepOver);
		}
		ImGui::SameLine();
		if (ImGui::Button("Step Out")) {
			Session::Resume(selected_thread, Session::ResumeMode::StepOut);
		}
		ImGui::EndDisabled();

		ImGui::SameLine();
		ImGui::BeginDisabled(paused);
		if (ImGui::Button("Pause")) {
			Session::Pause();
		}
		ImGui::EndDisabled();

		ImGui::SameLine(0.0f, 24.0f);
		ImGui::TextColored(Palette::TextDim, "|");
		ImGui::SameLine(0.0f, 24.0f);

		WindowToggle("Debugger", &show_debugger);
		ImGui::SameLine();
		WindowToggle("Memory", &show_memory);
		ImGui::SameLine();
		WindowToggle("Graphics", &show_graphics);
		ImGui::SameLine();
		WindowToggle("Modules", &show_lookup);

		ImGui::SameLine(0.0f, 16.0f);
		bool break_on_fatal = Session::BreakOnFatalEnabled();
		if (ImGui::Checkbox("break on fatal", &break_on_fatal)) {
			Session::SetBreakOnFatal(break_on_fatal);
		}

		// Where we stopped, on the right.
		if (selected != nullptr) {
			const auto  location = Symbols::Format(selected->address);
			const auto  text     = std::string(StopReasonName(selected->reason)) + "  " + location;
			const float width    = ImGui::CalcTextSize(text.c_str()).x;
			const float avail    = ImGui::GetContentRegionAvail().x;
			if (avail > width + 24.0f) {
				ImGui::SameLine(0.0f, avail - width - 8.0f);
				ImGui::TextColored(Palette::TextDim, "%s", text.c_str());
			}
		}

		control_bar_height = ImGui::GetWindowHeight();

		ImGui::End();

		// ImGui orders windows by focus and offers no public way to pin one to the front.
		// SetWindowFocus would do it but would steal focus every frame, so typing into a panel
		// would be impossible. This is the same internal call ImGui's own docking host uses.
		if (auto* window = ImGui::FindWindowByName("##controlbar"); window != nullptr) {
			ImGui::BringWindowToDisplayFront(window);
		}
	}

	void DrawThreadsPane(const std::vector<StoppedThread>& stopped, float height) {
		if (!ImGui::BeginChild("##threads", {0.0f, height}, ImGuiChildFlags_Borders)) {
			ImGui::EndChild();
			return;
		}

		PaneTitle("THREADS");

		std::vector<Libs::LibKernel::GuestThreadInfo> threads;
		Libs::LibKernel::PthreadEnumerate(threads);

		if (threads.empty()) {
			ImGui::TextColored(Palette::TextDim, "no guest threads yet");
			ImGui::EndChild();
			return;
		}

		if (ImGui::BeginTable("threads", 3,
		                      ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
		                          ImGuiTableFlags_SizingStretchProp)) {
			ImGui::TableSetupColumn("id", ImGuiTableColumnFlags_WidthFixed, 34.0f);
			ImGui::TableSetupColumn("thread");
			ImGui::TableSetupColumn("state", ImGuiTableColumnFlags_WidthFixed, 72.0f);
			ImGui::TableSetupScrollFreeze(0, 1);
			ImGui::TableHeadersRow();

			for (const auto& thread: threads) {
				const StoppedThread* stop = nullptr;
				for (const auto& candidate: stopped) {
					if (candidate.unique_id == thread.unique_id) {
						stop = &candidate;
						break;
					}
				}

				ImGui::TableNextRow();
				ImGui::TableNextColumn();

				ImGui::PushID(thread.unique_id);

				const bool is_selected = selected_thread == thread.unique_id;
				if (ImGui::Selectable("##row", is_selected,
				                      ImGuiSelectableFlags_SpanAllColumns |
				                          ImGuiSelectableFlags_AllowOverlap)) {
					selected_thread = thread.unique_id;
					if (stop != nullptr) {
						FollowHaltedThread(stop->address);
					}
				}
				ImGui::SameLine(0.0f, 0.0f);
				ImGui::TextColored(Palette::TextDim, "%d", thread.unique_id);

				ImGui::TableNextColumn();
				ImGui::TextUnformatted(thread.name.empty() ? "(unnamed)" : thread.name.c_str());

				ImGui::TableNextColumn();
				if (stop != nullptr) {
					ImGui::TextColored(Palette::Amber, "%s", StopReasonName(stop->reason));
				} else if (thread.alive) {
					ImGui::TextColored(Palette::Green, "running");
				} else {
					ImGui::TextColored(Palette::TextDim, "done");
				}

				ImGui::PopID();
			}

			ImGui::EndTable();
		}

		ImGui::EndChild();
	}

	void DrawCallStackPane(const StoppedThread* selected) {
		if (!ImGui::BeginChild("##callstack", {0.0f, 0.0f}, ImGuiChildFlags_Borders)) {
			ImGui::EndChild();
			return;
		}

		PaneTitle("CALL STACK");

		if (selected == nullptr) {
			ImGui::TextColored(Palette::TextDim, "no thread halted");
			ImGui::EndChild();
			return;
		}

		const auto frames = Session::Backtrace(selected->unique_id);
		for (size_t i = 0; i < frames.size(); i++) {
			ImGui::PushID(static_cast<int>(i));

			ImGui::TextColored(Palette::TextDim, "%02zu", i);
			ImGui::SameLine(0.0f, 8.0f);
			if (ImGui::Selectable(frames[i].description.c_str(),
			                      disasm_address == frames[i].address)) {
				GoTo(frames[i].address);
			}

			ImGui::PopID();
		}

		if (frames.size() > 1) {
			ImGui::Dummy({0.0f, 4.0f});
			ImGui::TextColored(Palette::TextDim, "frame-pointer walk; truncates in FPO code");
		}

		ImGui::EndChild();
	}

	void DrawDisassemblyPane(const StoppedThread* selected, float width, float height) {
		if (!ImGui::BeginChild("##disasm", {width, height}, ImGuiChildFlags_Borders)) {
			ImGui::EndChild();
			return;
		}

		PaneTitle("DISASSEMBLY");

		// Follow rip without fighting the user. Re-anchoring the listing every frame would put
		// rip on row 0, where centring it clamps the scroll to the top, so any scrolling was
		// undone on the very next frame. Instead: move the anchor only when rip leaves the
		// decoded range, and scroll only when the halt address actually changes.
		if (selected != nullptr && follow_rip && selected->address != followed_address) {
			const bool in_range = listing_end > listing_first &&
			                      selected->address >= listing_first &&
			                      selected->address < listing_end;
			if (!in_range) {
				disasm_address = FindListingStart(selected->address, 48);
			}
			followed_address  = selected->address;
			scroll_to_current = true;
		}

		ImGui::SetNextItemWidth(220.0f);
		if (ImGui::InputTextWithHint("##goto", "symbol, module+off, or 0x...", goto_buffer.data(),
		                             goto_buffer.size(), ImGuiInputTextFlags_EnterReturnsTrue)) {
			uint64_t resolved = 0;
			if (Symbols::Resolve(goto_buffer.data(), resolved)) {
				GoTo(resolved);
			}
		}
		ImGui::SameLine();
		if (ImGui::Checkbox("follow rip", &follow_rip) && follow_rip) {
			followed_address = 0; // re-anchor and scroll on the next frame
		}
		ImGui::SameLine();
		if (ImGui::SmallButton("-page")) {
			GoTo(disasm_address - 64);
		}
		ImGui::SameLine();
		if (ImGui::SmallButton("+page")) {
			GoTo(disasm_address + 64);
		}

		ImGui::Dummy({0.0f, 2.0f});

		const auto instructions = Disasm::Decode(disasm_address, 128);
		const auto breakpoints  = Session::Breakpoints();

		// Remember what this listing covers so the next frame can tell whether rip is still
		// inside it without decoding again.
		listing_first = disasm_address;
		listing_end   = instructions.empty()
		                    ? disasm_address
		                    : instructions.back().address + instructions.back().length;

		if (instructions.empty()) {
			ImGui::TextColored(Palette::TextDim, "nothing mapped at %s",
			                   Hex64(disasm_address).c_str());
			ImGui::EndChild();
			return;
		}

		if (ImGui::BeginTable("disasm", 4,
		                      ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit)) {
			ImGui::TableSetupColumn("##bp", ImGuiTableColumnFlags_WidthFixed, 18.0f);
			ImGui::TableSetupColumn("##addr", ImGuiTableColumnFlags_WidthFixed, 136.0f);
			ImGui::TableSetupColumn("##bytes", ImGuiTableColumnFlags_WidthFixed, 132.0f);
			ImGui::TableSetupColumn("##text", ImGuiTableColumnFlags_WidthStretch);

			// Jumping somewhere new replaces the whole listing, so an inherited scroll offset
			// would leave the caller looking well past what they asked for.
			if (scroll_to_top) {
				ImGui::SetScrollY(0.0f);
				scroll_to_top = false;
			}

			uint32_t remove_breakpoint = 0;
			uint64_t add_breakpoint    = 0;

			for (const auto& instruction: instructions) {
				const bool is_current =
				    selected != nullptr && selected->address == instruction.address;

				// A label row above the first instruction of a known symbol.
				if (!instruction.symbol.empty()) {
					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					ImGui::TableNextColumn();
					ImGui::TableNextColumn();
					ImGui::TableNextColumn();
					ImGui::TextColored(Palette::Amber, "%s:", instruction.symbol.c_str());
				}

				ImGui::TableNextRow();
				if (is_current) {
					ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
					                       ImGui::GetColorU32(Palette::CurrentRow));
				}

				ImGui::PushID(static_cast<int>(instruction.address));

				// Breakpoint gutter.
				ImGui::TableNextColumn();
				const auto existing =
				    std::find_if(breakpoints.begin(), breakpoints.end(), [&](const Breakpoint& bp) {
					    return bp.address == instruction.address && !bp.one_shot;
				    });
				const bool has_breakpoint = existing != breakpoints.end();

				const ImVec2 gutter = ImGui::GetCursorScreenPos();
				if (ImGui::InvisibleButton("##bp", {14.0f, ImGui::GetTextLineHeight()})) {
					if (has_breakpoint) {
						remove_breakpoint = existing->id;
					} else {
						add_breakpoint = instruction.address;
					}
				}
				if (has_breakpoint) {
					ImGui::GetWindowDrawList()->AddCircleFilled(
					    {gutter.x + 6.0f, gutter.y + ImGui::GetTextLineHeight() * 0.5f}, 4.5f,
					    ImGui::GetColorU32(Palette::Red));
				} else if (ImGui::IsItemHovered()) {
					ImGui::GetWindowDrawList()->AddCircle(
					    {gutter.x + 6.0f, gutter.y + ImGui::GetTextLineHeight() * 0.5f}, 4.5f,
					    ImGui::GetColorU32(Palette::TextDim));
				}

				// Address, with a marker on the halted instruction.
				ImGui::TableNextColumn();
				if (is_current) {
					ImGui::TextColored(Palette::Green, ">");
					ImGui::SameLine(0.0f, 4.0f);
					ImGui::TextColored(Palette::Text, "%s", Hex64(instruction.address).c_str());
					if (scroll_to_current) {
						ImGui::SetScrollHereY(0.35f);
						g_scroll_requests++;
					}
				} else {
					ImGui::TextColored(Palette::TextDim, "  %s",
					                   Hex64(instruction.address).c_str());
				}

				ImGui::TableNextColumn();
				ImGui::TextColored(Palette::TextDim, "%s", instruction.bytes.c_str());

				ImGui::TableNextColumn();
				DrawInstructionText(instruction.text);

				ImGui::PopID();
			}

			ImGui::EndTable();

			if (remove_breakpoint != 0) {
				Session::RemoveBreakpoint(remove_breakpoint);
			}
			if (add_breakpoint != 0) {
				Session::AddBreakpointAt(add_breakpoint);
			}
		}

		// Cleared unconditionally: rip may not be in the listing at all (a jump into unmapped
		// memory, say), and a request that never found its row must not persist and re-fire.
		scroll_to_current = false;

		ImGui::EndChild();
	}

	void DrawRegistersPane(const StoppedThread* selected, float height) {
		if (!ImGui::BeginChild("##registers", {0.0f, height}, ImGuiChildFlags_Borders)) {
			ImGui::EndChild();
			return;
		}

		PaneTitle("REGISTERS");

		if (selected == nullptr || !selected->regs.valid) {
			ImGui::TextColored(Palette::TextDim, "no thread halted");
			ImGui::EndChild();
			return;
		}

		const auto& regs = selected->regs;

		struct Entry {
			const char* name;
			uint64_t    value;
			uint64_t    previous;
		};

		const std::array<Entry, 18> entries {{
		    {"rax", regs.rax, compare_basis.rax},
		    {"rbx", regs.rbx, compare_basis.rbx},
		    {"rcx", regs.rcx, compare_basis.rcx},
		    {"rdx", regs.rdx, compare_basis.rdx},
		    {"rsi", regs.rsi, compare_basis.rsi},
		    {"rdi", regs.rdi, compare_basis.rdi},
		    {"rbp", regs.rbp, compare_basis.rbp},
		    {"rsp", regs.rsp, compare_basis.rsp},
		    {"r8", regs.r8, compare_basis.r8},
		    {"r9", regs.r9, compare_basis.r9},
		    {"r10", regs.r10, compare_basis.r10},
		    {"r11", regs.r11, compare_basis.r11},
		    {"r12", regs.r12, compare_basis.r12},
		    {"r13", regs.r13, compare_basis.r13},
		    {"r14", regs.r14, compare_basis.r14},
		    {"r15", regs.r15, compare_basis.r15},
		    {"rip", regs.rip, compare_basis.rip},
		    {"efl", regs.rflags, compare_basis.rflags},
		}};

		if (ImGui::BeginTable("registers", 2,
		                      ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit)) {
			ImGui::TableSetupColumn("##name", ImGuiTableColumnFlags_WidthFixed, 34.0f);
			ImGui::TableSetupColumn("##value", ImGuiTableColumnFlags_WidthStretch);

			for (const auto& entry: entries) {
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				ImGui::TextColored(Palette::TextDim, "%s", entry.name);
				ImGui::TableNextColumn();

				// Registers the last step changed are worth spotting at a glance.
				const bool changed = compare_basis.valid && entry.value != entry.previous;
				ImGui::TextColored(changed ? Palette::Amber : Palette::Text, "%s",
				                   Hex64(entry.value).c_str());
			}

			ImGui::EndTable();
		}

		ImGui::Dummy({0.0f, 4.0f});
		ImGui::TextColored(Palette::TextDim, "at");
		ImGui::TextWrapped("%s", Symbols::Format(regs.rip).c_str());

		ImGui::EndChild();
	}

	void DrawBreakpointsTab() {
		ImGui::SetNextItemWidth(260.0f);
		const bool submit = ImGui::InputTextWithHint(
		    "##location", "symbol, module+off, or 0x...", breakpoint_buffer.data(),
		    breakpoint_buffer.size(), ImGuiInputTextFlags_EnterReturnsTrue);
		ImGui::SameLine();
		if ((ImGui::Button("Add") || submit) && breakpoint_buffer[0] != '\0') {
			Session::AddBreakpoint(breakpoint_buffer.data());
			breakpoint_buffer.fill('\0');
		}
		ImGui::SameLine();
		if (ImGui::Button("Clear all")) {
			Session::ClearBreakpoints();
		}

		ImGui::Dummy({0.0f, 2.0f});

		uint32_t remove_id = 0;

		if (ImGui::BeginTable("breakpoints", 5,
		                      ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
		                          ImGuiTableFlags_SizingStretchProp)) {
			ImGui::TableSetupColumn("on", ImGuiTableColumnFlags_WidthFixed, 30.0f);
			ImGui::TableSetupColumn("location");
			ImGui::TableSetupColumn("hits", ImGuiTableColumnFlags_WidthFixed, 50.0f);
			ImGui::TableSetupColumn("state", ImGuiTableColumnFlags_WidthFixed, 70.0f);
			ImGui::TableSetupColumn("##x", ImGuiTableColumnFlags_WidthFixed, 26.0f);
			ImGui::TableSetupScrollFreeze(0, 1);
			ImGui::TableHeadersRow();

			bool any = false;
			for (const auto& bp: Session::Breakpoints()) {
				if (bp.one_shot) {
					continue; // internal step-over helpers
				}
				any = true;

				ImGui::TableNextRow();
				ImGui::PushID(static_cast<int>(bp.id));

				ImGui::TableNextColumn();
				bool enabled = bp.enabled;
				if (ImGui::Checkbox("##enabled", &enabled)) {
					Session::SetBreakpointEnabled(bp.id, enabled);
				}

				ImGui::TableNextColumn();
				if (ImGui::Selectable(bp.label.c_str())) {
					GoTo(bp.address);
				}

				ImGui::TableNextColumn();
				ImGui::TextColored(bp.hit_count != 0 ? Palette::Text : Palette::TextDim, "%llu",
				                   static_cast<unsigned long long>(bp.hit_count));

				ImGui::TableNextColumn();
				if (bp.pending) {
					ImGui::TextColored(Palette::Amber, "pending");
				} else if (bp.armed) {
					ImGui::TextColored(Palette::Green, "armed");
				} else {
					ImGui::TextColored(Palette::TextDim, "off");
				}

				ImGui::TableNextColumn();
				if (ImGui::SmallButton("x")) {
					remove_id = bp.id;
				}

				ImGui::PopID();
			}

			ImGui::EndTable();

			if (!any) {
				ImGui::TextColored(Palette::TextDim,
				                   "no breakpoints - click the gutter in the disassembly, or add "
				                   "one above");
			}
		}

		if (remove_id != 0) {
			Session::RemoveBreakpoint(remove_id);
		}
	}

	// The hex view is a window of MEMORY_SPAN bytes anchored around wherever you navigated to,
	// so the scroll wheel works over a useful range without pretending to scroll all 2^64.
	// Going further afield is what the address box, the register buttons and pointer-following
	// are for.
	static constexpr uint64_t MEMORY_SPAN = 0x10000;

	void GoToMemory(uint64_t address, bool record_history = true) {
		if (record_history && memory_selected != 0) {
			if (memory_history.size() >= 32) {
				memory_history.erase(memory_history.begin());
			}
			memory_history.push_back(memory_selected);
		}

		memory_selected = address;
		memory_base     = (address > MEMORY_SPAN / 2) ? (address - MEMORY_SPAN / 2) : 0;
		memory_base -= memory_base % memory_bytes_per_row;
		memory_scroll_pending = true;
		memory_snapshot_valid = false;
		memory_edit_address   = 0;
	}

	void DrawMemoryToolbar(const StoppedThread* selected) {
		ImGui::SetNextItemWidth(230.0f);
		if (ImGui::InputTextWithHint("##address", "symbol, module+off, or 0x...",
		                             memory_buffer.data(), memory_buffer.size(),
		                             ImGuiInputTextFlags_EnterReturnsTrue)) {
			uint64_t resolved = 0;
			if (Symbols::Resolve(memory_buffer.data(), resolved)) {
				GoToMemory(resolved);
			}
		}

		ImGui::SameLine();
		ImGui::BeginDisabled(memory_history.empty());
		if (ImGui::Button("Back")) {
			const uint64_t previous = memory_history.back();
			memory_history.pop_back();
			GoToMemory(previous, false);
		}
		ImGui::EndDisabled();

		// Jumping to the stack or the current instruction is most of what this view is for while
		// something is halted.
		if (selected != nullptr && selected->regs.valid) {
			ImGui::SameLine(0.0f, 16.0f);
			ImGui::TextColored(Palette::TextDim, "go to");
			ImGui::SameLine();
			if (ImGui::SmallButton("rsp")) {
				GoToMemory(selected->regs.rsp);
			}
			ImGui::SameLine();
			if (ImGui::SmallButton("rbp")) {
				GoToMemory(selected->regs.rbp);
			}
			ImGui::SameLine();
			if (ImGui::SmallButton("rip")) {
				GoToMemory(selected->regs.rip);
			}
		}

		ImGui::SameLine(0.0f, 16.0f);
		ImGui::SetNextItemWidth(90.0f);
		const char* widths[] = {"8/row", "16/row", "32/row"};
		int width_index      = memory_bytes_per_row == 8 ? 0 : (memory_bytes_per_row == 32 ? 2 : 1);
		if (ImGui::Combo("##width", &width_index, widths, IM_ARRAYSIZE(widths))) {
			memory_bytes_per_row  = width_index == 0 ? 8 : (width_index == 2 ? 32 : 16);
			memory_snapshot_valid = false;
			memory_scroll_pending = true;
		}

		// Where we are, in module terms, so the address means something.
		ImGui::SameLine(0.0f, 16.0f);
		const auto location = Symbols::Describe(memory_selected);
		if (location.resolved) {
			ImGui::TextColored(Palette::Accent, "%s", Symbols::Format(memory_selected).c_str());
		} else {
			ImGui::TextColored(Palette::TextDim, "unknown region");
		}
	}

	// Capture the visible span at each halt, and keep the *previous* capture as the thing live
	// bytes are compared against — comparing against a snapshot taken at the current halt would
	// always match, so nothing would ever highlight. Same shape as the register basis.
	//
	// The guest is frozen while halted, so "differs from the previous halt" is exactly "the
	// guest wrote this since the last step".
	void RefreshMemorySnapshot() {
		const bool base_changed = memory_snapshot_base != memory_base;
		const bool halt_changed = memory_snapshot_halt != basis_rip;

		if (memory_snapshot_valid && !base_changed && !halt_changed) {
			return;
		}

		if (halt_changed && !base_changed && memory_snapshot_valid) {
			memory_compare         = memory_snapshot;
			memory_compare_present = memory_snapshot_present;
			memory_compare_valid   = true;
		} else {
			// Moved somewhere new: there is nothing meaningful to diff against yet.
			memory_compare_valid = false;
		}

		const auto rows = MEMORY_SPAN / memory_bytes_per_row;

		memory_snapshot.assign(MEMORY_SPAN, 0);
		memory_snapshot_present.assign(rows, 0);

		for (uint64_t row = 0; row < rows; row++) {
			const uint64_t address       = memory_base + row * memory_bytes_per_row;
			memory_snapshot_present[row] = static_cast<uint8_t>(
			    Session::ReadMemory(address, memory_snapshot.data() + row * memory_bytes_per_row,
			                        memory_bytes_per_row));
		}

		memory_snapshot_base  = memory_base;
		memory_snapshot_halt  = basis_rip;
		memory_snapshot_valid = true;
	}

	void DrawMemoryInspector() {
		PaneTitle("INSPECTOR");

		std::array<uint8_t, 8> raw {};
		const bool readable = Session::ReadMemory(memory_selected, raw.data(), raw.size());

		ImGui::TextColored(Palette::TextDim, "at");
		ImGui::SameLine();
		ImGui::TextUnformatted(Hex64(memory_selected).c_str());

		if (!readable) {
			ImGui::TextColored(Palette::Red, "unmapped");
			return;
		}

		// memcpy rather than reinterpret_cast: the buffer is a byte array, so casting it to a
		// wider type and dereferencing would be an unaligned access and undefined behaviour even
		// though x86 tolerates it.
		uint64_t u64 = 0;
		uint32_t u32 = 0;
		uint16_t u16 = 0;
		float    f32 = 0.0f;
		double   f64 = 0.0;
		std::memcpy(&u64, raw.data(), sizeof(u64));
		std::memcpy(&u32, raw.data(), sizeof(u32));
		std::memcpy(&u16, raw.data(), sizeof(u16));
		std::memcpy(&f32, raw.data(), sizeof(f32));
		std::memcpy(&f64, raw.data(), sizeof(f64));

		struct Row {
			const char* label;
			std::string value;
		};

		std::array<char, 64> scratch {};
		const auto           format = [&scratch](const char* fmt, auto value) {
			std::snprintf(scratch.data(), scratch.size(), fmt, value);
			return std::string(scratch.data());
		};

		const std::array<Row, 9> rows {{
		    {"i8", format("%d", static_cast<int>(static_cast<int8_t>(raw[0])))},
		    {"u8", format("%u", static_cast<unsigned>(raw[0]))},
		    {"i16", format("%d", static_cast<int>(static_cast<int16_t>(u16)))},
		    {"u16", format("%u", static_cast<unsigned>(u16))},
		    {"i32", format("%d", static_cast<int>(static_cast<int32_t>(u32)))},
		    {"u32", format("%u", static_cast<unsigned>(u32))},
		    {"i64", format("%lld", static_cast<long long>(static_cast<int64_t>(u64)))},
		    {"f32", format("%g", static_cast<double>(f32))},
		    {"f64", format("%g", f64)},
		}};

		if (ImGui::BeginTable("inspector", 2,
		                      ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit)) {
			ImGui::TableSetupColumn("##label", ImGuiTableColumnFlags_WidthFixed, 34.0f);
			ImGui::TableSetupColumn("##value", ImGuiTableColumnFlags_WidthStretch);

			for (const auto& row: rows) {
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				ImGui::TextColored(Palette::TextDim, "%s", row.label);
				ImGui::TableNextColumn();
				ImGui::TextUnformatted(row.value.c_str());
			}

			ImGui::EndTable();
		}

		ImGui::Dummy({0.0f, 4.0f});

		// If the qword here reads as an address that is itself mapped, it is probably a pointer,
		// which makes walking structures possible without leaving the panel.
		uint8_t    probe              = 0;
		const bool looks_like_pointer = u64 != 0 && Session::ReadMemory(u64, &probe, 1);

		ImGui::TextColored(Palette::TextDim, "ptr");
		ImGui::SameLine();
		ImGui::TextUnformatted(Hex64(u64).c_str());

		if (looks_like_pointer) {
			if (ImGui::Button("Follow pointer")) {
				GoToMemory(u64);
			}
			const auto target = Symbols::Describe(u64);
			if (target.resolved) {
				ImGui::TextColored(Palette::Accent, "%s", Symbols::Format(u64).c_str());
			}
		} else {
			ImGui::TextColored(Palette::TextDim, "not a mapped address");
		}

		ImGui::Dummy({0.0f, 6.0f});
		ImGui::TextColored(Palette::TextDim, "click a byte to edit it");
	}

	void DrawMemoryGrid() {
		const uint32_t per_row   = memory_bytes_per_row;
		const auto     row_count = static_cast<int>(MEMORY_SPAN / per_row);

		RefreshMemorySnapshot();

		const int columns = 2 + static_cast<int>(per_row);

		if (!ImGui::BeginTable("hex", columns,
		                       ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit |
		                           ImGuiTableFlags_NoPadOuterX)) {
			return;
		}

		ImGui::TableSetupColumn("##addr", ImGuiTableColumnFlags_WidthFixed, 132.0f);
		for (uint32_t i = 0; i < per_row; i++) {
			ImGui::TableSetupColumn("##b", ImGuiTableColumnFlags_WidthFixed,
			                        ImGui::CalcTextSize("00").x + 6.0f);
		}
		ImGui::TableSetupColumn("##ascii", ImGuiTableColumnFlags_WidthStretch);

		if (memory_scroll_pending) {
			const auto row = static_cast<float>((memory_selected - memory_base) / per_row);
			ImGui::SetScrollY(std::max(0.0f, (row - 6.0f) * ImGui::GetTextLineHeightWithSpacing()));
			memory_scroll_pending = false;
		}

		std::vector<uint8_t> row_data(per_row);

		ImGuiListClipper clipper;
		clipper.Begin(row_count);
		while (clipper.Step()) {
			for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
				const uint64_t row_address = memory_base + static_cast<uint64_t>(row) * per_row;
				const bool     readable =
				    Session::ReadMemory(row_address, row_data.data(), row_data.size());

				ImGui::TableNextRow();
				ImGui::PushID(row);

				ImGui::TableNextColumn();
				ImGui::TextColored(Palette::TextDim, "%s", Hex64(row_address).c_str());

				if (!readable) {
					ImGui::TableNextColumn();
					ImGui::TextColored(Palette::TextDim, "unmapped");
					ImGui::PopID();
					continue;
				}

				std::string ascii;
				for (uint32_t i = 0; i < per_row; i++) {
					ImGui::TableNextColumn();
					ImGui::PushID(static_cast<int>(i));

					const uint64_t address = row_address + i;
					const uint8_t  byte    = row_data[i];

					if (memory_edit_address == address) {
						ImGui::SetNextItemWidth(ImGui::CalcTextSize("00").x + 6.0f);
						if (memory_edit_focus) {
							ImGui::SetKeyboardFocusHere();
							memory_edit_focus = false;
							// Focus only lands after this frame, so the cancel-on-focus-loss
							// check below must not run yet or editing ends immediately.
							memory_edit_grace = 2;
						}
						if (ImGui::InputText("##edit", memory_edit_buffer.data(),
						                     memory_edit_buffer.size(),
						                     ImGuiInputTextFlags_CharsHexadecimal |
						                         ImGuiInputTextFlags_EnterReturnsTrue |
						                         ImGuiInputTextFlags_AutoSelectAll)) {
							unsigned value = 0;
							if (std::sscanf(memory_edit_buffer.data(), "%x", &value) == 1) {
								const auto written = static_cast<uint8_t>(value);
								Session::WriteMemory(address, &written, 1);
								memory_snapshot_valid = false;
							}
							// Roll onto the next byte so a run can be typed straight through.
							memory_edit_address = address + 1;
							memory_edit_focus   = true;
							memory_edit_buffer.fill('\0');
						} else if (memory_edit_grace > 0) {
							memory_edit_grace--;
						} else if (!ImGui::IsItemActive()) {
							memory_edit_address = 0; // focus lost: stop editing
						}
					} else {
						const bool changed =
						    memory_compare_valid &&
						    row < static_cast<int>(memory_compare_present.size()) &&
						    memory_compare_present[row] != 0 &&
						    memory_compare[static_cast<size_t>(row) * per_row + i] != byte;

						const bool is_selected = address == memory_selected;

						std::array<char, 4> cell {};
						std::snprintf(cell.data(), cell.size(), "%02x", byte);

						ImVec4 colour = Palette::Text;
						if (changed) {
							colour = Palette::Amber;
						} else if (byte == 0) {
							colour = Palette::TextDim;
						}

						ImGui::PushStyleColor(ImGuiCol_Text, colour);
						if (ImGui::Selectable(cell.data(), is_selected,
						                      ImGuiSelectableFlags_AllowDoubleClick)) {
							memory_selected = address;
							if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
								memory_edit_address = address;
								memory_edit_focus   = true;
								memory_edit_buffer.fill('\0');
							}
						}
						ImGui::PopStyleColor();
					}

					ImGui::PopID();

					ascii += (byte >= 0x20 && byte < 0x7f) ? static_cast<char>(byte) : '.';
				}

				ImGui::TableNextColumn();
				ImGui::TextColored(Palette::TextDim, "%s", ascii.c_str());

				ImGui::PopID();
			}
		}

		ImGui::EndTable();
	}

	void DrawMemoryTab(const StoppedThread* selected) {
		// Open on something worth looking at rather than address zero: the halted thread's stack
		// if there is one, otherwise the first loaded module.
		if (memory_selected == 0 && memory_base == 0) {
			if (selected != nullptr && selected->regs.valid) {
				GoToMemory(selected->regs.rsp, false);
			} else if (const auto modules = Symbols::Modules(); !modules.empty()) {
				GoToMemory(modules.front().base_vaddr, false);
			}
		}

		if (memory_selected == 0) {
			ImGui::TextColored(Palette::TextDim,
			                   "nothing to show yet - halt the guest, or type an address, symbol "
			                   "or module+offset above once a game is running");
			DrawMemoryToolbar(selected);
			return;
		}

		DrawMemoryToolbar(selected);
		ImGui::Dummy({0.0f, 2.0f});

		const float inspector_width = 230.0f;
		const float grid_width =
		    std::max(240.0f, ImGui::GetContentRegionAvail().x - inspector_width - 8.0f);

		ImGui::BeginChild("##hexgrid", {grid_width, 0.0f});
		DrawMemoryGrid();
		ImGui::EndChild();

		ImGui::SameLine();

		ImGui::BeginChild("##inspector", {0.0f, 0.0f}, ImGuiChildFlags_Borders);
		DrawMemoryInspector();
		ImGui::EndChild();
	}

	void DrawFrameTab() {
		const auto stats = Gfx::GetStats();

		if (ImGui::BeginTable("gpustats", 6,
		                      ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
			const auto cell = [](const char* label, const std::string& value) {
				ImGui::TableNextColumn();
				ImGui::TextColored(Palette::TextDim, "%s", label);
				ImGui::TextUnformatted(value.c_str());
			};

			ImGui::TableNextRow();
			cell("frame", std::to_string(stats.frame));
			cell("draws (last frame)", std::to_string(stats.draws_last_frame));
			cell("dispatches (last frame)", std::to_string(stats.dispatches_last_frame));
			cell("draws total", std::to_string(stats.total_draws));
			cell("dispatches total", std::to_string(stats.total_dispatches));
			cell("shaders", std::to_string(stats.shader_count));

			ImGui::EndTable();
		}

		if (stats.truncated) {
			ImGui::TextColored(Palette::Amber,
			                   "frame draw list truncated - only the first entries are kept");
		}

		ImGui::Dummy({0.0f, 2.0f});

		const auto draws = Gfx::LastFrame();
		if (draws.empty()) {
			ImGui::TextColored(Palette::TextDim,
			                   "no draws captured yet - the list fills once the game submits a "
			                   "frame");
			return;
		}

		if (ImGui::BeginTable("draws", 6,
		                      ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
		                          ImGuiTableFlags_SizingStretchProp)) {
			ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 44.0f);
			ImGui::TableSetupColumn("kind", ImGuiTableColumnFlags_WidthFixed, 96.0f);
			ImGui::TableSetupColumn("count", ImGuiTableColumnFlags_WidthFixed, 80.0f);
			ImGui::TableSetupColumn("inst", ImGuiTableColumnFlags_WidthFixed, 52.0f);
			ImGui::TableSetupColumn("vertex / compute shader");
			ImGui::TableSetupColumn("pixel shader");
			ImGui::TableSetupScrollFreeze(0, 1);
			ImGui::TableHeadersRow();

			for (const auto& draw: draws) {
				ImGui::TableNextRow();

				ImGui::TableNextColumn();
				ImGui::TextColored(Palette::TextDim, "%u", draw.index);

				ImGui::TableNextColumn();
				ImGui::TextUnformatted(Gfx::KindName(draw.kind));

				ImGui::TableNextColumn();
				if (draw.kind == Gfx::DrawKind::Dispatch) {
					ImGui::Text("%ux%ux%u", draw.groups[0], draw.groups[1], draw.groups[2]);
				} else {
					ImGui::Text("%u", draw.count);
				}

				ImGui::TableNextColumn();
				if (draw.kind == Gfx::DrawKind::Dispatch) {
					ImGui::TextColored(Palette::TextDim, "-");
				} else {
					ImGui::Text("%u", draw.instances);
				}

				ImGui::TableNextColumn();
				const uint64_t first =
				    draw.kind == Gfx::DrawKind::Dispatch ? draw.cs_address : draw.vs_address;
				ImGui::TextColored(first != 0 ? Palette::Text : Palette::TextDim, "%s",
				                   first != 0 ? Hex64(first).c_str() : "-");

				ImGui::TableNextColumn();
				ImGui::TextColored(draw.ps_address != 0 ? Palette::Text : Palette::TextDim, "%s",
				                   draw.ps_address != 0 ? Hex64(draw.ps_address).c_str() : "-");
			}

			ImGui::EndTable();
		}
	}

	void DrawShadersTab() {
		const auto shaders = Gfx::Shaders();

		ImGui::SetNextItemWidth(220.0f);
		ImGui::InputTextWithHint("##shaderfilter", "filter by hash or stage", shader_filter.data(),
		                         shader_filter.size());
		ImGui::SameLine();
		ImGui::TextColored(Palette::TextDim, "%zu shaders", shaders.size());

		if (shaders.empty()) {
			ImGui::Dummy({0.0f, 4.0f});
			ImGui::TextColored(Palette::TextDim,
			                   "none recompiled yet - they appear as the game draws");
			return;
		}

		ImGui::Dummy({0.0f, 2.0f});

		const std::string filter = shader_filter.data();

		const float list_width = 340.0f;

		ImGui::BeginChild("##shaderlist", {list_width, 0.0f});
		if (ImGui::BeginTable("shaders", 3,
		                      ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
		                          ImGuiTableFlags_SizingStretchProp)) {
			ImGui::TableSetupColumn("stage", ImGuiTableColumnFlags_WidthFixed, 62.0f);
			ImGui::TableSetupColumn("hash");
			ImGui::TableSetupColumn("bytes", ImGuiTableColumnFlags_WidthFixed, 60.0f);
			ImGui::TableSetupScrollFreeze(0, 1);
			ImGui::TableHeadersRow();

			for (const auto& shader: shaders) {
				const auto hash_text = Hex64(shader.hash);
				const auto stage     = std::string(Gfx::StageName(shader.stage));

				if (!filter.empty() && hash_text.find(filter) == std::string::npos &&
				    stage.find(filter) == std::string::npos) {
					continue;
				}

				ImGui::TableNextRow();
				ImGui::PushID(static_cast<int>(shader.sequence));

				ImGui::TableNextColumn();
				if (ImGui::Selectable(stage.c_str(), selected_shader == shader.hash,
				                      ImGuiSelectableFlags_SpanAllColumns)) {
					selected_shader   = shader.hash;
					shader_code_valid = false;
				}

				ImGui::TableNextColumn();
				ImGui::TextColored(Palette::TextDim, "%s", hash_text.c_str());

				ImGui::TableNextColumn();
				ImGui::TextColored(Palette::TextDim, "%u", shader.gcn_bytes);

				ImGui::PopID();
			}

			ImGui::EndTable();
		}
		ImGui::EndChild();

		ImGui::SameLine();

		ImGui::BeginChild("##shadercode", {0.0f, 0.0f}, ImGuiChildFlags_Borders);
		DrawShaderCode();
		ImGui::EndChild();
	}

	void DrawShaderCode() {
		if (selected_shader == 0) {
			ImGui::TextColored(Palette::TextDim, "select a shader");
			return;
		}

		// Disassembling SPIR-V is not free, so fetch once per selection rather than per frame.
		if (!shader_code_valid) {
			shader_code_valid = Gfx::GetShaderCode(selected_shader, shader_code);
			if (!shader_code_valid) {
				ImGui::TextColored(Palette::Red, "shader is no longer available");
				return;
			}
		}

		ImGui::TextColored(Palette::Accent, "%s", Hex64(selected_shader).c_str());
		ImGui::SameLine();
		if (ImGui::SmallButton("Save to _Shaders")) {
			std::string path;
			if (Gfx::DumpShader(selected_shader, path)) {
				shader_saved_to = path;
			}
		}
		if (!shader_saved_to.empty()) {
			ImGui::SameLine();
			ImGui::TextColored(Palette::Green, "saved to %s", shader_saved_to.c_str());
		}

		const auto view = [](const char* label, const std::string& text) {
			if (!ImGui::BeginTabItem(label)) {
				return;
			}
			if (text.empty()) {
				ImGui::TextColored(Palette::TextDim, "not captured for this shader");
			} else {
				ImGui::BeginChild("##text", {0.0f, 0.0f}, ImGuiChildFlags_None,
				                  ImGuiWindowFlags_HorizontalScrollbar);
				ImGui::TextUnformatted(text.c_str());
				ImGui::EndChild();
			}
			ImGui::EndTabItem();
		};

		if (ImGui::BeginTabBar("##shadercodetabs")) {
			view("RDNA2", shader_code.isa);
			view("IR", shader_code.ir);
			view("SPIR-V", shader_code.spirv);
			ImGui::EndTabBar();
		}
	}

	void DrawGraphicsTab() {
		if (!Gfx::IsCapturing()) {
			ImGui::TextColored(Palette::TextDim, "graphics capture is off");
			return;
		}

		// force_tab is only set by the layout test, so both tabs' contents get walked.
		const auto tab_flags = [this](int index) {
			return force_tab == index ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
		};

		if (ImGui::BeginTabBar("##gfxtabs")) {
			if (ImGui::BeginTabItem("Frame", nullptr, tab_flags(0))) {
				DrawFrameTab();
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Shaders", nullptr, tab_flags(1))) {
				DrawShadersTab();
				ImGui::EndTabItem();
			}
			ImGui::EndTabBar();
		}
	}

	void DrawModulesTab() {
		if (ImGui::BeginTable("modules", 3,
		                      ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
		                          ImGuiTableFlags_SizingStretchProp)) {
			ImGui::TableSetupColumn("module");
			ImGui::TableSetupColumn("base", ImGuiTableColumnFlags_WidthFixed, 150.0f);
			ImGui::TableSetupColumn("size", ImGuiTableColumnFlags_WidthFixed, 90.0f);
			ImGui::TableSetupScrollFreeze(0, 1);
			ImGui::TableHeadersRow();

			bool any = false;
			for (const auto& module: Symbols::Modules()) {
				any = true;
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				if (ImGui::Selectable(module.name.c_str(), false,
				                      ImGuiSelectableFlags_SpanAllColumns)) {
					GoTo(module.base_vaddr);
				}
				ImGui::TableNextColumn();
				ImGui::TextColored(Palette::TextDim, "%s", Hex64(module.base_vaddr).c_str());
				ImGui::TableNextColumn();
				ImGui::TextColored(Palette::TextDim, "%llu KiB",
				                   static_cast<unsigned long long>(module.size / 1024));
			}

			ImGui::EndTable();

			if (!any) {
				ImGui::TextColored(Palette::TextDim, "no modules loaded yet");
			}
		}
	}

	void DrawSymbolsTab() {
		ImGui::SetNextItemWidth(260.0f);
		ImGui::InputTextWithHint("##filter", "filter symbols", symbol_filter.data(),
		                         symbol_filter.size());

		ImGui::Dummy({0.0f, 2.0f});

		if (ImGui::BeginTable("symbols", 3,
		                      ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
		                          ImGuiTableFlags_SizingStretchProp)) {
			ImGui::TableSetupColumn("symbol");
			ImGui::TableSetupColumn("module", ImGuiTableColumnFlags_WidthFixed, 130.0f);
			ImGui::TableSetupColumn("address", ImGuiTableColumnFlags_WidthFixed, 150.0f);
			ImGui::TableSetupScrollFreeze(0, 1);
			ImGui::TableHeadersRow();

			for (const auto& match: Symbols::Search(symbol_filter.data(), 400)) {
				ImGui::TableNextRow();
				ImGui::PushID(static_cast<int>(match.address));

				ImGui::TableNextColumn();
				if (ImGui::Selectable(match.name.c_str(), false,
				                      ImGuiSelectableFlags_SpanAllColumns)) {
					GoTo(match.address);
				}
				ImGui::TableNextColumn();
				ImGui::TextColored(Palette::TextDim, "%s", match.module.c_str());
				ImGui::TableNextColumn();
				ImGui::TextColored(Palette::TextDim, "%s", Hex64(match.address).c_str());

				ImGui::PopID();
			}

			ImGui::EndTable();
		}
	}

	// Default placement for a panel, expressed in fractions of the work area so it lands
	// somewhere sensible on any window size. Only applied the first time, so anything the user
	// drags stays where they put it.
	void PlaceWindow(float x, float y, float width, float height) {
		const auto* viewport = ImGui::GetMainViewport();

		constexpr float MARGIN   = 12.0f;
		const float     top      = viewport->WorkPos.y + control_bar_height + MARGIN;
		const float     usable_w = viewport->WorkSize.x - MARGIN * 2.0f;
		const float     usable_h = viewport->WorkSize.y - control_bar_height - MARGIN * 2.0f;

		ImGui::SetNextWindowPos({viewport->WorkPos.x + MARGIN + usable_w * x, top + usable_h * y},
		                        ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize({usable_w * width, usable_h * height}, ImGuiCond_FirstUseEver);
	}

	// Execution: threads and call stack on the left, disassembly in the middle, registers on the
	// right, breakpoints along the bottom. These belong together because stepping is driven by
	// looking at all of them at once.
	void DrawDebuggerWindow(const std::vector<StoppedThread>& stopped,
	                        const StoppedThread*              selected) {
		PlaceWindow(0.0f, 0.0f, 0.63f, 0.74f);

		if (!ImGui::Begin("Debugger", &show_debugger)) {
			ImGui::End();
			return;
		}

		// A fatal report is the most important thing on screen when it happens, so it goes above
		// everything rather than into a pane that might be scrolled away.
		if (selected != nullptr && selected->reason == StopReason::Fatal &&
		    !selected->message.empty()) {
			ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4 {0.22f, 0.07f, 0.07f, 1.0f});
			if (ImGui::BeginChild("##fatal", {0.0f, ImGui::GetTextLineHeightWithSpacing() * 5.0f},
			                      ImGuiChildFlags_Borders, ImGuiWindowFlags_HorizontalScrollbar)) {
				ImGui::TextColored(Palette::Red, "FATAL ERROR - the emulator will exit when you "
				                                 "continue");
				ImGui::Separator();
				ImGui::TextUnformatted(selected->message.c_str());
			}
			ImGui::EndChild();
			ImGui::PopStyleColor();
			ImGui::Dummy({0.0f, 2.0f});
		}

		const float total_height = ImGui::GetContentRegionAvail().y;
		const float total_width  = ImGui::GetContentRegionAvail().x;

		constexpr float MIN_CENTRE  = 260.0f;
		constexpr float MIN_SIDE    = 150.0f;
		const float     side_budget = total_width - MIN_CENTRE - 2.0f * SPLITTER_THICKNESS;

		bottom_height = std::clamp(bottom_height, 70.0f, std::max(70.0f, total_height - 180.0f));
		left_width =
		    std::clamp(left_width, MIN_SIDE, std::max(MIN_SIDE, side_budget - right_width));
		right_width =
		    std::clamp(right_width, MIN_SIDE, std::max(MIN_SIDE, side_budget - left_width));

		const float main_height =
		    std::max(120.0f, total_height - bottom_height - SPLITTER_THICKNESS - 4.0f);
		const float centre_width = std::max(MIN_CENTRE, total_width - left_width - right_width -
		                                                    2.0f * SPLITTER_THICKNESS);

		ImGui::BeginChild("##left", {left_width, main_height});
		{
			const float column_height = ImGui::GetContentRegionAvail().y;
			const float column_width  = ImGui::GetContentRegionAvail().x;

			threads_height =
			    std::clamp(threads_height, 70.0f, std::max(70.0f, column_height - 100.0f));

			DrawThreadsPane(stopped, threads_height);
			HorizontalSplitter("##split_threads", column_width, &threads_height, 70.0f,
			                   std::max(70.0f, column_height - 100.0f), true);
			DrawCallStackPane(selected);
		}
		ImGui::EndChild();

		VerticalSplitter("##split_left", main_height, &left_width, MIN_SIDE, 640.0f, true);

		DrawDisassemblyPane(selected, centre_width, main_height);

		VerticalSplitter("##split_right", main_height, &right_width, MIN_SIDE, 640.0f, false);

		DrawRegistersPane(selected, main_height);

		HorizontalSplitter("##split_bottom", total_width, &bottom_height, 70.0f,
		                   std::max(70.0f, total_height - 180.0f), false);

		if (ImGui::BeginChild("##breakpoints", {0.0f, 0.0f}, ImGuiChildFlags_Borders)) {
			PaneTitle("BREAKPOINTS");
			DrawBreakpointsTab();
		}
		ImGui::EndChild();

		ImGui::End();
	}

	void DrawMemoryWindow(const StoppedThread* selected) {
		PlaceWindow(0.0f, 0.76f, 0.63f, 0.24f);

		if (!ImGui::Begin("Memory", &show_memory)) {
			ImGui::End();
			return;
		}

		DrawMemoryTab(selected);
		ImGui::End();
	}

	void DrawGraphicsWindow() {
		PlaceWindow(0.65f, 0.0f, 0.35f, 0.56f);

		if (!ImGui::Begin("Graphics", &show_graphics)) {
			ImGui::End();
			return;
		}

		DrawGraphicsTab();
		ImGui::End();
	}

	void DrawLookupWindow() {
		PlaceWindow(0.65f, 0.58f, 0.35f, 0.42f);

		if (!ImGui::Begin("Modules & symbols", &show_lookup)) {
			ImGui::End();
			return;
		}

		// force_tab is only set by the layout test, so both tabs' contents get walked.
		const auto tab_flags = [this](int index) {
			return force_tab == index ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
		};

		if (ImGui::BeginTabBar("##lookuptabs")) {
			if (ImGui::BeginTabItem("Modules", nullptr, tab_flags(0))) {
				DrawModulesTab();
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Symbols", nullptr, tab_flags(1))) {
				DrawSymbolsTab();
				ImGui::EndTabItem();
			}
			ImGui::EndTabBar();
		}

		ImGui::End();
	}

	void HandleShortcuts() {
		if (!Session::IsPaused()) {
			return;
		}

		if (ImGui::IsKeyPressed(ImGuiKey_F5, false)) {
			Session::ResumeAll();
		} else if (ImGui::IsKeyPressed(ImGuiKey_F11, false)) {
			Session::Resume(selected_thread, Session::ResumeMode::StepInto);
		} else if (ImGui::IsKeyPressed(ImGuiKey_F10, false)) {
			Session::Resume(selected_thread, Session::ResumeMode::StepOver);
		}
	}

	// Keep a register snapshot from the previous halt so the register pane can highlight what
	// the last step changed. `compare_basis` stays fixed for as long as we sit at one address.
	void UpdateRegisterBasis(const StoppedThread* selected) {
		if (selected == nullptr || !selected->regs.valid) {
			return;
		}

		if (selected->unique_id != basis_thread) {
			compare_basis = selected->regs; // switching threads is not a step; show no diff
		} else if (selected->regs.rip != basis_rip) {
			compare_basis = basis_regs; // basis_regs still holds the previous halt
		}

		basis_regs   = selected->regs;
		basis_rip    = selected->regs.rip;
		basis_thread = selected->unique_id;
	}

	void Draw() {
		const auto stopped = Session::Stopped();

		// Default the selection to whichever thread halted first.
		if (!stopped.empty()) {
			const bool still_stopped =
			    std::any_of(stopped.begin(), stopped.end(), [this](const StoppedThread& thread) {
				    return thread.unique_id == selected_thread;
			    });
			if (!still_stopped) {
				selected_thread = stopped.front().unique_id;
				FollowHaltedThread(stopped.front().address);
			}
		}

		const StoppedThread* selected = nullptr;
		for (const auto& thread: stopped) {
			if (thread.unique_id == selected_thread) {
				selected = &thread;
				break;
			}
		}

		UpdateRegisterBasis(selected);
		HandleShortcuts();

		// The control bar goes first so its height is known before the panels are placed, and is
		// pinned to the front from inside DrawControlBar.
		DrawControlBar(selected);

		if (show_debugger) {
			DrawDebuggerWindow(stopped, selected);
		}
		if (show_memory) {
			DrawMemoryWindow(selected);
		}
		if (show_graphics) {
			DrawGraphicsWindow();
		}
		if (show_lookup) {
			DrawLookupWindow();
		}
	}

	bool PrepareFrame(vk::Extent2D frame_extent, vk::Format format, uint32_t image_count) {
		if (!g_visible.load(std::memory_order_acquire)) {
			return false;
		}

		EnsureVulkan(format, image_count);
		DrainInput();

		auto& io       = ImGui::GetIO();
		io.DisplaySize = {static_cast<float>(frame_extent.width),
		                  static_cast<float>(frame_extent.height)};

		// The bundled font is a 13px bitmap; scale it up on tall displays so the panes stay
		// legible at 1440p and above.
		io.FontGlobalScale = frame_extent.height >= 1800   ? 2.0f
		                     : frame_extent.height >= 1200 ? 1.35f
		                                                   : 1.0f;

		const auto now = std::chrono::steady_clock::now();
		io.DeltaTime   = last_frame == std::chrono::steady_clock::time_point {}
		                     ? 1.0f / 60.0f
		                     : std::clamp(std::chrono::duration<float>(now - last_frame).count(),
		                                  1.0f / 1000.0f, 0.1f);
		last_frame     = now;

		ImGui_ImplVulkan_NewFrame();
		ImGui::NewFrame();
		Draw();
		ImGui::Render();

		// Heartbeat for the fatal-halt watchdog: reaching here means presentation is still
		// running, so a halt on some other thread is safe to hold indefinitely.
		Session::NotifyOverlayDrawn();

		extent = frame_extent;
		return true;
	}

	void Record(vk::CommandBuffer command, vk::ImageView target) {
		vk::RenderingAttachmentInfo color {};
		color.sType       = vk::StructureType::eRenderingAttachmentInfo;
		color.imageView   = target;
		color.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		color.loadOp      = vk::AttachmentLoadOp::eLoad;
		color.storeOp     = vk::AttachmentStoreOp::eStore;

		vk::RenderingInfo rendering {};
		rendering.sType                = vk::StructureType::eRenderingInfo;
		rendering.renderArea.extent    = extent;
		rendering.layerCount           = 1;
		rendering.colorAttachmentCount = 1;
		rendering.pColorAttachments    = &color;

		command.beginRendering(rendering);
		{
			Common::LockGuard queue_lock(graphics.queue_mutex);
			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(),
			                                static_cast<VkCommandBuffer>(command));
		}
		command.endRendering();
	}

	void ReleaseVulkan() {
		if (!vulkan_initialized) {
			return;
		}
		ImGui::SetCurrentContext(imgui_context);
		ImGui_ImplVulkan_Shutdown();
		vulkan_initialized = false;
	}

	GraphicContext& graphics;
	ImGuiContext*   imgui_context      = nullptr;
	bool            vulkan_initialized = false;
	int             selected_thread    = 0;
	int             force_tab          = -1;

	// Which panel windows are open, and how much room the pinned control bar takes.
	bool     show_debugger      = true;
	bool     show_memory        = true;
	bool     show_graphics      = true;
	bool     show_lookup        = false;
	float    control_bar_height = 0.0f;
	uint64_t disasm_address     = 0;

	// Memory view: the window currently mapped into the grid, the byte under the cursor, the
	// byte being typed into, a snapshot for change highlighting, and a small jump history.
	uint64_t              memory_base           = 0;
	uint64_t              memory_selected       = 0;
	uint32_t              memory_bytes_per_row  = 16;
	bool                  memory_scroll_pending = false;
	uint64_t              memory_edit_address   = 0;
	bool                  memory_edit_focus     = false;
	int                   memory_edit_grace     = 0;
	std::array<char, 4>   memory_edit_buffer {};
	std::vector<uint8_t>  memory_snapshot;
	std::vector<uint8_t>  memory_snapshot_present;
	std::vector<uint8_t>  memory_compare;
	std::vector<uint8_t>  memory_compare_present;
	uint64_t              memory_snapshot_base  = 0;
	uint64_t              memory_snapshot_halt  = 0;
	bool                  memory_snapshot_valid = false;
	bool                  memory_compare_valid  = false;
	std::vector<uint64_t> memory_history;
	bool                  follow_rip = true;
	// Address range the current listing covers, so the next frame can tell whether rip is still
	// inside it; the halt address already followed; and one-shot scroll requests.
	uint64_t              listing_first     = 0;
	uint64_t              listing_end       = 0;
	uint64_t              followed_address  = 0;
	bool                  scroll_to_current = false;
	bool                  scroll_to_top     = false;
	float                 left_width        = 300.0f;
	float                 right_width       = 210.0f;
	float                 bottom_height     = 220.0f;
	float                 threads_height    = 200.0f;
	Registers             compare_basis {};
	Registers             basis_regs {};
	uint64_t              basis_rip    = 0;
	int                   basis_thread = -1;
	std::array<char, 128> goto_buffer {};
	std::array<char, 128> breakpoint_buffer {};
	std::array<char, 128> memory_buffer {};
	std::array<char, 128> symbol_filter {};

	// Graphics tab.
	std::array<char, 128> shader_filter {};
	uint64_t              selected_shader   = 0;
	bool                  shader_code_valid = false;
	Gfx::ShaderCode       shader_code;
	std::string           shader_saved_to;

	vk::Extent2D                          extent {};
	std::chrono::steady_clock::time_point last_frame;
};

DebuggerOverlay::DebuggerOverlay(GraphicContext& graphics)
    : m_impl(std::make_unique<Impl>(graphics)) {}

DebuggerOverlay::~DebuggerOverlay() = default;

bool DebuggerOverlay::PrepareFrame(vk::Extent2D extent, vk::Format format, uint32_t image_count) {
	return m_impl->PrepareFrame(extent, format, image_count);
}

void DebuggerOverlay::Record(vk::CommandBuffer command, vk::ImageView target) {
	m_impl->Record(command, target);
}

void DebuggerOverlay::ReleaseVulkan() {
	m_impl->ReleaseVulkan();
}

void DebuggerOverlay::DrawPanelsForTesting(int force_tab) {
	// Draw() only touches ImGui and the debug session. The graphics context is required by the
	// constructor but never dereferenced on this path, and the Impl keeps its own ImGui context
	// null so the caller's context is the one that gets drawn into (and is not destroyed here).
	static Libs::Graphics::GraphicContext headless {};
	static Impl                           impl(headless);

	// Every window open, so the walk covers all of them rather than whatever happens to be up.
	impl.show_debugger = true;
	impl.show_memory   = true;
	impl.show_graphics = true;
	impl.show_lookup   = true;
	impl.force_tab     = force_tab;
	impl.Draw();
}

uint64_t DebuggerOverlay::ScrollRequestCountForTesting() {
	return g_scroll_requests;
}

} // namespace Debugger::Ui
