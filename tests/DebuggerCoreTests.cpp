// Regression tests for the debugger's headless core: the host-exception handler chain, the
// fault-free memory accessors, breakpoint arm/disarm with the read overlay, and instruction
// decoding. None of these need a window, a GPU, or a loaded guest.

#include "common/emulatorConfig.h"
#include "common/hostException.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "debugger/core/session.h"
#include "debugger/debugger.h"
#include "debugger/remote/server.h"
#include "debugger/target/graphics.h"
#include "debugger/target/io.h"
#include "debugger/ui/overlay.h"
#include "imgui.h"
#include "imgui_impl_null.h"
// Internal, but it is how ImGui's own test engine observes recoverable layout errors. In a
// Release build IM_ASSERT compiles away, so the error callback is the only way to see them.
#include "imgui_internal.h"
#include "debugger/symbols/symbols.h"
#include "debugger/target/memory.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <thread>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#endif

namespace {

using Common::HostException::ExceptionInfo;
using Common::HostException::ExceptionType;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "DebuggerCoreTests: failed: %s\n", message);
		std::abort();
	}
}

// ---- Handler chain --------------------------------------------------------------------------

std::vector<int> g_call_order;
int              g_trap_count      = 0;
bool             g_low_claims      = false;
constexpr int    MAX_TRAP_ATTEMPTS = 8;

// Guards against the whole suite hanging if resuming from a breakpoint does not make forward
// progress: the trap would otherwise re-fire forever.
void GuardAgainstTrapLoop() {
	if (++g_trap_count > MAX_TRAP_ATTEMPTS) {
		std::fprintf(stderr, "DebuggerCoreTests: failed: breakpoint trap did not resume\n");
		std::abort();
	}
}

bool g_rip_pointed_at_int3 = false;

// Step past the one-byte int3 the test just executed.
//
// ExceptionInfo::rip is contracted to be the address of the trap instruction on every platform
// (Windows reports it that way natively, POSIX does not and hostException normalises it), so
// resuming means moving one byte forward. The flag records whether that contract actually held,
// which is checked outside the handler — getting this wrong silently broke every breakpoint on
// Windows once already.
void ResumePastBreakpoint(const ExceptionInfo& info) {
	uint8_t at_rip = 0;
	g_rip_pointed_at_int3 =
	    Debugger::Target::SafeRead(info.rip, &at_rip, 1) && at_rip == 0xCC;

	Common::HostException::SetInstructionPointer(info.native_context, info.rip + 1);
}

bool LowPriorityHandler(const ExceptionInfo& info) {
	if (info.type != ExceptionType::Breakpoint) {
		return false;
	}
	GuardAgainstTrapLoop();
	g_call_order.push_back(0);
	if (!g_low_claims) {
		return false;
	}
	ResumePastBreakpoint(info);
	return true;
}

bool HighPriorityHandler(const ExceptionInfo& info) {
	if (info.type != ExceptionType::Breakpoint) {
		return false;
	}
	GuardAgainstTrapLoop();
	g_call_order.push_back(1);
	ResumePastBreakpoint(info);
	return true;
}

void RaiseBreakpoint() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	__debugbreak();
#else
	__asm__ volatile("int3");
#endif
}

void TestHandlerChainOrdering() {
	// Registered out of priority order on purpose: dispatch must sort them.
	Check(Common::HostException::AddHandler(LowPriorityHandler, 100), "AddHandler(low) failed");
	Check(Common::HostException::AddHandler(HighPriorityHandler, 0), "AddHandler(high) failed");

	// Re-registering the same handler is a no-op, not a second chain entry.
	Check(Common::HostException::AddHandler(LowPriorityHandler, 100),
	      "duplicate AddHandler should succeed");

	g_call_order.clear();
	g_trap_count          = 0;
	g_rip_pointed_at_int3 = false;
	RaiseBreakpoint();

	Check(g_call_order.size() == 1, "priority 0 handler did not claim the trap alone");
	Check(g_call_order[0] == 1, "handlers ran out of priority order");
	Check(g_rip_pointed_at_int3,
	      "ExceptionInfo::rip is not normalised to the breakpoint address");

	// With the first handler removed, the fault must fall through to the next one.
	Check(Common::HostException::RemoveHandler(HighPriorityHandler), "RemoveHandler failed");
	Check(!Common::HostException::RemoveHandler(HighPriorityHandler),
	      "removing an absent handler should fail");

	g_call_order.clear();
	g_trap_count = 0;
	g_low_claims = true;
	RaiseBreakpoint();

	Check(g_call_order.size() == 1, "fall-through handler did not run");
	Check(g_call_order[0] == 0, "wrong handler ran after removal");

	Check(Common::HostException::RemoveHandler(LowPriorityHandler), "cleanup RemoveHandler failed");
}

// Distinct bodies on purpose: identical captureless lambdas get folded to a single function by
// the linker's identical-code folding, which would make them look like one repeated
// registration rather than MAX_HANDLERS + 1 different ones.
volatile int g_dummy_sink = 0;

template <int N>
bool DummyHandler(const ExceptionInfo&) {
	g_dummy_sink += N;
	return false;
}

void TestHandlerCapacity() {
	// The table is fixed-size; registration past the limit must fail rather than overflow.
	constexpr uint32_t limit = Common::HostException::MAX_HANDLERS;
	static_assert(limit == 8, "update the handler list below to match MAX_HANDLERS");

	const std::array<Common::HostException::Handler, limit + 1> handlers {
	    DummyHandler<1>, DummyHandler<2>, DummyHandler<3>, DummyHandler<4>, DummyHandler<5>,
	    DummyHandler<6>, DummyHandler<7>, DummyHandler<8>, DummyHandler<9>,
	};

	uint32_t added = 0;
	for (uint32_t i = 0; i < limit + 1; i++) {
		if (Common::HostException::AddHandler(handlers[i], static_cast<int32_t>(i))) {
			added++;
		}
	}

	Check(added == limit, "handler table accepted more than MAX_HANDLERS entries");

	// Registering a handler twice is idempotent, not a second slot.
	Check(Common::HostException::AddHandler(handlers[0], 0), "duplicate registration failed");

	for (uint32_t i = 0; i < limit + 1; i++) {
		Common::HostException::RemoveHandler(handlers[i]);
	}

	// Once emptied, the table has room again.
	Check(Common::HostException::AddHandler(handlers[0], 0), "table did not free its slots");
	Check(Common::HostException::RemoveHandler(handlers[0]), "cleanup RemoveHandler failed");
}

// ---- Memory ---------------------------------------------------------------------------------

volatile uint64_t g_probe_target = 0x0123456789abcdefULL;

void TestSafeMemoryAccess() {
	uint64_t read_back = 0;
	Check(Debugger::Target::SafeRead(reinterpret_cast<uint64_t>(&g_probe_target), &read_back,
	                                 sizeof(read_back)),
	      "SafeRead of a mapped variable failed");
	Check(read_back == 0x0123456789abcdefULL, "SafeRead returned the wrong bytes");

	const uint64_t written = 0xfeedfacecafebeefULL;
	Check(Debugger::Target::SafeWrite(reinterpret_cast<uint64_t>(&g_probe_target), &written,
	                                  sizeof(written)),
	      "SafeWrite to a mapped variable failed");
	Check(g_probe_target == 0xfeedfacecafebeefULL, "SafeWrite did not take effect");

	// An unmapped address must be reported, not faulted on. The debugger's memory panel scrolls
	// through arbitrary addresses, so this is the property that keeps it from killing the
	// process.
	uint8_t scratch = 0;
	Check(!Debugger::Target::SafeRead(0x10, &scratch, 1), "SafeRead accepted a null-page address");
	Check(!Debugger::Target::SafeRead(0x7ff0'0000'0000'0000ULL, &scratch, 1),
	      "SafeRead accepted a non-canonical address");
	Check(!Debugger::Target::SafeRead(0, &scratch, 1), "SafeRead accepted address 0");
	Check(!Debugger::Target::SafeWrite(0x10, &scratch, 1),
	      "SafeWrite accepted a null-page address");
}

// ---- Breakpoints ----------------------------------------------------------------------------

// A function whose first byte the tests patch with a breakpoint. It must not be inlined or
// folded: the halted-thread test calls it for real and needs the call to land on exactly the
// address the breakpoint was planted at.
#if defined(_MSC_VER)
#define KYTY_TEST_NOINLINE __declspec(noinline)
#else
#define KYTY_TEST_NOINLINE __attribute__((noinline))
#endif

// NOLINTNEXTLINE(readability-identifier-naming)
extern "C" KYTY_TEST_NOINLINE void KytyDebuggerTestTarget() {
	// Deliberately non-trivial, so the first instruction is longer than one byte and the body is
	// distinctive enough that identical-code folding cannot merge it with another function.
	g_probe_target = g_probe_target * 3 + 1;
}

// A caller/leaf pair, so stepping has a real call to step over and a real frame to step out of.
// NOLINTNEXTLINE(readability-identifier-naming)
extern "C" KYTY_TEST_NOINLINE void KytyDebuggerStepLeaf() {
	g_probe_target = g_probe_target + 7;
}

using TestTargetFn = void (*)();

// NOLINTNEXTLINE(readability-identifier-naming)
extern "C" KYTY_TEST_NOINLINE void KytyDebuggerStepCaller() {
	volatile TestTargetFn leaf = &KytyDebuggerStepLeaf;
	leaf();
	g_probe_target = g_probe_target ^ 0x11;
}

// Called through a volatile pointer so the compiler cannot devirtualise, inline, or route the
// call past the address the breakpoint sits on.
void CallTestTargetIndirectly() {
	volatile TestTargetFn target = &KytyDebuggerTestTarget;
	target();
}

void CallStepCallerIndirectly() {
	volatile TestTargetFn target = &KytyDebuggerStepCaller;
	target();
}

void TestBreakpointArmAndOverlay() {
	Check(Debugger::Session::Initialize(), "Session::Initialize failed");

	const auto address = reinterpret_cast<uint64_t>(&KytyDebuggerTestTarget);

	std::array<uint8_t, 8> original {};
	Check(Debugger::Target::SafeRead(address, original.data(), original.size()),
	      "could not read the test target's code");

	const auto id = Debugger::Session::AddBreakpointAt(address);
	Check(id != 0, "AddBreakpointAt failed");

	// The raw byte is patched...
	uint8_t raw = 0;
	Check(Debugger::Target::SafeRead(address, &raw, 1), "raw read after arming failed");
	Check(raw == 0xCC, "arming did not plant an int3");

	// ...but the debugger's own read must show the instruction stream the guest would run,
	// otherwise the disassembly and hex panels display our patches.
	std::array<uint8_t, 8> overlaid {};
	Check(Debugger::Session::ReadMemory(address, overlaid.data(), overlaid.size()),
	      "Session::ReadMemory failed");
	Check(std::memcmp(overlaid.data(), original.data(), original.size()) == 0,
	      "breakpoint byte leaked through the read overlay");

	const auto listed = Debugger::Session::Breakpoints();
	Check(listed.size() == 1, "breakpoint list has the wrong size");
	Check(listed[0].armed && listed[0].address == address, "breakpoint recorded incorrectly");

	// Disabling must restore the original byte and re-enabling must plant it again.
	Check(Debugger::Session::SetBreakpointEnabled(id, false), "disabling the breakpoint failed");
	Check(Debugger::Target::SafeRead(address, &raw, 1), "raw read after disarming failed");
	Check(raw == original[0], "disarming did not restore the original byte");

	Check(Debugger::Session::SetBreakpointEnabled(id, true), "re-enabling the breakpoint failed");
	Check(Debugger::Target::SafeRead(address, &raw, 1), "raw read after re-arming failed");
	Check(raw == 0xCC, "re-arming did not plant an int3");

	Check(Debugger::Session::RemoveBreakpoint(id), "RemoveBreakpoint failed");
	Check(Debugger::Target::SafeRead(address, &raw, 1), "raw read after removal failed");
	Check(raw == original[0], "removal did not restore the original byte");
	Check(Debugger::Session::Breakpoints().empty(), "breakpoint list not emptied");

	Check(!Debugger::Session::RemoveBreakpoint(id), "removing an absent breakpoint should fail");
	Check(Debugger::Session::AddBreakpointAt(0) == 0, "a breakpoint at address 0 should fail");

	Debugger::Session::Shutdown();
}

// The memory editor writes through Session::WriteMemory, which has to cope with the user
// editing a byte that currently holds a breakpoint patch.
void TestWriteUnderBreakpoint() {
	Check(Debugger::Session::Initialize(), "Session::Initialize failed");

	const auto address = reinterpret_cast<uint64_t>(&KytyDebuggerTestTarget);

	std::array<uint8_t, 1> original {};
	Check(Debugger::Target::SafeRead(address, original.data(), 1), "could not read the target");

	const auto id = Debugger::Session::AddBreakpointAt(address);
	Check(id != 0, "could not arm the breakpoint");

	const uint8_t edited = static_cast<uint8_t>(original[0] ^ 0x5A);
	Check(Debugger::Session::WriteMemory(address, &edited, 1), "WriteMemory failed");

	// The patch must survive the edit, or the breakpoint quietly stops working.
	uint8_t raw = 0;
	Check(Debugger::Target::SafeRead(address, &raw, 1), "raw read failed");
	Check(raw == 0xCC, "editing a patched byte destroyed the breakpoint");

	// ...and the debugger's own view must show what was written, not the patch.
	uint8_t seen = 0;
	Check(Debugger::Session::ReadMemory(address, &seen, 1), "overlaid read failed");
	Check(seen == edited, "the edit is not visible through the read overlay");

	// Removing the breakpoint must leave the edit in place, not the pre-edit byte.
	Check(Debugger::Session::RemoveBreakpoint(id), "RemoveBreakpoint failed");
	Check(Debugger::Target::SafeRead(address, &raw, 1), "raw read after removal failed");
	Check(raw == edited, "lifting the breakpoint reverted the user's edit");

	// Put the function back exactly as it was; later tests execute it.
	Check(Debugger::Session::WriteMemory(address, original.data(), 1), "restore failed");
	Check(Debugger::Target::SafeRead(address, &raw, 1), "raw read after restore failed");
	Check(raw == original[0], "the target was not restored");

	Debugger::Session::Shutdown();
}

// ---- Disassembly ------------------------------------------------------------------------------

void TestDisassembly() {
	const auto address = reinterpret_cast<uint64_t>(&KytyDebuggerTestTarget);

	const auto instructions = Debugger::Disasm::Decode(address, 4);
	Check(!instructions.empty(), "Decode returned nothing for a known function");
	Check(instructions[0].address == address, "first decoded instruction has the wrong address");
	Check(instructions[0].length > 0, "first decoded instruction has zero length");
	Check(!instructions[0].text.empty(), "first decoded instruction has no text");

	// Addresses must advance by exactly the reported lengths.
	for (size_t i = 1; i < instructions.size(); i++) {
		Check(instructions[i].address ==
		          instructions[i - 1].address + instructions[i - 1].length,
		      "decoded instruction addresses are not contiguous");
	}

	Check(Debugger::Disasm::InstructionLength(address) == instructions[0].length,
	      "InstructionLength disagrees with Decode");

	// Unreadable memory must decode to nothing rather than crashing.
	Check(Debugger::Disasm::Decode(0x10, 4).empty(), "Decode accepted an unmapped address");
	Check(Debugger::Disasm::InstructionLength(0x10) == 0,
	      "InstructionLength accepted an unmapped address");

	uint32_t call_length = 0;
	Check(!Debugger::Disasm::IsCall(0x10, call_length), "IsCall accepted an unmapped address");
}

void TestSymbolResolution() {
	// With no guest module loaded, a hex literal must still resolve and a bogus name must not.
	uint64_t address = 0;
	Check(Debugger::Symbols::Resolve("0x1234", address) && address == 0x1234,
	      "hex address did not resolve");
	Check(Debugger::Symbols::Resolve("1234", address) && address == 0x1234,
	      "bare hex address did not resolve");
	Check(!Debugger::Symbols::Resolve("definitely_not_a_symbol_xyz", address),
	      "an unknown symbol resolved");
	Check(!Debugger::Symbols::Resolve("", address), "an empty location resolved");

	// Format never returns an empty string, so UI rows always have something to show.
	Check(!Debugger::Symbols::Format(0x1234).empty(), "Format returned an empty string");
}

// ---- Facade ------------------------------------------------------------------------------

void TestDebuggerStaysInertWithoutTheFlag() {
	Config::ConfigOptions options {};
	options.debugger_enabled = false;
	Config::Load(options);

	Debugger::Initialize();
	Check(!Debugger::IsEnabled(),
	      "the debugger armed itself even though --debugger was not passed");
	Check(!Debugger::IsOverlayVisible(), "the overlay is visible without --debugger");
	Debugger::Shutdown();
}

void TestDebuggerHonoursConfiguredBreakpoints() {
	char location[32] {};
	std::snprintf(location, sizeof(location), "0x%llx",
	              static_cast<unsigned long long>(reinterpret_cast<uint64_t>(
	                  &KytyDebuggerTestTarget)));

	Config::ConfigOptions options {};
	options.debugger_enabled     = true;
	options.debugger_ui_visible  = true;
	options.debugger_breakpoints = {location};
	Config::Load(options);

	Debugger::Initialize();
	Check(Debugger::IsEnabled(), "--debugger did not enable the debugger");
	Check(Debugger::IsOverlayVisible(), "--debugger-ui did not show the overlay");

	const auto breakpoints = Debugger::Session::Breakpoints();
	Check(breakpoints.size() == 1, "the configured breakpoint was not installed");
	Check(breakpoints[0].address == reinterpret_cast<uint64_t>(&KytyDebuggerTestTarget),
	      "the configured breakpoint landed at the wrong address");
	Check(breakpoints[0].armed, "the configured breakpoint was not armed");

	Debugger::Shutdown();
	Check(!Debugger::IsEnabled(), "Shutdown did not disable the debugger");
	Check(!Debugger::IsOverlayVisible(), "Shutdown left the overlay visible");

	// Shutdown must lift every patch it planted, or the process is left with stray int3s.
	uint8_t byte = 0;
	Check(Debugger::Target::SafeRead(reinterpret_cast<uint64_t>(&KytyDebuggerTestTarget), &byte, 1),
	      "post-shutdown read failed");
	Check(byte != 0xCC, "Shutdown left a breakpoint patch behind");
}

// ---- Overlay layout ---------------------------------------------------------------------

int  g_imgui_errors      = 0;
bool g_expect_imgui_error = false;

void OnImGuiError(ImGuiContext* /*ctx*/, void* /*user_data*/, const char* message) {
	g_imgui_errors++;
	if (g_expect_imgui_error) {
		return; // the self-check below deliberately provokes one
	}
	std::fprintf(stderr, "DebuggerCoreTests: ImGui layout error: %s\n",
	             message != nullptr ? message : "(none)");
}

// Walk every pane against a headless ImGui context. ImGui reports mismatched Begin/End pairs,
// unbalanced PushID/PopID and table column overruns through its error-recovery path; the
// callback above turns each into a test failure.
void RunOverlayFrames() {
	// Two tab bars remain (Graphics, and Modules & symbols); the rest are separate windows now
	// and are drawn every frame with all of them forced open.
	for (int tab = 0; tab < 2; tab++) {
		// Two frames per tab: ImGui settles tab selection and auto-sizing on the second.
		for (int frame = 0; frame < 2; frame++) {
			ImGui_ImplNull_NewFrame();
			ImGui::NewFrame();
			Debugger::Ui::DebuggerOverlay::DrawPanelsForTesting(tab);
			ImGui::Render();
			ImGui_ImplNullRender_RenderDrawData(ImGui::GetDrawData());
		}
	}
}

void TestOverlayLayout() {
	IMGUI_CHECKVERSION();
	ImGuiContext* context = ImGui::CreateContext();
	ImGui::SetCurrentContext(context);

	auto& io       = ImGui::GetIO();
	io.IniFilename = nullptr;
	io.LogFilename = nullptr;

	// Report errors through the callback rather than asserting or drawing a tooltip: asserts are
	// compiled out in Release, and a tooltip would just be discarded by the null renderer.
	io.ConfigErrorRecoveryEnableAssert  = false;
	io.ConfigErrorRecoveryEnableTooltip = false;

	context->ErrorCallback = OnImGuiError;
	g_imgui_errors         = 0;

	Check(ImGui_ImplNull_Init(), "the headless ImGui backend failed to start");

	// Prove the detector works before trusting a clean run: an unbalanced PushID is exactly the
	// class of mistake this test exists to catch, so if it slips through, so would a real one.
	g_expect_imgui_error             = true;
	io.ConfigErrorRecoveryEnableDebugLog = false; // keep the provoked error out of the log
	ImGui_ImplNull_NewFrame();
	ImGui::NewFrame();
	ImGui::PushID("deliberately-unbalanced");
	ImGui::Render();
	Check(g_imgui_errors > 0, "ImGui error detection is not working; the layout test cannot fail");
	g_imgui_errors                       = 0;
	g_expect_imgui_error                 = false;
	io.ConfigErrorRecoveryEnableDebugLog = true;

	// Empty state: no session, no modules, no halted thread.
	RunOverlayFrames();
	Check(g_imgui_errors == 0, "the overlay reported ImGui layout errors when empty");

	// Populated state: an armed breakpoint gives the disassembly a gutter marker and the
	// breakpoint tab a row to draw.
	Check(Debugger::Session::Initialize(), "Session::Initialize failed for the layout test");
	const auto id =
	    Debugger::Session::AddBreakpointAt(reinterpret_cast<uint64_t>(&KytyDebuggerTestTarget));
	Check(id != 0, "layout test could not arm a breakpoint");

	RunOverlayFrames();
	Check(g_imgui_errors == 0, "the overlay reported ImGui layout errors with content");

	Check(Debugger::Session::RemoveBreakpoint(id), "layout test cleanup failed");
	Debugger::Session::Shutdown();

	ImGui_ImplNull_Shutdown();
	ImGui::DestroyContext(context);
	ImGui::SetCurrentContext(nullptr);
}

// ---- Graphics observation -------------------------------------------------------------------

void TestGraphicsRegistry() {
	Check(Debugger::Session::Initialize(), "Session::Initialize failed");
	Debugger::Graphics::Reset();

	Check(Debugger::Graphics::IsCapturing(), "capture should follow the debugger being enabled");
	Check(Debugger::Graphics::Shaders().empty(), "the shader registry did not start empty");

	// A minimal valid SPIR-V module: magic, version, generator, bound, schema.
	const std::vector<uint32_t> spirv {0x07230203, 0x00010300, 0, 1, 0};
	Debugger::Graphics::ShaderCode::Resource texture_resource {};
	texture_resource.kind         = "image";
	texture_resource.index        = 3;
	texture_resource.source       = 11;
	texture_resource.first_use_pc = 0x48;
	texture_resource.read         = true;
	texture_resource.descriptor   = {0x12345678, 0x9abcdef0};
	texture_resource.address      = 0x280000000;
	texture_resource.width        = 1920;
	texture_resource.height       = 1080;
	texture_resource.depth        = 1;
	texture_resource.format       = 71;
	texture_resource.tile         = 27;

	Debugger::Graphics::RecordShader(Debugger::Graphics::ShaderStage::Vertex, 0xAAAA, 0x1000, 64,
	                                 spirv.data(), spirv.size(), "v_mov_b32 v0, v1", "ir-text",
	                                 {texture_resource});
	Debugger::Graphics::RecordShader(Debugger::Graphics::ShaderStage::Pixel, 0xBBBB, 0x2000, 32,
	                                 spirv.data(), spirv.size(), "", "");

	auto shaders = Debugger::Graphics::Shaders();
	Check(shaders.size() == 2, "the shader registry did not record both shaders");
	Check(shaders[0].hash == 0xAAAA, "shaders are not ordered by first compile");
	Check(shaders[0].stage == Debugger::Graphics::ShaderStage::Vertex, "wrong stage recorded");
	Check(shaders[0].gcn_bytes == 64, "wrong GCN size recorded");
	Check(shaders[0].resource_count == 1, "wrong shader resource count recorded");

	// Recompiling the same shader must not duplicate it, and must not lose text it already has.
	Debugger::Graphics::RecordShader(Debugger::Graphics::ShaderStage::Vertex, 0xAAAA, 0x1000, 64,
	                                 spirv.data(), spirv.size(), "", "");
	Check(Debugger::Graphics::Shaders().size() == 2, "a recompile duplicated a shader");

	Debugger::Graphics::ShaderCode code {};
	Check(Debugger::Graphics::GetShaderCode(0xAAAA, code), "GetShaderCode failed");
	Check(code.isa == "v_mov_b32 v0, v1", "a recompile dropped the captured ISA text");
	Check(code.ir == "ir-text", "a recompile dropped the captured IR text");
	Check(code.resources.size() == 1, "the shader resource snapshot was not retained");
	Check(code.resources[0].kind == "image" && code.resources[0].index == 3 &&
	          code.resources[0].read && code.resources[0].descriptor.size() == 2 &&
	          code.resources[0].address == 0x280000000 && code.resources[0].width == 1920 &&
	          code.resources[0].height == 1080 && code.resources[0].format == 71 &&
	          code.resources[0].tile == 27,
	      "the shader resource snapshot has the wrong contents");
	Check(!Debugger::Graphics::GetShaderCode(0x1234, code), "an unknown shader resolved");

	Debugger::Graphics::RecordImageEvent("upload", 0x280000000, 0x870000, 1920, 1080, 1, 4,
	                                     22, 37, 27);
	Debugger::Graphics::RecordImageEvent("gpu write", 0x290000000, 0x10000, 128, 128, 1, 4,
	                                     22, 37, 27);
	const auto image_events = Debugger::Graphics::ResourceHistory(0x280000000);
	Check(image_events.size() == 1 && image_events[0].action == "upload" &&
	          image_events[0].width == 1920 && image_events[0].bytes_per_block == 4,
	      "the filtered image resource history is wrong");

	const auto preview_id = Debugger::Graphics::RequestResourcePreview(0x280000000, 0x870000, true);
	const auto nonfinite_condition = Debugger::Graphics::AddBreakCondition(
	    Debugger::Graphics::BreakConditionKind::NonFinite, 0x280000000, {}, true);
	Debugger::Graphics::PreviewRequest preview_request {};
	Check(preview_id != 0 && Debugger::Graphics::TakePreviewRequest(preview_request) &&
	          preview_request.address == 0x280000000 && preview_request.size == 0x870000,
	      "the resource preview request was not delivered to the GPU side");
	Debugger::Graphics::PublishResourcePreview(preview_id, 0x280000000, 1920, 1080, 37, 2, 1,
	                                           {1, 2, 3, 4, 5, 6, 7, 8}, {},
	                                           {.content_hash = 0x12345678,
	                                            .total_pixels = 2073600,
	                                            .zero_pixels = 17,
	                                            .non_finite_pixels = 3,
	                                            .non_finite_components = 5});
	const auto preview = Debugger::Graphics::GetResourcePreview(0x280000000, 0x870000);
	Check(preview.status == "ready" && preview.width == 2 && preview.rgba.size() == 8 &&
	          preview.content_hash == 0x12345678 && preview.total_pixels == 2073600 &&
	          preview.non_finite_pixels == 3 && preview.non_finite_components == 5,
	      "the completed resource preview was not retained");
	std::string break_reason;
	Check(nonfinite_condition != 0 && Debugger::Graphics::TakeBreakRequest(break_reason) &&
	          break_reason.find("non-finite") != std::string::npos,
	      "the non-finite preview break condition did not request a safe-point pause");
	const auto preview_conditions = Debugger::Graphics::BreakConditions();
	Check(preview_conditions.size() == 1 && preview_conditions[0].hits == 1 &&
	          !preview_conditions[0].enabled,
	      "a one-shot non-finite condition did not retain its hit and disable itself");

	Debugger::Io::Reset();
	Debugger::Io::Record("read", 7, "/app0/movie.bk2", "D:/game/movie.bk2", 4096, 8192, 4096, 0);
	const auto io_events = Debugger::Io::History("movie", 16);
	Check(io_events.size() == 1 && io_events[0].descriptor == 7 &&
	          io_events[0].requested == 8192 && io_events[0].result == 4096,
	      "the filtered I/O history is wrong");
	Debugger::Io::Record("pread", 7, "/app0/movie.bk2", "D:/game/movie.bk2", 8192, 2048,
	                     1024, 0);
	Debugger::Io::Record("open", 8, "/app0/config.bin", "D:/game/config.bin", 0, 0, 8, 0);
	const auto io_files = Debugger::Io::Files({}, 16);
	const auto movie = std::find_if(io_files.begin(), io_files.end(), [](const auto& file) {
		return file.guest_path == "/app0/movie.bk2";
	});
	Check(io_files.size() == 2 && movie != io_files.end() && movie->reads == 2 &&
	          movie->bytes_read == 5120,
	      "the session-wide I/O file inventory did not aggregate file reads");

	// Draws accumulate into the current frame and only become readable when a flip closes it.
	Debugger::Graphics::DrawRecord draw {};
	draw.kind  = Debugger::Graphics::DrawKind::DrawIndexed;
	draw.count = 300;
	draw.vs_address = 0x1000;
	const auto shader_condition = Debugger::Graphics::AddBreakCondition(
	    Debugger::Graphics::BreakConditionKind::Shader, 0xAAAA, {}, false);
	Debugger::Graphics::RecordDraw(draw);
	Check(shader_condition != 0 && Debugger::Graphics::TakeBreakRequest(break_reason) &&
	          break_reason.find("shader") != std::string::npos,
	      "the shader hash/base break condition did not match a draw");

	draw.kind      = Debugger::Graphics::DrawKind::Dispatch;
	draw.groups[0] = 4;
	draw.submit_id = 77;
	draw.vs_address = 0;
	draw.cs_address = 0x2089bad00;
	Debugger::Graphics::RecordDraw(draw);
	const uint32_t draw_pm4[] = {0xc0011000, 0x11223344, 0x55667788};
	const uint32_t constant_pm4[] = {0xc0007600, 0xaabbccdd};
	Debugger::Graphics::RecordSubmission(77, 0, false, draw_pm4, std::size(draw_pm4),
	                                     constant_pm4, std::size(constant_pm4), true, false);
	Debugger::Graphics::SubmissionRecord submission {};
	Check(Debugger::Graphics::GetSubmission(77, submission) && submission.submit_id == 77 &&
	          submission.commands == std::vector<uint32_t>(std::begin(draw_pm4), std::end(draw_pm4)) &&
	          submission.constant_commands ==
	              std::vector<uint32_t>(std::begin(constant_pm4), std::end(constant_pm4)) &&
	          !submission.compute && !submission.truncated && submission.interrupt_on_done,
	      "the exact top-level PM4 submission was not retained for command capture");

	Debugger::Graphics::ResourceEvent traced {};
	traced.action = "gpu write";
	traced.note = "native image became authoritative";
	traced.image_index = 12;
	traced.image_generation = 4;
	traced.address = 0x280000000;
	traced.size = 0x870000;
	traced.metadata_address = 0x2ac000000;
	traced.metadata_size = 0x14000;
	traced.width = 1920;
	traced.height = 1080;
	traced.bytes_per_block = 4;
	traced.gpu_modified = true;
	traced.usage_storage = true;
	const auto resource_condition = Debugger::Graphics::AddBreakCondition(
	    Debugger::Graphics::BreakConditionKind::Resource, 0x280000100, "write", false);
	Debugger::Graphics::RecordImageEvent(traced);
	Check(resource_condition != 0 && Debugger::Graphics::TakeBreakRequest(break_reason) &&
	          break_reason.find("resource") != std::string::npos,
	      "the filtered resource-address break condition did not match an ownership event");
	const auto traced_events = Debugger::Graphics::ResourceTrace(0x280000100, 16, 16);
	Check(traced_events.size() == 2 && traced_events.back().image_index == 12 &&
	          traced_events.back().submit_id == 77 && traced_events.back().command_index == 1 &&
	          traced_events.back().cs_address == 0x2089bad00,
	      "resource trace did not retain image identity and command context");
	const auto traced_metadata = Debugger::Graphics::ResourceTrace(0x2ac000010, 4, 16);
	Check(traced_metadata.size() == 1 && traced_metadata[0].metadata_address == 0x2ac000000,
	      "resource trace did not follow an overlapping metadata range");
	const auto aliases = Debugger::Graphics::ResourceAliases(0x280000100, 16);
	Check(aliases.size() == 1 && aliases[0].active && aliases[0].gpu_modified &&
	          aliases[0].usage_storage,
	      "resource trace did not expose the latest alias ownership state");

	Debugger::Graphics::ResourceEvent watch_image {};
	watch_image.action = "upload";
	watch_image.image_index = 30;
	watch_image.image_generation = 2;
	watch_image.address = 0x2b0000000;
	watch_image.size = 0x10000;
	watch_image.width = 128;
	watch_image.height = 128;
	watch_image.depth = 1;
	watch_image.pitch = 128;
	watch_image.bytes_per_block = 4;
	watch_image.guest_format = 56; // Prospero::BufferFormat::k8_8_8_8UNorm
	watch_image.tile_mode = 27;    // Prospero::TileMode::kRenderTarget
	Debugger::Graphics::RecordImageEvent(watch_image);
	const auto watch_id = Debugger::Graphics::AddPixelWatch(watch_image.address, 3, 5);
	const auto pixel_condition = Debugger::Graphics::AddBreakCondition(
	    Debugger::Graphics::BreakConditionKind::PixelWatch, watch_id, {}, false);
	watch_image.action = "gpu write";
	Debugger::Graphics::RecordImageEvent(watch_image);
	const auto watches = Debugger::Graphics::PixelWatches();
	const auto watch_hits = Debugger::Graphics::PixelWatchHits();
	Check(watch_id != 0 && watches.size() == 1 && watches[0].exact &&
	          watches[0].pixel_address >= watch_image.address &&
	          watches[0].pixel_address + watches[0].pixel_size <=
	              watch_image.address + watch_image.size &&
	          !watch_hits.empty() && watch_hits.back().watch_id == watch_id &&
	          watch_hits.back().event.image_index == watch_image.image_index,
	      "pixel watchpoint did not resolve the PS5 tiled pixel or retain its resource event");
	Check(pixel_condition != 0 && Debugger::Graphics::TakeBreakRequest(break_reason) &&
	          break_reason.find("pixel") != std::string::npos,
	      "the pixel-watch break condition did not match a watched image event");
	Check(Debugger::Graphics::RemovePixelWatch(watch_id) &&
	          Debugger::Graphics::PixelWatches().empty(),
	      "pixel watchpoint removal failed");

	Check(Debugger::Graphics::LastFrame().empty(),
	      "draws became visible before a flip closed the frame");

	Debugger::Graphics::RecordFlip();

	const auto frame = Debugger::Graphics::LastFrame();
	Check(frame.size() == 2, "the completed frame did not capture both draws");
	Check(frame[0].index == 0 && frame[1].index == 1, "draw indices are wrong");

	const auto stats = Debugger::Graphics::GetStats();
	Check(stats.frame == 1, "the frame counter did not advance on flip");
	Check(stats.draws_last_frame == 1, "wrong draw count for the completed frame");
	Check(stats.dispatches_last_frame == 1, "wrong dispatch count for the completed frame");
	Check(stats.total_draws == 1 && stats.total_dispatches == 1, "wrong totals");
	Check(stats.shader_count == 2, "wrong shader count");

	Debugger::Graphics::Reset();
	Check(Debugger::Graphics::Shaders().empty(), "Reset did not clear the registry");
	Check(Debugger::Graphics::BreakConditions().empty(),
	      "Reset did not clear GPU break conditions");
	Check(Debugger::Graphics::ResourceHistory().empty(), "Reset did not clear resource history");
	Check(!Debugger::Graphics::GetSubmission(77, submission),
	      "Reset did not clear captured command streams");
	Debugger::Io::Reset();
	Check(Debugger::Io::History().empty(), "Reset did not clear I/O history");
	Check(Debugger::Io::Files().empty(), "Reset did not clear the I/O file inventory");

	Debugger::Session::Shutdown();

	// With the debugger off, nothing should be captured at all.
	Debugger::Graphics::RecordShader(Debugger::Graphics::ShaderStage::Vertex, 0xCCCC, 0, 0, nullptr,
	                                 0, "", "");
	Check(Debugger::Graphics::Shaders().empty(),
	      "shaders were captured with the debugger disabled");
}

// ---- A genuinely halted thread ------------------------------------------------------------

std::atomic<bool> g_worker_finished {false};

// Sits behind the debugger in the chain: anything reaching it is a breakpoint the debugger
// declined, which is the failure mode worth reporting rather than letting the process die with
// STATUS_BREAKPOINT and no explanation.
bool UnclaimedBreakpointReporter(const ExceptionInfo& info) {
	if (info.type != ExceptionType::Breakpoint) {
		return false;
	}

	std::fprintf(stderr,
	             "DebuggerCoreTests: breakpoint at rip-1=%016llx was not claimed by the debugger "
	             "(test target is %016llx)\n",
	             static_cast<unsigned long long>(info.rip - 1),
	             static_cast<unsigned long long>(reinterpret_cast<uint64_t>(
	                 &KytyDebuggerTestTarget)));

	ResumePastBreakpoint(info);
	return true;
}

void HaltableWorker() {
	CallTestTargetIndirectly(); // a breakpoint is planted on the target's first byte
	g_worker_finished.store(true, std::memory_order_release);
}

// Poll with a deadline so a failure reports rather than hanging the suite.
template <typename Predicate>
bool WaitFor(Predicate predicate, int timeout_ms) {
	for (int waited = 0; waited < timeout_ms; waited += 5) {
		if (predicate()) {
			return true;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}
	return predicate();
}

void DrawOneOverlayFrame() {
	ImGui_ImplNull_NewFrame();
	ImGui::NewFrame();
	Debugger::Ui::DebuggerOverlay::DrawPanelsForTesting();
	ImGui::Render();
	ImGui_ImplNullRender_RenderDrawData(ImGui::GetDrawData());
}

// Drives a real breakpoint hit on a worker thread. This is the only path that reaches the
// halted-thread branches of the UI — registers with values, the call stack, the current-row
// marker — and it is what makes the follow-rip scrolling behaviour observable.
void TestHaltedThreadAndScrolling() {
	const auto target = reinterpret_cast<uint64_t>(&KytyDebuggerTestTarget);

	IMGUI_CHECKVERSION();
	ImGuiContext* context = ImGui::CreateContext();
	ImGui::SetCurrentContext(context);

	auto& io                            = ImGui::GetIO();
	io.IniFilename                      = nullptr;
	io.LogFilename                      = nullptr;
	io.ConfigErrorRecoveryEnableAssert  = false;
	io.ConfigErrorRecoveryEnableTooltip = false;
	context->ErrorCallback              = OnImGuiError;
	g_imgui_errors                      = 0;

	Check(ImGui_ImplNull_Init(), "the headless ImGui backend failed to start");
	Check(Debugger::Session::Initialize(), "Session::Initialize failed");
	Check(Common::HostException::AddHandler(UnclaimedBreakpointReporter, 50),
	      "could not install the unclaimed-breakpoint reporter");

	const auto id = Debugger::Session::AddBreakpointAt(target);
	Check(id != 0, "could not arm the worker's breakpoint");

	// The patch must actually be in the code before the worker runs.
	uint8_t planted = 0;
	Check(Debugger::Target::SafeRead(target, &planted, 1), "could not read back the target");
	Check(planted == 0xCC, "the breakpoint byte was not planted at the target");

	g_worker_finished.store(false, std::memory_order_release);
	std::thread worker(HaltableWorker);

	const bool halted =
	    WaitFor([] { return !Debugger::Session::Stopped().empty(); }, 10000);

	// Whatever happens next, the worker must be released or the join below never returns.
	if (halted) {
		const auto stopped = Debugger::Session::Stopped();
		Check(stopped.size() == 1, "expected exactly one halted thread");
		Check(stopped[0].reason == Debugger::StopReason::Breakpoint,
		      "the worker halted for the wrong reason");
		Check(stopped[0].address == target, "the worker halted at the wrong address");
		Check(stopped[0].regs.valid, "no registers were captured at the halt");
		Check(stopped[0].regs.rip == target, "rip was not rewound onto the breakpoint");

		Check(!Debugger::Session::Backtrace(stopped[0].unique_id).empty(),
		      "the halted thread produced an empty backtrace");

		// The regression: following rip must scroll once when the halt address changes, not on
		// every frame. Re-scrolling every frame pins the view at the top and undoes any
		// scrolling the user does while the guest is halted.
		const auto before = Debugger::Ui::DebuggerOverlay::ScrollRequestCountForTesting();
		DrawOneOverlayFrame();
		const auto after_first = Debugger::Ui::DebuggerOverlay::ScrollRequestCountForTesting();
		Check(after_first > before, "following rip never scrolled to the halted instruction");

		for (int frame = 0; frame < 8; frame++) {
			DrawOneOverlayFrame();
		}
		Check(Debugger::Ui::DebuggerOverlay::ScrollRequestCountForTesting() == after_first,
		      "the disassembly keeps scrolling to rip every frame, which resets the scrollbar");

		Check(g_imgui_errors == 0, "the overlay reported ImGui layout errors while halted");
	}

	Debugger::Session::ResumeAll();

	Check(WaitFor([] { return g_worker_finished.load(std::memory_order_acquire); }, 10000),
	      "the worker never resumed past its breakpoint");
	worker.join();

	if (!halted) {
		// Separate "the patch was never planted" from "the worker never executed there".
		uint8_t byte = 0;
		Debugger::Target::SafeRead(target, &byte, 1);
		std::fprintf(stderr,
		             "DebuggerCoreTests: worker did not halt; byte at target = 0x%02x "
		             "(0xcc means the patch was planted and the call missed it)\n",
		             byte);
	}
	Check(halted, "the worker thread never halted on its breakpoint");

	Debugger::Session::RemoveBreakpoint(id);
	Common::HostException::RemoveHandler(UnclaimedBreakpointReporter);
	Debugger::Session::Shutdown();

	ImGui_ImplNull_Shutdown();
	ImGui::DestroyContext(context);
	ImGui::SetCurrentContext(nullptr);
}

// ---- Stepping -----------------------------------------------------------------------------

std::atomic<bool> g_stepper_finished {false};

void SteppingWorker() {
	CallStepCallerIndirectly();
	g_stepper_finished.store(true, std::memory_order_release);
}

// Wait for a halt whose address differs from `previous`, so a resume that steps is observably
// distinct from one that has not landed yet.
bool WaitForNewHalt(uint64_t previous, uint64_t& address_out, int timeout_ms) {
	const bool ok = WaitFor(
	    [previous] {
		    const auto stopped = Debugger::Session::Stopped();
		    return !stopped.empty() && stopped.front().address != previous;
	    },
	    timeout_ms);

	if (!ok) {
		return false;
	}

	const auto stopped = Debugger::Session::Stopped();
	address_out        = stopped.front().address;
	return true;
}

// Disassembly must show the guest's instructions, not the debugger's patches. Step-over derives
// an instruction length from this, so a patched read makes it plant a breakpoint in the middle
// of an instruction and corrupt the code stream.
void TestDisassemblyIgnoresBreakpointBytes() {
	Check(Debugger::Session::Initialize(), "Session::Initialize failed");

	const auto address = reinterpret_cast<uint64_t>(&KytyDebuggerStepCaller);

	const auto clean = Debugger::Disasm::Decode(address, 1);
	Check(!clean.empty(), "could not decode the step target");

	const auto id = Debugger::Session::AddBreakpointAt(address);
	Check(id != 0, "could not arm the breakpoint");

	const auto patched = Debugger::Disasm::Decode(address, 1);
	Check(!patched.empty(), "could not decode with a breakpoint armed");
	Check(patched[0].length == clean[0].length,
	      "the breakpoint patch changed the decoded instruction length");
	Check(patched[0].text == clean[0].text,
	      "the disassembly shows the breakpoint patch instead of the instruction");

	Check(Debugger::Session::RemoveBreakpoint(id), "RemoveBreakpoint failed");
	Debugger::Session::Shutdown();
}

// Drives all three step modes against a genuinely halted thread. Every one of them used to
// either corrupt memory (a breakpoint planted mid-instruction, or at whatever rbp happened to
// point at) or kill the process outright.
void TestStepping() {
	Check(Debugger::Session::Initialize(), "Session::Initialize failed");

	const auto entry = reinterpret_cast<uint64_t>(&KytyDebuggerStepCaller);
	const auto id    = Debugger::Session::AddBreakpointAt(entry);
	Check(id != 0, "could not arm the step entry breakpoint");

	g_stepper_finished.store(false, std::memory_order_release);
	std::thread worker(SteppingWorker);

	bool halted = WaitFor([] { return !Debugger::Session::Stopped().empty(); }, 10000);

	if (halted) {
		auto stopped = Debugger::Session::Stopped();
		Check(stopped.front().address == entry, "halted somewhere unexpected");
		const int thread_id = stopped.front().unique_id;

		// The entry breakpoint would otherwise be hit again by the steps below.
		Check(Debugger::Session::RemoveBreakpoint(id), "could not remove the entry breakpoint");

		uint64_t address = entry;

		for (int step = 0; step < 6 && halted; step++) {
			const uint64_t previous = address;
			Debugger::Session::Resume(thread_id, Debugger::Session::ResumeMode::StepInto);
			halted = WaitForNewHalt(previous, address, 10000);
			Check(halted, "step into did not reach a new halt");
		}

		if (halted) {
			const uint64_t previous = address;
			Debugger::Session::Resume(thread_id, Debugger::Session::ResumeMode::StepOver);
			halted = WaitForNewHalt(previous, address, 10000);
			Check(halted, "step over did not reach a new halt");
		}

		if (halted) {
			// Step out either lands in the caller or runs the thread to completion; both are
			// fine, and neither may corrupt memory on the way.
			Debugger::Session::Resume(thread_id, Debugger::Session::ResumeMode::StepOut);
			WaitFor(
			    [] {
				    return !Debugger::Session::Stopped().empty() ||
				           g_stepper_finished.load(std::memory_order_acquire);
			    },
			    10000);
		}
	}

	Debugger::Session::ResumeAll();

	Check(WaitFor([] { return g_stepper_finished.load(std::memory_order_acquire); }, 10000),
	      "the stepping worker never ran to completion");
	worker.join();

	Check(halted, "the stepping worker never halted");

	// Stepping must not leave stray patches behind.
	Debugger::Session::ClearBreakpoints();
	uint8_t byte = 0;
	Check(Debugger::Target::SafeRead(entry, &byte, 1), "post-step read failed");
	Check(byte != 0xCC, "stepping left a breakpoint patch on the entry point");

	Debugger::Session::Shutdown();
}

// ---- Break on fatal error -------------------------------------------------------------------

std::atomic<bool> g_fatal_returned {false};

void FatalWorker() {
	// ReportFatal returns once the halt is released; the real caller then exits.
	Debugger::Session::ReportFatal("--- Fatal Error ---\nNot implemented (test) in test.cpp:1");
	g_fatal_returned.store(true, std::memory_order_release);
}

// A fatal error halts the failing thread for inspection, whichever thread it is. That includes
// the ones that draw the overlay, where parking would mean nothing can ever present — so the
// halt watches for the overlay drawing and releases itself if it never does. Without that, a
// fatal in the renderer would hang the emulator instead of exiting it, which is strictly worse
// than not stopping at all.
//
// Nothing draws the overlay in this test, so every halt here takes the watchdog path.
void TestBreakOnFatal() {
	Check(Debugger::Session::Initialize(), "Session::Initialize failed");
	Check(Debugger::Session::BreakOnFatalEnabled(), "break on fatal should default to on");

	// Switched off, nothing halts and the call is immediate.
	Debugger::Session::SetBreakOnFatal(false);
	Check(!Debugger::Session::BreakOnFatalEnabled(), "SetBreakOnFatal did not take effect");

	const auto before = std::chrono::steady_clock::now();
	Debugger::Session::ReportFatal("ignored");
	const auto disabled_elapsed = std::chrono::steady_clock::now() - before;

	Check(Debugger::Session::Stopped().empty(), "a fatal halted with the option disabled");
	Check(disabled_elapsed < std::chrono::seconds(2), "a disabled fatal still blocked");

	// Switched on, the halt happens and the watchdog releases it because no overlay appears.
	Debugger::Session::SetBreakOnFatal(true);

	g_fatal_returned.store(false, std::memory_order_release);
	std::thread worker(FatalWorker);

	const bool returned =
	    WaitFor([] { return g_fatal_returned.load(std::memory_order_acquire); }, 30000);

	Debugger::Session::ResumeAll();
	worker.join();

	Check(returned,
	      "a fatal halt never released, so a crash in the renderer would hang the emulator "
	      "rather than exit it");

	Debugger::Session::Shutdown();
}

// ---- External debugger transport -----------------------------------------------------------

void TestRemoteDebuggerAttach() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	Check(Debugger::Session::Initialize(), "remote Session::Initialize failed");
	const uint32_t replay_pm4[] = {0xc0011000, 0x00000001, 0x00000002};
	Debugger::Graphics::RecordSubmission(99123, 0, false, replay_pm4, std::size(replay_pm4),
	                                     nullptr, 0, false, false);
	Debugger::Graphics::DrawRecord replay_draw {};
	replay_draw.submit_id = 99123;
	replay_draw.kind = Debugger::Graphics::DrawKind::Draw;
	replay_draw.count = 3;
	Debugger::Graphics::RecordDraw(replay_draw);
	Debugger::Graphics::RecordFlip();
	Check(Debugger::Remote::Start(), "remote debugger server did not start");

	const auto pid = GetCurrentProcessId();
	const auto descriptor = std::filesystem::path(std::getenv("LOCALAPPDATA")) / "KytyPS5" /
	                        "DebuggerSessions" / (std::to_string(pid) + ".json");
	std::ifstream file(descriptor, std::ios::binary);
	Check(file.good(), "remote debugger did not publish its session descriptor");
	const auto session = nlohmann::json::parse(file, nullptr, false);
	Check(!session.is_discarded(), "remote session descriptor is invalid JSON");
	file.close();

	const auto endpoint = session.value("endpoint", std::string());
	const auto token = session.value("token", std::string());
	Check(!endpoint.empty() && !token.empty(), "remote descriptor omitted endpoint credentials");
	const auto pipe_name = L"\\\\.\\pipe\\" + std::wstring(endpoint.begin(), endpoint.end());
	Check(WaitNamedPipeW(pipe_name.c_str(), 5000) != FALSE, "remote named pipe did not appear");
	const auto pipe = CreateFileW(pipe_name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
	                              OPEN_EXISTING, 0, nullptr);
	Check(pipe != INVALID_HANDLE_VALUE, "remote debugger attach failed");

	auto request = nlohmann::json({{"command", "summary"}, {"token", token}}).dump() + "\n";
	DWORD written = 0;
	Check(WriteFile(pipe, request.data(), static_cast<DWORD>(request.size()), &written, nullptr) !=
	          FALSE &&
	          written == request.size(),
	      "remote request write failed");
	char response_bytes[4096] = {};
	DWORD read = 0;
	Check(ReadFile(pipe, response_bytes, sizeof(response_bytes), &read, nullptr) != FALSE && read > 0,
	      "remote response read failed");
	const auto response = nlohmann::json::parse(response_bytes, response_bytes + read, nullptr, false);
	Check(!response.is_discarded() && response.value("ok", false) &&
	          response.value("protocol", 0) == 1,
	      "remote summary response failed protocol validation");

	request = nlohmann::json({{"command", "capture_command"}, {"token", token},
	                          {"submit", "99123"}}).dump() + "\n";
	Check(WriteFile(pipe, request.data(), static_cast<DWORD>(request.size()), &written, nullptr) !=
	          FALSE && written == request.size(),
	      "command-capture request write failed");
	std::memset(response_bytes, 0, sizeof(response_bytes));
	read = 0;
	Check(ReadFile(pipe, response_bytes, sizeof(response_bytes), &read, nullptr) != FALSE && read > 0,
	      "command-capture response read failed");
	const auto capture = nlohmann::json::parse(response_bytes, response_bytes + read, nullptr, false);
	Check(!capture.is_discarded() && capture.value("ok", false) &&
	          !capture.value("executable", true) && capture.value("command_words", 0) == 3,
	      "remote command capture did not report its exact PM4/non-executable boundary");
	const auto capture_path = std::filesystem::path(capture.value("path", std::string()));
	Check(std::filesystem::exists(capture_path / "manifest.json") &&
	          std::filesystem::file_size(capture_path / "commands.pm4") == sizeof(replay_pm4),
	      "remote command capture did not publish the expected bundle files");

	request = nlohmann::json({{"command", "renderdoc_status"}, {"token", token}}).dump() + "\n";
	Check(WriteFile(pipe, request.data(), static_cast<DWORD>(request.size()), &written, nullptr) !=
	          FALSE && written == request.size(),
	      "RenderDoc-status request write failed");
	std::memset(response_bytes, 0, sizeof(response_bytes));
	read = 0;
	Check(ReadFile(pipe, response_bytes, sizeof(response_bytes), &read, nullptr) != FALSE && read > 0,
	      "RenderDoc-status response read failed");
	const auto renderdoc =
	    nlohmann::json::parse(response_bytes, response_bytes + read, nullptr, false);
	Check(!renderdoc.is_discarded() && renderdoc.value("ok", false) &&
	          renderdoc.value("state", std::string()) == "idle" &&
	          renderdoc.contains("available") && renderdoc.contains("completed_captures"),
	      "remote RenderDoc status did not expose its availability and lifecycle");
	std::error_code cleanup_error;
	std::filesystem::remove_all(capture_path, cleanup_error);
	CloseHandle(pipe);

	Debugger::Remote::Stop();
	Check(!std::filesystem::exists(descriptor), "remote descriptor survived server shutdown");
	Debugger::Session::Shutdown();
#endif
}

} // namespace

int main() {
	// Symbolization reaches the RuntimeLinker singleton, whose constructor asserts it runs on
	// the main thread — which is only knowable once the thread registry has been initialized.
	Common::InitializeThreads();

	TestHandlerChainOrdering();
	TestHandlerCapacity();
	TestSafeMemoryAccess();
	TestBreakpointArmAndOverlay();
	TestWriteUnderBreakpoint();
	TestDisassembly();
	TestSymbolResolution();

	// The facade logs through Config and Log, so both have to exist before it runs.
	Config::Initialize();
	Log::Initialize();
	TestDebuggerStaysInertWithoutTheFlag();
	TestDebuggerHonoursConfiguredBreakpoints();
	TestOverlayLayout();
	TestHaltedThreadAndScrolling();
	TestDisassemblyIgnoresBreakpointBytes();
	TestStepping();
	TestGraphicsRegistry();
	TestBreakOnFatal();
	TestRemoteDebuggerAttach();
	Log::Shutdown();
	Config::Shutdown();

	std::printf("DebuggerCoreTests: all tests passed\n");
	return 0;
}
