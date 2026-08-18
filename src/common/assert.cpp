#include "common/assert.h"

#include "common/debug.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "kytyGitVersion.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <string>

namespace Common {

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS && KYTY_BUILD == KYTY_BUILD_DEBUG &&                    \
    KYTY_COMPILER == KYTY_COMPILER_CLANG
constexpr int PRINT_STACK_FROM = 4;
#else
constexpr int PRINT_STACK_FROM = 2;
#endif

static std::string BuildFatalReport(const char* title, std::string_view text, const char* file,
                                    int line) {
	DebugStack stack;
	DebugStack::Trace(&stack);

	std::string report = "--- Build ---\n" KYTY_BUILD_LABEL "\n--- Stack Trace ---\n";
	for (int i = PRINT_STACK_FROM; i < stack.depth; i++) {
		report += fmt::format("[{}] {:016x}\n", i - PRINT_STACK_FROM,
		                      static_cast<uint64_t>(stack.GetAddr(i)));
	}
	report += fmt::format("{}\n{} in {}:{}\n", title, text, file, line);
	return report;
}

static std::atomic<FatalHandler> g_fatal_handler {nullptr};

void SetFatalHandler(FatalHandler handler) {
	g_fatal_handler.store(handler, std::memory_order_release);
}

// Offer the failing thread to the debugger before the emulator starts tearing itself down, so
// there is still something to inspect when it stops.
static void OfferToDebugger(const std::string& report) {
	if (const auto handler = g_fatal_handler.load(std::memory_order_acquire); handler != nullptr) {
		handler(report.c_str());
	}
}

static int DbgReport(const char* title, std::string_view text, const char* file, int line) {
	const auto report = BuildFatalReport(title, text, file, line);
	Log::WriteFatal(report);
	OfferToDebugger(report);
	Subsystems::EmergencyShutdownActive();
	return 1;
}

int DbgExitIfHandler(const char* expr, const char* file, int line) {
	return DbgReport("--- Fatal Error ---", fmt::format("Error: condition ({}) is true", expr),
	                 file, line);
}

int DbgNotImplementedHandler(const char* expr, const char* file, int line) {
	return DbgReport("--- Fatal Error ---", fmt::format("Not implemented ({})", expr), file, line);
}

int DbgExitHandler(const char* file, int line, std::string_view text) {
	const auto report = BuildFatalReport("--- Error ---", text, file, line);
	Log::WriteFatal(report);
	OfferToDebugger(report);
	return 1;
}

int DbgExitHandler(const char* file, int line, fmt::text_style style, std::string_view text) {
	const auto report = BuildFatalReport("--- Error ---", text, file, line);
	Log::WriteFatal(style, report);
	OfferToDebugger(report);
	return 1;
}

void DbgExit(int status) {
	std::fflush(nullptr);
	std::_Exit(status);
}

} // namespace Common
