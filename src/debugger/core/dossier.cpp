#include "debugger/core/dossier.h"

#include "common/file.h"
#include "common/threads.h"
#include "debugger/symbols/symbols.h"
#include "debugger/target/graphics.h"
#include "debugger/target/memory.h"
#include "kernel/pthread.h"

#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace Debugger::Dossier {

namespace {

namespace Gfx = Debugger::Graphics;

std::atomic<uint32_t> g_sequence {0};

std::string Hex(uint64_t value) {
	std::array<char, 32> buffer {};
	std::snprintf(buffer.data(), buffer.size(), "%016llx", static_cast<unsigned long long>(value));
	return buffer.data();
}

std::string Line(const char* format, ...) {
	std::array<char, 512> buffer {};
	va_list               args = nullptr;
	va_start(args, format);
	std::vsnprintf(buffer.data(), buffer.size(), format, args);
	va_end(args);
	return buffer.data();
}

void AppendRegisters(std::string& out, const Registers& regs) {
	out += "## Registers\n\n";

	if (!regs.valid) {
		out += "Not captured on this platform.\n\n";
		return;
	}

	out += "```\n";
	out += Line("rax %s  rbx %s  rcx %s  rdx %s\n", Hex(regs.rax).c_str(), Hex(regs.rbx).c_str(),
	            Hex(regs.rcx).c_str(), Hex(regs.rdx).c_str());
	out += Line("rsi %s  rdi %s  rbp %s  rsp %s\n", Hex(regs.rsi).c_str(), Hex(regs.rdi).c_str(),
	            Hex(regs.rbp).c_str(), Hex(regs.rsp).c_str());
	out += Line("r8  %s  r9  %s  r10 %s  r11 %s\n", Hex(regs.r8).c_str(), Hex(regs.r9).c_str(),
	            Hex(regs.r10).c_str(), Hex(regs.r11).c_str());
	out += Line("r12 %s  r13 %s  r14 %s  r15 %s\n", Hex(regs.r12).c_str(), Hex(regs.r13).c_str(),
	            Hex(regs.r14).c_str(), Hex(regs.r15).c_str());
	out += Line("rip %s  flg %s\n", Hex(regs.rip).c_str(), Hex(regs.rflags).c_str());
	out += "```\n\n";
	out += "rip resolves to " + Symbols::Format(regs.rip) + "\n\n";
}

void AppendThreads(std::string& out) {
	out += "## Guest threads\n\n";

	std::vector<Libs::LibKernel::GuestThreadInfo> threads;
	Libs::LibKernel::PthreadEnumerate(threads);

	if (threads.empty()) {
		out += "None.\n\n";
		return;
	}

	out += "| id | name | state | stack |\n|---|---|---|---|\n";
	for (const auto& thread: threads) {
		out += Line("| %d | %s | %s | %s +%llu KiB |\n", thread.unique_id,
		            thread.name.empty() ? "(unnamed)" : thread.name.c_str(),
		            thread.alive ? "running" : "done", Hex(thread.stack_addr).c_str(),
		            static_cast<unsigned long long>(thread.stack_size / 1024));
	}
	out += "\n";
}

void AppendModules(std::string& out) {
	out += "## Modules\n\n";

	const auto modules = Symbols::Modules();
	if (modules.empty()) {
		out += "None loaded.\n\n";
		return;
	}

	out += "| module | base | size |\n|---|---|---|\n";
	for (const auto& module: modules) {
		out += Line("| %s | %s | %llu KiB |\n", module.name.c_str(), Hex(module.base_vaddr).c_str(),
		            static_cast<unsigned long long>(module.size / 1024));
	}
	out += "\n";
}

void AppendGraphics(std::string& out) {
	out += "## Graphics\n\n";

	const auto stats = Gfx::GetStats();
	out += Line("Frame %u, %u draws and %u dispatches in the last frame, %llu draws and %llu "
	            "dispatches in total, %u shaders recompiled.\n\n",
	            stats.frame, stats.draws_last_frame, stats.dispatches_last_frame,
	            static_cast<unsigned long long>(stats.total_draws),
	            static_cast<unsigned long long>(stats.total_dispatches), stats.shader_count);

	const auto draws = Gfx::LastFrame();
	if (!draws.empty()) {
		out += "### Last completed frame\n\n";
		out += "| # | kind | count | instances | vs/cs | ps |\n|---|---|---|---|---|---|\n";

		constexpr size_t MAX_ROWS = 200;
		for (size_t i = 0; i < draws.size() && i < MAX_ROWS; i++) {
			const auto& draw = draws[i];
			out += Line(
			    "| %u | %s | %u | %u | %s | %s |\n", draw.index, Gfx::KindName(draw.kind),
			    draw.kind == Gfx::DrawKind::Dispatch ? draw.groups[0] : draw.count, draw.instances,
			    Hex(draw.kind == Gfx::DrawKind::Dispatch ? draw.cs_address : draw.vs_address)
			        .c_str(),
			    Hex(draw.ps_address).c_str());
		}
		if (draws.size() > MAX_ROWS) {
			out += Line("\n%zu further draws not listed.\n", draws.size() - MAX_ROWS);
		}
		out += "\n";
	}

	const auto shaders = Gfx::Shaders();
	if (!shaders.empty()) {
		out += "### Shaders\n\n";
		out += "| stage | hash | gcn bytes |\n|---|---|---|\n";

		constexpr size_t MAX_ROWS = 100;
		for (size_t i = 0; i < shaders.size() && i < MAX_ROWS; i++) {
			out += Line("| %s | %s | %u |\n", Gfx::StageName(shaders[i].stage),
			            Hex(shaders[i].hash).c_str(), shaders[i].gcn_bytes);
		}
		if (shaders.size() > MAX_ROWS) {
			out += Line("\n%zu further shaders not listed.\n", shaders.size() - MAX_ROWS);
		}
		out += "\n";
	}
}

void AppendMemory(std::string& out, const char* label, uint64_t address) {
	if (address == 0) {
		return;
	}

	out += Line("### %s around %s\n\n", label, Hex(address).c_str());

	constexpr uint64_t BEFORE = 64;
	constexpr uint32_t ROWS   = 8;

	const uint64_t base = address > BEFORE ? address - BEFORE : 0;

	out += "```\n";
	for (uint32_t row = 0; row < ROWS; row++) {
		const uint64_t          row_address = base + row * 16;
		std::array<uint8_t, 16> data {};

		if (!Target::SafeRead(row_address, data.data(), data.size())) {
			out += Hex(row_address) + "  <unmapped>\n";
			continue;
		}

		std::string text = Hex(row_address) + "  ";
		for (const auto byte: data) {
			text += Line("%02x ", byte);
		}
		out += text + "\n";
	}
	out += "```\n\n";
}

} // namespace

std::string Write(const std::string& report, const Registers& regs) {
	std::string out;

	out += "# KytyPS5 crash dossier\n\n";
	out += Line("Failing thread: %d\n\n", Common::Thread::GetThreadIdUnique());

	out += "## Report\n\n```\n";
	out += report;
	out += "\n```\n\n";

	AppendRegisters(out, regs);
	AppendThreads(out);
	AppendModules(out);
	AppendGraphics(out);

	out += "## Memory\n\n";
	AppendMemory(out, "Code", regs.rip);
	AppendMemory(out, "Stack", regs.rsp);

	const auto folder = std::filesystem::path("_logs");
	Common::File::CreateDirectories(folder);

	const auto index = g_sequence.fetch_add(1, std::memory_order_relaxed);
	auto       path  = folder / Line("crash_%u.md", index);

	Common::File file;
	file.Create(path);
	if (file.IsInvalid()) {
		return {};
	}

	file.Printf("%s", out.c_str());
	file.Close();

	return path.string();
}

} // namespace Debugger::Dossier
