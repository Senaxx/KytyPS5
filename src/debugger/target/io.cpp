#include "debugger/target/io.h"

#include "common/stringUtils.h"
#include "common/threads.h"
#include "debugger/core/session.h"
#include "debugger/symbols/symbols.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <unordered_map>

namespace Debugger::Io {
namespace {

constexpr size_t MAX_EVENTS = 16384;
constexpr size_t MAX_FILES  = 4096;

std::mutex                                   g_mutex;
std::vector<Event>                           g_events;
std::unordered_map<std::string, FileSummary> g_files;
uint64_t                                     g_sequence = 0;
const auto                                   g_start    = std::chrono::steady_clock::now();

void UpdateFileSummary(const Event& event) {
	if (event.guest_path.empty() && event.host_path.empty()) return;
	// Reads from AvPlayer often omit the already-resolved host path. Key primarily by guest path so
	// those events remain part of the same file row as the preceding open.
	const auto key = !event.guest_path.empty() ? "g:" + event.guest_path : "h:" + event.host_path;
	if (!g_files.contains(key) && g_files.size() >= MAX_FILES) {
		auto oldest = g_files.begin();
		for (auto it = g_files.begin(); it != g_files.end(); ++it) {
			if (it->second.last_sequence < oldest->second.last_sequence) oldest = it;
		}
		g_files.erase(oldest);
	}
	auto& file = g_files[key];
	if (file.opens == 0 && file.closes == 0 && file.reads == 0 && file.writes == 0 &&
	    file.seeks == 0 && file.stats == 0) {
		file.guest_path = event.guest_path;
		file.host_path = event.host_path;
		file.first_sequence = event.sequence;
	}
	if (file.host_path.empty() && !event.host_path.empty()) file.host_path = event.host_path;
	file.last_sequence = event.sequence;
	file.last_timestamp_us = event.timestamp_us;
	file.last_module = event.module;
	const auto operation = Common::ToLower(event.operation);
	if (operation.find("open") != std::string::npos) file.opens++;
	if (operation.find("close") != std::string::npos) file.closes++;
	if (operation.find("read") != std::string::npos) {
		file.reads++;
		if (event.result > 0) file.bytes_read += static_cast<uint64_t>(event.result);
	}
	if (operation.find("write") != std::string::npos) {
		file.writes++;
		if (event.result > 0) file.bytes_written += static_cast<uint64_t>(event.result);
	}
	if (operation.find("seek") != std::string::npos) file.seeks++;
	if (operation.find("stat") != std::string::npos) file.stats++;
}

} // namespace

void Record(const char* operation, int descriptor, const std::string& guest_path,
            const std::string& host_path, int64_t offset, uint64_t requested, int64_t result,
            uint64_t caller) {
	if (!Session::IsEnabled() || operation == nullptr) return;

	Event event {};
	event.timestamp_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                                std::chrono::steady_clock::now() - g_start)
	                                                .count());
	event.thread_id  = Common::Thread::GetThreadIdUnique();
	event.caller     = caller;
	event.module     = Symbols::Describe(caller).module;
	event.operation  = operation;
	event.descriptor = descriptor;
	event.guest_path = guest_path;
	event.host_path  = host_path;
	event.offset     = offset;
	event.requested  = requested;
	event.result     = result;

	const std::lock_guard lock(g_mutex);
	event.sequence = g_sequence++;
	if (g_events.size() >= MAX_EVENTS) {
		g_events.erase(g_events.begin(), g_events.begin() + MAX_EVENTS / 4);
	}
	UpdateFileSummary(event);
	g_events.push_back(std::move(event));
}

std::vector<Event> History(const std::string& filter, size_t limit) {
	const auto needle = Common::ToLower(filter);
	const std::lock_guard lock(g_mutex);
	std::vector<Event> out;
	limit = std::min(limit, g_events.size());
	for (auto it = g_events.rbegin(); it != g_events.rend() && out.size() < limit; ++it) {
		if (!needle.empty()) {
			const auto text = Common::ToLower(it->operation + " " + it->guest_path + " " +
			                                      it->host_path + " " + it->module);
			if (text.find(needle) == std::string::npos) continue;
		}
		out.push_back(*it);
	}
	std::reverse(out.begin(), out.end());
	return out;
}

std::vector<FileSummary> Files(const std::string& filter, size_t limit) {
	const auto needle = Common::ToLower(filter);
	const std::lock_guard lock(g_mutex);
	std::vector<FileSummary> out;
	out.reserve(g_files.size());
	for (const auto& [key, file]: g_files) {
		(void)key;
		if (!needle.empty()) {
			const auto text = Common::ToLower(file.guest_path + " " + file.host_path + " " +
			                                  file.last_module);
			if (text.find(needle) == std::string::npos) continue;
		}
		out.push_back(file);
	}
	std::sort(out.begin(), out.end(), [](const FileSummary& left, const FileSummary& right) {
		return left.last_sequence > right.last_sequence;
	});
	if (out.size() > limit) out.resize(limit);
	return out;
}

void Reset() {
	const std::lock_guard lock(g_mutex);
	g_events.clear();
	g_files.clear();
	g_sequence = 0;
}

} // namespace Debugger::Io
