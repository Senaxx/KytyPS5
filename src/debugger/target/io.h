#ifndef EMULATOR_SRC_DEBUGGER_TARGET_IO_H_
#define EMULATOR_SRC_DEBUGGER_TARGET_IO_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Debugger::Io {

struct Event {
	uint64_t    sequence       = 0;
	uint64_t    timestamp_us   = 0;
	int         thread_id      = 0;
	uint64_t    caller         = 0;
	std::string module;
	std::string operation;
	int         descriptor     = -1;
	std::string guest_path;
	std::string host_path;
	int64_t     offset         = -1;
	uint64_t    requested      = 0;
	int64_t     result         = 0;
};

struct FileSummary {
	std::string guest_path;
	std::string host_path;
	std::string last_module;
	uint64_t    first_sequence    = 0;
	uint64_t    last_sequence     = 0;
	uint64_t    last_timestamp_us = 0;
	uint64_t    opens             = 0;
	uint64_t    closes            = 0;
	uint64_t    reads             = 0;
	uint64_t    writes            = 0;
	uint64_t    seeks             = 0;
	uint64_t    stats             = 0;
	uint64_t    bytes_read        = 0;
	uint64_t    bytes_written     = 0;
};

// Records metadata only. File payload bytes are intentionally not retained by default.
void Record(const char* operation, int descriptor, const std::string& guest_path,
            const std::string& host_path, int64_t offset, uint64_t requested, int64_t result,
            uint64_t caller);

std::vector<Event> History(const std::string& filter = {}, size_t limit = 2048);
// Session-wide distinct file inventory. Unlike History, entries remain visible after their
// detailed events rotate out of the bounded event ring.
std::vector<FileSummary> Files(const std::string& filter = {}, size_t limit = 4096);
void               Reset();

} // namespace Debugger::Io

#endif // EMULATOR_SRC_DEBUGGER_TARGET_IO_H_
