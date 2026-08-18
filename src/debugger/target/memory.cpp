#include "debugger/target/memory.h"

#include "common/virtualMemory.h"

#include <cstring>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#else
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace Debugger::Target {

namespace {

constexpr uint64_t PAGE_SIZE = 0x1000;

uint64_t PageBase(uint64_t address) {
	return address & ~(PAGE_SIZE - 1);
}

// Byte span of every page the range touches.
void PageSpan(uint64_t address, size_t size, uint64_t& base, uint64_t& length) {
	base         = PageBase(address);
	uint64_t end = address + size;
	length       = PageBase(end + PAGE_SIZE - 1) - base;
}

} // namespace

bool SafeRead(uint64_t address, void* dst, size_t size) {
	if (address == 0 || dst == nullptr || size == 0) {
		return false;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	SIZE_T read = 0;
	if (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(address), dst, size,
	                      &read) == 0) {
		return false;
	}
	return read == size;
#elif defined(__APPLE__)
	mach_vm_size_t out    = 0;
	const auto     status = mach_vm_read_overwrite(
	    mach_task_self(), static_cast<mach_vm_address_t>(address),
	    static_cast<mach_vm_size_t>(size), reinterpret_cast<mach_vm_address_t>(dst), &out);
	return status == KERN_SUCCESS && out == size;
#else
	iovec local {dst, size};
	iovec remote {reinterpret_cast<void*>(address), size};

	const auto moved = ::process_vm_readv(::getpid(), &local, 1, &remote, 1, 0);
	return moved >= 0 && static_cast<size_t>(moved) == size;
#endif
}

bool SafeWrite(uint64_t address, const void* src, size_t size) {
	if (address == 0 || src == nullptr || size == 0) {
		return false;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// WriteProcessMemory transparently relaxes page protection for the duration of the store,
	// which is what makes it usable on read-only guest data.
	SIZE_T written = 0;
	if (WriteProcessMemory(GetCurrentProcess(), reinterpret_cast<LPVOID>(address), src, size,
	                       &written) == 0) {
		return false;
	}
	FlushInstructionCache(GetCurrentProcess(), reinterpret_cast<LPCVOID>(address), size);
	return written == size;
#elif defined(__APPLE__)
	const auto status = mach_vm_write(mach_task_self(), static_cast<mach_vm_address_t>(address),
	                                  reinterpret_cast<vm_offset_t>(const_cast<void*>(src)),
	                                  static_cast<mach_msg_type_number_t>(size));
	return status == KERN_SUCCESS;
#else
	iovec local {const_cast<void*>(src), size};
	iovec remote {reinterpret_cast<void*>(address), size};

	const auto moved = ::process_vm_writev(::getpid(), &local, 1, &remote, 1, 0);
	return moved >= 0 && static_cast<size_t>(moved) == size;
#endif
}

bool WriteCode(uint64_t address, const void* src, size_t size) {
	if (address == 0 || src == nullptr || size == 0) {
		return false;
	}

	// The platform write already handles protection on Windows and macOS; only the Linux path
	// needs the mapping opened explicitly.
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS || defined(__APPLE__)
	if (!SafeWrite(address, src, size)) {
		return false;
	}
#else
	uint64_t base   = 0;
	uint64_t length = 0;
	PageSpan(address, size, base, length);

	using Common::VirtualMemory::Mode;
	Mode old_mode = Mode::NoAccess;
	if (!Common::VirtualMemory::Protect(base, length, Mode::ExecuteReadWrite, &old_mode)) {
		return false;
	}

	std::memcpy(reinterpret_cast<void*>(address), src, size);

	Common::VirtualMemory::Protect(base, length, old_mode);
#endif

	Common::VirtualMemory::FlushInstructionCache(address, size);
	return true;
}

} // namespace Debugger::Target
