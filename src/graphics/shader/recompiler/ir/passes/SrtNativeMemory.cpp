// Executable memory for the native SRT walker (SrtNative.cpp). A file of its own so that the OS
// headers stay out of the IR headers, and so that the walker does not depend on the common
// library (its standalone test targets link only fmt).

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace Libs::Graphics::ShaderRecompiler::IR {

// Copies `size` bytes of code into fresh memory and makes it executable (read + execute only).
// Returns null on failure.
void* SrtNativeAllocateCode(const void* code, size_t size) {
#if defined(_WIN32)
	void* memory = VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
	if (memory == nullptr) {
		return nullptr;
	}
	std::memcpy(memory, code, size);
	DWORD old = 0;
	if (VirtualProtect(memory, size, PAGE_EXECUTE_READ, &old) == 0) {
		VirtualFree(memory, 0, MEM_RELEASE);
		return nullptr;
	}
	FlushInstructionCache(GetCurrentProcess(), memory, size);
	return memory;
#else
	void* memory = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (memory == MAP_FAILED) {
		return nullptr;
	}
	std::memcpy(memory, code, size);
	if (mprotect(memory, size, PROT_READ | PROT_EXEC) != 0) {
		munmap(memory, size);
		return nullptr;
	}
	return memory;
#endif
}

void SrtNativeFreeCode(void* memory, size_t size) {
	if (memory == nullptr) {
		return;
	}
#if defined(_WIN32)
	(void)size;
	VirtualFree(memory, 0, MEM_RELEASE);
#else
	munmap(memory, size);
#endif
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
