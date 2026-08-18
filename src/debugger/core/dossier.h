#ifndef EMULATOR_SRC_DEBUGGER_CORE_DOSSIER_H_
#define EMULATOR_SRC_DEBUGGER_CORE_DOSSIER_H_

#include "common/common.h"
#include "debugger/core/types.h"

#include <string>

namespace Debugger::Dossier {

// Write everything the debugger's panels would have shown, as one file.
//
// Not every fatal error can be shown interactively: the renderer holds its own mutex across the
// frame it dies in, so when the failing thread is the one that presents, nothing else can draw
// the overlay. The dossier is what makes that case still useful — the report, the failing
// thread's registers, every guest thread, the module list, the GPU capture and memory around the
// failure, in a file that can be attached to a bug report.
//
// Returns the path written, or an empty string on failure.
std::string Write(const std::string& report, const Registers& regs);

} // namespace Debugger::Dossier

#endif // EMULATOR_SRC_DEBUGGER_CORE_DOSSIER_H_
