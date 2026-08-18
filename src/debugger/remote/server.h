#ifndef EMULATOR_SRC_DEBUGGER_REMOTE_SERVER_H_
#define EMULATOR_SRC_DEBUGGER_REMOTE_SERVER_H_

#include "common/common.h"

namespace Debugger::Remote {

bool Start();
void Stop();
bool IsRunning();

} // namespace Debugger::Remote

#endif // EMULATOR_SRC_DEBUGGER_REMOTE_SERVER_H_
