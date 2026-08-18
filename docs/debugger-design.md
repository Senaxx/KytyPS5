# KytyPS5 Debugger — Design

A design for a full-featured **in-emulator overlay debugger**. It is written against the code as
it exists today: guest code executes **natively** on host threads (there is no CPU interpreter or
JIT for guest x86-64 — only `src/loader/x64InstructionEmulator.cpp` for instructions the host
cannot run, and `src/loader/redZonePatcher.cpp` for stack-red-zone fixups). That single fact
shapes everything below: the CPU debugger is a *self-debugger* built on the host's trap
machinery, not on an instruction-stepping loop.

The debugger is an ImGui overlay inside the emulator window. There is no remote/serial protocol
and no console-log front end (§13).

---

## 0. Implementation status

Phases 0–2 of §11 are implemented and building. What exists today:

| Area | State |
|---|---|
| `hostException` trap types + priority handler chain + `SetGpr`/`SetInstructionPointer`/`SetFlagsRegister` | done — `src/common/hostException.{h,cpp}` |
| `--debugger` runtime gate, `KYTY_DEBUGGER` build option, launcher-visible config fields | done — `emulatorConfig`, `main.cpp`, `CMakeLists.txt` |
| Guest thread registry (`PthreadEnumerate`, `PthreadKillHostByOsId`) | done — `src/kernel/pthread.{h,cpp}` |
| Debug core: session, breakpoints (0xCC + shadow bytes + read overlay), park/resume, step into/over/out | done — `src/debugger/core/` |
| Fault-free memory access, code patching | done — `src/debugger/target/memory.cpp` |
| Thread suspend/context (Windows), signal-based stop (POSIX) | done — `src/debugger/target/threads.cpp` |
| Symbolizer, module list, symbol search, Zydis disassembler | done — `src/debugger/symbols/` |
| Overlay: toolbar, threads, registers, disassembly, call stack, breakpoints, memory, modules & symbols | done — `src/debugger/ui/overlay.cpp` |
| Input arbiter, halted-presentation path | done — `window.cpp`, `swapchain.cpp`, `videoOut.cpp` |
| GPU observation: shader registry (ISA/IR/SPIR-V), per-frame draw list, live counters | done — `src/debugger/target/graphics.cpp`, Graphics window |
| Break on fatal error: failing asserts and unhandled faults halt for inspection before the process exits | done — `Common::SetFatalHandler`, `Session::ReportFatal`, last-chance fault handler |
| Crash dossier: report, registers, threads, modules, GPU capture and memory written to `_logs` on any fatal | done — `src/debugger/core/dossier.cpp` |
| Regression tests | done — `tests/DebuggerCoreTests.cpp`, ctest target `debugger_core`, including a real breakpoint hit on a worker thread and a headless walk of every panel |

The GPU side is **observation only**. Shaders and draws are reported by the graphics code and
rendered by the overlay; nothing halts or reorders the GPU thread. Interactive PM4 debugging —
break on draw, step a packet, inspect bound resources at a draw — is the part still missing, and
it is missing deliberately: a PM4 breakpoint has to stop the GPU thread *between* submits and
never while it holds the command scheduler (§3.2, §4.4), and getting that wrong deadlocks
presentation along with the debugger meant to be showing it.

Not yet built: the watchdog (§3.2 item 4), command palette and Lua scripting (§3.5), HLE/kernel
panels (§4.2), memory scanner and watchpoints (§4.3), GPU halting and the draw-state/resource
inspector (§4.4), shader invocation tracing and live reload (§4.5), trace rings and the crash
dossier (§4.6), record/replay (§7), and the `.eh_frame` unwinder (§5.4 — the shipped backtrace
is the frame-pointer walk, which truncates in FPO code).

One platform difference is worth knowing before touching this code: **the reported instruction
pointer for a breakpoint trap is not the same on Windows and POSIX.** The Windows kernel backs
`CONTEXT.Rip` up onto the `int3`; a POSIX `SIGTRAP` reports the address after it. `ExceptionInfo`
normalises this so `rip` always means the trap instruction, and `Session::OnTrap` does no
arithmetic of its own. Assuming the POSIX convention everywhere silently broke every breakpoint
on Windows once already, so the regression test asserts the contract directly rather than
inferring it.

Known limitations of the current implementation:

- **Windows is the only build-verified platform.** The POSIX stop path (`SIGUSR2` → `raise(SIGTRAP)`
  → park) is written but has not been compiled or run. Note that the normalisation above means
  the POSIX branch is the *untested* side of a difference that has already caused one bug.
- **ImGui has no docking** on the vendored `1.92.9b` master build, so the panels are plain
  floating windows. Switching the submodule to the `docking` branch (§3.1) is still the
  recommendation but has not been done — it is a submodule change worth making deliberately.
- **A fatal error inside the renderer cannot be shown interactively.** `Presenter::PrepareFrame`
  holds `renderer.GetMutex()` across the frame, and `Presenter::Present` needs the same mutex, so
  when the failing thread is the one presenting, no other thread can ever draw the overlay while
  it is parked. Fatal halts therefore watch for the overlay drawing and release themselves after
  five seconds if it never does — a debugger that hangs the emulator is worse than one that
  misses a stop. The crash dossier covers that case instead. Making it interactive would mean
  restructuring the presenter so the overlay can be driven independently of the frame lock.
- Registers at a fatal stop are captured on Windows only; elsewhere the stop reports the message
  and thread but no registers.
- **Step-out relies on a heuristic.** With no `.eh_frame` unwinder yet, it finds a return address
  by scanning the stack for a value that decodes as code and is immediately preceded by a call.
  It can pick the wrong frame in deeply recursive code; what it will not do is plant a
  breakpoint on an address that fails that check, which is what made it fatal before.
- **The trap handler takes `g_breakpoints_mutex`** for hit counts and step-over bookkeeping.
  The armed-breakpoint lookup itself is lock-free, and the UI never holds that mutex while a
  thread is parked, so the wait is bounded — but it is not async-signal-safe on POSIX and
  should move to a lock-free structure before the POSIX path is trusted.

---

## 1. What already exists (and should be reused, not rebuilt)

| Facility | Location | Reuse for |
|---|---|---|
| Vectored/POSIX fault handler | `src/common/hostException.{h,cpp}` | breakpoint & watchpoint traps |
| Guest thread registry primitives | `src/kernel/pthread.h` (`PthreadGetHostThreadId`, `PthreadGetUniqueId`, `PthreadGetGuestStack`, `PthreadGetname`) | thread list, stop-the-world |
| Guest futex equivalent | `src/kernel/syncOnAddress.cpp` | scheduler control, wait-for graph, record/replay (§7) |
| Module/symbol database | `src/loader/{runtimeLinker,symbolDatabase}.h` (`FindProgramByAddr`, `SymbolRecord::dbg_name`) | symbolization, module list |
| Guest stack walk | `RuntimeLinker::StackTrace`, `src/common/debug.h`, `sysDbg.h` | backtraces |
| Guest memory map & backing access | `src/kernel/memory.h` (`KernelVirtualQuery`, `TryReadBacking`, `WriteBacking`, `RegisterCallbacks`) | memory view, alloc tracking, checkpoints |
| Page-protection tracking | `src/graphics/host_gpu/{pageManager,memoryTracker}.cpp` | watchpoints, dirty-page checkpoints (and a conflict to resolve — §6.3) |
| **Zydis** (x86-64 decoder) + **Xbyak** | already linked, `CMakeLists.txt:295` | disassembly view, instruction length for step-over |
| **ImGui 1.92.9b + Vulkan backend** | `src/graphics/presentation/imeOverlay.cpp` | the overlay — same pattern (but see §3.1) |
| `imgui_impl_sdl2.cpp` | vendored in `3rdparty/imgui/backends/`, **currently unused** | overlay mouse/keyboard input |
| PM4 command processor | `src/graphics/guest_gpu/command_processor/` | GPU breakpoints / stepping |
| GPU thread with command queue | `src/graphics/guest_gpu/graphicsRun.h` (`SendCommandSync`, `IsGpuThread`) | halting/resuming the GPU |
| Shader pipeline (ISA→IR→SPIR-V) | `src/graphics/shader/recompiler/` | shader debugger views |
| RenderDoc trigger, buffer/shader dumps | `renderDoc.cpp`, `--command-buffer-dump`, `--shader-log-direction` | frame capture integration |
| Tracy profiler | `src/common/profiler.h` | **do not duplicate** — timing stays in Tracy |
| Subsystem lifecycle + CLI + launcher config | `subsystems.h`, `emulatorConfig.h`, `src/main.cpp`, `src/launcher/` | wiring and options |

The two biggest levers are already in the tree: **Zydis** (free disassembler) and **an ImGui
Vulkan overlay** (free UI). Neither needs a new dependency.

---

## 2. Architecture

```
   ┌───────────────────────────────┐        ┌──────────────────────────────────────────┐
   │  src/debugger/ui  (ImGui)     │        │            src/debugger/core             │
   │  overlay + panels             │◄──────►│  DebugSession   (owns everything)        │
   │  input arbiter                │        │  ExecutionController (run/pause/step)    │
   │  layout persistence           │        │  BreakpointManager                       │
   └───────────────────────────────┘        │  ExpressionEvaluator                     │
   ┌───────────────────────────────┐        │  EventBus + TraceRings                   │
   │  src/debugger/script (Lua)    │◄──────►│  Symbolizer / Disassembler (Zydis)       │
   └───────────────────────────────┘        │  Unwinder / Recorder (§7)                │
                                            └────┬───────┬───────┬────────┬────────────┘
                                            CpuTarget MemTarget GpuTarget ShaderTarget
                                                 │       │        │        │
                                            pthread.cpp memory.cpp pm4*.cpp recompiler/
```

**Rules that keep this maintainable:**

1. The core is **headless and UI-free** — it links without Vulkan, ImGui, or SDL, so it is
   unit-testable under `tests/` with ctest. The overlay is a client of it, not the other way
   round. (This is worth preserving even though the overlay is the only front end: it is what
   makes the debugger testable in CI.)
2. Targets talk to the emulator through **narrow, explicit interfaces** (a `DebugHooks` header
   per subsystem). The debugger never reaches into private state.
3. Every hot-path hook is a single relaxed atomic load behind `KYTY_UNLIKELY`, and the whole
   module compiles out under `-DKYTY_DEBUGGER=OFF`.

### 2.1 Layout

```
src/debugger/
  core/    session.{h,cpp}  execution.{h,cpp}  breakpoints.{h,cpp}
           expression.{h,cpp}  events.{h,cpp}  trace.{h,cpp}  recorder.{h,cpp}
  target/  cpu.{h,cpp}  memory.{h,cpp}  hle.{h,cpp}  gpu.{h,cpp}  shader.{h,cpp}
  symbols/ symbolizer.{h,cpp}  disasm.{h,cpp}  unwind.{h,cpp}
  script/  lua bindings
  ui/      overlay.{h,cpp}  input.{h,cpp}  layout.{h,cpp}  panel_*.cpp
```

Two CMake targets: `kyty_debugger_core` (no graphics deps, unit-tested) and `kyty_debugger_ui`
(links ImGui/Vulkan/SDL). Option `KYTY_DEBUGGER` (default `ON`).

---

## 3. The overlay

### 3.1 ImGui prerequisites — two changes needed up front

**Docking.** The vendored ImGui is `1.92.9b` on **master**, where docking is absent
(`ImGuiConfigFlags_DockingEnable` and `DockSpace` appear only in commented-out lines of
`imgui.h`). A fifteen-panel debugger without docking means hand-rolling splitters and window
management. Move the `3rdparty/imgui` submodule to the **`docking` branch** at the matching
`v1.92.x` tag: same repository, same `imgui_impl_vulkan` API, and the docking branch is a strict
superset — `imeOverlay.cpp` keeps working unchanged. Enable `DockingEnable`; leave
multi-viewport (tear-off panels into real OS windows) **off** initially, since it adds swapchain
management for secondary windows.

**Input.** `imeOverlay.cpp` uses only `imgui_impl_vulkan.h` and feeds ImGui no mouse or keyboard
— fine for a passive IME overlay, useless for a debugger. `imgui_impl_sdl2.cpp` is already
vendored in `3rdparty/imgui/backends/` but not compiled. Add it to the build and route SDL
events into it from the window event loop (`src/graphics/presentation/window/window.cpp`).

**Fonts.** The disassembly, hex and register panels need a monospace font. Ship one embedded
(as the SPIR-V blobs are, via `src/embed_spirv.cmake`'s pattern) rather than depending on a
system font.

### 3.2 Presentation while the guest is halted — the central invariant

When guest threads are frozen the guest stops calling flip, so nothing drives presentation. In a
design where the overlay is the *only* front end, this is not a detail — it is the difference
between a debugger and a hang.

Requirements:

1. The overlay is driven from the presenter loop (`WindowRun()`, main thread), which is already
   independent of the guest thread and the GPU thread.
2. While halted, the presenter **re-presents the last frame** every iteration with the overlay
   composited on top, using its own command pool and its own swapchain acquire. It must not wait
   on any guest-produced fence.
3. A PM4 breakpoint halts the GPU thread **between** submits — never while it holds the
   `CommandScheduler`. `ExecutionController` exposes "GPU halted at a safe point" as a distinct
   state from "GPU halted", and only the former is enterable.
4. **Watchdog:** if no frame has presented for N seconds, force-halt and raise the overlay
   automatically. This is what turns "the emulator hung" into "the debugger is showing you where
   it hung", and it is the main mitigation for the one real weakness of an overlay-only design
   (§13).

### 3.3 Input arbitration

Host keyboard and mouse currently feed the guest through `HostInputKey` / `HostInputMouseButton`
(`window/hostInput.h`) and the IME via `ProcessImeInput`. With the debugger visible these
conflict. Add a small arbiter in the SDL event path with a fixed priority:

```
debugger overlay (if visible and wants capture)  →  IME overlay  →  guest pad/keyboard
```

`ImGuiIO::WantCaptureKeyboard` / `WantCaptureMouse` decide capture. Toggle key: `F1` or backtick
— **not** F11/Alt+Enter, which fullscreen already owns (commit 566a450). Gamepad input keeps
flowing to the guest unless explicitly grabbed, so you can hold the debugger open while playing.

### 3.4 Panels

Execution: **Threads** · **Disassembly** (Zydis, symbol-annotated, jump arrows, follow-RIP) ·
**Registers** (GPR/RFLAGS/XMM/YMM/FS base, editable) · **Call stack** · **Breakpoints** ·
**Watch**.

Memory: **Memory map** · **Hex editor** · **Value scanner** · **Allocations**.

Emulator: **Modules & symbols** · **HLE calls** (live trace, counts, unimplemented) ·
**Kernel objects** (threads, mutexes, semaphores, event flags, event queues, fds) ·
**Event log**.

Graphics: **Draw list** · **GPU state** (with diff) · **Resources** (texture preview) ·
**Shaders** (four-view).

Time travel: **Timeline** (§7).

Layout persistence via ImGui's `.ini` in the emulator's working directory, with a "reset layout"
command and 2–3 built-in presets (CPU debugging / GPU debugging / compatibility triage).

### 3.5 Command palette and scripting

A command bar (`Ctrl+P`) over the same command registry the Lua bindings use, so every action is
scriptable and every script function is reachable by hand:

```lua
bp.hle("sceKernelMapDirectMemory", function(a) log(("len=%x"):format(a.rsi)) end)
bp.mem_write(0x00800000, 4, function(w) trace.dump("mem.json") end)
gpu.break_on_draw(1892)
```

Scripted conditional breakpoints are what make an emulator debugger productive — you can attach
a condition to a hot HLE function without paying a trap per call.

---

## 4. Feature set by domain

### 4.1 Guest CPU

- Run / pause / step-into / step-over / step-out / run-to-cursor, per thread and all-threads.
- Software breakpoints (`0xCC`), hardware breakpoints, conditional + hit-count + one-shot.
- Breakpoints on module-relative addresses (`eboot.bin+0x1234`) so they survive reload.
- Register view/edit including FS base (guest TLS lives there).
- Disassembly with symbol annotation and cross-references.
- Backtrace with frame navigation. Locals are out of scope — no DWARF in retail ELFs.
- Break on: unhandled exception, illegal instruction, unresolved-NID stub call, thread
  create/exit, module load.
- Runtime byte patching, with **export in the format `src/loader/gamePatch.cpp` already
  consumes**.

### 4.2 HLE / kernel

The domain where an emulator debugger beats a general-purpose one, and the codebase is already
most of the way there: every HLE function self-registers through `LIB_FUNC` in `src/libs/libs.h`
with both its NID and its C++ `dbg_name`.

- **Change `LIB_ADD` to assign each function a dense id and register it in a global table**
  (`{nid, dbg_name, library, module, address, flags}`). Then, at runtime:
  - per-function / per-module / per-library trace toggles — replacing today's per-translation-unit
    `thread_local bool PRINT_NAME_ENABLED`, which cannot be controlled at runtime;
  - breakpoints on HLE entry/exit with SysV argument capture (RDI, RSI, RDX, RCX, R8, R9, XMM0–7);
  - call counts, and a **"stubbed / unimplemented calls" panel** — the fastest possible answer to
    "why does this game not boot".
- Kernel object inspectors backed by the existing implementations: threads (`pthread.cpp`),
  mutexes/rwlocks/condvars, semaphores, event flags, event queues, fds and mounts, memory pools.
- **Deadlock detector**: build a wait-for graph (thread → object → owner) from the sync
  primitives and `syncOnAddress`, run cycle detection on a timer, report the cycle with
  backtraces. Emulator hangs are usually this, and today they are diagnosed by reading logs.
- Guest `printf` capture (`guestPrintf.cpp`) into the event log.

### 4.3 Memory

- Memory map from `KernelVirtualQuery` + `KernelSetVirtualRangeName`: range, protection, type,
  direct/flexible/pooled/stack flags, name, owning module.
- Hex viewer/editor over `TryReadBacking` / `WriteBacking`, with type overlays and pointer
  following.
- Watchpoints (read/write/access) — see §6.3.
- Allocation and leak tracking via the already-present
  `Memory::RegisterCallbacks(alloc_func, free_func)`.
- **Value scanner** (exact / unknown-initial / changed / increased), with results exportable
  straight into a game patch.
- Page-fault heat map: which guest pages the GPU tracker invalidates per frame — doubles as a
  performance tool for CPU↔GPU sync.

### 4.4 GPU / PM4

- Halt/step the GPU thread: one PM4 packet, one draw, one submit, one frame.
- Breakpoints on opcode/packet type, draw index in a frame, shader hash, writes to a named GPU
  register, or any access to a guest address range.
- Draw-call browser: per draw, the captured `HW::Context` / `HW::Shader` / `HW::UserConfig`,
  bound render targets, textures, samplers, buffers, viewport/scissor, blend and depth state.
  `src/graphics/host_gpu/renderer/debug.h` (`hw_print`, `rt_print`, `uc_print`) already produces
  most of this as text — reuse it and add a structured form.
- **State diff between consecutive draws** — the fastest way to find "draw 41 renders, draw 42
  does not".
- Resource inspection with in-overlay texture preview (readback already exists for the
  `_Textures` dump path) and buffer hex.
- "Capture next frame": triggers RenderDoc (`renderDoc.cpp`) and the PM4 dump together into one
  timestamped folder.
- PM4 stream save/replay — command buffers are already dumpable; a replay path makes renderer
  bugs reproducible without the game.

### 4.5 Shader

- Four synchronized views per shader: raw bytes → decoded RDNA2 ISA (`ShaderDecoder`) → IR
  (`ShaderIR`, which already has `ShaderIRLog.cpp`) → SPIR-V disassembly (spirv-tools is linked;
  `spvBinaryToText`).
- Per-stage binding table (`BindingLayout`, `ResourceTracking`) with the descriptor values
  actually bound for the selected draw.
- **Invocation debugging without vendor support**: instrument the emitted SPIR-V to write a
  per-instruction value trace into a storage buffer, gated on a selected invocation
  (`gl_FragCoord == picked pixel`, or a chosen thread id), read back, and display the value
  history. This is the realistic equivalent of shader single-stepping. The existing
  `--spirv-debug-printf` is the cheap version of the same idea and should stay.
- **Live shader replace/reload**: load SPIR-V for a shader hash from `_Shaders/`, invalidate
  `pipelineCache`, re-render. Iterating on a recompiler bug without a rebuild is a large win
  given how much of this project *is* the recompiler.

### 4.6 Trace and the crash dossier

Per-thread lock-free ring buffers recording HLE calls, GPU submits, memory map changes, thread
state transitions, page faults and exceptions. Always on (cheap), dumped on demand.

> **Crash dossier** — on an unhandled fault, write one folder containing registers, symbolized
> guest backtrace, module map, the last N HLE calls per thread, the last N GPU submits, a hex
> dump around the faulting address, the current frame's draw list, the memory map, and the
> config. The README asks users to "attach the complete log file"; this replaces that with a
> self-contained report. It is also the fallback for the cases an in-process overlay cannot
> cover (§13).

---

## 5. How the CPU debugger actually works

### 5.1 Trap plumbing — the prerequisite change

`Common::HostException` today supports **exactly one handler** (`InstallHandler`, CAS-guarded
single slot) and classifies only `AccessViolation` and `IllegalInstruction`. Needed:

1. **New exception types** — `Breakpoint` (Windows `EXCEPTION_BREAKPOINT` 0x80000003 / `SIGTRAP`)
   and `SingleStep` (`EXCEPTION_SINGLE_STEP` 0x80000004 / `SIGTRAP` with TF). Add `SIGTRAP` to
   the POSIX `sigaction` set in `InstallHandler`.
2. **A priority handler chain** instead of one slot:
   `Debugger(0) → GpuPageFault(10) → GuestMemory(20) → X64Emulator(30)`. First handler to claim
   wins. The debugger must claim *only* trap and watchpoint faults and must never swallow the GPU
   tracker's page faults — getting this wrong silently breaks CPU↔GPU coherency.
3. **Full register context** in `ExceptionInfo` (GPRs only today): RIP, RFLAGS, XMM/YMM, FS/GS
   base. `native_context` is already mutable, which is what makes stepping and RIP rewind work.

Small, self-contained changes to `hostException.{h,cpp}`; land them first.

### 5.2 Stopping guest threads

A **hybrid** model, because neither half suffices:

- **Cooperative safepoints** — a `Debug::Poll()` (one relaxed atomic load) at known-safe points:
  HLE entry/exit, PM4 submit, vblank, thread create/exit, guest signal delivery. Stops here have
  clean, consistent state and are the default.
- **Forced async stop** — for a thread spinning in guest code:
  - Windows: `SuspendThread` + `GetThreadContext` / `SetThreadContext`.
  - Linux/macOS: `pthread_kill` with a dedicated signal whose handler parks the thread on a
    condvar and publishes its `ucontext_t`. **Not `SIGUSR1` on macOS** — that is already the
    guest signal-dispatch interrupt, and `hostException.cpp` deliberately blocks it inside the
    fault handler. Add the debugger's stop signal to that same block mask.

Hard rule: while any guest thread is suspended, the debugger must not acquire a lock a guest
thread could hold (the emulator heap, the renderer mutex, the linker mutex). Memory and symbol
reads go through snapshots or raw pointers. Violating this deadlocks the process, and it will
happen the first time someone calls `LOGF` from a stopped state.

### 5.3 Breakpoints

- **Software**: write `0xCC`, keep the original byte in a shadow map. Two requirements: make the
  page writable via `Memory::ProtectGuestMemory` and restore protection after; and **overlay
  original bytes in every memory read path** so the disassembler, the hex panel and any guest
  self-checksum see unpatched code.
- **Resume over a breakpoint**: restore byte → set TF → single-step one instruction → re-arm →
  clear TF. Must be per-thread and race-free when several threads sit on one address; keep the
  "stepping over bp @ addr" state per thread in `ExecutionController`.
- **Hardware breakpoints/watchpoints**: DR0–DR3 via `CONTEXT_DEBUG_REGISTERS` on Windows (4 max,
  set on *every* guest thread). Linux userspace cannot write DRs without `ptrace`, so
  **watchpoints there use page protection** (§6.3) — the same mechanism the GPU tracker already
  uses. Don't plan a `ptrace` helper process; the protection path is portable and reuses existing
  code.
- **Step-over/step-out**: Zydis gives instruction length for the temporary breakpoint after a
  `call`; step-out uses the unwinder's return address.

### 5.4 Symbolization and unwinding

- Address → module via `RuntimeLinker::FindProgramByAddr`; address → symbol via the per-program
  `SymbolDatabase` plus the HLE table (§4.2). Guest exports are NIDs — keep an optional NID→name
  map so known ones display as real names.
- Unwinding: `RuntimeLinker::StackTrace` (RBP chain) is the starting point, but guest code is
  frame-pointer-omitted and it will truncate. Implement a **`.eh_frame` CFI unwinder** — PS5 ELFs
  carry `.eh_frame` — with heuristic stack scanning as fallback. Accurate backtraces are the
  difference between a usable and a decorative debugger.

---

## 6. Hazards to design around

### 6.1 Presentation while halted
§3.2. The single most likely source of "the debugger hung the emulator".

### 6.2 Logging and allocation from a stopped state
The core needs its own bump allocator and a lock-free log sink. It must not call `LOGF`,
`spdlog`, or the emulator heap while guest threads are suspended.

### 6.3 Page-protection arbitration
`pageManager` / `memoryTracker` already mutate guest page protection for GPU coherency, and
`Memory::ProtectGuestHostMemory` exists specifically for "transient watch state that does not
change semantic protection". Debugger watchpoints, breakpoint page unprotection, and
checkpoint dirty-tracking (§7) all collide with that.

**Required:** a single arbiter owning host protection for guest pages, keyed per page by a reason
mask — `{guest_semantic, gpu_tracking, debugger_watch, debugger_patch, checkpoint_dirty}` —
computing effective protection as the intersection and routing faults to the right owner. Every
existing protection call site moves behind it. This is the one piece of the design that touches
existing code meaningfully; done wrong, the debugger corrupts GPU memory tracking in ways that
look like random rendering bugs.

### 6.4 Input arbitration
§3.3. Without it, typing an address into the debugger also presses buttons in the game.

---

## 7. Deterministic record/replay and reverse execution

I previously called full deterministic record/replay intractable here. That was wrong, and the
reason is worth stating precisely, because it changes the plan: **this emulator implements the
guest's synchronization primitives itself.** Guest mutexes, condvars, rwlocks, semaphores, event
flags and the futex equivalent (`syncOnAddress.cpp`) are all HLE code. Almost all cross-thread
ordering that matters therefore flows through code you control. That is a materially better
starting position than a general-purpose native recorder like `rr`, which has to infer all of it
from outside the process.

What follows is the design space, with honest costs.

### 7.1 What has to be reproduced

| Source of nondeterminism | Difficulty | Handling |
|---|---|---|
| Controller / IME / network / file I/O | easy | record at the HLE boundary, replay from the log |
| Clock reads, `rdtsc`, timers, vblank | easy | record returned values; `rdtsc` needs a trap or a patched read path |
| Allocator-returned guest addresses | easy | record, or make guest address assignment deterministic |
| **Host pointers handed to the guest as opaque handles** | easy but *easy to miss* | `Pthread` is literally `PthreadPrivate*`; these values are allocation- and ASLR-dependent, leak into guest memory, and must be reproduced or virtualized behind a stable handle table |
| GPU → guest-memory writes (EOP fences, queries, readback) | moderate | record as *device input*: (memory range, value, delivery point). Fences and queries are tiny; surface readback is bulky but rare |
| **Thread interleaving on shared memory** | **the hard one** | §7.2 |

### 7.2 The hard one, and three ways to solve it

Guest threads run natively on multiple host cores. Any lock-free algorithm in the game — a UE
task graph, a job queue, a spinlock fast path — produces an interleaving that is not reproduced
by replaying boundary events alone. Three tiers, in increasing cost and fidelity:

**Tier 1 — boundary recording with divergence detection.** Record everything in the table above
except interleaving. Replay feeds recorded values back. This reproduces every input, timing, file
and GPU-feedback bug, and every bug in single-threaded or lock-based guest code. It will diverge
on genuine lock-free races.

The thing that makes Tier 1 *honest* rather than misleading: at every safepoint, hash the
recording thread's registers plus the dirty-page set, and store the hash. On replay, compare. On
mismatch, stop and report "replay diverged at HLE call #412,338" instead of silently showing a
different execution. An approximate recorder with divergence detection is a real tool; one
without it is a liability.

*Cost:* low. Recording overhead is a few percent. Trace size is modest (kilobytes per frame,
dominated by any surface readback).

**Tier 2 — deterministic scheduling (true determinism).** Run **one guest thread at a time** under
an emulator-owned scheduler, and record the schedule (thread id + switch point). Because the
emulator owns `PthreadCreate` and every sync primitive, the switch points are already in your
code. Replay reproduces the schedule exactly, so every interleaving is reproduced, races included.

The sub-problem that decides whether this works: **forced preemption at a deterministic point.**
If you only switch at instrumented points, a thread spinning on a plain atomic never yields, and
because no other thread runs, the flag it spins on can never be set — a deadlock the recording
itself creates. Options:

- *PMU instruction/branch counting* (what `rr` does: retired conditional branches + RIP).
  Straightforward on Linux via `perf_event_open`. On **Windows — the project's primary platform —
  it needs a kernel driver or Intel PT plumbing.** This is the genuine blocker, and it is a
  blocker of engineering scale, not of possibility.
- *Cooperative + single-step escalation* (driver-free, and my recommendation). Default to
  switching at instrumented points. If a thread exceeds a wall-clock budget without reaching one,
  escalate: set TF and single-step it, counting instructions, until the next instrumented point,
  recording the count. Replay re-runs exactly that instruction count. Single-stepping costs
  roughly a microsecond per instruction, so this is only viable in short bursts — which matches
  reality, since well-written engine spinlocks back off into a futex wait (an HLE call) within a
  bounded number of iterations. A game that spins unboundedly on a plain atomic degrades to
  single-step speed, and the debugger should say so out loud.
- *Breakpoint + hit count* as an alternative encoding of the same idea: record (RIP, Nth hit
  since the last switch), replay with a hardware breakpoint. Exact, driver-free, but pays a trap
  per hit of a possibly hot address.

*Cost:* recording loses guest parallelism, so expect a large slowdown on multi-threaded titles —
this is a debugging mode, not a play mode. Replay runs at similar speed. The GPU is handled as a
recorded device (§7.1), so it does not need to be deterministic itself.

**Tier 3 — value-level recording** (record the value of every guest memory load, as Microsoft's
TTD does). Deterministic by construction, indifferent to interleaving, ~10–50× slowdown and huge
traces. It requires instrumenting every guest instruction, i.e. a dynamic binary translation
engine — **the one thing this project deliberately does not have.** *This* is where "we'd have to
build a whole new execution subsystem" is the accurate statement, and it is why Tier 3 is out.

### 7.3 Reverse execution — what people actually want

Reverse-step and reverse-continue don't require running backwards. They require **checkpoints +
deterministic replay forward from the nearest one**:

- Periodic full guest-memory checkpoints, made incremental via dirty-page tracking — the
  page-protection machinery already exists (§6.3 makes it shareable).
- "Step back one instruction" = restore the nearest checkpoint, replay forward to (current − 1).
- The **Timeline panel** scrubs across checkpoints, with markers for frames, HLE calls, GPU
  submits and breakpoint hits.

This works on Tier 1 traces too (within a divergence-free window), which is why it is worth
building the checkpoint machinery early — it pays off before Tier 2 exists.

### 7.4 Recommendation

Build **Tier 1 + checkpoints + reverse execution** first: it is a few weeks, it delivers the
reverse-debugging UX, and it covers input, timing, GPU-feedback and lock-based bugs. Then add
**Tier 2** behind a `--record-deterministic` flag for the races Tier 1 flags as divergent. Skip
Tier 3.

The correct summary is therefore: full deterministic record/replay is *achievable* here and is
made easier than usual by HLE-owned synchronization; what it costs is guest parallelism during
recording, plus one genuinely hard sub-problem (deterministic preemption on Windows) for which
the driver-free escalation path above is the answer.

---

## 8. Configuration

### 8.1 Two independent gates

- **Build-time `KYTY_DEBUGGER`** (CMake, default `ON`) — when `OFF`, the whole module and every
  hook compile out to nothing. For shipping a minimal build.
- **Runtime `--debugger`** — the subsystem is **inert unless this flag is passed.** Without it,
  no exception handler is chained, no hooks arm, no ImGui context is created, no memory is
  reserved for trace rings or checkpoints, and the cost is the atomic loads described in §9 (all
  reading `false`). This is the flag you asked for and it is the master switch: every other
  `--debugger-*` option below implies it, so `--debugger-break=main` alone is enough.

### 8.2 Options

Add to `Config::ConfigOptions` (`src/common/emulatorConfig.h`), parse in `src/main.cpp` alongside
the existing flags, and surface in the launcher's `configurationEditDialog.cpp` (a "Debugger"
section with an enable checkbox that gates the rest of the controls):

```
--debugger                     enable the debugger (master switch; implied by all options below)
--debugger-ui                  show the overlay at startup (default: hidden, F1 to open)
--debugger-break-entry         halt at guest entry
--debugger-break=<sym|addr>    initial breakpoint (repeatable)
--debugger-trace=hle,gpu,mem   enable trace rings
--debugger-script=<file.lua>   run a script at startup
--debugger-record=<dir>        record a session (§7)
--debugger-replay=<dir>        replay a recorded session
--debugger-dossier-dir=<dir>   crash dossier output (default _logs)
```

The crash dossier (§4.6) is the one piece worth considering enabling without `--debugger`, since
its value is in reports from users who were not debugging. Suggested: dossier-on-crash defaults
to on, everything else requires the flag.

Initialize as `Debugger::Lifecycle` in `Emulator::Init` (`src/emulator.cpp:107`), after
`Config`/`Log` and **before** `Memory` and `Graphics`, so the exception chain and the protection
arbiter exist before anything registers a fault handler. The lifecycle's `initialize` checks
`Config::DebuggerEnabled()` first and returns immediately when false — the ordering constraint
applies only to the enabled path.

---

## 9. Performance

| Hook | Cost when off | Mitigation |
|---|---|---|
| `Debug::Poll()` at safepoints | one relaxed load | inline, `KYTY_UNLIKELY` |
| HLE entry hook | one relaxed load per call | per-function flag byte, indexed by the id from `LIB_ADD` |
| PM4 per-packet hook | one relaxed load per packet | gate at submit granularity unless packet stepping is armed; riskiest hot path — measure it |
| Trace rings | ~20 ns/record | per-thread, lock-free, power-of-two ring |
| Draw-state capture | significant | only while the GPU panel is open |
| Overlay rendering | one extra render pass | only while visible |
| Recording (Tier 1) | low single-digit % | boundary only |
| Recording (Tier 2) | large — no guest parallelism | opt-in flag, documented as a debugging mode |

"Off" above means **built in but launched without `--debugger`** (§8.1) — the realistic default
for every user. Ship a benchmark comparing three configurations on a booting title:
`KYTY_DEBUGGER=OFF`, built-in-but-no-`--debugger`, and `--debugger` active. The middle column
must be within noise of the first, or the gates are in the wrong place.

---

## 10. Testing

The headless core is testable with the existing ctest setup (`tests/`, target `kyty_tests`):

- breakpoint set/clear/overlay-read against a fake target;
- resume-over-breakpoint state machine, including two threads on one address;
- expression evaluator;
- `.eh_frame` unwinder against synthetic stacks;
- **protection arbiter**: property test that effective protection always equals the intersection
  of active reasons, and that removing a debugger watch restores the GPU tracker's state;
- **record/replay**: record a synthetic HLE event stream, replay it, assert bit-identical
  checkpoint hashes — and assert that an *injected* divergence is actually detected.

---

## 11. Roadmap

| Phase | Content | Why here |
|---|---|---|
| **0. Plumbing** ✅ | `hostException` trap types + handler chain; guest thread registry; core skeleton; `KYTY_DEBUGGER` build flag and the `--debugger` runtime gate | everything depends on it; touches existing code, so land it alone. (The protection arbiter, §6.3, is deferred with watchpoints.) |
| **1. Overlay shell** ✅ | input arbiter; halted-presentation path; panel layout. (Docking branch and watchdog deferred — see §0.) | the front end is the product here — get the hard part (§3.2) working before there is anything to show |
| **2. CPU debugging** ✅ | execution control, breakpoints, disassembly, registers, stack, memory panels, symbolizer | the core debugging loop |
| **3. Trace + dossier + HLE** | trace rings, HLE function table and panels, kernel objects, deadlock detector, crash dossier | improves every bug report; mostly independent of phase 2 |
| **4. GPU** | PM4 stepping, draw browser, state diff, resource inspector, capture integration | where this project spends most of its debugging time |
| **5. Record/replay Tier 1 + reverse execution** | boundary recorder, checkpoints, divergence detection, timeline panel | the reverse-debugging UX, at moderate cost |
| **6. Shader** | four-view inspector, invocation trace, live reload | closes the loop on recompiler work |
| **7. Record/replay Tier 2** | deterministic scheduler, single-step escalation | true determinism, for the races phase 5 flags |
| **8. Extras** | value scanner + patch export, Lua scripting surface | unblocked by everything above |

Phases 1–3 are independently useful and can ship without 4–8.

---

## 12. Open decisions

1. **ImGui docking branch** — recommended (§3.1), but it is a submodule change affecting an
   existing overlay. Alternative is hand-rolled splitters, which costs more in the long run.
2. **Multi-viewport (tear-off panels into OS windows)** — very good for a debugger on a second
   monitor, but needs secondary swapchains. Deferred, not rejected.
3. **Lua** as the scripting language — MIT, ~200 KB. The alternative is a fixed command set with
   no scripting.

## 13. Deliberately out of scope

- **Remote/serial debugging (GDB stub) and a console front end.** Excluded by design decision:
  the debugger is an overlay. One consequence worth naming — an in-process overlay cannot debug a
  crash that takes down the window or the renderer, and cannot debug startup before Vulkan is up.
  The mitigations are the watchdog (§3.2) and the crash dossier (§4.6), which cover the realistic
  cases. If external tooling (IDA/Ghidra) is ever wanted, an RSP stub is a self-contained add-on
  against the same core — but it is not part of this plan.
- **Value-level record/replay (Tier 3)** — requires a DBT engine the project does not have (§7.2).
- **Source-level debugging of guest code** — no DWARF in retail ELFs.
- **Profiling and timeline-of-time views** — Tracy already covers this; the debugger links to it,
  it does not reimplement it.
- **Debugging the emulator's own C++** — stays with LLDB/Visual Studio on the host process.
