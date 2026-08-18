# External debugger

Kyty can expose its debugger core to a separate desktop application. This is a cooperative
attach, not process injection: guest memory, thread suspension, breakpoints and GPU observations
still pass through the in-process debugger's safety rules.

## Starting it

In the launcher, enable **External debugger** for the global or per-game configuration. The
launcher passes `--debugger-server`, starts `kyty_debugger`, and the debugger automatically
attaches when the emulator publishes its session.

The **Debugger overlay** option is independent. Either interface or both can be enabled.

For a command-line launch, start the emulator with `--debugger-server`, then run
`kyty_debugger.exe` from the same install directory.

## Capabilities

- discover and attach to local Kyty processes;
- show running/paused state and graphics counters;
- pause, continue, and step guest threads;
- inspect stopped registers, call stacks, nearby disassembly, and breakpoints;
- add and remove execution breakpoints;
- perform bounded guest-memory reads and writes;
- list loaded modules and search symbols globally or within one module;
- list captured vertex, pixel and compute shaders;
- inspect RDNA2 ISA, recompiler IR, SPIR-V, and decoded captured shader resources, preview a
  selected live image binding, and save shader dumps;
- show the previous frame's draw and dispatch commands;
- save an exact top-level PM4 command bundle for a selected submit, including the graphics and
  constant streams, draw/dispatch summaries, shader identities, and matching resource events;
- inspect a bounded history of image uploads and GPU writes;
- trace a guest image range across live and retired native aliases, ownership changes, uploads,
  downloads, clears, sampled reads, buffer/CPU invalidations, shader hashes, submits, metadata and
  stencil ranges. The trace is a bounded metadata ring and does not retain texture payloads;
- request a scheduler-safe thumbnail of a selected live color image;
- inspect and filter guest file opens, reads, writes, positional I/O, seeks, and closes, including
  guest/host paths, offsets, byte counts, results, threads, calling modules, and caller addresses;
- browse a session-wide distinct-file inventory with operation and byte totals even after detailed
  events rotate out of the recent-event view. All debugger tables support click-to-sort columns.

The endpoint is disabled by default, uses a per-process Windows named pipe, and requires the
random token published in the current user's local session descriptor. Protocol messages are
newline-delimited JSON with `protocol: 1` responses.

## Safety and current boundary

Memory operations remain bounded and are serviced by the cooperative in-process endpoint. Image
previews are explicit, asynchronous requests: the GPU thread records the copy at its normal
resource-maintenance safe point, completion converts a maximum 480×270 thumbnail, and the client
only polls the result. Supported preview formats cover common 8-, 16-, and 32-bit one-, two-, and
four-channel integer, normalized, and floating-point color images, plus 10:10:10:2 and R11G11B10
HDR images. D16, D24/S8, D32F and D32F/S8 depth aspects are previewed as grayscale. The lazily
allocated readback supports up to 256 MiB images; compressed, multisampled, and larger images fail
closed instead of disturbing rendering.

The I/O viewer retains metadata, not file payload bytes, and caps its detailed event ring. Its
bounded session-wide file inventory preserves aggregate visibility for older resources. This keeps
it useful for locating streamed movies, packages, saves, and configuration reads without silently
copying potentially large or sensitive game data. GPU breakpoints still require a command-processor
safe point for the same renderer-lock reason.

## Resource trace

`Graphics > Resource trace` accepts a guest address and byte range. Its upper table shows the last
known state of every overlapping image generation, including the current authoritative owner,
dirty flags, usage, format, tiling, host image handle, stencil and compression-metadata ranges. The
lower table provides the chronological provenance trail and links each event to its frame, command,
submit and active shader addresses when that context is available. Selecting either table requests
an explicit current-image preview; it does not perform continuous GPU readback.

The address map above the tables draws each image generation as a lane over one shared guest-address
axis: blue is pixel storage, purple is an associated stencil plane, and orange is compression
metadata. Retired generations are dimmed and a green outline marks the native image as the current
authority. This makes partial and differently sized aliases visible without replacing the exact
numeric ranges in the table.

`Trace resource` from a shader image and `Trace range` from Resource history transfer the selected
address directly into this view. The event ring keeps the newest 32,768 transitions so it can span
several frames without allowing an unbounded debug-session memory leak.

The same page can add a mip-0 pixel watchpoint by X/Y coordinate. For supported single-sample PS5
layouts, Kyty resolves the coordinate through the guest tile equation and shows the exact backing
byte beside every resource event that changed, uploaded, downloaded, cleared or reinterpreted its
image. The event is currently image-granular: it identifies the command that could affect the
pixel's image, but does not yet claim that a partial draw covered that pixel. Unsupported layouts
are retained with a clear reason rather than reported as exact. Up to 64 watches and 8,192 recent
hits are kept in memory.

Resource previews support ordinary integer, normalized, floating-point and packed color formats,
plus D16, D24/S8, D32F and D32F/S8 depth images (shown as grayscale using the depth aspect).
Multisampled images still report that an explicit sample/resolve path is required; Kyty does not
silently choose a sample or reinterpret unresolved storage.

Every completed preview also scans the full downloaded mip (not merely the thumbnail) and reports
a stable content hash, zero-pixel count, and NaN/Inf pixel and component counts for FP16, FP32,
HDR and floating-point depth resources. This makes the first non-finite producer visible while
walking a Resource Trace instead of relying on the final black or green frame as the symptom.

The **Break conditions** sub-page can pause on a shader hash or guest shader base, a resource
address with an optional action filter, an existing pixel-watchpoint ID, or a preview that contains
NaN/Inf values. Conditions retain their hit count and reason; one-shot conditions disable
themselves after the first match. Matching is deliberately separated from stopping: callbacks only
record the request while renderer/cache locks may be held, and the GPU worker pauses guest threads
after the complete submission reaches a safe point. This avoids freezing an ownership transition
halfway through while still stopping before subsequent guest work can hide the first fault.

## Command capture and replay readiness

Select a command under `Graphics > Last frame` and choose **Save command bundle**. Kyty writes a
versioned `kyty_gpu_command_capture` directory under `_CommandReplays` with `commands.pm4`,
`constant.pm4`, and a readable `manifest.json`. The manifest records the selected submit's command
list, shader identities, resource ownership events, queue and reset/interrupt state. The debugger
retains the newest 128 top-level submissions within a 64 MiB PM4 budget.

This first layer intentionally reports `replay.executable: false`. Re-executing PM4 while using the
game's *current* indirect buffers and texture contents would look like replay but could silently
produce a different frame or corrupt live state. The manifest lists the missing deterministic
inputs explicitly: nested indirect PM4, pre/post resource contents, and descriptor-address
remapping. Capturing those inputs is the next replay layer; until then the bundle is an exact,
portable command-stream and provenance artifact, not a pretend replay button.

## RenderDoc capture

When the game was started with the launcher's **Enable RenderDoc capture** option, the external
debugger's `Graphics > Last frame` page reports the live RenderDoc lifecycle and can request
**Capture next frame in RenderDoc**. A request starts at the next complete GPU submission and ends
at Kyty's normal frame boundary, using the same renderer-safe mechanism as the F1 shortcut. The UI
reports requested/capturing/idle state, the capture template under `_RenderDoc`, completion count,
and whether the last capture succeeded. If RenderDoc was not enabled before launch, the button stays
disabled instead of injecting/loading it into an already-running renderer.
