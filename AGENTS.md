# AGENTS.md: Marvel's Wolverine (PPSA03671) handoff

Guidance for AI coding agents, and people, picking up this tree. State as of 2026-09-25.

**This work is not public yet.** Do not push it, open PRs from it, or name the title in public
commits, issues or chats without checking with Mac first.

## What this tree is

A source snapshot of KytyPS5: upstream `KytyPS5/KytyPS5` main at **`db7745ef`** ("update README",
2026-09-24) plus three layers of changes, delivered as one tree:

1. The fixes from [PR 780](https://github.com/KytyPS5/KytyPS5/pull/780), the public Spider-Man: Miles
   Morales boot and gameplay work.
2. EmK530's renderer optimisations (`https://github.com/EmK530/KytyPS5`, branch
   `perf/renderer-optimizations`): batched RELEASE_MEM flushes, debounced DCC clear checks and periodic
   draw flushes, plus follow-up fixes to them.
3. The work for this title, described below.

The package has no git history. `../changes-vs-upstream-db7745ef.diff` holds the whole delta against
upstream, if you would rather apply it than copy the tree. The diff only adds and modifies files;
nothing upstream is deleted.

## Getting a buildable checkout

The submodules (`3rdparty/*`) are not in the package. They are unchanged from upstream `db7745ef`.

```
git clone https://github.com/KytyPS5/KytyPS5.git
cd KytyPS5
git checkout db7745ef
git submodule update --init --recursive
git apply ../changes-vs-upstream-db7745ef.diff
```

Copying this tree over the checkout gives the same result.

## Build (Windows)

Use the same toolchain as upstream CI (`.github/workflows/Build.yml`): Ninja, clang-cl, Qt 6
(msvc2022_64), and glslang's tools on `PATH`, all from a Visual Studio x64 developer prompt.

```
cmake -S . -B _Build/windows -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_PREFIX_PATH=<Qt>/msvc2022_64
cmake --build _Build/windows --target kyty_emulator launcher
cmake --install _Build/windows --prefix _Build/windows/install
```

- **Tests:** run `cmake --build _Build/windows --target kyty_tests -- -k 0`, then
  `ctest --test-dir _Build/windows -E kernel_file_system`. All 39 pass. `kernel_file_system_tests`
  does not link on Windows, because SDL's `#define main SDL_main` reaches it through
  `kernel/fileSystem.h`. That is an upstream issue: it fails on unmodified `db7745ef` too.
- **Tracy:** if Tracy fails to compile on `SystemProcessProviderGuid` and similar, your Windows SDK is
  older than Tracy 0.14.1 needs. Add `-DTRACY_NO_SYSTEM_TRACING=ON`.
- **Stale binaries:** kill any running `kyty_emulator.exe` before building, because it locks its own
  binary. `cmake --install` can also fail on `launcher.exe` and leave an old emulator in `install\`,
  so check the timestamp before trusting a measurement.

`../KytyPS5-bin/` holds a Release install built from exactly this tree.

## Running the title

In the launcher, add the game folder, edit its configuration and tick:

- **Enable readback**
- **Enable tessellation support**
- **Windows SysV red zone crash protection.** Required: without it, pressing Cross on the first menu
  crashes, because Windows exception dispatch overwrites the guest's SysV red zone.
- **Bindless textures.** Required: without it the game never gets past a skybox (see below).

The command-line equivalent:

```
kyty_emulator.exe --game "<dump>\eboot.bin" --readback-linear-images true --tessellation --redzone --bindless --printf-direction File --printf-output-file _kyty.txt
```

No game patch is needed.

Launcher settings live in `C:\ProgramData\Kyty\Kyty.ini`, which every launcher copy on the machine
shares. If a run behaves as though a flag is missing, check there first.

**Keys:**

| Key | Button |
|---|---|
| J | Cross |
| L | Circle |
| I / K | Triangle / Square |
| arrows | d-pad |
| WASD | left stick |
| F / H / T / G | right stick |
| Q / E | L1 / R1 |
| Z / C | L2 / R2 |
| Shift / Ctrl | L3 / R3 |
| Enter | Options |

Space pauses the emulator and is not a pad button.

The first run compiles every shader, so it is slow; later runs use the pipeline cache. Rebuilding the
emulator invalidates that cache. Measure progress by log markers such as the `EndOfPipe` count, not
by wall time: a cold cache changes timing by about 2x without changing how far the game gets.

## Where it stands (RTX 3060 laptop)

- Both intro videos play at 45–50 fps.
- On first boot the screen-reader prompt draws fully: text, icons and button glyphs.
- Pressing Cross goes through the load and new-game menus to the 3D title screen, "PRESS ⊗ TO START",
  over a fully textured desert scene with depth of field.
- After Start, subtitles appear and the voice-over plays choppily. Frames keep advancing at 2–3 fps.
  Nobody knows yet whether it reaches gameplay.
- That last run also flooded its log (see Save data below, now fixed), which may explain the
  slowness. Re-measure it first.

## The work for this title, by area

### Bindless textures (`--bindless`, off by default)

**Why it exists:** the material system reads image descriptors from one global texture heap, using a
key computed on the GPU per material and per pixel. Upstream's indirect images enumerate every
candidate on the CPU, bind up to 64 and switch between them in the shader. This title's heap starts
at about 7,000 descriptors and grows past 14,500 as it streams, which is far beyond what enumeration
can handle, so nothing drew.

**Recompiler.** In `ir/passes/ResourceTracking.cpp`, `TryMakeIndirectImage` marks the image
`IndirectImage::bindless` instead of enumerating it, when bindless is on and the table is a buffer
(V#) heap. `MaterializeBindlessImage` in `ResourceMaterialization.cpp` then:

- evaluates only the heap V# on the CPU;
- reserves two flattened-SRT words (translation region base and entry count);
- records a `ResourceSnapshot::bindless_heaps` use.

**SPIR-V.** The emitter code is in `backend/spirv/spirvEmitterModule.cpp` (`DefineBindlessImages`),
`spirvEmitterImage.cpp` (`BindlessSlot`) and `spirvEmitterAnalysis.cpp`. Descriptor set 1 holds:

| Binding | Contents |
|---|---|
| 0 | runtime array of 2D sampled images |
| 1 | runtime array of 2D-array sampled images |
| 2 | runtime array of cube sampled images |
| 3 | runtime array of 3D sampled images |
| 4 | translation storage buffer |
| 5 | feedback storage buffer |

The constants are in `ir/BindlessBindings.h`. The shader computes
`slot = translation[key < count ? base + key : 0]`, and the array access and sampled image are
decorated `NonUniform`. A pending entry (`0xFFFFFFFF`) stores a request in `feedback[base + key]` and
samples a placeholder.

**Host table.** `renderer/pipeline/bindlessTable.{h,cpp}` is owned by `RenderContext`. It holds:

- the set-1 layout, pool and set, created with UPDATE_AFTER_BIND and PARTIALLY_BOUND;
- the translation buffer (1M entries) and the feedback buffer, both in host-visible device-local
  memory;
- one region per heap, keyed by (heap base, table offset, view binding). A region gets headroom (at
  least 16,384 entries, or twice the count) and moves when a draw's descriptor covers more keys.

**Resolution.** `ResolveBindlessRequests` in `pipeline/descriptors.cpp` runs once per presented frame
and resolves up to 128 keys. For each key it:

1. reads the heap T#;
2. calls `ResolveTexture` and `FindTexture`;
3. checks shape and format compatibility (`BindlessCompatible`);
4. allocates a slot and writes the descriptor;
5. publishes the slot in the translation buffer.

Each draw patches its own region and count into its uploaded flattened SRT, and transitions the
resolved images to read-only layouts.

**Texture cache.** Bindless images are pinned: the GC skips `Image::bindless_pinned`. Unregistering
one repoints its slot.

**Device.** `presentation/window/vulkanWindow.cpp` enables the descriptor-indexing features when the
device supports them, and logs `Vulkan bindless images: enabled (...)`.

**Placeholders** are deliberately loud colours. Change them to black or transparent once things
settle.

| Slot | Colour | Meaning |
|---|---|---|
| 0 | grey | image dropped (unregistered) and awaiting re-resolve |
| 1 | red | key beyond the heap's entry count |
| 2 | blue | requested but not resolved yet |

**Gaps, in rough priority order:**

1. **Stale textures.** A resolved slot is never re-synced, so mips the game streams in later are not
   picked up. Look for blurry textures that never sharpen.
2. **Slot reuse.** Slots are never reused; the allocator only grows.
3. **Samplers.** Sampler heaps are not bindless yet. `PlanDefaultSamplers` substitutes a default
   linear/wrap sampler and logs `bindless sampler: using a default sampler`. The next step is bindless
   samplers built the same way, from a heap with a 16-byte stride.
4. **Image kinds.** Only float sampled images (2D, 2D array, cube, 3D) are supported. Storage images,
   uint and sint images, and the atomic float image are not. The atomic float image (format 22)
   appears in two pixel shaders and prints `atomic image descriptor 13 uses unsupported format 22` on
   stderr.

### Resource tracking (`ir/passes/ResourceTracking.cpp`)

Relaxations so the title's material chains are recognised:

- An `IsWaveUniform` whitelist.
- `MatchMaterialOffset` accepts a shift or an IMul by a uniform selector, and folds field offsets.
- Keys are peeled out of bit-fields (`key_shift`, `key_mask`).
- Shared image reads are allowed.
- Dead phi webs are ignored (`FeedsOnlyDeadPhis`).

A rejection logs `indirect image rejected at line N`, naming the check that failed.

### CFG dispatcher safety cap (`backend/spirv/spirvEmitterProgram.cpp`)

Shaders whose CFG cannot be structurised run through a dispatcher loop. One of them intermittently
hung the GPU with `VK_ERROR_DEVICE_LOST`: a blur pixel shader on the title menu, guest hash
`0x09161b28ae4039bb`. Its loop counts come from a per-pixel `buffer_load_dwordx3` (a radius, V# at
SRT+80), which appears to read garbage.

The dispatcher now stops after 4,096 block transitions per invocation, and the `shader_cfg` test
baselines were raised to match (248 words, 3 phis). The root cause is still open: why is the radius
data bad? A stale buffer, or the wrong V#?

### Decoder

- Unknown scalar source operands, such as `SRC_PRIVATE_BASE` (0xED), make the shader give up instead
  of aborting the emulator.
- `S_CBRANCH_CDBGUSER`, `S_CBRANCH_CDBGSYS_OR_USER` and `S_CBRANCH_CDBGSYS_AND_USER` decode like
  `S_CBRANCH_CDBGSYS`, which is never taken.

### AMPR (`libs/libAmpr.cpp`)

- The APR unit (file reads) and the AMM unit (memory maps) execute under separate locks, and a
  wait-free AMM batch runs inline when its ring is idle.
- Before this, a map could queue behind multi-megabyte file reads. The game then touched the
  streaming memory before it was mapped, which caused a host access violation while streaming.
- `AMPRTRACE` log lines trace every submission.

### ATRAC9 (`libs/ajm.cpp`, `libs/ajm/atrac9_decoder.h`)

Config data is validated before LibAtrac9 sees it: header 0xFE, the validation bit, and a channel
config below 6. The title passes an invalid config that crashed the decoder's init.

### Save data (`libs/libSaveData.cpp`)

`SaveDataGetEventResult` logs only the events it returns, plus its first four empty polls. The title
polls it in a tight loop after searching the autosave directory, and logging every poll wrote 74 GB
in 23 minutes.

### Launcher

The configuration dialog has a **Bindless textures** checkbox, which passes `--bindless`.

## Known problems (from the latest logs)

Shaders that give up are skipped, whether draw or dispatch. The log line is
`gave up hash=...: reason`.

- **Resource tracking failed**, 9 shaders: `0x01ba288cccc11730 0x296ff82e1c9d4136
  0x3362313b61a6c8b4 0x487054eeb2d7f3e7 0x4e71522e8bee1279 0x7e7bc09f4cadad5d 0xb8b17176ff277f18
  0xbf3eac34fedbad2d 0xc7a4d5f5ae678cf0`. The preceding `shader resource tracking:` lines give the
  reasons:
  - pixel shaders, 5 lines: `GetImageResource dword 0 is not a valid runtime value (indirect image
    rejected at line 1080)`;
  - compute shaders, 4 lines: `buffer descriptor is not a valid runtime value; GPU-selected access
    requires a raw DWORD x2/x4 load`. These are GPU-selected buffer descriptors, and bindless buffers
    are probably the next design question.
- **SOP1 0x21**, i.e. `S_SWAPPC_B64`, a shader function call: `0x65730dab35e283f7` and
  `0xc4df2a00067e0666`. The recompiler has no call support.
- **SOP1 0x04**, i.e. `S_MOV_B64` from operand 0xEB (`SRC_SHARED_BASE`, the flat aperture):
  `0xdc76e1223a9bf673`.
- **VOP3 0x22**, i.e. `V_CMP_EQ_F64`: `0xe6d76d24f59f8015`.

The opcode names above were decoded by hand from the raw words in the log; check them against the ISA
before relying on them.

## Diagnosing

- **Log lines to grep:** `gave up hash=`, `shader resource tracking:`, `Bindless heap region:`,
  `Bindless requests:`, `Bindless out-of-range key:`, `Bindless feedback:`, `Bindless patch:`,
  `AMPRTRACE`, `APR submit read failed` and `Unresolved import stub called`.
- **Unresolved imports:** the loader lists every unresolved import at startup, but only the
  `stub called` lines show which ones the game actually calls. Check those before blaming an import.
- **Upstream options:**
  - `--graphics-debug-dump true` dumps the decoded code and SPIR-V of every shader into the shader
    log folder.
  - `--shader-validation`, `--vulkan-validation` and `--gpu-assisted-validation`.
  - `--command-buffer-dump`.
  - `--profile` enables Tracy, on demand.
- **Host stacks:** every fatal error prints a symbolised host stack when the `.pdb` sits next to the
  exe. The build writes it to `_Build/windows`; the prebuilt package leaves it out.
- **Device loss:** the emulator writes the driver's vendor fault data to `_device_fault.nv-gpudmp`
  next to the log.
- **Validation layers:** overlay layers (RTSS, Steam, OBS, Epic, GOG, Overwolf and others) can crash
  `--vulkan-validation` at startup. Disable them with `VK_LOADER_LAYERS_DISABLE`.
- **GPU-AV:** GPU-assisted validation stops on a false-positive LDS data race, because lockstep GCN
  code does not use barriers the way the layer assumes. Set
  `VK_KHRONOS_VALIDATION_GPUAV_SHARED_MEMORY_DATA_RACE=false`.
