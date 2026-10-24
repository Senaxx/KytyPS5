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

## Build (Linux)

Tested by Ozmund on Bazzite (Fedora Atomic) with an RTX 3080 Ti (12 GB) on NVIDIA 615.71.09. The tree
builds unchanged. Bazzite's base system is read-only, so the build runs in a Fedora 44 Distrobox
container. Create it with `--nvidia`, or the container only sees llvmpipe. The packages are the ones in
upstream's README, under "Building on Fedora and Fedora Atomic".

```
git submodule update --init --recursive
cmake -S . -B _Build/linux -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build _Build/linux --target kyty_emulator launcher
cmake --install _Build/linux --prefix _Build/linux/install
```

The installed emulator runs on the host, outside the container.

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

**On Linux** leave out `--redzone`: it is Windows-only, and the Linux emulator rejects it as an
unknown option. On a 12 GB card the following gets to gameplay (see "Linux, 12 GB card" below).
`KYTY_HOST_BUFFER_MB` comes from the open host-buffers PR to this branch, and `KYTY_VRAM_BUDGET_MB`
from the vram-budget PR; without them, drop those variables and expect the cutscene to fill the
card and the texture cache to run clamped (see 2026-10-18 below). The skip list is Mac's.

```
SKIP=bad108e74fb72e9f,4e7f2c6bb9b158a1,8bfd230b9cd875a2,c4df2a00067e0666,dc76e1223a9bf673,e6d76d24f59f8015,12f18ba73d4b5119,198a7ca6e3dfbec7,1d81053379396859,21e4ffa9eaebf903,29be8047c4e86afe,2a69c6facfa03f42,2bd971cbf9fd85b9,2bf24cdfe17d445a,2db5771324658f9b,2ea3957af2daf690,30d445ded2c91412,329858dd50dedcf6,337df7adf22cbcf8,384f47083f0d5e08,3cf86dbfc447bb8c,40978f6a2b51a610,4b32d12a59b34410,76c1cc14d6f7641d,883049efd1de9ca7,919b497919097925,949d0da52574b6bb,9c18c0b9ff0f7997,a0d3108311ea1997,a81255bc7d83eb26,ad3e817088b42ce0,b0164a333a8443f1,ce6af5b8f4b5d395,cf2433f75f08295e,d8cc8c65f6c476d7,d925be771906e8da,deb8684b019495fc,e98e6c7a12cfbb83,f61c05c40f988ea2,f687ff30ac856e49,fbd3835b6b17af6a,fc0537b7278db60c,fceac1321aa192d2
KYTY_HOST_BUFFER_MB=1024 KYTY_TEXTURE_CACHE_MB=5632 KYTY_BUFFER_CACHE_MB=2560 KYTY_VRAM_TARGET_MB=11264 KYTY_VRAM_BUDGET_MB=8704 ./_Build/linux/install/kyty_emulator --game <dump> --readback-linear-images true --tessellation --bindless --skip-shaders $SKIP
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

### Linux, 12 GB card (RTX 3080 Ti, Ryzen 5 7600X, base 4802886, 2026-10-11)

- **Reaches gameplay.** Intros, INITIAL SETUP, the title screen (~3 fps), the opening cutscene
  ("30 YEARS AGO", Essex's line, ~5 fps), then the first jungle fight with its tutorials (leap
  attack, shred, sprint) at 5-10 fps. About 10 minutes played without a crash or freeze.
- **Screen-reader prompt.** It does not draw on Linux: the screen is blank, but J (Cross) still
  dismisses it. On the 3060 laptop it draws fully.
- **Cutscene hang: fixed by the skip list.** Without it, rendering stops partway through the
  cutscene (audio and AMPR streaming carry on) right after a very large pipeline starts
  compiling. With the skip list frames keep coming.
- **Video memory: one 2.7 GB buffer.** The game describes its whole heap at `0x1210000000` as
  one 2,801,664 KB buffer. In VRAM it is a quarter of the card, and the collector deletes and
  re-creates it (each time a 2.7 GB upload), which spikes device use by ~2.8 GB in a second.
  With the desktop's ~2 GB on top the card fills and the whole PC soft-locks. No cache cap
  avoids it: the emulator went 2.2 GB past `KYTY_VRAM_TARGET_MB` before collecting.
  `KYTY_HOST_BUFFER_MB=1024` puts it in system memory, and fps went up (2-5 to 5-7), since the
  re-uploads stopped.
- **CPU: write-fault lock contention.** perf in the fight showed ~13% of all CPU in game
  threads spinning on the tracker's region lock in the write-fault handler, behind a holder
  doing mprotect and a TLB shootdown per 4 KB page. Invalidating a 64 KB granule per buffer
  write fault cut the spin-wait to 2.2%, but froze the game within minutes (rendering and AMM
  streaming stop, audio carries on), so it was dropped; the spin lock pause stayed.
- **GPU drains.** Collecting the 2.7 GB system-memory buffer downloads it first, a ~100 ms
  drain. Keeping system-memory buffers out of the buffer GC (host-buffers PR) cut the worst
  wait to ~33 ms; fps in the fight 5-20, no freeze in 6+ minutes.
- **Textures.** Ground and tree textures stay low resolution. On 2026-10-18 this turned out to be
  the cache bounds, not stale bindless slots (see below).
- **Visual.** Explosion smoke draws as solid black blobs (a skipped shader, or one of the
  unsupported ones). Lighting, fire, fog and characters look right.
- **Profiling on Linux.** Release builds are stripped, but the linker map
  (`_Build/linux/kyty_emulator_*.map`) resolves perf's addresses. A rootless Distrobox cannot
  attach perf or gdb to the emulator on the host; a rootful one (`distrobox create --root`)
  can, after `sysctl kernel.perf_event_paranoid=-1`. Running the emulator inside the
  container fails instead (`importing the supplied dmabufs failed`) and overwrites the
  pipeline cache with an empty one.

### Linux, 12 GB card, re-test (base 4802886 + host-buffers, 2026-10-18)

Same machine. Played into Trask Outpost; no crash. The controller (DualSense) works.

- **Missing and low-detail textures: the video memory budget.** With the 2026-10-11 settings
  (`KYTY_VRAM_TARGET_MB=8192`) and the desktop holding ~2.2 GB, `MemoryShortfall` was non-zero most
  of the session, which clamped the texture collector's thresholds to 512-728 MB while the cache
  held ~4.5 GB. It ran in "critical" every submission (229,500 runs by frame 2,852) and evicted
  bindless heap textures that draws still sampled (`TexGc ... bindless=141`, `125`, `41`).
- **The Linux NVIDIA driver's budget is the limit, not the target.** Raising the target to 11264
  changed nothing: `GetTotalMemoryBudget` takes the smaller of the target and the driver's budget
  less its 1 GB reserve, and the driver reported a budget that left the emulator ~7.9 GB
  (`Memory: short ... device 7374 MB of 7965 MB budget`). `KYTY_VRAM_BUDGET_MB` (vram-budget PR)
  replaces the driver's figure. 9728 overshot (11,493 MB on the card with the desktop, the run
  script's watchdog killed it); 8704 held: thresholds stayed at 3,788-4,812 MB, no bindless
  evictions, 11,010 MB peak on the card, and `Memory: short` only briefly near the top.
- **CPU: tracked page faults dominate.** One 5 min 23 s session (3,357 frames, ~3 fps in the
  outpost) took 9,950,000 tracked faults (379,201 on the GPU thread) at 1,680 sites. The twelve
  slowest sites alone spent ~747 s in the handler, summed over threads: more than two of the six
  cores for the whole session.
  - Reads of GPU-written bytes: site `0x903ee0a9d` took 46,682 reads at ~5.6 ms each (262 s),
    `0x903ee16ac` 9,968 at ~6 ms (61 s). Both last faulted in the heap buffer at `0x1210000000`,
    the one `KYTY_HOST_BUFFER_MB` puts in system memory. None were served by clean-read emulation.
  - Writes: site `0x903d64dc0` took 3,882,355 faults at ~51 us each (197 s), `0x903ee6cf6`
    798,359 (45 s), `0x903dc951d` 794,512 (26 s).
  - `RecordTrackedFault` takes one global mutex and searches the site list linearly on every
    fault, 10 million times here. It is diagnostic only and could be opt-in.
  - `GpuWaits`: 381,696 waits, 51.4 s in total.
- **Log volume.** The session wrote 3.8 million lines (230 MB). About half are libAjm per-call
  parameters (`ajm.cpp`, the `instance = ...` lines).
- **Build.** The host-buffers GC-keep lambda had two return types that clang rejects; fixed on
  that PR's branch. A build script that ignored ninja's exit status had hidden it.

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
