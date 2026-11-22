# Flint Engine

**Flint** (`flint`) is a standalone, **zero-C-dependency**, pure **Mojo 1.2** 2D & 3D game engine.

Every subsystem — from direct Wayland/XDG-shell sockets, PipeWire/PulseAudio & ALSA `ioctl` audio, `futex` work-stealing job graphs, 8-wide SIMD Structure-of-Arrays (SoA) ECS, TrueType SDF font baking, glTF 2.0 PBR & skeletal animation, XPBD/TGS physics solvers, and LDR-FLIP perceptual image diffing, all the way down to **hand-emitted Radeon RDNA GFX11/GFX12 machine code (`AMDGPU`) and Vulkan 1.2 SPIR-V 1.3 (`Kernel`)** — is written in 100% pure Mojo.

---

## Highlights

- **Pure Mojo 1.2, Zero Wrapper Libraries**: No GLFW, SDL, OpenGL, glslang, shaderc, miniaudio, freetype, or Box2D/Jolt C++ wrappers.
- **Unified 2D + 3D Rendering**:
  - **`SimdCanvas` (`flint.render.simd_canvas`)**: Sub-4 MiB 8-wide SIMD software rasterizer & depth-buffered sprite/3D mesh/beam compositor (`Camera3D` perspective + `blit_sprite_simd`) for instant startup, laptop efficiency, and deterministic headless CI screenshots.
- **Dual-GPU Compute & PBR Raster (`flint.render.kernel`, `gpu_raster`, `vulkan_backend`, `amdgpu`, `scene_renderer`)**:
  - Write compute kernels once in Mojo (`Kernel`) and emit **both** native AMD RDNA GFX11/GFX12 ISA dwords and Vulkan SPIR-V 1.3 modules.
  - Full `GpuSceneRenderer` parity across Vulkan SPIR-V, AMDGPU GFX11/GFX12 ISA, and CPU PBR: Cook-Torrance GGX microfacet specular, Schlick Fresnel, soft-penumbra sphere shadows, GPU directional planar mesh/model shadows (`draw_planar_shadow_mesh` / `draw_planar_shadow_model`), 48 tiled dynamic point lights, AgX filmic tonemapping, 5-tap Gaussian + 2x2 bilinear HDR bloom, 3D ribbons/plumes, `draw_colossal_beam_3d`, and volumetric `draw_cloud_deck`.
- **Native GLB 2.0 Forge, `.flpk` Asset Packager & `.flrp` Replay Streams (`flint.assets`)**:
  - Parametric curved `MeshBuilder`, binary glTF 2.0 `.glb` writer/loader (`glb_writer`, `glb_native`), single-file `.flpk` VFS packager (`write_flpk_pack`, `tools/flint.sh pack`), LZ4 + FNV-64 binary state checkpoints (`checkpoint.mojo`), and deterministic per-tick input replay streams (`replay_stream.mojo`).
- **Concert-Pitch Music + Procedural Audio Mixer (`flint.platform.audio`)**:
  - Direct PipeWire/PulseAudio + ALSA `NativeAudio` mixer with `BUS_MUSIC` concert-pitch stereo loops (`set_music_clip`), velocity-following ambient bus, `BUS_SPEECH` radio/comms, and 8-voice priority SFX.
- **First-Class DevTools & AI Agent LiveLink (`flint.devtools`)**:
  - Live drop-down `~` `DevConsole` (`ConsoleVars` tab-completion), `F2` frame/zone profiler HUD, `F3` `MemCharge` CPU/VRAM budget inspector, generic `inotify` `AssetWatcher`, `LDR-FLIP` perceptual regression testing, and a non-blocking `AF_UNIX` `FlintMcpLiveEndpoint` socket server.
- **Continuous Upstream Harvesting (`tools/sync_from_deepsong.sh`)**:
  - Cleanly syncs battle-tested game-agnostic pure-Mojo primitives from `Deepsong`, rewrites namespaces to `flint.*`, applies Flint VFS/input/glyph enrichments, and enforces a strict **Zero-Game-Leak Gate** (`0` `gameplay.*` or `world.*` imports).

---

## Quickstart

### Prerequisites
- Linux x86_64 (Wayland desktop for live window; any headless Linux container for `test` / `shot` / `gate`)
- [Pixi](https://pixi.sh) (installs Mojo `>=1.2.0,<1.3` automatically) or `mojo` 1.2 on `PATH`

```bash
# Install Mojo 1.2 toolchain (first time on a new machine)
pixi install

# Run the full verification gate (Zero-Leak check + build + 5 subsystem test suites + headless PNG)
bash tools/flint.sh gate

# Launch the interactive 2D + 3D + 2,048-SIMD-Spark showcase
bash tools/flint.sh play

# Scaffold a brand-new standalone game powered by Flint
bash tools/flint.sh new ../my-flint-game
```

---

## Open source in TurboMojo

To open the project in its own graphical IDE window, run `bash ~/flint/tools/flint.sh studio --ide-gui` from the game directory. You can append a file or `FILE:LINE:COLUMN`. Build the desktop frontend with `pixi run build-gui` in the TurboMojo checkout. Flint looks for `tmide-gui` on PATH or in `~/IDE/build`. Set `FLINT_TURBOMOJO_GUI` to select another build.

Click **TurboMojo [F8]** in the Studio toolbar or press **F8**. TurboMojo opens inside the Studio window. Press **F2** to save, or **Ctrl+F8** to return to the scene. Press **F8** from the scene to resume the same buffer, including unsaved edits. Inside TurboMojo, **F8** runs the current configuration. From a game directory, it opens the `[project] entry` in `flint.toml`. The motion editor and Shrike's combat and gallery rehearsals open their own Mojo entry points.

```bash
bash ~/flint/tools/flint.sh studio --ide
bash ~/flint/tools/flint.sh studio --ide src/main.mojo:40:5
```

The command runs in the current interactive terminal. Later requests for the same project reuse its Studio IDE session. Flint finds `tmide` on PATH or in `~/IDE/build/tmide`. Set `FLINT_TURBOMOJO` to another executable path if needed. The standalone launcher uses GNOME Terminal or `x-terminal-emulator` when invoked without an interactive terminal.

Flint supplies its Mojo toolchain and links `build/flint-ide/engine-src` to the engine source for games with a `flint.toml`. To enable compiler and language-server imports, add `include_paths = ["src", "build/flint-ide/engine-src"]` under `[tools]` in the game's `.turbomojo.toml`. Shrike includes this configuration. [TurboMojo on GitHub](https://github.com/sabelg/turbomojo)

Launch Studio from a Flint game directory and choose **Tools → Move Library**, or run `bash ~/flint/tools/flint.sh studio --moves` directly. The scene menu bar groups file operations, edits, transforms, playback, and project tools. Press **F10** to open it, use the arrows and **Enter** to choose a command, or **Esc** to dismiss it. Select a project move, open its motion or combat editor, or launch a room that uses the preset.

## Choreograph a move in Studio

From a game directory, run `bash ~/flint/tools/flint.sh studio --motion path/to/move.motion.json`. The editor loads a skinned GLB, blends two named clips and draws the hands, feet and ledge at any point on the timeline. Drag the playhead or step at 30 Hz with Left and Right. Use Space for playback, M for the skeleton, G for the gameplay camera, and S to save. Move or rotate the ledge with Up, Down and R. Keys 1 and 2 select clips, minus and equals change the overlap, and T sets the audible cue. Tab selects the opening or closing camera key. A/D orbit that key and W/X change its height.

A preset uses the following JSON fields. Model and WAV paths resolve from the preset directory. Replace the clip and joint names with names from the model. An empty contact name allows gestures to play in place. The optional support joint uses its two parent joints for IK along the ledge.

```json
{
  "version": 1, "model": "../assets/actor.glb",
  "first": "kick", "second": "hang", "overlap": 0.5,
  "contact": "LeftHand", "support": "RightHand",
  "markers": ["LeftHand", "RightHand", "LeftFoot", "RightFoot"],
  "start": [-0.5, -1.4, -0.4], "scale": 1,
  "ledge": [0, 1.7, 0], "yaw": 0,
  "cameraStart": [2, 0.5, -3], "cameraEnd": [1, 0.3, -3],
  "cameraAim": [0, -0.5, 0], "cue": 0.55, "sound": ""
}
```

The overlap is a fraction of the shorter clip. The cue uses normalized sequence time and plays the selected WAV, or a timing click when empty. Camera keys, aim and starting position are relative to the ledge. Yaw is in degrees. Use `--check` after the preset path to validate a translated and rotated instance, entry direction, reach, seeking and cue crossing. Use `--shot frame.png 0.8` or `--game-shot frame.png 0.8` for a headless frame.

In gameplay, load a `MotionSequence` and bind one `MotionPlayer` per actor to its loaded model. Test `can_catch` with the airborne hand, velocity and reach. Pass each wall's `Placement`, elapsed seconds and the actor's position at takeoff to `sample`. Render `player.bones` at the returned placement. A game can supply its camera by setting `[studio] motion_entry` in `flint.toml` to a Mojo entry point that calls `run_motion_editor[YourCamera]()`. Implement `MotionCamera.view` to return the gameplay camera for the sampled actor. The default view is a third-person follow camera.

## Rehearse a combat encounter

From a game with a `[studio] combat_entry` in its `flint.toml`, run `bash ~/flint/tools/flint.sh studio --combat path/to/move.combat.json`. The entry point supplies the game simulation, camera, rendering and reset. Shrike's Warden plunge is a playable example.

`flint.anim.combat_timing.CombatTiming` loads and saves version 1 presets with integer `fps`, `impact`, `close`, `chain` and `end` fields and a positive `reach` in world units. Damage opens at `impact` and ends just before `close`. Follow-ups open at `chain` and end just before `end`. Pass the elapsed animation frame to `active` and `can_chain` on each simulation step. The game applies damage once per target. `flint.studio.combat_editor.CombatTimeline` handles frame scrubbing, dragging and keyboard nudges against the same preset.

## Place a traversal room

A game can register a Mojo `room_entry` under `[studio]` and launch it with `studio --room path/to/gallery.room.json`. `flint.studio.traversal_room.TraversalRoom` stores a lip transform and local landing, takeoff and defender markers alongside references to motion and combat presets. The landing is relative to the lip in all three axes. Takeoff and defender use the lip's horizontal position and yaw and store height from the ground. Shrike's gallery editor uses these placements to draw a platform, play either exit, review recorded simulation ticks and save a second layout with the same animations.

## Compare performance captures

From a game directory, use `bash ~/flint/tools/flint.sh studio --compare before.json after.json` to open two completed Flint Trace Event files. Click the graph to inspect the same frame index in both runs. A and B jump to each run's slowest frame. Up and Down select CPU scopes or GPU passes. Use `--text` for per-zone means, 95th percentiles, deltas in milliseconds and sample counts, or `--shot comparison.png` for a saved view.

For a listening development game, run `studio --capture path/to/live.trace.json` from the same filesystem, or `studio --capture --ssh user@device /absolute/device/live.trace.json local.trace.json` over an existing SSH connection. The game writes the trace on its own clock. Studio retrieves it after completion. Shrike supplies a working producer through `--profile-live`.

A frame includes its pacing wait. “Frame less pace” subtracts the explicit wait and includes render submission and synchronization. Repeated scopes in a frame are summed. Statistics use frames that contain the scope. A missing GPU timestamp is excluded. A scene scope can contain skinning work and overlap the GPU, so these rows should be read individually. Use the same rehearsal, machine, resolution and settings when comparing builds.

## Build profiles

Add named sections such as the following to a game's `flint.toml`.

```toml
[build_profiles.linux-development]
platform = "linux"
optimization = "O1"
loose_assets = true

[build_profiles.windows-wsl2]
platform = "windows-wsl2"
optimization = "O3"
loose_assets = true
```

`studio --build-profiles` opens the native picker. Enter builds the selected profile and Escape closes it. The same profiles are available through `package --list-profiles` and `package --profile NAME`. By default each profile has its own binary cache and output under `build/packages/NAME`. The bundle manifest records the architecture, platform, optimization and profile.

The Windows WSL2 bundle includes `Play-WSL.cmd`. Run it on a Windows machine with a configured WSL2 Linux distribution, GUI support and a working Vulkan driver inside that distribution for 3D games. A matching Linux CPU architecture uses the bundled player. On other architectures the launcher uses Pixi in WSL to compile the bundled sources and caches the executable. Pass `--verify` to Shrike's launcher for a headless mission check. Use `FLINT_WSL_REBUILD=1 bash run-wsl.sh --verify` inside Linux to exercise the source-build path.

## Repository Layout

| Path | Purpose |
| :--- | :--- |
| `src/flint/core/` | Math, deterministic RNG/trig, `JobPool`, `MemCharge`/`Arena`, `OffsetAllocator`, `BlobWriter`, `LZ4`, `ecs.mojo`, `DevConsole`, `Profiler` |
| `src/flint/platform/` | Direct Wayland client, `evdev` gamepads, PipeWire/PulseAudio + ALSA `NativeAudio` (with raised-cosine ambient drone), Vulkan loader, `AMDGPU` DRM, RDNA ISA assembler |
| `src/flint/assets/` | `.flpk` mmap VFS, strict JSON DOM/writer, PNG/TGA/HDR codecs, glTF 2.0 loader, BCn GPU compressor, `inotify` `AssetWatcher` |
| `src/flint/text/` | Pure-Mojo TrueType `.ttf` parser, multi-range SDF atlas baker, UTF-8 typesetter, kinetic `lettering` |
| `src/flint/ui/` | Retained UI widget tree, focus navigation, input prompt glyphs, subtitles |
| `src/flint/render/` | `SimdCanvas` (2D depth sprites + 3D lit triangles/lines), write-once `Kernel` (Radeon ISA + SPIR-V 1.3), PBR shaders, TAA, bloom, GTAO, `LDR-FLIP` |
| `src/flint/physics/` | Parallel hash-grid `Broadphase`, sleeping `Islands`, GJK/EPA `Convex`, TGS contact `Solver`, XPBD constraints, raycasts |
| `src/flint/anim/` | TRS `Skeleton`, animation clip sampling & blending, Two-Bone `IK`, procedural `Spring` |
| `src/flint/devtools/` | `FlintDevOverlays` (`~` Console, `F2` Profiler, `F3` Memory) & `Flint MCP` JSON responder |
| `examples/minimal_spark/` | Flagship `< 180`-line 3D crystal + 2,048 parallel SIMD depth-sprite + audio + DevTools showcase |
| `tests/test_flint.mojo` | Pure-Mojo engine verification suite (ECS/LZ4, 1T-vs-7T JobPool determinism, OffsetAllocator, Dual-GPU ISA+SPIR-V codegen, Broadphase + FLIP + MCP) |
| `tools/` | `flint.sh` unified CLI & `sync_from_deepsong.sh` upstream library harvester |
