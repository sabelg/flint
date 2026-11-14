# Flint Engine

**Flint** (`flint`) is a standalone, **zero-C-dependency**, pure **Mojo 1.2** 2D & 3D game engine.

Every subsystem — from direct Wayland/XDG-shell sockets, PipeWire/PulseAudio & ALSA `ioctl` audio, `futex` work-stealing job graphs, 8-wide SIMD Structure-of-Arrays (SoA) ECS, TrueType SDF font baking, glTF 2.0 PBR & skeletal animation, XPBD/TGS physics solvers, and LDR-FLIP perceptual image diffing, all the way down to **hand-emitted Radeon RDNA GFX11/GFX12 machine code (`AMDGPU`) and Vulkan 1.2 SPIR-V 1.3 (`Kernel`)** — is written in 100% pure Mojo.

---

## Highlights

- **Pure Mojo 1.2, Zero Wrapper Libraries**: No GLFW, SDL, OpenGL, glslang, shaderc, miniaudio, freetype, or Box2D/Jolt C++ wrappers.
- **Unified 2D + 3D Rendering**:
  - **`SimdCanvas` (`flint.render.simd_canvas`)**: Sub-4 MiB 8-wide SIMD software rasterizer & depth-buffered sprite/3D mesh/beam compositor (`Camera3D` perspective + `blit_sprite_simd`) for instant startup, laptop efficiency, and deterministic headless CI screenshots.
  - **Dual-GPU Compute & Raster (`flint.render.kernel`, `gpu_raster`, `vulkan_backend`, `amdgpu`)**: Write compute kernels once in Mojo (`Kernel`) and emit **both** native AMD RDNA GFX11/GFX12 ISA dwords and Vulkan SPIR-V 1.3 modules.
- **Data-Oriented Parallel ECS (`flint.core.ecs`)**:
  - Generational 64-bit `EntityHandle`, dynamic `ArchetypeRegistry`, deterministic `ArchetypeSlots` with deferred tick-boundary commits, 8-lane SIMD-padded `SimdColumn[dtype]` with automatic `MemCharge` RAM accounting, `BlobWriter` + `LZ4` persistence, and Mojo-lifetime-safe `run_parallel_ctx` over `JobPool`.
- **Full Physics & Animation Foundations (`flint.physics`, `flint.anim`)**:
  - Parallel spatial-hash `Broadphase`, union-find sleeping `Islands`, GJK/EPA convex narrowphase, TGS velocity contact solver, XPBD cloth/rope constraints, skeletal TRS skinning, Two-Bone IK, and spring procedural motion.
- **First-Class DevTools & AI Agent Introspection (`flint.devtools`)**:
  - Live drop-down `~` `DevConsole` (`ConsoleVars` tab-completion), `F2` frame/zone profiler HUD, `F3` `MemCharge` CPU/VRAM budget inspector, generic `inotify` `AssetWatcher`, `LDR-FLIP` perceptual regression testing, and a game-agnostic `Flint MCP` JSON responder.
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
