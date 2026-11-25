# Flint Engine v5 — Agent & Developer Guidelines (`AGENTS.md`)

## 1. Engine Mission & Standalone Architecture
- **100% Pure Mojo 1.2**: Zero C/C++ wrapper libraries (no SDL, GLFW, OpenGL, shaderc, miniaudio, freetype, ENet, or Box2D/Jolt). Flint talks directly to OS protocols/syscalls (`AF_INET6` `recvmmsg`/`sendmmsg`, Wayland, X11, DRM/KMS, `evdev`, `inotify`, PulseAudio/ALSA) and Vulkan drivers.
- **100% Standalone Repository & Zero-Warning Gate**:
  - `flint` compiles strictly with `-I src -I tests -O3 --fp-mode contract=off --Werror` and **zero external game repository `-I` paths**.
  - `src/flint/` must never import `gameplay.*`, `world.*`, or downstream game modules. Run `bash tools/flint.sh gate` before committing.
- **Engine Subsystems (`src/flint/`)**:
  - `core`: Deterministic math (`detmath`), stable sorting/heaps (`detsort`), generational SIMD ECS (`ecs`), scene graph & prefabs, `JobPool`, LZ4, and versioned `BlobWriter`/`BlobReader` serialization.
  - `platform` & `input`: Native Wayland, X11, DRM/KMS, Vulkan 1.3 loader, Linux `evdev` gamepads with force-feedback rumble, and declarative `InputMap`.
  - `render`: Scalable 2-tier Vulkan renderer (`VisBuffer` + hardware RT + SVGF/ReBLUR + TSR + VSM cascades), SPIR-V `OpSpecConstant` material graph, VAT crowd baking, and GPU SDF UI.
  - `physics` & `nav`: XPBD/TGS rigid bodies, `VehicleChassis`, `CharacterController`, Quickhull/V-HACD convex decomposition, heightfields, Recast `NavMesh`, and SIMD8 ORCA crowds.
  - `anim`, `ai` & `camera`: `.flanim` state machines, bone sockets, `RagdollStateMachine`, foot-plant IK, `BehaviorTree`, `UtilityAiScorer`, and spring-arm `CameraRig`.
  - `audio` & `cycles`: Pure-Mojo QOA 5:1 codec, BVH8 raycast acoustics/Sabine RT60, multithreaded `AudioServer` with Woodworth HRTF & BS.1770 LUFS limiter, and `cycles` live pattern synth.
  - `assets`, `procedural`, `ui`, `studio` & `devtools`: Incremental `AssetServer`, `.flpk` mmap VFS, SubD/ArchKit procedural kits, Flexbox `DockSpace`, `Flint Studio` visual editor, and MCP v2 server.
  - `ml`: Pure-Mojo ONNX loader/interpreter (mmapped zero-copy weights, SIMD + `JobPool` CPU kernels) with MAX `max.gpu` matrix products on the GPU, the MotionBricks SOMA motion-inbetweening runtime (`flint.ml.motionbricks`), and UniRig auto-rigging (`flint.ml.unirig_rig`: OPT-350M skeleton decoder, voxel-geodesic skin prior, PTv3 skin network with fused GPU attention / sparse-conv gather-GEMM; `examples/unirig`, graphs converted by `tools/unirig/export.py`). Models and clips stay outside the repo, see `licenses/`.
  - `net` & `lab`: Dual-stack IPv6/IPv4-mapped non-blocking `UdpSocket`, `BitWriter`/`BitReader` (32-bit Smallest-Three `Quat`), deterministic `VirtualNetSim`, reliable-ordered `NetChannel` (`1180 B` MTU fragmentation), 4-peer GGPO `RollbackSession`, AOI `SnapshotReplicator`, and `EngineBenchSuite` (`flint.lab`).

## 2. Mojo 1.2 Syntax & Lifetime Rules
- Use `def` for all functions/methods (`fn` was removed); place `raises` before `->`.
- Use `comptime` (`alias` and `@parameter` were removed) and `var` (`let` was removed).
- Argument conventions: immutable borrow default, `mut`, `var`, `out self` in `__init__`, `deinit self` in `__deinit__`.
- Structs: `@fieldwise_init` with explicit `Copyable`, `Movable`; transfer non-`ImplicitlyCopyable` values with `^`. Inside generic structs `Foo[T: Trait]`, qualify `Self.T`.
- Stdlib imports use `std.` prefix (`from std.testing import assert_equal, assert_true`); UTF-8 strings use `s.byte_length()` and `s[byte=i]`.
- **ASAP Destructor & Parallel Job Safety**: Dispatch `JobPool` work via `flint.core.ecs.run_parallel_ctx`, call `.keep_alive()` on `SimdColumn`s passed by raw `base_addr()`, and prefer `hash_list(bytes)` over `block_hash(address_of(bytes), ...)` on last-use `List[UInt8]` values.

## 3. Common Commands
```bash
pixi install                                         # Install Mojo 1.2 toolchain on a new machine
bash tools/flint.sh doctor                           # Check Mojo 1.2, Vulkan ICD, Wayland/X11/KMS, Audio, evdev & inotify
bash tools/flint.sh lab <physics|anim|render|audio_ui|net|all> [--fast]  # Compile & run <2s subsystem labs (-O1 with --fast)
bash tools/flint.sh studio [--headless-smoke|--mcp-stdio]                # Build & launch Flint Studio visual editor / MCP v2 host
bash tools/flint.sh gate                             # Standalone zero-leak + --Werror build + unit/e2e/v5 tests + headless PNG
bash tools/flint.sh test [unit|e2e|v5]               # Run standalone unit, E2E, or v5 subsystem verification suites
bash tools/flint.sh play                             # Run interactive minimal_spark 2D+3D showcase
bash tools/flint.sh preview <spark|world|studio|dir> # Live preview: rebuild on .mojo save and hot-swap window
bash tools/flint.sh kms [--probe]                    # Bare-display DRM/KMS demo from a text console (no X11/Wayland)
bash tools/flint.sh kms3d                            # GPU 3D on the bare display (Vulkan swapchain via VK_KHR_display)
bash tools/flint.sh sync /path/to/Deepsong           # Harvest latest game-agnostic pure-Mojo libraries
bash tools/flint.sh new ../my-game                   # Scaffold a standalone game repo with flint.toml, pixi.toml & tools/run.sh
```
