# Changelog

All notable changes to Flint are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning and the deprecation process
are described in [docs/API_STABILITY.md](docs/API_STABILITY.md).

There are no release tags yet: every commit on the v5 line reports engine version `5.0.0`, and
games pin a commit with `flint_version = "5.0.0+<commit>"`. The history below was seeded from
`git log` (2026-11-14 to 2026-11-26) and is grouped by subsystem, newest first within each group.
Short hashes point at the commits.

## [Unreleased]

### Added

- **Docs and CI:** getting-started, ECS and gameplay guides, an expanded architecture map, this
  changelog, an API stability policy, `CONTRIBUTING.md`, the CPU-only `tools/ci.sh` and a GitHub
  Actions workflow.
- **CI:** an `e2e-studio` job runs Studio e2e tiers 2-4 (`flint.sh test e2e-studio --extended`;
  `--all` runs tiers 1-4). The gate keeps tier 1.
- **Studio:** `.flscn`/`.flscnb` store Enabled and Visible separately (`StudioEntityMetaComponent`
  v3; older files still load), and the undo byte cap is configurable (`undo_byte_budget`).

## [5.0.0] - development line, 2026-11-14 to 2026-11-26

### Studio and devtools

- Fixed: second review of the Studio redesign. MCP security: approvals replay the exact gated call
  and expire after 120 s, hidden/tool-config paths are rejected, builds only write `.so` files
  under `build/`, socket clients are limited to 4 async builds. Data loss: undoing a scene load
  restores path and dirty state, autosave recovery keeps the user's path. Gizmo, status bar,
  Open/Save As dialog, shortcuts and `tests/test_studio_review_fixes.mojo` (`1f500fd`).
- Added: Studio redesign phases 0-7: unified viewport coordinates (`ViewportTransform`), stable
  editor ids (`StudioEditorIdComponent`), per-connection MCP sessions, undo origins, autosave to
  `.flint/autosave`, async `code.build_hot_reload`; Studio suites join the gate (`1eec997`).
- Added: 200% HiDPI SDF UI scaling (`UiDrawList.scale_all`, `--ui-scale`, `FLINT_UI_SCALE`,
  Ctrl+= / Ctrl+- / Ctrl+0) (`c72c926`).
- Fixed: MCP request bursts no longer freeze Studio (4 requests / 12 ms per frame); agent
  visualisation follows spawned entities (`080c666`).
- Added: whole-game milestones M0-M3 (asset pipeline, `.flmat`/`.flrig`/`.flprefab`, hot-reloaded
  `gameplay.so`, PIE checkpoints, packaging), `examples/test_harness_game`, the `tests/e2e_studio`
  suite, a UI redesign with themes and icons, and the Visual AI Agent Theatre (`f027600`).
- Changed: expanded motion authoring in Studio (`7f64fbb`).
- Added: Wave 2 Studio `.flscene`, Inspector and AssetBrowser (`f1bbfa5`).
- Fixed: Phase 0+2+3 audit fixes across Studio, assets, UI and procedural kits (`16f0b7c`).
- Added: Studio/MCP NPR inspectors (`c29afd2`).

### Platform and windowing

- Added: native Wayland `xdg_toplevel` window controls (minimize, maximize, move, resize) and
  client-side controls in Studio's header bar (`e84e8a0`).
- Added: Vulkan 64 MiB `OffsetAllocator` sub-allocation, ALSA `AudioServer` fallback, XCB WSI,
  ARM64-safe KMS rescue (`91ed6b8`).
- Fixed: Vulkan/Wayland leaks, real-time audio safety, 4-pad evdev sync and rumble, `InputMap`
  mouse-delta clamp (`dfc45d4`).
- Fixed: shared libraries load through the stdlib `dlopen`/`dlsym` shapes; `dynlib` uses
  `ptr()` instead of the deprecated `unsafe_ptr()` (`88cdae2`, `aa7b174`).
- Added: zero-copy fullscreen for GPU games on Wayland (`a27b7ab`).
- Added: DRM/KMS bare-display backend (`e0fba84`), logind input when `/dev/input` is unreadable
  (`afa66a2`), Vulkan direct-to-display (`VK_KHR_display`) and KMS GPU mode (`bf92a12`).
- Fixed: `FLINT_KMS_MODE` picks the fastest refresh at the requested size (`ec68e87`).
- Added: aarch64 support (`7e0f4f0`).

### Core, ECS, scene and app

- Added: Wave 2: `par_for_each3`, `read_only` / `changed_since` iteration, chunk reclamation on
  despawn, dirty-tracked transform propagation, dynamic CVars and profiler zones, SIMD `Mat4`
  (`01d54bb`).
- Fixed: Phase 0+1 audit: syscall-free `par_for_each`, sparse-set exclusion, replay safety, O(1)
  lookups (`1af156b`).
- Fixed: `World.restore_blob` validates counts and entity tables before allocating (`a083194`).
- Added: LZ4 checkpoints (`1feb775`), deterministic `.flrp` replay streams (`8ed7b58`).

### Rendering

- Fixed: `long_way_home` presents non-zero-copy Vulkan readback frames (`e11082f`).
- Added: Wave 2 render: 1D meshlet worklist, `AaaRenderGraph` barriers, OBB decals, SpecId 1
  (`2b6a7d1`).
- Fixed: Phase 0+2 Vulkan synchronisation, resize, async VSM, visibility buffer and Hi-Z
  (`63a6d1b`).
- Added: anime NPR cel/outline pipeline, eroding 3D VFX, morph/smear skinning (`c29afd2`).
- Fixed: Vulkan/GpuRaster host scratch freed on teardown (`dbfc3cf`); `GpuSceneRenderer` closes its
  Vulkan device (`6178774`).
- Added: GPU particles (`671672b`), shaped water surfaces (`1410d15`), ray-traced refractive glass
  (`85b3eb1`), denoised ray-traced sun shadows (`9ce7893`), saturation/contrast grade (`adde4d2`).
- Fixed: glass stays see-through with all RT features off (`6cc3580`).
- Added: AAA renderer v2 (`e49058e`); V12-V15 Vulkan 1.3/1.4 renderer (visibility buffer, VSM,
  ray queries, TSR) and cinematic V6-V11 upgrades (`593f9bd`).
- Added: GPU planar mesh shadows (`8ed7b58`), AMDGPU GFX12 PBR/bloom ISA parity and
  cloud/beam VFX (`1feb775`), Vulkan GGX PBR pipeline and TTF SDF fonts (`7e0f4f0`).

### UI and text

- Added: `PRIM_POLYGON` non-zero-winding filled polygons in the GPU UI overlay (`8110de0`).

### Physics, navigation, AI and animation

- Added: Wave 2 unified 6-DoF solver, parallel narrowphase, water early-out, NavMesh + ORCA
  unification (`fa552de`).
- Fixed: Phase 0+1 performance and correctness across physics, nav, AI, camera and anim
  (`2825431`); ORCA doorway jamb steering (`dbfc3cf`).
- Fixed: out-of-range `SolverBodies` batches are rejected instead of corrupting the heap
  (`3cac630`).
- Added: `DirectSpringChain` (`c29afd2`), vehicle physics (`e49058e`).

### Networking

- Added: Wave 2: ChaCha20-Poly1305 AEAD sessions, cookie handshake, AIMD/RTO backoff,
  unordered and raw channels, despawn/AOI/interpolation, hub batching (`5db6b41`).
- Security: fixed DoS, sequence wraparound, deadlock, UDP and bit-packing bugs; fewer replicator
  allocations (`21eb045`).
- Added: `NetHub`, one transport shared by a `NetChannel` per peer (`34547ee`).

### Audio

- Added: `cycles` live-coding audio (`e49058e`), concert `BUS_MUSIC` (`8ed7b58`).

### Machine learning

- Added: MotionBricks SOMA runtime and GPU ONNX inference through MAX (`622a670`).

### Assets and tools

- Added: Wave 2 async `AssetServer`, hardened PNG decoder, incremental `flint.sh` builds
  (`f1bbfa5`).
- Added: `flint.sh preview`, rebuild on save and swap the window (`1360c0f`).
- Added: `.flpk` VFS bundler (`8ed7b58`), LiveLink MCP socket (`1feb775`), GLB 2.0 writer and
  mesh builder (`7e0f4f0`), portable Mojo discovery (`48753b6`).

### Examples

- Added: *The Long Way Home*, a 4-chapter storybook game with a live Studio diorama
  (`7dd1b13`), and Wayland controls / HiDPI scaling for it (`e84e8a0`, `c72c926`).

### Verification

- Changed: 24 Tier 5 adversarial hardening rounds across v5 subsystems (`2ef84e9`); broad
  subsystem and suite updates (`4744bba`).
- Added: V5 recheck notes (`6bbb534`).

### Repository

- Added: working-tree snapshot for flint-py (`8650aa9`), `AGENTS.md` and the desktop GPU roadmap
  (`48753b6`), initial standalone pure-Mojo 2D/3D engine (`436b132`).
