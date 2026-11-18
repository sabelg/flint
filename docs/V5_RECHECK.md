# Flint v5 re-check: open items (2026-11-18 09:35)

This follows up [V5_REVIEW.md](V5_REVIEW.md). It checks the working tree as of 09:31, while fixes were still in progress.

## Status of the review's "fix first" list
All 9 items are fixed in code. Under ASAN, the v5 suites audio_ui_input, core_ecs_scene, net_rollback and physics_nav_ai pass at `-O1 -g1`, and v5 render and assets_studio_mcp pass at `-O3 -g1`. The unit suite passes at `-O3 -g1` apart from a driver issue at exit (item 4). E2E had 1 failure under ASAN at 09:31; after the 09:49 fix it passes 505/505 at -O3 without ASAN, not yet re-run under ASAN.
- `.flpk` v3 was fuzzed with 20,000 corrupted packs, CRC resealed so they reach validation: 0 ASAN errors.
- Navmesh carving now takes 0.05 ms, the 60 m diagonal path is 81.2 m, and ORCA shows 0 overlaps at 1,800 and 3,600 ticks.

## Still open: please fix

### 1. `World.restore_blob` crashes on a corrupted snapshot (src/flint/ecs/world.mojo)
Counts read from the blob are used to allocate before anyone checks that the blob holds that many bytes:
- `ent_len`, around line 2518: `List[EntityRecord](capacity=ent_len)` plus `resize(unsafe_uninit_length=ent_len)`.
- `free_len`, around line 2532: same pattern.
- The per-archetype chunk and row counts, around line 2579, which allocate through the constructor around line 729.

A flipped byte gives a multi-terabyte allocation, so ASAN reports `allocation-size-too-big`. Without ASAN it is an OOM, or a crash on a null pointer.

**Fix:** before each allocation, raise `Error` unless `count >= 0 and count * record_size <= reader.remaining()`. `BlobReader.remaining()` already exists in core/serial.mojo:139. A fixed sane cap also works, for example `MAX_ENTITIES`.

**Repro:** build a world, call `snapshot_blob`, flip random bytes, then call `restore_blob` into a fresh world and use it afterwards. Fuzzers are in `docs/v5_recheck/`:
`mojo build -I src -O1 -g1 --sanitize address docs/v5_recheck/ecs_fuzz.mojo -o build/ecs_fuzz`, then `./build/ecs_fuzz 5000 [skip_header_bytes]`.
`flpk_fuzz.mojo` is built the same way; run it as `./build/flpk_fuzz 20000 /tmp/fz.flpk`.

After the fix, rerun the fuzzer with `skip_header_bytes` set to 0 and to 200. It currently stops at the first abort, so the code after these allocations hasn't been fuzzed yet.

### 2. Minor: the render suite misses its own time target
`test_flint_v5_render` prints "PASSED in 2.73 s (<2.0s target)" at `-O1`. Check it at `-O3` in the gate.

### 3. Minor: ORCA with walls
At 3,600 ticks, 253 of 256 agents arrive and 1 is reported `not_through_wall`. It's worth a look, but it doesn't block the release.

### 4. Minor: use-after-free at process exit inside the NVIDIA driver (test_flint)
Every test passes, but at `exit()` ASAN reports a heap-use-after-free between two of `libnvidia`'s own cleanup handlers. The memory was allocated by `vkCreateDevice` (vulkan.mojo:1128 via `create_device`, from `SceneRenderer.__init__` in `test_v11_subnative_tsr_sdf_shadows_weather_precip_and_dyn_reflect`, test_flint.mojo:2094). The likely cause is a `VkDevice` that is never destroyed before exit. Destroy the GPU or renderer at the end of that test.

## Update 09:54
- The e2e failure `test_f22_hiz_two_phase_cull_split` ("page (63, 63) maps to cascade 0 max slot 4095") seen at 09:31 is fixed by the 09:49 `aaa_vsm.mojo` change. E2E passes 505/505 at -O3.
- v5 render and v5 assets/studio/MCP are clean under ASAN at -O3. Under ASAN the render suite exceeds its own time ceiling (5.2 s), which is expected given the sanitizer's slowdown.
- Toolchain note: suites built with `-O1 -g1 --sanitize address` that use `spirv.emit` crash inside `libKGENCompilerRTShared`. This is a Mojo toolchain issue, not Flint. Use `-O3 -g1 --sanitize address`.
- Item 1 (`restore_blob` allocation) is still open; world.mojo is unchanged since 08:35.

## Before committing
Run `bash tools/flint.sh gate`. The gate order fix and the `integrate_velocities(0, ...)` e2e fix are already in the working tree; keep both.

## Update 10:25: fixes are ready to take
Items 1 and 4 are fixed and committed in flint-py's copy of Flint (`~/flint-py/vendor/flint`, branch `v5-flint-py`), on top of a snapshot of this working tree taken at 10:01:
- `a083194` fix(ecs): restore_blob validates counts and entity tables before allocating (adds suite (h) to test_flint_v5_core_ecs_scene.mojo)
- `6178774` fix(render): GpuSceneRenderer closes its Vulkan device when destroyed (`shutdown()` + `__deinit__`)

The full gate passes on that copy. To take them: `git -C ~/flint-py/vendor/flint format-patch -2 6178774`, then `git am` the patches here once v5 is committed. They only touch `src/flint/ecs/world.mojo`, `src/flint/render/scene_renderer.mojo` and the ECS test.

## Update 2026-11-18 evening: fixes and render features are applied to this working tree

The commits from flint-py's engine copy are now applied here as **uncommitted** changes, on
top of the v5 working tree (which had not changed since the 10:01 snapshot). Nothing was
committed on `main`; you still own the v5 commit. The same work, as separate commits, is on
the local branch `v5-flint-py` (8650aa9 snapshot, then 7 commits up to 671672b):

- a083194 fix(ecs): restore_blob validates counts and entity tables before allocating
- 6178774 fix(render): GpuSceneRenderer closes its Vulkan device when destroyed
- 9ce7893 feat(render): soft, denoised ray-traced sun shadows (new aaa_rt_shadow.mojo)
- 85b3eb1 feat(render): ray-traced refractive glass materials
- 1410d15 feat(render): shaped, calm water surfaces (round ponds, wave and foam control)
- 6cc3580 fix(render): glass is see-through even with every RT feature off
- 671672b feat(render): generic GPU particles (color/size/alpha over life, additive or alpha blend)

The engine tests gained ECS suite (h) and render Suite 7. If you edit these files, start
from what is on disk now; the patch is `git diff 8650aa9 671672b`.
