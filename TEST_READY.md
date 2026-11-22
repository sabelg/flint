# E2E Test Suite Ready

## Test Runner
- Command: `bash tests/e2e_opaque/run_e2e_opaque.sh --strict`
- Default (live-only) Command: `bash tests/e2e_opaque/run_e2e_opaque.sh`
- Expected: All 11 live suites (140 checks) and all 4 M1–M6 contract suites (29 checks) compile cleanly with `-I src -I tests -O3 --fp-mode contract=off --Werror` and pass `169/169` with exit code `0` (`[SUMMARY] live=11/11 passed (0 failed), gated=4/4 passed (0 pending, 0 failed), strict=1`).

## Coverage Summary
| Tier | Count | Description |
|------|------:|-------------|
| 1. Feature Coverage | 61 | >=6–11 per feature area (FA1: 11, FA2: 8, FA3: 7, FA4: 7, FA5: 7, FA6: 7, FA7: 8, FA8: 6) |
| 2. Boundary & Corner | 58 | >=7–8 per feature area (FA1: 7, FA2: 8, FA3: 7, FA4: 7, FA5: 7, FA6: 7, FA7: 8, FA8: 7) |
| 3. Cross-Feature | 10 | Pairwise combinations (`test_t3_cross_feature.mojo`) |
| 4. Real-World Application | 11 | Multi-step workflows (`test_t4_workflows.mojo`: 5, `test_t4_cli_workflows.sh`: 6) |
| M1–M6 Contract Suites (`gated/`) | 29 | End-to-end M1–M6 contracts (`F1`–`F34`: M1 9, M2 5, M3/M4 7, M5/M6 8) |
| **Total** | **169** | **15 suites (`11` live + `4` gated), all `--Werror` green** |

## Feature Checklist
| Feature Area | Tier 1 | Tier 2 | Tier 3 | Tier 4 | M1–M6 Contract |
|--------------|-------:|-------:|:------:|:------:|:--------------:|
| FA1. Engine CLI, Build, Scaffolding & Packaging (`test_cli_e2e.sh`, `test_t4_cli_workflows.sh`, `gated/test_gated_m5_m6_cli.sh`) | 11 | 7 | ✓ | ✓ | 8 (F29–F34) |
| FA2. App Runtime, ECS, Scene Graph & Jobs/Math (`test_t1_core_net.mojo`, `test_t2_core_net.mojo`, `gated/test_gated_m1_core_net.mojo`) | 8 | 8 | ✓ | ✓ | 5 (F1–F5) |
| FA3. Simulation: Physics, Vehicle, Aero, Cloth, Ragdoll & Anim (`test_t1_sim_render.mojo`, `test_t2_sim_render.mojo`, `gated/test_gated_m2_plugins.mojo`) | 7 | 7 | ✓ | ✓ | 5 (F10–F14) |
| FA4. Rendering, GPU Residency & Verification Metrics (`test_t1_sim_render.mojo`, `test_t2_sim_render.mojo`, `gated/test_gated_m3_m4_gpu_plat.mojo`) | 7 | 7 | ✓ | ✓ | ✓ (F15–F20, F22) |
| FA5. Asset Pipeline, Codecs & VFS (`test_t1_assets_audio_ui.mojo`, `test_t2_assets_audio_ui.mojo`, `gated/test_gated_m3_m4_gpu_plat.mojo`) | 7 | 7 | ✓ | ✓ | ✓ (F20–F21) |
| FA6. Audio, Input, Windowing, Text, UI & Multi-OS (`test_t1_assets_audio_ui.mojo`, `test_t2_assets_audio_ui.mojo`, `gated/test_gated_m3_m4_gpu_plat.mojo`) | 7 | 7 | ✓ | ✓ | ✓ (F23–F28) |
| FA7. Networking, Rollback & Replication (`test_t1_core_net.mojo`, `test_t2_core_net.mojo`, `gated/test_gated_m1_core_net.mojo`) | 8 | 8 | ✓ | ✓ | 4 (F6–F9) |
| FA8. Studio, DevTools & Bench/Perf Gates (`test_t1_t2_studio_lab.mojo`, `test_t4_workflows.mojo`, `gated/test_gated_m5_m6_cli.sh`) | 6 | 7 | ✓ | ✓ | ✓ (F30, F32) |

## Suite Inventory
- `tests/e2e_opaque/run_e2e_opaque.sh` — Master opaque-box runner (`--strict`, `--filter <substr>`, `-O3 --fp-mode contract=off --Werror`)
- `tests/e2e_opaque/test_cli_e2e.sh` — FA1 Tiers 1–2 (`18/18`)
- `tests/e2e_opaque/test_t1_core_net.mojo` — FA2 & FA7 Tier 1 (`16/16`)
- `tests/e2e_opaque/test_t2_core_net.mojo` — FA2 & FA7 Tier 2 (`16/16`)
- `tests/e2e_opaque/test_t1_sim_render.mojo` — FA3 & FA4 Tier 1 (`14/14`)
- `tests/e2e_opaque/test_t2_sim_render.mojo` — FA3 & FA4 Tier 2 (`14/14`)
- `tests/e2e_opaque/test_t1_assets_audio_ui.mojo` — FA5 & FA6 Tier 1 (`14/14`)
- `tests/e2e_opaque/test_t2_assets_audio_ui.mojo` — FA5 & FA6 Tier 2 (`14/14`)
- `tests/e2e_opaque/test_t1_t2_studio_lab.mojo` — FA8 Tiers 1–2 (`13/13`)
- `tests/e2e_opaque/test_t3_cross_feature.mojo` — Tier 3 Pairwise Cross-Feature (`10/10`)
- `tests/e2e_opaque/test_t4_workflows.mojo` — Tier 4 Real-World Mojo Workflows (`5/5`)
- `tests/e2e_opaque/test_t4_cli_workflows.sh` — Tier 4 Real-World CLI Production Pipeline (`6/6`)
- `tests/e2e_opaque/gated/test_gated_m1_core_net.mojo` — M1 F1–F9 (`9/9`)
- `tests/e2e_opaque/gated/test_gated_m2_plugins.mojo` — M2 F10–F14 (`5/5`)
- `tests/e2e_opaque/gated/test_gated_m3_m4_gpu_plat.mojo` — M3/M4 F15–F28 (`7/7`)
- `tests/e2e_opaque/gated/test_gated_m5_m6_cli.sh` — M5/M6 F29–F34 (`8/8`)

## Notes for M7-Final
- All 169 tests across live and gated suites pass under `bash tests/e2e_opaque/run_e2e_opaque.sh --strict` (verified in `9m46.903s`, exit `0`, zero leaked temp directories).
- Out-of-scope UI finding flagged during E2E-M4 adversarial probing for M7 Phase 2 Tier 5 hardening: `src/flint/ui/context.mojo:1093` (`UiContext.button` triggers on drag-from-outside release when `hovered and self.mouse_released_l` without checking `self.active_id == wid`).
