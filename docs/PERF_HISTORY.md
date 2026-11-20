# Flint Engine v5 Performance History & Regression Gate (`docs/PERF_HISTORY.md`)

## Gate Rules (`src/flint/core/perf_history.mojo`)
- **History Table**: `docs/PERF_HISTORY.csv` stores one row per recorded release/baseline (`date`, `commit`, `package`, `result`, plus `<case>_p50_ms` and `<case>_p99_ms` columns emitted by `EngineBenchSuite` across all 5 subsystem labs).
- **Regression Limit**: `10%` (`REGRESSION_LIMIT = 0.10`) above the median of the last `BASELINE_ROWS = 3` passing or accepted rows for that column, provided the delta also exceeds the metric's calibrated noise floor (`floor`).
- **Absolute Budget**: Any metric with `budget > 0` fails the gate immediately if its measured value exceeds `budget`.
- **Memory Growth Limit**: Memory metrics use `MEMORY_GROWTH_LIMIT = 0.02` (`2%`).
- **Drift Warning**: `DRIFT_WARNING = 0.25` reports >25% drift since phase start / `v5.0` baseline as a summary warning without failing the gate.
- **Normal Gate Hygiene**: `bash tools/flint.sh gate` validates current `build/labs/lab_*_perf.csv` runs against `docs/PERF_HISTORY.csv` without modifying the committed CSV. Set `FLINT_PERF_RECORD=1` when intentionally appending a new recorded baseline row.

## Calibrated Subsystem Budgets & Noise Floors (`v5.0` `-O3 --Werror`)
Noise floors are set to at least `1.5 x` the observed min-to-max range (`NOISE_RANGES = 1.5`) across repeated `-O3` runs on the reference Vulkan + Linux workstation.

| Metric | Lab Source | Noise Floor (ms) | p99 / Abs Budget (ms) |
|---|---|---:|---:|
| `physics_200body_step_p50_ms` | `lab_physics` | 0.05 | 2.00 |
| `physics_200body_step_p99_ms` | `lab_physics` | 0.50 | 4.00 |
| `physics_rollback_det_120_300_p99_ms` | `lab_physics` | 90.00 | 1000.00 |
| `physics_shapes_vhacd_dda_p99_ms` | `lab_physics` | 0.05 | 2.00 |
| `physics_kcc_stair_90t_p99_ms` | `lab_physics` | 0.02 | 0.25 |
| `nav_obstacle_carve_p99_ms` | `lab_physics` | 0.02 | 0.25 |
| `nav_orca_256_tick_p50_ms` | `lab_physics` | 0.10 | 2.00 |
| `nav_orca_256_tick_p99_ms` | `lab_physics` | 0.15 | 3.00 |
| `anim_flanim_roundtrip_p99_ms` | `lab_anim` | 0.05 | 0.50 |
| `anim_socket_foot_ik_p99_ms` | `lab_anim` | 0.05 | 0.50 |
| `anim_ragdoll_sm_tick_p99_ms` | `lab_anim` | 0.09 | 0.50 |
| `ai_bt_utility_perc_cam_p99_ms` | `lab_anim` | 0.02 | 0.50 |
| `anim_ai_state_det_p99_ms` | `lab_anim` | 7.00 | 50.00 |
| `render_spirv_mat_compile_p99_ms` | `lab_render` | 0.20 | 25.00 |
| `render_vat_crowd_bake_p99_ms` | `lab_render` | 0.10 | 25.00 |
| `render_world_256x192_frame_p50_ms` | `lab_render` | 0.25 | 25.00 |
| `render_world_256x192_frame_p99_ms` | `lab_render` | 0.30 | 50.00 |
| `render_ui_readback_check_p99_ms` | `lab_render` | 0.05 | 50.00 |
| `audio_qoa_codec_p99_ms` | `lab_audio_ui` | 0.90 | 5.00 |
| `audio_bvh_acoustics_p99_ms` | `lab_audio_ui` | 0.02 | 1.00 |
| `audio_server_quantum_p99_ms` | `lab_audio_ui` | 0.15 | 2.00 |
| `ui_dockspace_100k_frame_p99_ms` | `lab_audio_ui` | 0.02 | 0.50 |
| `input_lockstep_det_p99_ms` | `lab_audio_ui` | 0.02 | 5.00 |
| `udp_dual_stack_32_mmsg_p99_ms` | `lab_net` | 0.30 | 50.00 |
| `bitpack_10k_quat32_fuzz_p99_ms` | `lab_net` | 6.00 | 50.00 |
| `channel_reliable_8kib_loss12_p99_ms` | `lab_net` | 0.50 | 50.00 |
| `rollback_8tick_resim_latency_p99_ms` | `lab_net` | 0.02 | 1.20 |
| `rollback_4peer_10000_ticks_p99_ms` | `lab_net` | 25.00 | 1500.00 |
| `physics_world_rollback_spoof_auth_p99_ms` | `lab_net` | 1.00 | 1000.00 |
| `replicator_16c_256e_tick_p50_ms` | `lab_net` | 0.05 | 3.00 |
| `replicator_16c_256e_tick_p99_ms` | `lab_net` | 0.05 | 5.00 |
| `replicator_bandwidth_gate_24kbs_p99_ms` | `lab_net` | 1.50 | 100.00 |
| `lab_harness_flip_and_csv_p99_ms` | `lab_net` | 0.10 | 50.00 |

## Accepted Performance Changes
Add lines in the format `- accept <package> <column|all>: <why>` below when an intentional feature trade-off alters a baseline metric.
