"""Flint Engine E2E Test Suite Master Runner.

Aggregates and executes all 505 tests across Tiers 1–5:
- Tier 1: Feature Coverage (F01–F43: 215 tests).
- Tier 2: Boundary & Corner Cases (F01–F43: 215 tests).
- Tier 3: Cross-Feature Pairwise Interactions (45 tests).
- Tier 4: Real-World Application Scenarios (5 tests).
- Tier 5: Adversarial Coverage Hardening (25 tests).
"""
from e2e.common.harness import TestContext

# Tier 1 Feature Coverage
import e2e.tier1_coverage.test_t1_m1_phase0 as t1_m1
import e2e.tier1_coverage.test_t1_m2_gpu_loop as t1_m2
import e2e.tier1_coverage.test_t1_m3_rt_streaming as t1_m3
import e2e.tier1_coverage.test_t1_m4_world_sim as t1_m4
import e2e.tier1_coverage.test_t1_m5_compat_gate as t1_m5

# Tier 2 Boundary & Corner Cases
import e2e.tier2_boundary.test_t2_m1_phase0 as t2_m1
import e2e.tier2_boundary.test_t2_m2_gpu_loop as t2_m2
import e2e.tier2_boundary.test_t2_m3_rt_streaming as t2_m3
import e2e.tier2_boundary.test_t2_m4_world_sim as t2_m4
import e2e.tier2_boundary.test_t2_m5_compat_gate as t2_m5

# Tier 3 Cross-Feature Combinations
import e2e.tier3_cross_feature.test_t3_render_pipeline as t3_render
import e2e.tier3_cross_feature.test_t3_physics_anim as t3_physics
import e2e.tier3_cross_feature.test_t3_platform_assets as t3_platform

# Tier 4 Real-World Application Scenarios
import e2e.tier4_real_world.test_t4_neondrift_scenario as t4_neondrift
import e2e.tier4_real_world.test_t4_rollercoaster_scenario as t4_rollercoaster
import e2e.tier4_real_world.test_t4_openworld_scenario as t4_openworld
import e2e.tier4_real_world.test_t4_asset_pipeline_scenario as t4_asset
import e2e.tier4_real_world.test_t4_character_combat_scenario as t4_combat

# Tier 5 Adversarial Coverage Hardening
import e2e.tier5_adversarial.test_t5_render_assets as t5_render_assets
import e2e.tier5_adversarial.test_t5_physics_anim_platform as t5_phys_anim


def main() raises:
    print("============================================================")
    print("FLINT ENGINE COMPREHENSIVE E2E TEST SUITE (TIERS 1 - 5)")
    print("Pure Mojo 1.2 Opaque-Box Requirement Verification")
    print("============================================================")

    var ctx = TestContext()

    # Tier 1 Feature Coverage (215 tests)
    t1_m1.run_all(ctx)
    t1_m2.run_all(ctx)
    t1_m3.run_all(ctx)
    t1_m4.run_all(ctx)
    t1_m5.run_all(ctx)

    # Tier 2 Boundary & Corner Cases (215 tests)
    t2_m1.run_all(ctx)
    t2_m2.run_all(ctx)
    t2_m3.run_all(ctx)
    t2_m4.run_all(ctx)
    t2_m5.run_all(ctx)

    # Tier 3 Cross-Feature Pairwise Interactions (45 tests)
    t3_render.run_all(ctx)
    t3_physics.run_all(ctx)
    t3_platform.run_all(ctx)

    # Tier 4 Real-World Application Scenarios (5 tests)
    t4_neondrift.run_all(ctx)
    t4_rollercoaster.run_all(ctx)
    t4_openworld.run_all(ctx)
    t4_asset.run_all(ctx)
    t4_combat.run_all(ctx)

    # Tier 5 Adversarial Coverage Hardening (25 tests)
    t5_render_assets.run_all(ctx)
    t5_phys_anim.run_all(ctx)

    ctx.print_summary()

    if ctx.failed_tests > 0:
        print("[FAIL] E2E test suite failed with " + String(ctx.failed_tests) + " failures.")
        raise Error("E2E tests failed")
    else:
        print("[PASS] All " + String(ctx.total_tests) + " E2E tests across Tiers 1-5 completed successfully with 0 errors!")
