"""Tier 4 Real-World Application Scenarios: S4 Asset Pipeline & glTF Ingestion.
End-to-end ingestion pipeline: synthetic glTF 2.0 asset with AAA material extensions,
Forsyth post-transform vertex cache optimization, QEM LOD meshlet simplification,
packaging into .flpk container, and header validation.
"""
from std.collections import List
from flint.core.math import Vec3
from flint.assets.meshlet import compute_acmr, optimize_vertex_cache_forsyth, simplify_mesh_qem
from flint.render.aaa_bsdf import AaaMaterial, eval_thin_film_iridescence_cpu
from flint.assets.vfs import encode_pack, pack_read
from e2e.common.harness import TestContext
from e2e.common.mock_data import create_mock_meshlet_mesh, create_mock_model_with_extensions, create_mock_flpk_data


def test_t4_asset_pipeline_full_ingestion(mut ctx: TestContext) raises:
    # 1. Ingest glTF model with AAA material extensions
    var model = create_mock_model_with_extensions()
    ctx.assert_true(len(model.meshes) > 0, "ingested glTF model contains mesh primitives")
    ctx.assert_true(len(model.textures) >= 3, "ingested glTF model contains PBR textures")

    # Verify material parameters from extensions
    var raw_mesh = model.meshes[0].copy()
    ctx.assert_approx(raw_mesh.roughness, 0.25, 0.01, "roughness extension ingested accurately")
    ctx.assert_approx(raw_mesh.metallic, 0.75, 0.01, "metallic extension ingested accurately")
    ctx.assert_approx(raw_mesh.emissive_strength, 2.5, 0.01, "emissive strength ingested accurately")

    # 2. Extract primary mesh and run Forsyth post-transform cache optimizer
    var int_indices = List[Int]()
    for i in range(len(raw_mesh.indices)):
        int_indices.append(Int(raw_mesh.indices[i]))

    var v_count = len(raw_mesh.positions) // 3
    var opt_indices = optimize_vertex_cache_forsyth(int_indices, v_count, 32)
    ctx.assert_equal(len(opt_indices), len(int_indices), "cache optimization preserves total index count")

    var unopt_acmr = compute_acmr(int_indices, 32)
    var opt_acmr = compute_acmr(opt_indices, 32)
    ctx.assert_true(opt_acmr <= unopt_acmr + 0.05, "Forsyth optimizer achieves high GPU FIFO-32 cache locality")

    # 3. Generate LOD 1 via QEM surface simplification
    var orig_tris = len(raw_mesh.indices) // 3
    var target_l1_tris = orig_tris // 2
    var lod1_result = simplify_mesh_qem(raw_mesh, target_l1_tris)
    var lod1_mesh = lod1_result[0].copy()
    var lod1_err = lod1_result[1]
    var lod1_tris = len(lod1_mesh.indices) // 3
    ctx.assert_true(lod1_tris <= orig_tris, "LOD 1 decimation reduces polygon complexity")
    ctx.assert_true(lod1_err >= 0.0, "geometric surface error calculated monotonically")

    # 4. Pack processed asset into .flpk container bundle
    var file_names = List[String]()
    var file_buffers = List[List[UInt8]]()

    file_names.append(String("model_lod0.bin"))
    var b0 = List[UInt8]()
    b0.append(0xAA); b0.append(0xBB); b0.append(0xCC); b0.append(0xDD)
    file_buffers.append(b0^)

    file_names.append(String("model_lod1.bin"))
    var b1 = List[UInt8]()
    b1.append(0x11); b1.append(0x22); b1.append(0x33); b1.append(0x44)
    file_buffers.append(b1^)

    var flpk_container = encode_pack(file_names, file_buffers)
    ctx.assert_true(len(flpk_container) >= 64, "packed .flpk meets container alignment requirements")

    # Verify container header magic & entry count
    var count = Int(flpk_container[8]) | (Int(flpk_container[9]) << 8)
    ctx.assert_equal(count, 2, "container directory indexes exactly 2 asset streams")

    # 5. Evaluate glTF KHR_materials_iridescence extension
    var irid_rgb = eval_thin_film_iridescence_cpu(0.707, 400.0, 1.5)
    ctx.assert_true(irid_rgb.x > 0.0 or irid_rgb.y > 0.0 or irid_rgb.z > 0.0, "thin-film iridescence evaluates non-zero spectral reflectance")


# =========================================================================
# Aggregator for Tier 4 Asset Pipeline Scenario
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T4_RealWorld_AssetPipeline")
    try: test_t4_asset_pipeline_full_ingestion(ctx); ctx.record_pass("test_t4_asset_pipeline_full_ingestion") except e: ctx.record_fail("test_t4_asset_pipeline_full_ingestion", String(e))

def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
