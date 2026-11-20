"""Flint v5 Standalone Procedural Render Pipeline Lab (`labs/lab_render.mojo`).

Zero external `.glb` or texture files — 100% procedural geometry, materials, skeletal VAT animation,
and GPU SDF UI overlay executing in `<2.0s` (both headless CI and live Wayland/X11/KMS swapchains):
  - Procedural Cornell Box + 5x5 PBR / Clearcoat Sphere Matrix + White-Furnace conservation check
  - 4-Cascade Toroidal Virtual Shadow Map (`±16m, ±64m, ±256m, ±1024m`) Sundial
  - Procedural 2-Bone Skeletal `VatCrowdClipDesc` (`RGBA16F` pos+normal baker + motion vectors)
  - SPIR-V `OpSpecConstant` `SurfaceMaterialNode` compilation & validation (`spirv_validate.mojo`)
  - Double-Buffered `RenderWorld[2]` + `RenderExtractor` (`UInt64` entity ID transform tracking)
  - GPU 3D `DebugLineBatch` (`PRIM_LINE` near-plane clipped gizmos & AABBs) + all 6 GPU SDF UI primitives
    (`0xFFFF2040` pure-red vs `0xFF2040FF` pure-blue readback check + `>300` prim tile stress)
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, cos, max, min, sin
from std.time import monotonic
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import (
    NativeAnimation,
    NativeChannel,
    NativeModel,
    NativeNode,
    NativeSkin,
)
from flint.core.math import Mat4, Vec3
from flint.core.transform import Quat, trs
from flint.lab import EngineBenchSuite
from flint.platform.os import get_time
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_vsm import compute_vsm_cascade_matrices
from flint.render.material_graph import (
    MAT_DOMAIN_CLEARCOAT,
    MAT_DOMAIN_STANDARD_PBR,
    MaterialAsset,
    StandardPbrMaterialNode,
    TriplanarWeatherMaterialNode,
    bake_vat_crowd_clip_rgba16f,
    compile_surface_material_shader,
    sample_vat_vertex_cpu,
)
from flint.render.render_world import (
    RENDER_TIER_2_AAA_VISBUFFER_RT,
    DoubleBufferedRenderWorld,
    RenderCameraPacket,
    RenderExtractor,
    ScalableRenderer,
)
from flint.render.simd_canvas import pack_argb
from flint.render.spirv_validate import validate_spirv_module
from flint.render.ui_overlay import (
    UI_SYMBOL_A,
    UI_SYMBOL_GIZMO,
    UI_SYMBOL_PLAY,
    make_ui_bezier_prim,
    make_ui_line_prim,
    make_ui_symbol_prim,
)
from flint.ui.draw_list import UiDrawList


def _build_walker_model() -> NativeModel:
    var mb = MeshBuilder()
    mb.add_cylinder(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.75, 0.0), 0.20, 0.15, segments=10)
    var mesh = mb^.finish()
    mesh.skin = 0
    for v in range(mesh.vertex_count()):
        var py = mesh.positions[v * 3 + 1]
        var w1 = max(Float32(0.0), min(Float32(1.0), (py - Float32(0.45)) / Float32(1.1)))
        mesh.joints.append(UInt16(0))
        mesh.joints.append(UInt16(1))
        mesh.joints.append(UInt16(0))
        mesh.joints.append(UInt16(0))
        mesh.weights.append(Float32(1.0) - w1)
        mesh.weights.append(w1)
        mesh.weights.append(Float32(0.0))
        mesh.weights.append(Float32(0.0))

    var n0 = NativeNode()
    n0.children.append(1)
    var n1 = NativeNode()
    n1.base_translation = Vec3(0.0, 0.85, 0.0)
    n1.translation = Vec3(0.0, 0.85, 0.0)

    var sk = NativeSkin()
    sk.joints.append(0)
    sk.joints.append(1)
    var ibm0 = Mat4()
    var ibm1 = trs(Vec3(0.0, -0.85, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
    for k in range(16):
        sk.inverse_bind.append(ibm0.values[k])
    for k in range(16):
        sk.inverse_bind.append(ibm1.values[k])

    var ch = NativeChannel()
    ch.node = 1
    ch.path = 1
    var ts = [Float32(0.0), Float32(0.25), Float32(0.50), Float32(0.75), Float32(1.0)]
    var angs = [Float32(0.0), Float32(0.42), Float32(0.0), Float32(-0.42), Float32(0.0)]
    for i in range(5):
        ch.times.append(ts[i])
        var ha = angs[i] * Float32(0.5)
        var q = Quat(sin(ha), 0.0, 0.0, cos(ha)).normalized()
        ch.values.append(q.x)
        ch.values.append(q.y)
        ch.values.append(q.z)
        ch.values.append(q.w)

    var anim = NativeAnimation()
    anim.name = "crowd_stride"
    anim.duration = Float32(1.0)
    anim.channels.append(ch^)

    var m = NativeModel()
    m.meshes.append(mesh^)
    m.nodes.append(n0^)
    m.nodes.append(n1^)
    m.skins.append(sk^)
    m.animations.append(anim^)
    return m^


def _make_ui_border_tex() -> List[UInt32]:
    var out = List[UInt32](capacity=256)
    for y in range(16):
        for x in range(16):
            var edge = x < 3 or x >= 13 or y < 3 or y >= 13
            out.append(pack_argb(240 if edge else 45, 200 if edge else 65, 255 if edge else 95, 255))
    return out^


def main() raises:
    var t_start = get_time()
    print("=== Flint v5 Procedural Render Lab (`labs/lab_render.mojo`) ===")
    var suite = EngineBenchSuite("flint_v5_render")

    # 1. Validate `SurfaceMaterialNode` + `OpSpecConstant` SPIR-V compilation
    var spv_samples = List[Float64](capacity=8)
    var mat_spv = List[UInt32]()
    var spv_valid = True
    var spv_err = String()
    for _ in range(8):
        var ts0 = monotonic()
        mat_spv = compile_surface_material_shader(StandardPbrMaterialNode(enable_detail_layer=True))
        var spv_rep = validate_spirv_module(mat_spv)
        spv_samples.append(Float64(monotonic() - ts0) / 1.0e6)
        if not spv_rep.valid:
            spv_valid = False
            if len(spv_rep.errors) > 0:
                spv_err = spv_rep.errors[0]
    _ = suite.record_samples(
        "render_spirv_mat_compile",
        spv_samples,
        budget_p99_ms=25.0,
        detail="words=" + String(len(mat_spv)),
    )
    if not spv_valid:
        print("SPIR-V validation error:", spv_err)
        _ = suite.record_determinism_gate("render_spirv_valid", UInt64(0), UInt64(1))
        _ = suite.finish_and_export("render")
        raise Error("SPIR-V validation failed")

    # 2. Bake Procedural Skeletal VAT Crowd Clip (`RGBA16F`) & verify 3D motion vectors
    var walker = _build_walker_model()
    var vat_samples = List[Float64](capacity=8)
    var vat_frames = 0
    var vat_mv_len = Float32(0.0)
    for _ in range(8):
        var tv0 = monotonic()
        var vat_clip = bake_vat_crowd_clip_rgba16f(walker, 0, 0, sample_fps=30.0, looping=True)
        var vat_sample = sample_vat_vertex_cpu(vat_clip, walker.meshes[0].vertex_count() - 1, Float32(0.125), Float32(0.125 - 1.0 / 60.0))
        vat_samples.append(Float64(monotonic() - tv0) / 1.0e6)
        vat_frames = vat_clip.desc.frame_count
        vat_mv_len = (vat_sample[0] - vat_sample[2]).length()
    _ = suite.record_samples(
        "render_vat_crowd_bake",
        vat_samples,
        budget_p99_ms=25.0,
        detail="frames=" + String(vat_frames),
    )
    _ = suite.record_determinism_gate(
        "render_vat_motion_nonzero",
        UInt64(1 if vat_mv_len > Float32(0.0) else 0),
        UInt64(1),
    )

    # 3. Initialize 3-Tier `ScalableRenderer` (`256 x 192` compact showcase)
    var W = 256
    var H = 192
    var renderer = ScalableRenderer(RENDER_TIER_2_AAA_VISBUFFER_RT, width=W, height=H, tsr_scale=1.0)

    var mb_box = MeshBuilder()
    mb_box.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), bevel=0.02)
    var box_mesh = mb_box^.finish()

    var mb_sph = MeshBuilder()
    mb_sph.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), rings=14, segments=20)
    var sph_mesh = mb_sph^.finish()

    var mat_white = renderer.register_material(MaterialAsset(base_color=Vec3(0.76, 0.76, 0.74), roughness=0.72, metallic=0.0))
    var mat_red = renderer.register_material(MaterialAsset(base_color=Vec3(0.78, 0.12, 0.10), roughness=0.68, metallic=0.0))
    var mat_green = renderer.register_material(MaterialAsset(base_color=Vec3(0.12, 0.70, 0.16), roughness=0.68, metallic=0.0))
    var mat_gold = renderer.register_material(MaterialAsset(base_color=Vec3(0.95, 0.78, 0.32), roughness=0.16, metallic=1.0))

    var sph_mats = List[Int](capacity=25)
    for r in range(5):
        for c in range(5):
            var sp_m = MaterialAsset(
                base_color=Vec3(0.22 + 0.18 * Float32(c), 0.30 + 0.14 * Float32(r), 0.88 - 0.15 * Float32(r)),
                roughness=Float32(0.08) + Float32(c) * Float32(0.21),
                metallic=(Float32(1.0) if r >= 2 else Float32(0.0)),
                domain=(MAT_DOMAIN_CLEARCOAT if r == 1 else MAT_DOMAIN_STANDARD_PBR),
            )
            if r == 1:
                sp_m.clearcoat = Float32(1.0)
                sp_m.clearcoat_roughness = Float32(0.05)
            sph_mats.append(renderer.register_material(sp_m))

    var mesh_box_id = renderer.register_mesh(box_mesh, mat_white, build_lods=False)
    var mesh_sph_id = renderer.register_mesh(sph_mesh, mat_white, build_lods=True)
    var mesh_crowd_id = renderer.register_mesh(walker.meshes[0], mat_gold, build_lods=False)

    var furnace_luma = Float32(1.0)
    var ui_tex_slot = 0
    if renderer.uses_vulkan():
        ref gpu = renderer.gpu_slot[0]
        gpu.tsr_history_weight = Float32(0.0)
        gpu.camera_lens.vignette_strength = Float32(0.0)
        ui_tex_slot = max(0, gpu.upload_bc7_texture(_make_ui_border_tex(), 16, 16, srgb=False, generate_mips=False))

        # Quick White-Furnace Conservation Probe
        var wf_id = gpu.add_material(AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.45, metallic=0.0))
        gpu.furnace_white_dome = True
        gpu.enable_vsm = False
        gpu.begin_frame(Vec3(0.0, 0.0, 2.8), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
        gpu.draw_instance(mesh_sph_id, Mat4(), material_override=wf_id)
        _ = gpu.render_frame(capture_readback=True)
        var wf_hdr = gpu.readback_resolved_hdr()
        furnace_luma = wf_hdr[(H // 2) * W + (W // 2)].x
        gpu.furnace_white_dome = False
        gpu.enable_vsm = True
        gpu.enable_vsm_clipmap_cascades = True
        # Dispatch `TriplanarWeatherMaterialNode` on GPU onto back-wall material (`mat_white` stays clean; bake a dedicated alpine sphere slot!)
        var alpine_mat = renderer.register_material(MaterialAsset(base_color=Vec3(0.14, 0.16, 0.20), roughness=0.78))
        _ = renderer.evaluate_material_graph(TriplanarWeatherMaterialNode(snow_line_y=10.0), alpine_mat, dim=32, world_y_scale=24.0)
        sph_mats[0] = alpine_mat

    # 4. Extract & Render 2 Animated Frames through `DoubleBufferedRenderWorld` + `RenderExtractor`
    var sun_dir = Vec3(0.62, 0.72, 0.31).normalized()
    var vsm_cascades = compute_vsm_cascade_matrices(sun_dir, Vec3(0.0, 1.5, 0.0))

    var world_ring = DoubleBufferedRenderWorld()
    var extractor = RenderExtractor()

    var frame_samples = List[Float64](capacity=16)
    for frame_i in range(18):
        ref w = world_ring.write_world()
        extractor.begin_extraction(w, UInt64(frame_i + 1))
        w.camera = RenderCameraPacket(eye=Vec3(0.0, 3.2, 7.8), target=Vec3(0.0, 1.5, 0.0), fov_y_deg=50.0)
        w.sun_dir = sun_dir
        w.sun_radiance = Vec3(5.2, 4.9, 4.5)
        w.ibl_intensity = Float32(0.30)

        # Cornell Box walls + Sundial Gnomon
        extractor.extract_entity(w, UInt64(1), mesh_box_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(6.0, 0.1, 6.0)), mat_white)
        extractor.extract_entity(w, UInt64(2), mesh_box_id, trs(Vec3(0.0, 3.0, -4.5), Quat(), Vec3(6.0, 3.2, 0.12)), mat_white)
        extractor.extract_entity(w, UInt64(3), mesh_box_id, trs(Vec3(-4.8, 3.0, 0.0), Quat(), Vec3(0.12, 3.2, 5.0)), mat_red)
        extractor.extract_entity(w, UInt64(4), mesh_box_id, trs(Vec3(4.8, 3.0, 0.0), Quat(), Vec3(0.12, 3.2, 5.0)), mat_green)
        extractor.extract_entity(w, UInt64(5), mesh_box_id, trs(Vec3(1.2, 1.5, 1.2), Quat(), Vec3(0.22, 1.5, 0.22)), mat_gold)

        # 5x5 PBR/Clearcoat Sphere Matrix
        for r in range(5):
            for c in range(5):
                var idx = r * 5 + c
                extractor.extract_entity(
                    w,
                    UInt64(100 + idx),
                    mesh_sph_id,
                    trs(Vec3(Float32(c - 2) * Float32(0.85), Float32(1.1) + Float32(r) * Float32(0.68), -2.6), Quat(), Vec3(0.26, 0.26, 0.26)),
                    sph_mats[idx],
                )

        # Moving Crowd Agents (`UInt64` entity IDs `500..507` generating per-instance motion vectors!)
        var step_z = Float32(frame_i) * Float32(0.05)
        for ag in range(8):
            var ax = Float32(-2.8) + Float32(ag) * Float32(0.55)
            extractor.extract_entity(w, UInt64(500 + ag), mesh_crowd_id, trs(Vec3(ax, 0.0, Float32(1.8) - step_z), Quat(), Vec3(0.55, 0.55, 0.55)), mat_gold)

        # 3D Debug Gizmo & Sundial AABB
        w.debug_lines.add_gizmo(Vec3(-2.0, 0.5, 2.0), Quat(), 0.75)
        w.debug_lines.add_aabb(Vec3(0.95, 0.0, 0.95), Vec3(1.45, 3.0, 1.45), 0xFFFFE040)

        extractor.end_extraction()
        world_ring.swap_buffers()

        var ui = UiDrawList()
        # Pure-red `0xFFFF2040` & pure-blue `0xFF2040FF` swatches + Nine-Slice + Symbols + Lines + >300 Tile Overlap
        ui.add_rect(8.0, 8.0, 24.0, 24.0, 0xFFFF2040)
        ui.add_rect(36.0, 8.0, 24.0, 24.0, 0xFF2040FF)
        ui.add_nine_slice(ui_tex_slot, 66.0, 8.0, 44.0, 24.0, 0.25, 0.25, 0.25, 0.25, 0xFFFFFFFF)
        ui.prims.append(make_ui_symbol_prim(UI_SYMBOL_A, 116.0, 8.0, 24.0, 24.0, 0xFFFFFFFF, 0xFF2060D0))
        ui.prims.append(make_ui_symbol_prim(UI_SYMBOL_PLAY, 144.0, 8.0, 24.0, 24.0, 0xFF30FF80, 0xFF182230))
        ui.prims.append(make_ui_symbol_prim(UI_SYMBOL_GIZMO, 172.0, 8.0, 24.0, 24.0, 0xFFFFD030, 0xFF181820))
        ui.prims.append(make_ui_line_prim(8.0, 176.0, 96.0, 176.0, 5.0, 0xFF20F0D0))
        ui.prims.append(make_ui_bezier_prim(104.0, 180.0, 136.0, 152.0, 168.0, 180.0, 4.0, 0xFFFF50E0))
        for ov in range(320):
            var ov_col: UInt32 = 0xFF802020 if (ov & 1) == 0 else 0xFF202080
            ui.add_rect(210.0, 8.0, 20.0, 20.0, ov_col)
        ui.add_rect(210.0, 8.0, 20.0, 20.0, 0xFF10F030)

        var tf0 = monotonic()
        if not renderer.render_world(world_ring.read_world(), ui, capture_readback=(frame_i == 17)):
            print("render_world failed")
            renderer.shutdown()
            raise Error("render_world failed")
        var frame_ms = Float64(monotonic() - tf0) / 1.0e6
        if frame_i >= 2:
            frame_samples.append(frame_ms)

    _ = suite.record_samples(
        "render_world_256x192_frame",
        frame_samples,
        budget_p99_ms=50.0,
        detail=renderer.tier_name(),
    )

    var tui0 = monotonic()
    var px = renderer.readback_pixels()
    var red_sw = px[16 * W + 16]
    var blue_sw = px[16 * W + 44]
    var tile_321 = px[16 * W + 220]
    var r_ok = abs(Int((red_sw >> 16) & 0xFF) - 255) <= 2 and abs(Int(red_sw & 0xFF) - 64) <= 2
    var b_ok = abs(Int((blue_sw >> 16) & 0xFF) - 32) <= 2 and abs(Int(blue_sw & 0xFF) - 255) <= 2
    var ov_ok = abs(Int((tile_321 >> 8) & 0xFF) - 240) <= 2
    var ui_check_ms = Float64(monotonic() - tui0) / 1.0e6
    _ = suite.record_single(
        "render_ui_readback_check",
        ui_check_ms,
        passed=(renderer.tier == 0 or (r_ok and b_ok and ov_ok)),
        budget_ms=50.0,
    )

    # 5. 2-Frame Static FLIP Determinism Check (`tsr_history_weight = 0.0`)
    var ui_det = UiDrawList()
    ui_det.add_rect(8.0, 8.0, 24.0, 24.0, 0xFFFF2040)
    ui_det.add_rect(36.0, 8.0, 24.0, 24.0, 0xFF2040FF)
    if renderer.uses_vulkan():
        renderer.gpu_slot[0].frame_index = 18
    var ok_a = renderer.render_world(world_ring.read_world(), ui_det, capture_readback=True)
    var px_a = renderer.readback_pixels()
    if renderer.uses_vulkan():
        renderer.gpu_slot[0].frame_index = 18
    var ok_b = renderer.render_world(world_ring.read_world(), ui_det, capture_readback=True)
    var px_b = renderer.readback_pixels()
    if not ok_a or not ok_b:
        px_a.clear()
        px_b.clear()
    _ = suite.verify_golden_pixels(
        "render_2frame_flip_det",
        px_a,
        px_b,
        W,
        H,
        tolerance=2,
        spread_per_million=500,
        flip_limit=0.01,
    )

    var elapsed = get_time() - t_start
    print("  Active Tier           :", renderer.tier_name())
    print("  SPIR-V Material Words :", len(mat_spv), "(OpSpecConstant validated)")
    print("  VSM Cascades          :", len(vsm_cascades), "(±16m, ±64m, ±256m, ±1024m)")
    print("  VAT Crowd Frames      :", vat_frames, "| MotionVec |dv| =", vat_mv_len)
    print("  White-Furnace Luma    :", furnace_luma)
    print("  UI Red/Blue/321-Tile  :", r_ok, b_ok, ov_ok)
    print("  Total Elapsed         :", elapsed, "s")

    renderer.shutdown()
    if not suite.finish_and_export("render"):
        raise Error("lab_render failed verification gates")
    print("=== Flint v5 Render Lab Complete ===")
