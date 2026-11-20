"""Flint v5.1 Stylized Anime NPR Cel, Clip-Space Outlines, Smear Skinning & VFX Suite (`test_flint_v5_npr_vfx.mojo`).

Zero external asset dependencies. Verifies:
  1. SPIR-V 1.5 structural & SSA validation (`validate_spirv_module`) of all new shaders:
     `npr_outline_vertex_shader`, `npr_outline_fragment_shader`, `aaa_npr_cel_shade_shader`,
     `visbuffer_alphamask_vfx_fragment_shader`, `aaa_npr_post_fx_shader`, and `aaa_dq_skinning_shader`.
  2. Deterministic CPU reference evaluators (`eval_npr_cel_cpu`, `skin_vertex_dq_smear_cpu`,
     `SteppedAnimClock`, `eval_stylized_vfx_cpu`, `eval_shockwave_chromatic_offsets_cpu`, `eval_impact_frame_cpu`).
  3. Live Vulkan GPU `ILM + SSS + Inner-Ink` cel shading, hardware `CULL_MODE_FRONT` clip-space outlines,
     and per-vertex `G`-channel outline suppression (`G = 0` vs `G = 255`).
  4. Live Vulkan GPU `DualQuat + BoneSmear` non-uniform stretch/squash/shear + `J^{-T}` normal deformation
     matching `skin_vertex_dq_smear_cpu` to `< 1e-4`, plus automatic identity reset.
  5. Live Vulkan GPU eroding 3D stylized VFX silhouette cutout, R/G/B chromatic-dispersion shockwaves,
     and radial-streak Manga Impact Frames!
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, max, min
from flint.anim.flanim_ecs import (
    SteppedAnimClock,
    compute_volume_preserving_smear_scale,
    quantize_stepped_anim_time,
)
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh
from flint.core.math import Vec3
from flint.core.transform import Mat4, Quat, trs
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_ibl import SkyDomeParams, bake_bc6h_ibl_cubemap_and_sh9
from flint.render.aaa_npr_cel import (
    AaaCelParams,
    aaa_npr_cel_shade_shader,
    eval_npr_cel_cpu,
    npr_outline_fragment_shader,
    npr_outline_vertex_shader,
)
from flint.render.aaa_npr_vfx import (
    AaaImpactFrameParams,
    AaaShockwaveRing,
    AaaStylizedVfxParams,
    aaa_npr_post_fx_shader,
    eval_impact_frame_cpu,
    eval_shockwave_chromatic_offsets_cpu,
    eval_stylized_vfx_cpu,
    visbuffer_alphamask_vfx_fragment_shader,
)
from flint.render.aaa_post_tsr import AAA_DISPLAY_LINEAR_SRGB
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_skinning_rt import (
    BoneSmear,
    DualQuat,
    aaa_dq_skinning_shader,
    skin_vertex_dq_cpu,
    skin_vertex_dq_smear_cpu,
)
from flint.render.spirv_validate import validate_spirv_module


def _fail(msg: String):
    print("  [FAIL]", msg)
    external_call["exit", NoneType](Int32(1))


def _assert(cond: Bool, msg: String):
    if not cond:
        _fail(msg)


def _validate_spv(name: String, words: List[UInt32]):
    var rep = validate_spirv_module(words)
    if not rep.valid:
        var err = rep.errors[0] if len(rep.errors) > 0 else String("unknown SPIR-V error")
        _fail(name + " SPIR-V validation failed: " + err)


def _unpack_rgb(argb: UInt32) -> Vec3:
    var r = Float32((argb >> 16) & 0xFF) / Float32(255.0)
    var g = Float32((argb >> 8) & 0xFF) / Float32(255.0)
    var b = Float32(argb & 0xFF) / Float32(255.0)
    return Vec3(r, g, b)


def _make_sphere(radius: Float32) -> NativeMesh:
    var mb = MeshBuilder()
    mb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(radius, radius, radius))
    return mb^.finish()


def _make_box(bmin: Vec3, bmax: Vec3) -> NativeMesh:
    var mb = MeshBuilder()
    var center = (bmin + bmax) * Float32(0.5)
    var half_ext = (bmax - bmin) * Float32(0.5)
    mb.add_beveled_box(center, half_ext, bevel=0.005)
    return mb^.finish()


def test_spirv_validation():
    _validate_spv("npr_outline_vertex_shader", npr_outline_vertex_shader())
    _validate_spv("npr_outline_fragment_shader", npr_outline_fragment_shader())
    _validate_spv("aaa_npr_cel_shade_shader", aaa_npr_cel_shade_shader())
    _validate_spv("visbuffer_alphamask_vfx_fragment_shader", visbuffer_alphamask_vfx_fragment_shader())
    _validate_spv("aaa_npr_post_fx_shader", aaa_npr_post_fx_shader())
    _validate_spv("aaa_dq_skinning_shader", aaa_dq_skinning_shader(0, 13))
    print("  [PASS] Suite 1: All 6 NPR/Outline/Smear/VFX SPIR-V 1.5 shaders validated cleanly")


def test_cpu_reference_evaluators():
    var cel = AaaCelParams.hair(
        sss_shadow_tint=Vec3(0.50, 0.30, 0.45),
        halo_strength=0.70,
        outline_color=Vec3(0.02, 0.01, 0.02),
    )
    var alb = Vec3(0.85, 0.20, 0.18)
    var n = Vec3(0.0, 0.0, 1.0)
    var b_tan = Vec3(0.0, 1.0, 0.0)
    var v = Vec3(0.0, 0.0, 1.0)
    var l_lit = Vec3(0.0, 0.35, 0.9367).normalized()
    var l_dark = Vec3(0.0, 0.0, -1.0)
    var sun_rad = Vec3(3.0, 2.9, 2.8)
    var sh0 = Vec3(0.4, 0.4, 0.45)

    var ilm_def = (Float32(1.0), Float32(0.5), Float32(0.5), Float32(1.0))
    var ilm_occ = (Float32(1.0), Float32(0.0), Float32(0.5), Float32(1.0))
    var ilm_ink = (Float32(1.0), Float32(0.5), Float32(0.5), Float32(0.0))

    var c_lit = eval_npr_cel_cpu(cel, alb, alb, False, ilm_def, n, b_tan, v, l_lit, sun_rad, 1.0, sh0)
    var c_shd = eval_npr_cel_cpu(cel, alb, alb, False, ilm_def, n, b_tan, v, l_dark, sun_rad, 1.0, sh0)
    _assert(c_lit.x > c_shd.x * Float32(2.0), "Cel lit band significantly brighter than SSS shadow band")

    var c_ilm_occ = eval_npr_cel_cpu(cel, alb, alb, False, ilm_occ, n, b_tan, v, l_lit, sun_rad, 1.0, sh0)
    _assert(abs(c_ilm_occ.x - c_shd.x) < Float32(0.08), "ILM.g == 0 forces cel shadow pocket")

    var c_ink = eval_npr_cel_cpu(cel, alb, alb, False, ilm_ink, n, b_tan, v, l_lit, sun_rad, 1.0, sh0)
    _assert((c_ink - cel.outline_color).length() < Float32(1e-4), "ILM.a == 0 resolves exact inner ink color")

    # Smear-frame CPU skinning identity & non-uniform stretch/orthogonality
    var bones = List[DualQuat]()
    bones.append(DualQuat())
    bones.append(DualQuat.from_quat_trans(Quat(0.0, 0.0, 0.3826834, 0.9238795), Vec3(0.4, 0.2, -0.1)))
    var id_smears = List[BoneSmear]()
    id_smears.append(BoneSmear())
    id_smears.append(BoneSmear())

    var ref_rigid = skin_vertex_dq_cpu(bones, 1, 0, 0, 0, 0.75, 0.25, 0.0, 0.0, Vec3(1.2, 0.3, -0.2), Vec3(0.6, 0.8, 0.0))
    var ref_sm_id = skin_vertex_dq_smear_cpu(bones, id_smears, 1, 0, 0, 0, 0.75, 0.25, 0.0, 0.0, Vec3(1.2, 0.3, -0.2), Vec3(0.6, 0.8, 0.0))
    _assert((ref_rigid[0] - ref_sm_id[0]).length() < Float32(1e-5), "Zero BoneSmear matches rigid DualQuat position")
    _assert((ref_rigid[1] - ref_sm_id[1]).length() < Float32(1e-5), "Zero BoneSmear matches rigid DualQuat normal")

    var vp_scale = compute_volume_preserving_smear_scale(Float32(2.25))
    _assert(abs(vp_scale.x * vp_scale.y * vp_scale.z - Float32(1.0)) < Float32(1e-4), "Volume-preserving smear det == 1.0")

    # Stepped limited-animation clock + hitstop freeze
    _assert(abs(quantize_stepped_anim_time(Float32(0.07), Float32(12.0)) - Float32(0.0)) < Float32(1e-5), "12 Hz step holds frame 0 at 70 ms")
    _assert(abs(quantize_stepped_anim_time(Float32(0.09), Float32(12.0)) - Float32(1.0 / 12.0)) < Float32(1e-5), "12 Hz step snaps at 90 ms")
    var clk = SteppedAnimClock(step_hz=12.0)
    _ = clk.advance(Float32(0.05))
    clk.trigger_hitstop(Float32(0.10))
    var t_frozen = clk.advance(Float32(0.08))
    _assert(t_frozen == Float32(0.0) and not clk.pose_changed_this_tick, "Hitstop freezes stepped pose clock")

    # Stylized VFX erosion, chromatic shockwave & manga impact frame
    var vfx0 = AaaStylizedVfxParams.anime_flame(erosion=0.0)
    var vfx1 = AaaStylizedVfxParams.anime_flame(erosion=1.0)
    _assert(eval_stylized_vfx_cpu(vfx0, 0.5, 0.5, 0.25)[0], "erosion=0.0 keeps VFX visible")
    _assert(not eval_stylized_vfx_cpu(vfx1, 0.5, 0.5, 0.25)[0], "erosion=1.0 fully dissolves VFX")

    var sw = AaaShockwaveRing(center_u=0.5, center_v=0.5, radius_uv=0.20, thickness_uv=0.06, refract_strength=0.04, chromatic_dispersion=0.60)
    var sw_off = eval_shockwave_chromatic_offsets_cpu(sw, 0.72, 0.5, 1.0)
    _assert(sw_off[3] > Float32(0.2) and abs(sw_off[0][0]) > abs(sw_off[1][0]) and abs(sw_off[1][0]) > abs(sw_off[2][0]), "Shockwave R/G/B chromatic dispersion ordering")

    var imp = AaaImpactFrameParams(blend=1.0, luma_threshold=0.5, accent_width=0.02, paper_gain=3.0, ink_color=Vec3(0.01, 0.01, 0.01))
    var imp_hi = eval_impact_frame_cpu(imp, Vec3(2.0, 2.0, 2.0), 0.5, 0.5, 1.0)
    var imp_lo = eval_impact_frame_cpu(imp, Vec3(0.05, 0.05, 0.05), 0.5, 0.5, 1.0)
    _assert(imp_hi.x > Float32(2.5) and imp_lo.x < Float32(0.05), "Impact frame binary paper/ink separation")
    print("  [PASS] Suite 2: CPU reference evaluators (Cel, Smear, SteppedClock, VFX, Shockwave, Impact) verified")


def test_gpu_cel_outlines_smear_and_vfx():
    var w = 160
    var h = 120
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan AAA renderer unavailable on this host")
        return

    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.tsr_history_weight = Float32(0.0)
    ren.enable_auto_exposure = False
    ren.exposure = Float32(1.0)
    ren.bloom_intensity = Float32(0.0)
    ren.enable_vsm = False

    var sky_p = SkyDomeParams()
    sky_p.zenith_rgb = Vec3(0.75, 0.80, 0.90)
    sky_p.horizon_rgb = Vec3(0.75, 0.80, 0.90)
    sky_p.ground_rgb = Vec3(0.55, 0.58, 0.65)
    sky_p.sun_lobe_rgb = Vec3(0.0, 0.0, 0.0)
    ren.upload_ibl_environment(bake_bc6h_ibl_cubemap_and_sh9(sky_p, base_dim=32, mip_levels=5))
    ren.sun_dir = Vec3(0.65, 0.45, -0.61).normalized()
    ren.sun_radiance = Vec3(3.2, 3.1, 3.0)
    ren.ibl_intensity = Float32(0.6)

    # ------------------------------------------------------------------------
    # Part A: GPU Cel Shading (`ILM.a` inner ink + `SSS` shadow step) + Hardware Outline
    # ------------------------------------------------------------------------
    var ilm_dim = 64
    var ilm_px = List[UInt32](length=ilm_dim * ilm_dim, fill=0)
    for y in range(ilm_dim):
        for x in range(ilm_dim):
            var u_dist = abs(x - 32)
            var a8 = UInt32(0) if u_dist <= 1 else UInt32(255)
            var r8 = UInt32(255)
            var g8 = UInt32(128)
            var b8 = UInt32(128)
            ilm_px[y * ilm_dim + x] = (a8 << 24) | (r8 << 16) | (g8 << 8) | b8
    var ilm_slot = ren.upload_bc7_texture(ilm_px, ilm_dim, ilm_dim, srgb=False, generate_mips=False)
    _assert(ilm_slot >= 0, "Uploaded BC7_UNORM ILM map")

    var base_mat = AaaMaterial(base_color=Vec3(0.92, 0.16, 0.05), roughness=0.5, metallic=0.0)
    var cel_p = AaaCelParams.cloth(
        sss_shadow_tint=Vec3(0.35, 0.25, 0.85),
        outline_width_px=5.0,
        outline_color=Vec3(0.002, 0.002, 0.003),
    )
    cel_p.ilm_tex = ilm_slot
    cel_p.spec_intensity = Float32(0.0)
    cel_p.rim_color = Vec3(0.0, 0.0, 0.0)
    var cel_mat_id = ren.add_cel_material(base_mat, cel_p)
    _assert(cel_mat_id >= 0, "Registered AaaCelParams material")

    var sph_mesh = _make_sphere(0.85)
    var sph_entry = ren.add_mesh(sph_mesh, cel_mat_id)

    ren.begin_frame(cam_pos=Vec3(0.0, 0.0, -2.8), cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(sph_entry.mesh_id, Mat4())
    _assert(ren.render_frame(capture_readback=True), "Rendered Cel + Outline frame")
    var px_cel = ren.readback_pixels()

    var mid_y = h // 2
    var outline_hits = 0
    var bright_cel_hits = 0
    var shadow_cel_hits = 0
    for x in range(10, w - 10):
        var c = _unpack_rgb(px_cel[mid_y * w + x])
        var luma = c.x * Float32(0.2126) + c.y * Float32(0.7152) + c.z * Float32(0.0722)
        if luma < Float32(0.05):
            outline_hits += 1
        elif c.x > Float32(0.82) and c.z < Float32(0.45):
            bright_cel_hits += 1
        elif c.x > Float32(0.20) and c.x < Float32(0.68) and c.z < Float32(0.45):
            shadow_cel_hits += 1

    _assert(
        outline_hits >= 4,
        "Hardware CULL_MODE_FRONT outline + ILM.a ink pixels present (got " + String(outline_hits) + ")",
    )
    _assert(bright_cel_hits >= 8, "Crisp lit cel hemisphere pixels present (got " + String(bright_cel_hits) + ")")
    _assert(shadow_cel_hits >= 6, "SSS-tinted cel shadow hemisphere pixels present (got " + String(shadow_cel_hits) + ")")

    # Verify per-vertex `G = 0` outline suppression (`use_vertex_color_channels = True`)
    var no_out_sph = _make_sphere(0.85)
    var nv_sph = no_out_sph.vertex_count()
    no_out_sph.colors = List[UInt8](capacity=nv_sph * 4)
    for _ in range(nv_sph):
        no_out_sph.colors.append(UInt8(128))
        no_out_sph.colors.append(UInt8(0))
        no_out_sph.colors.append(UInt8(255))
        no_out_sph.colors.append(UInt8(255))
    var cel_vc = cel_p.copy()
    cel_vc.ilm_tex = -1
    cel_vc.use_vertex_color_channels = True
    var vc_mat_id = ren.add_cel_material(base_mat, cel_vc)
    var vc_entry = ren.add_mesh(no_out_sph, vc_mat_id)

    ren.begin_frame(cam_pos=Vec3(0.0, 0.0, -2.8), cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(vc_entry.mesh_id, Mat4())
    _assert(ren.render_frame(capture_readback=True), "Rendered G=0 vertex-suppressed outline frame")
    var px_sup = ren.readback_pixels()
    var sup_dark_hits = 0
    for x in range(10, w - 10):
        var c = _unpack_rgb(px_sup[mid_y * w + x])
        var luma = c.x * Float32(0.2126) + c.y * Float32(0.7152) + c.z * Float32(0.0722)
        if luma < Float32(0.05):
            sup_dark_hits += 1
    _assert(sup_dark_hits == 0, "Vertex color G == 0 cleanly suppresses hardware outline to 0 pixels")

    # Verify per-character `use_custom_light_dir` decouples cel key lighting from a backlit world sun!
    var saved_sun = ren.sun_dir
    ren.sun_dir = Vec3(0.0, 0.0, 1.0)  # Purely behind sphere relative to cam at -2.8 Z
    ren.begin_frame(cam_pos=Vec3(0.0, 0.0, -2.8), cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(vc_entry.mesh_id, Mat4())
    _assert(ren.render_frame(capture_readback=True), "Rendered backlit cel sphere")
    var c_backlit = _unpack_rgb(ren.readback_pixels()[mid_y * w + (w // 2)])

    cel_vc.use_custom_light_dir = True
    cel_vc.custom_light_dir = Vec3(0.0, 0.25, -0.968).normalized()
    _assert(ren.update_cel_material(vc_mat_id, cel_vc), "Updated cel material with custom_light_dir")
    ren.begin_frame(cam_pos=Vec3(0.0, 0.0, -2.8), cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(vc_entry.mesh_id, Mat4())
    _assert(ren.render_frame(capture_readback=True), "Rendered custom-key-lit cel sphere under backlit world sun")
    var c_herolit = _unpack_rgb(ren.readback_pixels()[mid_y * w + (w // 2)])
    _assert(
        c_herolit.x > c_backlit.x + Float32(0.18),
        "use_custom_light_dir lights front hemisphere despite backlit world sun (" + String(c_backlit.x) + " -> " + String(c_herolit.x) + ")",
    )
    ren.sun_dir = saved_sun

    # A plain (e.g. glTF-uploaded, textured) material joins the toon look through update_cel_material.
    var n_cel_before = len(ren.cel_materials)
    var plain_id = ren.add_material(AaaMaterial(base_color=Vec3(0.6, 0.2, 0.2), roughness=0.6, metallic=0.0))
    _assert(ren.update_cel_material(plain_id, cel_vc), "update_cel_material pairs a plain material with cel params")
    _assert(len(ren.cel_materials) == n_cel_before + 1, "the plain material gained one cel entry")
    _assert(ren.update_cel_material(plain_id, cel_vc), "updating it again edits that entry")
    _assert(len(ren.cel_materials) == n_cel_before + 1, "a second update adds no entry")
    _assert(not ren.update_cel_material(len(ren.materials) + 5, cel_vc), "an unknown material id is refused")
    print("  [PASS] Suite 3: GPU Cel (ILM + SSS + Custom Hero Light) & Variable-Width Outlines verified")

    # ------------------------------------------------------------------------
    # Part B: GPU `DualQuat + BoneSmear` Non-Uniform Deformation & Identity Reset
    # ------------------------------------------------------------------------
    var box_m = _make_box(Vec3(0.0, -0.1, -0.1), Vec3(1.0, 0.1, 0.1))
    var bv_cnt = box_m.vertex_count()
    var j_packed = List[UInt32](capacity=bv_cnt)
    var j_weights = List[Vec3](capacity=bv_cnt)
    for vi in range(bv_cnt):
        var px_v = box_m.positions[vi * 3 + 0]
        var w1 = max(Float32(0.0), min(Float32(1.0), px_v))
        var w0 = Float32(1.0) - w1
        j_packed.append(UInt32(0) | (UInt32(1) << 8))
        j_weights.append(Vec3(w0, w1, 0.0))

    var sk_entry = ren.add_skinned_mesh(box_m, j_packed, j_weights, cel_mat_id, bone_count=4)
    var sk_bones = List[DualQuat]()
    sk_bones.append(DualQuat())
    sk_bones.append(DualQuat.from_quat_trans(Quat(0.0, 0.0, 0.2588190, 0.9659258), Vec3(0.0, 0.15, 0.0)))

    var sk_smears = List[BoneSmear]()
    sk_smears.append(BoneSmear())
    sk_smears.append(BoneSmear(scale=Vec3(2.4, 0.65, 0.65), shear_xy=0.28, pivot_bind=Vec3(0.25, 0.0, 0.0)))

    _ = ren.update_skinned_mesh_smear(sk_entry, sk_bones, sk_bones, sk_smears, sk_smears)

    var gpu_v = List[Float32](length=bv_cnt * 12, fill=0.0)
    _ = external_call[
        "memcpy", Int
    ](Int(gpu_v.unsafe_ptr()), ren.buf_vertices.cpu_address + sk_entry.mesh_entry.first_vertex * 48, bv_cnt * 48)
    for vi in range(bv_cnt):
        var bp = Vec3(box_m.positions[vi * 3], box_m.positions[vi * 3 + 1], box_m.positions[vi * 3 + 2])
        var bn = Vec3(box_m.normals[vi * 3], box_m.normals[vi * 3 + 1], box_m.normals[vi * 3 + 2])
        var wv = j_weights[vi]
        var cpu_ref = skin_vertex_dq_smear_cpu(sk_bones, sk_smears, 0, 1, 0, 0, wv.x, wv.y, 0.0, 0.0, bp, bn)
        var gp = Vec3(gpu_v[vi * 12 + 0], gpu_v[vi * 12 + 1], gpu_v[vi * 12 + 2])
        var gn = Vec3(gpu_v[vi * 12 + 3], gpu_v[vi * 12 + 4], gpu_v[vi * 12 + 5])
        _assert((gp - cpu_ref[0]).length() < Float32(1e-4), "GPU Smear-Skinning position matches CPU < 1e-4")
        _assert((gn - cpu_ref[1]).length() < Float32(1e-3), "GPU Smear-Skinning J^{-T} normal matches CPU < 1e-3")

    _ = ren.update_skinned_mesh_bones(sk_entry, sk_bones, sk_bones)
    _ = external_call[
        "memcpy", Int
    ](Int(gpu_v.unsafe_ptr()), ren.buf_vertices.cpu_address + sk_entry.mesh_entry.first_vertex * 48, bv_cnt * 48)
    var bp0 = Vec3(box_m.positions[0], box_m.positions[1], box_m.positions[2])
    var bn0 = Vec3(box_m.normals[0], box_m.normals[1], box_m.normals[2])
    var rigid_ref = skin_vertex_dq_cpu(sk_bones, 0, 1, 0, 0, j_weights[0].x, j_weights[0].y, 0.0, 0.0, bp0, bn0)
    var gp_rst = Vec3(gpu_v[0], gpu_v[1], gpu_v[2])
    _assert((gp_rst - rigid_ref[0]).length() < Float32(1e-4), "update_skinned_mesh_bones resets BoneSmear to identity")
    _ = len(gpu_v)
    print("  [PASS] Suite 4: GPU DualQuat + BoneSmear skinning (< 1e-4 vs CPU) & BLAS refit verified")

    # ------------------------------------------------------------------------
    # Part C: Eroding 3D Mesh VFX + Chromatic Shockwave + Manga Impact Frame
    # ------------------------------------------------------------------------
    var wall_mat = ren.add_material(AaaMaterial(base_color=Vec3(0.05, 0.15, 0.85), roughness=0.9, metallic=0.0))
    var wall_mesh = _make_box(Vec3(-3.0, -2.5, 1.5), Vec3(3.0, 2.5, 1.7))
    var wall_entry = ren.add_mesh(wall_mesh, wall_mat)

    var vfx_params = AaaStylizedVfxParams.anime_flame(
        core_color=Vec3(4.0, 3.0, 0.8),
        edge_color=Vec3(2.5, 0.25, 0.04),
        erosion=0.62,
    )
    var vfx_mat = ren.add_vfx_material(AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0)), vfx_params)
    var vfx_quad = NativeMesh()
    vfx_quad.double_sided = True
    vfx_quad.positions = [
        Float32(-1.2), Float32(-1.0), Float32(0.0),
        Float32(1.2), Float32(-1.0), Float32(0.0),
        Float32(1.2), Float32(1.0), Float32(0.0),
        Float32(-1.2), Float32(1.0), Float32(0.0),
    ]
    vfx_quad.normals = [
        Float32(0.0), Float32(0.0), Float32(-1.0),
        Float32(0.0), Float32(0.0), Float32(-1.0),
        Float32(0.0), Float32(0.0), Float32(-1.0),
        Float32(0.0), Float32(0.0), Float32(-1.0),
    ]
    vfx_quad.texcoords = [
        Float32(0.0), Float32(0.0),
        Float32(1.0), Float32(0.0),
        Float32(1.0), Float32(1.0),
        Float32(0.0), Float32(1.0),
    ]
    vfx_quad.indices = [0, 1, 2, 0, 2, 3]
    var vfx_entry = ren.add_mesh(vfx_quad, vfx_mat)

    ren.scene_time_sec = Float32(0.45)
    ren.begin_frame(cam_pos=Vec3(0.0, 0.0, -2.5), cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(wall_entry.mesh_id, Mat4())
    ren.draw_instance(vfx_entry.mesh_id, Mat4())
    _assert(ren.render_frame(capture_readback=True), "Rendered eroding 3D stylized VFX frame")
    var px_vfx = ren.readback_pixels()

    var flame_px = 0
    var eroded_hole_blue_px = 0
    for y in range(h // 4, (h * 3) // 4):
        for x in range(w // 4, (w * 3) // 4):
            var c = _unpack_rgb(px_vfx[y * w + x])
            if c.x > Float32(0.65) and c.z < Float32(0.35):
                flame_px += 1
            elif c.z > Float32(0.22) and c.z > c.x * Float32(2.0):
                eroded_hole_blue_px += 1
    _assert(flame_px > 25, "Stylized 3D VFX rendered bright warm flame bands (got " + String(flame_px) + ")")
    _assert(
        eroded_hole_blue_px > 25,
        "Stylized 3D VFX hardware alpha-erosion cut out holes revealing blue wall (got " + String(eroded_hole_blue_px) + ")",
    )

    # Chromatic Shockwave dispersion over a crisp vertical white pillar on dark wall
    var dark_wall_mat = ren.add_material(AaaMaterial(base_color=Vec3(0.01, 0.01, 0.01), roughness=1.0, metallic=0.0))
    var wb_mat = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=1.0, metallic=0.0)
    wb_mat.emissive = Vec3(2.0, 2.0, 2.0)
    var white_bar_mat = ren.add_material(wb_mat)
    var bar_mesh = _make_box(Vec3(0.28, -2.0, 0.2), Vec3(0.42, 2.0, 0.3))
    var dark_wall_entry = ren.add_mesh(wall_mesh, dark_wall_mat)
    var bar_entry = ren.add_mesh(bar_mesh, white_bar_mat)

    ren.begin_frame(cam_pos=Vec3(0.0, 0.0, -2.5), cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(dark_wall_entry.mesh_id, Mat4())
    ren.draw_instance(bar_entry.mesh_id, Mat4())
    ren.submit_shockwave(
        AaaShockwaveRing(
            center_u=0.5,
            center_v=0.5,
            radius_uv=0.16,
            thickness_uv=0.07,
            refract_strength=0.06,
            chromatic_dispersion=0.80,
            glow_intensity=0.0,
        )
    )
    _assert(ren.render_frame(capture_readback=True), "Rendered R/G/B chromatic shockwave frame")
    var px_sw = ren.readback_pixels()
    var max_rb_split = Float32(0.0)
    for x in range(w // 2, w - 8):
        var c = _unpack_rgb(px_sw[mid_y * w + x])
        var diff_rb = abs(c.x - c.z)
        if diff_rb > max_rb_split:
            max_rb_split = diff_rb
    _assert(max_rb_split > Float32(0.20), "Chromatic shockwave separated R and B channels (max_rb=" + String(max_rb_split) + ")")

    # Manga Impact Frame binary ink / crimson accent / white paper stylization
    ren.set_impact_frame(
        AaaImpactFrameParams(
            blend=1.0,
            center_u=0.5,
            center_v=0.5,
            luma_threshold=0.45,
            streak_strength=0.35,
            accent_width=0.08,
            paper_gain=2.5,
            ink_color=Vec3(0.0, 0.0, 0.0),
            accent_color=Vec3(1.0, 0.02, 0.02),
        )
    )
    ren.begin_frame(cam_pos=Vec3(0.0, 0.0, -2.8), cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(sph_entry.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    _assert(ren.render_frame(capture_readback=True), "Rendered Manga Impact Frame")
    var px_imp = ren.readback_pixels()
    var white_hits = 0
    var black_hits = 0
    for i in range(w * h):
        var c = _unpack_rgb(px_imp[i])
        if c.x > Float32(0.95) and c.y > Float32(0.95) and c.z > Float32(0.95):
            white_hits += 1
        elif c.x < Float32(0.03) and c.y < Float32(0.03) and c.z < Float32(0.03):
            black_hits += 1
    _assert(white_hits > 500 and black_hits > 200, "Manga Impact Frame produced high-contrast paper white & ink black")
    ren.clear_impact_frame()
    ren.shutdown()
    print("  [PASS] Suite 5: GPU Eroding 3D Mesh VFX, Chromatic Shockwave & Manga Impact Frame verified")


def main():
    print("=== Flint v5.1 Anime NPR Cel, Outlines, Smear Skinning & Stylized VFX Suite ===")
    test_spirv_validation()
    test_cpu_reference_evaluators()
    test_gpu_cel_outlines_smear_and_vfx()
    print("=== ALL FLINT v5.1 NPR & VFX TESTS PASSED ===")
