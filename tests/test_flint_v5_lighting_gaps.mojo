"""Raster lighting fallbacks without ray tracing (`test_flint_v5_lighting_gaps.mojo`).

Zero external assets. Verifies `aaa_raster_fallbacks.mojo` with every RT feature off:
  1. SPIR-V validation of the GTAO, SSR, local-shadow and blended-mesh shaders + CPU helpers.
  2. GTAO darkens the crease between a floor and a wall, and leaves open floor alone.
  3. SSR shows a bright emissive box in a glossy floor.
  4. A point light casts a raster shadow-map shadow from a box onto the floor.
  5. A transparent quad blends with the opaque wall behind it, and one behind the wall stays hidden.
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, max, min
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh
from flint.core.math import Vec3
from flint.core.transform import Mat4
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_clustered_lights import (
    AaaClusterLight,
    local_shadow_point_face_basis,
    local_shadow_spot_basis,
    select_local_shadow_lights,
)
from flint.render.aaa_ibl import SkyDomeParams, bake_bc6h_ibl_cubemap_and_sh9
from flint.render.aaa_post_tsr import AAA_DISPLAY_LINEAR_SRGB
from flint.render.aaa_raster_fallbacks import (
    AAA_FALLBACK_AUTO,
    AAA_FALLBACK_OFF,
    AAA_FALLBACK_ON,
    blend_composite_shader,
    blend_mesh_fragment_shader,
    blend_mesh_vertex_shader,
    fallback_active,
    gtao_cosine_visibility_cpu,
    gtao_denoise_apply_shader,
    gtao_trace_shader,
    local_shadow_depth_vertex_shader,
    ssr_composite_shader,
    ssr_pyramid_levels,
    ssr_pyramid_shader,
    ssr_trace_shader,
)
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.material_graph import AAA_MAX_DECALS
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


def _luma(c: Vec3) -> Float32:
    return c.x * Float32(0.2126) + c.y * Float32(0.7152) + c.z * Float32(0.0722)


def _lin_luma(c: Vec3) -> Float32:
    """Linear luminance of an sRGB-encoded readback pixel (ratios of light, not of encoded values)."""
    return _luma(Vec3(c.x ** Float32(2.2), c.y ** Float32(2.2), c.z ** Float32(2.2)))


def _make_box(bmin: Vec3, bmax: Vec3) -> NativeMesh:
    var mb = MeshBuilder()
    var center = (bmin + bmax) * Float32(0.5)
    var half_ext = (bmax - bmin) * Float32(0.5)
    mb.add_beveled_box(center, half_ext, bevel=0.002)
    return mb^.finish()


def _setup(mut ren: FlintAaaVulkanRenderer, sun: Vec3, ibl: Float32):
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.tsr_history_weight = Float32(0.0)
    ren.enable_auto_exposure = False
    ren.exposure = Float32(1.0)
    ren.bloom_intensity = Float32(0.0)
    ren.enable_vsm = False
    ren.enable_rt_shadows = False
    ren.enable_rt_reflections = False
    ren.enable_rt_gi = False
    var sky_p = SkyDomeParams()
    sky_p.zenith_rgb = Vec3(0.55, 0.65, 0.85)
    sky_p.horizon_rgb = Vec3(0.60, 0.68, 0.85)
    sky_p.ground_rgb = Vec3(0.45, 0.47, 0.50)
    sky_p.sun_lobe_rgb = Vec3(0.0, 0.0, 0.0)
    ren.upload_ibl_environment(bake_bc6h_ibl_cubemap_and_sh9(sky_p, base_dim=32, mip_levels=5))
    ren.sun_dir = Vec3(0.3, 0.9, -0.3).normalized()
    ren.sun_radiance = sun
    ren.ibl_intensity = ibl
    ren.raster_fallbacks.gtao_mode = AAA_FALLBACK_OFF
    ren.raster_fallbacks.ssr_mode = AAA_FALLBACK_OFF
    ren.raster_fallbacks.local_shadow_mode = AAA_FALLBACK_OFF


def test_spirv_and_cpu():
    _validate_spv("gtao_trace_shader", gtao_trace_shader())
    _validate_spv("gtao_denoise_apply_shader", gtao_denoise_apply_shader())
    _validate_spv("ssr_pyramid_shader", ssr_pyramid_shader())
    _validate_spv("ssr_trace_shader", ssr_trace_shader())
    _validate_spv("ssr_composite_shader", ssr_composite_shader())
    _validate_spv("local_shadow_depth_vertex_shader", local_shadow_depth_vertex_shader())
    _validate_spv("blend_mesh_vertex_shader", blend_mesh_vertex_shader())
    _validate_spv("blend_mesh_fragment_shader", blend_mesh_fragment_shader())
    _validate_spv("blend_composite_shader", blend_composite_shader())
    print("  [PASS] 9 raster-fallback SPIR-V shaders validated")

    # GTAO slice integral: open hemisphere = 1, a horizon at 45 deg on one side occludes
    var open_v = gtao_cosine_visibility_cpu(Float32(-1.5707963), Float32(1.5707963), Float32(0.0))
    var half_v = gtao_cosine_visibility_cpu(Float32(-1.5707963), Float32(0.7853982), Float32(0.0))
    _assert(abs(open_v - Float32(1.0)) < Float32(1e-4), "GTAO open slice visibility == 1 (got " + String(open_v) + ")")
    _assert(half_v < Float32(0.95) and half_v > Float32(0.6), "GTAO 45 deg horizon occludes (got " + String(half_v) + ")")

    # Modes: AUTO only without ray query
    _assert(fallback_active(AAA_FALLBACK_AUTO, False) and not fallback_active(AAA_FALLBACK_AUTO, True), "AUTO follows ray query")
    _assert(fallback_active(AAA_FALLBACK_ON, True) and not fallback_active(AAA_FALLBACK_OFF, False), "ON / OFF modes")

    # Pyramid levels end at 1x1 and halve
    var lv = ssr_pyramid_levels(160, 120, 0)
    _assert(lv[0][1] == 160 and lv[0][2] == 120 and lv[1][1] == 80, "SSR pyramid level sizes")
    _assert(lv[len(lv) - 1][1] == 1 and lv[len(lv) - 1][2] == 1, "SSR pyramid ends at 1x1")

    # Cube face bases are orthonormal; selection respects the tile budget
    for f in range(6):
        var b = local_shadow_point_face_basis(f)
        _assert(abs(b[0].dot(b[1])) < Float32(1e-6), "cube face basis orthogonal")
    var sb = local_shadow_spot_basis(Vec3(0.0, -1.0, 0.0))
    _assert(abs(sb[0].dot(sb[1])) < Float32(1e-5) and abs(sb[1].length() - Float32(1.0)) < Float32(1e-4), "spot basis orthonormal")
    var lights = List[AaaClusterLight]()
    for k in range(6):
        lights.append(AaaClusterLight.point(Vec3(Float32(k) * Float32(3.0), 2.0, 0.0), Vec3(4.0, 4.0, 4.0), 6.0))
    lights.append(AaaClusterLight.spot(Vec3(0.0, 4.0, 0.0), Vec3(0.0, -1.0, 0.0), Vec3(9.0, 9.0, 9.0), 10.0))
    var picks = select_local_shadow_lights(lights, Vec3(0.0, 2.0, -4.0), 8, 24)
    var tiles = 0
    for p in picks:
        tiles += 6 if lights[p].kind == UInt32(0) else 1
    _assert(len(picks) >= 2 and tiles <= 24, "local shadow light selection fits the 24-tile atlas")
    _assert(AAA_MAX_DECALS == 256, "decal cap raised to 256")
    print("  [PASS] CPU helpers: GTAO integral, modes, Hi-Z pyramid, cube/spot bases, light selection, decal cap")


def test_gpu_gtao_crease():
    var w = 160
    var h = 120
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan AAA renderer unavailable on this host")
        return
    _setup(ren, Vec3(0.0, 0.0, 0.0), Float32(1.0))
    var mat = ren.add_material(AaaMaterial(base_color=Vec3(0.7, 0.7, 0.7), roughness=0.9, metallic=0.0))
    var floor_e = ren.add_mesh(_make_box(Vec3(-3.0, -0.2, -4.0), Vec3(3.0, 0.0, 3.0)), mat)
    var wall_e = ren.add_mesh(_make_box(Vec3(-3.0, 0.0, 1.0), Vec3(3.0, 3.0, 1.3)), mat)

    var frames = List[List[UInt32]]()
    for pass_i in range(2):
        ren.raster_fallbacks.gtao_mode = AAA_FALLBACK_OFF if pass_i == 0 else AAA_FALLBACK_ON
        ren.begin_frame(cam_pos=Vec3(0.0, 1.3, -2.0), cam_target=Vec3(0.0, 0.4, 1.0), fov_y_deg=60.0)
        ren.draw_instance(floor_e.mesh_id, Mat4())
        ren.draw_instance(wall_e.mesh_id, Mat4())
        _assert(ren.render_frame(capture_readback=True), "rendered GTAO frame")
        frames.append(ren.readback_pixels())
    _assert(ren.raster_fallbacks.last_gtao_ran, "GTAO pass ran")

    # Darkest ratio along the center column is the crease; the bottom rows (open floor near the camera) stay put
    var min_ratio = Float32(10.0)
    var min_row = -1
    var dark_px = 0
    for y in range(8, h - 8):
        for x in range(w // 2 - 20, w // 2 + 20):
            var off = _lin_luma(_unpack_rgb(frames[0][y * w + x]))
            var on = _lin_luma(_unpack_rgb(frames[1][y * w + x]))
            if off > Float32(0.01):
                var r = on / off
                if r < Float32(0.92):
                    dark_px += 1
                if x == w // 2 and r < min_ratio:
                    min_ratio = r
                    min_row = y
    var open_off = Float32(0.0)
    var open_on = Float32(0.0)
    for y in range(h - 6, h):
        for x in range(w // 2 - 20, w // 2 + 20):
            open_off += _lin_luma(_unpack_rgb(frames[0][y * w + x]))
            open_on += _lin_luma(_unpack_rgb(frames[1][y * w + x]))
    var open_ratio = open_on / max(open_off, Float32(1e-4))
    _assert(min_ratio < Float32(0.9), "GTAO darkens the crease (min ratio " + String(min_ratio) + " at row " + String(min_row) + ")")
    _assert(dark_px >= 30, "GTAO crease band has width (got " + String(dark_px) + " px)")
    _assert(open_ratio > Float32(0.95), "GTAO leaves open floor lit (ratio " + String(open_ratio) + ")")
    print(
        "  [PASS] GTAO darkens the floor/wall crease (min ratio " + String(min_ratio) + ", " + String(dark_px)
        + " px < 0.92, linear) and keeps open floor (ratio " + String(open_ratio) + ")"
    )
    ren.shutdown()


def test_gpu_ssr_reflection():
    var w = 160
    var h = 120
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan AAA renderer unavailable on this host")
        return
    _setup(ren, Vec3(0.0, 0.0, 0.0), Float32(0.5))
    var floor_mat = ren.add_material(AaaMaterial(base_color=Vec3(0.9, 0.9, 0.9), roughness=0.05, metallic=1.0))
    var emis = AaaMaterial(base_color=Vec3(0.0, 0.0, 0.0), roughness=0.9, metallic=0.0)
    emis.emissive = Vec3(4.0, 0.3, 0.1)
    var box_mat = ren.add_material(emis)
    var floor_e = ren.add_mesh(_make_box(Vec3(-4.0, -0.2, -4.0), Vec3(4.0, 0.0, 6.0)), floor_mat)
    var box_e = ren.add_mesh(_make_box(Vec3(-0.6, 0.0, 1.5), Vec3(0.6, 1.2, 2.3)), box_mat)

    var frames = List[List[UInt32]]()
    for pass_i in range(2):
        ren.raster_fallbacks.ssr_mode = AAA_FALLBACK_OFF if pass_i == 0 else AAA_FALLBACK_ON
        ren.begin_frame(cam_pos=Vec3(0.0, 1.0, -2.5), cam_target=Vec3(0.0, 0.2, 1.5), fov_y_deg=60.0)
        ren.draw_instance(floor_e.mesh_id, Mat4())
        ren.draw_instance(box_e.mesh_id, Mat4())
        _assert(ren.render_frame(capture_readback=True), "rendered SSR frame")
        frames.append(ren.readback_pixels())
    _assert(ren.raster_fallbacks.last_ssr_ran, "SSR pass ran")

    # Lowest row of the (unchanged) emissive box, then reflected red pixels below it
    var box_bottom = -1
    for y in range(h):
        var c = _unpack_rgb(frames[0][y * w + w // 2])
        if c.x > Float32(0.9) and c.z < Float32(0.5):
            box_bottom = y
    _assert(box_bottom > 0 and box_bottom < h - 4, "emissive box visible (bottom row " + String(box_bottom) + ")")
    var refl_px = 0
    var max_gain = Float32(0.0)
    for y in range(box_bottom + 2, h):
        for x in range(w // 2 - 15, w // 2 + 15):
            var off = _unpack_rgb(frames[0][y * w + x])
            var on = _unpack_rgb(frames[1][y * w + x])
            var gain = (on.x - on.z) - (off.x - off.z)
            max_gain = max(max_gain, gain)
            if gain > Float32(0.2):
                refl_px += 1
    var box_c_off = _unpack_rgb(frames[0][(box_bottom - 3) * w + w // 2])
    var box_c_on = _unpack_rgb(frames[1][(box_bottom - 3) * w + w // 2])
    _assert(refl_px >= 30, "SSR reflects the emissive box in the floor (" + String(refl_px) + " px, max gain " + String(max_gain) + ")")
    _assert((box_c_on - box_c_off).length() < Float32(0.08), "SSR leaves the box itself unchanged")
    print("  [PASS] SSR shows the emissive box in the glossy floor (" + String(refl_px) + " reflected px, max red gain " + String(max_gain) + ")")
    ren.shutdown()


def test_gpu_local_light_shadow():
    var w = 160
    var h = 120
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan AAA renderer unavailable on this host")
        return
    _setup(ren, Vec3(0.0, 0.0, 0.0), Float32(0.05))
    var mat = ren.add_material(AaaMaterial(base_color=Vec3(0.8, 0.8, 0.8), roughness=0.8, metallic=0.0))
    var floor_e = ren.add_mesh(_make_box(Vec3(-4.0, -0.2, -4.0), Vec3(4.0, 0.0, 4.0)), mat)
    var occ_e = ren.add_mesh(_make_box(Vec3(-0.35, 0.9, -0.35), Vec3(0.35, 1.0, 0.35)), mat)

    var frames = List[List[UInt32]]()
    for pass_i in range(2):
        ren.raster_fallbacks.local_shadow_mode = AAA_FALLBACK_OFF if pass_i == 0 else AAA_FALLBACK_ON
        ren.begin_frame(cam_pos=Vec3(0.0, 3.5, -3.5), cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=55.0)
        ren.add_cluster_light(AaaClusterLight.point(Vec3(0.0, 2.0, 0.0), Vec3(6.0, 6.0, 6.0), 7.0))
        ren.draw_instance(floor_e.mesh_id, Mat4())
        ren.draw_instance(occ_e.mesh_id, Mat4())
        _assert(ren.render_frame(capture_readback=True), "rendered local-light frame")
        frames.append(ren.readback_pixels())
    _assert(ren.raster_fallbacks.last_shadow_tiles == 6, "point light rendered 6 cube-face tiles (got " + String(ren.raster_fallbacks.last_shadow_tiles) + ")")

    var shadow_px = 0
    var brighter_px = 0
    for y in range(h):
        for x in range(w):
            var off = _lin_luma(_unpack_rgb(frames[0][y * w + x]))
            var on = _lin_luma(_unpack_rgb(frames[1][y * w + x]))
            if off > Float32(0.03) and on < off * Float32(0.5):
                shadow_px += 1
            if on > off * Float32(1.1) + Float32(0.005):
                brighter_px += 1
    _assert(shadow_px >= 40, "point light casts a shadow under the box (" + String(shadow_px) + " px)")
    _assert(brighter_px <= 4, "shadow map never brightens (" + String(brighter_px) + " px)")
    print("  [PASS] Point light casts a raster cube shadow under the box (" + String(shadow_px) + " px darkened > 2x)")

    # Same scene lit by a downward spot light: one frustum tile
    var spot_frames = List[List[UInt32]]()
    for pass_i in range(2):
        ren.raster_fallbacks.local_shadow_mode = AAA_FALLBACK_OFF if pass_i == 0 else AAA_FALLBACK_ON
        ren.begin_frame(cam_pos=Vec3(0.0, 3.5, -3.5), cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=55.0)
        ren.add_cluster_light(AaaClusterLight.spot(Vec3(0.3, 2.2, 0.0), Vec3(-0.1, -1.0, 0.0), Vec3(10.0, 10.0, 10.0), 7.0, 30.0, 50.0))
        ren.draw_instance(floor_e.mesh_id, Mat4())
        ren.draw_instance(occ_e.mesh_id, Mat4())
        _assert(ren.render_frame(capture_readback=True), "rendered spot-light frame")
        spot_frames.append(ren.readback_pixels())
    _assert(ren.raster_fallbacks.last_shadow_tiles == 1, "spot light rendered 1 tile (got " + String(ren.raster_fallbacks.last_shadow_tiles) + ")")
    var spot_shadow_px = 0
    for y in range(h):
        for x in range(w):
            var off = _lin_luma(_unpack_rgb(spot_frames[0][y * w + x]))
            var on = _lin_luma(_unpack_rgb(spot_frames[1][y * w + x]))
            if off > Float32(0.03) and on < off * Float32(0.5):
                spot_shadow_px += 1
    _assert(spot_shadow_px >= 40, "spot light casts a shadow under the box (" + String(spot_shadow_px) + " px)")
    print("  [PASS] Spot light casts a raster frustum shadow under the box (" + String(spot_shadow_px) + " px darkened > 2x)")
    ren.shutdown()


def test_gpu_transparent_blend():
    var w = 160
    var h = 120
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=1.0)
    if not ren.active:
        print("  [SKIP] Vulkan AAA renderer unavailable on this host")
        return
    _setup(ren, Vec3(2.5, 2.5, 2.5), Float32(0.6))
    ren.sun_dir = Vec3(0.0, 0.3, -1.0).normalized()
    var wall_mat = ren.add_material(AaaMaterial(base_color=Vec3(0.85, 0.08, 0.05), roughness=0.9, metallic=0.0))
    var glass = AaaMaterial(base_color=Vec3(0.05, 0.2, 0.95), roughness=0.9, metallic=0.0)
    var glass_mat = ren.add_transparent_material(glass, Float32(0.5))
    var wall_e = ren.add_mesh(_make_box(Vec3(-3.0, -2.0, 1.0), Vec3(3.0, 2.0, 1.2)), wall_mat)
    var front_e = ren.add_mesh(_make_box(Vec3(-1.4, -0.6, 0.0), Vec3(-0.2, 0.6, 0.02)), glass_mat)
    var back_e = ren.add_mesh(_make_box(Vec3(0.2, -0.6, 2.0), Vec3(1.4, 0.6, 2.02)), glass_mat)

    var frames = List[List[UInt32]]()
    for pass_i in range(2):
        ren.begin_frame(cam_pos=Vec3(0.0, 0.0, -3.0), cam_target=Vec3(0.0, 0.0, 1.0), fov_y_deg=50.0)
        ren.draw_instance(wall_e.mesh_id, Mat4())
        if pass_i == 1:
            ren.draw_instance(front_e.mesh_id, Mat4())
            ren.draw_instance(back_e.mesh_id, Mat4())
        _assert(ren.render_frame(capture_readback=True), "rendered blend frame")
        frames.append(ren.readback_pixels())
    _assert(ren.raster_fallbacks.last_blend_draws == 2, "2 blended draws recorded (got " + String(ren.raster_fallbacks.last_blend_draws) + ")")

    # Front quad covers the left of center, the hidden quad the right of center
    var lx = w // 2 - 20
    var rx = w // 2 + 20
    var cy = h // 2
    var base_l = _unpack_rgb(frames[0][cy * w + lx])
    var mix_l = _unpack_rgb(frames[1][cy * w + lx])
    var base_r = _unpack_rgb(frames[0][cy * w + rx])
    var mix_r = _unpack_rgb(frames[1][cy * w + rx])
    _assert(base_l.x > Float32(0.3) and base_l.x > base_l.z + Float32(0.3), "red wall visible without the quad")
    _assert(mix_l.x < base_l.x * Float32(0.8) and mix_l.x > base_l.x * Float32(0.2), "blended pixel keeps part of the wall (" + String(base_l.x) + " -> " + String(mix_l.x) + ")")
    _assert(mix_l.z > base_l.z + Float32(0.1), "blended pixel picks up the blue quad (" + String(base_l.z) + " -> " + String(mix_l.z) + ")")
    _assert((mix_r - base_r).length() < Float32(0.03), "quad behind the wall is depth-rejected")
    print(
        "  [PASS] Transparent quad blends over the wall (r " + String(base_l.x) + " -> " + String(mix_l.x) + ", b " + String(base_l.z)
        + " -> " + String(mix_l.z) + ") and the occluded quad stays hidden"
    )
    ren.shutdown()


def main():
    print("Flint v5 raster lighting fallbacks (RT off)")
    test_spirv_and_cpu()
    test_gpu_gtao_crease()
    test_gpu_ssr_reflection()
    test_gpu_local_light_shadow()
    test_gpu_transparent_blend()
    print("  [PASS] all raster lighting fallback checks")
