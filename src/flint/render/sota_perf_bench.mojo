"""Area F render SOTA: large-scene benchmark harness (`flint.render.sota_perf_bench`).

Loads heavy glTF/GLB scenes through Flint's own loader (`load_native_model_file` -> `upload_native_model`)
and measures, per scene x resolution x variant:
- per-pass GPU time (`get_gpu_pass_timings`, timestamp queries), as medians over N frames,
- whole-frame GPU time (median and p95), CPU time split into the fence wait (`begin_frame`) and the
  submission work (draw list + `render_frame`), VRAM use (`VK_EXT_memory_budget` heap usage),
- image quality: mean FLIP of the real-time static frame against Flint's own path tracer in photo mode
  (Area B `SOTA_GI_MODE_PATH`, progressive accumulation, 8 bounces).

Variants: static camera, moving camera (fly-through along the scene's long axis), moving light (the sun
sweeps 90 degrees of azimuth) and skinned characters (glTF-skinned walkers with GPU dual-quaternion
skinning + BLAS refit) - the dynamic variants are what the robustness metrics in
`test_flint_v5_sota_perf.mojo` are built on.

Scene assets never live in the repo: by default they are read from `~/flint-models/_scenes/`
(see LICENSES.md there). Everything here is plain CPU orchestration over the public renderer API, so it
adds no passes and no hooks to the frame.
"""
from std.collections import List
from std.math import abs, cos, max, min, sin, sqrt
from flint.assets.glb_native import load_native_model_file
from flint.assets.meshlet import AAA_VERTEX_BYTES
from flint.assets.model import NativeModel
from flint.core.math import Mat4, Vec3
from flint.core.transform import Quat, trs
from std.os.path import exists
from flint.platform.os import env_now, get_time
from flint.platform.vulkan import VulkanGpu, call2, get32, get64, put32, put64
from flint.render.aaa_renderer import AaaUploadedModel, FlintAaaVulkanRenderer
from flint.render.aaa_skinning_rt import DualQuat
from flint.render.flip import flip_error_map, flip_mean
from flint.render.sota_perf_async import SotaFramePacing, sota_overlap_fraction
from flint.render.sota_gi_world_cache import SOTA_GI_MODE_HYBRID, SOTA_GI_MODE_LEGACY, SOTA_GI_MODE_PATH

comptime SOTA_BENCH_STATIC = 0
comptime SOTA_BENCH_MOVING_CAMERA = 1
comptime SOTA_BENCH_MOVING_LIGHT = 2
comptime SOTA_BENCH_SKINNED = 3

comptime SOTA_BENCH_PRESET_DEFAULT = 0  # renderer defaults: VSM + TSR, no RT GI
comptime SOTA_BENCH_PRESET_HIGH = 1  # + RT GI (Area B hybrid surface cache + world radiance cache) + RT reflections

comptime _VK_STYPE_MEMORY_PROPERTIES_2: UInt32 = 1000059006
comptime _VK_STYPE_MEMORY_BUDGET_EXT: UInt32 = 1000237000


def sota_bench_variant_name(v: Int) -> String:
    if v == SOTA_BENCH_MOVING_CAMERA:
        return "moving_cam"
    if v == SOTA_BENCH_MOVING_LIGHT:
        return "moving_light"
    if v == SOTA_BENCH_SKINNED:
        return "skinned"
    return "static"


def sota_bench_scene_dir() -> String:
    var d = env_now("FLINT_SCENES_DIR")
    if d.byte_length() == 0:
        d = env_now("HOME") + "/flint-models/_scenes"
    return d


# ---------------------------------------------------------------------------------------------
# VRAM: VK_EXT_memory_budget heap usage (device-local heaps), 0 when the query is unavailable
# ---------------------------------------------------------------------------------------------
def sota_perf_vram_usage(mut vk: VulkanGpu) -> Tuple[Int, Int]:
    """(bytes used by this process, budget bytes) summed over device-local heaps."""
    if vk.physical == 0:
        return (0, 0)
    var has_budget = False
    for e in vk.device_extensions():
        if e == "VK_EXT_memory_budget":
            has_budget = True
    if not has_budget:
        return (0, 0)
    var f = vk.instance_call("vkGetPhysicalDeviceMemoryProperties2")
    if f == 0:
        return (0, 0)
    var props = vk.scratch(544)
    var budget = vk.scratch(272)
    put32(props, _VK_STYPE_MEMORY_PROPERTIES_2)
    put64(props + 8, budget)
    put32(budget, _VK_STYPE_MEMORY_BUDGET_EXT)
    _ = call2(f, vk.physical, props)
    var heap_count = min(16, Int(get32(props + 16 + 4 + 32 * 8)))
    var used = 0
    var total = 0
    for h in range(heap_count):
        var flags = get32(props + 16 + 264 + h * 16 + 8)
        if (flags & 1) != 0:
            used += get64(budget + 16 + 128 + h * 8)
            total += get64(budget + 16 + h * 8)
    vk.rewind()
    return (used, total)


# ---------------------------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------------------------
struct SotaBenchScene(Movable):
    var name: String
    var path: String
    var model: NativeModel
    var uploaded: AaaUploadedModel
    var bmin: Vec3
    var bmax: Vec3
    var unique_tris: Int
    var drawn_tris: Int
    var draw_count: Int
    var load_s: Float64
    var upload_s: Float64
    var cam_eye0: Vec3
    var cam_tgt0: Vec3
    var cam_eye1: Vec3
    var cam_tgt1: Vec3
    var fov_deg: Float32
    var near_z: Float32
    var sun_dir: Vec3
    var exposure: Float32
    var loaded: Bool

    def __init__(out self, name: String, path: String):
        self.name = name
        self.path = path
        self.model = NativeModel()
        self.uploaded = AaaUploadedModel()
        self.bmin = Vec3(0.0, 0.0, 0.0)
        self.bmax = Vec3(0.0, 0.0, 0.0)
        self.unique_tris = 0
        self.drawn_tris = 0
        self.draw_count = 0
        self.load_s = 0.0
        self.upload_s = 0.0
        self.cam_eye0 = Vec3(0.0, 1.6, -5.0)
        self.cam_tgt0 = Vec3(0.0, 1.6, 0.0)
        self.cam_eye1 = Vec3(0.0, 1.6, -5.0)
        self.cam_tgt1 = Vec3(0.0, 1.6, 0.0)
        self.fov_deg = Float32(60.0)
        self.near_z = Float32(0.1)
        self.sun_dir = Vec3(0.30, 0.86, -0.41).normalized()
        self.exposure = Float32(1.5)
        self.loaded = False


def _scene_bounds(mut scene: SotaBenchScene):
    var globals = scene.model.global_transforms()
    var mesh_aabbs = List[List[Float32]]()
    for m in range(len(scene.model.meshes)):
        mesh_aabbs.append(scene.model.mesh_bounds(m))
        scene.unique_tris += scene.model.meshes[m].triangle_count()
    var lo = Vec3(1e30, 1e30, 1e30)
    var hi = Vec3(-1e30, -1e30, -1e30)
    for n in range(len(scene.model.nodes)):
        var meshes = scene.model.nodes[n].mesh_indices.copy()
        if len(meshes) == 0 and scene.model.nodes[n].mesh >= 0:
            meshes.append(scene.model.nodes[n].mesh)
        for mi in meshes:
            if mi < 0 or mi >= len(mesh_aabbs) or len(mesh_aabbs[mi]) < 6:
                continue
            scene.drawn_tris += scene.model.meshes[mi].triangle_count()
            scene.draw_count += 1
            ref bb = mesh_aabbs[mi]
            for c in range(8):
                var p = Vec3(
                    bb[3] if (c & 1) != 0 else bb[0],
                    bb[4] if (c & 2) != 0 else bb[1],
                    bb[5] if (c & 4) != 0 else bb[2],
                )
                var w = globals[n].transform_point(p)
                lo = Vec3(min(lo.x, w.x), min(lo.y, w.y), min(lo.z, w.z))
                hi = Vec3(max(hi.x, w.x), max(hi.y, w.y), max(hi.z, w.z))
    if len(scene.model.nodes) == 0:
        for m in range(len(mesh_aabbs)):
            if len(mesh_aabbs[m]) < 6:
                continue
            scene.drawn_tris += scene.model.meshes[m].triangle_count()
            scene.draw_count += 1
            lo = Vec3(min(lo.x, mesh_aabbs[m][0]), min(lo.y, mesh_aabbs[m][1]), min(lo.z, mesh_aabbs[m][2]))
            hi = Vec3(max(hi.x, mesh_aabbs[m][3]), max(hi.y, mesh_aabbs[m][4]), max(hi.z, mesh_aabbs[m][5]))
    scene.bmin = lo
    scene.bmax = hi


def _auto_camera(mut scene: SotaBenchScene):
    """Inside the bounds, at 1/4 height, looking along the longest horizontal axis; the fly-through moves
    60 % of that axis. Known scenes override this in `sota_bench_load_scene`."""
    var c = (scene.bmin + scene.bmax) * Float32(0.5)
    var e = (scene.bmax - scene.bmin) * Float32(0.5)
    var y = scene.bmin.y + min(Float32(1.7), e.y * Float32(0.5))
    if e.x >= e.z:
        scene.cam_eye0 = Vec3(c.x - e.x * Float32(0.7), y, c.z)
        scene.cam_tgt0 = Vec3(c.x + e.x, y + e.y * Float32(0.15), c.z)
        scene.cam_eye1 = Vec3(c.x + e.x * Float32(0.2), y, c.z + e.z * Float32(0.2))
        scene.cam_tgt1 = Vec3(c.x + e.x * Float32(1.5), y + e.y * Float32(0.15), c.z - e.z * Float32(0.2))
    else:
        scene.cam_eye0 = Vec3(c.x, y, c.z - e.z * Float32(0.7))
        scene.cam_tgt0 = Vec3(c.x, y + e.y * Float32(0.15), c.z + e.z)
        scene.cam_eye1 = Vec3(c.x + e.x * Float32(0.2), y, c.z + e.z * Float32(0.2))
        scene.cam_tgt1 = Vec3(c.x - e.x * Float32(0.2), y + e.y * Float32(0.15), c.z + e.z * Float32(1.5))
    scene.near_z = max(Float32(0.02), min(Float32(0.2), e.length() * Float32(0.002)))


def sota_bench_load_scene(name: String, path: String) -> SotaBenchScene:
    """CPU load (parse + decode) of one scene; `sota_bench_upload_scene` puts it on a renderer.
    `loaded` stays False when the file is missing or the loader rejects it."""
    var scene = SotaBenchScene(name, path)
    if not exists(path):
        return scene^
    var t0 = get_time()
    scene.model = load_native_model_file(path)
    scene.load_s = get_time() - t0
    if len(scene.model.meshes) == 0:
        return scene^
    _scene_bounds(scene)
    _auto_camera(scene)
    if name == "sponza":
        # Khronos Sponza (metres, Y up): the classic nave view and a walk down the nave.
        scene.cam_eye0 = Vec3(9.5, 1.7, -0.4)
        scene.cam_tgt0 = Vec3(-9.0, 3.6, 0.2)
        scene.cam_eye1 = Vec3(-6.0, 1.9, 1.6)
        scene.cam_tgt1 = Vec3(-14.0, 3.0, -2.0)
        scene.sun_dir = Vec3(0.18, 0.95, 0.25).normalized()
        scene.near_z = Float32(0.05)
        scene.exposure = Float32(3.0)
    elif name == "bistro":
        scene.exposure = Float32(2.5)
    elif name == "forest":
        # 28 m square of CC0 vegetation: eye height walk-in from one corner towards the jacarandas.
        scene.cam_eye0 = Vec3(-13.0, 1.6, -13.0)
        scene.cam_tgt0 = Vec3(4.0, 1.2, 3.0)
        scene.cam_eye1 = Vec3(-4.0, 1.6, -8.0)
        scene.cam_tgt1 = Vec3(12.0, 1.4, 2.0)
        scene.sun_dir = Vec3(0.42, 0.55, 0.72).normalized()
        scene.near_z = Float32(0.05)
        scene.exposure = Float32(1.2)
    scene.loaded = True
    return scene^


def sota_bench_scene_drawable(scene: SotaBenchScene) -> Bool:
    """Loaded, uploaded and with at least one drawable mesh: every probe refuses to measure otherwise."""
    if not scene.loaded or len(scene.uploaded.mesh_entries) == 0:
        return False
    for e in scene.uploaded.mesh_entries:
        if e.mesh_id >= 0:
            return True
    return False


def sota_bench_upload_scene(
    mut ren: FlintAaaVulkanRenderer, mut scene: SotaBenchScene, max_tex_dim: Int = 1024, build_lods: Bool = True
) -> Bool:
    """Upload (BC7/BC5 texture encode, meshlets + LOD DAG, BLAS) onto `ren`; replaces any earlier upload."""
    if len(scene.model.meshes) == 0:
        return False
    var t1 = get_time()
    scene.uploaded = ren.upload_native_model(scene.model, build_lods=build_lods, max_tex_dim=max_tex_dim)
    scene.upload_s = get_time() - t1
    scene.loaded = len(scene.uploaded.mesh_entries) > 0
    return scene.loaded


# ---------------------------------------------------------------------------------------------
# Skinned walkers (glTF skin + clip -> dual quaternions -> GPU skinning + BLAS refit)
# ---------------------------------------------------------------------------------------------
struct SotaBenchCharacters(Movable):
    var model: NativeModel
    var uploaded: AaaUploadedModel
    var count: Int
    var loaded: Bool
    var prev_bones: List[List[DualQuat]]
    var prev_world: List[Mat4]

    def __init__(out self):
        self.model = NativeModel()
        self.uploaded = AaaUploadedModel()
        self.count = 0
        self.loaded = False
        self.prev_bones = List[List[DualQuat]]()
        self.prev_world = List[Mat4]()


def sota_bench_load_characters(mut ren: FlintAaaVulkanRenderer, path: String, count: Int) -> SotaBenchCharacters:
    """`count` independently skinned copies of one skinned glTF (one GPU skinned mesh each)."""
    var ch = SotaBenchCharacters()
    if not exists(path):
        return ch^
    ch.model = load_native_model_file(path)
    if len(ch.model.meshes) == 0 or len(ch.model.skins) == 0:
        return ch^
    for _ in range(count):
        var up = ren.upload_native_model(ch.model, build_lods=False, max_tex_dim=512, upload_skinned=True)
        if len(up.skinned_entries) == 0:
            return ch^
        if ch.count == 0:
            ch.uploaded = up^
        else:
            for e in up.skinned_entries:
                ch.uploaded.skinned_entries.append(e.copy())
            for e in up.mesh_entries:
                ch.uploaded.mesh_entries.append(e.copy())
        ch.count += 1
        ch.prev_bones.append(List[DualQuat]())
        ch.prev_world.append(Mat4())
    ch.loaded = ch.count > 0
    return ch^


def _skin_palette(mut model: NativeModel, skin: Int) -> List[DualQuat]:
    var out = List[DualQuat]()
    var globals = model.global_transforms()
    ref sk = model.skins[skin]
    for j in range(len(sk.joints)):
        var node = sk.joints[j]
        var m = globals[node].copy() if node >= 0 and node < len(globals) else Mat4()
        if len(sk.inverse_bind) >= (j + 1) * 16:
            var ib = Mat4()
            for k in range(16):
                ib.values[k] = sk.inverse_bind[j * 16 + k]
            m = m.multiply(ib)
        out.append(DualQuat.from_mat4(m))
    return out^


def sota_bench_draw_characters(
    mut ren: FlintAaaVulkanRenderer, mut ch: SotaBenchCharacters, t: Float32, origin: Vec3, spacing: Float32
):
    """Animate (clip 0 at time t, phase-shifted per walker), skin on the GPU, and draw every walker
    moving along +x with real previous transforms (motion vectors)."""
    if not ch.loaded:
        return
    var per = len(ch.uploaded.skinned_entries) // max(1, ch.count)
    for i in range(ch.count):
        var ti = t + Float32(i) * Float32(0.37)
        if len(ch.model.animations) > 0:
            var dur = max(Float32(0.1), ch.model.animations[0].duration)
            var tt = ti - dur * Float32(Int(ti / dur))
            ch.model.sample_clip(0, tt)
        var skin = ch.model.meshes[0].skin if ch.model.meshes[0].skin >= 0 else 0
        var bones = _skin_palette(ch.model, skin)
        var prev = ch.prev_bones[i].copy() if len(ch.prev_bones[i]) == len(bones) else bones.copy()
        for k in range(per):
            _ = ren.update_skinned_mesh_bones(ch.uploaded.skinned_entries[i * per + k], bones, prev, defer_to_frame=True)
        ch.prev_bones[i] = bones^
        var x = origin.x + Float32(i) * spacing + ti * Float32(1.1)
        var w = trs(Vec3(x, origin.y, origin.z + Float32(i % 2) * spacing * Float32(0.5)), Quat(), Vec3(1.0, 1.0, 1.0))
        for k in range(per):
            var mid = ch.uploaded.skinned_entries[i * per + k].mesh_entry.mesh_id
            ren.draw_instance(mid, w, ch.prev_world[i], -1, Float32(1.0), t > Float32(0.0))
        ch.prev_world[i] = w^


# ---------------------------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------------------------
@fieldwise_init
struct SotaPassStat(Copyable, Movable):
    var name: String
    var median_ms: Float64


struct SotaBenchResult(Copyable, Movable):
    var scene: String
    var variant: Int
    var preset: Int
    var width: Int
    var height: Int
    var render_w: Int
    var render_h: Int
    var frames: Int
    var gpu_ms: Float64  # median whole-frame GPU time
    var gpu_p95_ms: Float64
    var period_ms: Float64  # wall-clock frame period over the measured frames (throughput incl. pipelining)
    var cpu_ms: Float64  # median draw-list + render_frame wall time (excludes the fence wait)
    var wait_ms: Float64  # median begin_frame fence wait
    var vram_mb: Float64
    var record_ms: Float64  # median command-buffer record time (render_frame internal)
    var submit_ms: Float64  # median vkQueueSubmit + own-fence wait (synchronous) inside render_frame
    var retire_ms: Float64  # median wait on the previous frame's fence (pipelined mode)
    var interval_p99_ms: Float64  # p99 wall-clock frame interval
    var hitch: Float64  # p99 / median frame interval
    var pipelined: Bool
    var passes: List[SotaPassStat]  # sorted, most expensive first
    var flip: Float64  # mean FLIP vs the photo-mode path tracer (static variant only, -1 otherwise)
    var pixels: List[UInt32]  # final frame (sRGB/AgX display output)
    var ok: Bool

    def __init__(out self, scene: String, variant: Int, preset: Int, width: Int, height: Int):
        self.scene = scene
        self.variant = variant
        self.preset = preset
        self.width = width
        self.height = height
        self.render_w = width
        self.render_h = height
        self.frames = 0
        self.gpu_ms = 0.0
        self.gpu_p95_ms = 0.0
        self.period_ms = 0.0
        self.cpu_ms = 0.0
        self.wait_ms = 0.0
        self.vram_mb = 0.0
        self.record_ms = 0.0
        self.submit_ms = 0.0
        self.retire_ms = 0.0
        self.interval_p99_ms = 0.0
        self.hitch = 0.0
        self.pipelined = False
        self.passes = List[SotaPassStat]()
        self.flip = -1.0
        self.pixels = List[UInt32]()
        self.ok = False

    def top_passes(self, n: Int) -> String:
        var s = String("")
        for i in range(min(n, len(self.passes))):
            if i > 0:
                s += ", "
            s += self.passes[i].name + " " + sota_fmt(self.passes[i].median_ms, 2)
        return s


def sota_median(v: List[Float64]) -> Float64:
    if len(v) == 0:
        return 0.0
    var s = v.copy()
    sort(s)
    return s[len(s) // 2]


def sota_percentile(v: List[Float64], p: Float64) -> Float64:
    if len(v) == 0:
        return 0.0
    var s = v.copy()
    sort(s)
    return s[min(len(s) - 1, Int(Float64(len(s) - 1) * p + 0.5))]


def sota_fmt(x: Float64, digits: Int = 3) -> String:
    var scale = 1
    for _ in range(digits):
        scale *= 10
    var neg = x < 0.0
    var v = Int(abs(x) * Float64(scale) + 0.5)
    var frac = String(v % scale)
    while frac.byte_length() < digits:
        frac = "0" + frac
    var s = String(v // scale)
    if digits > 0:
        s += "." + frac
    return ("-" + s) if neg else s


def sota_bench_apply_preset(mut ren: FlintAaaVulkanRenderer, preset: Int, scene: SotaBenchScene):
    ren.sun_dir = scene.sun_dir
    ren.exposure = scene.exposure
    ren.vsm_scene_center = (scene.bmin + scene.bmax) * Float32(0.5)
    ren.vsm_ortho_half_extent = max(Float32(16.0), ((scene.bmax - scene.bmin) * Float32(0.5)).length())
    if preset == SOTA_BENCH_PRESET_HIGH and ren.vk.has_ray_query:
        ren.enable_rt_gi = True
        ren.enable_rt_reflections = True
        ren.gi_rays_per_pixel = 4
        ren.svgf_passes = 2
        ren.sota_gi.mode = SOTA_GI_MODE_HYBRID
        ren.sota_gi.pixel_rays = 4
        ren.sota_gi.accumulate = False
        ren.sota_gi.wrc_center_fixed = False
    else:
        ren.enable_rt_gi = False
        ren.enable_rt_reflections = False
        ren.sota_gi.mode = SOTA_GI_MODE_LEGACY


def _lerp(a: Vec3, b: Vec3, t: Float32) -> Vec3:
    return a + (b - a) * t


def sota_bench_camera(scene: SotaBenchScene, variant: Int, f: Int, frames: Int) -> Tuple[Vec3, Vec3]:
    if variant != SOTA_BENCH_MOVING_CAMERA or frames <= 1:
        return (scene.cam_eye0, scene.cam_tgt0)
    var t = Float32(f) / Float32(frames - 1)
    var s = t * t * (Float32(3.0) - Float32(2.0) * t)
    return (_lerp(scene.cam_eye0, scene.cam_eye1, s), _lerp(scene.cam_tgt0, scene.cam_tgt1, s))


def sota_bench_sun(scene: SotaBenchScene, variant: Int, f: Int, frames: Int) -> Vec3:
    if variant != SOTA_BENCH_MOVING_LIGHT or frames <= 1:
        return scene.sun_dir
    var t = Float32(f) / Float32(frames - 1)
    var ang = t * Float32(1.5707963)
    var d = scene.sun_dir
    var c = cos(ang)
    var s = sin(ang)
    return Vec3(d.x * c - d.z * s, d.y, d.x * s + d.z * c).normalized()


def sota_bench_draw_scene(mut ren: FlintAaaVulkanRenderer, mut scene: SotaBenchScene):
    ren.draw_native_model(scene.model, scene.uploaded)


def sota_bench_run(
    mut ren: FlintAaaVulkanRenderer,
    mut scene: SotaBenchScene,
    mut chars: SotaBenchCharacters,
    variant: Int,
    preset: Int,
    frames: Int,
    warmup: Int = 8,
) -> SotaBenchResult:
    """Render `warmup + frames` frames of one variant and collect medians over the last `frames`."""
    var res = SotaBenchResult(scene.name, variant, preset, ren.output_w, ren.output_h)
    res.render_w = ren.render_w
    res.render_h = ren.render_h
    if not ren.active or not scene.loaded:
        return res^
    sota_bench_apply_preset(ren, preset, scene)
    var names = List[String]()
    var samples = List[List[Float64]]()
    var gpu = List[Float64]()
    var cpu = List[Float64]()
    var wait = List[Float64]()
    var rec = List[Float64]()
    var sub = List[Float64]()
    var ret = List[Float64]()
    var pacing = SotaFramePacing()
    res.pipelined = ren.deferred_readback
    var total = warmup + frames
    var char_origin = _lerp(scene.cam_eye0, scene.cam_tgt0, Float32(0.12))
    var t_measure = 0.0
    for f in range(total):
        if f == warmup:
            t_measure = get_time()
        var fm = max(0, f - warmup)
        var cam = sota_bench_camera(scene, variant, fm, frames)
        ren.sun_dir = sota_bench_sun(scene, variant, fm, frames)
        var t0 = get_time()
        ren.begin_frame(cam[0], cam[1], fov_y_deg=scene.fov_deg, near_z=scene.near_z)
        var t1 = get_time()
        sota_bench_draw_scene(ren, scene)
        if variant == SOTA_BENCH_SKINNED:
            sota_bench_draw_characters(ren, chars, Float32(f) * Float32(1.0 / 30.0), Vec3(char_origin.x - 2.0, scene.bmin.y, char_origin.z), Float32(1.2))
        if not ren.render_frame():
            return res^
        var t2 = get_time()
        if f < warmup:
            continue
        pacing.mark(t2)
        wait.append((t1 - t0) * 1000.0)
        cpu.append((t2 - t1) * 1000.0)
        rec.append(ren.last_cpu_record_ms)
        sub.append(ren.last_cpu_submit_wait_ms)
        ret.append(ren.last_cpu_prev_wait_ms)
        gpu.append(ren.last_frame_gpu_ms)
        for t in ren.get_gpu_pass_timings():
            var idx = -1
            for k in range(len(names)):
                if names[k] == t.pass_name:
                    idx = k
                    break
            if idx < 0:
                names.append(t.pass_name)
                samples.append(List[Float64]())
                idx = len(names) - 1
            samples[idx].append(t.gpu_ms)
    res.frames = frames
    res.period_ms = (get_time() - t_measure) * 1000.0 / Float64(max(1, frames))
    res.gpu_ms = sota_median(gpu)
    res.gpu_p95_ms = sota_percentile(gpu, 0.95)
    res.cpu_ms = sota_median(cpu)
    res.wait_ms = sota_median(wait)
    res.record_ms = sota_median(rec)
    res.submit_ms = sota_median(sub)
    res.retire_ms = sota_median(ret)
    res.interval_p99_ms = pacing.p99_ms()
    res.hitch = pacing.hitch_ratio()
    ren.flush_deferred_frame()  # pipelined mode: the last frame is still in flight
    var vr = sota_perf_vram_usage(ren.vk)
    res.vram_mb = Float64(vr[0]) / (1024.0 * 1024.0)
    for k in range(len(names)):
        res.passes.append(SotaPassStat(names[k], sota_median(samples[k])))
    # most expensive first (insertion sort: a few dozen passes)
    for i in range(1, len(res.passes)):
        var j = i
        while j > 0 and res.passes[j].median_ms > res.passes[j - 1].median_ms:
            var tmp = res.passes[j].copy()
            res.passes[j] = res.passes[j - 1].copy()
            res.passes[j - 1] = tmp^
            j -= 1
    res.pixels = ren.readback_pixels()
    res.ok = len(res.pixels) == ren.output_w * ren.output_h
    return res^


def sota_bench_photo_reference(
    mut ren: FlintAaaVulkanRenderer, mut scene: SotaBenchScene, frames: Int, bounces: Int = 8
) -> List[UInt32]:
    """Flint's path tracer in photo mode (Area B): 4 paths/px/frame, progressive accumulation on the static
    camera, no SVGF, WRC termination of the last vertex. Restores the HIGH preset afterwards."""
    if not ren.active or not scene.loaded or not ren.vk.has_ray_query:
        return List[UInt32]()
    sota_bench_apply_preset(ren, SOTA_BENCH_PRESET_HIGH, scene)
    ren.sota_gi.mode = SOTA_GI_MODE_PATH
    ren.sota_gi.pixel_rays = 4
    ren.sota_gi.pt_bounces = bounces
    ren.sota_gi.pt_wrc_terminate = True
    ren.sota_gi.accumulate = True
    var svgf = ren.svgf_passes
    ren.svgf_passes = 0
    for _ in range(frames):
        ren.begin_frame(scene.cam_eye0, scene.cam_tgt0, fov_y_deg=scene.fov_deg, near_z=scene.near_z)
        sota_bench_draw_scene(ren, scene)
        if not ren.render_frame():
            break
    var px = ren.readback_pixels()
    ren.svgf_passes = svgf
    ren.sota_gi.accumulate = False
    sota_bench_apply_preset(ren, SOTA_BENCH_PRESET_HIGH, scene)
    return px^


def sota_bench_flip(reference: List[UInt32], test: List[UInt32], width: Int, height: Int) -> Float64:
    if len(reference) != width * height or len(test) != width * height:
        return -1.0
    return flip_mean(flip_error_map(reference, test, width, height))


def sota_bench_hiz_shadow_probe(
    mut ren: FlintAaaVulkanRenderer, mut scene: SotaBenchScene, frames: Int = 6
) -> Tuple[Float64, Float64, List[UInt32], List[UInt32]]:
    """Area B's report: with two-phase HiZ occlusion culling on, sun shadows vanish from frame 2 on
    (occluded casters culled from the shadow passes too). Renders the static view with HiZ on and off and
    returns (mean FLIP on vs off, FLIP off vs the next off frame = temporal noise floor, on, off).
    A correct renderer stays at the floor: culling must not change pixels."""
    var keep = ren.enable_hiz_cull
    ren.enable_hiz_cull = True
    for _ in range(frames):
        ren.begin_frame(scene.cam_eye0, scene.cam_tgt0, fov_y_deg=scene.fov_deg, near_z=scene.near_z)
        sota_bench_draw_scene(ren, scene)
        _ = ren.render_frame()
    var on = ren.readback_pixels()
    ren.enable_hiz_cull = False
    for _ in range(frames):
        ren.begin_frame(scene.cam_eye0, scene.cam_tgt0, fov_y_deg=scene.fov_deg, near_z=scene.near_z)
        sota_bench_draw_scene(ren, scene)
        _ = ren.render_frame()
    var off = ren.readback_pixels()
    ren.begin_frame(scene.cam_eye0, scene.cam_tgt0, fov_y_deg=scene.fov_deg, near_z=scene.near_z)
    sota_bench_draw_scene(ren, scene)
    _ = ren.render_frame()
    var off2 = ren.readback_pixels()
    ren.enable_hiz_cull = keep
    var w = ren.output_w
    var h = ren.output_h
    return (sota_bench_flip(off, on, w, h), sota_bench_flip(off, off2, w, h), on^, off^)


# ---------------------------------------------------------------------------------------------
# Dynamic-scene robustness: light-change lag (GI caches / denoiser history) and camera-motion ghosting
# ---------------------------------------------------------------------------------------------
struct SotaLagResult(Movable):
    var flip_curve: List[Float64]  # FLIP(frame k after the change, converged reference), k = 1..frames
    var noise_floor: Float64  # FLIP between two converged frames of the reference (temporal noise)
    var lag_frames: Int  # first k with FLIP <= floor * 1.25 + 0.005 (frames + 1 if never)
    var first: List[UInt32]  # frame 1 after the change
    var reference: List[UInt32]
    var failed_frames: Int  # render_frame() returned False (stale readback: the curve is meaningless if > 0)

    def __init__(out self):
        self.failed_frames = 0
        self.flip_curve = List[Float64]()
        self.noise_floor = 0.0
        self.lag_frames = 0
        self.first = List[UInt32]()
        self.reference = List[UInt32]()


def _render_n(mut ren: FlintAaaVulkanRenderer, mut scene: SotaBenchScene, eye: Vec3, tgt: Vec3, n: Int) -> Int:
    """Renders `n` frames; returns how many `render_frame` calls failed."""
    var failed = 0
    for _ in range(n):
        ren.begin_frame(eye, tgt, fov_y_deg=scene.fov_deg, near_z=scene.near_z)
        sota_bench_draw_scene(ren, scene)
        if not ren.render_frame():
            failed += 1
    return failed


def sota_bench_light_lag(
    mut ren: FlintAaaVulkanRenderer, mut scene: SotaBenchScene, sun_a: Vec3, sun_b: Vec3,
    converge: Int = 48, frames: Int = 24,
) -> SotaLagResult:
    """Converge under sun A, cut to sun B, measure how many frames the image needs to reach a reference that
    converged under B from scratch. A high lag means stale GI caches / denoiser history (light lag);
    a curve that settles above the floor means a cache never fully re-converges."""
    var out = SotaLagResult()
    if not sota_bench_scene_drawable(scene):
        # round 2: an unloaded scene renders sky only, identical for every sun direction; that is what the
        # round-1 "no sunlight in a fresh process" lag of exactly 1 frame with a ~0 floor was
        out.failed_frames = frames + 1
        out.lag_frames = -1
        return out^
    var eye = scene.cam_eye0
    var tgt = scene.cam_tgt0
    # reference: B converged (history from A is flushed by `converge` frames under B)
    ren.sun_dir = sun_a
    out.failed_frames += _render_n(ren, scene, eye, tgt, 4)
    ren.sun_dir = sun_b
    out.failed_frames += _render_n(ren, scene, eye, tgt, converge * 2)
    out.reference = ren.readback_pixels()
    out.failed_frames += _render_n(ren, scene, eye, tgt, 1)
    var again = ren.readback_pixels()
    out.noise_floor = sota_bench_flip(out.reference, again, ren.output_w, ren.output_h)
    # the cut
    ren.sun_dir = sun_a
    out.failed_frames += _render_n(ren, scene, eye, tgt, converge)
    ren.sun_dir = sun_b
    out.lag_frames = frames + 1
    for k in range(frames):
        out.failed_frames += _render_n(ren, scene, eye, tgt, 1)
        var px = ren.readback_pixels()
        var f = sota_bench_flip(out.reference, px, ren.output_w, ren.output_h)
        out.flip_curve.append(f)
        if k == 0:
            out.first = px.copy()
        if out.lag_frames > frames and f <= out.noise_floor * 1.25 + 0.005:
            out.lag_frames = k + 1
    ren.sun_dir = scene.sun_dir
    return out^


def sota_bench_motion_ghosting(
    mut ren: FlintAaaVulkanRenderer, mut scene: SotaBenchScene, frames: Int = 24, converge: Int = 48
) -> Tuple[Float64, Float64, List[UInt32], List[UInt32]]:
    """Fly the moving-camera path for `frames` frames, then compare the last frame with the same view after
    `converge` static frames: (FLIP moving vs converged, FLIP converged vs converged+1 noise floor,
    moving frame, converged frame). The excess over the floor is ghosting / history lag / disocclusion."""
    for f in range(frames):
        var cam = sota_bench_camera(scene, SOTA_BENCH_MOVING_CAMERA, f, frames)
        ren.begin_frame(cam[0], cam[1], fov_y_deg=scene.fov_deg, near_z=scene.near_z)
        sota_bench_draw_scene(ren, scene)
        _ = ren.render_frame()
    var moving = ren.readback_pixels()
    var last = sota_bench_camera(scene, SOTA_BENCH_MOVING_CAMERA, frames - 1, frames)
    _ = _render_n(ren, scene, last[0], last[1], converge)
    var conv = ren.readback_pixels()
    _ = _render_n(ren, scene, last[0], last[1], 1)
    var conv2 = ren.readback_pixels()
    var w = ren.output_w
    var h = ren.output_h
    return (sota_bench_flip(conv, moving, w, h), sota_bench_flip(conv, conv2, w, h), moving^, conv^)


def sota_bench_geometry_mb(scene: SotaBenchScene) -> Int:
    """Geometry-pool size (MB) for `FlintAaaVulkanRenderer(geometry_mb=...)`: vertices and indices of every
    primitive with 3x headroom for the QEM LOD levels (2x overflowed the forest), never below the v5 default of 32 MB."""
    var verts = 0
    var idx = 0
    for m in scene.model.meshes:
        verts += m.vertex_count()
        idx += len(m.indices)
    var v_mb = verts * AAA_VERTEX_BYTES * 3 // (1024 * 1024)
    var i_mb = idx * 4 * 3 // (1024 * 1024)
    return max(32, max(v_mb, i_mb) + 16)


def sota_bench_side_queue_overlap(
    mut ren: FlintAaaVulkanRenderer, mut scene: SotaBenchScene, reps: Int = 7, mb: Int = 256, copies: Int = 6
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Round 2 async-compute probe: does work on `VulkanGpu`'s side queue (a compute-only family on NVIDIA)
    run beside a frame? Times (median wall ms) a synchronous frame alone, a side-queue job alone (`copies`
    buffer copies of `mb` MB) and both submitted together; returns (frame, side, both, overlap fraction,
    see `sota_overlap_fraction`). Returns all -1 without a side queue."""
    if not ren.active or not ren.vk.has_side_queue() or not scene.loaded:
        return (-1.0, -1.0, -1.0, -1.0)
    var idx = ren.vk.allocate_side_command_buffers(1)
    if idx < 0:
        return (-1.0, -1.0, -1.0, -1.0)
    var bytes = mb * 1024 * 1024
    var a = ren.vk.create_buffer(bytes, local=True)
    var b = ren.vk.create_buffer(bytes, local=True)
    if a.size == 0 or b.size == 0:
        ren.vk.destroy_buffer(a)
        ren.vk.destroy_buffer(b)
        return (-1.0, -1.0, -1.0, -1.0)
    var keep_defer = ren.deferred_readback
    ren.deferred_readback = False
    var t_frame = List[Float64]()
    var t_side = List[Float64]()
    var t_both = List[Float64]()
    for r in range(reps + 1):
        # frame alone
        var t0 = get_time()
        ren.begin_frame(scene.cam_eye0, scene.cam_tgt0, fov_y_deg=scene.fov_deg, near_z=scene.near_z)
        sota_bench_draw_scene(ren, scene)
        _ = ren.render_frame()
        var t1 = get_time()
        # side job alone
        _ = ren.vk.begin_commands(idx)
        for c in range(copies):
            if c % 2 == 0:
                ren.vk.copy_buffer(idx, a, 0, b, 0, bytes)
            else:
                ren.vk.copy_buffer(idx, b, 0, a, 0, bytes)
        var f = ren.vk.submit_side(idx)
        _ = ren.vk.wait_side(f)
        var t2 = get_time()
        # both: side job first, then the frame; wait for both
        _ = ren.vk.begin_commands(idx)
        for c in range(copies):
            if c % 2 == 0:
                ren.vk.copy_buffer(idx, a, 0, b, 0, bytes)
            else:
                ren.vk.copy_buffer(idx, b, 0, a, 0, bytes)
        var f2 = ren.vk.submit_side(idx)
        ren.begin_frame(scene.cam_eye0, scene.cam_tgt0, fov_y_deg=scene.fov_deg, near_z=scene.near_z)
        sota_bench_draw_scene(ren, scene)
        _ = ren.render_frame()
        _ = ren.vk.wait_side(f2)
        var t3 = get_time()
        if r == 0:
            continue  # warm-up
        t_frame.append((t1 - t0) * 1000.0)
        t_side.append((t2 - t1) * 1000.0)
        t_both.append((t3 - t2) * 1000.0)
    ren.deferred_readback = keep_defer
    ren.vk.wait_idle()
    ren.vk.destroy_buffer(a)
    ren.vk.destroy_buffer(b)
    var mf = sota_median(t_frame)
    var ms = sota_median(t_side)
    var mbth = sota_median(t_both)
    return (mf, ms, mbth, sota_overlap_fraction(mf, ms, mbth))
