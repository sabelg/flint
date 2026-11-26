"""Area G (shadows) render SOTA, round 2 (`test_flint_v5_sota_shadow.mojo`).

1. CPU: toroidal clipmap page addressing (slot <-> window-local round trip, scroll invalidation = only the pages that
   enter the window, depth-snap index).
2. GPU (skipped without Vulkan): a 200 m field of pillars, spheres and a hilly ground under a low sun.
   - VSM command truncation regression (the Forest "HiZ drops shadow casters" bug): a renderer created with a tiny
     `max_draws` must render the same shadows as one with a large budget, and must not drop or cache truncated pages.
   - Toroidal VSM: a camera dolly re-renders far fewer pages than v5 cascade-relative addressing and converges to the
     same image as a cold start at the final camera.
   - Screen-space contact shadows, GPU page packing and RT-shadow comparisons (see the functions below).
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, cos, max, min, sin, sqrt
from std.os import getenv
from flint.assets.image import write_png
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh
from flint.core.math import Vec3
from flint.core.transform import Quat, trs
from flint.platform.os import get_time, make_dirs
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_vsm import VSM_CASCADE_PAGES_DIM, VSM_VIRTUAL_PAGE_COUNT, VsmPageManager
from flint.render.flip import flip_error_map, flip_mean
from flint.render.image_diff import compare_pixels
from flint.render.sota_shadow_vsm import (
    sota_shadow_depth_snap_index,
    sota_shadow_local_from_slot,
    sota_shadow_scroll_invalidate,
    sota_shadow_slot_from_local,
    sota_shadow_toroidal_offset,
    sota_shadow_toroidal_offsets,
)


def _fail(msg: String):
    print("FAIL:", msg)
    external_call["exit", NoneType](Int32(1))


def _assert(cond: Bool, msg: String):
    if not cond:
        _fail(msg)


def _eval_dir() -> String:
    var home = getenv("HOME", "/tmp")
    var d = home + "/flint-models/_eval/render/G-r2"
    _ = make_dirs(d)
    return d + "/"


def _fmt(x: Float64) -> String:
    var r = Int(x * 1000.0 + 0.5) if x >= 0.0 else -Int(-x * 1000.0 + 0.5)
    var neg = r < 0
    if neg:
        r = -r
    var frac = String(r % 1000)
    while frac.byte_length() < 3:
        frac = "0" + frac
    return ("-" if neg else "") + String(r // 1000) + "." + frac


def _median(v: List[Float64]) -> Float64:
    if len(v) == 0:
        return 0.0
    var c = v.copy()
    for i in range(1, len(c)):
        var j = i
        while j > 0 and c[j] < c[j - 1]:
            var t = c[j]
            c[j] = c[j - 1]
            c[j - 1] = t
            j -= 1
    return c[len(c) // 2]


# ---------------------------------------------------------------------------------------------
# 1. CPU
# ---------------------------------------------------------------------------------------------
def _valid_in_cascade(vsm: VsmPageManager, c: Int) -> Int:
    var n = 0
    var qx = (c & 1) * 64
    var qy = (c >> 1) * 64
    for y in range(64):
        for x in range(64):
            if vsm.virtual_valid[(qy + y) * 128 + qx + x]:
                n += 1
    return n


def test_cpu_toroidal():
    var origins = List[Tuple[Int, Int]]()
    origins.append((5, -3))
    origins.append((-70, 12))
    origins.append((0, 0))
    origins.append((1000, -1000))
    var offs = sota_shadow_toroidal_offsets(origins)
    for c in range(4):
        for ly in range(0, 64, 7):
            for lx in range(0, 64, 5):
                var v = sota_shadow_slot_from_local(c, lx, ly, offs)
                var back = sota_shadow_local_from_slot(v & 127, v >> 7, offs)
                _assert(back[0] == lx and back[1] == ly, "toroidal slot -> local round trip")
                _assert((v & 127) // 64 == (c & 1) and (v >> 7) // 64 == (c >> 1), "slot stays in its cascade quadrant")
    # a world page keeps its slot when the window scrolls: window page lx at origin ox == page lx - 1 at ox + 1
    var o0 = List[Tuple[Int, Int]]()
    var o1 = List[Tuple[Int, Int]]()
    for _ in range(4):
        o0.append((10, 20))
        o1.append((11, 19))  # +1 page in x, -1 in light y (= +1 window row)
    var f0 = sota_shadow_toroidal_offsets(o0)
    var f1 = sota_shadow_toroidal_offsets(o1)
    _assert(sota_shadow_slot_from_local(0, 30, 40, f0) == sota_shadow_slot_from_local(0, 29, 39, f1), "world page keeps its slot")
    # scroll invalidation: one page in x and one row in y -> 64 + 64 - 1 entering pages, the rest stay valid
    var vsm = VsmPageManager()
    for i in range(VSM_VIRTUAL_PAGE_COUNT):
        vsm.virtual_valid[i] = True
    var n = sota_shadow_scroll_invalidate(vsm, 0, (10, 20), (11, 19))
    _assert(n == 127, "diagonal one-page scroll invalidates 127 pages, got " + String(n))
    _assert(_valid_in_cascade(vsm, 0) == 4096 - 127, "the other cascade-0 pages stay cached")
    _assert(_valid_in_cascade(vsm, 1) == 4096, "other cascades untouched")
    # the invalidated slots are exactly the new window's last column and last row
    for ly in range(64):
        for lx in range(64):
            var v = sota_shadow_slot_from_local(0, lx, ly, f1)
            var entering = lx == 63 or ly == 63
            _assert(vsm.virtual_valid[v] != entering, "entering page <-> invalidated slot")
    var n_big = sota_shadow_scroll_invalidate(vsm, 2, (0, 0), (64, 0))
    _assert(n_big == 4096 and _valid_in_cascade(vsm, 2) == 0, "a full-window jump invalidates the cascade")
    _assert(sota_shadow_toroidal_offset((32, -32)) == (0, 0), "offset of the reference origin")
    var sd0 = sota_shadow_depth_snap_index(Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 0.0))
    var sd1 = sota_shadow_depth_snap_index(Vec3(0.0, 1.0, 0.0), Vec3(0.0, 40.0, 0.0))
    _assert(sd0 != sd1, "40 m along the light changes the 32 m depth snap")
    print("[PASS] toroidal VSM page addressing (CPU)")


# ---------------------------------------------------------------------------------------------
# 2. GPU scene
# ---------------------------------------------------------------------------------------------
def _terrain(n: Int, size: Float32) -> NativeMesh:
    var mb = MeshBuilder()
    for z in range(n + 1):
        for x in range(n + 1):
            var fx = Float32(x) / Float32(n)
            var fz = Float32(z) / Float32(n)
            var h = Float32(0.004) * sin(fx * 31.0) * cos(fz * 27.0)
            _ = mb.add_vertex(Vec3((fx - 0.5) * size, h * size, (fz - 0.5) * size), Vec3(0.0, 1.0, 0.0), fx * 8.0, fz * 8.0)
    for z in range(n):
        for x in range(n):
            var a = z * (n + 1) + x
            mb.add_quad(a, a + 1, a + n + 2, a + n + 1)
    return mb^.finish()


struct ShadowScene(Movable):
    var ren: FlintAaaVulkanRenderer
    var ok: Bool
    var ground: Int
    var pillar: Int
    var ball: Int
    var tile: Int
    var mat_g: Int
    var mat_p: Int
    var mat_b: Int

    def __init__(out self, W: Int, H: Int, max_draws: Int = 131072):
        self.ren = FlintAaaVulkanRenderer(width=W, height=H, max_draws=max_draws)
        self.ok = self.ren.active
        self.ground = -1
        self.pillar = -1
        self.ball = -1
        self.tile = -1
        self.mat_g = -1
        self.mat_p = -1
        self.mat_b = -1
        if not self.ok:
            return
        self.ren.tsr_history_weight = Float32(0.0)
        self.ren.enable_auto_exposure = False
        self.ren.bloom_intensity = Float32(0.0)
        self.ren.camera_lens.vignette_strength = Float32(0.0)
        self.ren.enable_rt_gi = False
        self.ren.enable_rt_reflections = False
        self.ren.enable_rt_shadows = False
        self.ren.sun_dir = Vec3(0.55, 0.42, 0.30).normalized()
        self.ren.sun_radiance = Vec3(3.2, 3.0, 2.8)
        self.mat_g = self.ren.add_material(AaaMaterial(base_color=Vec3(0.55, 0.52, 0.48), roughness=0.9, metallic=0.0))
        self.mat_p = self.ren.add_material(AaaMaterial(base_color=Vec3(0.7, 0.7, 0.72), roughness=0.6, metallic=0.0))
        self.mat_b = self.ren.add_material(AaaMaterial(base_color=Vec3(0.75, 0.25, 0.2), roughness=0.5, metallic=0.0))
        self.ground = self.ren.add_mesh(_terrain(96, 200.0), self.mat_g).mesh_id
        var pb = MeshBuilder()
        pb.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), 0.02)
        self.pillar = self.ren.add_mesh(pb^.finish(), self.mat_p).mesh_id
        var sb = MeshBuilder()
        sb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), rings=24, segments=48)
        self.ball = self.ren.add_mesh(sb^.finish(), self.mat_b).mesh_id
        var tb = MeshBuilder()
        tb.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), 0.01)
        self.tile = self.ren.add_mesh(tb^.finish(), self.mat_b).mesh_id

    def draw(mut self):
        self.ren.draw_instance(self.ground, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        for gz in range(-12, 13):
            for gx in range(-12, 13):
                var p = Vec3(Float32(gx) * 3.1, 0.0, Float32(gz) * 3.3)
                if (gx + gz) % 2 == 0:
                    var hgt = Float32(1.2) + Float32((gx * 7 + gz * 3) % 5) * Float32(0.5)
                    self.ren.draw_instance(self.pillar, trs(Vec3(p.x, hgt * 0.5, p.z), Quat(), Vec3(0.18, hgt * 0.5, 0.18)))
                else:
                    self.ren.draw_instance(self.ball, trs(Vec3(p.x, 0.75, p.z), Quat(), Vec3(0.6, 0.6, 0.6)), uniform_scale=0.6)
                # thin contact casters: 3 cm slabs lying on the ground (the VSM bias loses most of their shadow)
                self.ren.draw_instance(self.tile, trs(Vec3(p.x + 1.3, 0.035, p.z + 0.4), Quat(), Vec3(0.35, 0.03, 0.22)))

    def frame(mut self, eye: Vec3, tgt: Vec3) -> Bool:
        self.ren.begin_frame(eye, tgt, fov_y_deg=60.0, near_z=0.05)
        self.draw()
        return self.ren.render_frame(capture_readback=True)

    def shutdown(mut self):
        if self.ok:
            self.ren.shutdown()


def test_gpu_vsm_truncation():
    """Regression: VSM indirect commands were capped at `max_draws` and truncated pages cached as lit."""
    var W = 320
    var H = 180
    var big = ShadowScene(W, H, 131072)
    if not big.ok:
        print("[SKIP] Vulkan unavailable: GPU shadow tests skipped")
        return
    var eye = Vec3(-6.0, 2.2, -14.0)
    var tgt = Vec3(4.0, 0.4, 2.0)
    for _ in range(6):
        _ = big.frame(eye, tgt)
    var ref_px = big.ren.readback_pixels()
    var ref_cmds = big.ren.sota_shadow.vsm_cmds_last
    big.shutdown()
    var small = ShadowScene(W, H, 256)
    var first_cmds = 0
    for f in range(6):
        _ = small.frame(eye, tgt)
        if f == 0:
            first_cmds = small.ren.sota_shadow.vsm_cmds_last
    var px = small.ren.readback_pixels()
    var d = compare_pixels(px, ref_px, W, H, 6)
    var line = String("  VSM commands frame 0: ") + String(first_cmds) + " (max_draws 256 -> command capacity "
    line += String(small.ren.sota_shadow.vsm_cmd_capacity) + ", grows " + String(small.ren.sota_shadow.vsm_cmd_grows)
    line += ", dropped " + String(small.ren.sota_shadow.vsm_cmds_dropped_total) + "); image vs big budget: " + d.summary()
    print(line)
    _ = write_png(_eval_dir() + "trunc_small_budget.png", px, W, H, stamped=False)
    _ = write_png(_eval_dir() + "trunc_big_budget.png", ref_px, W, H, stamped=False)
    _assert(first_cmds > 256, "scene needs more VSM commands than max_draws (else the test proves nothing)")
    _assert(small.ren.sota_shadow.vsm_cmds_dropped_total == 0, "no VSM commands dropped")
    _assert(small.ren.sota_shadow.vsm_cmd_capacity >= first_cmds, "VSM command buffer grew to the need")
    _assert(d.over_tolerance <= W * H // 500, "small max_draws renders the same shadows: " + d.summary())
    _ = ref_cmds
    small.shutdown()
    print("[PASS] VSM command budget is independent of max_draws (Forest shadow-caster bug)")


def _dolly_run(toroidal: Bool, W: Int, H: Int, frames: Int, mut pages: List[Int], mut px_out: List[UInt32]) -> Bool:
    var sc = ShadowScene(W, H)
    if not sc.ok:
        return False
    sc.ren.sota_shadow.toroidal = toroidal
    for f in range(frames):
        var t = Float32(f) * Float32(0.31)
        var eye = Vec3(-8.0 + t, 2.0, -10.0 + t * 0.6)
        _ = sc.frame(eye, eye + Vec3(6.0, -1.2, 9.0))
        pages.append(sc.ren.last_vsm_dirty_pages)
    px_out = sc.ren.readback_pixels()
    sc.shutdown()
    return True


def test_gpu_toroidal():
    var W = 320
    var H = 180
    var frames = 40
    var p_tor = List[Int]()
    var p_v5 = List[Int]()
    var px_tor = List[UInt32]()
    var px_v5 = List[UInt32]()
    if not _dolly_run(True, W, H, frames, p_tor, px_tor):
        print("[SKIP] Vulkan unavailable: toroidal test skipped")
        return
    _ = _dolly_run(False, W, H, frames, p_v5, px_v5)
    # cold-start reference at the final camera
    var sc = ShadowScene(W, H)
    var t = Float32(frames - 1) * Float32(0.31)
    var eye = Vec3(-8.0 + t, 2.0, -10.0 + t * 0.6)
    for _ in range(6):
        _ = sc.frame(eye, eye + Vec3(6.0, -1.2, 9.0))
    var ref_px = sc.ren.readback_pixels()
    sc.shutdown()
    var s_tor = 0
    var s_v5 = 0
    for f in range(4, frames):
        s_tor += p_tor[f]
        s_v5 += p_v5[f]
    var d_tor = compare_pixels(px_tor, ref_px, W, H, 6)
    var d_v5 = compare_pixels(px_v5, ref_px, W, H, 6)
    print("  dolly " + String(frames - 4) + " frames: pages rendered toroidal " + String(s_tor) + " vs v5 " + String(s_v5)
          + "; final vs cold start: toroidal " + d_tor.summary() + " | v5 " + d_v5.summary())
    _ = write_png(_eval_dir() + "toroidal_final.png", px_tor, W, H, stamped=False)
    _ = write_png(_eval_dir() + "toroidal_v5_final.png", px_v5, W, H, stamped=False)
    _ = write_png(_eval_dir() + "toroidal_cold_ref.png", ref_px, W, H, stamped=False)
    _assert(s_tor * 2 < s_v5, "toroidal re-renders at most half the pages of cascade-relative addressing")
    _assert(d_tor.over_tolerance <= W * H // 500, "toroidal cache converges to the cold-start image: " + d_tor.summary())
    print("[PASS] toroidal clipmap VSM: cached pages survive camera motion")


struct ShadowCfg(Copyable, Movable):
    var px: List[UInt32]
    var deferred_ms: Float64
    var rt_ms: Float64
    var vsm_ms: Float64
    var gpu_ms: Float64

    def __init__(out self):
        self.px = List[UInt32]()
        self.deferred_ms = 0.0
        self.rt_ms = 0.0
        self.vsm_ms = 0.0
        self.gpu_ms = 0.0


def _render_cfg(mode: Int, W: Int, H: Int, eye: Vec3, tgt: Vec3) -> ShadowCfg:
    """mode 0: VSM PCSS; 1: VSM + contact shadows; 2: RT sun shadows (ray query, denoised)."""
    var out = ShadowCfg()
    var sc = ShadowScene(W, H)
    if not sc.ok:
        return out^
    sc.ren.enable_gpu_profiler = True
    sc.ren.tsr_history_weight = Float32(0.9)  # static camera: let TSR integrate the RT / contact noise
    sc.ren.sota_shadow.contact = mode == 1
    sc.ren.enable_rt_shadows = mode == 2 and sc.ren.vk.has_ray_query
    var d = List[Float64]()
    var r = List[Float64]()
    var v = List[Float64]()
    var g = List[Float64]()
    for f in range(40):
        _ = sc.frame(eye, tgt)
        if f < 16:
            continue
        var dd = 0.0
        var rr = 0.0
        var vv = 0.0
        for t in sc.ren.get_gpu_pass_timings():
            if t.pass_name == "DeferredMultiLobeRtShade":
                dd += t.gpu_ms
            elif t.pass_name == "RtSoftSunShadows":
                rr += t.gpu_ms
            elif t.pass_name == "VsmMarkPages" or t.pass_name == "VsmTileRaster":
                vv += t.gpu_ms
        d.append(dd)
        r.append(rr)
        v.append(vv)
        g.append(sc.ren.last_frame_gpu_ms)
    out.px = sc.ren.readback_pixels()
    out.deferred_ms = _median(d)
    out.rt_ms = _median(r)
    out.vsm_ms = _median(v)
    out.gpu_ms = _median(g)
    sc.shutdown()
    return out^


def test_gpu_contact_vs_rt():
    """VSM, VSM + contact shadows and RT shadows on the same view: quality (FLIP against RT) and cost."""
    var W = 640
    var H = 360
    var eye = Vec3(-3.2, 1.1, -5.0)
    var tgt = Vec3(1.0, 0.1, 1.5)
    var vsm = _render_cfg(0, W, H, eye, tgt)
    if len(vsm.px) == 0:
        print("[SKIP] Vulkan unavailable: contact shadow test skipped")
        return
    var con = _render_cfg(1, W, H, eye, tgt)
    _ = write_png(_eval_dir() + "contact_off_vsm.png", vsm.px, W, H, stamped=False)
    _ = write_png(_eval_dir() + "contact_on_vsm.png", con.px, W, H, stamped=False)
    var dc = compare_pixels(con.px, vsm.px, W, H, 6)
    var line = String("  contact shadows: changed ") + dc.summary() + "; deferred pass " + _fmt(vsm.deferred_ms)
    line += " -> " + _fmt(con.deferred_ms) + " ms"
    print(line)
    _assert(dc.over_tolerance > W * H // 2000, "contact shadows darken some contacts")
    _assert(dc.over_tolerance < W * H // 4, "contact shadows only touch contacts, not the whole frame")
    var rt = _render_cfg(2, W, H, eye, tgt)
    _ = write_png(_eval_dir() + "contact_rt_reference.png", rt.px, W, H, stamped=False)
    if len(rt.px) == W * H:
        var f_vsm = flip_mean(flip_error_map(rt.px, vsm.px, W, H))
        var f_con = flip_mean(flip_error_map(rt.px, con.px, W, H))
        print("  FLIP vs RT shadows: VSM " + _fmt(f_vsm) + ", VSM + contact " + _fmt(f_con)
              + " | GPU frame VSM " + _fmt(vsm.gpu_ms) + " ms (VSM passes " + _fmt(vsm.vsm_ms)
              + "), +contact " + _fmt(con.gpu_ms) + ", RT " + _fmt(rt.gpu_ms) + " ms (RtSoftSunShadows "
              + _fmt(rt.rt_ms) + ")")
        _assert(f_con <= f_vsm + 0.002, "contact shadows do not move the VSM away from the RT reference")
    print("[PASS] screen-space contact shadows")


struct PagesRun(Copyable, Movable):
    var px: List[UInt32]
    var cold_wall_ms: Float64  # first frame (every page dirty)
    var move_wall_ms: Float64  # median wall time of the dolly frames
    var move_cpu_vsm_ms: Float64
    var move_gpu_vsm_ms: Float64  # median VSM GPU passes during the dolly
    var static_gpu_vsm_ms: Float64
    var pages: Int
    var gpu_active: Bool

    def __init__(out self):
        self.px = List[UInt32]()
        self.cold_wall_ms = 0.0
        self.move_wall_ms = 0.0
        self.move_cpu_vsm_ms = 0.0
        self.move_gpu_vsm_ms = 0.0
        self.static_gpu_vsm_ms = 0.0
        self.pages = 0
        self.gpu_active = False


def _vsm_gpu_ms(mut sc: ShadowScene) -> Float64:
    var v = 0.0
    for t in sc.ren.get_gpu_pass_timings():
        if t.pass_name == "VsmMarkPages" or t.pass_name == "VsmTileRaster" or t.pass_name == "SotaShadowVsmGpu":
            v += t.gpu_ms
    return v


def _pages_run(gpu: Bool, W: Int, H: Int) -> PagesRun:
    var out = PagesRun()
    var sc = ShadowScene(W, H)
    if not sc.ok:
        return out^
    sc.ren.enable_gpu_profiler = True
    sc.ren.sota_shadow.toroidal = True
    sc.ren.sota_shadow.gpu_pages = gpu
    var eye0 = Vec3(-8.0, 2.0, -10.0)
    _ = sc.frame(eye0, eye0 + Vec3(6.0, -1.2, 9.0))  # pipelines, first upload
    out.gpu_active = sc.ren.sota_shadow.gpu_active
    var st = List[Float64]()
    for _ in range(8):
        _ = sc.frame(eye0, eye0 + Vec3(6.0, -1.2, 9.0))
        st.append(_vsm_gpu_ms(sc))
    # teleport: every page dirty (the CPU path also does its synchronous request readback here)
    var cold = List[Float64]()
    for k in range(3):
        var e_far = Vec3(40.0 * Float32(k + 1), 2.0, -30.0)
        var t0 = get_time()
        _ = sc.frame(e_far, e_far + Vec3(-6.0, -1.2, 9.0))
        cold.append((get_time() - t0) * 1000.0)
        _ = sc.frame(e_far, e_far + Vec3(-6.0, -1.2, 9.0))
    out.cold_wall_ms = _median(cold)
    var wall = List[Float64]()
    var cpu = List[Float64]()
    var gv = List[Float64]()
    for f in range(30):
        var tt = Float32(f) * Float32(0.31)
        var eye = Vec3(-8.0 + tt, 2.0, -10.0 + tt * 0.6)
        var t1 = get_time()
        _ = sc.frame(eye, eye + Vec3(6.0, -1.2, 9.0))
        wall.append((get_time() - t1) * 1000.0)
        cpu.append(sc.ren.last_cpu_vsm_ms)
        gv.append(_vsm_gpu_ms(sc))
        out.pages += sc.ren.last_vsm_dirty_pages
    # settle at the last camera, then read back
    var tl = Float32(29) * Float32(0.31)
    var eye_l = Vec3(-8.0 + tl, 2.0, -10.0 + tl * 0.6)
    for _ in range(6):
        _ = sc.frame(eye_l, eye_l + Vec3(6.0, -1.2, 9.0))
    out.px = sc.ren.readback_pixels()
    out.move_wall_ms = _median(wall)
    out.move_cpu_vsm_ms = _median(cpu)
    out.move_gpu_vsm_ms = _median(gv)
    out.static_gpu_vsm_ms = _median(st)
    if gpu:
        print("  GPU path: buffer items " + String(sc.ren.sota_shadow_gpu.item_cap) + ", draws/list "
              + String(sc.ren.sota_shadow_gpu.cmd_cap) + ", max wanted " + String(sc.ren.sota_shadow_gpu.wanted_max)
              + ", overflowed page draws " + String(sc.ren.sota_shadow_gpu.overflow_total) + ", grows "
              + String(sc.ren.sota_shadow_gpu.grows))
    sc.shutdown()
    return out^


def test_gpu_pages():
    """GPU page allocation + cull + one-pass raster against the CPU path: same image, cost on both sides."""
    var W = 480
    var H = 270
    var cpu = _pages_run(False, W, H)
    if len(cpu.px) == 0:
        print("[SKIP] Vulkan unavailable: GPU page test skipped")
        return
    var gpu = _pages_run(True, W, H)
    if not gpu.gpu_active:
        print("[SKIP] no shaderClipDistance / drawIndirectCount: GPU page path unavailable")
        return
    var d = compare_pixels(gpu.px, cpu.px, W, H, 6)
    print("  teleport frame wall (median of 3): CPU path " + _fmt(cpu.cold_wall_ms) + " ms, GPU path " + _fmt(gpu.cold_wall_ms) + " ms")
    print("  dolly frame wall (median): CPU path " + _fmt(cpu.move_wall_ms) + " ms (VSM CPU " + _fmt(cpu.move_cpu_vsm_ms)
          + " ms), GPU path " + _fmt(gpu.move_wall_ms) + " ms (VSM CPU " + _fmt(gpu.move_cpu_vsm_ms) + " ms)")
    print("  VSM GPU passes (median): dolly CPU path " + _fmt(cpu.move_gpu_vsm_ms) + " ms, GPU path " + _fmt(gpu.move_gpu_vsm_ms)
          + " ms; static CPU path " + _fmt(cpu.static_gpu_vsm_ms) + " ms, GPU path " + _fmt(gpu.static_gpu_vsm_ms) + " ms")
    print("  pages rendered during the dolly: CPU path " + String(cpu.pages) + ", GPU path " + String(gpu.pages)
          + "; final image GPU vs CPU path: " + d.summary())
    _ = write_png(_eval_dir() + "gpu_pages_gpu.png", gpu.px, W, H, stamped=False)
    _ = write_png(_eval_dir() + "gpu_pages_cpu.png", cpu.px, W, H, stamped=False)
    # The test scene requests ~1700 pages for 1024 physical tiles, so some pages stay unmapped on both paths; the CPU
    # path drops them in table order, the GPU path drops the coarsest cascades first. Hence a tolerance, not equality.
    _assert(d.over_tolerance <= W * H // 100 and d.flip_mean < 0.01, "GPU page path matches the CPU path: " + d.summary())
    _assert(gpu.pages > 0, "GPU page path rendered pages")
    print("[PASS] GPU-driven VSM pages match the CPU path")


def main():
    var only = getenv("SOTA_SHADOW_ONLY")  # e.g. "pages" to run one GPU test while iterating
    if only == "" or only == "toroidal":
        test_cpu_toroidal()
    if only == "" or only == "trunc":
        test_gpu_vsm_truncation()
    if only == "" or only == "toroidal":
        test_gpu_toroidal()
    if only == "" or only == "contact":
        test_gpu_contact_vs_rt()
    if only == "" or only == "pages":
        test_gpu_pages()
    print("[PASS] test_flint_v5_sota_shadow")
