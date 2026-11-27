"""Flint v5 Unified Render World, RH `-Z` / Reverse-Z Projection, GPU 3D DebugLineBatch & 3-Tier `ScalableRenderer` (`render_world.mojo`).

Architecture:
  1. Unified Right-Handed (`-Z` forward, `+X` right, `+Y` up) view & Vulkan Reverse-Z (`near -> 1.0`, `far/inf -> 0.0`)
     projection helpers (`make_rh_look_at`, `make_rh_reverse_z_infinite_perspective`, `make_rh_reverse_z_perspective`,
     `project_rh_world_to_screen`).
  2. `GpuDebugVertex` & `DebugLineBatch`: GPU-compatible 3D debug lines, AABBs, OBBs, spheres, capsules, 3-axis gizmos,
     and `DebugDraw` ingestion that analytically clips 3D segments against the camera near plane (`z_cam <= -near_z`),
     applies depth-cued / X-ray styling (`0xAARRGGBB`), and emits antialiased GPU SDF `PRIM_LINE` capsules into
     `UiDrawList` so 3D debug overlays render at 0 CPU framebuffer cost on zero-copy Wayland/X11/KMS swapchains!
  3. Double-buffered `RenderWorld` (`DoubleBufferedRenderWorld` / `RenderWorld[2]`) & `RenderExtractor`:
     decouples ECS gameplay simulation from GPU command recording, keying previous world transforms by `UInt64`
     entity ID (`Dict[UInt64, Int]`) so newly spawned entities never streak and despawned entities are pruned cleanly.
  4. 3-Tier `ScalableRenderer`:
     - `RENDER_TIER_0_HEADLESS_SIMD` (`0`): Pure-CPU 8-wide SIMD depth-buffered `SimdCanvas` (`<4 MiB`, zero GPU required).
     - `RENDER_TIER_1_FORWARD_CLUSTERED` (`1`): Fast Direct Clustered Lights + VSM Vulkan profile (zero RT/SVGF/DoF cost).
     - `RENDER_TIER_2_AAA_VISBUFFER_RT` (`2`): Flagship Meshlet VisBuffer + Hardware `SPV_KHR_ray_query` RT + SVGF +
       4-Cascade Toroidal VSM + FlintTSR + GPU SDF `UiSdfOverlayPass`.
"""
from std.collections import Dict, List
from std.math import abs, cos, isfinite, isnan, max, min, pi, sin, sqrt, tan
from std.memory import bitcast
from flint.assets.bc_compress import downsample_argb_2x
from flint.assets.meshlet import AAA_VERTEX_WORDS, AaaGeometryPool, CookedMeshEntry
from flint.assets.model import NativeMesh, NativeModel
from flint.core.math import Mat4, Vec3
from flint.core.transform import Quat
from flint.app.engine_app import Resource
from flint.ecs.world import comp_hash64
from flint.platform.wayland import WaylandWindow
from flint.render.aaa_clustered_lights import AaaClusterLight
from flint.render.aaa_gpu_particles import AaaParticleEmitter
from flint.render.cpu_particles import CpuParticles
from flint.render.aaa_renderer import (
    AaaSkinnedMeshEntry,
    AaaUploadedModel,
    FlintAaaVulkanRenderer,
    srgb_u8_to_linear,
)
from flint.render.aaa_skinning_rt import DualQuat
from flint.render.debug_draw import (
    DebugDraw,
    SHAPE_BOX,
    SHAPE_CAPSULE,
    SHAPE_LINE,
    SHAPE_SPHERE,
    perpendicular,
    side_of,
)
from flint.render.material_graph import (
    BakedVatCrowdClip,
    GpuDecalVolume,
    MAT_DOMAIN_CLEARCOAT,
    MAT_FLAG_ALPHA_MASKED,
    MAT_FLAG_CLEARCOAT,
    MAT_FLAG_DOUBLE_SIDED,
    MAT_FLAG_EMISSIVE_BLOOM,
    MAT_FLAG_IRIDESCENCE,
    MAT_FLAG_SHEEN,
    MaterialAsset,
    SurfaceMaterialNode,
    TerrainClipmapDesc,
    sample_vat_vertex_cpu,
)
from flint.render.font import SdfFont
from flint.render.simd_canvas import Camera3D, SimdCanvas, pack_argb
from flint.render.swapchain_surface import (
    DISPLAY_SDR_SRGB,
    PRESENT_MAILBOX,
    SwapchainSurface,
)
from flint.render.ui_cpu_raster import rasterize_ui_draw_list_cpu
from flint.render.ui_overlay import make_ui_line_prim
from flint.ui.draw_list import UiDrawList

# ============================================================================
# 1. Unified Right-Handed `-Z` Forward View & Reverse-Z Projection Math
# ============================================================================

comptime DEG_TO_RAD_F32: Float32 = 0.017453292519943295


def make_rh_look_at(
    eye: Vec3,
    target: Vec3,
    up_hint: Vec3 = Vec3(0.0, 1.0, 0.0),
) -> Mat4:
    """Builds a Right-Handed view matrix where camera forward is `-Z`, right is `+X`, and up is `+Y`.
    Any world point `P` in front of `eye` along `(target - eye)` transforms to `z_view = -depth < 0`."""
    var raw_fwd = target - eye
    var fwd = raw_fwd.normalized() if raw_fwd.length_squared() > Float32(1e-10) else Vec3(0.0, 0.0, -1.0)
    var up_n = up_hint.normalized() if up_hint.length_squared() > Float32(1e-10) else Vec3(0.0, 1.0, 0.0)
    if abs(fwd.dot(up_n)) > Float32(0.985):
        up_n = Vec3(0.0, 0.0, -1.0) if abs(fwd.z) < Float32(0.9) else Vec3(1.0, 0.0, 0.0)
    var right = fwd.cross(up_n).normalized()
    var up = right.cross(fwd).normalized()
    var back = -fwd  # +Z camera axis points opposite forward (`det([right, up, back]) == +1`)

    var m = Mat4()
    m.set(0, 0, right.x)
    m.set(0, 1, right.y)
    m.set(0, 2, right.z)
    m.set(0, 3, -right.dot(eye))

    m.set(1, 0, up.x)
    m.set(1, 1, up.y)
    m.set(1, 2, up.z)
    m.set(1, 3, -up.dot(eye))

    m.set(2, 0, back.x)
    m.set(2, 1, back.y)
    m.set(2, 2, back.z)
    m.set(2, 3, -back.dot(eye))

    m.set(3, 0, Float32(0.0))
    m.set(3, 1, Float32(0.0))
    m.set(3, 2, Float32(0.0))
    m.set(3, 3, Float32(1.0))
    return m^


def make_rh_reverse_z_infinite_perspective(
    fov_y_rad: Float32,
    aspect: Float32,
    near_z: Float32,
    flip_y: Bool = False,
) -> Mat4:
    """Right-Handed `-Z` forward Infinite Reverse-Z perspective matrix:
    - `z_view = -near_z` -> `z_ndc = 1.0`
    - `z_view -> -infinity` -> `z_ndc = 0.0`
    - `w_clip = -z_view > 0`."""
    var half_tan = max(Float32(1e-4), tan(fov_y_rad * Float32(0.5)))
    var f = Float32(1.0) / half_tan
    var safe_aspect = max(Float32(1e-4), aspect)
    var nz = max(Float32(1e-4), near_z)

    var m = Mat4()
    for i in range(16):
        m.values[i] = Float32(0.0)
    m.set(0, 0, f / safe_aspect)
    m.set(1, 1, -f if flip_y else f)
    m.set(2, 2, Float32(0.0))
    m.set(2, 3, nz)
    m.set(3, 2, Float32(-1.0))
    return m^


def make_rh_reverse_z_perspective(
    fov_y_rad: Float32,
    aspect: Float32,
    near_z: Float32,
    far_z: Float32,
    flip_y: Bool = False,
) -> Mat4:
    """Right-Handed `-Z` forward Finite Reverse-Z perspective matrix:
    - `z_view = -near_z` -> `z_ndc = 1.0`
    - `z_view = -far_z`  -> `z_ndc = 0.0`."""
    var half_tan = max(Float32(1e-4), tan(fov_y_rad * Float32(0.5)))
    var f = Float32(1.0) / half_tan
    var safe_aspect = max(Float32(1e-4), aspect)
    var nz = max(Float32(1e-4), near_z)
    var fz = max(nz + Float32(1e-2), far_z)
    var denom = fz - nz

    var m = Mat4()
    for i in range(16):
        m.values[i] = Float32(0.0)
    m.set(0, 0, f / safe_aspect)
    m.set(1, 1, -f if flip_y else f)
    m.set(2, 2, nz / denom)
    m.set(2, 3, (fz * nz) / denom)
    m.set(3, 2, Float32(-1.0))
    return m^


def project_rh_world_to_screen(
    vp: Mat4,
    world_pt: Vec3,
    screen_w: Int,
    screen_h: Int,
    y_flipped_in_proj: Bool = False,
) -> Tuple[Float32, Float32, Float32, Bool]:
    """Projects `world_pt` via `vp` (`proj * view`) to pixel `(sx, sy, z_ndc, in_front)`.
    Rejects points behind the near plane (`w_clip <= 0` or `z_ndc > 1.0`)."""
    var cx = vp.at(0, 0) * world_pt.x + vp.at(0, 1) * world_pt.y + vp.at(0, 2) * world_pt.z + vp.at(0, 3)
    var cy = vp.at(1, 0) * world_pt.x + vp.at(1, 1) * world_pt.y + vp.at(1, 2) * world_pt.z + vp.at(1, 3)
    var cz = vp.at(2, 0) * world_pt.x + vp.at(2, 1) * world_pt.y + vp.at(2, 2) * world_pt.z + vp.at(2, 3)
    var cw = vp.at(3, 0) * world_pt.x + vp.at(3, 1) * world_pt.y + vp.at(3, 2) * world_pt.z + vp.at(3, 3)
    if cw <= Float32(1e-5):
        return (Float32(0.0), Float32(0.0), Float32(0.0), False)
    var inv_w = Float32(1.0) / cw
    var ndc_x = cx * inv_w
    var ndc_y = cy * inv_w
    var ndc_z = cz * inv_w
    if ndc_z < Float32(-1e-4) or ndc_z > Float32(1.0001):
        return (Float32(0.0), Float32(0.0), ndc_z, False)
    var sx = (ndc_x * Float32(0.5) + Float32(0.5)) * Float32(screen_w)
    var sy = (
        (ndc_y * Float32(0.5) + Float32(0.5)) * Float32(screen_h)
        if y_flipped_in_proj
        else (Float32(0.5) - ndc_y * Float32(0.5)) * Float32(screen_h)
    )
    return (sx, sy, ndc_z, True)


# ============================================================================
# 2. `GpuDebugVertex` & GPU 3D `DebugLineBatch` -> `UiDrawList` (`PRIM_LINE`)
# ============================================================================

@always_inline
def _normalize_flint_rgba8(col: UInt32) -> UInt32:
    """Normalizes `0xRRGGBB` debug constants (alpha == 0) to opaque `0xFFRRGGBB` while preserving explicit alpha."""
    if (col & UInt32(0xFF000000)) == 0 and col != 0:
        return col | UInt32(0xFF000000)
    return col


@always_inline
def _scale_rgba8_alpha(col: UInt32, alpha_mul: Float32) -> UInt32:
    var base_a = Float32((col >> 24) & UInt32(0xFF))
    var safe_mul = Float32(0.0) if isnan(alpha_mul) else max(Float32(0.0), min(Float32(1.0), alpha_mul))
    var new_a = UInt32(Int(max(Float32(0.0), min(Float32(255.0), base_a * safe_mul + Float32(0.5)))))
    return (col & UInt32(0x00FFFFFF)) | (new_a << 24)


struct GpuDebugVertex(Copyable, ImplicitlyCopyable, Movable):
    """GPU-ready 3D debug vertex carrying world position, Flint `0xAARRGGBB` color, stroke width, and X-ray flag."""
    var pos: Vec3
    var rgba8: UInt32
    var thickness_px: Float32
    var xray: Bool

    def __init__(
        out self,
        pos: Vec3 = Vec3(),
        rgba8: UInt32 = 0xFFFFFFFF,
        thickness_px: Float32 = 1.75,
        xray: Bool = False,
    ):
        self.pos = pos
        self.rgba8 = _normalize_flint_rgba8(rgba8)
        self.thickness_px = max(Float32(0.75), thickness_px)
        self.xray = xray


struct DebugLineBatch(Copyable, Movable):
    """3D debug geometry batch projecting world-space lines, AABBs, OBBs, spheres, capsules, and gizmos
    into camera-near-clipped antialiased GPU SDF `PRIM_LINE` capsules on `UiDrawList`."""
    var vertices: List[GpuDebugVertex]
    var depth_fade_far_m: Float32
    var xray_alpha_scale: Float32

    def __init__(out self):
        self.vertices = List[GpuDebugVertex]()
        self.depth_fade_far_m = Float32(90.0)
        self.xray_alpha_scale = Float32(0.45)

    def clear(mut self):
        self.vertices.clear()

    @always_inline
    def segment_count(self) -> Int:
        return len(self.vertices) // 2

    def add_line(
        mut self,
        a: Vec3,
        b: Vec3,
        rgba8: UInt32 = 0xFFFFFFFF,
        thickness_px: Float32 = 1.75,
        xray: Bool = False,
    ):
        var col = _normalize_flint_rgba8(rgba8)
        var th = max(Float32(0.75), thickness_px)
        self.vertices.append(GpuDebugVertex(a, col, th, xray))
        self.vertices.append(GpuDebugVertex(b, col, th, xray))

    def add_aabb(
        mut self,
        bmin: Vec3,
        bmax: Vec3,
        rgba8: UInt32 = 0xFF40FF70,
        thickness_px: Float32 = 1.5,
        xray: Bool = False,
    ):
        var c000 = Vec3(bmin.x, bmin.y, bmin.z)
        var c100 = Vec3(bmax.x, bmin.y, bmin.z)
        var c010 = Vec3(bmin.x, bmax.y, bmin.z)
        var c110 = Vec3(bmax.x, bmax.y, bmin.z)
        var c001 = Vec3(bmin.x, bmin.y, bmax.z)
        var c101 = Vec3(bmax.x, bmin.y, bmax.z)
        var c011 = Vec3(bmin.x, bmax.y, bmax.z)
        var c111 = Vec3(bmax.x, bmax.y, bmax.z)
        # Bottom face
        self.add_line(c000, c100, rgba8, thickness_px, xray)
        self.add_line(c100, c101, rgba8, thickness_px, xray)
        self.add_line(c101, c001, rgba8, thickness_px, xray)
        self.add_line(c001, c000, rgba8, thickness_px, xray)
        # Top face
        self.add_line(c010, c110, rgba8, thickness_px, xray)
        self.add_line(c110, c111, rgba8, thickness_px, xray)
        self.add_line(c111, c011, rgba8, thickness_px, xray)
        self.add_line(c011, c010, rgba8, thickness_px, xray)
        # Vertical pillars
        self.add_line(c000, c010, rgba8, thickness_px, xray)
        self.add_line(c100, c110, rgba8, thickness_px, xray)
        self.add_line(c101, c111, rgba8, thickness_px, xray)
        self.add_line(c001, c011, rgba8, thickness_px, xray)

    def add_obb(
        mut self,
        center: Vec3,
        half_extents: Vec3,
        rotation: Quat = Quat(),
        rgba8: UInt32 = 0xFFFFE040,
        thickness_px: Float32 = 1.5,
        xray: Bool = False,
    ):
        var ax = rotation.rotate(Vec3(half_extents.x, 0.0, 0.0))
        var ay = rotation.rotate(Vec3(0.0, half_extents.y, 0.0))
        var az = rotation.rotate(Vec3(0.0, 0.0, half_extents.z))
        var c000 = center - ax - ay - az
        var c100 = center + ax - ay - az
        var c010 = center - ax + ay - az
        var c110 = center + ax + ay - az
        var c001 = center - ax - ay + az
        var c101 = center + ax - ay + az
        var c011 = center - ax + ay + az
        var c111 = center + ax + ay + az
        self.add_line(c000, c100, rgba8, thickness_px, xray)
        self.add_line(c100, c101, rgba8, thickness_px, xray)
        self.add_line(c101, c001, rgba8, thickness_px, xray)
        self.add_line(c001, c000, rgba8, thickness_px, xray)
        self.add_line(c010, c110, rgba8, thickness_px, xray)
        self.add_line(c110, c111, rgba8, thickness_px, xray)
        self.add_line(c111, c011, rgba8, thickness_px, xray)
        self.add_line(c011, c010, rgba8, thickness_px, xray)
        self.add_line(c000, c010, rgba8, thickness_px, xray)
        self.add_line(c100, c110, rgba8, thickness_px, xray)
        self.add_line(c101, c111, rgba8, thickness_px, xray)
        self.add_line(c001, c011, rgba8, thickness_px, xray)

    def add_sphere(
        mut self,
        center: Vec3,
        radius: Float32,
        rgba8: UInt32 = 0xFF50A0FF,
        segments: Int = 16,
        thickness_px: Float32 = 1.5,
        xray: Bool = False,
    ):
        var segs = max(6, segments)
        var r = max(Float32(0.001), radius)
        var step = Float32(2.0) * Float32(pi) / Float32(segs)
        for i in range(segs):
            var a0 = Float32(i) * step
            var a1 = Float32(i + 1) * step
            var c0 = cos(a0) * r
            var s0 = sin(a0) * r
            var c1 = cos(a1) * r
            var s1 = sin(a1) * r
            # XY, XZ, YZ rings
            self.add_line(center + Vec3(c0, s0, 0.0), center + Vec3(c1, s1, 0.0), rgba8, thickness_px, xray)
            self.add_line(center + Vec3(c0, 0.0, s0), center + Vec3(c1, 0.0, s1), rgba8, thickness_px, xray)
            self.add_line(center + Vec3(0.0, c0, s0), center + Vec3(0.0, c1, s1), rgba8, thickness_px, xray)

    def add_capsule(
        mut self,
        a: Vec3,
        b: Vec3,
        radius: Float32,
        rgba8: UInt32 = 0xFF40F0F0,
        segments: Int = 12,
        thickness_px: Float32 = 1.5,
        xray: Bool = False,
    ):
        var axis_raw = b - a
        var axis_len = axis_raw.length()
        var up_ax = axis_raw * (Float32(1.0) / axis_len) if axis_len > Float32(1e-5) else Vec3(0.0, 1.0, 0.0)
        var r_ax = perpendicular(up_ax)
        var f_ax = side_of(up_ax, r_ax).normalized()
        var r = max(Float32(0.001), radius)
        var segs = max(6, segments)
        var step = Float32(2.0) * Float32(pi) / Float32(segs)
        # 4 longitudinal rails
        self.add_line(a + r_ax * r, b + r_ax * r, rgba8, thickness_px, xray)
        self.add_line(a - r_ax * r, b - r_ax * r, rgba8, thickness_px, xray)
        self.add_line(a + f_ax * r, b + f_ax * r, rgba8, thickness_px, xray)
        self.add_line(a - f_ax * r, b - f_ax * r, rgba8, thickness_px, xray)
        # End-cap rings
        for i in range(segs):
            var a0 = Float32(i) * step
            var a1 = Float32(i + 1) * step
            var p0 = r_ax * (cos(a0) * r) + f_ax * (sin(a0) * r)
            var p1 = r_ax * (cos(a1) * r) + f_ax * (sin(a1) * r)
            self.add_line(a + p0, a + p1, rgba8, thickness_px, xray)
            self.add_line(b + p0, b + p1, rgba8, thickness_px, xray)

    def add_gizmo(
        mut self,
        origin: Vec3,
        rotation: Quat = Quat(),
        scale: Float32 = 1.0,
        thickness_px: Float32 = 2.5,
        xray: Bool = True,
    ):
        """Adds an RGB 3-axis transform gizmo (`+X` red, `+Y` green, `+Z` blue) with arrowheads."""
        var s = max(Float32(0.05), scale)
        var tip_len = s * Float32(0.18)
        var tip_rad = s * Float32(0.07)

        var x_dir = rotation.rotate(Vec3(1.0, 0.0, 0.0))
        var y_dir = rotation.rotate(Vec3(0.0, 1.0, 0.0))
        var z_dir = rotation.rotate(Vec3(0.0, 0.0, 1.0))

        var x_end = origin + x_dir * s
        var y_end = origin + y_dir * s
        var z_end = origin + z_dir * s

        var col_x: UInt32 = 0xFFFF3838
        var col_y: UInt32 = 0xFF38FF58
        var col_z: UInt32 = 0xFF4090FF

        self.add_line(origin, x_end, col_x, thickness_px, xray)
        self.add_line(x_end, x_end - x_dir * tip_len + y_dir * tip_rad, col_x, thickness_px, xray)
        self.add_line(x_end, x_end - x_dir * tip_len - y_dir * tip_rad, col_x, thickness_px, xray)

        self.add_line(origin, y_end, col_y, thickness_px, xray)
        self.add_line(y_end, y_end - y_dir * tip_len + x_dir * tip_rad, col_y, thickness_px, xray)
        self.add_line(y_end, y_end - y_dir * tip_len - x_dir * tip_rad, col_y, thickness_px, xray)

        self.add_line(origin, z_end, col_z, thickness_px, xray)
        self.add_line(z_end, z_end - z_dir * tip_len + y_dir * tip_rad, col_z, thickness_px, xray)
        self.add_line(z_end, z_end - z_dir * tip_len - y_dir * tip_rad, col_z, thickness_px, xray)

    def ingest_debug_draw(mut self, dd: DebugDraw, xray: Bool = False):
        """Translates CPU `DebugDraw` world-space shapes into GPU `DebugLineBatch` segments so existing
        gameplay `debug.line` / `debug.box` / `debug.sphere` / `debug.capsule` calls render on zero-copy swapchains."""
        if not dd.enabled or len(dd.shapes) == 0:
            return
        var th = max(Float32(1.25), Float32(dd.thickness))
        for i in range(len(dd.shapes)):
            ref sh = dd.shapes[i]
            var col = _normalize_flint_rgba8(sh.color)
            if sh.kind == SHAPE_LINE:
                self.add_line(sh.a, sh.b, col, th, xray)
            elif sh.kind == SHAPE_BOX:
                self.add_obb(sh.a, sh.b, sh.rotation, col, th, xray)
            elif sh.kind == SHAPE_SPHERE:
                self.add_sphere(sh.a, sh.radius, col, 16, th, xray)
            elif sh.kind == SHAPE_CAPSULE:
                self.add_capsule(sh.a, sh.b, sh.radius, col, 12, th, xray)

    def emit_to_ui_draw_list(
        self,
        mut ui: UiDrawList,
        eye: Vec3,
        target: Vec3,
        fov_y_deg: Float32,
        near_z: Float32,
        screen_w: Int,
        screen_h: Int,
        up_hint: Vec3 = Vec3(0.0, 1.0, 0.0),
        depth_cue: Bool = True,
    ) -> Int:
        """Transforms 3D segments into RH `-Z` camera space, analytically clips segments straddling `z_cam > -near_z`
        so lines extending behind the camera never invert or explode, applies depth/X-ray styling, and appends
        antialiased GPU SDF `PRIM_LINE` capsules to `ui`."""
        var n_segs = self.segment_count()
        if n_segs <= 0 or screen_w <= 0 or screen_h <= 0:
            return 0
        if not (
            isfinite(eye.x) and isfinite(eye.y) and isfinite(eye.z)
            and isfinite(target.x) and isfinite(target.y) and isfinite(target.z)
            and isfinite(up_hint.x) and isfinite(up_hint.y) and isfinite(up_hint.z)
            and isfinite(fov_y_deg) and isfinite(near_z)
        ):
            return 0

        var view = make_rh_look_at(eye, target, up_hint)
        for m_i in range(16):
            if not isfinite(view.values[m_i]):
                return 0
        var clip_near = max(Float32(0.02), near_z)
        var safe_fov = max(Float32(1.0), min(Float32(179.0), fov_y_deg))
        var half_tan = max(Float32(1e-4), tan(safe_fov * DEG_TO_RAD_F32 * Float32(0.5)))
        var focal_px = (Float32(screen_h) * Float32(0.5)) / half_tan
        var cx_px = Float32(screen_w) * Float32(0.5)
        var cy_px = Float32(screen_h) * Float32(0.5)
        var sw_f = Float32(screen_w)
        var sh_f = Float32(screen_h)
        var cur_clip = ui.current_clip()
        var emitted = 0

        for i in range(n_segs):
            var v0 = self.vertices[i * 2]
            var v1 = self.vertices[i * 2 + 1]
            if not (
                isfinite(v0.pos.x) and isfinite(v0.pos.y) and isfinite(v0.pos.z)
                and isfinite(v1.pos.x) and isfinite(v1.pos.y) and isfinite(v1.pos.z)
                and isfinite(v0.thickness_px) and isfinite(v1.thickness_px)
            ):
                continue

            # Transform endpoints to RH `-Z` camera space (`depth = -z_cam > 0` in front of eye)
            var ax = view.at(0, 0) * v0.pos.x + view.at(0, 1) * v0.pos.y + view.at(0, 2) * v0.pos.z + view.at(0, 3)
            var ay = view.at(1, 0) * v0.pos.x + view.at(1, 1) * v0.pos.y + view.at(1, 2) * v0.pos.z + view.at(1, 3)
            var az = view.at(2, 0) * v0.pos.x + view.at(2, 1) * v0.pos.y + view.at(2, 2) * v0.pos.z + view.at(2, 3)

            var bx = view.at(0, 0) * v1.pos.x + view.at(0, 1) * v1.pos.y + view.at(0, 2) * v1.pos.z + view.at(0, 3)
            var by = view.at(1, 0) * v1.pos.x + view.at(1, 1) * v1.pos.y + view.at(1, 2) * v1.pos.z + view.at(1, 3)
            var bz = view.at(2, 0) * v1.pos.x + view.at(2, 1) * v1.pos.y + view.at(2, 2) * v1.pos.z + view.at(2, 3)

            var da = -az
            var db = -bz
            if da < clip_near and db < clip_near:
                continue

            # Analytical camera near-plane clipping (`da >= clip_near` and `db >= clip_near`)
            if da < clip_near:
                var t = (clip_near - da) / max(Float32(1e-6), db - da)
                ax = ax + (bx - ax) * t
                ay = ay + (by - ay) * t
                da = clip_near
            elif db < clip_near:
                var t = (clip_near - db) / max(Float32(1e-6), da - db)
                bx = bx + (ax - bx) * t
                by = by + (ay - by) * t
                db = clip_near

            var inv_da = Float32(1.0) / da
            var inv_db = Float32(1.0) / db
            var sx0 = cx_px + ax * inv_da * focal_px
            var sy0 = cy_px - ay * inv_da * focal_px
            var sx1 = cx_px + bx * inv_db * focal_px
            var sy1 = cy_px - by * inv_db * focal_px

            var avg_d = Float32(0.5) * (da + db)
            var base_th = Float32(0.5) * (v0.thickness_px + v1.thickness_px)
            var is_xray = v0.xray or v1.xray

            var eff_alpha = Float32(1.0)
            var eff_th = base_th
            if is_xray:
                eff_alpha = self.xray_alpha_scale
                eff_th = max(Float32(1.25), base_th * Float32(0.9))
            elif depth_cue:
                var fade = Float32(1.0) / (Float32(1.0) + avg_d / max(Float32(5.0), self.depth_fade_far_m))
                eff_alpha = max(Float32(0.28), fade)
                eff_th = max(Float32(1.0), base_th * max(Float32(0.55), fade))

            var pad = eff_th + Float32(3.0)
            if not (
                isfinite(sx0) and isfinite(sy0) and isfinite(sx1) and isfinite(sy1)
                and isfinite(eff_th) and isfinite(eff_alpha)
            ):
                continue
            if (
                max(sx0, sx1) < -pad
                or min(sx0, sx1) > sw_f + pad
                or max(sy0, sy1) < -pad
                or min(sy0, sy1) > sh_f + pad
            ):
                continue

            var styled_col = _scale_rgba8_alpha(v0.rgba8, eff_alpha)
            ui.prims.append(
                make_ui_line_prim(
                    sx0,
                    sy0,
                    sx1,
                    sy1,
                    eff_th,
                    styled_col,
                    Float32(1.0),
                    cur_clip[0],
                    cur_clip[1],
                    cur_clip[2],
                    cur_clip[3],
                )
            )
            emitted += 1

        return emitted


# ============================================================================
# 3. Double-Buffered `RenderWorld` (`RenderWorld[2]`) & `RenderExtractor`
# ============================================================================

struct RenderCameraPacket(Copyable, ImplicitlyCopyable, Movable):
    """Extracted immutable camera state for a single render frame."""
    var eye: Vec3
    var target: Vec3
    var up: Vec3
    var fov_y_deg: Float32
    var near_z: Float32
    var far_z: Float32
    var exposure: Float32

    def __init__(
        out self,
        eye: Vec3 = Vec3(0.0, 2.0, 6.0),
        target: Vec3 = Vec3(0.0, 0.5, 0.0),
        up: Vec3 = Vec3(0.0, 1.0, 0.0),
        fov_y_deg: Float32 = 55.0,
        near_z: Float32 = 0.10,
        far_z: Float32 = 1000.0,
        exposure: Float32 = 1.0,
    ):
        self.eye = eye
        self.target = target
        self.up = up
        self.fov_y_deg = fov_y_deg
        self.near_z = near_z
        self.far_z = far_z
        self.exposure = exposure

    def view_matrix(self) -> Mat4:
        return make_rh_look_at(self.eye, self.target, self.up)

    def projection_matrix(self, aspect: Float32, infinite_far: Bool = True, flip_y: Bool = False) -> Mat4:
        var fov_rad = self.fov_y_deg * DEG_TO_RAD_F32
        if infinite_far:
            return make_rh_reverse_z_infinite_perspective(fov_rad, aspect, self.near_z, flip_y)
        return make_rh_reverse_z_perspective(fov_rad, aspect, self.near_z, self.far_z, flip_y)


struct RenderInstancePacket(Copyable, Movable):
    """Extracted renderable mesh instance keyed to a persistent `UInt64` ECS entity ID."""
    var entity_id: UInt64
    var mesh_id: Int
    var material_id: Int
    var world_mat: Mat4
    var prev_world_mat: Mat4
    var uniform_scale: Float32
    var has_prev: Bool
    var cast_shadows: Bool

    def __init__(
        out self,
        entity_id: UInt64,
        mesh_id: Int,
        world_mat: Mat4,
        prev_world_mat: Mat4,
        material_id: Int = -1,
        uniform_scale: Float32 = 1.0,
        has_prev: Bool = False,
        cast_shadows: Bool = True,
    ):
        self.entity_id = entity_id
        self.mesh_id = mesh_id
        self.material_id = material_id
        self.world_mat = world_mat.copy()
        self.prev_world_mat = prev_world_mat.copy()
        self.uniform_scale = uniform_scale
        self.has_prev = has_prev
        self.cast_shadows = cast_shadows


@fieldwise_init
struct VatCrowdInstancePacket(Copyable, Movable):
    """Per-instance VAT crowd packet stored on `RenderWorld` (pure `Copyable, Movable`)."""
    var entity_id: UInt64
    var clip_id: Int
    var phase_sec: Float32
    var prev_phase_sec: Float32
    var world_mat: Mat4
    var prev_world_mat: Mat4
    var material_id: Int
    var uniform_scale: Float32
    var has_prev: Bool


@fieldwise_init
struct TerrainClipmapPacket(Copyable, Movable):
    """Terrain clipmap descriptor packet stored on `RenderWorld` (`Copyable, Movable`)."""
    var active: Bool
    var desc: TerrainClipmapDesc
    var material_id: Int

    def __init__(out self):
        self.active = False
        self.desc = TerrainClipmapDesc()
        self.material_id = -1


struct RenderWorld(Resource, Copyable, Movable):
    """Self-contained extracted snapshot of one simulation frame ready for GPU/SIMD submission."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.render.RenderWorld")
    var frame_number: UInt64
    var delta_time: Float32
    var camera: RenderCameraPacket
    var sun_dir: Vec3
    var sun_radiance: Vec3
    var ibl_intensity: Float32
    var instances: List[RenderInstancePacket]
    var vat_instances: List[VatCrowdInstancePacket]
    var terrain_clipmap: TerrainClipmapPacket
    var cluster_lights: List[AaaClusterLight]
    var particle_emitters: List[AaaParticleEmitter]
    var decals: List[GpuDecalVolume]
    var debug_lines: DebugLineBatch
    var overlay_lines: DebugLineBatch

    def __init__(out self):
        self.frame_number = 0
        self.delta_time = Float32(1.0 / 60.0)
        self.camera = RenderCameraPacket()
        self.sun_dir = Vec3(0.38, 0.80, -0.46).normalized()
        self.sun_radiance = Vec3(3.2, 3.0, 2.75)
        self.ibl_intensity = Float32(1.0)
        self.instances = List[RenderInstancePacket]()
        self.vat_instances = List[VatCrowdInstancePacket]()
        self.terrain_clipmap = TerrainClipmapPacket()
        self.cluster_lights = List[AaaClusterLight]()
        self.particle_emitters = List[AaaParticleEmitter]()
        self.decals = List[GpuDecalVolume]()
        self.debug_lines = DebugLineBatch()
        self.overlay_lines = DebugLineBatch()

    def clear_frame(
        mut self,
        frame_number: UInt64 = 0,
        delta_time: Float32 = 1.0 / 60.0,
        clear_overlay: Bool = False,
    ):
        self.frame_number = frame_number
        self.delta_time = delta_time
        self.instances.clear()
        self.vat_instances.clear()
        self.cluster_lights.clear()
        self.particle_emitters.clear()
        self.decals.clear()
        self.debug_lines.clear()
        if clear_overlay:
            self.overlay_lines.clear()

    @always_inline
    def instance_count(self) -> Int:
        return len(self.instances)


struct DoubleBufferedRenderWorld(Copyable, Movable):
    """Ping-pong `RenderWorld[2]` ring decoupling ECS extraction (`write_world`) from rendering (`read_world`)."""
    var slots: List[RenderWorld]
    var write_index: Int
    var read_index: Int

    def __init__(out self):
        self.slots = List[RenderWorld](capacity=2)
        self.slots.append(RenderWorld())
        self.slots.append(RenderWorld())
        self.write_index = 0
        self.read_index = 1

    @always_inline
    def __getitem__(ref self, idx: Int) -> ref [self] RenderWorld:
        return Pointer[RenderWorld, origin_of(self)](unsafe_from_address=Int(Pointer(to=self.slots[idx & 1])))[]

    @always_inline
    def write_world(mut self) -> ref [self] RenderWorld:
        return Pointer[RenderWorld, origin_of(self)](unsafe_from_address=Int(Pointer(to=self.slots[self.write_index])))[]

    @always_inline
    def read_world(self) -> ref [self] RenderWorld:
        return Pointer[RenderWorld, origin_of(self)](unsafe_from_address=Int(Pointer(to=self.slots[self.read_index])))[]

    def swap_buffers(mut self):
        var tmp = self.write_index
        self.write_index = self.read_index
        self.read_index = tmp


@always_inline
def _extract_max_scale(m: Mat4) -> Float32:
    var s0 = m.values[0] * m.values[0] + m.values[1] * m.values[1] + m.values[2] * m.values[2]
    var s1 = m.values[4] * m.values[4] + m.values[5] * m.values[5] + m.values[6] * m.values[6]
    var s2 = m.values[8] * m.values[8] + m.values[9] * m.values[9] + m.values[10] * m.values[10]
    return sqrt(max(Float32(1e-8), max(s0, max(s1, s2))))


struct RenderExtractor(Copyable, Movable):
    """Stateful frame-to-frame ECS -> `RenderWorld` extractor keying `prev_world_mat` by `UInt64` entity ID."""
    var id_to_slot: Dict[UInt64, Int]
    var slot_ids: List[UInt64]
    var slot_mats: List[Mat4]
    var slot_last_frame: List[UInt64]
    var active_frame: UInt64

    def __init__(out self):
        self.id_to_slot = Dict[UInt64, Int]()
        self.slot_ids = List[UInt64]()
        self.slot_mats = List[Mat4]()
        self.slot_last_frame = List[UInt64]()
        self.active_frame = 1

    @always_inline
    def tracked_entity_count(self) -> Int:
        return len(self.slot_ids)

    def begin_extraction(mut self, mut dst: RenderWorld, frame_number: UInt64, delta_time: Float32 = 1.0 / 60.0):
        self.active_frame = frame_number + 1
        dst.clear_frame(frame_number, delta_time)

    def extract_entity(
        mut self,
        mut dst: RenderWorld,
        entity_id: UInt64,
        mesh_id: Int,
        world_mat: Mat4,
        material_id: Int = -1,
        cast_shadows: Bool = True,
    ):
        """Extracts `entity_id` with exact `prev_world_mat` lookup:
        - Existing entity: `prev_world_mat` is last frame's `world_mat`, `has_prev = True`.
        - Newly spawned entity: `prev_world_mat = world_mat`, `has_prev = False` (zero TSR/motion-blur streaking!)."""
        var u_scale = _extract_max_scale(world_mat)
        var found = self.id_to_slot.get(entity_id)
        if found:
            var slot = found.value()
            var prev_m = self.slot_mats[slot].copy()
            self.slot_mats[slot] = world_mat.copy()
            self.slot_last_frame[slot] = self.active_frame
            dst.instances.append(
                RenderInstancePacket(
                    entity_id,
                    mesh_id,
                    world_mat,
                    prev_m,
                    material_id,
                    u_scale,
                    True,
                    cast_shadows,
                )
            )
        else:
            var new_slot = len(self.slot_ids)
            self.id_to_slot[entity_id] = new_slot
            self.slot_ids.append(entity_id)
            self.slot_mats.append(world_mat.copy())
            self.slot_last_frame.append(self.active_frame)
            dst.instances.append(
                RenderInstancePacket(
                    entity_id,
                    mesh_id,
                    world_mat,
                    world_mat,
                    material_id,
                    u_scale,
                    False,
                    cast_shadows,
                )
            )

    def extract_vat_crowd_entity(
        mut self,
        mut world: RenderWorld,
        entity_id: UInt64,
        clip_id: Int,
        phase_sec: Float32,
        prev_phase_sec: Float32,
        world_mat: Mat4,
        material_id: Int = -1,
    ):
        """Extracts a VAT crowd instance into `world.vat_instances`, resolving `prev_world_mat` by `entity_id`."""
        var u_scale = _extract_max_scale(world_mat)
        var found = self.id_to_slot.get(entity_id)
        if found:
            var slot = found.value()
            var prev_m = self.slot_mats[slot].copy()
            self.slot_mats[slot] = world_mat.copy()
            self.slot_last_frame[slot] = self.active_frame
            world.vat_instances.append(
                VatCrowdInstancePacket(
                    entity_id,
                    clip_id,
                    phase_sec,
                    prev_phase_sec,
                    world_mat.copy(),
                    prev_m^,
                    material_id,
                    u_scale,
                    True,
                )
            )
        else:
            var new_slot = len(self.slot_ids)
            self.id_to_slot[entity_id] = new_slot
            self.slot_ids.append(entity_id)
            self.slot_mats.append(world_mat.copy())
            self.slot_last_frame.append(self.active_frame)
            world.vat_instances.append(
                VatCrowdInstancePacket(
                    entity_id,
                    clip_id,
                    phase_sec,
                    prev_phase_sec,
                    world_mat.copy(),
                    world_mat.copy(),
                    material_id,
                    u_scale,
                    False,
                )
            )

    def end_extraction(mut self):
        """Prunes despawned entity IDs that were not extracted in `active_frame`."""
        var stale = 0
        for i in range(len(self.slot_last_frame)):
            if self.slot_last_frame[i] != self.active_frame:
                stale += 1
        if stale == 0:
            return

        var next_map = Dict[UInt64, Int]()
        var next_ids = List[UInt64](capacity=len(self.slot_ids) - stale)
        var next_mats = List[Mat4](capacity=len(self.slot_ids) - stale)
        var next_frames = List[UInt64](capacity=len(self.slot_ids) - stale)
        for i in range(len(self.slot_ids)):
            if self.slot_last_frame[i] == self.active_frame:
                var new_idx = len(next_ids)
                var eid = self.slot_ids[i]
                next_map[eid] = new_idx
                next_ids.append(eid)
                next_mats.append(self.slot_mats[i].copy())
                next_frames.append(self.slot_last_frame[i])
        self.id_to_slot = next_map^
        self.slot_ids = next_ids^
        self.slot_mats = next_mats^
        self.slot_last_frame = next_frames^


# ============================================================================
# 4. 3-Tier `ScalableRenderer` (`TIER_0_HEADLESS_SIMD`, `TIER_1`, `TIER_2`)
# ============================================================================

comptime RENDER_TIER_0_HEADLESS_SIMD: UInt32 = 0
comptime RENDER_TIER_1_FORWARD_CLUSTERED: UInt32 = 1
comptime RENDER_TIER_2_AAA_VISBUFFER_RT: UInt32 = 2


def material_asset_from_native_mesh(m: NativeMesh) -> MaterialAsset:
    """Constructs a `MaterialAsset` matching `NativeMesh` GLB PBR/emissive/clearcoat/sheen properties."""
    var bc = m.base_color
    var lin_bc = Vec3(
        srgb_u8_to_linear(Int((bc >> 16) & 0xFF)),
        srgb_u8_to_linear(Int((bc >> 8) & 0xFF)),
        srgb_u8_to_linear(Int(bc & 0xFF)),
    )
    var ma = MaterialAsset(base_color=lin_bc, roughness=m.roughness, metallic=m.metallic)
    if m.emissive_strength > Float32(1e-4):
        ma.emissive = m.emissive_color * m.emissive_strength
        ma.flags |= MAT_FLAG_EMISSIVE_BLOOM
    ma.clearcoat = m.clearcoat
    ma.clearcoat_roughness = m.clearcoat_roughness
    if m.clearcoat > Float32(0.001):
        ma.flags |= MAT_FLAG_CLEARCOAT
        ma.domain = MAT_DOMAIN_CLEARCOAT
    ma.specular_factor = m.specular_factor
    ma.iridescence = m.iridescence
    ma.iridescence_thickness_nm = m.iridescence_thickness_nm
    ma.iridescence_ior = m.iridescence_ior
    if m.iridescence > Float32(0.001):
        ma.flags |= MAT_FLAG_IRIDESCENCE
    ma.sheen_color = m.sheen_color
    ma.sheen_roughness = m.sheen_roughness
    if m.sheen_color.x + m.sheen_color.y + m.sheen_color.z > Float32(0.001):
        ma.flags |= MAT_FLAG_SHEEN
    if m.alpha_mode == 1:
        ma.opacity = Float32((bc >> 24) & 255) / 255
        ma.alpha_cutoff = m.alpha_cutoff
        ma.flags |= MAT_FLAG_ALPHA_MASKED
    if m.double_sided:
        ma.flags |= MAT_FLAG_DOUBLE_SIDED
    return ma^


struct ScalableRenderer(Resource, Movable):
    """Unified 3-Tier Flint v5 Renderer scaling seamlessly from `<4 MiB` CPU SIMD CI (`Tier 0`)
    to Clustered Forward/VisBuffer+VSM (`Tier 1`) to Flagship Hardware RT+SVGF+TSR (`Tier 2`)."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.render.ScalableRenderer")
    var tier: UInt32
    var requested_tier: UInt32
    var width: Int
    var height: Int
    var surface: SwapchainSurface
    var cpu_canvas: SimdCanvas
    var cpu_particles: CpuParticles
    var cpu_meshes: List[NativeMesh]
    var cpu_mesh_materials: List[Int]
    var free_cpu_skin_mesh_ids: List[Int]
    var cpu_skin_bind_positions: Dict[Int, List[Float32]]
    var cpu_vat_clips: List[BakedVatCrowdClip]
    var cpu_vat_meshes: List[NativeMesh]
    var cpu_vat_materials: List[Int]
    var gpu_terrain_synced: Bool
    var gpu_terrain_desc: TerrainClipmapDesc
    var gpu_terrain_mat_id: Int
    var cpu_terrain_heights: List[Float32]
    var cpu_terrain_hm_w: Int
    var cpu_terrain_hm_h: Int
    var cpu_ui_fonts: List[SdfFont]
    var ui_time_sec: Float32
    var ui_dpi_scale: Float32
    var ui_global_flags: UInt32
    var materials: List[MaterialAsset]
    var gpu_slot: List[FlintAaaVulkanRenderer]

    def __init__(
        out self,
        tier: UInt32 = RENDER_TIER_2_AAA_VISBUFFER_RT,
        width: Int = 640,
        height: Int = 360,
        tsr_scale: Float32 = 1.0,
    ):
        self.requested_tier = tier
        self.tier = tier
        self.width = max(64, width)
        self.height = max(64, height)
        self.surface = SwapchainSurface()
        _ = self.surface.initialize_headless(self.width, self.height)
        self.cpu_canvas = SimdCanvas(self.width, self.height)
        self.cpu_particles = CpuParticles()
        self.cpu_meshes = List[NativeMesh]()
        self.cpu_mesh_materials = List[Int]()
        self.free_cpu_skin_mesh_ids = List[Int]()
        self.cpu_skin_bind_positions = Dict[Int, List[Float32]]()
        self.cpu_vat_clips = List[BakedVatCrowdClip]()
        self.cpu_vat_meshes = List[NativeMesh]()
        self.cpu_vat_materials = List[Int]()
        self.gpu_terrain_synced = False
        self.gpu_terrain_desc = TerrainClipmapDesc()
        self.gpu_terrain_mat_id = -1
        self.cpu_terrain_heights = List[Float32]()
        self.cpu_terrain_hm_w = 0
        self.cpu_terrain_hm_h = 0
        self.cpu_ui_fonts = List[SdfFont]()
        self.ui_time_sec = 0.0
        self.ui_dpi_scale = 1.0
        self.ui_global_flags = 0
        self.materials = List[MaterialAsset]()
        self.gpu_slot = List[FlintAaaVulkanRenderer]()

        if tier != RENDER_TIER_0_HEADLESS_SIMD:
            var gpu = FlintAaaVulkanRenderer(
                width=self.width,
                height=self.height,
                tsr_scale=(Float32(1.0) if tier == RENDER_TIER_1_FORWARD_CLUSTERED else tsr_scale),
                max_draws=(16384 if tier == RENDER_TIER_1_FORWARD_CLUSTERED else 65536),
                max_instances=(4096 if tier == RENDER_TIER_1_FORWARD_CLUSTERED else 16384),
                env_cube_dim=32,
            )
            if gpu.active:
                if tier == RENDER_TIER_1_FORWARD_CLUSTERED:
                    gpu.enable_rt_shadows = False
                    gpu.enable_rt_reflections = False
                    gpu.enable_rt_gi = False
                    gpu.svgf_passes = 0
                    gpu.enable_spec_reblur = False
                    gpu.enable_froxel_fog = False
                    gpu.enable_bokeh_dof = False
                    gpu.enable_motion_blur = False
                    gpu.enable_vsm = True
                    gpu.tsr_history_weight = Float32(0.0)
                else:
                    gpu.enable_vsm = True
                    gpu.enable_vsm_clipmap_cascades = True
                    gpu.enable_rt_shadows = gpu.vk.has_ray_query
                    gpu.enable_rt_reflections = gpu.vk.has_ray_query
                    gpu.enable_rt_gi = gpu.vk.has_ray_query
                self.gpu_slot.append(gpu^)
            else:
                gpu.shutdown()
                self.tier = RENDER_TIER_0_HEADLESS_SIMD

    def tier_name(self) -> String:
        if self.tier == RENDER_TIER_2_AAA_VISBUFFER_RT:
            return "Tier2-AAA-VisBuffer-RT"
        if self.tier == RENDER_TIER_1_FORWARD_CLUSTERED:
            return "Tier1-Forward-Clustered"
        return "Tier0-Headless-SIMD"

    @always_inline
    def uses_vulkan(self) -> Bool:
        return self.tier != RENDER_TIER_0_HEADLESS_SIMD and len(self.gpu_slot) > 0 and self.gpu_slot[0].active

    def register_ui_font(mut self, font: SdfFont) -> Int:
        """Registers an `SdfFont` atlas on both Vulkan bindless storage and Tier-0 CPU parity slots."""
        if self.uses_vulkan():
            var slot = self.gpu_slot[0].upload_ui_font_atlas(font)
            if slot >= 0:
                while len(self.cpu_ui_fonts) <= slot:
                    self.cpu_ui_fonts.append(SdfFont())
                self.cpu_ui_fonts[slot] = font.copy()
            return slot
        var slot = len(self.cpu_ui_fonts)
        self.cpu_ui_fonts.append(font.copy())
        return slot

    def attach_window_surface(
        mut self,
        mut window: WaylandWindow,
        present_mode: UInt32 = PRESENT_MAILBOX,
        display_mode: UInt32 = DISPLAY_SDR_SRGB,
    ) -> Bool:
        if not self.uses_vulkan():
            return self.surface.initialize_headless(self.width, self.height, display_mode, present_mode)
        var ok = self.surface.initialize(window, self.gpu_slot[0].vk, self.width, self.height, present_mode, display_mode)
        self.gpu_slot[0].display_mode = self.surface.display_mode
        return ok

    def register_material(mut self, mat: MaterialAsset) -> Int:
        if self.uses_vulkan():
            var gpu_id = self.gpu_slot[0].add_material_asset(mat)
            if gpu_id < 0:
                return -1
            while len(self.materials) < gpu_id:
                self.materials.append(MaterialAsset())
            self.materials.append(mat.copy())
            return gpu_id
        var id = len(self.materials)
        self.materials.append(mat.copy())
        return id

    def update_material_asset_deferred(mut self, mat_id: Int, mat: MaterialAsset):
        """Updates CPU `materials[mat_id]` immediately and queues a non-stalling GPU SSBO upload."""
        if mat_id < 0 or mat_id >= len(self.materials):
            return
        self.materials[mat_id] = mat.copy()
        if self.uses_vulkan():
            self.gpu_slot[0].queue_material_asset_update(mat_id, mat)

    def evaluate_material_graph[
        NodeT: SurfaceMaterialNode
    ](
        mut self,
        node: NodeT,
        mat_id: Int,
        dim: Int = 32,
        time_sec: Float32 = 0.0,
        world_y_scale: Float32 = 24.0,
    ) -> Tuple[Int, Int]:
        """Dispatches `node` (`SurfaceMaterialNode`) on `FlintAaaVulkanRenderer` against unified `24`-word `buf_materials`
        and binds the GPU-evaluated surface textures onto `mat_id`."""
        if not self.uses_vulkan() or mat_id < 0 or mat_id >= len(self.materials):
            return (-1, -1)
        var slots = self.gpu_slot[0].evaluate_material_graph_to_bindless(
            node, mat_id, dim=dim, time_sec=time_sec, world_y_scale=world_y_scale
        )
        if slots[0] >= 0 and slots[1] >= 0:
            self.materials[mat_id].set_albedo_slot(slots[0])
            self.materials[mat_id].set_orm_slot(slots[1])
        return slots

    def register_mesh(mut self, mesh: NativeMesh, mat_id: Int, build_lods: Bool = False) -> Int:
        if self.uses_vulkan():
            var entry = self.gpu_slot[0].add_mesh(mesh, mat_id, build_lods=build_lods)
            while len(self.cpu_meshes) < entry.mesh_id:
                self.cpu_meshes.append(NativeMesh())
                self.cpu_mesh_materials.append(0)
            self.cpu_meshes.append(mesh.copy())
            self.cpu_mesh_materials.append(mat_id)
            return entry.mesh_id
        var id = len(self.cpu_meshes)
        self.cpu_meshes.append(mesh.copy())
        self.cpu_mesh_materials.append(mat_id)
        return id

    def register_cooked_mesh_asset(mut self, geom: AaaGeometryPool, mat_id: Int = -1) -> Int:
        """Uploads a cooked `AaaGeometryPool` (reconstructed `NativeMesh`) into Tier 0/1/2."""
        if len(geom.meshes) == 0:
            return -1
        ref gm = geom.meshes[0]
        var vc = gm.vertex_count
        if vc <= 0 or (gm.first_vertex + vc) * AAA_VERTEX_WORDS > len(geom.vertex_words):
            return -1
        var nm = NativeMesh()
        nm.positions.reserve(vc * 3)
        nm.normals.reserve(vc * 3)
        nm.texcoords.reserve(vc * 2)
        for vi in range(vc):
            var base_w = (gm.first_vertex + vi) * AAA_VERTEX_WORDS
            nm.positions.append(bitcast[DType.float32](geom.vertex_words[base_w + 0]))
            nm.positions.append(bitcast[DType.float32](geom.vertex_words[base_w + 1]))
            nm.positions.append(bitcast[DType.float32](geom.vertex_words[base_w + 2]))
            nm.normals.append(bitcast[DType.float32](geom.vertex_words[base_w + 3]))
            nm.normals.append(bitcast[DType.float32](geom.vertex_words[base_w + 4]))
            nm.normals.append(bitcast[DType.float32](geom.vertex_words[base_w + 5]))
            nm.texcoords.append(bitcast[DType.float32](geom.vertex_words[base_w + 9]))
            nm.texcoords.append(bitcast[DType.float32](geom.vertex_words[base_w + 10]))
        var n_idx = gm.triangle_count * 3
        if gm.first_index + n_idx <= len(geom.index_words):
            nm.indices.reserve(n_idx)
            for ii in range(n_idx):
                var g_v = Int(geom.index_words[gm.first_index + ii]) - gm.first_vertex
                if g_v >= 0 and g_v < vc:
                    nm.indices.append(g_v)
        var eff_mat = mat_id
        if eff_mat < 0:
            eff_mat = self.register_material(material_asset_from_native_mesh(NativeMesh()))
        return self.register_mesh(nm, eff_mat, build_lods=False)

    def build_native_mesh_material(
        mut self,
        m: NativeMesh,
        model: NativeModel = NativeModel(),
        max_tex_dim: Int = 2048,
    ) -> MaterialAsset:
        """Builds a `MaterialAsset` from `NativeMesh`, uploading embedded GLB textures on Vulkan tiers."""
        var ma = material_asset_from_native_mesh(m)
        if not self.uses_vulkan():
            return ma^
        var n_tex = len(model.textures)
        if n_tex <= 0:
            return ma^
        var cap_dim = max(16, max_tex_dim)
        if m.texture >= 0 and m.texture < n_tex:
            ref tx = model.textures[m.texture]
            if tx.width > 0 and tx.height > 0 and len(tx.pixels) >= tx.width * tx.height:
                var cur_px = tx.pixels.copy()
                var cw = tx.width
                var ch = tx.height
                while (cw > cap_dim or ch > cap_dim) and cw >= 8 and ch >= 8:
                    var ds = downsample_argb_2x(cur_px, cw, ch)
                    cur_px = ds[0].copy()
                    cw = ds[1]
                    ch = ds[2]
                var alb_slot = self.gpu_slot[0].upload_bc7_texture(cur_px, cw, ch, srgb=True, generate_mips=True)
                if alb_slot >= 0:
                    ma.set_albedo_slot(alb_slot)
        if m.normal_orm_texture >= 0 and m.normal_orm_texture < n_tex:
            ref ntx = model.textures[m.normal_orm_texture]
            if ntx.width > 0 and ntx.height > 0 and len(ntx.pixels) >= ntx.width * ntx.height:
                var cur_np = ntx.pixels.copy()
                var nw = ntx.width
                var nh = ntx.height
                while (nw > cap_dim or nh > cap_dim) and nw >= 8 and nh >= 8:
                    var ds = downsample_argb_2x(cur_np, nw, nh)
                    cur_np = ds[0].copy()
                    nw = ds[1]
                    nh = ds[2]
                var n_vecs = List[Vec3](capacity=nw * nh)
                for p in cur_np:
                    var nx = (Float32((p >> 16) & 0xFF) / Float32(255.0)) * Float32(2.0) - Float32(1.0)
                    var ny = (Float32((p >> 8) & 0xFF) / Float32(255.0)) * Float32(2.0) - Float32(1.0)
                    var nz = (Float32(p & 0xFF) / Float32(255.0)) * Float32(2.0) - Float32(1.0)
                    n_vecs.append(Vec3(nx, ny, max(Float32(0.05), nz)).normalized())
                var nrm_slot = self.gpu_slot[0].upload_bc5_normal_texture(n_vecs, nw, nh, generate_mips=True)
                if nrm_slot >= 0:
                    ma.set_normal_slot(nrm_slot)
        if m.orm_texture >= 0 and m.orm_texture < n_tex:
            ref otx = model.textures[m.orm_texture]
            if otx.width > 0 and otx.height > 0 and len(otx.pixels) >= otx.width * otx.height:
                var cur_op = otx.pixels.copy()
                var ow = otx.width
                var oh = otx.height
                while (ow > cap_dim or oh > cap_dim) and ow >= 8 and oh >= 8:
                    var ds = downsample_argb_2x(cur_op, ow, oh)
                    cur_op = ds[0].copy()
                    ow = ds[1]
                    oh = ds[2]
                var has_sep_occ = (
                    m.occlusion_texture >= 0
                    and m.occlusion_texture < n_tex
                    and m.occlusion_texture != m.orm_texture
                    and model.textures[m.occlusion_texture].width > 0
                    and model.textures[m.occlusion_texture].height > 0
                    and len(model.textures[m.occlusion_texture].pixels)
                    >= model.textures[m.occlusion_texture].width * model.textures[m.occlusion_texture].height
                )
                var orm_px = List[UInt32](capacity=ow * oh)
                if has_sep_occ:
                    ref octx = model.textures[m.occlusion_texture]
                    for py in range(oh):
                        var sy = min(octx.height - 1, (py * octx.height) // oh)
                        for px in range(ow):
                            var p = cur_op[py * ow + px]
                            var sx = min(octx.width - 1, (px * octx.width) // ow)
                            var r_ao = Float32((octx.pixels[sy * octx.width + sx] >> 16) & 0xFF)
                            var scaled_ao = UInt32(
                                max(
                                    0,
                                    min(
                                        255,
                                        Int(
                                            Float32(255.0) * (Float32(1.0) - m.occlusion_strength)
                                            + r_ao * m.occlusion_strength
                                            + Float32(0.5)
                                        ),
                                    ),
                                )
                            )
                            orm_px.append(UInt32(0xFF000000) | (scaled_ao << 16) | (p & 0x0000FFFF))
                else:
                    for p in cur_op:
                        var r_ao = (p >> 16) & 0xFF
                        var use_ao: UInt32
                        if m.occlusion_texture == m.orm_texture:
                            use_ao = UInt32(
                                max(
                                    0,
                                    min(
                                        255,
                                        Int(
                                            Float32(255.0) * (Float32(1.0) - m.occlusion_strength)
                                            + Float32(r_ao) * m.occlusion_strength
                                            + Float32(0.5)
                                        ),
                                    ),
                                )
                            )
                        else:
                            use_ao = UInt32(255) if r_ao == 0 else r_ao
                        orm_px.append(UInt32(0xFF000000) | (use_ao << 16) | (p & 0x0000FFFF))
                var orm_slot = self.gpu_slot[0].upload_bc7_texture(orm_px, ow, oh, srgb=False, generate_mips=True)
                if orm_slot >= 0:
                    ma.set_orm_slot(orm_slot)
        elif m.occlusion_texture >= 0 and m.occlusion_texture < n_tex:
            ref octx = model.textures[m.occlusion_texture]
            if octx.width > 0 and octx.height > 0 and len(octx.pixels) >= octx.width * octx.height:
                var cur_op = octx.pixels.copy()
                var ow = octx.width
                var oh = octx.height
                while (ow > cap_dim or oh > cap_dim) and ow >= 8 and oh >= 8:
                    var ds = downsample_argb_2x(cur_op, ow, oh)
                    cur_op = ds[0].copy()
                    ow = ds[1]
                    oh = ds[2]
                var rough_u8 = UInt32(max(12, min(255, Int(m.roughness * Float32(255.0) + Float32(0.5)))))
                var metal_u8 = UInt32(max(0, min(255, Int(m.metallic * Float32(255.0) + Float32(0.5)))))
                var orm_px = List[UInt32](capacity=ow * oh)
                for p in cur_op:
                    var r_ao = Float32((p >> 16) & 0xFF)
                    var scaled_ao = UInt32(
                        max(
                            0,
                            min(
                                255,
                                Int(
                                    Float32(255.0) * (Float32(1.0) - m.occlusion_strength)
                                    + r_ao * m.occlusion_strength
                                    + Float32(0.5)
                                ),
                            ),
                        )
                    )
                    orm_px.append(UInt32(0xFF000000) | (scaled_ao << 16) | (rough_u8 << 8) | metal_u8)
                var orm_slot = self.gpu_slot[0].upload_bc7_texture(orm_px, ow, oh, srgb=False, generate_mips=True)
                if orm_slot >= 0:
                    ma.set_orm_slot(orm_slot)
        if m.emissive_texture >= 0 and m.emissive_texture < n_tex:
            ref etx = model.textures[m.emissive_texture]
            if etx.width > 0 and etx.height > 0 and len(etx.pixels) >= etx.width * etx.height:
                var cur_ep = etx.pixels.copy()
                var ew = etx.width
                var eh = etx.height
                while (ew > cap_dim or eh > cap_dim) and ew >= 8 and eh >= 8:
                    var ds = downsample_argb_2x(cur_ep, ew, eh)
                    cur_ep = ds[0].copy()
                    ew = ds[1]
                    eh = ds[2]
                var emi_slot = self.gpu_slot[0].upload_bc7_texture(cur_ep, ew, eh, srgb=True, generate_mips=True)
                if emi_slot >= 0:
                    ma.set_emissive_slot(emi_slot)
        return ma^

    def upload_native_model(
        mut self,
        model: NativeModel,
        build_lods: Bool = True,
        max_tex_dim: Int = 2048,
        upload_skinned: Bool = False,
    ) -> AaaUploadedModel:
        """Uploads a `NativeModel` across Tier 0 CPU and Tier 1/2 Vulkan, keeping `materials` & `cpu_meshes` aligned."""
        if len(model.meshes) == 0:
            return AaaUploadedModel()
        if self.uses_vulkan():
            var out = self.gpu_slot[0].upload_native_model(
                model, build_lods=build_lods, max_tex_dim=max_tex_dim, upload_skinned=upload_skinned
            )
            for mi in range(len(out.material_ids)):
                var mat_id = out.material_ids[mi]
                if mat_id >= 0:
                    while len(self.materials) <= mat_id:
                        self.materials.append(MaterialAsset())
                    if mi < len(model.meshes):
                        var ma = material_asset_from_native_mesh(model.meshes[mi])
                        if mat_id < len(self.gpu_slot[0].materials):
                            ref gm = self.gpu_slot[0].materials[mat_id]
                            ma.set_albedo_slot(gm.albedo_tex)
                            ma.set_normal_slot(gm.normal_tex)
                            ma.set_orm_slot(gm.orm_tex)
                        self.materials[mat_id] = ma^
            for mi in range(len(out.mesh_entries)):
                var mesh_id = out.mesh_entries[mi].mesh_id
                var mat_id = out.material_ids[mi] if mi < len(out.material_ids) else -1
                if mesh_id >= 0:
                    while len(self.cpu_meshes) <= mesh_id:
                        self.cpu_meshes.append(NativeMesh())
                        self.cpu_mesh_materials.append(0)
                    if mi < len(model.meshes):
                        self.cpu_meshes[mesh_id] = model.meshes[mi].copy()
                        self.cpu_mesh_materials[mesh_id] = mat_id
            for si in range(len(out.skinned_entries)):
                var sm_id = out.skinned_entries[si].mesh_entry.mesh_id
                if sm_id >= 0 and sm_id < len(self.cpu_meshes):
                    self.cpu_skin_bind_positions[sm_id] = self.cpu_meshes[sm_id].positions.copy()
            return out^

        var out = AaaUploadedModel()
        for mi in range(len(model.meshes)):
            ref m = model.meshes[mi]
            var ma = material_asset_from_native_mesh(m)
            var mat_id = self.register_material(ma)
            out.material_ids.append(mat_id)

            var mesh_id = self.register_mesh(m, mat_id, build_lods=build_lods)
            var entry = CookedMeshEntry()
            entry.mesh_id = mesh_id
            entry.material_id = mat_id
            entry.vertex_count = m.vertex_count()
            entry.triangle_count = m.triangle_count()
            entry.bounds_center = m.bounds_center
            entry.bounds_radius = m.bounds_radius
            out.mesh_entries.append(entry.copy())

            var mv_cnt = m.vertex_count()
            if upload_skinned and mv_cnt > 0 and len(m.joints) >= mv_cnt * 4 and len(m.weights) >= mv_cnt * 4:
                var b_cnt = 64
                if m.skin >= 0 and m.skin < len(model.skins) and len(model.skins[m.skin].joints) > 0:
                    b_cnt = len(model.skins[m.skin].joints)
                self.cpu_skin_bind_positions[mesh_id] = m.positions.copy()
                var sk_entry = AaaSkinnedMeshEntry()
                sk_entry.mesh_entry = entry^
                sk_entry.vertex_count = mv_cnt
                sk_entry.bone_count = b_cnt
                out.skinned_entries.append(sk_entry^)
        return out^

    def add_skinned_mesh(
        mut self, mesh: NativeMesh, mat_id: Int, bone_count: Int = 64
    ) -> AaaSkinnedMeshEntry:
        """Allocates an independent skinned mesh entry (and bone palette) across Tier 0/1/2."""
        if self.uses_vulkan():
            var empty_jp = List[UInt32]()
            var empty_jw = List[Vec3]()
            var sk = self.gpu_slot[0].add_skinned_mesh(mesh, empty_jp, empty_jw, mat_id, bone_count=bone_count)
            var mid = sk.mesh_entry.mesh_id
            if mid >= 0:
                while len(self.cpu_meshes) <= mid:
                    self.cpu_meshes.append(NativeMesh())
                    self.cpu_mesh_materials.append(0)
                self.cpu_meshes[mid] = mesh.copy()
                self.cpu_mesh_materials[mid] = mat_id
                self.cpu_skin_bind_positions[mid] = mesh.positions.copy()
            return sk^

        var mid: Int
        if len(self.free_cpu_skin_mesh_ids) > 0:
            mid = self.free_cpu_skin_mesh_ids.pop()
            self.cpu_meshes[mid] = mesh.copy()
            self.cpu_mesh_materials[mid] = mat_id
        else:
            mid = self.register_mesh(mesh, mat_id, build_lods=False)
        self.cpu_skin_bind_positions[mid] = mesh.positions.copy()
        var entry = CookedMeshEntry()
        entry.mesh_id = mid
        entry.material_id = mat_id
        entry.vertex_count = mesh.vertex_count()
        entry.triangle_count = mesh.triangle_count()
        entry.bounds_center = mesh.bounds_center
        entry.bounds_radius = mesh.bounds_radius
        var sk_entry = AaaSkinnedMeshEntry()
        sk_entry.mesh_entry = entry^
        sk_entry.vertex_count = mesh.vertex_count()
        sk_entry.bone_count = bone_count
        return sk_entry^

    def remove_skinned_mesh(mut self, entry: AaaSkinnedMeshEntry):
        var mid = entry.mesh_entry.mesh_id
        if mid < 0 or mid not in self.cpu_skin_bind_positions:
            return
        if self.uses_vulkan():
            _ = self.gpu_slot[0].remove_skinned_mesh(entry)
        else:
            self.free_cpu_skin_mesh_ids.append(mid)
        _ = self.cpu_skin_bind_positions.pop(mid, List[Float32]())
        self.cpu_meshes[mid] = NativeMesh()
        self.cpu_mesh_materials[mid] = -1

    def update_skinned_mesh_bones(
        mut self,
        skin_entry: AaaSkinnedMeshEntry,
        cur_bones: List[DualQuat],
        prev_bones: List[DualQuat],
        defer_to_frame: Bool = True,
        morph_weights: List[Float32] = List[Float32](),
        prev_morph_weights: List[Float32] = List[Float32](),
    ):
        """Updates DualQuat bone palettes on GPU (Tier 1/2) and CPU (Tier 0)."""
        if self.uses_vulkan():
            if len(morph_weights) > 0:
                self.gpu_slot[0].update_skinned_mesh_morphs(skin_entry, morph_weights, prev_morph_weights)
            _ = self.gpu_slot[0].update_skinned_mesh_bones(
                skin_entry, cur_bones, prev_bones, invalidate_vsm=False, defer_to_frame=defer_to_frame
            )
            return
        var mid = skin_entry.mesh_entry.mesh_id
        if mid < 0 or mid >= len(self.cpu_meshes) or len(cur_bones) == 0:
            return
        var bind_opt = self.cpu_skin_bind_positions.get(mid)
        if not bind_opt:
            return
        ref bind_pos = bind_opt.value()
        ref m = self.cpu_meshes[mid]
        var vc = len(bind_pos) // 3
        if len(m.joints) < vc * 4 or len(m.weights) < vc * 4 or len(m.positions) < vc * 3:
            return
        var nb = len(cur_bones)
        for vi in range(vc):
            var p = Vec3(bind_pos[vi * 3], bind_pos[vi * 3 + 1], bind_pos[vi * 3 + 2])
            var targets = min(len(morph_weights), min(m.morph_count, len(m.morph_positions) // (vc * 3)))
            for j in range(targets):
                var offset = (j * vc + vi) * 3
                p = p + Vec3(m.morph_positions[offset], m.morph_positions[offset+1], m.morph_positions[offset+2]) * morph_weights[j]
            var out_p = Vec3()
            var w_sum = Float32(0.0)
            for k in range(4):
                var w = m.weights[vi * 4 + k]
                var ji = Int(m.joints[vi * 4 + k])
                if w > Float32(0.0) and ji >= 0 and ji < nb:
                    out_p = out_p + cur_bones[ji].transform_point(p) * w
                    w_sum += w
            if w_sum > Float32(1.0e-5):
                var inv_w = Float32(1.0) / w_sum
                m.positions[vi * 3 + 0] = out_p.x * inv_w
                m.positions[vi * 3 + 1] = out_p.y * inv_w
                m.positions[vi * 3 + 2] = out_p.z * inv_w

    def register_vat_crowd_clip(
        mut self,
        clip: BakedVatCrowdClip,
        template_mesh: NativeMesh,
        mat_id: Int = 0,
        max_phase_slots: Int = 8,
    ) -> Int:
        var id = len(self.cpu_vat_clips)
        if self.uses_vulkan():
            var gpu_id = self.gpu_slot[0].register_vat_crowd_clip(clip, template_mesh, mat_id, max_phase_slots=max_phase_slots)
            if gpu_id < 0:
                return -1
            while len(self.cpu_vat_clips) < gpu_id:
                self.cpu_vat_clips.append(BakedVatCrowdClip(clip.desc))
                self.cpu_vat_meshes.append(NativeMesh())
                self.cpu_vat_materials.append(0)
            id = gpu_id
        self.cpu_vat_clips.append(clip.copy())
        self.cpu_vat_meshes.append(template_mesh.copy())
        self.cpu_vat_materials.append(mat_id)
        return id

    def set_terrain_clipmap(
        mut self,
        desc: TerrainClipmapDesc,
        material_id: Int = -1,
        heights: List[Float32] = List[Float32](),
        height_w: Int = 0,
        height_h: Int = 0,
    ) -> TerrainClipmapDesc:
        """Uploads optional `heights` heightmap (`RGBA16F`) and activates `TerrainClipmapDesc` on GPU/Tier-0."""
        var final_desc = desc.copy()
        if len(heights) > 0 and height_w > 0 and height_h > 0:
            self.cpu_terrain_heights = heights.copy()
            self.cpu_terrain_hm_w = height_w
            self.cpu_terrain_hm_h = height_h
            if self.uses_vulkan():
                var slot = self.gpu_slot[0].upload_terrain_heightmap_rgba16f(heights, height_w, height_h)
                if slot >= 0:
                    final_desc.heightmap_slot = UInt16(slot)
        self.gpu_terrain_desc = final_desc.copy()
        self.gpu_terrain_mat_id = material_id
        if self.uses_vulkan():
            self.gpu_slot[0].set_terrain_clipmap(final_desc, material_id)
            self.gpu_terrain_synced = True
        return final_desc.copy()

    def clear_terrain_clipmap(mut self):
        self.gpu_terrain_synced = False
        if self.uses_vulkan():
            self.gpu_slot[0].clear_terrain_clipmap()

    def _material_color_argb(self, mat_id: Int) -> UInt32:
        if mat_id >= 0 and mat_id < len(self.materials):
            var bc = self.materials[mat_id].base_color
            var cx = Float32(0.0) if isnan(bc.x) else max(Float32(0.0), min(Float32(1.0), bc.x))
            var cy = Float32(0.0) if isnan(bc.y) else max(Float32(0.0), min(Float32(1.0), bc.y))
            var cz = Float32(0.0) if isnan(bc.z) else max(Float32(0.0), min(Float32(1.0), bc.z))
            var r = Int(sqrt(cx) * Float32(255.0) + Float32(0.5))
            var g = Int(sqrt(cy) * Float32(255.0) + Float32(0.5))
            var b = Int(sqrt(cz) * Float32(255.0) + Float32(0.5))
            return pack_argb(r, g, b, 255)
        return 0xFFCCCCCC

    @staticmethod
    @always_inline
    def _modulate_env_argb(col: UInt32, sun_radiance: Vec3, exposure: Float32) -> UInt32:
        var exp_s = max(Float32(0.02), min(Float32(4.0), exposure)) if isfinite(exposure) else Float32(1.0)
        var rx = max(Float32(0.02), min(Float32(4.0), sun_radiance.x * Float32(1.0 / 3.2) * exp_s)) if isfinite(sun_radiance.x) else exp_s
        var ry = max(Float32(0.02), min(Float32(4.0), sun_radiance.y * Float32(1.0 / 3.0) * exp_s)) if isfinite(sun_radiance.y) else exp_s
        var rz = max(Float32(0.02), min(Float32(4.0), sun_radiance.z * Float32(1.0 / 2.75) * exp_s)) if isfinite(sun_radiance.z) else exp_s
        if rx == Float32(1.0) and ry == Float32(1.0) and rz == Float32(1.0):
            return col
        var r = min(255, max(0, Int(Float32((col >> 16) & 0xFF) * rx)))
        var g = min(255, max(0, Int(Float32((col >> 8) & 0xFF) * ry)))
        var b = min(255, max(0, Int(Float32(col & 0xFF) * rz)))
        return pack_argb(r, g, b, Int((col >> 24) & 0xFF))

    def _render_tier0_simd(mut self, world: RenderWorld, ui: UiDrawList) -> Bool:
        self.cpu_canvas.clear(0x101828, 0x1E2D46)
        var cam = Camera3D(
            screen_w=self.width,
            screen_h=self.height,
            eye=world.camera.eye,
            target=world.camera.target,
            fov_deg=world.camera.fov_y_deg,
        )
        cam.near_z = max(Float32(0.05), world.camera.near_z)
        cam.up = world.camera.up
        cam.update_basis()

        for i in range(len(world.instances)):
            ref inst = world.instances[i]
            if inst.mesh_id < 0 or inst.mesh_id >= len(self.cpu_meshes):
                continue
            ref m = self.cpu_meshes[inst.mesh_id]
            var mat_idx = inst.material_id if inst.material_id >= 0 else self.cpu_mesh_materials[inst.mesh_id]
            var col = Self._modulate_env_argb(self._material_color_argb(mat_idx), world.sun_radiance, world.camera.exposure)
            var n_tris = len(m.indices) // 3
            var n_verts = len(m.positions) // 3
            for t in range(n_tris):
                var i0 = m.indices[t * 3 + 0]
                var i1 = m.indices[t * 3 + 1]
                var i2 = m.indices[t * 3 + 2]
                if i0 < 0 or i1 < 0 or i2 < 0 or i0 >= n_verts or i1 >= n_verts or i2 >= n_verts:
                    continue
                var p0 = inst.world_mat.transform_point(
                    Vec3(m.positions[i0 * 3 + 0], m.positions[i0 * 3 + 1], m.positions[i0 * 3 + 2])
                )
                var p1 = inst.world_mat.transform_point(
                    Vec3(m.positions[i1 * 3 + 0], m.positions[i1 * 3 + 1], m.positions[i1 * 3 + 2])
                )
                var p2 = inst.world_mat.transform_point(
                    Vec3(m.positions[i2 * 3 + 0], m.positions[i2 * 3 + 1], m.positions[i2 * 3 + 2])
                )
                self.cpu_canvas.draw_tri_3d_lit(cam, p0, p1, p2, col, world.sun_dir)

        for v_i in range(len(world.vat_instances)):
            ref vi = world.vat_instances[v_i]
            if vi.clip_id < 0 or vi.clip_id >= len(self.cpu_vat_clips):
                continue
            ref clip = self.cpu_vat_clips[vi.clip_id]
            ref vm = self.cpu_vat_meshes[vi.clip_id]
            var mat_idx = vi.material_id if vi.material_id >= 0 else self.cpu_vat_materials[vi.clip_id]
            var col = Self._modulate_env_argb(self._material_color_argb(mat_idx), world.sun_radiance, world.camera.exposure)
            var n_tris = len(vm.indices) // 3
            var n_verts = vm.vertex_count()
            for t in range(n_tris):
                var i0 = vm.indices[t * 3 + 0]
                var i1 = vm.indices[t * 3 + 1]
                var i2 = vm.indices[t * 3 + 2]
                if i0 < 0 or i1 < 0 or i2 < 0 or i0 >= n_verts or i1 >= n_verts or i2 >= n_verts:
                    continue
                var lp0 = sample_vat_vertex_cpu(clip, i0, vi.phase_sec, vi.prev_phase_sec)[0]
                var lp1 = sample_vat_vertex_cpu(clip, i1, vi.phase_sec, vi.prev_phase_sec)[0]
                var lp2 = sample_vat_vertex_cpu(clip, i2, vi.phase_sec, vi.prev_phase_sec)[0]
                var p0 = vi.world_mat.transform_point(lp0)
                var p1 = vi.world_mat.transform_point(lp1)
                var p2 = vi.world_mat.transform_point(lp2)
                self.cpu_canvas.draw_tri_3d_lit(cam, p0, p1, p2, col, world.sun_dir)

        if world.terrain_clipmap.active:
            var td = world.terrain_clipmap.desc.copy()
            var col = Self._modulate_env_argb(
                self._material_color_argb(world.terrain_clipmap.material_id),
                world.sun_radiance,
                world.camera.exposure,
            )
            var ox = td.snapped_ring_origin_xz(world.camera.eye, 0)[0]
            var oz = td.snapped_ring_origin_xz(world.camera.eye, 0)[1]
            var step = td.ring_cell_size(0) * Float32(4.0)
            var hm_w = self.cpu_terrain_hm_w
            var hm_h = self.cpu_terrain_hm_h
            for gz in range(-4, 4):
                for gx in range(-4, 4):
                    var x0 = ox + Float32(gx) * step
                    var x1 = x0 + step
                    var z0 = oz + Float32(gz) * step
                    var z1 = z0 + step
                    var y00 = td.height_offset_m + Float32(0.5) * td.height_scale_m
                    if hm_w > 0 and hm_h > 0 and len(self.cpu_terrain_heights) >= hm_w * hm_h and isfinite(x0) and isfinite(z0):
                        var u0 = max(Float32(0.0), min(Float32(1.0), (x0 + Float32(32.0)) / Float32(64.0)))
                        var v0 = max(Float32(0.0), min(Float32(1.0), (z0 + Float32(32.0)) / Float32(64.0)))
                        if isfinite(u0) and isfinite(v0):
                            var hi = min(hm_h - 1, Int(v0 * Float32(hm_h - 1))) * hm_w + min(hm_w - 1, Int(u0 * Float32(hm_w - 1)))
                            y00 = td.height_offset_m + self.cpu_terrain_heights[hi] * td.height_scale_m
                    var p00 = Vec3(x0, y00, z0)
                    var p10 = Vec3(x1, y00, z0)
                    var p01 = Vec3(x0, y00, z1)
                    var p11 = Vec3(x1, y00, z1)
                    self.cpu_canvas.draw_tri_3d_lit(cam, p00, p01, p11, col, world.sun_dir)
                    self.cpu_canvas.draw_tri_3d_lit(cam, p00, p11, p10, col, world.sun_dir)

        var particle_light = max(Float32(0.1), world.ibl_intensity*0.4 + max(Float32(0), world.sun_dir.y)*0.6)
        self.cpu_particles.render(self.cpu_canvas, cam, world.particle_emitters, world.delta_time, particle_light)

        var n_dbg = world.debug_lines.segment_count()
        for s_i in range(n_dbg):
            var v0 = world.debug_lines.vertices[s_i * 2]
            var v1 = world.debug_lines.vertices[s_i * 2 + 1]
            var thick = Int(min(Float32(64.0), max(Float32(1.0), v0.thickness_px + Float32(0.5)))) if isfinite(v0.thickness_px) else 1
            self.cpu_canvas.draw_line_3d(cam, v0.pos, v1.pos, v0.rgba8, thick)

        var n_ov = world.overlay_lines.segment_count()
        for s_i in range(n_ov):
            var v0 = world.overlay_lines.vertices[s_i * 2]
            var v1 = world.overlay_lines.vertices[s_i * 2 + 1]
            var thick = Int(min(Float32(64.0), max(Float32(1.0), v0.thickness_px + Float32(0.5)))) if isfinite(v0.thickness_px) else 1
            self.cpu_canvas.draw_line_3d(cam, v0.pos, v1.pos, v0.rgba8, thick)

        rasterize_ui_draw_list_cpu(
            self.cpu_canvas,
            ui,
            self.cpu_ui_fonts,
            self.ui_time_sec,
            self.ui_global_flags,
        )

        return True

    def resize(mut self, width: Int, height: Int) -> Bool:
        """Resizes the active tier (`Tier 0` CPU canvas or `Tier 1/2` Vulkan `SwapchainSurface` + render targets)."""
        var nw = max(16, width)
        var nh = max(16, height)
        if not self.uses_vulkan():
            self.width = nw
            self.height = nh
            self.cpu_canvas = SimdCanvas(nw, nh)
            _ = self.surface.initialize_headless(nw, nh, self.surface.display_mode, self.surface.present_mode)
            self.surface.resize_generations += 1
            return True

        var ok_sc = self.surface.recreate(self.gpu_slot[0].vk, nw, nh)
        var target_w = self.surface.width if self.surface.width > 0 else nw
        var target_h = self.surface.height if self.surface.height > 0 else nh
        var ok_ren = self.gpu_slot[0].resize(target_w, target_h)
        self.width = self.gpu_slot[0].output_w
        self.height = self.gpu_slot[0].output_h
        self.gpu_slot[0].display_mode = self.surface.display_mode
        self.gpu_slot[0].swapchain_out_of_date = False
        return ok_sc and ok_ren

    def render_world(
        mut self,
        world: RenderWorld,
        mut ui: UiDrawList,
        capture_readback: Bool = False,
        force_readback: Bool = False,
    ) -> Bool:
        """Renders `world` + 3D `DebugLineBatch` + `ui` through the active tier (`Tier 0`, `Tier 1`, or `Tier 2`)."""
        if not self.uses_vulkan():
            return self._render_tier0_simd(world, ui)

        if (
            self.gpu_slot[0].swapchain_out_of_date
            or (
                self.surface.is_zero_copy_gpu()
                and (self.gpu_slot[0].output_w != self.width or self.gpu_slot[0].output_h != self.height)
            )
        ):
            _ = self.resize(self.width, self.height)

        ref gpu = self.gpu_slot[0]
        if world.terrain_clipmap.active:
            var td = world.terrain_clipmap.desc.copy()
            var needs_geom_sync = (
                not self.gpu_terrain_synced
                or td.ring_count != self.gpu_terrain_desc.ring_count
                or td.grid_dim_per_ring != self.gpu_terrain_desc.grid_dim_per_ring
                or abs(td.base_cell_size_m - self.gpu_terrain_desc.base_cell_size_m) > Float32(1e-5)
                or abs(td.height_scale_m - self.gpu_terrain_desc.height_scale_m) > Float32(1e-5)
                or abs(td.height_offset_m - self.gpu_terrain_desc.height_offset_m) > Float32(1e-5)
                or world.terrain_clipmap.material_id != self.gpu_terrain_mat_id
            )
            var needs_slot_sync = (
                abs(td.morph_start_ratio - self.gpu_terrain_desc.morph_start_ratio) > Float32(1e-5)
                or td.heightmap_slot != self.gpu_terrain_desc.heightmap_slot
                or td.normalmap_slot != self.gpu_terrain_desc.normalmap_slot
                or td.splat_control_slot != self.gpu_terrain_desc.splat_control_slot
            )
            if not needs_slot_sync:
                if len(td.layer_albedo_slots) != len(self.gpu_terrain_desc.layer_albedo_slots):
                    needs_slot_sync = True
                else:
                    for k_s in range(len(td.layer_albedo_slots)):
                        if td.layer_albedo_slots[k_s] != self.gpu_terrain_desc.layer_albedo_slots[k_s]:
                            needs_slot_sync = True
                            break
                if not needs_slot_sync:
                    if len(td.layer_normal_slots) != len(self.gpu_terrain_desc.layer_normal_slots):
                        needs_slot_sync = True
                    else:
                        for k_s in range(len(td.layer_normal_slots)):
                            if td.layer_normal_slots[k_s] != self.gpu_terrain_desc.layer_normal_slots[k_s]:
                                needs_slot_sync = True
                                break
            if needs_geom_sync:
                gpu.set_terrain_clipmap(td, world.terrain_clipmap.material_id)
                self.gpu_terrain_desc = td.copy()
                self.gpu_terrain_mat_id = world.terrain_clipmap.material_id
                self.gpu_terrain_synced = True
            elif needs_slot_sync:
                gpu.terrain_clipmap_desc = td.copy()
                gpu.terrain_clipmap_dirty = True
                self.gpu_terrain_desc = td.copy()
        elif self.gpu_terrain_synced:
            gpu.clear_terrain_clipmap()
            self.gpu_terrain_synced = False

        gpu.sun_dir = world.sun_dir.normalized()
        gpu.sun_radiance = world.sun_radiance
        gpu.ibl_intensity = world.ibl_intensity
        gpu.exposure = world.camera.exposure
        gpu.ui_time_sec = self.ui_time_sec
        gpu.ui_dpi_scale = self.ui_dpi_scale
        gpu.ui_global_flags = self.ui_global_flags

        gpu.begin_frame(
            cam_pos=world.camera.eye,
            cam_target=world.camera.target,
            fov_y_deg=world.camera.fov_y_deg,
            near_z=world.camera.near_z,
        )
        for i in range(len(world.instances)):
            ref inst = world.instances[i]
            gpu.draw_instance(
                inst.mesh_id,
                inst.world_mat,
                inst.prev_world_mat,
                inst.material_id,
                inst.uniform_scale,
                inst.has_prev,
            )
        for v_i in range(len(world.vat_instances)):
            ref vi = world.vat_instances[v_i]
            gpu.draw_vat_crowd_instance(
                vi.clip_id,
                vi.phase_sec,
                vi.prev_phase_sec,
                vi.world_mat,
                vi.prev_world_mat,
                vi.material_id,
                vi.uniform_scale,
                vi.has_prev,
            )
        gpu.particle_dt_sec = world.delta_time
        for emitter in world.particle_emitters:
            gpu.submit_particle_emitter(emitter)
        for l_i in range(len(world.cluster_lights)):
            gpu.add_cluster_light(world.cluster_lights[l_i])
        for d_i in range(len(world.decals)):
            gpu.add_decal(world.decals[d_i])

        # Copy caller UI primitives + project 3D DebugLineBatch into GPU SDF `PRIM_LINE` segments
        for p_i in range(ui.count()):
            gpu.ui_draw_list.prims.append(ui.prims[p_i])
        _ = world.debug_lines.emit_to_ui_draw_list(
            gpu.ui_draw_list,
            world.camera.eye,
            world.camera.target,
            world.camera.fov_y_deg,
            world.camera.near_z,
            self.width,
            self.height,
            world.camera.up,
        )
        _ = world.overlay_lines.emit_to_ui_draw_list(
            gpu.ui_draw_list,
            world.camera.eye,
            world.camera.target,
            world.camera.fov_y_deg,
            world.camera.near_z,
            self.width,
            self.height,
            world.camera.up,
        )
        var ok = gpu.render_frame(capture_readback=capture_readback, force_readback=force_readback)
        if self.gpu_slot[0].swapchain_out_of_date:
            var rw = self.gpu_slot[0].vk.swapchain_width if self.gpu_slot[0].vk.swapchain_width > 0 else self.width
            var rh = self.gpu_slot[0].vk.swapchain_height if self.gpu_slot[0].vk.swapchain_height > 0 else self.height
            _ = self.resize(rw, rh)
        return ok

    def readback_pixels(self) -> List[UInt32]:
        if self.uses_vulkan():
            return self.gpu_slot[0].readback_pixels()
        return self.cpu_canvas.pixels.copy()

    def shutdown(mut self):
        if len(self.gpu_slot) > 0:
            self.surface.shutdown_vk(self.gpu_slot[0].vk)
            self.gpu_slot[0].shutdown()
            self.gpu_slot.clear()

    def __deinit__(deinit self):
        self.shutdown()
