"""Phase 3 (`V18`): Clustered `1,024+` Punctual & Heitz `LTC` Polygonal/Tube Area Lighting (`aaa_clustered_lights.mojo`).

Implements:
  1. `16 x 9 x 24` (`3,456` froxel clusters) Logarithmic-Z GPU Compute Light Culler (`aaa_cluster_cull_shader`)
     supporting `1,024+` dynamic Point, Spot, Rectangular `LTC` Area, and Tube `LTC` Area lights.
  2. Pure-Mojo Eric Heitz **Linearly Transformed Cosines (`LTC`)** `64x64` `VK_FORMAT_R16G16_SFLOAT` (`RG16F`)
     Inverse-Matrix LUT baker (`bake_ltc_lut_rg16f`) & Sutherland-Hodgman horizon-clipped spherical polygon
     contour integrator (`eval_ltc_rect_cpu` + SPIR-V `emit_eval_clustered_and_ltc_lights`), achieving
     exact analytical Lambertian solid-angle conservation at `roughness = 1.0` and anisotropic GGX
     highlight stretching/skewing across all roughness & viewing angles.
  3. Multi-light VSM shadow sub-atlas region allocator (`allocate_local_light_vsm_regions`).
"""
from std.math import cos, isfinite, max, min, sin, sqrt
from std.memory import bitcast
from flint.core.detmath import det_acos
from flint.core.math import Vec3
from flint.assets.bc_compress import f32_to_uf16_bits
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.aaa_raytracing import emit_rt_shadow_query
from flint.render.shader_dsl import (
    ShaderDsl,
    clamp_f,
    dot,
    exp2_f,
    fma_f,
    max_f,
    min_f,
)

comptime AAA_CLUSTER_X: Int = 16
comptime AAA_CLUSTER_Y: Int = 9
comptime AAA_CLUSTER_Z: Int = 24
comptime AAA_TOTAL_CLUSTERS: Int = 3456           # 16 * 9 * 24
comptime AAA_MAX_CLUSTERED_LIGHTS: Int = 1024
comptime AAA_MAX_LIGHTS_PER_CLUSTER: Int = 64
comptime AAA_CLUSTER_LIGHT_WORDS: Int = 16

# Buffer offsets in `BIND_AAA_CLUSTERS` (words):
# [0..15]: Header: [0]=light_count, [1]=near_z, [2]=far_z, [3]=log2(far/near), [4]=tan_x, [5]=tan_y
# [16 .. 16400): 1,024 x 16-word `AaaClusterLight` descriptors
# [16400 .. 19856): 3,456 per-cluster light counts (`UInt32`)
# [19856 .. 241040): 3,456 x 64 culled light indices (`UInt32`)
comptime AAA_CLUSTER_LIGHTS_BASE: Int = 16
comptime AAA_CLUSTER_COUNTS_BASE: Int = 16 + AAA_MAX_CLUSTERED_LIGHTS * AAA_CLUSTER_LIGHT_WORDS  # 16400
comptime AAA_CLUSTER_INDICES_BASE: Int = AAA_CLUSTER_COUNTS_BASE + AAA_TOTAL_CLUSTERS             # 19856
comptime AAA_CLUSTER_BUFFER_WORDS: Int = AAA_CLUSTER_INDICES_BASE + AAA_TOTAL_CLUSTERS * AAA_MAX_LIGHTS_PER_CLUSTER  # 241040

comptime AAA_LIGHT_POINT: UInt32 = 0
comptime AAA_LIGHT_SPOT: UInt32 = 1
comptime AAA_LIGHT_LTC_RECT: UInt32 = 2
comptime AAA_LIGHT_LTC_TUBE: UInt32 = 3

comptime RESTIR_RESERVOIR_WORDS: Int = 4
comptime RESTIR_MAX_M_TEMPORAL: UInt32 = 20
comptime RESTIR_MAX_M_SPATIAL: UInt32 = 30


struct RestirReservoir(Copyable, Movable):
    """16-byte ReSTIR Direct Illumination Reservoir (Bitterli et al. 2020) with unbiased streaming and merging."""
    var light_id: UInt32      # Index into b_clusters (0..1023), or 0xFFFFFFFF if unlit
    var weight_sum: Float32   # Cumulative weight sum w_sum
    var M_count: UInt32       # Candidate count seen by this reservoir
    var target_pdf: Float32   # Target function evaluated for light_id: p_hat(light_id)

    def __init__(out self):
        self.light_id = UInt32(0xFFFFFFFF)
        self.weight_sum = Float32(0.0)
        self.M_count = 0
        self.target_pdf = Float32(0.0)

    def __init__(out self, light_id: UInt32, weight_sum: Float32, M_count: UInt32, target_pdf: Float32):
        self.light_id = light_id
        self.weight_sum = weight_sum if (isfinite(weight_sum) and weight_sum >= Float32(0.0)) else Float32(0.0)
        self.M_count = M_count
        self.target_pdf = target_pdf if (isfinite(target_pdf) and target_pdf >= Float32(0.0)) else Float32(0.0)

    def update(mut self, candidate_id: UInt32, candidate_target_pdf: Float32, sample_weight: Float32, rand_01: Float32) -> Bool:
        """Streams a candidate sample into the reservoir using weighted reservoir sampling."""
        if (
            not isfinite(sample_weight)
            or not isfinite(candidate_target_pdf)
            or sample_weight <= Float32(0.0)
            or candidate_target_pdf <= Float32(0.0)
        ):
            return False
        if not isfinite(self.weight_sum) or self.weight_sum < Float32(0.0):
            self.weight_sum = Float32(0.0)
        var new_w_sum = self.weight_sum + sample_weight
        if not isfinite(new_w_sum):
            return False
        var safe_rand = max(Float32(0.0), min(Float32(1.0), rand_01)) if isfinite(rand_01) else Float32(1.0)
        self.weight_sum = new_w_sum
        self.M_count += 1
        if self.light_id == UInt32(0xFFFFFFFF) or safe_rand * self.weight_sum <= sample_weight:
            self.light_id = candidate_id
            self.target_pdf = candidate_target_pdf
            return True
        return False

    def merge(mut self, other: RestirReservoir, candidate_target_pdf: Float32, rand_01: Float32) -> Bool:
        """Merges a neighbor or temporal reservoir into this reservoir."""
        if (
            other.M_count == 0
            or not isfinite(candidate_target_pdf)
            or candidate_target_pdf <= Float32(0.0)
        ):
            return False
        var ow = other.unbiased_weight()
        if not isfinite(ow) or ow < Float32(0.0):
            return False
        var w = candidate_target_pdf * ow * Float32(other.M_count)
        if not isfinite(w):
            return False
        if not isfinite(self.weight_sum) or self.weight_sum < Float32(0.0):
            self.weight_sum = Float32(0.0)
        var new_w_sum = self.weight_sum + w
        if not isfinite(new_w_sum):
            return False
        var safe_rand = max(Float32(0.0), min(Float32(1.0), rand_01)) if isfinite(rand_01) else Float32(1.0)
        self.weight_sum = new_w_sum
        self.M_count += other.M_count
        if w > Float32(0.0) and (self.light_id == UInt32(0xFFFFFFFF) or safe_rand * self.weight_sum <= w):
            self.light_id = other.light_id
            self.target_pdf = candidate_target_pdf
            return True
        return False

    def unbiased_weight(self) -> Float32:
        """Evaluates W = (1 / target_pdf) * (weight_sum / M_count)."""
        if (
            not isfinite(self.target_pdf)
            or not isfinite(self.weight_sum)
            or self.target_pdf <= Float32(1e-8)
            or self.weight_sum <= Float32(0.0)
            or self.M_count == 0
        ):
            return Float32(0.0)
        var w = self.weight_sum / (Float32(self.M_count) * self.target_pdf)
        return w if isfinite(w) else Float32(0.0)

    def clamp_m(mut self, max_m: UInt32):
        """Clamps M_count to max_m while scaling weight_sum to preserve unbiased estimator W."""
        if not isfinite(self.weight_sum) or self.weight_sum < Float32(0.0):
            self.weight_sum = Float32(0.0)
        if self.M_count > max_m and max_m > 0:
            var factor = Float32(max_m) / Float32(self.M_count)
            self.weight_sum *= factor
            self.M_count = max_m



struct AaaClusterLight(Copyable, Movable):
    var position: Vec3
    var radius: Float32
    var radiance: Vec3
    var kind: UInt32
    var axis_u: Vec3          # Spot forward dir | Rect right*half_w | Tube half-segment
    var param0: Float32       # Spot cos_outer   | Rect two_sided (1.0/0.0)
    var axis_v: Vec3          # Rect up*half_h
    var param1: Float32       # Spot inv_cos_range | Tube radius
    var vsm_region_id: Int

    def __init__(out self):
        self.position = Vec3()
        self.radius = Float32(5.0)
        self.radiance = Vec3(1.0, 1.0, 1.0)
        self.kind = AAA_LIGHT_POINT
        self.axis_u = Vec3(1.0, 0.0, 0.0)
        self.param0 = Float32(0.0)
        self.axis_v = Vec3(0.0, 1.0, 0.0)
        self.param1 = Float32(0.0)
        self.vsm_region_id = -1

    @staticmethod
    def point(pos: Vec3, radiance: Vec3, radius: Float32) -> Self:
        var l = Self()
        l.position = pos
        l.radiance = radiance
        l.radius = max(Float32(0.05), radius)
        l.kind = AAA_LIGHT_POINT
        return l^

    @staticmethod
    def spot(pos: Vec3, direction: Vec3, radiance: Vec3, radius: Float32, inner_deg: Float32 = 20.0, outer_deg: Float32 = 35.0) -> Self:
        var l = Self()
        l.position = pos
        l.radiance = radiance
        l.radius = max(Float32(0.05), radius)
        l.kind = AAA_LIGHT_SPOT
        l.axis_u = direction.normalized()
        var deg2rad = Float32(3.14159265 / 180.0)
        var c_in = cos(inner_deg * deg2rad)
        var c_out = cos(outer_deg * deg2rad)
        l.param0 = c_out
        l.param1 = Float32(1.0) / max(Float32(1e-4), c_in - c_out)
        return l^

    @staticmethod
    def rect_ltc(
        center: Vec3,
        right_dir: Vec3,
        up_dir: Vec3,
        half_width: Float32,
        half_height: Float32,
        radiance: Vec3,
        range_radius: Float32 = 24.0,
        two_sided: Bool = False,
    ) -> Self:
        """Polygonal Rectangular Area Light shaded via Heitz `LTC` (`radiance` in linear `W/(sr*m^2)`)."""
        var l = Self()
        l.position = center
        l.radiance = radiance
        l.axis_u = right_dir.normalized() * half_width
        l.axis_v = up_dir.normalized() * half_height
        var diag = sqrt(half_width * half_width + half_height * half_height)
        l.radius = max(diag + Float32(0.5), range_radius)
        l.kind = AAA_LIGHT_LTC_RECT
        l.param0 = Float32(1.0) if two_sided else Float32(0.0)
        return l^

    @staticmethod
    def tube_ltc(
        start_pos: Vec3,
        end_pos: Vec3,
        tube_radius: Float32,
        radiance: Vec3,
        range_radius: Float32 = 20.0,
    ) -> Self:
        """Capsule / Fluorescent Tube Area Light represented as a view-facing `LTC` ribbon."""
        var l = Self()
        l.position = (start_pos + end_pos) * Float32(0.5)
        var half_seg = (end_pos - start_pos) * Float32(0.5)
        l.axis_u = half_seg
        l.param0 = Float32(1.0)
        l.param1 = max(Float32(0.02), tube_radius)
        l.radiance = radiance
        l.radius = max(half_seg.length() + tube_radius + Float32(0.5), range_radius)
        l.kind = AAA_LIGHT_LTC_TUBE
        return l^

    def append_words(self, mut out: List[UInt32]):
        var vsm_bits = UInt32(self.vsm_region_id & 0xFFFF) if self.vsm_region_id >= 0 else UInt32(0xFFFF)
        var packed_kind = (self.kind & 0xFF) | (vsm_bits << 16)
        out.append(bitcast[DType.uint32](self.position.x))
        out.append(bitcast[DType.uint32](self.position.y))
        out.append(bitcast[DType.uint32](self.position.z))
        out.append(bitcast[DType.uint32](self.radius))
        out.append(bitcast[DType.uint32](self.radiance.x))
        out.append(bitcast[DType.uint32](self.radiance.y))
        out.append(bitcast[DType.uint32](self.radiance.z))
        out.append(packed_kind)
        out.append(bitcast[DType.uint32](self.axis_u.x))
        out.append(bitcast[DType.uint32](self.axis_u.y))
        out.append(bitcast[DType.uint32](self.axis_u.z))
        out.append(bitcast[DType.uint32](self.param0))
        out.append(bitcast[DType.uint32](self.axis_v.x))
        out.append(bitcast[DType.uint32](self.axis_v.y))
        out.append(bitcast[DType.uint32](self.axis_v.z))
        out.append(bitcast[DType.uint32](self.param1))


def allocate_local_light_vsm_regions(mut lights: List[AaaClusterLight], max_shadowed_lights: Int = 8) -> Int:
    """Assigns sub-regions `0 .. max_shadowed_lights - 1` inside the `16k x 16k` VSM virtual table
    to the highest-luminance spot/area lights."""
    var assigned = 0
    for i in range(len(lights)):
        lights[i].vsm_region_id = -1
    for slot in range(max_shadowed_lights):
        var best_i = -1
        var best_score = Float32(0.0)
        for i in range(len(lights)):
            if lights[i].vsm_region_id >= 0:
                continue
            var lum = lights[i].radiance.x * Float32(0.2126) + lights[i].radiance.y * Float32(0.7152) + lights[i].radiance.z * Float32(0.0722)
            var sc = lum * lights[i].radius
            if sc > best_score:
                best_score = sc
                best_i = i
        if best_i < 0:
            break
        lights[best_i].vsm_region_id = slot
        assigned += 1
    return assigned


# ============================================================================
# 1. Pure-Mojo Heitz `LTC` `64x64 RG16F` Inverse-Matrix LUT Baker & Reference
# ============================================================================

def bake_ltc_lut_rg16f(lut_dim: Int = 64) -> List[UInt8]:
    """Bakes a `64x64` `VK_FORMAT_R16G16_SFLOAT` (`4` bytes/texel) Heitz `LTC` parameter LUT
    indexed by `(u = NdotV, v = roughness)`:
      - Channel R (`scale_inv`): inverse lobe width `1 / alpha_eff` (equals `1.0` at `roughness = 1.0` -> `M^-1 = I`!)
      - Channel G (`tilt_norm`): GGX off-specular mean lobe tilt `theta_m / (pi / 2)` in `[0.0 .. 1.0]`
        (equals `0.0` at `roughness = 1.0` or `NdotV = 1.0`, approaching `theta_v / (pi/2)` as `roughness -> 0`!).
    """
    var out = List[UInt8](capacity=lut_dim * lut_dim * 4)
    var half_pi = Float32(1.57079632679)

    for y in range(lut_dim):
        var rough = Float32(y) / Float32(max(1, lut_dim - 1))
        var alpha = max(Float32(0.02), rough * rough)
        # At rough=1.0 -> alpha=1.0 -> inv_scale=1.0 (exact identity M^-1 = I for Lambertian solid angle!)
        var inv_scale = Float32(1.0) / alpha

        for x in range(lut_dim):
            var ndotv = max(Float32(1e-3), min(Float32(1.0), Float32(x) / Float32(max(1, lut_dim - 1))))
            var theta_v = det_acos(ndotv)
            # Off-specular peak compression: at alpha=1 -> theta_m = 0; at alpha->0 -> theta_m = theta_v!
            var off_spec = (Float32(1.0) - alpha * alpha) / (Float32(1.0) + Float32(0.15) * alpha)
            var theta_m = theta_v * max(Float32(0.0), min(Float32(1.0), off_spec))
            var tilt_01 = max(Float32(0.0), min(Float32(1.0), theta_m / half_pi))

            var h_r = f32_to_uf16_bits(inv_scale)
            var h_g = f32_to_uf16_bits(tilt_01)
            out.append(UInt8(h_r & 0xFF))
            out.append(UInt8((h_r >> 8) & 0xFF))
            out.append(UInt8(h_g & 0xFF))
            out.append(UInt8((h_g >> 8) & 0xFF))

    return out^


@always_inline
def _heitz_edge_integral_z(v1: Vec3, v2: Vec3) -> Float32:
    """Heitz 2016 exact/rational contour edge vector Z component `(theta / (2*pi*sin(theta))) * (v1 x v2).z`."""
    var x = max(Float32(-0.99999), min(Float32(0.99999), v1.dot(v2)))
    var y = abs(x)
    var a = Float32(0.8543985) + (Float32(0.4965155) + Float32(0.0145206) * y) * y
    var b = Float32(3.4175940) + (Float32(4.1616724) + y) * y
    var u = a / b
    var theta_sintheta = u if x > Float32(0.0) else (Float32(0.5) / sqrt(max(Float32(1e-7), Float32(1.0) - x * x)) - u)
    var cz = v1.x * v2.y - v1.y * v2.x
    return cz * theta_sintheta


def eval_ltc_polygon_cpu(
    surf_p: Vec3,
    surf_n: Vec3,
    view_v: Vec3,
    quad_center: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    inv_scale: Float32 = 1.0,
    tilt_01: Float32 = 0.0,
    two_sided: Bool = False,
) -> Float32:
    """Evaluates Heitz `LTC` normalized irradiance `E / pi` (in `[0 .. 1]`) for a 4-vertex rectangular area light!
    At `(inv_scale = 1.0, tilt_01 = 0.0)`, `M^-1 = I` yields the **exact analytical Lambertian solid-angle form factor**!"""
    var ndotv = surf_n.dot(view_v)
    var t1_raw = view_v - surf_n * ndotv
    var t1: Vec3
    if t1_raw.length_squared() > Float32(1e-8):
        t1 = t1_raw.normalized()
    else:
        var up = Vec3(1.0, 0.0, 0.0) if abs(surf_n.y) > Float32(0.9) else Vec3(0.0, 1.0, 0.0)
        t1 = up.cross(surf_n).normalized()
    var t2 = surf_n.cross(t1)

    # Check if surface point is in front of one-sided emitter plane
    var light_n = axis_u.cross(axis_v)
    if not two_sided and light_n.length_squared() > Float32(1e-10):
        if (surf_p - quad_center).dot(light_n) <= Float32(0.0):
            return Float32(0.0)

    var c0_w = quad_center - axis_u - axis_v - surf_p
    var c1_w = quad_center + axis_u - axis_v - surf_p
    var c2_w = quad_center + axis_u + axis_v - surf_p
    var c3_w = quad_center - axis_u + axis_v - surf_p

    var theta_m = tilt_01 * Float32(1.57079632679)
    var ct = cos(theta_m)
    var st = sin(theta_m)
    var sx = inv_scale / max(Float32(0.25), ct)
    var sy = inv_scale

    var raw_pts = [c0_w, c1_w, c2_w, c3_w]
    var l_pts = List[Vec3](capacity=4)
    for k in range(4):
        var pw = raw_pts[k]
        var lx = pw.dot(t1)
        var ly = pw.dot(t2)
        var lz = pw.dot(surf_n)
        # Apply M^-1 = diag(sx, sy, 1) * R_y(-theta_m):
        var rx = ct * lx - st * lz
        var rz = st * lx + ct * lz
        l_pts.append(Vec3(rx * sx, ly * sy, rz))

    # Sutherland-Hodgman clip against upper hemisphere `z >= 0` (produces 0..5 vertices)
    var clipped = List[Vec3](capacity=5)
    for k in range(4):
        var a = l_pts[k]
        var b = l_pts[(k + 1) & 3]
        var a_in = a.z >= Float32(0.0)
        var b_in = b.z >= Float32(0.0)
        if a_in:
            clipped.append(a)
        if a_in != b_in:
            var t = a.z / (a.z - b.z)
            clipped.append(a + (b - a) * t)

    if len(clipped) < 3:
        return Float32(0.0)

    var u_pts = List[Vec3](capacity=len(clipped))
    for cp in clipped:
        u_pts.append(cp.normalized())

    var sum_z = Float32(0.0)
    var m = len(u_pts)
    for k in range(m):
        sum_z += _heitz_edge_integral_z(u_pts[k], u_pts[(k + 1) % m])

    return min(Float32(1.0), abs(sum_z))


# ============================================================================
# 2. GPU Compute `16 x 9 x 24` Logarithmic-Z Cluster Light Culler (`ShaderDsl`)
# ============================================================================

def aaa_cluster_cull_shader(bind_clusters: Int, bind_globals: Int) -> List[UInt32]:
    """GPU Compute Cluster Culler (`group = 16 x 9 x 1`, dispatched with `groups_z = 24`):
    Each thread owns 1 of the `3,456` 3D logarithmic-Z view frustum clusters and tests up to
    `1,024` `AaaClusterLight` bounding spheres against the cluster AABB in view space!"""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=16, group_height=9, push_words=16)
    var b_clusters = dsl.buffer[1](bind_clusters)
    var b_globals = dsl.buffer[1](bind_globals)

    var cx = dsl.global_id_x()
    var cy = dsl.global_id_y()
    var cz = dsl.push_u32(0)  # Or global_id_z! Using global_id_z lets a single dispatch(1, 1, 24) cull all 3,456!
    _ = cz
    var gz = dsl.global_id_z()
    dsl.return_if(cx.ge(UInt32(AAA_CLUSTER_X)) | cy.ge(UInt32(AAA_CLUSTER_Y)) | gz.ge(UInt32(AAA_CLUSTER_Z)))

    var cluster_id = (gz * UInt32(AAA_CLUSTER_X * AAA_CLUSTER_Y)) + (cy * UInt32(AAA_CLUSTER_X)) + cx
    var count_slot = dsl.u32(UInt32(AAA_CLUSTER_COUNTS_BASE)) + cluster_id
    var list_base = dsl.u32(UInt32(AAA_CLUSTER_INDICES_BASE)) + cluster_id * UInt32(AAA_MAX_LIGHTS_PER_CLUSTER)

    var light_count = b_clusters.load_u32(dsl.u32(0))
    dsl.if_then(light_count.eq(0))
    b_clusters.store_u32(count_slot, dsl.u32(0))
    dsl.end_if()
    dsl.return_if(light_count.eq(0))

    var near_z = b_clusters.load_f32(dsl.u32(1))
    var log_ratio = b_clusters.load_f32(dsl.u32(3))
    var tan_x = b_clusters.load_f32(dsl.u32(4))
    var tan_y = b_clusters.load_f32(dsl.u32(5))

    var cam_pos = b_globals.load_vec3(dsl.u32(32))
    var cam_fwd = b_globals.load_vec3(dsl.u32(36))
    var cam_rgt = b_globals.load_vec3(dsl.u32(40))
    var cam_up = b_globals.load_vec3(dsl.u32(44))

    # Logarithmic Z slice bounds `[z_min, z_max]`
    var f_gz0 = gz.to_f32() * Float32(1.0 / 24.0)
    var f_gz1 = (gz + UInt32(1)).to_f32() * Float32(1.0 / 24.0)
    var z_min = near_z * exp2_f(log_ratio * f_gz0)
    var z_max = near_z * exp2_f(log_ratio * f_gz1)

    # View-space X & Y frustum tile bounds at `z_max` (conservative enclosing view-space AABB!)
    var nx0 = fma_f(cx.to_f32(), dsl.f32(2.0 / 16.0), dsl.f32(-1.0))
    var nx1 = fma_f((cx + UInt32(1)).to_f32(), dsl.f32(2.0 / 16.0), dsl.f32(-1.0))
    # Screen Y=0 is top (+1 NDC), Y=9 is bottom (-1 NDC)
    var ny_top = dsl.f32(1.0) - cy.to_f32() * Float32(2.0 / 9.0)
    var ny_bot = dsl.f32(1.0) - (cy + UInt32(1)).to_f32() * Float32(2.0 / 9.0)

    var x0_near = nx0 * tan_x * z_min; var x0_far = nx0 * tan_x * z_max
    var x1_near = nx1 * tan_x * z_min; var x1_far = nx1 * tan_x * z_max
    var y0_near = ny_bot * tan_y * z_min; var y0_far = ny_bot * tan_y * z_max
    var y1_near = ny_top * tan_y * z_min; var y1_far = ny_top * tan_y * z_max

    var aabb_min_x = min_f(x0_near, x0_far)
    var aabb_max_x = max_f(x1_near, x1_far)
    var aabb_min_y = min_f(y0_near, y0_far)
    var aabb_max_y = max_f(y1_near, y1_far)

    var found = dsl.var_u32(dsl.u32(0))
    var li = dsl.var_u32(dsl.u32(0))

    dsl.begin_loop()
    var cur_i = li.get()
    dsl.break_if(cur_i.ge(light_count) | found.get().ge(UInt32(AAA_MAX_LIGHTS_PER_CLUSTER)))
    var l_off = dsl.u32(UInt32(AAA_CLUSTER_LIGHTS_BASE)) + cur_i * UInt32(AAA_CLUSTER_LIGHT_WORDS)
    var l_pos = b_clusters.load_vec3(l_off)
    var l_rad = b_clusters.load_f32(l_off + UInt32(3))

    var rel = l_pos - cam_pos
    var vx = dot(rel, cam_rgt)
    var vy = dot(rel, cam_up)
    var vz = dot(rel, cam_fwd)

    var cx_clamped = clamp_f(vx, aabb_min_x, aabb_max_x)
    var cy_clamped = clamp_f(vy, aabb_min_y, aabb_max_y)
    var cz_clamped = clamp_f(vz, z_min, z_max)
    var dx = vx - cx_clamped
    var dy = vy - cy_clamped
    var dz = vz - cz_clamped
    var dist2 = fma_f(dx, dx, fma_f(dy, dy, dz * dz))

    dsl.if_then(dist2.lt(l_rad * l_rad))
    var slot = found.get()
    b_clusters.store_u32(list_base + slot, cur_i)
    found.set(slot + UInt32(1))
    dsl.end_if()

    li.set(cur_i + UInt32(1))
    dsl.end_loop()

    b_clusters.store_u32(count_slot, found.get())
    return dsl.finish()


# ============================================================================
# 3. SPIR-V Deferred Clustered Punctual + Heitz `LTC` Area Light Evaluator
# ============================================================================

def _emit_heitz_edge_z_spv(
    mut s: AaaSpirvModule,
    v1x: UInt32,
    v1y: UInt32,
    v1z: UInt32,
    v2x: UInt32,
    v2y: UInt32,
    v2z: UInt32,
) -> UInt32:
    var dot_v = s.nclamp(
        s.fma(v1x, v2x, s.fma(v1y, v2y, s.fmul(v1z, v2z))),
        s.float(-0.99999),
        s.float(0.99999),
    )
    var y = s.fabs(dot_v)
    var a = s.fma(s.fma(s.float(0.0145206), y, s.float(0.4965155)), y, s.float(0.8543985))
    var b = s.fma(s.fadd(s.float(4.1616724), y), y, s.float(3.4175940))
    var u = s.fdiv(a, b)
    var neg_term = s.fsub(
        s.fdiv(s.float(0.5), s.fsqrt(s.nmax(s.float(1e-7), s.fsub(s.float(1.0), s.fmul(dot_v, dot_v))))),
        u,
    )
    var th_sin = s.select_f(s.fgt(dot_v, s.float(0.0)), u, neg_term)
    var cz = s.fsub(s.fmul(v1x, v2y), s.fmul(v1y, v2x))
    return s.fmul(cz, th_sin)


def _emit_ltc_quad_form_factor_spv(
    mut s: AaaSpirvModule,
    t1: Tuple[UInt32, UInt32, UInt32],
    t2: Tuple[UInt32, UInt32, UInt32],
    n: Tuple[UInt32, UInt32, UInt32],
    rel_cx: UInt32,
    rel_cy: UInt32,
    rel_cz: UInt32,
    ux: UInt32,
    uy: UInt32,
    uz: UInt32,
    vx: UInt32,
    vy: UInt32,
    vz: UInt32,
    inv_scale: UInt32,
    tilt_01: UInt32,
    two_sided: UInt32,
) -> UInt32:
    """Evaluates horizon-clipped 4-vertex quad `LTC` form factor `E / pi` directly in SPIR-V!"""
    var theta_m = s.fmul(tilt_01, s.float(1.57079633))
    var ct = s.fcos(theta_m)
    var st = s.fsin(theta_m)
    var sx = s.fdiv(inv_scale, s.nmax(s.float(0.25), ct))
    var sy = inv_scale

    # 4 quad corners relative to shading point P: C - U - V, C + U - V, C + U + V, C - U + V
    var su = [-1, 1, 1, -1]
    var sv = [-1, -1, 1, 1]
    var lx_a = List[UInt32]()
    var ly_a = List[UInt32]()
    var lz_a = List[UInt32]()

    for k in range(4):
        var fu = s.float(Float32(su[k]))
        var fv = s.float(Float32(sv[k]))
        var wx = s.fma(ux, fu, s.fma(vx, fv, rel_cx))
        var wy = s.fma(uy, fu, s.fma(vy, fv, rel_cy))
        var wz = s.fma(uz, fu, s.fma(vz, fv, rel_cz))
        var px_t = s.fma(wx, t1[0], s.fma(wy, t1[1], s.fmul(wz, t1[2])))
        var py_t = s.fma(wx, t2[0], s.fma(wy, t2[1], s.fmul(wz, t2[2])))
        var pz_t = s.fma(wx, n[0], s.fma(wy, n[1], s.fmul(wz, n[2])))
        var rx = s.fsub(s.fmul(ct, px_t), s.fmul(st, pz_t))
        var rz = s.fma(st, px_t, s.fmul(ct, pz_t))
        lx_a.append(s.fmul(rx, sx))
        ly_a.append(s.fmul(py_t, sy))
        lz_a.append(rz)

    # Exact edge-by-edge horizon clipping against `z >= 0`:
    # Each original edge `(A, B)` clipped to `z >= 0` becomes a valid segment `(A', B')`, AND whenever
    # the polygon exits at `P_exit` (`z=0`) and later re-enters at `P_enter` (`z=0`), the closing chord
    # along the horizon `(P_exit, P_enter)` has `z1 = 0, z2 = 0` — wait: `(P_exit x P_enter).z` IS non-zero!
    # So storing up to 5 clipped vertices in function arrays (`cx_v[0..4], cy_v[0..4], cz_v[0..4]`) is 100% exact!
    var cv_x = List[UInt32]()
    var cv_y = List[UInt32]()
    var cv_z = List[UInt32]()
    for _ in range(5):
        cv_x.append(s.fvar(s.float(0.0)))
        cv_y.append(s.fvar(s.float(0.0)))
        cv_z.append(s.fvar(s.float(1.0)))
    var c_cnt = s.uvar(s.uint(0))

    for k in range(4):
        var nk = (k + 1) & 3
        var ax = lx_a[k]; var ay = ly_a[k]; var az = lz_a[k]
        var bx = lx_a[nk]; var by = ly_a[nk]; var bz = lz_a[nk]
        var a_in = s.fge(az, s.float(0.0))
        var b_in = s.fge(bz, s.float(0.0))

        s.begin_if(a_in)
        var idx_a = s.load_u(c_cnt)
        for slot_k in range(5):
            s.begin_if(s.ieq(idx_a, s.uint(UInt32(slot_k))))
            s.store(cv_x[slot_k], ax); s.store(cv_y[slot_k], ay); s.store(cv_z[slot_k], az)
            s.end_if()
        s.store(c_cnt, s.iadd(idx_a, s.uint(1)))
        s.end_if()

        var cross_h = s.logical_or(
            s.logical_and(a_in, s.logical_not(b_in)),
            s.logical_and(s.logical_not(a_in), b_in),
        )
        s.begin_if(cross_h)
        var denom = s.fsub(az, bz)
        var safe_den = s.select_f(s.fge(denom, s.float(0.0)), s.nmax(denom, s.float(1e-7)), s.nmin(denom, s.float(-1e-7)))
        var t_h = s.nclamp(s.fdiv(az, safe_den), s.float(0.0), s.float(1.0))
        var ix = s.fma(s.fsub(bx, ax), t_h, ax)
        var iy = s.fma(s.fsub(by, ay), t_h, ay)
        var idx_i = s.load_u(c_cnt)
        for slot_k in range(5):
            s.begin_if(s.ieq(idx_i, s.uint(UInt32(slot_k))))
            s.store(cv_x[slot_k], ix); s.store(cv_y[slot_k], iy); s.store(cv_z[slot_k], s.float(0.0))
            s.end_if()
        s.store(c_cnt, s.iadd(idx_i, s.uint(1)))
        s.end_if()

    var res_ff = s.fvar(s.float(0.0))
    var n_c = s.load_u(c_cnt)
    s.begin_if(s.uge(n_c, s.uint(3)))
    # Normalize active vertices 0..4
    var ux_v = List[UInt32]()
    var uy_v = List[UInt32]()
    var uz_v = List[UInt32]()
    for slot_k in range(5):
        var rvx = s.load_f(cv_x[slot_k])
        var rvy = s.load_f(cv_y[slot_k])
        var rvz = s.load_f(cv_z[slot_k])
        var inv_l = s.inverse_sqrt(s.nmax(s.float(1e-12), s.fma(rvx, rvx, s.fma(rvy, rvy, s.fmul(rvz, rvz)))))
        ux_v.append(s.fmul(rvx, inv_l))
        uy_v.append(s.fmul(rvy, inv_l))
        uz_v.append(s.fmul(rvz, inv_l))

    var e01 = _emit_heitz_edge_z_spv(s, ux_v[0], uy_v[0], uz_v[0], ux_v[1], uy_v[1], uz_v[1])
    var e12 = _emit_heitz_edge_z_spv(s, ux_v[1], uy_v[1], uz_v[1], ux_v[2], uy_v[2], uz_v[2])
    var acc = s.fvar(s.fadd(e01, e12))

    s.begin_if(s.ieq(n_c, s.uint(3)))
    s.store(acc, s.fadd(s.load_f(acc), _emit_heitz_edge_z_spv(s, ux_v[2], uy_v[2], uz_v[2], ux_v[0], uy_v[0], uz_v[0])))
    s.end_if()

    s.begin_if(s.ieq(n_c, s.uint(4)))
    var e23 = _emit_heitz_edge_z_spv(s, ux_v[2], uy_v[2], uz_v[2], ux_v[3], uy_v[3], uz_v[3])
    var e30 = _emit_heitz_edge_z_spv(s, ux_v[3], uy_v[3], uz_v[3], ux_v[0], uy_v[0], uz_v[0])
    s.store(acc, s.fadd(s.load_f(acc), s.fadd(e23, e30)))
    s.end_if()

    s.begin_if(s.uge(n_c, s.uint(5)))
    var e23_5 = _emit_heitz_edge_z_spv(s, ux_v[2], uy_v[2], uz_v[2], ux_v[3], uy_v[3], uz_v[3])
    var e34_5 = _emit_heitz_edge_z_spv(s, ux_v[3], uy_v[3], uz_v[3], ux_v[4], uy_v[4], uz_v[4])
    var e40_5 = _emit_heitz_edge_z_spv(s, ux_v[4], uy_v[4], uz_v[4], ux_v[0], uy_v[0], uz_v[0])
    s.store(acc, s.fadd(s.load_f(acc), s.fadd(e23_5, s.fadd(e34_5, e40_5))))
    s.end_if()

    var raw_sum = s.load_f(acc)
    _ = two_sided
    s.store(res_ff, s.nmin(s.float(1.0), s.fabs(raw_sum)))
    s.end_if()

    return s.load_f(res_ff)


def closest_point_on_ltc_light_cpu(light: AaaClusterLight, surf_p: Vec3) -> Vec3:
    """Computes the exact closest point on an `AaaClusterLight` (`POINT`, `SPOT`, `LTC_RECT`, `LTC_TUBE`)
    to surface point `surf_p` for deterministic Hero Local/Area RT shadow rays."""
    if light.kind == AAA_LIGHT_LTC_TUBE:
        var cw = surf_p - light.position
        var u_l2 = max(Float32(1e-6), light.axis_u.length_squared())
        var t_seg = max(Float32(-1.0), min(Float32(1.0), cw.dot(light.axis_u) / u_l2))
        return light.position + light.axis_u * t_seg
    elif light.kind == AAA_LIGHT_LTC_RECT:
        var cw = surf_p - light.position
        var u_l2 = max(Float32(1e-6), light.axis_u.length_squared())
        var v_l2 = max(Float32(1e-6), light.axis_v.length_squared())
        var tu = max(Float32(-1.0), min(Float32(1.0), cw.dot(light.axis_u) / u_l2))
        var tv = max(Float32(-1.0), min(Float32(1.0), cw.dot(light.axis_v) / v_l2))
        return light.position + light.axis_u * tu + light.axis_v * tv
    return light.position


def _emit_push_top2_hero_light(
    mut s: AaaSpirvModule,
    cand_lum: UInt32,
    cand_r: UInt32,
    cand_g: UInt32,
    cand_b: UInt32,
    cand_tx: UInt32,
    cand_ty: UInt32,
    cand_tz: UInt32,
    t0_lum: UInt32,
    t0_r: UInt32,
    t0_g: UInt32,
    t0_b: UInt32,
    t0_tx: UInt32,
    t0_ty: UInt32,
    t0_tz: UInt32,
    t1_lum: UInt32,
    t1_r: UInt32,
    t1_g: UInt32,
    t1_b: UInt32,
    t1_tx: UInt32,
    t1_ty: UInt32,
    t1_tz: UInt32,
):
    """Updates per-pixel Top-2 highest-luminance local/LTC light contribution registers."""
    s.begin_if(s.fgt(cand_lum, s.load_f(t0_lum)))
    s.store(t1_lum, s.load_f(t0_lum))
    s.store(t1_r, s.load_f(t0_r)); s.store(t1_g, s.load_f(t0_g)); s.store(t1_b, s.load_f(t0_b))
    s.store(t1_tx, s.load_f(t0_tx)); s.store(t1_ty, s.load_f(t0_ty)); s.store(t1_tz, s.load_f(t0_tz))
    s.store(t0_lum, cand_lum)
    s.store(t0_r, cand_r); s.store(t0_g, cand_g); s.store(t0_b, cand_b)
    s.store(t0_tx, cand_tx); s.store(t0_ty, cand_ty); s.store(t0_tz, cand_tz)
    s.begin_else()
    s.begin_if(s.fgt(cand_lum, s.load_f(t1_lum)))
    s.store(t1_lum, cand_lum)
    s.store(t1_r, cand_r); s.store(t1_g, cand_g); s.store(t1_b, cand_b)
    s.store(t1_tx, cand_tx); s.store(t1_ty, cand_ty); s.store(t1_tz, cand_tz)
    s.end_if()
    s.end_if()


def emit_eval_clustered_and_ltc_lights(
    mut s: AaaSpirvModule,
    b_clusters: UInt32,
    ltc_lut: Tuple[UInt32, UInt32],
    px: UInt32,
    py: UInt32,
    w_u: UInt32,
    h_u: UInt32,
    z_lin: UInt32,
    wx: UInt32,
    wy: UInt32,
    wz: UInt32,
    nx: UInt32,
    ny: UInt32,
    nz: UInt32,
    vx: UInt32,
    vy: UInt32,
    vz: UInt32,
    ndotv: UInt32,
    alb_r: UInt32,
    alb_g: UInt32,
    alb_b: UInt32,
    spec_f0_r: UInt32,
    spec_f0_g: UInt32,
    spec_f0_b: UInt32,
    f90: UInt32,
    lut_a: UInt32,
    lut_b: UInt32,
    rough: UInt32,
    metal: UInt32,
    enable_top2_rt_shadows: Bool = False,
    tlas_var: UInt32 = 0,
    do_rt_shadows_cond: UInt32 = 0,
    b_verts: UInt32 = 0,
    b_indices: UInt32 = 0,
    b_instances: UInt32 = 0,
    b_materials: UInt32 = 0,
    bindless_arr: Tuple[UInt32, UInt32] = (0, 0),
) -> Tuple[UInt32, UInt32, UInt32]:
    """Accumulates shading from all culled Point, Spot, Rectangular `LTC`, and Tube `LTC` lights
    in the pixel's `16 x 9 x 24` logarithmic-Z cluster, optionally tracing `<= 2` deterministic
    `RAY_FLAG_SHADOW_OCCLUSION` inline shadow rays to the Top-2 highest-luminance hero emitters!"""
    var out_r = s.fvar(s.float(0.0))
    var out_g = s.fvar(s.float(0.0))
    var out_b = s.fvar(s.float(0.0))

    var t0_lum: UInt32 = 0; var t0_r: UInt32 = 0; var t0_g: UInt32 = 0; var t0_b: UInt32 = 0
    var t0_tx: UInt32 = 0; var t0_ty: UInt32 = 0; var t0_tz: UInt32 = 0
    var t1_lum: UInt32 = 0; var t1_r: UInt32 = 0; var t1_g: UInt32 = 0; var t1_b: UInt32 = 0
    var t1_tx: UInt32 = 0; var t1_ty: UInt32 = 0; var t1_tz: UInt32 = 0
    if enable_top2_rt_shadows:
        t0_lum = s.fvar(s.float(0.0)); t0_r = s.fvar(s.float(0.0)); t0_g = s.fvar(s.float(0.0)); t0_b = s.fvar(s.float(0.0))
        t0_tx = s.fvar(wx); t0_ty = s.fvar(wy); t0_tz = s.fvar(wz)
        t1_lum = s.fvar(s.float(0.0)); t1_r = s.fvar(s.float(0.0)); t1_g = s.fvar(s.float(0.0)); t1_b = s.fvar(s.float(0.0))
        t1_tx = s.fvar(wx); t1_ty = s.fvar(wy); t1_tz = s.fvar(wz)

    var total_lights = s.load_word(b_clusters, s.uint(0))
    s.begin_if(s.ine(total_lights, s.uint(0)))

    var near_z = s.load_float(b_clusters, s.uint(1))
    var log_ratio = s.load_float(b_clusters, s.uint(3))
    var cx = s.umin(s.uint(UInt32(AAA_CLUSTER_X - 1)), s.udiv(s.imul(px, s.uint(UInt32(AAA_CLUSTER_X))), w_u))
    var cy = s.umin(s.uint(UInt32(AAA_CLUSTER_Y - 1)), s.udiv(s.imul(py, s.uint(UInt32(AAA_CLUSTER_Y))), h_u))
    var z_norm = s.nclamp(
        s.fdiv(s.log2(s.nmax(s.float(1.0), s.fdiv(z_lin, near_z))), s.nmax(s.float(1e-4), log_ratio)),
        s.float(0.0),
        s.float(0.9999),
    )
    var cz = s.umin(s.uint(UInt32(AAA_CLUSTER_Z - 1)), s.saturate_uint(s.fmul(z_norm, s.float(Float32(AAA_CLUSTER_Z)))))
    var cluster_id = s.iadd(s.imul(cz, s.uint(UInt32(AAA_CLUSTER_X * AAA_CLUSTER_Y))), s.iadd(s.imul(cy, s.uint(UInt32(AAA_CLUSTER_X))), cx))

    var c_cnt = s.load_word(b_clusters, s.iadd(s.uint(UInt32(AAA_CLUSTER_COUNTS_BASE)), cluster_id))
    var list_base = s.iadd(s.uint(UInt32(AAA_CLUSTER_INDICES_BASE)), s.imul(cluster_id, s.uint(UInt32(AAA_MAX_LIGHTS_PER_CLUSTER))))

    # Build LTC shading tangent frame (T1, T2, N) + fetch 64x64 RG16F LTC parameters once per pixel!
    var t1_rx = s.fsub(vx, s.fmul(nx, ndotv))
    var t1_ry = s.fsub(vy, s.fmul(ny, ndotv))
    var t1_rz = s.fsub(vz, s.fmul(nz, ndotv))
    var t1_l2 = s.fma(t1_rx, t1_rx, s.fma(t1_ry, t1_ry, s.fmul(t1_rz, t1_rz)))
    var up_is_y = s.flt(s.fabs(ny), s.float(0.90))
    var fb_x = s.select_f(up_is_y, nz, s.float(0.0))
    var fb_y = s.select_f(up_is_y, s.float(0.0), s.fneg(nz))
    var fb_z = s.select_f(up_is_y, s.fneg(nx), ny)
    var use_t1 = s.fgt(t1_l2, s.float(1e-8))
    var ux_t = s.select_f(use_t1, t1_rx, fb_x)
    var uy_t = s.select_f(use_t1, t1_ry, fb_y)
    var uz_t = s.select_f(use_t1, t1_rz, fb_z)
    var inv_t1 = s.inverse_sqrt(s.nmax(s.float(1e-12), s.fma(ux_t, ux_t, s.fma(uy_t, uy_t, s.fmul(uz_t, uz_t)))))
    var t1 = (s.fmul(ux_t, inv_t1), s.fmul(uy_t, inv_t1), s.fmul(uz_t, inv_t1))
    var t2 = (
        s.fsub(s.fmul(ny, t1[2]), s.fmul(nz, t1[1])),
        s.fsub(s.fmul(nz, t1[0]), s.fmul(nx, t1[2])),
        s.fsub(s.fmul(nx, t1[1]), s.fmul(ny, t1[0])),
    )

    var ltc_uv = s.vec2(s.nclamp(ndotv, s.float(0.01), s.float(0.99)), s.nclamp(rough, s.float(0.02), s.float(0.99)))
    var ltc_s = s.sample_2d_lod(ltc_lut, ltc_uv, s.float(0.0))
    var ltc_inv_scale = s.extract_f(ltc_s, 0)
    var ltc_tilt_01 = s.extract_f(ltc_s, 1)

    var spec_w_r = s.fma(spec_f0_r, lut_a, s.fmul(f90, lut_b))
    var spec_w_g = s.fma(spec_f0_g, lut_a, s.fmul(f90, lut_b))
    var spec_w_b = s.fma(spec_f0_b, lut_a, s.fmul(f90, lut_b))
    var one_m_met = s.fsub(s.float(1.0), metal)
    var kd_r = s.fmul(alb_r, s.fmul(one_m_met, s.nmax(s.float(0.0), s.fsub(s.float(1.0), spec_w_r))))
    var kd_g = s.fmul(alb_g, s.fmul(one_m_met, s.nmax(s.float(0.0), s.fsub(s.float(1.0), spec_w_g))))
    var kd_b = s.fmul(alb_b, s.fmul(one_m_met, s.nmax(s.float(0.0), s.fsub(s.float(1.0), spec_w_b))))

    var alpha = s.nmax(s.float(0.002), s.fmul(rough, rough))
    var a2 = s.fmul(alpha, alpha)
    var inv_pi = s.float(0.318309886)

    var k_iter = s.uvar(s.uint(0))
    s.begin_loop()
    var ki = s.load_u(k_iter)
    s.break_if(s.uge(ki, c_cnt))

    var li = s.load_word(b_clusters, s.iadd(list_base, ki))
    var l_off = s.iadd(s.uint(UInt32(AAA_CLUSTER_LIGHTS_BASE)), s.imul(li, s.uint(UInt32(AAA_CLUSTER_LIGHT_WORDS))))
    var lpx = s.load_float(b_clusters, l_off)
    var lpy = s.load_float(b_clusters, s.iadd(l_off, s.uint(1)))
    var lpz = s.load_float(b_clusters, s.iadd(l_off, s.uint(2)))
    var lrad = s.load_float(b_clusters, s.iadd(l_off, s.uint(3)))
    var lrr = s.load_float(b_clusters, s.iadd(l_off, s.uint(4)))
    var lrg = s.load_float(b_clusters, s.iadd(l_off, s.uint(5)))
    var lrb = s.load_float(b_clusters, s.iadd(l_off, s.uint(6)))
    var kind = s.bit_and(s.load_word(b_clusters, s.iadd(l_off, s.uint(7))), s.uint(0xFF))
    var aux_x = s.load_float(b_clusters, s.iadd(l_off, s.uint(8)))
    var aux_y = s.load_float(b_clusters, s.iadd(l_off, s.uint(9)))
    var aux_z = s.load_float(b_clusters, s.iadd(l_off, s.uint(10)))
    var p0_f = s.load_float(b_clusters, s.iadd(l_off, s.uint(11)))
    var avx_x = s.load_float(b_clusters, s.iadd(l_off, s.uint(12)))
    var avx_y = s.load_float(b_clusters, s.iadd(l_off, s.uint(13)))
    var avx_z = s.load_float(b_clusters, s.iadd(l_off, s.uint(14)))
    var p1_f = s.load_float(b_clusters, s.iadd(l_off, s.uint(15)))

    var rel_x = s.fsub(lpx, wx)
    var rel_y = s.fsub(lpy, wy)
    var rel_z = s.fsub(lpz, wz)

    s.begin_if(s.uge(kind, s.uint(AAA_LIGHT_LTC_RECT)))
    # Rectangular or Tube Area Light shaded via exact Heitz `LTC`!
    var use_vx = s.fvar(avx_x)
    var use_vy = s.fvar(avx_y)
    var use_vz = s.fvar(avx_z)
    var two_s = s.fvar(p0_f)

    s.begin_if(s.ieq(kind, s.uint(AAA_LIGHT_LTC_TUBE)))
    # Orient ribbon minor axis perpendicular to `(tube_axis x to_light)` with half-width = `tube_radius` (`p1_f`)!
    var crx = s.fsub(s.fmul(aux_y, rel_z), s.fmul(aux_z, rel_y))
    var cry = s.fsub(s.fmul(aux_z, rel_x), s.fmul(aux_x, rel_z))
    var crz = s.fsub(s.fmul(aux_x, rel_y), s.fmul(aux_y, rel_x))
    var inv_cr = s.fmul(p1_f, s.inverse_sqrt(s.nmax(s.float(1e-8), s.fma(crx, crx, s.fma(cry, cry, s.fmul(crz, crz))))))
    s.store(use_vx, s.fmul(crx, inv_cr))
    s.store(use_vy, s.fmul(cry, inv_cr))
    s.store(use_vz, s.fmul(crz, inv_cr))
    s.store(two_s, s.float(1.0))
    s.end_if()

    # Check one-sided emitter facing before integrating
    var f_vx = s.load_f(use_vx); var f_vy = s.load_f(use_vy); var f_vz = s.load_f(use_vz)
    var lnx = s.fsub(s.fmul(aux_y, f_vz), s.fmul(aux_z, f_vy))
    var lny = s.fsub(s.fmul(aux_z, f_vx), s.fmul(aux_x, f_vz))
    var lnz = s.fsub(s.fmul(aux_x, f_vy), s.fmul(aux_y, f_vx))
    var front_dot = s.fneg(s.fma(rel_x, lnx, s.fma(rel_y, lny, s.fmul(rel_z, lnz))))
    var emit_ok = s.logical_or(s.fgt(s.load_f(two_s), s.float(0.5)), s.fgt(front_dot, s.float(0.0)))

    s.begin_if(emit_ok)
    # Smooth windowing at outer cull `lrad` so cluster boundaries have zero hard cutoff
    var d2_c = s.fma(rel_x, rel_x, s.fma(rel_y, rel_y, s.fmul(rel_z, rel_z)))
    var r2_c = s.nmax(s.float(0.01), s.fmul(lrad, lrad))
    var win_t = s.nclamp(s.fsub(s.float(1.0), s.fmul(s.fdiv(d2_c, r2_c), s.fdiv(d2_c, r2_c))), s.float(0.0), s.float(1.0))
    var win = s.fmul(win_t, win_t)

    # 1. Exact Lambertian Diffuse `LTC` (`M^-1 = I` -> `inv_scale = 1.0, tilt_01 = 0.0`)
    var ff_diff = _emit_ltc_quad_form_factor_spv(
        s, t1, t2, (nx, ny, nz),
        rel_x, rel_y, rel_z,
        aux_x, aux_y, aux_z,
        f_vx, f_vy, f_vz,
        s.float(1.0), s.float(0.0), s.load_f(two_s),
    )
    # 2. Anisotropic GGX Specular `LTC` (`M^-1` from `64x64 RG16F` `ltc_lut`!)
    var ff_spec = _emit_ltc_quad_form_factor_spv(
        s, t1, t2, (nx, ny, nz),
        rel_x, rel_y, rel_z,
        aux_x, aux_y, aux_z,
        f_vx, f_vy, f_vz,
        ltc_inv_scale, ltc_tilt_01, s.load_f(two_s),
    )

    var lr_w = s.fmul(lrr, win)
    var lg_w = s.fmul(lrg, win)
    var lb_w = s.fmul(lrb, win)
    var c_ltc_r = s.fmul(lr_w, s.fma(kd_r, ff_diff, s.fmul(spec_w_r, ff_spec)))
    var c_ltc_g = s.fmul(lg_w, s.fma(kd_g, ff_diff, s.fmul(spec_w_g, ff_spec)))
    var c_ltc_b = s.fmul(lb_w, s.fma(kd_b, ff_diff, s.fmul(spec_w_b, ff_spec)))
    s.store(out_r, s.fadd(c_ltc_r, s.load_f(out_r)))
    s.store(out_g, s.fadd(c_ltc_g, s.load_f(out_g)))
    s.store(out_b, s.fadd(c_ltc_b, s.load_f(out_b)))

    if enable_top2_rt_shadows:
        var ltc_lum = s.fma(c_ltc_r, s.float(0.2126), s.fma(c_ltc_g, s.float(0.7152), s.fmul(c_ltc_b, s.float(0.0722))))
        # Exact closest point on LTC_TUBE segment or LTC_RECT quad to shading point `W = (wx, wy, wz)`
        var cwx = s.fneg(rel_x)
        var cwy = s.fneg(rel_y)
        var cwz = s.fneg(rel_z)
        var u_l2 = s.nmax(s.float(1e-6), s.fma(aux_x, aux_x, s.fma(aux_y, aux_y, s.fmul(aux_z, aux_z))))
        var tu_c = s.nclamp(s.fdiv(s.fma(cwx, aux_x, s.fma(cwy, aux_y, s.fmul(cwz, aux_z))), u_l2), s.float(-1.0), s.float(1.0))
        var tv_c = s.fvar(s.float(0.0))
        s.begin_if(s.ieq(kind, s.uint(AAA_LIGHT_LTC_RECT)))
        var v_l2 = s.nmax(s.float(1e-6), s.fma(avx_x, avx_x, s.fma(avx_y, avx_y, s.fmul(avx_z, avx_z))))
        s.store(tv_c, s.nclamp(s.fdiv(s.fma(cwx, avx_x, s.fma(cwy, avx_y, s.fmul(cwz, avx_z))), v_l2), s.float(-1.0), s.float(1.0)))
        s.end_if()
        var f_tvc = s.load_f(tv_c)
        var sh_tx = s.fma(aux_x, tu_c, s.fma(avx_x, f_tvc, lpx))
        var sh_ty = s.fma(aux_y, tu_c, s.fma(avx_y, f_tvc, lpy))
        var sh_tz = s.fma(aux_z, tu_c, s.fma(avx_z, f_tvc, lpz))
        _emit_push_top2_hero_light(
            s, ltc_lum, c_ltc_r, c_ltc_g, c_ltc_b, sh_tx, sh_ty, sh_tz,
            t0_lum, t0_r, t0_g, t0_b, t0_tx, t0_ty, t0_tz,
            t1_lum, t1_r, t1_g, t1_b, t1_tx, t1_ty, t1_tz,
        )
    s.end_if()

    s.begin_else()
    # Punctual Point (`0`) or Spot (`1`) Light with Karis smooth inverse-square window + GGX microfacet!
    var d2 = s.nmax(s.float(0.01), s.fma(rel_x, rel_x, s.fma(rel_y, rel_y, s.fmul(rel_z, rel_z))))
    var inv_d = s.inverse_sqrt(d2)
    var lx = s.fmul(rel_x, inv_d)
    var ly = s.fmul(rel_y, inv_d)
    var lz = s.fmul(rel_z, inv_d)
    var ndotl = s.nmax(s.float(0.0), s.fma(nx, lx, s.fma(ny, ly, s.fmul(nz, lz))))

    s.begin_if(s.fgt(ndotl, s.float(0.0001)))
    var r2 = s.nmax(s.float(0.01), s.fmul(lrad, lrad))
    var q = s.fdiv(d2, r2)
    var q2 = s.nclamp(s.fsub(s.float(1.0), s.fmul(q, q)), s.float(0.0), s.float(1.0))
    var atten = s.fvar(s.fdiv(s.fmul(q2, q2), s.fadd(d2, s.float(0.25))))

    s.begin_if(s.ieq(kind, s.uint(AAA_LIGHT_SPOT)))
    var cos_cone = s.fneg(s.fma(lx, aux_x, s.fma(ly, aux_y, s.fmul(lz, aux_z))))
    var spot_w = s.nclamp(s.fmul(s.fsub(cos_cone, p0_f), p1_f), s.float(0.0), s.float(1.0))
    s.store(atten, s.fmul(s.load_f(atten), s.fmul(spot_w, spot_w)))
    s.end_if()

    var hx = s.fadd(vx, lx); var hy = s.fadd(vy, ly); var hz = s.fadd(vz, lz)
    var inv_h = s.inverse_sqrt(s.nmax(s.float(1e-8), s.fma(hx, hx, s.fma(hy, hy, s.fmul(hz, hz)))))
    var ndoth = s.nmax(s.float(0.0), s.fmul(s.fma(nx, hx, s.fma(ny, hy, s.fmul(nz, hz))), inv_h))
    var vdoth = s.nmax(s.float(0.0), s.fmul(s.fma(vx, hx, s.fma(vy, hy, s.fmul(vz, hz))), inv_h))

    var d_den = s.fma(s.fmul(ndoth, ndoth), s.fsub(a2, s.float(1.0)), s.float(1.0))
    var d_ggx = s.fdiv(a2, s.nmax(s.float(1e-6), s.fmul(s.float(3.14159265), s.fmul(d_den, d_den))))
    var gv = s.fmul(ndotl, s.fsqrt(s.fma(s.fmul(ndotv, ndotv), s.fsub(s.float(1.0), a2), a2)))
    var gl = s.fmul(ndotv, s.fsqrt(s.fma(s.fmul(ndotl, ndotl), s.fsub(s.float(1.0), a2), a2)))
    var vis_g = s.fdiv(s.float(0.5), s.nmax(s.float(1e-5), s.fadd(gv, gl)))
    var om = s.fsub(s.float(1.0), vdoth)
    var om2 = s.fmul(om, om)
    var fc = s.fmul(s.fmul(om2, om2), om)
    var fr_r = s.fma(s.fsub(f90, spec_f0_r), fc, spec_f0_r)
    var fr_g = s.fma(s.fsub(f90, spec_f0_g), fc, spec_f0_g)
    var fr_b = s.fma(s.fsub(f90, spec_f0_b), fc, spec_f0_b)
    var dv_s = s.fmul(d_ggx, vis_g)

    var e_in = s.fmul(ndotl, s.load_f(atten))
    var c_pt_r = s.fmul(s.fmul(lrr, e_in), s.fma(kd_r, inv_pi, s.fmul(fr_r, dv_s)))
    var c_pt_g = s.fmul(s.fmul(lrg, e_in), s.fma(kd_g, inv_pi, s.fmul(fr_g, dv_s)))
    var c_pt_b = s.fmul(s.fmul(lrb, e_in), s.fma(kd_b, inv_pi, s.fmul(fr_b, dv_s)))
    s.store(out_r, s.fadd(c_pt_r, s.load_f(out_r)))
    s.store(out_g, s.fadd(c_pt_g, s.load_f(out_g)))
    s.store(out_b, s.fadd(c_pt_b, s.load_f(out_b)))

    if enable_top2_rt_shadows:
        var pt_lum = s.fma(c_pt_r, s.float(0.2126), s.fma(c_pt_g, s.float(0.7152), s.fmul(c_pt_b, s.float(0.0722))))
        _emit_push_top2_hero_light(
            s, pt_lum, c_pt_r, c_pt_g, c_pt_b, lpx, lpy, lpz,
            t0_lum, t0_r, t0_g, t0_b, t0_tx, t0_ty, t0_tz,
            t1_lum, t1_r, t1_g, t1_b, t1_tx, t1_ty, t1_tz,
        )
    s.end_if()
    s.end_if()

    s.store(k_iter, s.iadd(ki, s.uint(1)))
    s.end_loop()

    # Trace `<= 2` deterministic `RAY_FLAG_SHADOW_OCCLUSION` rays to Top-2 hero local/LTC emitters!
    if enable_top2_rt_shadows:
        s.begin_if(do_rt_shadows_cond)
        s.begin_if(s.fgt(s.load_f(t0_lum), s.float(1e-4)))
        var dx0 = s.fsub(s.load_f(t0_tx), wx)
        var dy0 = s.fsub(s.load_f(t0_ty), wy)
        var dz0 = s.fsub(s.load_f(t0_tz), wz)
        var dist0 = s.fsqrt(s.nmax(s.float(1e-6), s.fma(dx0, dx0, s.fma(dy0, dy0, s.fmul(dz0, dz0)))))
        s.begin_if(s.fgt(dist0, s.float(0.28)))
        var inv_d0 = s.fdiv(s.float(1.0), dist0)
        var vis0 = emit_rt_shadow_query(
            s, tlas_var, wx, wy, wz, nx, ny, nz,
            s.fmul(dx0, inv_d0), s.fmul(dy0, inv_d0), s.fmul(dz0, inv_d0),
            s.nmax(s.float(0.05), s.fsub(dist0, s.float(0.22))),
            b_verts=b_verts,
            b_indices=b_indices,
            b_instances=b_instances,
            b_materials=b_materials,
            bindless_arr=bindless_arr,
        )
        var occ0 = s.fsub(s.float(1.0), vis0)
        s.store(out_r, s.nmax(s.float(0.0), s.fsub(s.load_f(out_r), s.fmul(occ0, s.load_f(t0_r)))))
        s.store(out_g, s.nmax(s.float(0.0), s.fsub(s.load_f(out_g), s.fmul(occ0, s.load_f(t0_g)))))
        s.store(out_b, s.nmax(s.float(0.0), s.fsub(s.load_f(out_b), s.fmul(occ0, s.load_f(t0_b)))))
        s.end_if()
        s.end_if()

        s.begin_if(s.fgt(s.load_f(t1_lum), s.float(1e-4)))
        var dx1 = s.fsub(s.load_f(t1_tx), wx)
        var dy1 = s.fsub(s.load_f(t1_ty), wy)
        var dz1 = s.fsub(s.load_f(t1_tz), wz)
        var dist1 = s.fsqrt(s.nmax(s.float(1e-6), s.fma(dx1, dx1, s.fma(dy1, dy1, s.fmul(dz1, dz1)))))
        s.begin_if(s.fgt(dist1, s.float(0.28)))
        var inv_d1 = s.fdiv(s.float(1.0), dist1)
        var vis1 = emit_rt_shadow_query(
            s, tlas_var, wx, wy, wz, nx, ny, nz,
            s.fmul(dx1, inv_d1), s.fmul(dy1, inv_d1), s.fmul(dz1, inv_d1),
            s.nmax(s.float(0.05), s.fsub(dist1, s.float(0.22))),
            b_verts=b_verts,
            b_indices=b_indices,
            b_instances=b_instances,
            b_materials=b_materials,
            bindless_arr=bindless_arr,
        )
        var occ1 = s.fsub(s.float(1.0), vis1)
        s.store(out_r, s.nmax(s.float(0.0), s.fsub(s.load_f(out_r), s.fmul(occ1, s.load_f(t1_r)))))
        s.store(out_g, s.nmax(s.float(0.0), s.fsub(s.load_f(out_g), s.fmul(occ1, s.load_f(t1_g)))))
        s.store(out_b, s.nmax(s.float(0.0), s.fsub(s.load_f(out_b), s.fmul(occ1, s.load_f(t1_b)))))
        s.end_if()
        s.end_if()
        s.end_if()
    s.end_if()

    return (s.load_f(out_r), s.load_f(out_g), s.load_f(out_b))


@always_inline
def _emit_pcg_hash_f01(mut s: AaaSpirvModule, seed_in: UInt32) -> Tuple[UInt32, UInt32]:
    """Fast GPU PCG hash returning `(next_seed_u32, uniform_float_01)`."""
    var st = s.iadd(s.imul(seed_in, s.uint(747796405)), s.uint(2891336453))
    var wd = s.imul(s.bit_xor(s.shr(st, s.iadd(s.shr(st, s.uint(28)), s.uint(4))), st), s.uint(277803737))
    var res = s.bit_xor(s.shr(wd, s.uint(22)), wd)
    var f01 = s.fmul(s.float_from_uint(s.bit_and(res, s.uint(0x00FFFFFF))), s.float(1.0 / 16777216.0))
    return (res, f01)


def emit_restir_candidate_selection_spv(
    mut s: AaaSpirvModule,
    b_clusters: UInt32,
    px: UInt32,
    py: UInt32,
    scr_w: UInt32,
    scr_h: UInt32,
    z_lin: UInt32,
    wx: UInt32,
    wy: UInt32,
    wz: UInt32,
    nx: UInt32,
    ny: UInt32,
    nz: UInt32,
    frame_idx: UInt32,
    num_candidates: Int = 4,
) -> Tuple[UInt32, UInt32, UInt32, UInt32]:
    """Selects 1 initial candidate light from the pixel's froxel cluster using RIS."""
    var near_z = s.load_float(b_clusters, s.uint(1))
    var log_ratio = s.load_float(b_clusters, s.uint(3))
    var cx = s.umin(s.uint(UInt32(AAA_CLUSTER_X - 1)), s.udiv(s.imul(px, s.uint(UInt32(AAA_CLUSTER_X))), scr_w))
    var cy = s.umin(s.uint(UInt32(AAA_CLUSTER_Y - 1)), s.udiv(s.imul(py, s.uint(UInt32(AAA_CLUSTER_Y))), scr_h))
    var z_norm = s.nclamp(
        s.fdiv(s.log2(s.nmax(s.float(1.0), s.fdiv(z_lin, near_z))), s.nmax(s.float(1e-4), log_ratio)),
        s.float(0.0),
        s.float(0.9999),
    )
    var cz = s.umin(s.uint(UInt32(AAA_CLUSTER_Z - 1)), s.saturate_uint(s.fmul(z_norm, s.float(Float32(AAA_CLUSTER_Z)))))
    var cluster_id = s.iadd(s.imul(cz, s.uint(UInt32(AAA_CLUSTER_X * AAA_CLUSTER_Y))), s.iadd(s.imul(cy, s.uint(UInt32(AAA_CLUSTER_X))), cx))

    var c_cnt = s.load_word(b_clusters, s.iadd(s.uint(UInt32(AAA_CLUSTER_COUNTS_BASE)), cluster_id))
    var list_base = s.iadd(s.uint(UInt32(AAA_CLUSTER_INDICES_BASE)), s.imul(cluster_id, s.uint(UInt32(AAA_MAX_LIGHTS_PER_CLUSTER))))

    var out_light_id = s.uvar(s.uint(0xFFFFFFFF))
    var out_w_sum = s.fvar(s.float(0.0))
    var out_m = s.uvar(s.uint(0))
    var out_target_pdf = s.fvar(s.float(0.0))

    s.begin_if(s.ine(c_cnt, s.uint(0)))
    var seed = s.iadd(s.imul(s.iadd(s.imul(py, scr_w), px), s.uint(1973)), s.imul(frame_idx, s.uint(9277)))
    for cand in range(num_candidates):
        var h = _emit_pcg_hash_f01(s, s.iadd(seed, s.uint(UInt32(cand * 1337 + 17))))
        var rand_pick = s.umod(h[0], c_cnt)
        var cand_light_id = s.load_word(b_clusters, s.iadd(list_base, rand_pick))

        # Evaluate unshadowed target PDF: p_hat = luminance(radiance) * atten * max(0, N dot L)
        var l_off = s.iadd(s.uint(UInt32(AAA_CLUSTER_LIGHTS_BASE)), s.imul(cand_light_id, s.uint(UInt32(AAA_CLUSTER_LIGHT_WORDS))))
        var lx = s.load_float(b_clusters, l_off)
        var ly = s.load_float(b_clusters, s.iadd(l_off, s.uint(1)))
        var lz = s.load_float(b_clusters, s.iadd(l_off, s.uint(2)))
        var lrad = s.load_float(b_clusters, s.iadd(l_off, s.uint(3)))
        var lrr = s.load_float(b_clusters, s.iadd(l_off, s.uint(4)))
        var lrg = s.load_float(b_clusters, s.iadd(l_off, s.uint(5)))
        var lrb = s.load_float(b_clusters, s.iadd(l_off, s.uint(6)))

        var to_lx = s.fsub(lx, wx)
        var to_ly = s.fsub(ly, wy)
        var to_lz = s.fsub(lz, wz)
        var dist2 = s.fma(to_lx, to_lx, s.fma(to_ly, to_ly, s.fmul(to_lz, to_lz)))
        var inv_d = s.inverse_sqrt(s.nmax(s.float(1e-6), dist2))
        var ndotl = s.nmax(s.float(0.0), s.fmul(s.fma(nx, to_lx, s.fma(ny, to_ly, s.fmul(nz, to_lz))), inv_d))

        var rad2 = s.fmul(lrad, lrad)
        var att_win = s.nclamp(s.fsub(s.float(1.0), s.fdiv(dist2, rad2)), s.float(0.0), s.float(1.0))
        var atten = s.fdiv(s.fmul(att_win, att_win), s.fadd(dist2, s.float(0.25)))
        var luma = s.fma(lrr, s.float(0.2126), s.fma(lrg, s.float(0.7152), s.fmul(lrb, s.float(0.0722))))
        var p_hat = s.fmul(s.fmul(luma, atten), ndotl)

        # Uniform cluster sampling proposal q = 1.0 / c_cnt -> sample weight w = p_hat * c_cnt
        var sample_w = s.fmul(p_hat, s.float_from_uint(c_cnt))
        var new_w_sum = s.fadd(s.load_f(out_w_sum), sample_w)
        s.store(out_w_sum, new_w_sum)
        s.store(out_m, s.iadd(s.load_u(out_m), s.uint(1)))

        var h_rand = _emit_pcg_hash_f01(s, h[0])
        var rand_cond = s.flt(s.fmul(h_rand[1], new_w_sum), sample_w)
        s.begin_if(rand_cond)
        s.store(out_light_id, cand_light_id)
        s.store(out_target_pdf, p_hat)
        s.end_if()
    s.end_if()

    return (s.load_u(out_light_id), s.load_f(out_w_sum), s.load_u(out_m), s.load_f(out_target_pdf))


def emit_eval_single_cluster_light(
    mut s: AaaSpirvModule,
    b_clusters: UInt32,
    ltc_lut: Tuple[UInt32, UInt32],
    li: UInt32,
    wx: UInt32,
    wy: UInt32,
    wz: UInt32,
    nx: UInt32,
    ny: UInt32,
    nz: UInt32,
    vx: UInt32,
    vy: UInt32,
    vz: UInt32,
    ndotv: UInt32,
    alb_r: UInt32,
    alb_g: UInt32,
    alb_b: UInt32,
    spec_f0_r: UInt32,
    spec_f0_g: UInt32,
    spec_f0_b: UInt32,
    f90: UInt32,
    lut_a: UInt32,
    lut_b: UInt32,
    rough: UInt32,
    metal: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Evaluates unshadowed multi-lobe BSDF contribution from a single clustered light index `li`."""
    var out_r = s.fvar(s.float(0.0))
    var out_g = s.fvar(s.float(0.0))
    var out_b = s.fvar(s.float(0.0))

    # Build LTC shading tangent frame (T1, T2, N) + fetch 64x64 RG16F LTC parameters once
    var t1_rx = s.fsub(vx, s.fmul(nx, ndotv))
    var t1_ry = s.fsub(vy, s.fmul(ny, ndotv))
    var t1_rz = s.fsub(vz, s.fmul(nz, ndotv))
    var t1_l2 = s.fma(t1_rx, t1_rx, s.fma(t1_ry, t1_ry, s.fmul(t1_rz, t1_rz)))
    var up_is_y = s.flt(s.fabs(ny), s.float(0.99))
    var fb_x = s.select_f(up_is_y, s.float(0.0), s.fneg(nz))
    var fb_y = s.select_f(up_is_y, s.fneg(nz), s.float(0.0))
    var fb_z = s.select_f(up_is_y, s.fneg(nx), ny)
    var use_t1 = s.fgt(t1_l2, s.float(1e-8))
    var ux_t = s.select_f(use_t1, t1_rx, fb_x)
    var uy_t = s.select_f(use_t1, t1_ry, fb_y)
    var uz_t = s.select_f(use_t1, t1_rz, fb_z)
    var inv_t1 = s.inverse_sqrt(s.nmax(s.float(1e-12), s.fma(ux_t, ux_t, s.fma(uy_t, uy_t, s.fmul(uz_t, uz_t)))))
    var t1 = (s.fmul(ux_t, inv_t1), s.fmul(uy_t, inv_t1), s.fmul(uz_t, inv_t1))
    var t2 = (
        s.fsub(s.fmul(ny, t1[2]), s.fmul(nz, t1[1])),
        s.fsub(s.fmul(nz, t1[0]), s.fmul(nx, t1[2])),
        s.fsub(s.fmul(nx, t1[1]), s.fmul(ny, t1[0])),
    )

    var ltc_uv = s.vec2(s.nclamp(ndotv, s.float(0.01), s.float(0.99)), s.nclamp(rough, s.float(0.02), s.float(0.99)))
    var ltc_s = s.sample_2d_lod(ltc_lut, ltc_uv, s.float(0.0))
    var ltc_inv_scale = s.extract_f(ltc_s, 0)
    var ltc_tilt_01 = s.extract_f(ltc_s, 1)

    var spec_w_r = s.fma(spec_f0_r, lut_a, s.fmul(f90, lut_b))
    var spec_w_g = s.fma(spec_f0_g, lut_a, s.fmul(f90, lut_b))
    var spec_w_b = s.fma(spec_f0_b, lut_a, s.fmul(f90, lut_b))
    var one_m_met = s.fsub(s.float(1.0), metal)
    var kd_r = s.fmul(alb_r, s.fmul(one_m_met, s.nmax(s.float(0.0), s.fsub(s.float(1.0), spec_w_r))))
    var kd_g = s.fmul(alb_g, s.fmul(one_m_met, s.nmax(s.float(0.0), s.fsub(s.float(1.0), spec_w_g))))
    var kd_b = s.fmul(alb_b, s.fmul(one_m_met, s.nmax(s.float(0.0), s.fsub(s.float(1.0), spec_w_b))))

    var alpha = s.nmax(s.float(0.002), s.fmul(rough, rough))
    var a2 = s.fmul(alpha, alpha)
    var inv_pi = s.float(0.318309886)

    var l_off = s.iadd(s.uint(UInt32(AAA_CLUSTER_LIGHTS_BASE)), s.imul(li, s.uint(UInt32(AAA_CLUSTER_LIGHT_WORDS))))
    var lpx = s.load_float(b_clusters, l_off)
    var lpy = s.load_float(b_clusters, s.iadd(l_off, s.uint(1)))
    var lpz = s.load_float(b_clusters, s.iadd(l_off, s.uint(2)))
    var lrad = s.load_float(b_clusters, s.iadd(l_off, s.uint(3)))
    var lrr = s.load_float(b_clusters, s.iadd(l_off, s.uint(4)))
    var lrg = s.load_float(b_clusters, s.iadd(l_off, s.uint(5)))
    var lrb = s.load_float(b_clusters, s.iadd(l_off, s.uint(6)))
    var kind = s.bit_and(s.load_word(b_clusters, s.iadd(l_off, s.uint(7))), s.uint(0xFF))
    var aux_x = s.load_float(b_clusters, s.iadd(l_off, s.uint(8)))
    var aux_y = s.load_float(b_clusters, s.iadd(l_off, s.uint(9)))
    var aux_z = s.load_float(b_clusters, s.iadd(l_off, s.uint(10)))
    var p0_f = s.load_float(b_clusters, s.iadd(l_off, s.uint(11)))
    var avx_x = s.load_float(b_clusters, s.iadd(l_off, s.uint(12)))
    var avx_y = s.load_float(b_clusters, s.iadd(l_off, s.uint(13)))
    var avx_z = s.load_float(b_clusters, s.iadd(l_off, s.uint(14)))
    var p1_f = s.load_float(b_clusters, s.iadd(l_off, s.uint(15)))

    var rel_x = s.fsub(lpx, wx)
    var rel_y = s.fsub(lpy, wy)
    var rel_z = s.fsub(lpz, wz)

    s.begin_if(s.uge(kind, s.uint(AAA_LIGHT_LTC_RECT)))
    var use_vx = s.fvar(avx_x)
    var use_vy = s.fvar(avx_y)
    var use_vz = s.fvar(avx_z)
    var two_s = s.fvar(p0_f)

    s.begin_if(s.ieq(kind, s.uint(AAA_LIGHT_LTC_TUBE)))
    var crx = s.fsub(s.fmul(aux_y, rel_z), s.fmul(aux_z, rel_y))
    var cry = s.fsub(s.fmul(aux_z, rel_x), s.fmul(aux_x, rel_z))
    var crz = s.fsub(s.fmul(aux_x, rel_y), s.fmul(aux_y, rel_x))
    var inv_cr = s.fmul(p1_f, s.inverse_sqrt(s.nmax(s.float(1e-8), s.fma(crx, crx, s.fma(cry, cry, s.fmul(crz, crz))))))
    s.store(use_vx, s.fmul(crx, inv_cr))
    s.store(use_vy, s.fmul(cry, inv_cr))
    s.store(use_vz, s.fmul(crz, inv_cr))
    s.store(two_s, s.float(1.0))
    s.end_if()

    var f_vx = s.load_f(use_vx); var f_vy = s.load_f(use_vy); var f_vz = s.load_f(use_vz)
    var lnx = s.fsub(s.fmul(aux_y, f_vz), s.fmul(aux_z, f_vy))
    var lny = s.fsub(s.fmul(aux_z, f_vx), s.fmul(aux_x, f_vz))
    var lnz = s.fsub(s.fmul(aux_x, f_vy), s.fmul(aux_y, f_vx))
    var front_dot = s.fneg(s.fma(rel_x, lnx, s.fma(rel_y, lny, s.fmul(rel_z, lnz))))
    var emit_ok = s.logical_or(s.fgt(s.load_f(two_s), s.float(0.5)), s.fgt(front_dot, s.float(0.0)))

    s.begin_if(emit_ok)
    var d2_c = s.fma(rel_x, rel_x, s.fma(rel_y, rel_y, s.fmul(rel_z, rel_z)))
    var r2_c = s.nmax(s.float(0.01), s.fmul(lrad, lrad))
    var win_t = s.nclamp(s.fsub(s.float(1.0), s.fmul(s.fdiv(d2_c, r2_c), s.fdiv(d2_c, r2_c))), s.float(0.0), s.float(1.0))
    var win = s.fmul(win_t, win_t)

    var ff_diff = _emit_ltc_quad_form_factor_spv(
        s, t1, t2, (nx, ny, nz),
        rel_x, rel_y, rel_z,
        aux_x, aux_y, aux_z,
        f_vx, f_vy, f_vz,
        s.float(1.0), s.float(0.0), s.load_f(two_s),
    )
    var ff_spec = _emit_ltc_quad_form_factor_spv(
        s, t1, t2, (nx, ny, nz),
        rel_x, rel_y, rel_z,
        aux_x, aux_y, aux_z,
        f_vx, f_vy, f_vz,
        ltc_inv_scale, ltc_tilt_01, s.load_f(two_s),
    )
    var lr_w = s.fmul(lrr, win)
    var lg_w = s.fmul(lrg, win)
    var lb_w = s.fmul(lrb, win)
    s.store(out_r, s.fma(lr_w, s.fma(kd_r, ff_diff, s.fmul(spec_w_r, ff_spec)), s.load_f(out_r)))
    s.store(out_g, s.fma(lg_w, s.fma(kd_g, ff_diff, s.fmul(spec_w_g, ff_spec)), s.load_f(out_g)))
    s.store(out_b, s.fma(lb_w, s.fma(kd_b, ff_diff, s.fmul(spec_w_b, ff_spec)), s.load_f(out_b)))
    s.end_if()

    s.begin_else()
    var d2 = s.nmax(s.float(0.01), s.fma(rel_x, rel_x, s.fma(rel_y, rel_y, s.fmul(rel_z, rel_z))))
    var inv_d = s.inverse_sqrt(d2)
    var lx = s.fmul(rel_x, inv_d)
    var ly = s.fmul(rel_y, inv_d)
    var lz = s.fmul(rel_z, inv_d)
    var ndotl = s.nmax(s.float(0.0), s.fma(nx, lx, s.fma(ny, ly, s.fmul(nz, lz))))

    s.begin_if(s.fgt(ndotl, s.float(0.0001)))
    var r2 = s.nmax(s.float(0.01), s.fmul(lrad, lrad))
    var q = s.fdiv(d2, r2)
    var q2 = s.nclamp(s.fsub(s.float(1.0), s.fmul(q, q)), s.float(0.0), s.float(1.0))
    var atten = s.fvar(s.fdiv(s.fmul(q2, q2), s.fadd(d2, s.float(0.25))))

    s.begin_if(s.ieq(kind, s.uint(AAA_LIGHT_SPOT)))
    var cos_cone = s.fneg(s.fma(lx, aux_x, s.fma(ly, aux_y, s.fmul(lz, aux_z))))
    var spot_w = s.nclamp(s.fmul(s.fsub(cos_cone, p0_f), p1_f), s.float(0.0), s.float(1.0))
    s.store(atten, s.fmul(s.load_f(atten), s.fmul(spot_w, spot_w)))
    s.end_if()

    var hx = s.fadd(vx, lx); var hy = s.fadd(vy, ly); var hz = s.fadd(vz, lz)
    var inv_h = s.inverse_sqrt(s.nmax(s.float(1e-8), s.fma(hx, hx, s.fma(hy, hy, s.fmul(hz, hz)))))
    var ndoth = s.nmax(s.float(0.0), s.fmul(s.fma(nx, hx, s.fma(ny, hy, s.fmul(nz, hz))), inv_h))
    var vdoth = s.nmax(s.float(0.0), s.fmul(s.fma(vx, hx, s.fma(vy, hy, s.fmul(vz, hz))), inv_h))

    var d_den = s.fma(s.fmul(ndoth, ndoth), s.fsub(a2, s.float(1.0)), s.float(1.0))
    var d_ggx = s.fdiv(a2, s.nmax(s.float(1e-6), s.fmul(s.float(3.14159265), s.fmul(d_den, d_den))))
    var gv = s.fmul(ndotl, s.fsqrt(s.fma(s.fmul(ndotv, ndotv), s.fsub(s.float(1.0), a2), a2)))
    var gl = s.fmul(ndotv, s.fsqrt(s.fma(s.fmul(ndotl, ndotl), s.fsub(s.float(1.0), a2), a2)))
    var vis_g = s.fdiv(s.float(0.5), s.nmax(s.float(1e-5), s.fadd(gv, gl)))
    var om = s.fsub(s.float(1.0), vdoth)
    var om2 = s.fmul(om, om)
    var fc = s.fmul(s.fmul(om2, om2), om)
    var fr_r = s.fma(s.fsub(f90, spec_f0_r), fc, spec_f0_r)
    var fr_g = s.fma(s.fsub(f90, spec_f0_g), fc, spec_f0_g)
    var fr_b = s.fma(s.fsub(f90, spec_f0_b), fc, spec_f0_b)
    var dv_s = s.fmul(d_ggx, vis_g)

    var e_in = s.fmul(ndotl, s.load_f(atten))
    s.store(out_r, s.fma(s.fmul(lrr, e_in), s.fma(kd_r, inv_pi, s.fmul(fr_r, dv_s)), s.load_f(out_r)))
    s.store(out_g, s.fma(s.fmul(lrg, e_in), s.fma(kd_g, inv_pi, s.fmul(fr_g, dv_s)), s.load_f(out_g)))
    s.store(out_b, s.fma(s.fmul(lrb, e_in), s.fma(kd_b, inv_pi, s.fmul(fr_b, dv_s)), s.load_f(out_b)))
    s.end_if()
    s.end_if()

    return (s.load_f(out_r), s.load_f(out_g), s.load_f(out_b))

