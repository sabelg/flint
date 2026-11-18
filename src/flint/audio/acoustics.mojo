"""Bvh8 Raycast Spatial Acoustics, Portal Diffraction & Sabine RT60 Probe (`flint.audio.acoustics`).

Provides geometry-driven 3D acoustic propagation over `Bvh8MeshCollider`:
  - `AcousticMaterial` & `AcousticMaterialTable` (frequency-dependent absorption & transmission loss)
  - Multi-hit direct `Bvh8.raycast` transmission loss (`occlusion_gain`, `occlusion_cutoff_hz`)
  - 4-ray orthogonal aperture portal fan distinguishing narrow pillar `obstruction_factor`
    from sealed-room `room_occlusion_factor`
  - 6-ray listener room-enclosure Sabine `RT60 = 0.161 * V / A` probe driving FDN reverb decay
"""
from std.collections import List
from std.math import sqrt, exp, abs
from flint.core.math import Vec3
from flint.physics.bvh import Bvh8MeshCollider

comptime MAT_STRUCTURAL_DEFAULT: UInt16 = 0
comptime MAT_CONCRETE: UInt16 = 1
comptime MAT_METAL: UInt16 = 2
comptime MAT_WOOD: UInt16 = 3
comptime MAT_GLASS: UInt16 = 4
comptime MAT_CARPET: UInt16 = 5
comptime MAT_ROCK: UInt16 = 6
comptime MAT_FOLIAGE: UInt16 = 7

comptime CLEAR_CUTOFF_HZ: Float32 = 20000.0


@fieldwise_init
struct AcousticMaterial(Copyable, ImplicitlyCopyable, Movable):
    """Acoustic surface properties for raycast transmission loss and Sabine reverb."""
    var id: UInt16
    var absorption_low: Float32
    var absorption_mid: Float32
    var absorption_high: Float32
    var transmission_loss_db: Float32
    var scattering: Float32

    def __init__(out self):
        self.id = MAT_STRUCTURAL_DEFAULT
        self.absorption_low = 0.05
        self.absorption_mid = 0.08
        self.absorption_high = 0.12
        self.transmission_loss_db = 24.0
        self.scattering = 0.15


struct AcousticMaterialTable(Copyable, Movable):
    """Lookup table mapping triangle `material_id` (`UInt16`) to `AcousticMaterial`."""
    var materials: List[AcousticMaterial]

    def __init__(out self):
        self.materials = List[AcousticMaterial]()
        # 0: Default structural wall (concrete plaster)
        self.materials.append(AcousticMaterial(MAT_STRUCTURAL_DEFAULT, 0.05, 0.08, 0.12, 24.0, 0.15))
        # 1: Dense poured concrete
        self.materials.append(AcousticMaterial(MAT_CONCRETE, 0.03, 0.06, 0.09, 28.0, 0.12))
        # 2: Steel / bulkhead hull
        self.materials.append(AcousticMaterial(MAT_METAL, 0.02, 0.04, 0.06, 32.0, 0.08))
        # 3: Timber / interior wood panel
        self.materials.append(AcousticMaterial(MAT_WOOD, 0.12, 0.15, 0.18, 14.0, 0.22))
        # 4: Architectural glass
        self.materials.append(AcousticMaterial(MAT_GLASS, 0.18, 0.08, 0.05, 10.0, 0.06))
        # 5: Heavy acoustic carpet / fabric
        self.materials.append(AcousticMaterial(MAT_CARPET, 0.15, 0.52, 0.72, 9.0, 0.45))
        # 6: Cavern / stone rockface
        self.materials.append(AcousticMaterial(MAT_ROCK, 0.04, 0.07, 0.11, 30.0, 0.55))
        # 7: Foliage / hedge canopy
        self.materials.append(AcousticMaterial(MAT_FOLIAGE, 0.10, 0.38, 0.62, 5.0, 0.75))

    def get(self, mat_id: UInt16) -> AcousticMaterial:
        var idx = Int(mat_id)
        if idx >= 0 and idx < len(self.materials):
            return self.materials[idx]
        return self.materials[0]

    def set_material(mut self, mat_id: UInt16, mat: AcousticMaterial):
        var idx = Int(mat_id)
        while len(self.materials) <= idx:
            self.materials.append(AcousticMaterial())
        self.materials[idx] = mat


@fieldwise_init
struct AcousticOcclusionResult(Copyable, ImplicitlyCopyable, Movable):
    """Result of direct + 4-ray aperture portal fan acoustic raycast."""
    var direct_blocked: Bool
    var wall_hits: Int
    var fan_open_rays: Int
    var portal_openness: Float32
    var obstruction_factor: Float32
    var room_occlusion_factor: Float32
    var transmission_loss_db: Float32
    var occlusion_gain: Float32
    var occlusion_cutoff_hz: Float32

    def __init__(out self):
        self.direct_blocked = False
        self.wall_hits = 0
        self.fan_open_rays = 4
        self.portal_openness = 1.0
        self.obstruction_factor = 0.0
        self.room_occlusion_factor = 0.0
        self.transmission_loss_db = 0.0
        self.occlusion_gain = 1.0
        self.occlusion_cutoff_hz = CLEAR_CUTOFF_HZ


@fieldwise_init
struct RoomProbeResult(Copyable, ImplicitlyCopyable, Movable):
    """Result of 6-ray listener enclosure probe and Sabine `RT60` solver."""
    var hit_count: Int
    var enclosure_factor: Float32
    var mean_distance_m: Float32
    var estimated_volume_m3: Float32
    var sabine_rt60_sec: Float32
    var fdn_feedback: Float32
    var wet_send: Float32

    def __init__(out self):
        self.hit_count = 0
        self.enclosure_factor = 0.0
        self.mean_distance_m = 30.0
        self.estimated_volume_m3 = 0.0
        self.sabine_rt60_sec = 0.18
        self.fdn_feedback = 0.45
        self.wet_send = 0.06


@always_inline
def _db_to_linear(db: Float32) -> Float32:
    # 10^(db / 20) = exp(db * ln(10) / 20)
    return exp(db * Float32(0.11512925465))


@always_inline
def _orthonormal_basis(n: Vec3) -> Tuple[Vec3, Vec3]:
    var helper = Vec3(0.0, 1.0, 0.0) if abs(n.y) < Float32(0.9) else Vec3(1.0, 0.0, 0.0)
    var u = n.cross(helper).normalized()
    var v = n.cross(u).normalized()
    return (u, v)


def evaluate_emitter_acoustics(
    bvh: Bvh8MeshCollider,
    materials: AcousticMaterialTable,
    listener_pos: Vec3,
    emitter_pos: Vec3,
    aperture_radius: Float32 = 1.15,
) -> AcousticOcclusionResult:
    """Evaluates direct `Bvh8` multi-surface transmission loss and a 4-ray orthogonal
    aperture fan around the line-of-sight midpoint to distinguish narrow pillar
    obstruction from sealed-room wall occlusion."""
    var res = AcousticOcclusionResult()
    var delta = emitter_pos - listener_pos
    var dist = delta.length()
    if dist <= Float32(0.08) or len(bvh.nodes) == 0 or bvh.root_node < 0:
        return res^

    var inv_d = Float32(1.0) / dist
    var dir = delta * inv_d
    var eps = Float32(0.025)

    # 1. Step along direct line-of-sight up to 4 walls to accumulate transmission loss
    var cur_orig = listener_pos + dir * eps
    var remaining = dist - Float32(2.0) * eps
    var total_tl_db = Float32(0.0)
    var hits = 0

    for _ in range(4):
        if remaining <= Float32(0.02):
            break
        var hit = bvh.raycast(cur_orig, dir, remaining)
        if not hit.hit:
            break
        hits += 1
        var mat = materials.get(hit.material_id)
        total_tl_db += mat.transmission_loss_db
        var advance = hit.t + eps
        cur_orig = cur_orig + dir * advance
        remaining -= advance

    res.direct_blocked = hits > 0
    res.wall_hits = hits

    # 2. 4-ray aperture portal fan via midpoint lateral probes:
    #    listener -> mid_offset -> emitter tests diffraction paths around columns/doorframes.
    var basis = _orthonormal_basis(dir)
    var u = basis[0]
    var v = basis[1]
    var mid = (listener_pos + emitter_pos) * Float32(0.5)
    var rad = min(aperture_radius, dist * Float32(0.45))

    var offsets = Array[Vec3, 4](fill=Vec3())
    offsets[0] = u * rad
    offsets[1] = u * (-rad)
    offsets[2] = v * rad
    offsets[3] = v * (-rad)

    var open_fan = 0
    for k in range(4):
        var waypoint = mid + offsets[k]
        var seg1 = waypoint - listener_pos
        var len1 = seg1.length()
        var seg2 = emitter_pos - waypoint
        var len2 = seg2.length()
        if len1 <= eps * Float32(2.0) or len2 <= eps * Float32(2.0):
            open_fan += 1
            continue
        var d1 = seg1 * (Float32(1.0) / len1)
        var d2 = seg2 * (Float32(1.0) / len2)
        # Overlap across `waypoint` by `eps` so a wall lying directly on the midpoint plane cannot slip between segments
        var h1 = bvh.raycast(listener_pos + d1 * eps, d1, len1)
        if not h1.hit:
            var h2 = bvh.raycast(waypoint - d2 * eps, d2, len2)
            if not h2.hit:
                open_fan += 1

    res.fan_open_rays = open_fan
    res.portal_openness = Float32(open_fan) * Float32(0.25)

    if not res.direct_blocked:
        # Direct line of sight is clear, though partial grazing near portals adds subtle HF shadow
        var grazing_block = Float32(1.0) - res.portal_openness
        res.obstruction_factor = grazing_block * Float32(0.35)
        res.room_occlusion_factor = 0.0
        res.transmission_loss_db = grazing_block * Float32(2.0)
        res.occlusion_gain = _db_to_linear(-res.transmission_loss_db)
        res.occlusion_cutoff_hz = CLEAR_CUTOFF_HZ - grazing_block * Float32(3500.0)
        return res^

    # Direct line of sight is blocked:
    # - High portal_openness (e.g. 3..4 fan rays open) => narrow pillar OBSTRUCTION (diffraction wraps around)
    # - Zero portal_openness (0 fan rays open) => sealed room OCCLUSION (full wall transmission loss)
    res.obstruction_factor = res.portal_openness
    res.room_occlusion_factor = Float32(1.0) - res.portal_openness

    # Effective dB loss blends diffracted pillar shadow (~6-9 dB) vs full wall transmission loss (24+ dB)
    var diffracted_db = min(total_tl_db, Float32(7.5))
    var eff_db = diffracted_db * res.portal_openness + total_tl_db * res.room_occlusion_factor
    res.transmission_loss_db = eff_db
    res.occlusion_gain = max(Float32(0.01), _db_to_linear(-eff_db))

    # Low-pass cutoff: 20 kHz clear -> ~2.4 kHz behind thin pillar -> < 500 Hz behind sealed concrete wall
    var sealed_cutoff = max(Float32(220.0), CLEAR_CUTOFF_HZ * exp(-total_tl_db * Float32(0.165)))
    var pillar_cutoff = Float32(2400.0)
    res.occlusion_cutoff_hz = pillar_cutoff * res.portal_openness + sealed_cutoff * res.room_occlusion_factor
    return res^


def probe_room_sabine_rt60(
    bvh: Bvh8MeshCollider,
    materials: AcousticMaterialTable,
    listener_pos: Vec3,
    max_probe_dist: Float32 = 50.0,
) -> RoomProbeResult:
    """Casts 6 cardinal rays `(±X, ±Y, ±Z)` from `listener_pos` to estimate enclosure
    volume `V`, material-weighted absorption `A`, and Sabine `RT60 = 0.161 * V / A`."""
    var out = RoomProbeResult()
    if len(bvh.nodes) == 0 or bvh.root_node < 0:
        return out^

    var dirs = Array[Vec3, 6](fill=Vec3())
    dirs[0] = Vec3(1.0, 0.0, 0.0)
    dirs[1] = Vec3(-1.0, 0.0, 0.0)
    dirs[2] = Vec3(0.0, 1.0, 0.0)
    dirs[3] = Vec3(0.0, -1.0, 0.0)
    dirs[4] = Vec3(0.0, 0.0, 1.0)
    dirs[5] = Vec3(0.0, 0.0, -1.0)

    var dists = Array[Float32, 6](fill=max_probe_dist)
    var alphas = Array[Float32, 6](fill=1.0)  # Open sky = 100% absorption (anechoic escape)
    var hits = 0
    var dist_sum = Float32(0.0)

    for i in range(6):
        var h = bvh.raycast(listener_pos, dirs[i], max_probe_dist)
        if h.hit:
            hits += 1
            var d = max(Float32(0.35), h.t)
            dists[i] = d
            dist_sum += d
            var mat = materials.get(h.material_id)
            alphas[i] = max(Float32(0.02), min(Float32(0.99), mat.absorption_mid))
        else:
            dist_sum += max_probe_dist

    out.hit_count = hits
    out.enclosure_factor = Float32(hits) / Float32(6.0)
    out.mean_distance_m = dist_sum / Float32(6.0)

    if hits == 0:
        return out^

    var wx = max(Float32(1.0), dists[0] + dists[1])
    var wy = max(Float32(1.0), dists[2] + dists[3])
    var wz = max(Float32(1.0), dists[4] + dists[5])
    var vol = wx * wy * wz
    out.estimated_volume_m3 = vol

    var area_yz = wy * wz
    var area_xz = wx * wz
    var area_xy = wx * wy
    var total_absorption = (
        area_yz * (alphas[0] + alphas[1])
        + area_xz * (alphas[2] + alphas[3])
        + area_xy * (alphas[4] + alphas[5])
    )

    var rt60 = Float32(0.161) * vol / max(Float32(0.1), total_absorption)
    out.sabine_rt60_sec = max(Float32(0.08), min(Float32(6.5), rt60))

    # Mean FDN delay line length in StereoReverbFDN is ~2074 frames @ 48 kHz = 0.0432 s:
    # g = 10^(-3 * tau_mean / RT60)
    var tau_mean = Float32(0.0432)
    var fb = exp(Float32(-6.907755) * tau_mean / out.sabine_rt60_sec)
    out.fdn_feedback = max(Float32(0.25), min(Float32(0.93), fb))
    out.wet_send = Float32(0.05) + out.enclosure_factor * Float32(0.28)
    return out^
