"""`16`-Bone Hard-Surface SubD Mechs & `12`-Pose Articulated Humanoid Crowd Bank (`crowd_rig.mojo`).

Implements Flint Engine v3 character & crowd assets:
  1. `SkinnedMeshData`, `build_subd_cyber_enforcer_rigged()`, and `build_subd_cyber_enforcer_rigged_v3()`:
     `~3,100`-triangle hard-surface faceted SubD MaxTac `Cyber-Enforcer` sentinel rigged across `16`
     hierarchical bones (`Pelvis..Rifle`) with 3 synchronized material submeshes sharing the exact
     16-bone palette:
       - `0` (`mesh_armor`): Coated slate-blue ceramite carapace, layered samurai pauldrons, angular
         knee-guards & NCPD white/crimson tactical stripes (`colors`).
       - `1` (`mesh_chassis`): Dark titanium frame, mirror-chrome hydraulic knee/elbow rams, split-toe
         talons, and bullpup pulse-rifle chassis (`colors`).
       - `2` (`mesh_emissive`): HDR cyan T-visor + quad-optics `[24, 244, 255]`, amber chest arc-reactor
         `[255, 165, 36]`, red `[255, 28, 52]` & blue `[28, 132, 255]` pauldron police strobes, dorsal
         ion vents, pulse-rifle plasma rails, and riot-shield energy rim!
  2. `build_articulated_crowd_pose_bank(with_umbrella: Bool = False)`:
     `12`-pose pre-skinned SubD humanoid crowd bank (`8` walk-cycle phases + `2` idle/conversation +
     `2` cheering/umbrella-brace poses) with rich 24-bit per-vertex `colors` (skin, hair, indigo denim,
     shoes, and chrome-ribbed navy/crimson/teal umbrellas) & grounded contact `ao`.
"""
from std.math import cos, max, min, sin, sqrt
from flint.core.math import Vec3
from flint.core.transform import Mat4, Quat, trs
from flint.assets.model import NativeMesh, NativeModel, NativeNode, NativeSkin
from flint.assets.mesh_builder import MeshBuilder
from flint.procedural.subd import SubdCage, subdivide_to_mesh

comptime _TAU_RIG: Float32 = 6.28318530
comptime ENFORCER_BONE_COUNT: Int = 16


@always_inline
def _quat_euler_xyz(rx: Float32, ry: Float32, rz: Float32) -> Quat:
    var hx = rx * Float32(0.5)
    var hy = ry * Float32(0.5)
    var hz = rz * Float32(0.5)
    var qx = Quat(sin(hx), 0.0, 0.0, cos(hx))
    var qy = Quat(0.0, sin(hy), 0.0, cos(hy))
    var qz = Quat(0.0, 0.0, sin(hz), cos(hz))
    return qy.multiply(qx).multiply(qz).normalized()


struct _BoneDQ(Copyable, ImplicitlyCopyable, Movable):
    """Internal volume-preserving unit Dual-Quaternion `(q_r, q_d)` for zero-collapse CPU pose baking."""
    var rx: Float32; var ry: Float32; var rz: Float32; var rw: Float32
    var dx: Float32; var dy: Float32; var dz: Float32; var dw: Float32

    def __init__(out self):
        self.rx = 0.0; self.ry = 0.0; self.rz = 0.0; self.rw = 1.0
        self.dx = 0.0; self.dy = 0.0; self.dz = 0.0; self.dw = 0.0

    @staticmethod
    def from_rot_trans(rot: Quat, trans: Vec3) -> Self:
        var q = rot.normalized()
        var s = Self()
        s.rx = q.x; s.ry = q.y; s.rz = q.z; s.rw = q.w
        s.dx = Float32(0.5) * (trans.x * q.w + trans.y * q.z - trans.z * q.y)
        s.dy = Float32(0.5) * (-trans.x * q.z + trans.y * q.w + trans.z * q.x)
        s.dz = Float32(0.5) * (trans.x * q.y - trans.y * q.x + trans.z * q.w)
        s.dw = Float32(0.5) * (-trans.x * q.x - trans.y * q.y - trans.z * q.z)
        return s


@always_inline
def _deform_vertex_dq(
    pos: Vec3,
    nrm: Vec3,
    b0: Int,
    b1: Int,
    w0: Float32,
    w1: Float32,
    dqs: List[_BoneDQ],
) -> Tuple[Vec3, Vec3]:
    """Blends 2-bone Dual-Quaternions with hemisphere antipodality sign alignment."""
    var dq0 = dqs[b0]
    var dq1 = dqs[b1]
    var dot_r = dq0.rx * dq1.rx + dq0.ry * dq1.ry + dq0.rz * dq1.rz + dq0.rw * dq1.rw
    var sgn = Float32(-1.0) if dot_r < Float32(0.0) else Float32(1.0)
    var wb = w1 * sgn

    var rx = dq0.rx * w0 + dq1.rx * wb
    var ry = dq0.ry * w0 + dq1.ry * wb
    var rz = dq0.rz * w0 + dq1.rz * wb
    var rw = dq0.rw * w0 + dq1.rw * wb
    var dx = dq0.dx * w0 + dq1.dx * wb
    var dy = dq0.dy * w0 + dq1.dy * wb
    var dz = dq0.dz * w0 + dq1.dz * wb
    var dw = dq0.dw * w0 + dq1.dw * wb

    var norm = sqrt(max(Float32(1e-12), rx * rx + ry * ry + rz * rz + rw * rw))
    var inv_n = Float32(1.0) / norm
    rx *= inv_n; ry *= inv_n; rz *= inv_n; rw *= inv_n
    dx *= inv_n; dy *= inv_n; dz *= inv_n; dw *= inv_n

    var q = Quat(rx, ry, rz, rw)
    var tx = Float32(2.0) * (-dw * rx + dx * rw - dy * rz + dz * ry)
    var ty = Float32(2.0) * (-dw * ry + dx * rz + dy * rw - dz * rx)
    var tz = Float32(2.0) * (-dw * rz - dx * ry + dy * rx + dz * rw)

    var out_p = q.rotate(pos) + Vec3(tx, ty, tz)
    var out_n = q.rotate(nrm).normalized()
    return (out_p, out_n)


def _append_skinned_submesh(
    mut dst: NativeMesh,
    src: NativeMesh,
    mut jp_out: List[UInt32],
    mut jw_out: List[Vec3],
    bone_primary: Int,
    bone_secondary: Int,
    pivot_pos: Vec3,
    blend_axis: Vec3,
    blend_radius: Float32,
    tint_color: UInt32 = 0xFFFFFFFF,
):
    """Appends `src` into `dst` while assigning smooth joint-boundary skin weights and 24-bit sRGB `colors`."""
    var base_v = dst.vertex_count()
    var svc = src.vertex_count()
    var has_col = len(src.colors) >= svc * 4
    var has_ao = len(src.ao) >= svc
    var has_uv = len(src.texcoords) >= svc * 2
    var has_nrm = len(src.normals) >= svc * 3

    var tr = UInt8((tint_color >> 16) & 255)
    var tg = UInt8((tint_color >> 8) & 255)
    var tb = UInt8(tint_color & 255)
    var ta = UInt8((tint_color >> 24) & 255)

    var inv_br = Float32(1.0) / max(Float32(0.01), blend_radius)

    for i in range(svc):
        var px = src.positions[i * 3 + 0]
        var py = src.positions[i * 3 + 1]
        var pz = src.positions[i * 3 + 2]
        dst.positions.append(px)
        dst.positions.append(py)
        dst.positions.append(pz)

        if has_nrm:
            dst.normals.append(src.normals[i * 3 + 0])
            dst.normals.append(src.normals[i * 3 + 1])
            dst.normals.append(src.normals[i * 3 + 2])
        else:
            dst.normals.append(0.0); dst.normals.append(1.0); dst.normals.append(0.0)

        if has_uv:
            dst.texcoords.append(src.texcoords[i * 2 + 0])
            dst.texcoords.append(src.texcoords[i * 2 + 1])
        else:
            dst.texcoords.append(px * Float32(0.5))
            dst.texcoords.append(py * Float32(0.5))

        if has_col and tint_color == 0xFFFFFFFF:
            dst.colors.append(src.colors[i * 4 + 0])
            dst.colors.append(src.colors[i * 4 + 1])
            dst.colors.append(src.colors[i * 4 + 2])
            dst.colors.append(src.colors[i * 4 + 3])
        else:
            dst.colors.append(tr)
            dst.colors.append(tg)
            dst.colors.append(tb)
            dst.colors.append(ta)

        dst.ao.append(src.ao[i] if has_ao else UInt8(235))

        var w0 = Float32(1.0)
        var w1 = Float32(0.0)
        if bone_secondary >= 0 and bone_secondary != bone_primary:
            var d_ax = (Vec3(px, py, pz) - pivot_pos).dot(blend_axis)
            var u = max(Float32(0.0), min(Float32(1.0), d_ax * inv_br * Float32(0.5) + Float32(0.5)))
            w0 = u * u * (Float32(3.0) - Float32(2.0) * u)
            w0 = Float32(0.55) + w0 * Float32(0.45)
            w1 = Float32(1.0) - w0

        var b0_u = max(0, bone_primary)
        var b1_u = max(0, bone_secondary if bone_secondary >= 0 else bone_primary)
        dst.joints.append(UInt16(b0_u))
        dst.joints.append(UInt16(b1_u))
        dst.joints.append(UInt16(0))
        dst.joints.append(UInt16(0))
        dst.weights.append(w0)
        dst.weights.append(w1)
        dst.weights.append(Float32(0.0))
        dst.weights.append(Float32(0.0))

        jp_out.append(UInt32(b0_u & 255) | (UInt32(b1_u & 255) << 8))
        jw_out.append(Vec3(w0, w1, Float32(0.0)))

    for idx in src.indices:
        dst.indices.append(base_v + idx)


def _make_subd_armor_box(
    center: Vec3,
    half_bot: Vec3,
    half_top: Vec3,
    top_offset_z: Float32 = 0.0,
    top_offset_x: Float32 = 0.0,
    levels: Int = 2,
    crease: Float32 = 3.5,
) -> NativeMesh:
    """Builds a hard-surface faceted Catmull-Clark SubD armor plate (`crease = 3.5` -> crisp split chamfers!)."""
    var c = SubdCage()
    var b0 = c.add_vert(center + Vec3(-half_bot.x, -half_bot.y, -half_bot.z))
    var b1 = c.add_vert(center + Vec3( half_bot.x, -half_bot.y, -half_bot.z))
    var b2 = c.add_vert(center + Vec3( half_bot.x, -half_bot.y,  half_bot.z))
    var b3 = c.add_vert(center + Vec3(-half_bot.x, -half_bot.y,  half_bot.z))

    var tx = top_offset_x
    var tz = top_offset_z
    var t0 = c.add_vert(center + Vec3(-half_top.x + tx,  half_top.y, -half_top.z + tz))
    var t1 = c.add_vert(center + Vec3( half_top.x + tx,  half_top.y, -half_top.z + tz))
    var t2 = c.add_vert(center + Vec3( half_top.x + tx,  half_top.y,  half_top.z + tz))
    var t3 = c.add_vert(center + Vec3(-half_top.x + tx,  half_top.y,  half_top.z + tz))

    c.add_quad(b0, b3, b2, b1)  # bottom (-Y)
    c.add_quad(t0, t1, t2, t3)  # top (+Y)
    c.add_quad(b3, t3, t2, b2)  # front (+Z)
    c.add_quad(b1, b2, t2, t1)  # right (+X)
    c.add_quad(b0, b1, t1, t0)  # back (-Z)
    c.add_quad(b3, b0, t0, t3)  # left (-X)

    if crease > Float32(0.0):
        c.set_crease_chain([b0, b1, b2, b3], crease, closed=True)
        c.set_crease_chain([t0, t1, t2, t3], crease, closed=True)
        c.set_edge_sharpness(b0, t0, crease * Float32(0.90))
        c.set_edge_sharpness(b1, t1, crease * Float32(0.90))
        c.set_edge_sharpness(b2, t2, crease * Float32(0.92))
        c.set_edge_sharpness(b3, t3, crease * Float32(0.92))

    return subdivide_to_mesh(c, levels=levels, split_sharp_threshold=1.35)


# ============================================================================
# 1. `16`-Bone 3-Submesh Hard-Surface MaxTac `Cyber-Enforcer` (`v3`)
# ============================================================================

struct SkinnedMeshData(Copyable, Movable):
    """Complete 16-bone SubD character/mech package ready for both `NativeModel`/`SkeletonInstance`
    FK/IK and `FlintAaaVulkanRenderer.add_skinned_mesh` GPU Dual-Quaternion skinning."""
    var mesh: NativeMesh
    var bone_count: Int
    var bone_names: List[String]
    var bone_parents: List[Int]
    var bind_pivots: List[Vec3]
    var bind_local_translations: List[Vec3]
    var inverse_bind_matrices: List[Mat4]
    var joint_indices_packed: List[UInt32]
    var joint_weights_w012: List[Vec3]
    var rig_model: NativeModel

    def __init__(out self):
        self.mesh = NativeMesh()
        self.bone_count = 0
        self.bone_names = List[String]()
        self.bone_parents = List[Int]()
        self.bind_pivots = List[Vec3]()
        self.bind_local_translations = List[Vec3]()
        self.inverse_bind_matrices = List[Mat4]()
        self.joint_indices_packed = List[UInt32]()
        self.joint_weights_w012 = List[Vec3]()
        self.rig_model = NativeModel()


def _init_enforcer_skeleton(mut target: SkinnedMeshData):
    target.bone_count = ENFORCER_BONE_COUNT
    var names: List[String] = [
        "Pelvis", "Spine", "Chest", "Head",
        "L_UpperArm", "L_Forearm", "R_UpperArm", "R_Forearm",
        "L_Thigh", "L_Shin", "L_Foot", "R_Thigh", "R_Shin", "R_Foot",
        "Shield", "Rifle",
    ]
    var parents: List[Int] = [
        -1, 0, 1, 2,
        2, 4, 2, 6,
        0, 8, 9, 0, 11, 12,
        5, 7,
    ]
    var pivots: List[Vec3] = [
        Vec3( 0.00, 2.52,  0.00),  # 0: Pelvis
        Vec3( 0.00, 2.86, -0.02),  # 1: Spine
        Vec3( 0.00, 3.38,  0.04),  # 2: Chest
        Vec3( 0.00, 4.32,  0.10),  # 3: Head
        Vec3(-1.08, 3.82,  0.02),  # 4: L_UpperArm
        Vec3(-1.32, 3.06,  0.18),  # 5: L_Forearm
        Vec3( 1.08, 3.82,  0.02),  # 6: R_UpperArm
        Vec3( 1.32, 3.06,  0.22),  # 7: R_Forearm
        Vec3(-0.52, 2.42,  0.04),  # 8: L_Thigh
        Vec3(-0.54, 1.34, -0.18),  # 9: L_Shin (reverse-knee apex)
        Vec3(-0.54, 0.24,  0.08),  # 10: L_Foot
        Vec3( 0.52, 2.42,  0.04),  # 11: R_Thigh
        Vec3( 0.54, 1.34, -0.18),  # 12: R_Shin (reverse-knee apex)
        Vec3( 0.54, 0.24,  0.08),  # 13: R_Foot
        Vec3(-1.56, 2.88,  0.42),  # 14: Shield
        Vec3( 1.38, 3.04,  0.88),  # 15: Rifle
    ]

    target.bone_names = names^
    target.bone_parents = parents^
    target.bind_pivots = pivots.copy()

    var skin = NativeSkin()
    for b in range(ENFORCER_BONE_COUNT):
        var p_idx = target.bone_parents[b]
        var loc_t = pivots[b] if p_idx < 0 else (pivots[b] - pivots[p_idx])
        target.bind_local_translations.append(loc_t)

        var node = NativeNode()
        node.name = target.bone_names[b]
        node.base_translation = loc_t
        node.translation = loc_t
        target.rig_model.nodes.append(node^)

        var inv_m = Mat4()
        inv_m.set(0, 3, -pivots[b].x)
        inv_m.set(1, 3, -pivots[b].y)
        inv_m.set(2, 3, -pivots[b].z)
        target.inverse_bind_matrices.append(inv_m.copy())
        skin.joints.append(b)
        for comp in range(16):
            skin.inverse_bind.append(inv_m.values[comp])

    for b in range(ENFORCER_BONE_COUNT):
        var p_idx = target.bone_parents[b]
        if p_idx < 0:
            target.rig_model.roots.append(b)
        else:
            target.rig_model.nodes[p_idx].children.append(b)
    target.rig_model.skins.append(skin^)


def build_subd_cyber_enforcer_rigged_v3() -> Tuple[SkinnedMeshData, SkinnedMeshData, SkinnedMeshData]:
    """Builds the 3 synchronized 16-bone skinned MaxTac `Cyber-Enforcer` submeshes (`~3,120` tris total):
    - `[0]`: `armor_rig` — Coated tactical slate-blue ceramite carapace, layered samurai pauldrons,
      angular reverse-knee cowls, backpack shroud & white/crimson NCPD tactical stripes (`colors`).
    - `[1]`: `chassis_rig` — Brushed dark titanium frame, mirror-chrome hydraulic knee/elbow actuators,
      swept antennas, tungsten talons, and bullpup pulse-rifle chassis (`colors`).
    - `[2]`: `emissive_rig` — HDR glowing cyan T-visor + quad-optics `[24, 244, 255]`, amber chest
      arc-reactor `[255, 165, 36]`, left-pauldron red strobe `[255, 28, 52]`, right-pauldron blue strobe
      `[28, 132, 255]`, dorsal thruster vents, pulse-rifle plasma rails & riot-shield energy rim!
    All 3 `SkinnedMeshData` parts share identical `16`-bone `rig_model` & `bind_pivots` so they animate
    from a single shared 16-bone GPU Dual-Quaternion palette slice!
    """
    var armor = SkinnedMeshData()
    var chassis = SkinnedMeshData()
    var emis = SkinnedMeshData()
    _init_enforcer_skeleton(armor)
    _init_enforcer_skeleton(chassis)
    _init_enforcer_skeleton(emis)

    ref pivots = armor.bind_pivots

    # Material tuning per submesh
    armor.mesh.base_color = 0xFFFFFFFF
    armor.mesh.roughness = 0.28
    armor.mesh.metallic = 0.34
    armor.mesh.clearcoat = 0.88
    armor.mesh.clearcoat_roughness = 0.05

    chassis.mesh.base_color = 0xFFFFFFFF
    chassis.mesh.roughness = 0.22
    chassis.mesh.metallic = 0.88

    emis.mesh.base_color = 0xFFFFFFFF
    emis.mesh.roughness = 0.08
    emis.mesh.metallic = 0.0
    emis.mesh.emissive_color = Vec3(1.0, 1.0, 1.0)
    emis.mesh.emissive_strength = 8.5

    # -------------------------------------------------------------------------
    # SUBMESH 0 (`armor`): Hard-Surface Slate-Blue Ceramite + White/Crimson NCPD Trim
    # -------------------------------------------------------------------------
    # Pelvic codpiece & hip tassets
    var m_pelvis = _make_subd_armor_box(Vec3(0.0, 2.54, 0.04), Vec3(0.54, 0.22, 0.38), Vec3(0.66, 0.20, 0.44), levels=2, crease=3.6)
    _append_skinned_submesh(armor.mesh, m_pelvis, armor.joint_indices_packed, armor.joint_weights_w012, 0, 1, pivots[1], Vec3(0.0, -1.0, 0.0), 0.18, 0xFF526684)

    # Angular V-sloped pectoral carapace + NCPD white breastplate stripe + dorsal jump-thruster backpack
    var m_chest = _make_subd_armor_box(Vec3(0.0, 3.72, 0.12), Vec3(0.66, 0.42, 0.44), Vec3(0.90, 0.42, 0.56), top_offset_z=0.10, levels=2, crease=3.7)
    var m_stripe = _make_subd_armor_box(Vec3(-0.46, 3.92, 0.52), Vec3(0.22, 0.12, 0.12), Vec3(0.26, 0.12, 0.10), levels=1, crease=3.8)
    var m_pack = _make_subd_armor_box(Vec3(0.0, 3.88, -0.40), Vec3(0.58, 0.40, 0.26), Vec3(0.68, 0.40, 0.30), top_offset_z=-0.08, levels=2, crease=3.7)
    _append_skinned_submesh(armor.mesh, m_chest, armor.joint_indices_packed, armor.joint_weights_w012, 2, 1, pivots[2], Vec3(0.0, 1.0, 0.0), 0.22, 0xFF566C8C)
    _append_skinned_submesh(armor.mesh, m_stripe, armor.joint_indices_packed, armor.joint_weights_w012, 2, 2, pivots[2], Vec3(0.0, 1.0, 0.0), 0.10, 0xFFF0F4FA)
    _append_skinned_submesh(armor.mesh, m_pack, armor.joint_indices_packed, armor.joint_weights_w012, 2, 1, pivots[2], Vec3(0.0, 1.0, 0.0), 0.22, 0xFF3E4E66)

    # Chamfered MaxTac sensor helmet + white forehead crest
    var m_head = _make_subd_armor_box(Vec3(0.0, 4.56, 0.14), Vec3(0.33, 0.24, 0.40), Vec3(0.29, 0.24, 0.35), top_offset_z=-0.06, levels=2, crease=3.7)
    var m_crest = _make_subd_armor_box(Vec3(0.0, 4.80, 0.18), Vec3(0.11, 0.06, 0.26), Vec3(0.08, 0.06, 0.22), levels=1, crease=3.8)
    _append_skinned_submesh(armor.mesh, m_head, armor.joint_indices_packed, armor.joint_weights_w012, 3, 2, pivots[3], Vec3(0.0, 1.0, 0.0), 0.12, 0xFF5E7496)
    _append_skinned_submesh(armor.mesh, m_crest, armor.joint_indices_packed, armor.joint_weights_w012, 3, 3, pivots[3], Vec3(0.0, 1.0, 0.0), 0.08, 0xFFF2F5FA)

    # Layered Samurai Shoulder Pauldrons (upper flared wing + stepped lower plate) & faceted Forearm Vambraces
    for side in range(2):
        var sx = Float32(-1.0) if side == 0 else Float32(1.0)
        var b_up = 4 if side == 0 else 6
        var b_lo = 5 if side == 0 else 7
        var trim_col: UInt32 = UInt32(0xFFDC263E) if side == 0 else UInt32(0xFFECEFF4)

        var m_paul_hi = _make_subd_armor_box(
            Vec3(sx * Float32(1.18), 3.92, 0.02),
            Vec3(0.30, 0.20, 0.34),
            Vec3(0.38, 0.18, 0.38),
            top_offset_x=sx * Float32(0.08),
            levels=2,
            crease=3.7,
        )
        var m_paul_lo = _make_subd_armor_box(
            Vec3(sx * Float32(1.25), 3.60, 0.04),
            Vec3(0.24, 0.14, 0.28),
            Vec3(0.28, 0.14, 0.32),
            levels=1,
            crease=3.7,
        )
        var m_fore = _make_subd_armor_box(
            Vec3(sx * Float32(1.38), 2.94, 0.36),
            Vec3(0.18, 0.18, 0.30),
            Vec3(0.19, 0.18, 0.30),
            levels=1,
            crease=3.6,
        )
        _append_skinned_submesh(armor.mesh, m_paul_hi, armor.joint_indices_packed, armor.joint_weights_w012, b_up, 2, pivots[b_up], Vec3(sx, 0.0, 0.0), 0.16, 0xFF566D8E)
        _append_skinned_submesh(armor.mesh, m_paul_lo, armor.joint_indices_packed, armor.joint_weights_w012, b_up, 2, pivots[b_up], Vec3(sx, 0.0, 0.0), 0.14, trim_col)
        _append_skinned_submesh(armor.mesh, m_fore, armor.joint_indices_packed, armor.joint_weights_w012, b_lo, b_up, pivots[b_lo], Vec3(0.0, -1.0, 0.2), 0.14, 0xFF4C607E)

    # Hard-surface Thigh Carapace, Angular Knee-Guard Cowls & Reverse-Knee Shin Greaves
    for side in range(2):
        var sx = Float32(-1.0) if side == 0 else Float32(1.0)
        var b_th = 8 if side == 0 else 11
        var b_sh = 9 if side == 0 else 12

        var m_thigh = _make_subd_armor_box(
            Vec3(sx * Float32(0.54), 1.92, -0.04),
            Vec3(0.22, 0.52, 0.25),
            Vec3(0.27, 0.52, 0.30),
            top_offset_z=0.14,
            levels=2,
            crease=3.6,
        )
        var m_kneeguard = _make_subd_armor_box(
            Vec3(sx * Float32(0.54), 1.44, 0.12),
            Vec3(0.18, 0.18, 0.14),
            Vec3(0.20, 0.18, 0.18),
            top_offset_z=0.06,
            levels=1,
            crease=3.8,
        )
        var m_shin = _make_subd_armor_box(
            Vec3(sx * Float32(0.54), 0.80, -0.02),
            Vec3(0.18, 0.52, 0.22),
            Vec3(0.23, 0.52, 0.26),
            top_offset_z=-0.16,
            levels=2,
            crease=3.7,
        )
        _append_skinned_submesh(armor.mesh, m_thigh, armor.joint_indices_packed, armor.joint_weights_w012, b_th, 0, pivots[b_th], Vec3(0.0, -1.0, 0.0), 0.16, 0xFF526888)
        _append_skinned_submesh(armor.mesh, m_kneeguard, armor.joint_indices_packed, armor.joint_weights_w012, b_th, b_sh, pivots[b_sh], Vec3(0.0, 1.0, 0.0), 0.12, 0xFFDFE5EE)
        _append_skinned_submesh(armor.mesh, m_shin, armor.joint_indices_packed, armor.joint_weights_w012, b_sh, b_th, pivots[b_sh], Vec3(0.0, -1.0, 0.0), 0.16, 0xFF485C78)

    # Curved Ballistic Riot Shield Armor Plate (`Bone 14`)
    var sh_cage = SubdCage()
    var sh_z: List[Float32] = [-0.08, 0.44, 0.92]
    var sh_x: List[Float32] = [-1.62, -1.69, -1.56]
    var sh_grid = List[List[Int]]()
    for sz_i in range(3):
        var row = List[Int](capacity=4)
        row.append(sh_cage.add_vert(Vec3(sh_x[sz_i], 1.96, sh_z[sz_i])))
        row.append(sh_cage.add_vert(Vec3(sh_x[sz_i], 3.78, sh_z[sz_i])))
        row.append(sh_cage.add_vert(Vec3(sh_x[sz_i] + Float32(0.09), 3.78, sh_z[sz_i])))
        row.append(sh_cage.add_vert(Vec3(sh_x[sz_i] + Float32(0.09), 1.96, sh_z[sz_i])))
        sh_grid.append(row^)
    for sz_i in range(2):
        for k in range(4):
            var kn = (k + 1) % 4
            sh_cage.add_quad(sh_grid[sz_i][k], sh_grid[sz_i + 1][k], sh_grid[sz_i + 1][kn], sh_grid[sz_i][kn])
            sh_cage.set_edge_sharpness(sh_grid[sz_i][k], sh_grid[sz_i + 1][k], Float32(3.6))
    sh_cage.add_quad(sh_grid[0][0], sh_grid[0][3], sh_grid[0][2], sh_grid[0][1])
    sh_cage.add_quad(sh_grid[2][0], sh_grid[2][1], sh_grid[2][2], sh_grid[2][3])
    var m_shield = subdivide_to_mesh(sh_cage, levels=1, split_sharp_threshold=1.35)
    _append_skinned_submesh(armor.mesh, m_shield, armor.joint_indices_packed, armor.joint_weights_w012, 14, 5, pivots[14], Vec3(-1.0, 0.0, 0.0), 0.12, 0xFF3E4E68)

    # -------------------------------------------------------------------------
    # SUBMESH 1 (`chassis`): Dark Titanium Frame + Chrome Hydraulic Rams + Bullpup Rifle
    # -------------------------------------------------------------------------
    var m_spine = _make_subd_armor_box(Vec3(0.0, 3.06, -0.02), Vec3(0.40, 0.26, 0.28), Vec3(0.50, 0.26, 0.34), levels=1, crease=3.5)
    _append_skinned_submesh(chassis.mesh, m_spine, chassis.joint_indices_packed, chassis.joint_weights_w012, 1, 0, pivots[1], Vec3(0.0, 1.0, 0.0), 0.18, 0xFF424B58)

    var mb_ant = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.22, metallic=0.90)
    mb_ant.add_cylinder(Vec3(-0.62, 4.18, -0.24), Vec3(-0.72, 5.36, -0.40), 0.038, 0.010, segments=6, caps=False, color=0xFFC8D2DE)
    mb_ant.add_cylinder(Vec3( 0.62, 4.18, -0.24), Vec3( 0.72, 5.36, -0.40), 0.038, 0.010, segments=6, caps=False, color=0xFFC8D2DE)
    _append_skinned_submesh(chassis.mesh, mb_ant^.finish(), chassis.joint_indices_packed, chassis.joint_weights_w012, 2, 2, pivots[2], Vec3(0.0, 1.0, 0.0), 0.10, 0xFFC8D2DE)

    for side in range(2):
        var sx = Float32(-1.0) if side == 0 else Float32(1.0)
        var b_up = 4 if side == 0 else 6
        var b_lo = 5 if side == 0 else 7
        var b_th = 8 if side == 0 else 11
        var b_sh = 9 if side == 0 else 12
        var b_ft = 10 if side == 0 else 13

        # Faceted bicep actuator block + bright chrome elbow hydraulic piston
        var m_bicep = _make_subd_armor_box(Vec3(sx * Float32(1.22), 3.38, 0.08), Vec3(0.15, 0.22, 0.17), Vec3(0.17, 0.22, 0.19), levels=1, crease=3.5)
        _append_skinned_submesh(chassis.mesh, m_bicep, chassis.joint_indices_packed, chassis.joint_weights_w012, b_up, b_lo, pivots[b_lo], Vec3(0.0, 1.0, 0.0), 0.14, 0xFF46505E)

        var mb_rams = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.18, metallic=0.94)
        # Twin mirror-chrome reverse-knee hydraulic rams (`0xFFDCE6F2`)
        mb_rams.add_cylinder(Vec3(sx * Float32(0.46), 0.34, -0.14), Vec3(sx * Float32(0.46), 1.42, -0.34), 0.048, 0.056, segments=8, caps=False, color=0xFFDCE6F2)
        mb_rams.add_cylinder(Vec3(sx * Float32(0.62), 0.34, -0.14), Vec3(sx * Float32(0.62), 1.42, -0.34), 0.048, 0.056, segments=8, caps=False, color=0xFFDCE6F2)
        _append_skinned_submesh(chassis.mesh, mb_rams^.finish(), chassis.joint_indices_packed, chassis.joint_weights_w012, b_sh, b_th, pivots[b_sh], Vec3(0.0, -1.0, 0.0), 0.20, 0xFFDCE6F2)

        # Articulated split-toe tungsten talons + rear stabilizer heel spur (`grounded AO = 88`)
        var m_toe_l = _make_subd_armor_box(Vec3(sx * Float32(0.44), 0.11, 0.22), Vec3(0.09, 0.11, 0.26), Vec3(0.08, 0.10, 0.22), levels=1, crease=3.6)
        var m_toe_r = _make_subd_armor_box(Vec3(sx * Float32(0.64), 0.11, 0.22), Vec3(0.09, 0.11, 0.26), Vec3(0.08, 0.10, 0.22), levels=1, crease=3.6)
        var m_heel = _make_subd_armor_box(Vec3(sx * Float32(0.54), 0.10, -0.18), Vec3(0.11, 0.10, 0.16), Vec3(0.09, 0.09, 0.14), levels=1, crease=3.6)
        for tv in range(len(m_toe_l.ao)):
            if m_toe_l.positions[tv * 3 + 1] < Float32(0.06):
                m_toe_l.ao[tv] = UInt8(88)
                m_toe_r.ao[tv] = UInt8(88)
                m_heel.ao[tv] = UInt8(88)
        _append_skinned_submesh(chassis.mesh, m_toe_l, chassis.joint_indices_packed, chassis.joint_weights_w012, b_ft, b_sh, pivots[b_ft], Vec3(0.0, -1.0, 0.0), 0.10, 0xFF4E5868)
        _append_skinned_submesh(chassis.mesh, m_toe_r, chassis.joint_indices_packed, chassis.joint_weights_w012, b_ft, b_sh, pivots[b_ft], Vec3(0.0, -1.0, 0.0), 0.10, 0xFF4E5868)
        _append_skinned_submesh(chassis.mesh, m_heel, chassis.joint_indices_packed, chassis.joint_weights_w012, b_ft, b_sh, pivots[b_ft], Vec3(0.0, -1.0, 0.0), 0.10, 0xFF68768A)

    # Bullpup Heavy Pulse Rifle (`Bone 15`)
    var m_rifle_body = _make_subd_armor_box(Vec3(1.40, 3.06, 0.90), Vec3(0.09, 0.16, 0.52), Vec3(0.08, 0.15, 0.48), levels=2, crease=3.7)
    var m_rifle_mag = _make_subd_armor_box(Vec3(1.40, 2.82, 0.62), Vec3(0.06, 0.14, 0.11), Vec3(0.06, 0.14, 0.10), levels=1, crease=3.6)
    var mb_barrel = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.20, metallic=0.90)
    mb_barrel.add_cylinder(Vec3(1.40, 3.12, 1.38), Vec3(1.40, 3.12, 2.06), 0.052, 0.046, segments=10, caps=True, color=0xFFB8C4D4)
    mb_barrel.add_cylinder(Vec3(1.40, 2.98, 1.38), Vec3(1.40, 2.98, 1.92), 0.040, 0.036, segments=8, caps=True, color=0xFF8A96A8)
    _append_skinned_submesh(chassis.mesh, m_rifle_body, chassis.joint_indices_packed, chassis.joint_weights_w012, 15, 7, pivots[15], Vec3(0.0, 0.0, 1.0), 0.25, 0xFF3E4856)
    _append_skinned_submesh(chassis.mesh, m_rifle_mag, chassis.joint_indices_packed, chassis.joint_weights_w012, 15, 15, pivots[15], Vec3(0.0, 0.0, 1.0), 0.10, 0xFF5A6678)
    _append_skinned_submesh(chassis.mesh, mb_barrel^.finish(), chassis.joint_indices_packed, chassis.joint_weights_w012, 15, 15, pivots[15], Vec3(0.0, 0.0, 1.0), 0.10, 0xFFB8C4D4)

    # -------------------------------------------------------------------------
    # SUBMESH 2 (`emissive`): HDR Cyan Visor/Quad-Optics, Arc-Reactor, Police Strobes & Plasma Rails
    # -------------------------------------------------------------------------
    var mb_em = MeshBuilder(
        base_color=0xFFFFFFFF, roughness=0.06, metallic=0.0,
        emissive_color=Vec3(1.0, 1.0, 1.0), emissive_strength=8.5,
    )

    # Bone 3: Recessed glowing cyan T-visor + 4 MaxTac quad-optic sensor eyes (`[24, 244, 255]`)
    var m_visor = _make_subd_armor_box(Vec3(0.0, 4.58, 0.52), Vec3(0.28, 0.065, 0.04), Vec3(0.26, 0.065, 0.04), levels=1, crease=3.8)
    _append_skinned_submesh(emis.mesh, m_visor, emis.joint_indices_packed, emis.joint_weights_w012, 3, 3, pivots[3], Vec3(0.0, 1.0, 0.0), 0.08, 0xFF18F4FF)
    for eye_i in range(4):
        var ex = (Float32(eye_i) - Float32(1.5)) * Float32(0.12)
        mb_em.add_cylinder(Vec3(ex, 4.66, 0.50), Vec3(ex, 4.66, 0.57), 0.028, 0.024, segments=6, color=0xFF18F4FF)
    _append_skinned_submesh(emis.mesh, mb_em^.finish(), emis.joint_indices_packed, emis.joint_weights_w012, 3, 3, pivots[3], Vec3(0.0, 1.0, 0.0), 0.08, 0xFF18F4FF)

    # Bone 2: Chest amber arc-reactor `[255, 165, 36]` + dorsal backpack twin cyan ion thruster vents
    var mb_chest_em = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.06, metallic=0.0)
    mb_chest_em.add_cylinder(Vec3(0.0, 3.76, 0.58), Vec3(0.0, 3.76, 0.67), 0.16, 0.13, segments=8, color=0xFFFFA524)
    _append_skinned_submesh(emis.mesh, mb_chest_em^.finish(), emis.joint_indices_packed, emis.joint_weights_w012, 2, 2, pivots[2], Vec3(0.0, 1.0, 0.0), 0.10, 0xFFFFA524)

    var mb_thr_em = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.06, metallic=0.0)
    mb_thr_em.add_cylinder(Vec3(-0.34, 3.68, -0.66), Vec3(-0.34, 3.62, -0.74), 0.12, 0.10, segments=8, color=0xFF20E4FF)
    mb_thr_em.add_cylinder(Vec3( 0.34, 3.68, -0.66), Vec3( 0.34, 3.62, -0.74), 0.12, 0.10, segments=8, color=0xFF20E4FF)
    _append_skinned_submesh(emis.mesh, mb_thr_em^.finish(), emis.joint_indices_packed, emis.joint_weights_w012, 2, 2, pivots[2], Vec3(0.0, 1.0, 0.0), 0.10, 0xFF20E4FF)

    # Bone 4 (Left Pauldron): Red police strobe `[255, 28, 52]`; Bone 6 (Right Pauldron): Blue police strobe `[28, 132, 255]`
    var m_strobe_l = _make_subd_armor_box(Vec3(-1.22, 4.12, 0.06), Vec3(0.14, 0.045, 0.16), Vec3(0.12, 0.045, 0.14), levels=1, crease=3.6)
    var m_strobe_r = _make_subd_armor_box(Vec3( 1.22, 4.12, 0.06), Vec3(0.14, 0.045, 0.16), Vec3(0.12, 0.045, 0.14), levels=1, crease=3.6)
    _append_skinned_submesh(emis.mesh, m_strobe_l, emis.joint_indices_packed, emis.joint_weights_w012, 4, 4, pivots[4], Vec3(-1.0, 0.0, 0.0), 0.10, 0xFFFF1C34)
    _append_skinned_submesh(emis.mesh, m_strobe_r, emis.joint_indices_packed, emis.joint_weights_w012, 6, 6, pivots[6], Vec3( 1.0, 0.0, 0.0), 0.10, 0xFF1C84FF)

    # Bone 14: Riot-shield cyan energy-field perimeter rim & holographic viewport strip
    var m_sh_rim = _make_subd_armor_box(Vec3(-1.70, 3.36, 0.44), Vec3(0.035, 0.09, 0.34), Vec3(0.035, 0.09, 0.34), levels=1, crease=3.8)
    _append_skinned_submesh(emis.mesh, m_sh_rim, emis.joint_indices_packed, emis.joint_weights_w012, 14, 14, pivots[14], Vec3(-1.0, 0.0, 0.0), 0.10, 0xFF1CF0FF)

    # Bone 15: Bullpup pulse-rifle glowing cyan plasma accelerator rails (`z = 0.85 .. 1.98`)
    var m_rail = _make_subd_armor_box(Vec3(1.40, 3.05, 1.42), Vec3(0.095, 0.032, 0.48), Vec3(0.095, 0.032, 0.48), levels=1, crease=3.8)
    _append_skinned_submesh(emis.mesh, m_rail, emis.joint_indices_packed, emis.joint_weights_w012, 15, 15, pivots[15], Vec3(0.0, 0.0, 1.0), 0.10, 0xFF20F4FF)

    armor.mesh.skin = 0
    chassis.mesh.skin = 0
    emis.mesh.skin = 0
    armor.mesh.ensure_bounds()
    chassis.mesh.ensure_bounds()
    emis.mesh.ensure_bounds()
    return (armor^, chassis^, emis^)


def _merge_skinned_submesh(mut unified: SkinnedMeshData, src_rig: SkinnedMeshData):
    ref sm = src_rig.mesh
    var base_v = unified.mesh.vertex_count()
    var svc = sm.vertex_count()
    for v in range(svc):
        unified.mesh.positions.append(sm.positions[v * 3 + 0])
        unified.mesh.positions.append(sm.positions[v * 3 + 1])
        unified.mesh.positions.append(sm.positions[v * 3 + 2])
        unified.mesh.normals.append(sm.normals[v * 3 + 0])
        unified.mesh.normals.append(sm.normals[v * 3 + 1])
        unified.mesh.normals.append(sm.normals[v * 3 + 2])
        unified.mesh.texcoords.append(sm.texcoords[v * 2 + 0])
        unified.mesh.texcoords.append(sm.texcoords[v * 2 + 1])
        unified.mesh.colors.append(sm.colors[v * 4 + 0])
        unified.mesh.colors.append(sm.colors[v * 4 + 1])
        unified.mesh.colors.append(sm.colors[v * 4 + 2])
        unified.mesh.colors.append(sm.colors[v * 4 + 3])
        unified.mesh.ao.append(sm.ao[v])
        for jc in range(4):
            unified.mesh.joints.append(sm.joints[v * 4 + jc])
            unified.mesh.weights.append(sm.weights[v * 4 + jc])
        unified.joint_indices_packed.append(src_rig.joint_indices_packed[v])
        unified.joint_weights_w012.append(src_rig.joint_weights_w012[v])
    for idx in sm.indices:
        unified.mesh.indices.append(base_v + idx)


def build_subd_cyber_enforcer_rigged() -> SkinnedMeshData:
    """Unified single-mesh 16-bone `Cyber-Enforcer` (`~3,120` tris) preserving all 24-bit per-vertex
    ceramite, chrome-servo, strobe, arc-reactor, and cyan-visor `colors`."""
    var parts = build_subd_cyber_enforcer_rigged_v3()
    var unified = parts[0].copy()
    _merge_skinned_submesh(unified, parts[1])
    _merge_skinned_submesh(unified, parts[2])
    unified.mesh.ensure_bounds()
    return unified^


# ============================================================================
# 2. `12`-Pose Articulated Humanoid Crowd Bank (`~688–760` tris/pose)
# ============================================================================

struct _CrowdRigTemplate(Movable):
    var upper: NativeMesh
    var lower: NativeMesh
    var upper_b0: List[Int]
    var upper_b1: List[Int]
    var upper_w0: List[Float32]
    var lower_b0: List[Int]
    var lower_b1: List[Int]
    var lower_w0: List[Float32]
    var parents: List[Int]
    var pivots: List[Vec3]

    def __init__(out self):
        self.upper = NativeMesh()
        self.lower = NativeMesh()
        self.upper_b0 = List[Int]()
        self.upper_b1 = List[Int]()
        self.upper_w0 = List[Float32]()
        self.lower_b0 = List[Int]()
        self.lower_b1 = List[Int]()
        self.lower_w0 = List[Float32]()
        self.parents = List[Int]()
        self.pivots = List[Vec3]()


def _append_crowd_part(
    mut dst: NativeMesh,
    mut b0_list: List[Int],
    mut b1_list: List[Int],
    mut w0_list: List[Float32],
    src: NativeMesh,
    j0: Int,
    j1: Int,
    pivot: Vec3,
    axis: Vec3,
    radius: Float32,
    col: UInt32,
):
    var base = dst.vertex_count()
    var vc = src.vertex_count()
    var has_src_col = len(src.colors) >= vc * 4
    var cr = UInt8((col >> 16) & 255)
    var cg = UInt8((col >> 8) & 255)
    var cb = UInt8(col & 255)
    var inv_r = Float32(1.0) / max(Float32(0.01), radius)

    for i in range(vc):
        var px = src.positions[i * 3 + 0]
        var py = src.positions[i * 3 + 1]
        var pz = src.positions[i * 3 + 2]
        dst.positions.append(px); dst.positions.append(py); dst.positions.append(pz)
        dst.normals.append(src.normals[i * 3 + 0])
        dst.normals.append(src.normals[i * 3 + 1])
        dst.normals.append(src.normals[i * 3 + 2])
        dst.texcoords.append(src.texcoords[i * 2 + 0])
        dst.texcoords.append(src.texcoords[i * 2 + 1])
        if has_src_col and col == 0xFFFFFFFF:
            dst.colors.append(src.colors[i * 4 + 0])
            dst.colors.append(src.colors[i * 4 + 1])
            dst.colors.append(src.colors[i * 4 + 2])
            dst.colors.append(src.colors[i * 4 + 3])
        else:
            dst.colors.append(cr); dst.colors.append(cg); dst.colors.append(cb); dst.colors.append(UInt8(255))
        var ao_v = src.ao[i] if i < len(src.ao) else UInt8(235)
        if py < Float32(0.08):
            ao_v = UInt8(min(Int(ao_v), 105))
        dst.ao.append(ao_v)

        var w0 = Float32(1.0)
        if j1 >= 0 and j1 != j0:
            var d = (Vec3(px, py, pz) - pivot).dot(axis)
            var u = max(Float32(0.0), min(Float32(1.0), d * inv_r * Float32(0.5) + Float32(0.5)))
            w0 = Float32(0.52) + (u * u * (Float32(3.0) - Float32(2.0) * u)) * Float32(0.48)
        b0_list.append(j0)
        b1_list.append(j1 if j1 >= 0 else j0)
        w0_list.append(w0)

    for idx in src.indices:
        dst.indices.append(base + idx)


def _add_segmented_limb_tube(
    mut dst: NativeMesh,
    mut b0_list: List[Int],
    mut b1_list: List[Int],
    mut w0_list: List[Float32],
    pts: List[Vec3],
    radii: List[Float32],
    seg_bones: List[Int],
    joint_pivot: Vec3,
    sides: Int,
    col_top: UInt32,
    col_bot: UInt32,
):
    """Builds a continuous multi-ring sleeve/trouser tube with top-to-cuff sRGB gradient & joint weights."""
    var n_st = len(pts)
    var mb = MeshBuilder(base_color=0xFFFFFFFF)
    var stride = sides + 1
    var base_v = mb.mesh.vertex_count()

    var r0 = Float32((col_top >> 16) & UInt32(255))
    var g0 = Float32((col_top >> 8) & UInt32(255))
    var b0_c = Float32(col_top & UInt32(255))
    var r1 = Float32((col_bot >> 16) & UInt32(255))
    var g1 = Float32((col_bot >> 8) & UInt32(255))
    var b1_c = Float32(col_bot & UInt32(255))

    for i in range(n_st):
        var dir = (pts[min(n_st - 1, i + 1)] - pts[max(0, i - 1)]).normalized()
        if dir.length_squared() < Float32(0.05):
            dir = Vec3(0.0, -1.0, 0.0)
        var ref_v = Vec3(0.0, 0.0, 1.0) if abs(dir.y) > Float32(0.9) else Vec3(0.0, 1.0, 0.0)
        var r_v = ref_v.cross(dir).normalized()
        var u_v = dir.cross(r_v).normalized()
        var tv = Float32(i) / Float32(n_st - 1)
        var ao_b = UInt8(max(120, min(250, Int((Float32(0.62) + Float32(0.35) * (Float32(1.0) - tv * Float32(0.3))) * Float32(255.0)))))
        var ring_col = (
            UInt32(0xFF000000)
            | (UInt32(Int(r0 + (r1 - r0) * tv)) << 16)
            | (UInt32(Int(g0 + (g1 - g0) * tv)) << 8)
            | UInt32(Int(b0_c + (b1_c - b0_c) * tv))
        )
        for s in range(stride):
            var tu = Float32(s) / Float32(sides)
            var a = tu * _TAU_RIG
            var n_out = (r_v * cos(a) + u_v * sin(a)).normalized()
            _ = mb.add_vertex(pts[i] + n_out * radii[i], n_out, tu, tv, color=ring_col, ao=ao_b)

    for i in range(n_st - 1):
        for s in range(sides):
            var i0 = base_v + i * stride + s
            mb.add_quad(i0, i0 + 1, i0 + stride + 1, i0 + stride)

    var src = mb^.finish()
    var dst_base = dst.vertex_count()

    for i in range(n_st):
        var jb = seg_bones[i]
        var other_b = seg_bones[min(n_st - 1, i + 1)] if i == 0 else seg_bones[max(0, i - 1)]
        var dist_piv = (pts[i] - joint_pivot).length()
        var w_main = Float32(0.62) if dist_piv < Float32(0.07) else Float32(0.95)
        for s in range(stride):
            var vi = i * stride + s
            dst.positions.append(src.positions[vi * 3 + 0])
            dst.positions.append(src.positions[vi * 3 + 1])
            dst.positions.append(src.positions[vi * 3 + 2])
            dst.normals.append(src.normals[vi * 3 + 0])
            dst.normals.append(src.normals[vi * 3 + 1])
            dst.normals.append(src.normals[vi * 3 + 2])
            dst.texcoords.append(src.texcoords[vi * 2 + 0])
            dst.texcoords.append(src.texcoords[vi * 2 + 1])
            dst.colors.append(src.colors[vi * 4 + 0])
            dst.colors.append(src.colors[vi * 4 + 1])
            dst.colors.append(src.colors[vi * 4 + 2])
            dst.colors.append(src.colors[vi * 4 + 3])
            dst.ao.append(src.ao[vi])
            b0_list.append(jb)
            b1_list.append(other_b)
            w0_list.append(w_main)

    for idx in src.indices:
        dst.indices.append(dst_base + idx)


def _build_base_crowd_rig(with_umbrella: Bool) -> _CrowdRigTemplate:
    """Constructs the bind-pose 16-bone SubD humanoid (`height ~ 1.72m`) with rich 24-bit per-part `colors`."""
    var tpl = _CrowdRigTemplate()
    tpl.parents = [-1, 0, 1, 2, 2, 4, 5, 2, 7, 8, 0, 10, 11, 0, 13, 14]
    tpl.pivots = [
        Vec3( 0.00, 0.92, 0.00),  # 0: Pelvis
        Vec3( 0.00, 1.08, 0.00),  # 1: Spine
        Vec3( 0.00, 1.32, 0.00),  # 2: Chest
        Vec3( 0.00, 1.48, 0.01),  # 3: Head
        Vec3(-0.20, 1.38, 0.00),  # 4: L_UpperArm
        Vec3(-0.24, 1.12, 0.02),  # 5: L_Forearm
        Vec3(-0.25, 0.88, 0.05),  # 6: L_Hand
        Vec3( 0.20, 1.38, 0.00),  # 7: R_UpperArm
        Vec3( 0.24, 1.12, 0.02),  # 8: R_Forearm
        Vec3( 0.25, 0.88, 0.05),  # 9: R_Hand
        Vec3(-0.10, 0.88, 0.00),  # 10: L_Thigh
        Vec3(-0.10, 0.48, 0.01),  # 11: L_Shin
        Vec3(-0.10, 0.08, 0.02),  # 12: L_Foot
        Vec3( 0.10, 0.88, 0.00),  # 13: R_Thigh
        Vec3( 0.10, 0.48, 0.01),  # 14: R_Shin
        Vec3( 0.10, 0.08, 0.02),  # 15: R_Foot
    ]

    tpl.upper.base_color = 0xFFFFFFFF
    tpl.upper.roughness = 0.56
    tpl.upper.metallic = 0.04
    tpl.lower.base_color = 0xFFFFFFFF
    tpl.lower.roughness = 0.60
    tpl.lower.metallic = 0.04

    # Upper Garment: Tailored SubD jacket/coat torso + flared lower coat skirt + sleeves
    var m_chest = _make_subd_armor_box(Vec3(0.0, 1.25, 0.01), Vec3(0.16, 0.19, 0.11), Vec3(0.20, 0.19, 0.13), levels=2, crease=1.8)
    var m_skirt = _make_subd_armor_box(Vec3(0.0, 0.90, 0.00), Vec3(0.21, 0.17, 0.15), Vec3(0.16, 0.17, 0.11), levels=1, crease=2.2)
    _append_crowd_part(tpl.upper, tpl.upper_b0, tpl.upper_b1, tpl.upper_w0, m_chest, 2, 1, tpl.pivots[1], Vec3(0.0, 1.0, 0.0), 0.16, 0xFFFBFDFE)
    _append_crowd_part(tpl.upper, tpl.upper_b0, tpl.upper_b1, tpl.upper_w0, m_skirt, 0, 1, tpl.pivots[1], Vec3(0.0, -1.0, 0.0), 0.14, 0xFFDCE2EC)

    for side in range(2):
        var sx = Float32(-1.0) if side == 0 else Float32(1.0)
        var bu = 4 if side == 0 else 7
        var bf = 5 if side == 0 else 8
        var sl_pts: List[Vec3] = [
            Vec3(sx * Float32(0.19), 1.40, 0.00),
            Vec3(sx * Float32(0.22), 1.25, 0.01),
            Vec3(sx * Float32(0.24), 1.12, 0.02),
            Vec3(sx * Float32(0.25), 0.99, 0.04),
            Vec3(sx * Float32(0.25), 0.89, 0.05),
        ]
        var sl_rad: List[Float32] = [0.064, 0.056, 0.050, 0.047, 0.045]
        var sl_bones: List[Int] = [bu, bu, bf, bf, bf]
        _add_segmented_limb_tube(
            tpl.upper, tpl.upper_b0, tpl.upper_b1, tpl.upper_w0,
            sl_pts, sl_rad, sl_bones, tpl.pivots[bf], sides=6,
            col_top=0xFFF8FAFC, col_bot=0xFFD8DFE9,
        )

    # Lower Body & Gear: Warm skin head/hands (`0xFFECBC9A`), dark espresso hair (`0xFF32221A`),
    # indigo denim trousers (`0xFF324C74` -> `0xFF243856`), dark shoes (`0xFF22262E`), and colorful umbrella!
    var m_head = _make_subd_armor_box(Vec3(0.0, 1.58, 0.01), Vec3(0.095, 0.115, 0.105), Vec3(0.105, 0.115, 0.115), levels=1, crease=1.0)
    var m_hair = _make_subd_armor_box(Vec3(0.0, 1.63, -0.01), Vec3(0.108, 0.085, 0.118), Vec3(0.102, 0.085, 0.112), levels=1, crease=1.6)
    _append_crowd_part(tpl.lower, tpl.lower_b0, tpl.lower_b1, tpl.lower_w0, m_head, 3, 2, tpl.pivots[3], Vec3(0.0, 1.0, 0.0), 0.08, 0xFFECBC9A)
    _append_crowd_part(tpl.lower, tpl.lower_b0, tpl.lower_b1, tpl.lower_w0, m_hair, 3, 3, tpl.pivots[3], Vec3(0.0, 1.0, 0.0), 0.05, 0xFF32221A)

    for side in range(2):
        var sx = Float32(-1.0) if side == 0 else Float32(1.0)
        var bh = 6 if side == 0 else 9
        var bth = 10 if side == 0 else 13
        var bsh = 11 if side == 0 else 14
        var bft = 12 if side == 0 else 15

        var mb_hand = MeshBuilder(base_color=0xFFFFFFFF)
        mb_hand.add_ellipsoid(Vec3(sx * Float32(0.25), 0.85, 0.06), Vec3(0.034, 0.045, 0.036), rings=4, segments=6, color=0xFFECBC9A)
        _append_crowd_part(tpl.lower, tpl.lower_b0, tpl.lower_b1, tpl.lower_w0, mb_hand^.finish(), bh, bh, tpl.pivots[bh], Vec3(0.0, -1.0, 0.0), 0.04, 0xFFECBC9A)

        var leg_pts: List[Vec3] = [
            Vec3(sx * Float32(0.10), 0.90, 0.00),
            Vec3(sx * Float32(0.10), 0.68, 0.01),
            Vec3(sx * Float32(0.10), 0.48, 0.01),
            Vec3(sx * Float32(0.10), 0.28, 0.00),
            Vec3(sx * Float32(0.10), 0.09, 0.00),
        ]
        var leg_rad: List[Float32] = [0.082, 0.072, 0.064, 0.058, 0.050]
        var leg_bones: List[Int] = [bth, bth, bsh, bsh, bsh]
        _add_segmented_limb_tube(
            tpl.lower, tpl.lower_b0, tpl.lower_b1, tpl.lower_w0,
            leg_pts, leg_rad, leg_bones, tpl.pivots[bsh], sides=6,
            col_top=0xFF34507A, col_bot=0xFF243856,
        )

        var m_shoe = _make_subd_armor_box(Vec3(sx * Float32(0.10), 0.048, 0.045), Vec3(0.052, 0.048, 0.115), Vec3(0.048, 0.045, 0.095), top_offset_z=-0.02, levels=1, crease=2.4)
        _append_crowd_part(tpl.lower, tpl.lower_b0, tpl.lower_b1, tpl.lower_w0, m_shoe, bft, bsh, tpl.pivots[bft], Vec3(0.0, -1.0, 0.0), 0.05, 0xFF22262E)

    # 12-ribbed domed rain umbrella with navy/crimson/teal panels & bright chrome shaft/ferrule!
    if with_umbrella:
        var mb_umb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.28, metallic=0.14, double_sided=True)
        var hand_p = Vec3(0.25, 0.88, 0.06)
        var top_p = Vec3(0.06, 1.96, 0.04)
        var axis_u = (top_p - hand_p).normalized()
        var r_u = Vec3(0.0, 0.0, 1.0).cross(axis_u).normalized()
        var f_u = axis_u.cross(r_u).normalized()

        # Brushed-chrome shaft + handle + ferrule spike (`12` tris)
        mb_umb.add_cylinder(hand_p - axis_u * Float32(0.07), top_p + axis_u * Float32(0.08), 0.011, 0.008, segments=6, color=0xFFDCE4EE, caps=False)

        # 12-gore domed canopy with per-gore sRGB color accents (`navy 0xFF1F3558`, `crimson 0xFFD92B4B`, `teal 0xFF1CA6B8`)
        for r_i in range(12):
            var a0 = (Float32(r_i) / Float32(12.0)) * _TAU_RIG
            var a1 = (Float32(r_i + 1) / Float32(12.0)) * _TAU_RIG
            var d0 = r_u * cos(a0) + f_u * sin(a0)
            var d1 = r_u * cos(a1) + f_u * sin(a1)

            var gore_col: UInt32
            if (r_i % 6) == 0:
                gore_col = UInt32(0xFFD92B4B)
            elif (r_i % 6) == 3:
                gore_col = UInt32(0xFF1CA6B8)
            else:
                gore_col = UInt32(0xFF1F3558)

            var pm0 = top_p - axis_u * Float32(0.09) + d0 * Float32(0.32)
            var pm1 = top_p - axis_u * Float32(0.09) + d1 * Float32(0.32)
            var pr0 = top_p - axis_u * Float32(0.25) + d0 * Float32(0.58)
            var pr1 = top_p - axis_u * Float32(0.25) + d1 * Float32(0.58)

            var nm0 = (axis_u * Float32(0.82) + d0 * Float32(0.57)).normalized()
            var nm1 = (axis_u * Float32(0.82) + d1 * Float32(0.57)).normalized()
            var nr0 = (axis_u * Float32(0.52) + d0 * Float32(0.85)).normalized()
            var nr1 = (axis_u * Float32(0.52) + d1 * Float32(0.85)).normalized()

            var v_ap = mb_umb.add_vertex(top_p, axis_u, 0.5, 0.5, color=0xFFDCE4EE, ao=UInt8(255))
            var vm0 = mb_umb.add_vertex(pm0, nm0, 0.3, 0.3, color=gore_col, ao=UInt8(235))
            var vm1 = mb_umb.add_vertex(pm1, nm1, 0.7, 0.3, color=gore_col, ao=UInt8(235))
            var vr0 = mb_umb.add_vertex(pr0, nr0, 0.1, 0.9, color=gore_col, ao=UInt8(195))
            var vr1 = mb_umb.add_vertex(pr1, nr1, 0.9, 0.9, color=gore_col, ao=UInt8(195))

            mb_umb.add_tri(v_ap, vm0, vm1)
            mb_umb.add_quad(vm0, vr0, vr1, vm1)

        # Pass `col = 0xFFFFFFFF` so `_append_crowd_part` preserves each gore & shaft vertex color!
        _append_crowd_part(tpl.lower, tpl.lower_b0, tpl.lower_b1, tpl.lower_w0, mb_umb^.finish(), 9, 9, hand_p, Vec3(0.0, 1.0, 0.0), 0.05, 0xFFFFFFFF)

    return tpl^


def _bake_crowd_pose(
    tpl: _CrowdRigTemplate,
    local_rots: List[Quat],
    pelvis_offset: Vec3,
) -> Tuple[NativeMesh, NativeMesh]:
    """Evaluates hierarchical 16-bone FK + Dual-Quaternion skinning and grounds the stance sole at `y = 0`."""
    var world_m = List[Mat4](length=16, fill=Mat4())
    var dqs = List[_BoneDQ](length=16, fill=_BoneDQ())

    for b in range(16):
        var p = tpl.parents[b]
        var loc_t = (tpl.pivots[0] + pelvis_offset) if p < 0 else (tpl.pivots[b] - tpl.pivots[p])
        var loc_m = trs(loc_t, local_rots[b], Vec3(1.0, 1.0, 1.0))
        var wm = loc_m.copy() if p < 0 else world_m[p].multiply(loc_m)
        world_m[b] = wm.copy()

        var q_world = local_rots[b]
        var cur_p = p
        while cur_p >= 0:
            q_world = local_rots[cur_p].multiply(q_world).normalized()
            cur_p = tpl.parents[cur_p]
        var world_piv = Vec3(wm.at(0, 3), wm.at(1, 3), wm.at(2, 3))
        var skin_t = world_piv - q_world.rotate(tpl.pivots[b])
        dqs[b] = _BoneDQ.from_rot_trans(q_world, skin_t)

    var out_u = tpl.upper.copy()
    var out_l = tpl.lower.copy()

    var min_y = Float32(1e9)
    for i in range(out_u.vertex_count()):
        var p_in = Vec3(tpl.upper.positions[i * 3], tpl.upper.positions[i * 3 + 1], tpl.upper.positions[i * 3 + 2])
        var n_in = Vec3(tpl.upper.normals[i * 3], tpl.upper.normals[i * 3 + 1], tpl.upper.normals[i * 3 + 2])
        var w0 = tpl.upper_w0[i]
        var def_vn = _deform_vertex_dq(p_in, n_in, tpl.upper_b0[i], tpl.upper_b1[i], w0, Float32(1.0) - w0, dqs)
        out_u.positions[i * 3 + 0] = def_vn[0].x
        out_u.positions[i * 3 + 1] = def_vn[0].y
        out_u.positions[i * 3 + 2] = def_vn[0].z
        out_u.normals[i * 3 + 0] = def_vn[1].x
        out_u.normals[i * 3 + 1] = def_vn[1].y
        out_u.normals[i * 3 + 2] = def_vn[1].z
        if def_vn[0].y < min_y:
            min_y = def_vn[0].y

    for i in range(out_l.vertex_count()):
        var p_in = Vec3(tpl.lower.positions[i * 3], tpl.lower.positions[i * 3 + 1], tpl.lower.positions[i * 3 + 2])
        var n_in = Vec3(tpl.lower.normals[i * 3], tpl.lower.normals[i * 3 + 1], tpl.lower.normals[i * 3 + 2])
        var w0 = tpl.lower_w0[i]
        var def_vn = _deform_vertex_dq(p_in, n_in, tpl.lower_b0[i], tpl.lower_b1[i], w0, Float32(1.0) - w0, dqs)
        out_l.positions[i * 3 + 0] = def_vn[0].x
        out_l.positions[i * 3 + 1] = def_vn[0].y
        out_l.positions[i * 3 + 2] = def_vn[0].z
        out_l.normals[i * 3 + 0] = def_vn[1].x
        out_l.normals[i * 3 + 1] = def_vn[1].y
        out_l.normals[i * 3 + 2] = def_vn[1].z
        if def_vn[0].y < min_y:
            min_y = def_vn[0].y

    for i in range(out_u.vertex_count()):
        out_u.positions[i * 3 + 1] -= min_y
    for i in range(out_l.vertex_count()):
        out_l.positions[i * 3 + 1] -= min_y

    out_u.ensure_bounds()
    out_l.ensure_bounds()
    return (out_u^, out_l^)


def build_articulated_crowd_pose_bank(with_umbrella: Bool = False) -> List[Tuple[NativeMesh, NativeMesh]]:
    """Generates `12` biomechanical crowd poses (`8` walk-cycle phases + `2` idle/conversation + `2` cheering/umbrella)
    returning `(garment_upper_mesh, lower_body_and_gear_mesh)` at `~688–760` tris/pose."""
    var tpl = _build_base_crowd_rig(with_umbrella)
    var bank = List[Tuple[NativeMesh, NativeMesh]](capacity=12)

    for k in range(8):
        var ph = (Float32(k) / Float32(8.0)) * _TAU_RIG
        var rots = List[Quat](length=16, fill=Quat())
        var s_ph = sin(ph)
        var c_ph = cos(ph)

        var bob_y = abs(c_ph) * Float32(0.028)
        var pelv_off = Vec3(s_ph * Float32(0.018), bob_y, 0.0)
        rots[0] = _quat_euler_xyz(Float32(0.04), s_ph * Float32(0.08), c_ph * Float32(0.03))
        rots[1] = _quat_euler_xyz(Float32(0.03), -s_ph * Float32(0.06), 0.0)
        rots[2] = _quat_euler_xyz(Float32(0.02), -s_ph * Float32(0.09), -c_ph * Float32(0.02))
        rots[3] = _quat_euler_xyz(Float32(-0.03), s_ph * Float32(0.05), 0.0)

        var l_hip = -s_ph * Float32(0.46)
        var r_hip =  s_ph * Float32(0.46)
        var l_knee = max(Float32(0.08), c_ph * Float32(0.72) + Float32(0.24))
        var r_knee = max(Float32(0.08), -c_ph * Float32(0.72) + Float32(0.24))
        rots[10] = _quat_euler_xyz(l_hip, 0.0, 0.0)
        rots[11] = _quat_euler_xyz(l_knee, 0.0, 0.0)
        rots[12] = _quat_euler_xyz(-(l_hip + l_knee) * Float32(0.45), 0.0, 0.0)

        rots[13] = _quat_euler_xyz(r_hip, 0.0, 0.0)
        rots[14] = _quat_euler_xyz(r_knee, 0.0, 0.0)
        rots[15] = _quat_euler_xyz(-(r_hip + r_knee) * Float32(0.45), 0.0, 0.0)

        var l_arm = s_ph * Float32(0.38)
        var l_elb = Float32(-0.34) - max(Float32(0.0), -s_ph) * Float32(0.30)
        rots[4] = _quat_euler_xyz(l_arm, 0.0, Float32(-0.06))
        rots[5] = _quat_euler_xyz(l_elb, 0.0, 0.0)

        if with_umbrella:
            rots[7] = _quat_euler_xyz(Float32(-0.42) + c_ph * Float32(0.03), Float32(-0.18), Float32(-0.12))
            rots[8] = _quat_euler_xyz(Float32(-0.92), 0.0, 0.0)
            rots[9] = _quat_euler_xyz(Float32(1.28), Float32(0.15), Float32(0.10))
        else:
            var r_arm = -s_ph * Float32(0.38)
            var r_elb = Float32(-0.34) - max(Float32(0.0), s_ph) * Float32(0.30)
            rots[7] = _quat_euler_xyz(r_arm, 0.0, Float32(0.06))
            rots[8] = _quat_euler_xyz(r_elb, 0.0, 0.0)

        bank.append(_bake_crowd_pose(tpl, rots, pelv_off))

    # Pose `8`: Contrapposto Idle
    var r_idle = List[Quat](length=16, fill=Quat())
    r_idle[0] = _quat_euler_xyz(0.0, 0.12, -0.05)
    r_idle[2] = _quat_euler_xyz(0.0, -0.08, 0.05)
    r_idle[3] = _quat_euler_xyz(-0.04, 0.24, 0.0)
    r_idle[4] = _quat_euler_xyz(0.06, 0.0, -0.08)
    r_idle[5] = _quat_euler_xyz(-0.28, 0.0, 0.0)
    r_idle[10] = _quat_euler_xyz(-0.12, 0.08, 0.04)
    r_idle[11] = _quat_euler_xyz(0.22, 0.0, 0.0)
    if with_umbrella:
        r_idle[7] = _quat_euler_xyz(-0.40, -0.16, -0.10)
        r_idle[8] = _quat_euler_xyz(-0.95, 0.0, 0.0)
        r_idle[9] = _quat_euler_xyz(1.30, 0.14, 0.08)
    else:
        r_idle[7] = _quat_euler_xyz(-0.08, 0.0, 0.08)
        r_idle[8] = _quat_euler_xyz(-0.42, -0.18, 0.0)
    bank.append(_bake_crowd_pose(tpl, r_idle, Vec3(0.02, 0.0, 0.0)))

    # Pose `9`: Expressive Conversation
    var r_talk = List[Quat](length=16, fill=Quat())
    r_talk[0] = _quat_euler_xyz(0.0, -0.15, 0.04)
    r_talk[2] = _quat_euler_xyz(0.03, 0.18, -0.04)
    r_talk[3] = _quat_euler_xyz(-0.06, -0.22, 0.05)
    r_talk[4] = _quat_euler_xyz(-0.45, 0.18, -0.12)
    r_talk[5] = _quat_euler_xyz(-0.88, 0.22, 0.0)
    r_talk[13] = _quat_euler_xyz(-0.10, -0.06, 0.0)
    r_talk[14] = _quat_euler_xyz(0.18, 0.0, 0.0)
    if with_umbrella:
        r_talk[7] = _quat_euler_xyz(-0.44, -0.14, -0.10)
        r_talk[8] = _quat_euler_xyz(-0.92, 0.0, 0.0)
        r_talk[9] = _quat_euler_xyz(1.28, 0.12, 0.08)
    else:
        r_talk[7] = _quat_euler_xyz(-0.38, -0.15, 0.12)
        r_talk[8] = _quat_euler_xyz(-0.76, -0.20, 0.0)
    bank.append(_bake_crowd_pose(tpl, r_talk, Vec3(-0.015, 0.0, 0.0)))

    # Pose `10`: Cheering / Rain-Wind Brace
    var r_p10 = List[Quat](length=16, fill=Quat())
    if with_umbrella:
        r_p10[1] = _quat_euler_xyz(0.12, 0.0, 0.0)
        r_p10[3] = _quat_euler_xyz(0.08, 0.0, 0.0)
        r_p10[4] = _quat_euler_xyz(-0.32, 0.25, -0.05)
        r_p10[5] = _quat_euler_xyz(-0.85, 0.0, 0.0)
        r_p10[7] = _quat_euler_xyz(-0.56, -0.22, -0.12)
        r_p10[8] = _quat_euler_xyz(-0.98, 0.0, 0.0)
        r_p10[9] = _quat_euler_xyz(1.18, 0.12, 0.08)
        r_p10[10] = _quat_euler_xyz(-0.22, 0.0, 0.0)
        r_p10[11] = _quat_euler_xyz(0.28, 0.0, 0.0)
        r_p10[13] = _quat_euler_xyz(0.18, 0.0, 0.0)
    else:
        r_p10[2] = _quat_euler_xyz(-0.08, 0.08, 0.0)
        r_p10[3] = _quat_euler_xyz(-0.22, 0.10, 0.0)
        r_p10[4] = _quat_euler_xyz(-2.35, 0.0, -0.28)
        r_p10[5] = _quat_euler_xyz(-0.32, 0.0, 0.0)
        r_p10[7] = _quat_euler_xyz(-0.42, 0.0, 0.14)
        r_p10[8] = _quat_euler_xyz(-0.72, 0.0, 0.0)
    bank.append(_bake_crowd_pose(tpl, r_p10, Vec3(0.0, 0.0, 0.0)))

    # Pose `11`: Dual-Arm Celebration / Skyward Neon Gaze
    var r_p11 = List[Quat](length=16, fill=Quat())
    r_p11[2] = _quat_euler_xyz(-0.10, -0.06, 0.0)
    r_p11[3] = _quat_euler_xyz(-0.28, -0.16, 0.0)
    if with_umbrella:
        r_p11[4] = _quat_euler_xyz(-0.18, 0.0, -0.12)
        r_p11[5] = _quat_euler_xyz(-0.35, 0.0, 0.0)
        r_p11[7] = _quat_euler_xyz(-0.32, -0.16, -0.10)
        r_p11[8] = _quat_euler_xyz(-0.88, 0.0, 0.0)
        r_p11[9] = _quat_euler_xyz(1.36, 0.14, 0.08)
    else:
        r_p11[4] = _quat_euler_xyz(-2.45, 0.0, -0.34)
        r_p11[5] = _quat_euler_xyz(-0.26, 0.0, 0.0)
        r_p11[7] = _quat_euler_xyz(-2.45, 0.0, 0.34)
        r_p11[8] = _quat_euler_xyz(-0.26, 0.0, 0.0)
    bank.append(_bake_crowd_pose(tpl, r_p11, Vec3(0.0, 0.01, 0.0)))

    return bank^
