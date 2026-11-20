"""Flint v5.2 GPU Morph-Target (`BlendShape`) Skinning & `DirectSpringChain` Physics Suite (`test_flint_v5_morph_spring.mojo`).

Zero external asset dependencies. Verifies:
  1. `BIND_AAA_SKINNING` v5.2 layout invariants, `FlintAaaSkinningArena.alloc_morph_targets` bounds clamping,
     and SPIR-V 1.5 validation of `aaa_dq_skinning_shader` with pre-skinning `BlendShape` morph loops.
  2. CPU `skin_vertex_morph_dq_smear_cpu` math: `0`-weight identity, multi-target morph blending +
     renormalized normal deltas + non-uniform `BoneSmear` + 4-bone `DualQuat` skinning.
  3. Standalone `DirectSpringChain` secondary hair/cape/ribbon physics: `120 Hz` gravity droop equilibrium,
     wind gust deflection + `world_link_rot` orientation, `100 m` teleport `max_root_step` clamping
     (zero NaNs / zero stretch explosion, exact link lengths `< 1e-4`), and sphere collider non-penetration.
  4. Live Vulkan GPU `FlintAaaVulkanRenderer` (`128 x 128`) morph-target skinning (`morph_count = 2`) with
     `add_cel_material` (`outline_width_px = 2.5`): verifies GPU `buf_vertices` + `prev_pos` match CPU `< 1e-4`
     and morph weight `1.0` visibly expands BOTH the rendered cel silhouette AND its `CULL_MODE_FRONT` outline
     to the expanded `+X`/`+Y` coordinates with zero NaNs!
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, isnan, isinf, max, min, sqrt
from flint.anim.correctives import RbfPoseDriver, swing_twist_y
from flint.anim.helper_joints import HelperJoints
from flint.anim.pose import Pose, SkeletonInstance, det_quat_to_scaled_angle
from flint.anim.procedural import Placement
from flint.anim.skinning import SKIN_LBS, PALETTE_FLOATS, SkinDraw, write_palette_entry, skin_vertex
from flint.anim.spring_bones import DirectSpringChain, DirectSpringSphereCollider, SpringBones
from flint.assets.manifest import ManifestBone, ManifestVolume, HELPER_TWIST
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh, NativeModel, NativeNode
from flint.core.detmath import PI
from flint.core.ik import angle_axis
from flint.core.math import Vec3
from flint.core.transform import Mat4, Quat, trs
from flint.platform.vulkan import (
    ACCESS_HOST_READ,
    ACCESS_TRANSFER_WRITE,
    PIPELINE_HOST,
    PIPELINE_TRANSFER,
)
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_ibl import SkyDomeParams, bake_bc6h_ibl_cubemap_and_sh9
from flint.render.aaa_npr_cel import AaaCelParams
from flint.render.aaa_post_tsr import AAA_DISPLAY_LINEAR_SRGB
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_skinning_rt import (
    AAA_MAX_SKIN_BONES,
    AAA_SKIN_BIND_STRIDE_WORDS,
    AAA_SKIN_BIND_VERTS_BASE,
    AAA_SKIN_BUFFER_WORDS,
    AAA_SKIN_CUR_BONES_BASE,
    AAA_SKIN_CUR_MORPH_WEIGHTS_BASE,
    AAA_SKIN_MAX_MORPH_DELTAS,
    AAA_SKIN_MAX_MORPH_SLOTS,
    AAA_SKIN_MAX_VERTS,
    AAA_SKIN_MORPH_DELTAS_BASE,
    AAA_SKIN_PREV_BONES_BASE,
    AAA_SKIN_PREV_MORPH_WEIGHTS_BASE,
    AAA_SKIN_PREV_POS_BASE,
    AAA_SKIN_PREV_SMEAR_BASE,
    BoneSmear,
    DualQuat,
    FlintAaaSkinningArena,
    aaa_dq_skinning_shader,
    normalize_and_sort_skin_weights_4,
    skin_vertex_dq_cpu,
    skin_vertex_dq_smear_cpu,
    skin_vertex_morph_dq_smear_cpu,
)
from flint.render.spirv_validate import validate_spirv_module


def _fail(msg: String):
    print("  [FAIL]", msg)
    external_call["exit", NoneType](Int32(1))


def _assert(cond: Bool, msg: String):
    if not cond:
        _fail(msg)


def _assert_approx(actual: Float32, expected: Float32, eps: Float32, msg: String):
    _assert(abs(actual - expected) <= eps, msg)


def _is_finite_vec3(v: Vec3) -> Bool:
    return (
        not isnan(v.x)
        and not isnan(v.y)
        and not isnan(v.z)
        and not isinf(v.x)
        and not isinf(v.y)
        and not isinf(v.z)
    )


def _unpack_rgb(argb: UInt32) -> Vec3:
    var r = Float32((argb >> 16) & 0xFF) / Float32(255.0)
    var g = Float32((argb >> 8) & 0xFF) / Float32(255.0)
    var b = Float32(argb & 0xFF) / Float32(255.0)
    return Vec3(r, g, b)


def _make_sphere(radius: Float32) -> NativeMesh:
    var mb = MeshBuilder()
    mb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(radius, radius, radius))
    return mb^.finish()


def _readback_device_local_floats(
    mut ren: FlintAaaVulkanRenderer, use_skinning_buf: Bool, src_byte_offset: Int, n_floats: Int
) -> List[Float32]:
    var out = List[Float32](length=n_floats, fill=Float32(0.0))
    if not ren.active or n_floats <= 0:
        return out^
    if use_skinning_buf and ren.buf_skinning.vk_buffer == 0:
        return out^
    if (not use_skinning_buf) and ren.buf_vertices.vk_buffer == 0:
        return out^
    var rb = ren.vk.create_buffer(n_floats * 4, local=False, cpu_reads=True)
    if rb.vk_buffer != 0 and rb.cpu_address != 0:
        ren.vk.wait_idle()
        if ren.vk.begin_commands(0):
            if use_skinning_buf:
                ren.vk.copy_buffer(0, ren.buf_skinning, src_byte_offset, rb, 0, n_floats * 4)
            else:
                ren.vk.copy_buffer(0, ren.buf_vertices, src_byte_offset, rb, 0, n_floats * 4)
            ren.vk.memory_barrier2(
                0, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ)
            )
            var f = ren.vk.submit(0)
            if f != 0 and ren.vk.wait(f):
                _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), rb.cpu_address, n_floats * 4)
    ren.vk.destroy_buffer(rb)
    return out^


def test_layout_arena_and_spirv():
    # Verify legacy v5.1 offsets [0 .. 1,933,312) are 100% preserved and v5.2 morph regions append cleanly
    _assert(AAA_MAX_SKIN_BONES == 2048, "AAA_MAX_SKIN_BONES == 2048")
    _assert(AAA_SKIN_CUR_BONES_BASE == 0, "AAA_SKIN_CUR_BONES_BASE == 0")
    _assert(AAA_SKIN_PREV_BONES_BASE == 16384, "AAA_SKIN_PREV_BONES_BASE == 16384")
    _assert(AAA_SKIN_BIND_VERTS_BASE == 32768, "AAA_SKIN_BIND_VERTS_BASE == 32768")
    _assert(AAA_SKIN_MAX_VERTS == 98304, "AAA_SKIN_MAX_VERTS == 98304")
    _assert(
        AAA_SKIN_PREV_POS_BASE == 32768 + 98304 * AAA_SKIN_BIND_STRIDE_WORDS,
        "AAA_SKIN_PREV_POS_BASE == 32768 + 98304 * AAA_SKIN_BIND_STRIDE_WORDS",
    )
    _assert(
        AAA_SKIN_PREV_SMEAR_BASE + AAA_MAX_SKIN_BONES * 8 == 1933312,
        "Legacy skinning region ends at exact word 1,933,312",
    )
    _assert(AAA_SKIN_MAX_MORPH_SLOTS == 256, "AAA_SKIN_MAX_MORPH_SLOTS == 256")
    _assert(AAA_SKIN_MAX_MORPH_DELTAS == 262144, "AAA_SKIN_MAX_MORPH_DELTAS == 262,144")
    _assert(AAA_SKIN_CUR_MORPH_WEIGHTS_BASE == 1933312, "AAA_SKIN_CUR_MORPH_WEIGHTS_BASE == 1,933,312")
    _assert(AAA_SKIN_PREV_MORPH_WEIGHTS_BASE == 1933568, "AAA_SKIN_PREV_MORPH_WEIGHTS_BASE == 1,933,568")
    _assert(AAA_SKIN_MORPH_DELTAS_BASE == 1933824, "AAA_SKIN_MORPH_DELTAS_BASE == 1,933,824")
    _assert(AAA_SKIN_BUFFER_WORDS == 3506688, "AAA_SKIN_BUFFER_WORDS == 3,506,688")

    # Verify FlintAaaSkinningArena morph sub-allocation and saturation clamping
    var arena = FlintAaaSkinningArena()
    var a0 = arena.alloc_morph_targets(4, 1000)
    _assert(a0[0] == 0 and a0[1] == 0 and a0[2] == 4, "First morph allocation starts at (0, 0) with 4 targets")
    var a1 = arena.alloc_morph_targets(3, 500)
    _assert(a1[0] == 4 and a1[1] == 4000 and a1[2] == 3, "Second morph allocation advances cursors to (4, 4000)")
    var a_clamp_slots = arena.alloc_morph_targets(300, 10)
    _assert(a_clamp_slots[2] == 256 - 7, "Morph slot overflow clamps cleanly to remaining AAA_SKIN_MAX_MORPH_SLOTS")
    arena.reset()
    _assert(arena.morph_weight_cursor == 0 and arena.morph_delta_cursor == 0, "arena.reset() zeroes morph cursors")
    var a_clamp_deltas = arena.alloc_morph_targets(20, 20000)
    _assert(
        a_clamp_deltas[2] == AAA_SKIN_MAX_MORPH_DELTAS // 20000,
        "Morph delta overflow clamps cleanly to AAA_SKIN_MAX_MORPH_DELTAS",
    )

    var spv = aaa_dq_skinning_shader(0, 13)
    var rep = validate_spirv_module(spv)
    if not rep.valid:
        var err = rep.errors[0] if len(rep.errors) > 0 else String("unknown SPIR-V error")
        _fail("aaa_dq_skinning_shader SPIR-V validation failed: " + err)
    print("  [PASS] Suite 1: BIND_AAA_SKINNING v5.2 layout, arena clamping & SPIR-V 1.5 shader validated")


def test_cpu_morph_dq_smear_math():
    var bones = List[DualQuat]()
    bones.append(DualQuat())
    bones.append(DualQuat.from_quat_trans(Quat(0.0, 0.0, 0.3826834, 0.9238795), Vec3(0.35, -0.15, 0.25)))

    var smears = List[BoneSmear]()
    smears.append(BoneSmear())
    smears.append(BoneSmear(scale=Vec3(1.65, 0.78, 0.78), shear_xy=0.22, pivot_bind=Vec3(0.10, 0.05, 0.0)))

    var bp = Vec3(0.45, 0.30, -0.20)
    var bn = Vec3(0.60, 0.80, 0.0).normalized()

    var pos_deltas = List[Vec3]()
    pos_deltas.append(Vec3(0.40, 0.25, -0.10))
    pos_deltas.append(Vec3(-0.15, 0.35, 0.20))
    pos_deltas.append(Vec3(0.05, -0.10, 0.30))

    var nrm_deltas = List[Vec3]()
    nrm_deltas.append(Vec3(0.25, 0.15, 0.05))
    nrm_deltas.append(Vec3(-0.10, 0.30, -0.15))
    nrm_deltas.append(Vec3(0.0, -0.05, 0.20))

    # 1. Empty weights & explicit 0.0 weights must match skin_vertex_dq_smear_cpu identically
    var empty_w = List[Float32]()
    var zero_w: List[Float32] = [Float32(0.0), Float32(0.0), Float32(0.0)]
    var ref_base = skin_vertex_dq_smear_cpu(bones, smears, 1, 0, 0, 0, 0.70, 0.30, 0.0, 0.0, bp, bn)
    var res_empty = skin_vertex_morph_dq_smear_cpu(
        bones, smears, empty_w, pos_deltas, nrm_deltas, 1, 0, 0, 0, 0.70, 0.30, 0.0, 0.0, bp, bn
    )
    var res_zero = skin_vertex_morph_dq_smear_cpu(
        bones, smears, zero_w, pos_deltas, nrm_deltas, 1, 0, 0, 0, 0.70, 0.30, 0.0, 0.0, bp, bn
    )
    _assert((res_empty[0] - ref_base[0]).length() < Float32(1e-6), "Empty morph weights match base smear-DQ position")
    _assert((res_empty[1] - ref_base[1]).length() < Float32(1e-6), "Empty morph weights match base smear-DQ normal")
    _assert((res_zero[0] - ref_base[0]).length() < Float32(1e-6), "Zero morph weights match base smear-DQ position")
    _assert((res_zero[1] - ref_base[1]).length() < Float32(1e-6), "Zero morph weights match base smear-DQ normal")

    # 2. Multi-target morph blend + BoneSmear + DualQuat rotation/translation
    var active_w: List[Float32] = [Float32(0.75), Float32(0.40), Float32(-0.20)]
    var exp_bp = bp + pos_deltas[0] * Float32(0.75) + pos_deltas[1] * Float32(0.40) + pos_deltas[2] * Float32(-0.20)
    var exp_bn = (
        bn + nrm_deltas[0] * Float32(0.75) + nrm_deltas[1] * Float32(0.40) + nrm_deltas[2] * Float32(-0.20)
    ).normalized()
    var exp_out = skin_vertex_dq_smear_cpu(bones, smears, 1, 0, 0, 0, 0.70, 0.30, 0.0, 0.0, exp_bp, exp_bn)
    var got_out = skin_vertex_morph_dq_smear_cpu(
        bones, smears, active_w, pos_deltas, nrm_deltas, 1, 0, 0, 0, 0.70, 0.30, 0.0, 0.0, bp, bn
    )
    _assert((got_out[0] - exp_out[0]).length() < Float32(1e-6), "Multi-target morph + Smear + DualQuat pos exact")
    _assert((got_out[1] - exp_out[1]).length() < Float32(1e-6), "Multi-target morph + Smear + DualQuat nrm exact")
    _assert(abs(got_out[1].length() - Float32(1.0)) < Float32(1e-5), "Deformed morph normal is unit length")
    _assert((got_out[0] - ref_base[0]).length() > Float32(0.25), "Active morph targets displace vertex significantly")

    # Also verify pure rigid DualQuat compatibility when smears are identity
    var id_smears: List[BoneSmear] = [BoneSmear(), BoneSmear()]
    var got_rigid_morph = skin_vertex_morph_dq_smear_cpu(
        bones, id_smears, active_w, pos_deltas, nrm_deltas, 1, 0, 0, 0, 0.70, 0.30, 0.0, 0.0, bp, bn
    )
    var exp_rigid_morph = skin_vertex_dq_cpu(bones, 1, 0, 0, 0, 0.70, 0.30, 0.0, 0.0, exp_bp, exp_bn)
    _assert(
        (got_rigid_morph[0] - exp_rigid_morph[0]).length() < Float32(1e-5),
        "Morph + identity smear matches pre-morphed rigid DualQuat",
    )
    print("  [PASS] Suite 2: CPU skin_vertex_morph_dq_smear_cpu multi-target + Smear + DualQuat verified")


def test_direct_spring_chain_physics():
    var rest_offs = List[Vec3]()
    rest_offs.append(Vec3(0.20, 0.0, 0.0))
    rest_offs.append(Vec3(0.20, 0.0, 0.0))
    rest_offs.append(Vec3(0.20, 0.0, 0.0))
    rest_offs.append(Vec3(0.20, 0.0, 0.0))

    # Verify default constructor parameters (stiffness=18.0, gravity=-6.5) droops finitely in -Y
    var stiff_bang = DirectSpringChain(rest_offs)
    for _ in range(30):
        stiff_bang.step(Float32(1.0 / 60.0), Vec3(0.0, 1.5, 0.0), Quat())
    _assert(stiff_bang.world_tail_pos(3).y < Float32(1.5) - Float32(0.008), "Default stiff chain droops cleanly in -Y")

    var chain = DirectSpringChain(
        rest_offs,
        stiffness=1.8,
        drag=0.14,
        gravity=-7.5,
        wind_scale=1.0,
        link_radius=0.025,
        max_root_step=0.35,
    )
    _assert(chain.link_count() == 4, "DirectSpringChain has 4 links")
    var root0 = Vec3(0.0, 1.5, 0.0)
    var rot0 = Quat()
    chain.reset(root0, rot0)

    # 1. Gravity droop equilibrium (2.0 seconds @ 60 Hz)
    for _ in range(120):
        chain.step(Float32(1.0 / 60.0), root0, rot0)

    _assert((chain.world_head_pos(0) - root0).length() < Float32(1e-6), "Link 0 head anchored at root_world_pos")
    for i in range(4):
        var h_p = chain.world_head_pos(i)
        var t_p = chain.world_tail_pos(i)
        _assert(_is_finite_vec3(h_p) and _is_finite_vec3(t_p), "Gravity equilibrium positions finite")
        var seg_len = (t_p - h_p).length()
        _assert(abs(seg_len - Float32(0.20)) < Float32(1e-4), "Gravity equilibrium preserves exact link length < 1e-4")
        if i > 0:
            _assert((h_p - chain.world_tail_pos(i - 1)).length() < Float32(1e-6), "Child head equals parent tail")
        _assert(t_p.y < h_p.y - Float32(0.02), "Horizontal spring link droops downward under gravity")
        var q_link = chain.world_link_rot(i)
        var oriented_seg = q_link.rotate(rest_offs[i])
        _assert(
            (oriented_seg - (t_p - h_p)).length() < Float32(1e-4),
            "world_link_rot orients rest_local_offsets[i] along live link",
        )

    # 2. Wind gust deflection (+Z wind)
    for _ in range(90):
        chain.step(Float32(1.0 / 60.0), root0, rot0, wind_world=Vec3(0.0, 0.0, 18.0))
    var tip_wind = chain.world_tail_pos(3)
    _assert(_is_finite_vec3(tip_wind) and tip_wind.z > Float32(0.20), "Wind gust deflects chain tip along +Z")
    for i in range(4):
        var seg_len = (chain.world_tail_pos(i) - chain.world_head_pos(i)).length()
        _assert(abs(seg_len - Float32(0.20)) < Float32(1e-4), "Wind gust preserves exact link length < 1e-4")

    # 3. 100 m instantaneous teleport (`max_root_step` clamping)
    var teleport_root = Vec3(100.0, 21.5, -50.0)
    chain.step(Float32(1.0 / 60.0), teleport_root, rot0, wind_world=Vec3(0.0, 0.0, 0.0))
    _assert(
        (chain.world_head_pos(0) - teleport_root).length() < Float32(1e-5),
        "Teleport anchors link 0 head at new 100 m root",
    )
    for i in range(4):
        var h_p = chain.world_head_pos(i)
        var t_p = chain.world_tail_pos(i)
        _assert(_is_finite_vec3(h_p) and _is_finite_vec3(t_p), "Post-teleport positions have zero NaNs/Infs")
        var seg_len = (t_p - h_p).length()
        _assert(abs(seg_len - Float32(0.20)) < Float32(1e-4), "Post-teleport preserves exact link length < 1e-4")
        _assert((t_p - teleport_root).length() <= Float32(0.805), "Post-teleport chain stays bounded within total reach")

    # Step 10 more frames after teleport to ensure Verlet velocity didn't explode
    for _ in range(10):
        chain.step(Float32(1.0 / 60.0), teleport_root, rot0)
    for i in range(4):
        var seg_len = (chain.world_tail_pos(i) - chain.world_head_pos(i)).length()
        _assert(_is_finite_vec3(chain.world_tail_pos(i)), "Post-teleport follow-up frames finite")
        _assert(abs(seg_len - Float32(0.20)) < Float32(1e-4), "Post-teleport follow-up preserves link length < 1e-4")

    # 4. Sphere collider non-penetration (hair strand draping over a head/shoulder sphere)
    var hair_offs = List[Vec3]()
    hair_offs.append(Vec3(0.02, -0.22, 0.0))
    hair_offs.append(Vec3(0.0, -0.22, 0.0))
    hair_offs.append(Vec3(0.0, -0.22, 0.0))
    var hair = DirectSpringChain(
        hair_offs,
        stiffness=10.0,
        drag=0.18,
        gravity=-8.0,
        wind_scale=1.0,
        link_radius=0.03,
        max_root_step=0.35,
    )
    var col_local = Vec3(0.0, -0.24, 0.0)
    var col_radius = Float32(0.18)
    var col_test_collider = DirectSpringSphereCollider(col_local, col_radius)
    _assert(abs(col_test_collider.radius - col_radius) < Float32(1e-6), "DirectSpringSphereCollider init")
    hair.add_sphere_collider(col_local, col_radius)

    var head_pos = Vec3(0.0, 1.70, 0.0)
    var head_rot = Quat(0.0, 0.2588190, 0.0, 0.9659258)  # 30 deg yaw to test root-rotated collider frame
    hair.reset(head_pos, head_rot)
    for _ in range(120):
        hair.step(Float32(1.0 / 60.0), head_pos, head_rot)

    var col_world = head_pos + head_rot.rotate(col_local)
    var min_reach = col_radius + hair.link_radius
    for i in range(hair.link_count()):
        var h_p = hair.world_head_pos(i)
        var t_p = hair.world_tail_pos(i)
        _assert(_is_finite_vec3(t_p), "Collider-draped hair tail finite")
        var d_col = (t_p - col_world).length()
        _assert(
            d_col >= min_reach - Float32(1e-4),
            "Sphere collider prevents hair penetration (dist=" + String(d_col) + ", min=" + String(min_reach) + ")",
        )
        var l_err = abs((t_p - h_p).length() - hair_offs[i].length())
        _assert(
            l_err < Float32(1e-4),
            "Collider push-out preserves exact inextensible link length < 1e-4 (err=" + String(l_err) + ")",
        )
    print("  [PASS] Suite 3: DirectSpringChain (gravity, wind, 100 m teleport clamp & sphere collider) verified")


def _analyze_cel_and_outline_bounds(
    px: List[UInt32], w: Int, h: Int
) -> Tuple[Int, Int, Int, Int, Int, Int, Int, Int, Int, Int]:
    """Returns `(cel_cnt, cel_min_x, cel_max_x, cel_min_y, cel_max_y, out_cnt, out_min_x, out_max_x, out_min_y, out_max_y)`."""
    var cel_cnt = 0
    var cel_min_x = w
    var cel_max_x = -1
    var cel_min_y = h
    var cel_max_y = -1

    var out_cnt = 0
    var out_min_x = w
    var out_max_x = -1
    var out_min_y = h
    var out_max_y = -1

    for y in range(2, h - 2):
        for x in range(2, w - 2):
            var c = _unpack_rgb(px[y * w + x])
            _assert(_is_finite_vec3(c), "Readback pixel finite")
            var luma = c.x * Float32(0.2126) + c.y * Float32(0.7152) + c.z * Float32(0.0722)
            # Dark ink hardware outline (`outline_color ~ 0.005`) vs cool sky background (`~0.55..0.85` blue-grey)
            # vs warm red-orange cel surface (`r > 0.30`, `r > b * 1.8`)
            if luma < Float32(0.06):
                out_cnt += 1
                if x < out_min_x:
                    out_min_x = x
                if x > out_max_x:
                    out_max_x = x
                if y < out_min_y:
                    out_min_y = y
                if y > out_max_y:
                    out_max_y = y
            elif c.x > Float32(0.28) and c.x > c.z * Float32(1.75):
                cel_cnt += 1
                if x < cel_min_x:
                    cel_min_x = x
                if x > cel_max_x:
                    cel_max_x = x
                if y < cel_min_y:
                    cel_min_y = y
                if y > cel_max_y:
                    cel_max_y = y

    return (
        cel_cnt, cel_min_x, cel_max_x, cel_min_y, cel_max_y,
        out_cnt, out_min_x, out_max_x, out_min_y, out_max_y,
    )


def test_gpu_morph_skinning_cel_and_outline():
    var w = 128
    var h = 128
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
    sky_p.zenith_rgb = Vec3(0.50, 0.58, 0.72)
    sky_p.horizon_rgb = Vec3(0.50, 0.58, 0.72)
    sky_p.ground_rgb = Vec3(0.45, 0.52, 0.65)
    sky_p.sun_lobe_rgb = Vec3(0.0, 0.0, 0.0)
    ren.upload_ibl_environment(bake_bc6h_ibl_cubemap_and_sh9(sky_p, base_dim=32, mip_levels=5))
    ren.sun_dir = Vec3(0.45, 0.45, -0.77).normalized()
    ren.sun_radiance = Vec3(3.2, 3.1, 3.0)
    ren.ibl_intensity = Float32(0.55)

    var base_mat = AaaMaterial(base_color=Vec3(0.92, 0.18, 0.06), roughness=0.5, metallic=0.0)
    var cel_p = AaaCelParams.skin(
        sss_shadow_tint=Vec3(0.75, 0.30, 0.25),
        outline_width_px=2.5,
        outline_color=Vec3(0.004, 0.004, 0.006),
    )
    cel_p.spec_intensity = Float32(0.0)
    cel_p.rim_color = Vec3(0.0, 0.0, 0.0)
    var cel_mat_id = ren.add_cel_material(base_mat, cel_p)
    _assert(cel_mat_id >= 0, "Registered cel + 2.5px outline material")

    # Build a skinned head/sphere mesh (`radius = 0.46`) with `morph_count = 2`:
    # - Morph 0: expands `+X` and `+Y` silhouette smoothly by `+0.45` (so upper-right silhouette expands from 0.46 -> 0.91!)
    # - Morph 1: expands `-X` silhouette by `+0.30`
    var head_mesh = _make_sphere(0.46)
    var v_cnt = head_mesh.vertex_count()
    head_mesh.morph_count = 2
    head_mesh.morph_positions = List[Float32](length=2 * v_cnt * 3, fill=0.0)
    head_mesh.morph_normals = List[Float32](length=2 * v_cnt * 3, fill=0.0)

    var j_packed = List[UInt32](capacity=v_cnt)
    var j_weights = List[Vec3](capacity=v_cnt)
    for vi in range(v_cnt):
        var nx = head_mesh.normals[vi * 3 + 0]
        var ny = head_mesh.normals[vi * 3 + 1]
        var nz = head_mesh.normals[vi * 3 + 2]

        # Morph 0: smooth +X and +Y expansion by +0.45 along positive X/Y lobes
        var wx_pos = max(Float32(0.0), nx)
        var wy_pos = max(Float32(0.0), ny)
        var dx0 = Float32(0.45) * wx_pos
        var dy0 = Float32(0.45) * wy_pos
        head_mesh.morph_positions[vi * 3 + 0] = dx0
        head_mesh.morph_positions[vi * 3 + 1] = dy0
        head_mesh.morph_positions[vi * 3 + 2] = Float32(0.0)
        # Keep morph 0 surface normals smooth & outward-pointing for clean CULL_MODE_FRONT outline extrusion
        var n0_target = Vec3(nx + Float32(0.25) * wx_pos, ny + Float32(0.25) * wy_pos, nz).normalized()
        head_mesh.morph_normals[vi * 3 + 0] = n0_target.x - nx
        head_mesh.morph_normals[vi * 3 + 1] = n0_target.y - ny
        head_mesh.morph_normals[vi * 3 + 2] = n0_target.z - nz

        # Morph 1: smooth -X expansion by +0.30
        var m1_base = (v_cnt + vi) * 3
        var wx_neg = max(Float32(0.0), -nx)
        head_mesh.morph_positions[m1_base + 0] = Float32(-0.30) * wx_neg
        head_mesh.morph_positions[m1_base + 1] = Float32(0.0)
        head_mesh.morph_positions[m1_base + 2] = Float32(0.0)
        head_mesh.morph_normals[m1_base + 0] = Float32(-0.15) * wx_neg
        head_mesh.morph_normals[m1_base + 1] = Float32(0.0)
        head_mesh.morph_normals[m1_base + 2] = Float32(0.0)

        var w1 = max(Float32(0.0), min(Float32(1.0), ny * Float32(0.5) + Float32(0.5)))
        j_packed.append(UInt32(0) | (UInt32(1) << 8))
        j_weights.append(Vec3(Float32(1.0) - w1, w1, 0.0))

    var sk_entry = ren.add_skinned_mesh(head_mesh, j_packed, j_weights, cel_mat_id, bone_count=4)
    _assert(sk_entry.morph_count == 2, "add_skinned_mesh registered morph_count == 2")

    # Frame 1: Render with morph weight = [0.0, 0.0]
    var w_zero: List[Float32] = [Float32(0.0), Float32(0.0)]
    ren.update_skinned_mesh_morphs(sk_entry, w_zero)
    ren.begin_frame(cam_pos=Vec3(0.0, 0.0, -2.8), cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(sk_entry.mesh_entry.mesh_id, Mat4())
    _assert(ren.render_frame(capture_readback=True), "Rendered morph weight 0.0 frame")
    var px_w0 = ren.readback_pixels()
    var b0 = _analyze_cel_and_outline_bounds(px_w0, w, h)
    _assert(b0[0] > 400, "Weight 0.0 rendered warm cel sphere interior (cnt=" + String(b0[0]) + ")")
    _assert(b0[5] > 80, "Weight 0.0 rendered CULL_MODE_FRONT dark outline ring (cnt=" + String(b0[5]) + ")")

    # Frame 2: Render with morph weight = [1.0, 0.0] (and prev_weights = [0.0, 0.0] to also test TSR prev_pos!)
    var w_one: List[Float32] = [Float32(1.0), Float32(0.0)]
    ren.update_skinned_mesh_morphs(sk_entry, w_one, w_zero)
    ren.begin_frame(cam_pos=Vec3(0.0, 0.0, -2.8), cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(sk_entry.mesh_entry.mesh_id, Mat4())
    _assert(ren.render_frame(capture_readback=True), "Rendered morph weight 1.0 frame")
    var px_w1 = ren.readback_pixels()
    var b1 = _analyze_cel_and_outline_bounds(px_w1, w, h)

    # Verify deformed GPU vertices & previous-frame positions match CPU `skin_vertex_morph_dq_smear_cpu` < 1e-4
    var gpu_v = _readback_device_local_floats(
        ren, False, sk_entry.mesh_entry.first_vertex * 48, v_cnt * 12
    )
    var gpu_prev = _readback_device_local_floats(
        ren, True, (AAA_SKIN_PREV_POS_BASE + sk_entry.skin_bind_first_slot * 3) * 4, v_cnt * 3
    )

    var id_bones: List[DualQuat] = [DualQuat(), DualQuat()]
    var id_smears: List[BoneSmear] = [BoneSmear(), BoneSmear()]
    for vi in range(v_cnt):
        var bp = Vec3(head_mesh.positions[vi * 3], head_mesh.positions[vi * 3 + 1], head_mesh.positions[vi * 3 + 2])
        var bn = Vec3(head_mesh.normals[vi * 3], head_mesh.normals[vi * 3 + 1], head_mesh.normals[vi * 3 + 2])
        var pd: List[Vec3] = [
            Vec3(
                head_mesh.morph_positions[vi * 3],
                head_mesh.morph_positions[vi * 3 + 1],
                head_mesh.morph_positions[vi * 3 + 2],
            ),
            Vec3(
                head_mesh.morph_positions[(v_cnt + vi) * 3],
                head_mesh.morph_positions[(v_cnt + vi) * 3 + 1],
                head_mesh.morph_positions[(v_cnt + vi) * 3 + 2],
            ),
        ]
        var nd: List[Vec3] = [
            Vec3(
                head_mesh.morph_normals[vi * 3],
                head_mesh.morph_normals[vi * 3 + 1],
                head_mesh.morph_normals[vi * 3 + 2],
            ),
            Vec3(
                head_mesh.morph_normals[(v_cnt + vi) * 3],
                head_mesh.morph_normals[(v_cnt + vi) * 3 + 1],
                head_mesh.morph_normals[(v_cnt + vi) * 3 + 2],
            ),
        ]
        var wv = j_weights[vi]
        var cpu_cur = skin_vertex_morph_dq_smear_cpu(
            id_bones, id_smears, w_one, pd, nd, 0, 1, 0, 0, wv.x, wv.y, 0.0, 0.0, bp, bn
        )
        var cpu_prv = skin_vertex_morph_dq_smear_cpu(
            id_bones, id_smears, w_zero, pd, nd, 0, 1, 0, 0, wv.x, wv.y, 0.0, 0.0, bp, bn
        )
        var gp = Vec3(gpu_v[vi * 12 + 0], gpu_v[vi * 12 + 1], gpu_v[vi * 12 + 2])
        var gn = Vec3(gpu_v[vi * 12 + 3], gpu_v[vi * 12 + 4], gpu_v[vi * 12 + 5])
        var g_prev = Vec3(gpu_prev[vi * 3 + 0], gpu_prev[vi * 3 + 1], gpu_prev[vi * 3 + 2])
        _assert(_is_finite_vec3(gp) and _is_finite_vec3(gn) and _is_finite_vec3(g_prev), "GPU deformed vertex finite")
        _assert((gp - cpu_cur[0]).length() < Float32(1e-4), "GPU morph position matches CPU < 1e-4")
        _assert((gn - cpu_cur[1]).length() < Float32(1e-3), "GPU morph normal matches CPU < 1e-3")
        _assert((g_prev - cpu_prv[0]).length() < Float32(1e-4), "GPU prev_weights position matches CPU < 1e-4")

    # Verify silhouette & hardware outline expansion in rendered 128x128 framebuffer!
    var cel_span_x0 = b0[2] - b0[1]
    var cel_span_y0 = b0[4] - b0[3]
    var cel_span_x1 = b1[2] - b1[1]
    var cel_span_y1 = b1[4] - b1[3]

    var out_span_x0 = b0[7] - b0[6]
    var out_span_y0 = b0[9] - b0[8]
    var out_span_x1 = b1[7] - b1[6]
    var out_span_y1 = b1[9] - b1[8]

    _assert(
        b1[0] > b0[0] + 350,
        "Morph weight 1.0 significantly expands cel silhouette area (" + String(b0[0]) + " -> " + String(b1[0]) + ")",
    )
    _assert(
        cel_span_x1 >= cel_span_x0 + 16 and cel_span_y1 >= cel_span_y0 + 16,
        "Morph weight 1.0 expands cel X/Y spans by >= 16 px (dx="
        + String(cel_span_x1 - cel_span_x0)
        + ", dy="
        + String(cel_span_y1 - cel_span_y0)
        + ")",
    )
    _assert(
        out_span_x1 >= out_span_x0 + 16 and out_span_y1 >= out_span_y0 + 16,
        "Morph weight 1.0 expands CULL_MODE_FRONT outline X/Y spans by >= 16 px (dx="
        + String(out_span_x1 - out_span_x0)
        + ", dy="
        + String(out_span_y1 - out_span_y0)
        + ")",
    )
    # Outline always wraps outside the expanded cel silhouette on all 4 sides
    _assert(
        b1[6] <= b1[1] and b1[7] >= b1[2] and b1[8] <= b1[3] and b1[9] >= b1[4],
        "CULL_MODE_FRONT outline cleanly wraps outside the expanded morph cel silhouette",
    )

    # Also test simultaneous multi-target morphs + DualQuat rotation + BoneSmear without duplicate dispatch
    var rot_bones: List[DualQuat] = [
        DualQuat(),
        DualQuat.from_quat_trans(Quat(0.0, 0.0, 0.1736482, 0.9848078), Vec3(0.05, 0.08, 0.0)),
    ]
    var act_smears: List[BoneSmear] = [
        BoneSmear(),
        BoneSmear(scale=Vec3(1.20, 0.90, 0.90), shear_xy=0.12, pivot_bind=Vec3(0.0, 0.0, 0.0)),
    ]
    var w_both: List[Float32] = [Float32(0.65), Float32(0.80)]
    ren.update_skinned_mesh_morphs(sk_entry, w_both, w_one)
    _ = ren.update_skinned_mesh_smear(sk_entry, rot_bones, id_bones, act_smears, id_smears, defer_to_frame=True)
    _assert(len(ren.pending_skin_updates) == 1, "Combined morph + smear deferred update deduplicates to 1 dispatch")

    ren.begin_frame(cam_pos=Vec3(0.0, 0.0, -2.8), cam_target=Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(sk_entry.mesh_entry.mesh_id, Mat4())
    _assert(ren.render_frame(capture_readback=True), "Rendered combined Morph + BoneSmear + DualQuat frame")
    gpu_v = _readback_device_local_floats(
        ren, False, sk_entry.mesh_entry.first_vertex * 48, v_cnt * 12
    )
    for vi in range(0, v_cnt, 7):
        var bp = Vec3(head_mesh.positions[vi * 3], head_mesh.positions[vi * 3 + 1], head_mesh.positions[vi * 3 + 2])
        var bn = Vec3(head_mesh.normals[vi * 3], head_mesh.normals[vi * 3 + 1], head_mesh.normals[vi * 3 + 2])
        var pd: List[Vec3] = [
            Vec3(
                head_mesh.morph_positions[vi * 3],
                head_mesh.morph_positions[vi * 3 + 1],
                head_mesh.morph_positions[vi * 3 + 2],
            ),
            Vec3(
                head_mesh.morph_positions[(v_cnt + vi) * 3],
                head_mesh.morph_positions[(v_cnt + vi) * 3 + 1],
                head_mesh.morph_positions[(v_cnt + vi) * 3 + 2],
            ),
        ]
        var nd: List[Vec3] = [
            Vec3(
                head_mesh.morph_normals[vi * 3],
                head_mesh.morph_normals[vi * 3 + 1],
                head_mesh.morph_normals[vi * 3 + 2],
            ),
            Vec3(
                head_mesh.morph_normals[(v_cnt + vi) * 3],
                head_mesh.morph_normals[(v_cnt + vi) * 3 + 1],
                head_mesh.morph_normals[(v_cnt + vi) * 3 + 2],
            ),
        ]
        var wv = j_weights[vi]
        var cpu_combo = skin_vertex_morph_dq_smear_cpu(
            rot_bones, act_smears, w_both, pd, nd, 0, 1, 0, 0, wv.x, wv.y, 0.0, 0.0, bp, bn
        )
        var gp = Vec3(gpu_v[vi * 12 + 0], gpu_v[vi * 12 + 1], gpu_v[vi * 12 + 2])
        var gn = Vec3(gpu_v[vi * 12 + 3], gpu_v[vi * 12 + 4], gpu_v[vi * 12 + 5])
        _assert((gp - cpu_combo[0]).length() < Float32(1e-4), "GPU Morph+Smear+DQ pos matches CPU < 1e-4")
        _assert((gn - cpu_combo[1]).length() < Float32(1e-3), "GPU Morph+Smear+DQ nrm matches CPU < 1e-3")

    _ = len(gpu_v)
    _ = len(gpu_prev)
    ren.shutdown()
    print(
        "  [PASS] Suite 4: Live Vulkan GPU Morph-Target (`morph_count=2`) Cel Silhouette + 2.5px Outline Expansion verified"
    )


def _v4_add_node(
    mut model: NativeModel,
    name: String,
    parent: Int,
    base_t: Vec3,
    base_r: Quat = Quat(),
) -> Int:
    var idx = len(model.nodes)
    var node = NativeNode()
    node.name = name
    node.translation = base_t
    node.base_translation = base_t
    node.rotation = base_r
    node.base_rotation = base_r
    node.scale = Vec3(1, 1, 1)
    node.base_scale = Vec3(1, 1, 1)
    model.nodes.append(node^)
    if parent >= 0:
        model.nodes[parent].children.append(idx)
    else:
        model.roots.append(idx)
    return idx


def test_v4_rbf_helpers_skinning_and_springbones():
    # 7A. Gaussian RBF Pose-Space Deformer: exact Kronecker delta at sculpted targets & bounded diagonal
    var rbf = RbfPoseDriver(0, False)
    rbf.add_sculpted_pose(0.85, 0.0, 0)
    rbf.add_sculpted_pose(0.0, 0.85, 1)
    rbf.finalize()

    var w_rest = rbf.evaluate(0.0, 0.0)
    _assert_approx(w_rest[0], 0.0, 0.01, "RBF rest pose morph 0 == 0")
    _assert_approx(w_rest[1], 0.0, 0.01, "RBF rest pose morph 1 == 0")

    var w_t1 = rbf.evaluate(0.85, 0.0)
    _assert_approx(w_t1[0], 1.0, 0.01, "RBF target 1 hits w_0 ~ 1.0")
    _assert_approx(w_t1[1], 0.0, 0.01, "RBF target 1 suppresses w_1 ~ 0.0")

    var w_t2 = rbf.evaluate(0.0, 0.85)
    _assert_approx(w_t2[0], 0.0, 0.01, "RBF target 2 suppresses w_0 ~ 0.0")
    _assert_approx(w_t2[1], 1.0, 0.01, "RBF target 2 hits w_1 ~ 1.0")

    var w_diag = rbf.evaluate(0.60, 0.60)
    var diag_sum = w_diag[0] + w_diag[1]
    _assert(diag_sum > 0.65 and diag_sum < 1.25, "RBF diagonal sum stays near 1.0 without 2x linear bulge")

    # 7B. Proximal Counter-Twist + Distal Twist HelperJoints
    var arm_model = NativeModel()
    var clavicle = _v4_add_node(arm_model, "Clavicle", -1, Vec3(0, 1.4, 0))
    var upper_arm = _v4_add_node(arm_model, "UpperArm", clavicle, Vec3(0.2, 0, 0))
    var upper_twist = _v4_add_node(arm_model, "UpperArm_Twist", upper_arm, Vec3(0.05, 0, 0))
    var forearm = _v4_add_node(arm_model, "Forearm", upper_arm, Vec3(0.32, 0, 0))
    var hand = _v4_add_node(arm_model, "Hand", forearm, Vec3(0.28, 0, 0))
    var forearm_twist = _v4_add_node(arm_model, "Forearm_Twist", forearm, Vec3(0.14, 0, 0))

    var manifest_bones = List[ManifestBone]()
    var mb_prox = ManifestBone("UpperArm_Twist", "UpperArm", Vec3(0.25, 1.4, 0))
    mb_prox.helper_kind = HELPER_TWIST
    mb_prox.helper_source = "UpperArm"
    mb_prox.helper_amount = 0.5
    manifest_bones.append(mb_prox^)

    var mb_dist = ManifestBone("Forearm_Twist", "Forearm", Vec3(0.66, 1.4, 0))
    mb_dist.helper_kind = HELPER_TWIST
    mb_dist.helper_source = "Hand"
    mb_dist.helper_amount = 0.5
    manifest_bones.append(mb_dist^)

    var helpers = HelperJoints()
    helpers.bind(arm_model, 1, manifest_bones)
    _assert(len(helpers.problems) == 0, "proximal child & distal sibling twist helpers bind with 0 problems")
    _assert(helpers.count() == 2, "both proximal and distal twist helpers bound")

    # Apply combined swing (45 deg about Z) + axial twist (60 deg about shaft +X) to UpperArm, and 80 deg +X roll to Hand
    var swing_z = angle_axis(PI * Float32(0.25), Vec3(0, 0, 1))
    var roll_x = angle_axis(PI / Float32(3.0), Vec3(1, 0, 0))
    arm_model.nodes[upper_arm].rotation = swing_z.multiply(roll_x).normalized()
    arm_model.nodes[hand].rotation = angle_axis(Float32(1.4), Vec3(1, 0, 0))
    helpers.apply(arm_model)

    # UpperArm_Twist world rotation should have 100% of swing_z and (1 - 0.5) = 50% of roll_x!
    var expected_prox_world = swing_z.multiply(angle_axis(PI / Float32(6.0), Vec3(1, 0, 0))).normalized()
    var actual_prox_world = arm_model.global_rotation(upper_twist)
    var prox_err = det_quat_to_scaled_angle(expected_prox_world.multiply(actual_prox_world.conjugate())).length()
    _assert_approx(prox_err, 0.0, 0.002, "proximal counter-twist cancels 50% axial roll while keeping 100% swing")

    var expected_dist_local = angle_axis(Float32(0.7), Vec3(1, 0, 0))
    var dist_err = det_quat_to_scaled_angle(
        expected_dist_local.multiply(arm_model.nodes[forearm_twist].rotation.conjugate())
    ).length()
    _assert_approx(dist_err, 0.0, 0.002, "distal sibling twist helper takes +50% of hand roll")
    _ = swing_twist_y(arm_model.nodes[upper_arm].rotation, Quat())

    # 7C. Cofactor SKIN_LBS normal transform under non-uniform scale diag(2.0, 0.5, 1.0)
    var palette = List[Float32](length=PALETTE_FLOATS, fill=0.0)
    var m_nonunif = Mat4()
    m_nonunif.set(0, 0, 2.0)
    m_nonunif.set(1, 1, 0.5)
    m_nonunif.set(2, 2, 1.0)
    _ = write_palette_entry(Int(palette.unsafe_ptr()), 0, m_nonunif)

    var draw = SkinDraw()
    draw.palette = Int(palette.unsafe_ptr())
    draw.joint_count = 1
    draw.method = SKIN_LBS
    var joints_v: List[UInt16] = [UInt16(0), UInt16(0), UInt16(0), UInt16(0)]
    var weights_v: List[Float32] = [Float32(1.0), Float32(0.0), Float32(0.0), Float32(0.0)]

    var n_bind = Vec3(1.0, 1.0, 0.0).normalized()
    var skinned_vn = skin_vertex(
        draw,
        Vec3(1.0, 1.0, 0.0),
        n_bind,
        Vec3(),
        Int(joints_v.unsafe_ptr()),
        Int(weights_v.unsafe_ptr()),
    )
    var deformed_tangent = Vec3(2.0, -0.5, 0.0)
    _assert_approx(skinned_vn[1].length(), 1.0, 0.0005, "cofactor skinned normal is unit length")
    _assert_approx(skinned_vn[1].dot(deformed_tangent), 0.0, 0.001, "cofactor normal stays perpendicular under non-uniform scale")
    _ = len(palette) + len(joints_v) + len(weights_v)

    # 7D. Inertia-Clamped SpringBones under a 100m single-frame teleport
    var cape_model = NativeModel()
    var spine_c = _v4_add_node(cape_model, "Spine", -1, Vec3(0, 1.5, 0))
    var cape1 = _v4_add_node(cape_model, "Cape1", spine_c, Vec3(0, 0, -0.15))
    _ = _v4_add_node(cape_model, "Cape2", cape1, Vec3(0, -0.40, 0))

    var sb_bones = List[ManifestBone]()
    var b1 = ManifestBone("Cape1", "Spine", Vec3(0, 1.5, -0.15))
    b1.spring = True
    b1.tail = Vec3(0, 1.1, -0.15)
    b1.stiffness = 12.0
    b1.drag = 0.25
    b1.gravity = 9.81
    sb_bones.append(b1^)

    var b2 = ManifestBone("Cape2", "Cape1", Vec3(0, 1.1, -0.15))
    b2.spring = True
    b2.tail = Vec3(0, 0.7, -0.15)
    b2.stiffness = 10.0
    b2.drag = 0.25
    b2.gravity = 9.81
    sb_bones.append(b2^)

    var vols = List[ManifestVolume]()
    var springs = SpringBones()
    springs.bind(cape_model, 1, sb_bones, vols)
    _assert(springs.count() == 2, "2-link spring chain bound")
    _assert(springs.bones[1].parent_spring == 0, "Cape2 linked to parent spring Cape1 for intra-substep propagation")

    springs.update(cape_model, Placement(Vec3(0, 0, 0)), Vec3(), Float32(1.0 / 60.0))
    springs.update(cape_model, Placement(Vec3(100.0, 0.0, 0.0)), Vec3(), Float32(1.0 / 60.0))

    var head0_w = Placement(Vec3(100.0, 0.0, 0.0)).to_world(cape_model.global_position(cape1))
    var dist0 = (springs.tails[0] - head0_w).length()
    var vel0_apparent = (springs.tails[0] - springs.previous[0]).length() * Float32(120.0)
    _assert(dist0 < springs.bones[0].length * 1.5, "spring bone tail stays within < 1.5x bone length on 100m teleport")
    _assert_approx(dist0, springs.bones[0].length, 0.01, "spring bone preserves exact bone length after teleport")
    _assert(vel0_apparent < 60.0, "root inertia clamp prevents 1200+ m/s Verlet launch")

    var cape_skel = SkeletonInstance()
    cape_skel.reset_to_rest(cape_model)
    springs.update_skeleton(
        cape_skel, cape_model, Placement(Vec3(100.2, 0.0, 0.0)), Vec3(2.0, 0.0, 0.0), Float32(1.0 / 60.0)
    )
    var cape1_rot = det_quat_to_scaled_angle(cape_skel.pose.r[cape1]).length()
    _assert(
        not isnan(cape1_rot) and cape1_rot > Float32(0.35) and cape1_rot < Float32(0.48),
        "springs.update_skeleton rotates Cape1 within stiffness-calibrated band (0.35..0.48 rad)",
    )

    var h_pose = Pose()
    h_pose.read(arm_model)
    h_pose.r[upper_twist] = Quat()
    h_pose.r[forearm_twist] = Quat()
    helpers.apply_to_pose(h_pose, arm_model)
    var pose_prox_err = det_quat_to_scaled_angle(
        arm_model.nodes[upper_twist].rotation.multiply(h_pose.r[upper_twist].conjugate())
    ).length()
    var pose_dist_err = det_quat_to_scaled_angle(
        arm_model.nodes[forearm_twist].rotation.multiply(h_pose.r[forearm_twist].conjugate())
    ).length()
    _assert_approx(pose_prox_err, 0.0, 0.002, "helpers.apply_to_pose matches proximal counter-twist")
    _assert_approx(pose_dist_err, 0.0, 0.002, "helpers.apply_to_pose matches distal sibling twist")

    arm_model.morph_names = [String("morph_0"), String("morph_1")]
    h_pose.r[clavicle] = angle_axis(Float32(0.85), Vec3(1, 0, 0))
    var morph_out = List[Float32]()
    rbf.apply_to_pose(h_pose, arm_model, morph_out)
    _assert(len(morph_out) == 2, "rbf.apply_to_pose populates morph_out for 2 morph targets")
    _assert_approx(morph_out[0], 1.0, 0.02, "rbf.apply_to_pose activates morph_out[0] ~ 1.0 at sculpted swing_x=0.85")
    _assert_approx(morph_out[1], 0.0, 0.02, "rbf.apply_to_pose suppresses morph_out[1] ~ 0.0")
    print("  [PASS] Suite 5: RBF correctives, HelperJoints, SKIN_LBS cofactor normals & SpringBones verified")


def test_skin_weight_sort_and_mat4_scale():
    # 11b. Dominant-Pivot Weight Sort, 4-Weight Normalization (Zero `j3=0` Root Leak) & Uniform Bone Scale
    var ns1 = normalize_and_sort_skin_weights_4(5, 0, 0, 0, Float32(0.35), Float32(0.0), Float32(0.0), Float32(0.0))
    _assert((ns1[0] & UInt32(0xFF)) == UInt32(5), "1-bone dominant joint == 5")
    _assert(abs(ns1[1] - Float32(1.0)) < Float32(1e-6), "1-bone normalized w0 == 1.0")
    _assert(ns1[2] == Float32(0.0), "1-bone w1 == 0.0")
    _assert(ns1[3] == Float32(0.0), "1-bone w2 == 0.0")
    _assert(Float32(1.0) - (ns1[1] + ns1[2] + ns1[3]) == Float32(0.0), "1-bone w3 == 0.0")

    # 2-bone unsorted & unnormalized `(j0=4, w=0.15), (j1=7, w=0.45)` -> `j0=7 (0.75), j1=4 (0.25)` with zero leak
    var ns2 = normalize_and_sort_skin_weights_4(4, 7, 0, 0, Float32(0.15), Float32(0.45), Float32(0.0), Float32(0.0))
    _assert((ns2[0] & UInt32(0xFF)) == UInt32(7), "2-bone sorted dominant joint == 7")
    _assert(((ns2[0] >> UInt32(8)) & UInt32(0xFF)) == UInt32(4), "2-bone sorted second joint == 4")
    _assert(abs(ns2[1] - Float32(0.75)) < Float32(1e-6), "2-bone normalized w0 == 0.75")
    _assert(abs(ns2[2] - Float32(0.25)) < Float32(1e-6), "2-bone normalized w1 == 0.25")
    _assert(ns2[3] == Float32(0.0), "2-bone w2 == 0.0")
    _assert(Float32(1.0) - (ns2[1] + ns2[2] + ns2[3]) == Float32(0.0), "2-bone w3 == 0.0")

    var leak_bones = List[DualQuat](length=8, fill=DualQuat())
    leak_bones[0] = DualQuat.from_quat_trans(Quat(), Vec3(500.0, -500.0, 500.0))
    leak_bones[4] = DualQuat.from_quat_trans(Quat(), Vec3(0.0, 4.0, 0.0))
    leak_bones[5] = DualQuat.from_quat_trans(Quat(), Vec3(0.0, 3.0, 0.0))
    leak_bones[7] = DualQuat.from_quat_trans(Quat(), Vec3(0.0, 8.0, 0.0))
    var sk1_cpu = skin_vertex_dq_cpu(
        leak_bones, 5, 0, 0, 0, Float32(0.35), Float32(0.0), Float32(0.0), Float32(0.0), Vec3(1.0, 0.0, 0.0)
    )
    _assert((sk1_cpu[0] - Vec3(1.0, 3.0, 0.0)).length() < Float32(1e-4), "1-bone DQ skinning has zero root bone 0 leak")
    var sk2_cpu = skin_vertex_dq_cpu(
        leak_bones, 4, 7, 0, 0, Float32(0.15), Float32(0.45), Float32(0.0), Float32(0.0), Vec3(0.0, 0.0, 0.0)
    )
    _assert((sk2_cpu[0] - Vec3(0.0, 7.0, 0.0)).length() < Float32(1e-4), "2-bone DQ skinning has zero root bone 0 leak")

    # Uniform bone scale in `DualQuat.from_mat4`
    var q_rot = Quat(0.0, Float32(0.70710678), 0.0, Float32(0.70710678)).normalized()
    var m_scaled = trs(Vec3(3.0, -2.0, 5.0), q_rot, Vec3(2.5, 2.5, 2.5))
    var dq_scaled = DualQuat.from_mat4(m_scaled)
    var p_test = Vec3(2.0, 1.0, 0.0)
    var p_dq = dq_scaled.transform_point(p_test)
    var p_m = Vec3(
        m_scaled.values[0] * p_test.x + m_scaled.values[4] * p_test.y + m_scaled.values[8] * p_test.z + m_scaled.values[12],
        m_scaled.values[1] * p_test.x + m_scaled.values[5] * p_test.y + m_scaled.values[9] * p_test.z + m_scaled.values[13],
        m_scaled.values[2] * p_test.x + m_scaled.values[6] * p_test.y + m_scaled.values[10] * p_test.z + m_scaled.values[14],
    )
    _assert((p_dq - p_m).length() < Float32(1e-3), "DualQuat.from_mat4 handles uniform scale 2.5x")
    print("  [PASS] Suite 6: normalize_and_sort_skin_weights_4, zero root-bone leak & DualQuat.from_mat4 scale verified")


def main():
    print("=== Flint v5.2 GPU Morph-Target (`BlendShape`) Skinning & `DirectSpringChain` Physics Suite ===")
    test_layout_arena_and_spirv()
    test_cpu_morph_dq_smear_math()
    test_direct_spring_chain_physics()
    test_v4_rbf_helpers_skinning_and_springbones()
    test_skin_weight_sort_and_mat4_scale()
    test_gpu_morph_skinning_cel_and_outline()
    print("=== ALL FLINT v5.2 MORPH & SPRING TESTS PASSED ===")
