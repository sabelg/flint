"""Sota:geometry — GPU two-pass occlusion culling with a conservative Hi-Z pyramid + continuous cluster-LOD cut.

Replaces the legacy `meshlet_cull_shader` dispatches when `SotaGeometryGpu.enabled` (default on):

  Phase 1  every (instance, meshlet): DAG LOD cut (or the legacy discrete-LOD test for non-DAG meshlets),
           normal cone, frustum, then occlusion against LAST frame's Hi-Z re-projected with the previous
           view-projection and previous instance transform. Visible -> `vkCmdDrawIndexedIndirectCount`;
           occluded -> candidate queue.
  Hi-Z     after the phase-1 raster: level 0 = exact copy of the reverse-Z depth, level k+1 = MIN (farthest in
           reverse-Z) of the 2x2 (clamped) texels below, ceil-divided sizes, so a level-L texel covers exactly
           the 2^L x 2^L pixels under it — no non-power-of-two holes, no bilinear averaging.
  Phase 2  candidates only, against the fresh Hi-Z with the CURRENT matrices; survivors are drawn. Phase 2 is
           conservative, so phase-1 mistakes cost a re-test, never a missing cluster.

Occlusion test: the 8 corners of the world AABB of the bounding sphere are projected (exact for any
projection, including jitter and the previous frame's matrix); the screen rect (+1 px guard) picks the Hi-Z
level where it spans <= 2x2 texels; occluded iff the sphere's nearest reverse-Z depth < the farthest Hi-Z depth.

The legacy path it replaces sampled the single-mip D32 depth with a LINEAR sampler at 5 taps (not
conservative: gaps between taps, filtered depths) — see render-sota-geometry.md.

Sources: Karis/Stone/Thomas "Nanite" (SIGGRAPH 2021); Haar & Aaltonen "GPU-Driven Rendering Pipelines"
(SIGGRAPH 2015); Kapoulkine "niagara" (MIT) for the two-pass structure.
"""
from std.collections import List
from std.ffi import external_call
from std.memory import bitcast
from std.math import max, min
from flint.platform.amdgpu import GpuBuffer
from flint.platform.vulkan import (
    VulkanGpu,
    get32,
    put32,
    ACCESS_INDIRECT_READ,
    ACCESS_SHADER_READ,
    ACCESS_SHADER_WRITE,
    PIPELINE_COMPUTE,
    PIPELINE_DRAW_INDIRECT,
)
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.aaa_visbuffer import (
    AAA_INSTANCE_WORDS,
    emit_mat4_mul_dir,
    emit_mat4_mul_point,
    emit_vec3_dot,
    emit_vec3_normalize,
)
from flint.render.sota_geometry_cluster_dag import SOTA_DAG_RECORD_WORDS, ClusterDag
from flint.render.sota_geometry_dag_cache import DagCacheStats, build_cluster_dag_cached
from flint.assets.model import NativeMesh
from flint.core.jobs import JobPool, default_worker_count

# Own descriptor-set layout (does not touch the shared AAA layout):
comptime SOTA_BIND_MESHLETS: Int = 0
comptime SOTA_BIND_INSTANCES: Int = 1
comptime SOTA_BIND_INDIRECT: Int = 2
comptime SOTA_BIND_GLOBALS: Int = 3
comptime SOTA_BIND_CANDIDATES: Int = 4
comptime SOTA_BIND_DAG: Int = 5
comptime SOTA_BIND_HIZ: Int = 6
comptime SOTA_BIND_STATS: Int = 7
comptime SOTA_STORAGE_COUNT: Int = 8
comptime SOTA_BIND_DEPTH: Int = 8          # combined image sampler (reverse-Z D32 depth)
comptime SOTA_SAMPLER_COUNT: Int = 1
comptime SOTA_PUSH_WORDS: Int = 16

comptime SOTA_HIZ_HEADER_WORDS: Int = 64   # [0..15] level word offsets, [16..31] (w | h << 16), [32] levels
comptime SOTA_HIZ_MAX_LEVELS: Int = 16

# Stats (per frame slot): [0] p1 drawn clusters [1] p1 tris [2] p2 drawn clusters [3] p2 tris
# [4] LOD-rejected [5] cone/frustum-rejected [6] p1 occluded (queued) [7] p2 occluded (final)
# [8] DAG clusters drawn (both phases)
comptime SOTA_STATS_WORDS: Int = 16


def _emit_uf16_to_f32(mut s: AaaSpirvModule, h_u16: UInt32) -> UInt32:
    var is_sub = s.ult(h_u16, s.uint(0x0400))
    var sub_f = s.fmul(s.float_from_uint(h_u16), s.float(5.96046448e-8))
    var norm_bits = s.shl(s.iadd(h_u16, s.uint(0x1C000)), s.uint(13))
    return s.select_f(is_sub, sub_f, s.bitcast_float(norm_bits))


def _emit_proj_error(
    mut s: AaaSpirvModule, err: UInt32, cx: UInt32, cy: UInt32, cz: UInt32, r: UInt32,
    eye_x: UInt32, eye_y: UInt32, eye_z: UInt32, near_z: UInt32, k_px: UInt32,
    lod_px: UInt32, rt_base: UInt32, rt_slope: UInt32,
) -> UInt32:
    """Same formula as `dag_projected_error`: `err * k / D`, D = max(|c - eye| - r, near), maxed with the
    RT-consistency cap `err * t / (base + slope * D)` when base > 0; root (err >= 1e30) -> 3e38."""
    var dx = s.fsub(cx, eye_x)
    var dy = s.fsub(cy, eye_y)
    var dz = s.fsub(cz, eye_z)
    var d = s.nmax(s.fsub(s.fsqrt(emit_vec3_dot(s, dx, dy, dz, dx, dy, dz)), r), near_z)
    var pe = s.fdiv(s.fmul(err, k_px), d)
    var cap = s.fdiv(s.fmul(err, lod_px), s.fma(rt_slope, d, rt_base))
    var pe2 = s.select_f(s.fgt(rt_base, s.float(0.0)), s.nmax(pe, cap), pe)
    return s.select_f(s.fge(err, s.float(1.0e30)), s.float(3.0e38), pe2)


def sota_cluster_cull_shader(route: Int = 0) -> List[UInt32]:
    """Push: [0] instance_count [1] max_draws [2] cone (1/0) [3] occlusion (1/0) [4] screen_w [5] screen_h
    [6] lod threshold px (f32) [7] phase (0/1) [8] stats (1/0) [9] RT bias base m (f32, 0 = off) [10] RT bias slope.  Phase 1 grid: X = instance / 8, Y = meshlet / 8;
    phase 2 grid: X = candidate / 64."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=SOTA_PUSH_WORDS)
    var b_meshlets = s.buffer(SOTA_BIND_MESHLETS)
    var b_instances = s.buffer(SOTA_BIND_INSTANCES)
    var b_indirect = s.buffer(SOTA_BIND_INDIRECT)
    var b_globals = s.buffer(SOTA_BIND_GLOBALS)
    var b_cand = s.buffer(SOTA_BIND_CANDIDATES)
    var b_dag = s.buffer(SOTA_BIND_DAG)
    var b_hiz = s.buffer(SOTA_BIND_HIZ)
    var b_stats = s.buffer(SOTA_BIND_STATS)

    var inst_count = s.push_word(0)
    var max_draws = s.push_word(1)
    var enable_cone = s.push_word(2)
    var enable_occ = s.push_word(3)
    var scr_wu = s.push_word(4)
    var scr_hu = s.push_word(5)
    var scr_w = s.float_from_uint(scr_wu)
    var scr_h = s.float_from_uint(scr_hu)
    var lod_px = s.push_float(6)
    var phase = s.push_word(7)
    var stats_on = s.ine(s.push_word(8), s.uint(0))
    var rt_base = s.push_float(9)
    var rt_slope = s.push_float(10)
    var is_p1 = s.ieq(phase, s.uint(0))

    var inst_var = s.uvar(s.uint(0))
    var ml_var = s.uvar(s.uint(0))
    var valid_var = s.uvar(s.uint(0))
    s.begin_if(is_p1)
    var p1_inst = s.invocation_axis(0)
    var p1_local = s.invocation_axis(1)
    s.begin_if(s.ult(p1_inst, inst_count))
    var ib1 = s.imul(p1_inst, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var first_ml = s.load_word(b_instances, s.iadd(ib1, s.uint(32)))
    var ml_count = s.load_word(b_instances, s.iadd(ib1, s.uint(33)))
    s.begin_if(s.ult(p1_local, ml_count))
    s.store(inst_var, p1_inst)
    s.store(ml_var, s.iadd(first_ml, p1_local))
    s.store(valid_var, s.uint(1))
    s.end_if()
    s.end_if()
    s.begin_else()
    var cand_id = s.iadd(s.imul(s.workgroup_axis(0), s.uint(64)), s.iadd(s.imul(s.local_axis(1), s.uint(8)), s.local_axis(0)))
    var cand_total = s.load_word(b_cand, s.uint(0))
    s.begin_if(s.logical_and(s.ult(cand_id, cand_total), s.ult(cand_id, max_draws)))
    var packed = s.load_word(b_cand, s.iadd(s.uint(4), cand_id))
    s.store(inst_var, s.shr(packed, s.uint(17)))
    s.store(ml_var, s.bit_and(packed, s.uint(0x1FFFF)))
    s.store(valid_var, s.uint(1))
    s.end_if()
    s.end_if()
    s.return_if(s.ieq(s.load_u(valid_var), s.uint(0)))

    var inst_idx = s.load_u(inst_var)
    var ml_id = s.load_u(ml_var)
    var inst_base = s.imul(inst_idx, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var inst_w39 = s.load_word(b_instances, s.iadd(inst_base, s.uint(39)))
    # --- sota:E hook --- route 1 = alpha-masked instances only, route 2 = cel-outlined (word 39 bit 19) only
    if route == 1:
        s.return_if(s.ieq(s.bit_and(inst_w39, s.uint(0x00020000)), s.uint(0)))
    elif route == 2:
        s.return_if(s.ieq(s.bit_and(inst_w39, s.uint(0x00080000)), s.uint(0)))
    else:
        s.return_if(s.ine(s.bit_and(inst_w39, s.uint(0x00020000)), s.uint(0)))   # alpha-masked: own pass
    # --- end sota:E hook ---

    var ml_base = s.imul(ml_id, s.uint(12))
    var lcx = s.load_float(b_meshlets, ml_base)
    var lcy = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(1)))
    var lcz = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(2)))
    var lrad = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(3)))
    var lax = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(4)))
    var lay = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(5)))
    var laz = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(6)))
    var cone_sin = s.load_float(b_meshlets, s.iadd(ml_base, s.uint(7)))
    var first_tri = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(8)))
    var tri_cnt = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(9)))
    var lod_pack = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(10)))

    var uscale = s.load_float(b_instances, s.iadd(inst_base, s.uint(35)))
    var wc = emit_mat4_mul_point(s, b_instances, inst_base, lcx, lcy, lcz)
    var wrad = s.fmul(lrad, uscale)
    # --- sota:E hook --- outline route: skinned bounds are stale, inflate like the CPU packer (r * 4 + 4)
    if route == 2:
        var sk = s.ine(s.bit_and(s.load_word(b_instances, s.iadd(inst_base, s.uint(34))), s.uint(0x00010000)), s.uint(0))
        wrad = s.select_f(sk, s.fma(wrad, s.float(4.0), s.float(4.0)), wrad)
    # --- end sota:E hook ---

    var cam_x = s.load_float(b_globals, s.uint(32))
    var cam_y = s.load_float(b_globals, s.uint(33))
    var cam_z = s.load_float(b_globals, s.uint(34))
    var near_z = s.load_float(b_globals, s.uint(35))
    var fwd_x = s.load_float(b_globals, s.uint(36))
    var fwd_y = s.load_float(b_globals, s.uint(37))
    var fwd_z = s.load_float(b_globals, s.uint(38))
    var tan_x = s.load_float(b_globals, s.uint(39))
    var rgt_x = s.load_float(b_globals, s.uint(40))
    var rgt_y = s.load_float(b_globals, s.uint(41))
    var rgt_z = s.load_float(b_globals, s.uint(42))
    var tan_y = s.load_float(b_globals, s.uint(43))
    var up_x = s.load_float(b_globals, s.uint(44))
    var up_y = s.load_float(b_globals, s.uint(45))
    var up_z = s.load_float(b_globals, s.uint(46))

    var rx = s.fsub(wc[0], cam_x)
    var ry = s.fsub(wc[1], cam_y)
    var rz = s.fsub(wc[2], cam_z)
    var dist = s.fsqrt(s.nmax(s.float(1e-8), emit_vec3_dot(s, rx, ry, rz, rx, ry, rz)))
    var vz = emit_vec3_dot(s, rx, ry, rz, fwd_x, fwd_y, fwd_z)

    var dag_base = s.imul(ml_id, s.uint(UInt32(SOTA_DAG_RECORD_WORDS)))
    var dag_flags = s.load_word(b_dag, s.iadd(dag_base, s.uint(10)))
    var is_dag = s.ine(s.bit_and(dag_flags, s.uint(1)), s.uint(0))

    # ---- Phase 1: LOD cut + cone + frustum ----------------------------------------------------
    s.begin_if(is_p1)
    var k_px = s.fdiv(scr_h, s.nmax(s.float(1e-6), s.fmul(s.float(2.0), tan_y)))
    var lod_ok_var = s.uvar(s.uint(1))
    s.begin_if(is_dag)
    var lvl = s.load_word(b_dag, s.iadd(dag_base, s.uint(11)))
    var sc = emit_mat4_mul_point(
        s, b_instances, inst_base,
        s.load_float(b_dag, dag_base), s.load_float(b_dag, s.iadd(dag_base, s.uint(1))), s.load_float(b_dag, s.iadd(dag_base, s.uint(2))),
    )
    var s_r = s.fmul(s.load_float(b_dag, s.iadd(dag_base, s.uint(3))), uscale)
    var s_e = s.fmul(s.load_float(b_dag, s.iadd(dag_base, s.uint(4))), uscale)
    var pc = emit_mat4_mul_point(
        s, b_instances, inst_base,
        s.load_float(b_dag, s.iadd(dag_base, s.uint(5))), s.load_float(b_dag, s.iadd(dag_base, s.uint(6))), s.load_float(b_dag, s.iadd(dag_base, s.uint(7))),
    )
    var p_r = s.fmul(s.load_float(b_dag, s.iadd(dag_base, s.uint(8))), uscale)
    var p_e_raw = s.load_float(b_dag, s.iadd(dag_base, s.uint(9)))
    var p_e = s.select_f(s.fge(p_e_raw, s.float(1.0e30)), p_e_raw, s.fmul(p_e_raw, uscale))
    var proj_self = _emit_proj_error(s, s_e, sc[0], sc[1], sc[2], s_r, cam_x, cam_y, cam_z, near_z, k_px, lod_px, rt_base, rt_slope)
    var proj_parent = _emit_proj_error(s, p_e, pc[0], pc[1], pc[2], p_r, cam_x, cam_y, cam_z, near_z, k_px, lod_px, rt_base, rt_slope)
    var cut_ok = s.logical_and(s.fle(proj_self, lod_px), s.fgt(proj_parent, lod_px))
    var lod0_only = s.ieq(lvl, s.uint(0))
    s.store(lod_ok_var, s.select_u(s.fgt(lod_px, s.float(0.0)), s.select_u(cut_ok, s.uint(1), s.uint(0)), s.select_u(lod0_only, s.uint(1), s.uint(0))))
    s.begin_else()
    # legacy discrete per-instance LOD interval (bit-identical to meshlet_cull_shader)
    var self_err = _emit_uf16_to_f32(s, s.bit_and(lod_pack, s.uint(0xFFFF)))
    var parent_err = _emit_uf16_to_f32(s, s.shr(lod_pack, s.uint(16)))
    var io_dx = s.fsub(s.load_float(b_instances, s.iadd(inst_base, s.uint(12))), cam_x)
    var io_dy = s.fsub(s.load_float(b_instances, s.iadd(inst_base, s.uint(13))), cam_y)
    var io_dz = s.fsub(s.load_float(b_instances, s.iadd(inst_base, s.uint(14))), cam_z)
    var d_inst = s.nmax(near_z, s.fsqrt(s.nmax(s.float(1e-8), emit_vec3_dot(s, io_dx, io_dy, io_dz, io_dx, io_dy, io_dz))))
    var px_per_m = s.fdiv(s.fmul(uscale, scr_h), s.nmax(s.float(1e-5), s.fmul(s.float(2.0), s.fmul(d_inst, tan_y))))
    var e_max = s.select_f(s.fgt(lod_px, s.float(0.0)), s.fdiv(lod_px, s.nmax(s.float(1e-5), px_per_m)), s.float(0.0))
    var too_coarse = s.fgt(self_err, e_max)
    var too_fine = s.logical_and(s.flt(parent_err, s.float(65000.0)), s.fge(e_max, parent_err))
    s.store(lod_ok_var, s.select_u(s.logical_or(too_coarse, too_fine), s.uint(0), s.uint(1)))
    s.end_if()
    s.begin_if(s.ieq(s.load_u(lod_ok_var), s.uint(0)))
    s.begin_if(stats_on)
    _ = s.atomic_add_word(b_stats, s.uint(4), s.uint(1))
    s.end_if()
    s.end_if()
    s.return_if(s.ieq(s.load_u(lod_ok_var), s.uint(0)))

    var vis_var = s.uvar(s.uint(1))
    s.begin_if(s.logical_and(s.ine(enable_cone, s.uint(0)), s.flt(cone_sin, s.float(1.5))))
    var wax_raw = emit_mat4_mul_dir(s, b_instances, inst_base, lax, lay, laz)
    var wax = emit_vec3_normalize(s, wax_raw[0], wax_raw[1], wax_raw[2])
    var cdot = emit_vec3_dot(s, rx, ry, rz, wax[0], wax[1], wax[2])
    s.begin_if(s.fge(cdot, s.fma(cone_sin, dist, wrad)))
    s.store(vis_var, s.uint(0))
    s.end_if()
    s.end_if()
    var vx = s.fabs(emit_vec3_dot(s, rx, ry, rz, rgt_x, rgt_y, rgt_z))
    var vy = s.fabs(emit_vec3_dot(s, rx, ry, rz, up_x, up_y, up_z))
    var sec_x = s.fsqrt(s.fma(tan_x, tan_x, s.float(1.0)))
    var sec_y = s.fsqrt(s.fma(tan_y, tan_y, s.float(1.0)))
    var vz_pos = s.nmax(s.float(0.0), vz)
    var out_f = s.logical_or(
        s.flt(s.fadd(vz, wrad), near_z),
        s.logical_or(
            s.fgt(s.fsub(vx, s.fmul(wrad, sec_x)), s.fmul(vz_pos, tan_x)),
            s.fgt(s.fsub(vy, s.fmul(wrad, sec_y)), s.fmul(vz_pos, tan_y)),
        ),
    )
    s.begin_if(out_f)
    s.store(vis_var, s.uint(0))
    s.end_if()
    s.begin_if(s.logical_and(stats_on, s.ieq(s.load_u(vis_var), s.uint(0))))
    _ = s.atomic_add_word(b_stats, s.uint(5), s.uint(1))
    s.end_if()
    s.return_if(s.ieq(s.load_u(vis_var), s.uint(0)))
    s.end_if()

    # ---- Occlusion against the conservative Hi-Z --------------------------------------------
    var n_levels = s.load_word(b_hiz, s.uint(32))
    s.begin_if(s.logical_and(s.ine(enable_occ, s.uint(0)), s.ult(s.uint(0), n_levels)))
    # phase 1: previous transform + previous view-proj (Hi-Z is last frame's); phase 2: current
    var mat_inst = s.select_u(is_p1, s.iadd(inst_base, s.uint(16)), inst_base)
    var pw = emit_mat4_mul_point(s, b_instances, mat_inst, lcx, lcy, lcz)
    var vp = s.select_u(is_p1, s.uint(16), s.uint(0))
    var cc = emit_mat4_mul_point(s, b_globals, vp, pw[0], pw[1], pw[2])
    var c0x = s.fmul(s.load_float(b_globals, vp), wrad)
    var c0y = s.fmul(s.load_float(b_globals, s.iadd(vp, s.uint(1))), wrad)
    var c0w = s.fmul(s.load_float(b_globals, s.iadd(vp, s.uint(3))), wrad)
    var c0z = s.fmul(s.load_float(b_globals, s.iadd(vp, s.uint(2))), wrad)
    var c1x = s.fmul(s.load_float(b_globals, s.iadd(vp, s.uint(4))), wrad)
    var c1y = s.fmul(s.load_float(b_globals, s.iadd(vp, s.uint(5))), wrad)
    var c1z = s.fmul(s.load_float(b_globals, s.iadd(vp, s.uint(6))), wrad)
    var c1w = s.fmul(s.load_float(b_globals, s.iadd(vp, s.uint(7))), wrad)
    var c2x = s.fmul(s.load_float(b_globals, s.iadd(vp, s.uint(8))), wrad)
    var c2y = s.fmul(s.load_float(b_globals, s.iadd(vp, s.uint(9))), wrad)
    var c2z = s.fmul(s.load_float(b_globals, s.iadd(vp, s.uint(10))), wrad)
    var c2w = s.fmul(s.load_float(b_globals, s.iadd(vp, s.uint(11))), wrad)
    var min_x = s.fvar(s.float(1.0e30))
    var min_y = s.fvar(s.float(1.0e30))
    var max_x = s.fvar(s.float(-1.0e30))
    var max_y = s.fvar(s.float(-1.0e30))
    var max_d = s.fvar(s.float(-1.0e30))
    var behind = s.uvar(s.uint(0))
    for corner in range(8):
        var sx = s.float(1.0) if (corner & 1) != 0 else s.float(-1.0)
        var sy = s.float(1.0) if (corner & 2) != 0 else s.float(-1.0)
        var sz = s.float(1.0) if (corner & 4) != 0 else s.float(-1.0)
        var px = s.fma(sz, c2x, s.fma(sy, c1x, s.fma(sx, c0x, cc[0])))
        var py = s.fma(sz, c2y, s.fma(sy, c1y, s.fma(sx, c0y, cc[1])))
        var pz = s.fma(sz, c2z, s.fma(sy, c1z, s.fma(sx, c0z, cc[2])))
        var pwc = s.fma(sz, c2w, s.fma(sy, c1w, s.fma(sx, c0w, cc[3])))
        s.begin_if(s.fle(pwc, s.fmul(near_z, s.float(0.5))))
        s.store(behind, s.uint(1))
        s.end_if()
        var iw = s.fdiv(s.float(1.0), s.nmax(pwc, s.float(1e-6)))
        var nx = s.fmul(px, iw)
        var ny = s.fmul(py, iw)
        var nd = s.fmul(pz, iw)
        s.store(min_x, s.nmin(s.load_f(min_x), nx))
        s.store(max_x, s.nmax(s.load_f(max_x), nx))
        s.store(min_y, s.nmin(s.load_f(min_y), ny))
        s.store(max_y, s.nmax(s.load_f(max_y), ny))
        s.store(max_d, s.nmax(s.load_f(max_d), nd))
    s.begin_if(s.ieq(s.load_u(behind), s.uint(0)))
    # pixel rect (+1 px guard for TSR jitter / rasterization rules), v = 0.5 - 0.5 * ndc_y
    var fx0 = s.fsub(s.ffloor(s.fmul(s.fma(s.load_f(min_x), s.float(0.5), s.float(0.5)), scr_w)), s.float(1.0))
    var fx1 = s.fadd(s.ffloor(s.fmul(s.fma(s.load_f(max_x), s.float(0.5), s.float(0.5)), scr_w)), s.float(1.0))
    var fy0 = s.fsub(s.ffloor(s.fmul(s.fsub(s.float(0.5), s.fmul(s.load_f(max_y), s.float(0.5))), scr_h)), s.float(1.0))
    var fy1 = s.fadd(s.ffloor(s.fmul(s.fsub(s.float(0.5), s.fmul(s.load_f(min_y), s.float(0.5))), scr_h)), s.float(1.0))
    var wm1 = s.fsub(scr_w, s.float(1.0))
    var hm1 = s.fsub(scr_h, s.float(1.0))
    var x0 = s.uint_from_float(s.nclamp(fx0, s.float(0.0), wm1))
    var x1 = s.uint_from_float(s.nclamp(fx1, s.float(0.0), wm1))
    var y0 = s.uint_from_float(s.nclamp(fy0, s.float(0.0), hm1))
    var y1 = s.uint_from_float(s.nclamp(fy1, s.float(0.0), hm1))
    var ext = s.umax(s.isub(x1, x0), s.isub(y1, y0))
    var lvl_f = s.fneg(s.ffloor(s.fneg(s.log2(s.nmax(s.float_from_uint(ext), s.float(1.0))))))
    var lvl_u = s.uint_from_float(s.nmax(lvl_f, s.float(0.0)))
    s.begin_if(s.ult(lvl_u, n_levels))
    var off = s.load_word(b_hiz, lvl_u)
    var dims = s.load_word(b_hiz, s.iadd(s.uint(16), lvl_u))
    var lw = s.bit_and(dims, s.uint(0xFFFF))
    var tx0 = s.shr(x0, lvl_u)
    var tx1 = s.shr(x1, lvl_u)
    var ty0 = s.shr(y0, lvl_u)
    var ty1 = s.shr(y1, lvl_u)
    var h00 = s.load_float(b_hiz, s.iadd(off, s.iadd(s.imul(ty0, lw), tx0)))
    var h10 = s.load_float(b_hiz, s.iadd(off, s.iadd(s.imul(ty0, lw), tx1)))
    var h01 = s.load_float(b_hiz, s.iadd(off, s.iadd(s.imul(ty1, lw), tx0)))
    var h11 = s.load_float(b_hiz, s.iadd(off, s.iadd(s.imul(ty1, lw), tx1)))
    var hz_far = s.nmin(s.nmin(h00, h10), s.nmin(h01, h11))
    # reverse-Z: nearer = larger. Occluded iff the sphere's NEAREST depth is behind the FARTHEST occluder.
    var occluded = s.flt(s.load_f(max_d), hz_far)
    s.begin_if(occluded)
    s.begin_if(is_p1)
    var cs = s.atomic_add_word(b_cand, s.uint(0), s.uint(1))
    s.begin_if(s.ult(cs, max_draws))
    s.store_word(b_cand, s.iadd(s.uint(4), cs), s.bit_or(s.shl(inst_idx, s.uint(17)), s.bit_and(ml_id, s.uint(0x1FFFF))))
    s.end_if()
    s.end_if()
    s.begin_if(stats_on)
    _ = s.atomic_add_word(b_stats, s.select_u(is_p1, s.uint(6), s.uint(7)), s.uint(1))
    s.end_if()
    s.end_if()
    s.return_if(occluded)
    s.end_if()
    s.end_if()
    s.end_if()

    # ---- Append VkDrawIndexedIndirectCommand ------------------------------------------------
    var slot = s.atomic_add_word(b_indirect, s.uint(0), s.uint(1))
    s.begin_if(s.ult(slot, max_draws))
    var cmd = s.iadd(s.uint(4), s.imul(slot, s.uint(5)))
    s.store_word(b_indirect, cmd, s.imul(tri_cnt, s.uint(3)))
    s.store_word(b_indirect, s.iadd(cmd, s.uint(1)), s.uint(1))
    s.store_word(b_indirect, s.iadd(cmd, s.uint(2)), s.imul(first_tri, s.uint(3)))
    s.store_word(b_indirect, s.iadd(cmd, s.uint(3)), s.uint(0))
    s.store_word(b_indirect, s.iadd(cmd, s.uint(4)), s.bit_or(s.shl(inst_idx, s.uint(17)), s.bit_and(ml_id, s.uint(0x1FFFF))))
    s.end_if()
    s.begin_if(stats_on)
    _ = s.atomic_add_word(b_stats, s.select_u(is_p1, s.uint(0), s.uint(2)), s.uint(1))
    _ = s.atomic_add_word(b_stats, s.select_u(is_p1, s.uint(1), s.uint(3)), tri_cnt)
    s.begin_if(is_dag)
    _ = s.atomic_add_word(b_stats, s.uint(8), s.uint(1))
    s.end_if()
    s.end_if()
    return s.finish()


def sota_hiz_copy_shader() -> List[UInt32]:
    """Level 0 = exact reverse-Z depth (texelFetch). Push: [0] w [1] h [2] dst word offset."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=SOTA_PUSH_WORDS)
    var b_hiz = s.buffer(SOTA_BIND_HIZ)
    var depth = s.sampled_image_2d(SOTA_BIND_DEPTH)
    var w = s.push_word(0)
    var h = s.push_word(1)
    var off = s.push_word(2)
    var x = s.invocation_axis(0)
    var y = s.invocation_axis(1)
    s.return_if(s.logical_or(s.uge(x, w), s.uge(y, h)))
    var d = s.extract_f(s.texel_fetch_f2d(depth, s.uvec2(x, y), s.uint(0)), 0)
    s.store_float(b_hiz, s.iadd(off, s.iadd(s.imul(y, w), x)), d)
    return s.finish()


def sota_hiz_reduce_shader() -> List[UInt32]:
    """dst(x, y) = min of src(2x..2x+1, 2y..2y+1) clamped to the source. Push: [0] src off [1] src w [2] src h
    [3] dst off [4] dst w [5] dst h."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=SOTA_PUSH_WORDS)
    var b_hiz = s.buffer(SOTA_BIND_HIZ)
    var soff = s.push_word(0)
    var sw = s.push_word(1)
    var sh = s.push_word(2)
    var doff = s.push_word(3)
    var dw = s.push_word(4)
    var dh = s.push_word(5)
    var x = s.invocation_axis(0)
    var y = s.invocation_axis(1)
    s.return_if(s.logical_or(s.uge(x, dw), s.uge(y, dh)))
    var sx0 = s.imul(x, s.uint(2))
    var sy0 = s.imul(y, s.uint(2))
    var sx1 = s.umin(s.iadd(sx0, s.uint(1)), s.isub(sw, s.uint(1)))
    var sy1 = s.umin(s.iadd(sy0, s.uint(1)), s.isub(sh, s.uint(1)))
    var a = s.load_float(b_hiz, s.iadd(soff, s.iadd(s.imul(sy0, sw), sx0)))
    var b = s.load_float(b_hiz, s.iadd(soff, s.iadd(s.imul(sy0, sw), sx1)))
    var c = s.load_float(b_hiz, s.iadd(soff, s.iadd(s.imul(sy1, sw), sx0)))
    var d = s.load_float(b_hiz, s.iadd(soff, s.iadd(s.imul(sy1, sw), sx1)))
    s.store_float(b_hiz, s.iadd(doff, s.iadd(s.imul(y, dw), x)), s.nmin(s.nmin(a, b), s.nmin(c, d)))
    return s.finish()


# ---------------------------------------------------------------------------------------------
# CPU reference for the Hi-Z pyramid + occlusion query (tests)
# ---------------------------------------------------------------------------------------------
def hiz_build_cpu(depth: List[Float32], w: Int, h: Int) -> Tuple[List[Float32], List[Int], List[Int], List[Int]]:
    """Returns (data, offsets, widths, heights) with the exact GPU layout (offsets relative to data[0])."""
    var data = depth.copy()
    var offs = List[Int]()
    var ws = List[Int]()
    var hs = List[Int]()
    offs.append(0); ws.append(w); hs.append(h)
    var cw = w
    var ch = h
    while (cw > 1 or ch > 1) and len(offs) < SOTA_HIZ_MAX_LEVELS:
        var nw = (cw + 1) // 2
        var nh = (ch + 1) // 2
        var so = offs[len(offs) - 1]
        var do_ = len(data)
        for y in range(nh):
            for x in range(nw):
                var x1 = min(2 * x + 1, cw - 1)
                var y1 = min(2 * y + 1, ch - 1)
                var m = min(min(data[so + 2 * y * cw + 2 * x], data[so + 2 * y * cw + x1]), min(data[so + y1 * cw + 2 * x], data[so + y1 * cw + x1]))
                data.append(m)
        offs.append(do_); ws.append(nw); hs.append(nh)
        cw = nw
        ch = nh
    return (data^, offs^, ws^, hs^)


# ---------------------------------------------------------------------------------------------
# Renderer-side state
# ---------------------------------------------------------------------------------------------
comptime SOTA_MAX_GLOBAL_MESHLETS: Int = 131070   # visbuffer word0 = inst << 17 | (meshlet + 1), capped at 0x1FFFF


def sota_geometry_grow_size(old: Int, need: Int) -> Int:
    """Next geometry-buffer size: at least 1.5x the old one and 1.25x the need, rounded up to 4 MB."""
    var target = max(old + old // 2, need + need // 4)
    var mb4 = 4 * 1024 * 1024
    return (target + mb4 - 1) // mb4 * mb4


struct SotaGeometryGpu(Movable):
    """Area A GPU state owned by `FlintAaaVulkanRenderer` (field `sota_geom`)."""
    var enabled: Bool               # cvar r.sota.geometry.cull: conservative Hi-Z two-pass + DAG cut (default on)
    var cluster_lod_auto: Bool      # cvar r.sota.geometry.cluster_lod: `add_mesh(build_lods=True)` cooks a DAG
    var collect_stats: Bool
    var rt_safe_lod: Bool           # cvar r.sota.geometry.rt_safe_lod: cap DAG world error by the RT self-hit bias
    var rt_bias_base: Float32       # half the aaa_rt_shadow sun-ray origin bias (0.004 m + 0.0006 * distance) for margin
    var rt_bias_slope: Float32
    var legacy_hiz_occlusion: Bool  # cvar r.sota.geometry.legacy_hiz: let the v5 5-tap culler occlusion-test (off:
                                    # it false-culls visible meshlets, e.g. the Cornell back wall from frame 2 on)
    var dag_build_threads: Int      # cvar r.sota.geometry.dag_threads: JobPool workers for DAG builds (-1 auto, 0 serial)
    var dag_cache_dir: String       # cvar r.sota.geometry.dag_cache: directory of content-keyed DAG .flpk files ("" off)
    var dag_pool: JobPool
    var dag_pool_started: Bool
    var dag_cache_stats: DagCacheStats
    var logged: List[String]        # renderer warnings already printed once
    var frame_failures: Int         # render_frame calls that returned False (each one is logged)
    var last_frame_error: String
    var grow_events: Int            # geometry buffer growths
    var draws_dropped_total: Int    # visible meshlets / candidates the GPU could not record (over max_draws)
    var draws_dropped_last: Int
    var draw_grow_events: Int
    var meshlet_id_overflows: Int   # meshes refused because their meshlet ids exceed the 17-bit encoding
    var active: Bool
    var dsl: Int
    var layout: Int
    var pool: Int
    var sets_p1: List[Int]
    var sets_p2: List[Int]
    var pipe_cull: Int
    var pipe_hiz_copy: Int
    var pipe_hiz_reduce: Int
    var buf_dag: GpuBuffer
    var buf_hiz: GpuBuffer
    var buf_stats: List[GpuBuffer]
    var dag_capacity: Int
    var hiz_levels: Int
    var hiz_off: List[Int]
    var hiz_w: List[Int]
    var hiz_h: List[Int]
    var hiz_valid: Bool
    var last_stats: List[UInt32]
    var dag_meshlets: Int
    var last_slot: Int

    def __init__(out self):
        self.enabled = True
        self.cluster_lod_auto = True
        self.collect_stats = True
        self.rt_safe_lod = True
        self.rt_bias_base = Float32(0.002)
        self.rt_bias_slope = Float32(0.0003)
        self.legacy_hiz_occlusion = False
        self.dag_build_threads = -1
        self.dag_cache_dir = String()
        self.dag_pool = JobPool()
        self.dag_pool_started = False
        self.dag_cache_stats = DagCacheStats()
        self.logged = List[String]()
        self.frame_failures = 0
        self.last_frame_error = String()
        self.grow_events = 0
        self.draws_dropped_total = 0
        self.draws_dropped_last = 0
        self.draw_grow_events = 0
        self.meshlet_id_overflows = 0
        self.active = False
        self.dsl = 0
        self.layout = 0
        self.pool = 0
        self.sets_p1 = List[Int]()
        self.sets_p2 = List[Int]()
        self.pipe_cull = 0
        self.pipe_hiz_copy = 0
        self.pipe_hiz_reduce = 0
        self.buf_dag = GpuBuffer()
        self.buf_hiz = GpuBuffer()
        self.buf_stats = List[GpuBuffer]()
        self.dag_capacity = 0
        self.hiz_levels = 0
        self.hiz_off = List[Int]()
        self.hiz_w = List[Int]()
        self.hiz_h = List[Int]()
        self.hiz_valid = False
        self.last_stats = List[UInt32](length=SOTA_STATS_WORDS, fill=0)
        self.dag_meshlets = 0
        self.last_slot = 0

    def use_cull(self) -> Bool:
        return self.enabled and self.active

    def init(
        mut self,
        mut vk: VulkanGpu,
        frames: Int,
        render_w: Int,
        render_h: Int,
        meshlet_capacity: Int,
        buf_meshlets: GpuBuffer,
        ring_instances: List[GpuBuffer],
        ring_indirect: List[GpuBuffer],
        ring_globals: List[GpuBuffer],
        buf_candidates: GpuBuffer,
        buf_indirect_phase2: GpuBuffer,
        depth_view: Int,
        sampler: Int,
    ) -> Bool:
        var lay = vk.create_aaa_layouts(SOTA_STORAGE_COUNT, SOTA_SAMPLER_COUNT, 0, False, SOTA_PUSH_WORDS * 4)
        self.dsl = lay[0]
        self.layout = lay[1]
        if self.dsl == 0 or self.layout == 0:
            return False
        var ps = vk.allocate_aaa_descriptor_sets(self.dsl, frames * 2, SOTA_STORAGE_COUNT, SOTA_SAMPLER_COUNT, 0, False)
        self.pool = ps[0]
        var sets = ps[1].copy()
        if len(sets) < frames * 2:
            return False
        self.pipe_cull = vk.create_aaa_compute_pipeline(self.layout, sota_cluster_cull_shader())
        self.pipe_hiz_copy = vk.create_aaa_compute_pipeline(self.layout, sota_hiz_copy_shader())
        self.pipe_hiz_reduce = vk.create_aaa_compute_pipeline(self.layout, sota_hiz_reduce_shader())
        if self.pipe_cull == 0 or self.pipe_hiz_copy == 0 or self.pipe_hiz_reduce == 0:
            return False
        self.dag_capacity = meshlet_capacity
        self.buf_dag = vk.create_buffer(max(4096, meshlet_capacity * SOTA_DAG_RECORD_WORDS * 4), local=True)
        if self.buf_dag.size == 0:
            return False
        _ = external_call["memset", Int](self.buf_dag.cpu_address, Int32(0), self.buf_dag.size)
        # Hi-Z layout
        self.hiz_off = List[Int]()
        self.hiz_w = List[Int]()
        self.hiz_h = List[Int]()
        var cw = render_w
        var ch = render_h
        var cur = SOTA_HIZ_HEADER_WORDS
        while True:
            self.hiz_off.append(cur)
            self.hiz_w.append(cw)
            self.hiz_h.append(ch)
            cur += cw * ch
            if (cw <= 1 and ch <= 1) or len(self.hiz_off) >= SOTA_HIZ_MAX_LEVELS:
                break
            cw = (cw + 1) // 2
            ch = (ch + 1) // 2
        self.hiz_levels = len(self.hiz_off)
        self.buf_hiz = vk.create_buffer(cur * 4, local=True)
        if self.buf_hiz.size == 0:
            return False
        var ha = self.buf_hiz.cpu_address
        _ = external_call["memset", Int](ha, Int32(0), SOTA_HIZ_HEADER_WORDS * 4)
        for l in range(self.hiz_levels):
            put32(ha + l * 4, UInt32(self.hiz_off[l]))
            put32(ha + (16 + l) * 4, UInt32(self.hiz_w[l]) | (UInt32(self.hiz_h[l]) << 16))
        put32(ha + 32 * 4, UInt32(self.hiz_levels))
        self.buf_stats = List[GpuBuffer]()
        for _ in range(frames):
            var b = vk.create_buffer(SOTA_STATS_WORDS * 4, local=False, cpu_reads=True)
            if b.size == 0:
                return False
            _ = external_call["memset", Int](b.cpu_address, Int32(0), b.size)
            self.buf_stats.append(b^)
        self.sets_p1 = List[Int]()
        self.sets_p2 = List[Int]()
        for f in range(frames):
            for ph in range(2):
                var ds = sets[f * 2 + ph]
                vk.write_descriptor(ds, SOTA_BIND_MESHLETS, buf_meshlets)
                vk.write_descriptor(ds, SOTA_BIND_INSTANCES, ring_instances[f])
                if ph == 0:
                    vk.write_descriptor(ds, SOTA_BIND_INDIRECT, ring_indirect[f])
                else:
                    vk.write_descriptor(ds, SOTA_BIND_INDIRECT, buf_indirect_phase2)
                vk.write_descriptor(ds, SOTA_BIND_GLOBALS, ring_globals[f])
                vk.write_descriptor(ds, SOTA_BIND_CANDIDATES, buf_candidates)
                vk.write_descriptor(ds, SOTA_BIND_DAG, self.buf_dag)
                vk.write_descriptor(ds, SOTA_BIND_HIZ, self.buf_hiz)
                vk.write_descriptor(ds, SOTA_BIND_STATS, self.buf_stats[f])
                vk.write_aaa_texture(ds, SOTA_BIND_DEPTH, 0, sampler, depth_view)
                if ph == 0:
                    self.sets_p1.append(ds)
                else:
                    self.sets_p2.append(ds)
        self.active = True
        return True

    def rebind_meshlets(mut self, mut vk: VulkanGpu, buf_meshlets: GpuBuffer, meshlet_capacity: Int) -> Bool:
        """The renderer grew its meshlet buffer: rebind it and grow the DAG records to match (contents kept)."""
        if not self.active:
            return True
        var ok = True
        if meshlet_capacity > self.dag_capacity:
            var nb = vk.create_buffer(max(4096, meshlet_capacity * SOTA_DAG_RECORD_WORDS * 4), local=True)
            if nb.size == 0 or nb.cpu_address == 0:
                vk.destroy_buffer(nb)
                ok = False
            else:
                _ = external_call["memset", Int](nb.cpu_address, Int32(0), nb.size)
                _ = external_call["memcpy", Int](nb.cpu_address, self.buf_dag.cpu_address, self.buf_dag.size)
                vk.destroy_buffer(self.buf_dag)
                self.buf_dag = nb^
                self.dag_capacity = meshlet_capacity
        for ds in self.sets_p1:
            vk.write_descriptor(ds, SOTA_BIND_MESHLETS, buf_meshlets)
            vk.write_descriptor(ds, SOTA_BIND_DAG, self.buf_dag)
        for ds in self.sets_p2:
            vk.write_descriptor(ds, SOTA_BIND_MESHLETS, buf_meshlets)
            vk.write_descriptor(ds, SOTA_BIND_DAG, self.buf_dag)
        return ok

    def rebind_draws(mut self, mut vk: VulkanGpu, ring_indirect: List[GpuBuffer], buf_indirect_phase2: GpuBuffer, buf_candidates: GpuBuffer):
        """The renderer grew its draw buffers (max_draws): rebind them."""
        if not self.active:
            return
        for f in range(len(self.sets_p1)):
            vk.write_descriptor(self.sets_p1[f], SOTA_BIND_INDIRECT, ring_indirect[f])
            vk.write_descriptor(self.sets_p1[f], SOTA_BIND_CANDIDATES, buf_candidates)
        for f in range(len(self.sets_p2)):
            vk.write_descriptor(self.sets_p2[f], SOTA_BIND_INDIRECT, buf_indirect_phase2)
            vk.write_descriptor(self.sets_p2[f], SOTA_BIND_CANDIDATES, buf_candidates)

    # -- DAG records ---------------------------------------------------------------------------------------
    def write_dag_records(mut self, first_meshlet: Int, records: List[UInt32]):
        if self.buf_dag.size == 0:
            return
        var n = len(records) // SOTA_DAG_RECORD_WORDS
        if first_meshlet < 0 or first_meshlet + n > self.dag_capacity:
            return
        _ = external_call["memcpy", Int](
            self.buf_dag.cpu_address + first_meshlet * SOTA_DAG_RECORD_WORDS * 4, Int(records.unsafe_ptr()), len(records) * 4
        )
        self.dag_meshlets += n

    def clear_dag_range(mut self, first_meshlet: Int, count: Int):
        """A meshlet range was freed / re-cooked by the legacy path: its records must read as legacy again."""
        if self.buf_dag.size == 0 or count <= 0 or first_meshlet < 0:
            return
        var n = min(count, self.dag_capacity - first_meshlet)
        if n <= 0:
            return
        _ = external_call["memset", Int](self.buf_dag.cpu_address + first_meshlet * SOTA_DAG_RECORD_WORDS * 4, Int32(0), n * SOTA_DAG_RECORD_WORDS * 4)

    # -- per frame --------------------------------------------------------------------------------------------
    def begin_slot(mut self, slot: Int):
        """CPU: harvest the slot's previous stats (its fence has signalled) and zero them for this frame."""
        if slot < 0 or slot >= len(self.buf_stats):
            return
        var a = self.buf_stats[slot].cpu_address
        for i in range(SOTA_STATS_WORDS):
            self.last_stats[i] = get32(a + i * 4)
        _ = external_call["memset", Int](a, Int32(0), SOTA_STATS_WORDS * 4)

    def check_draw_overflow(mut self, max_draws: Int) -> Int:
        """From the stats just harvested: meshlets that did not fit the indirect / candidate buffers
        (phase-1 draws, phase-2 draws and occluded candidates each have `max_draws` slots). Returns the
        `max_draws` the scene needs (0 = no overflow)."""
        if not self.collect_stats or len(self.last_stats) < 9:
            return 0
        var p1 = Int(self.last_stats[0])
        var p2 = Int(self.last_stats[2])
        var cand = Int(self.last_stats[6])
        var dropped = max(0, p1 - max_draws) + max(0, p2 - max_draws) + max(0, cand - max_draws)
        self.draws_dropped_last = dropped
        if dropped == 0:
            return 0
        self.draws_dropped_total += dropped
        return max(p1, max(p2, cand))

    def read_stats(mut self, slot: Int):
        """CPU: after the slot's frame completed (synchronous readback frames)."""
        if slot < 0 or slot >= len(self.buf_stats):
            return
        var a = self.buf_stats[slot].cpu_address
        for i in range(SOTA_STATS_WORDS):
            self.last_stats[i] = get32(a + i * 4)

    def record_cull(
        mut self, mut vk: VulkanGpu, slot: Int, phase: Int, n_inst: Int, max_draws: Int, max_ml_per_inst: Int,
        cone: Bool, occlusion: Bool, render_w: Int, render_h: Int, lod_px: Float32, rt_active: Bool = False,
    ):
        var push = List[UInt32](length=SOTA_PUSH_WORDS, fill=0)
        push[0] = UInt32(n_inst)
        push[1] = UInt32(max_draws)
        push[2] = UInt32(1) if cone else UInt32(0)
        push[3] = UInt32(1) if (occlusion and (phase == 1 or self.hiz_valid)) else UInt32(0)
        push[4] = UInt32(render_w)
        push[5] = UInt32(render_h)
        push[6] = bitcast[DType.uint32](lod_px)
        push[7] = UInt32(phase)
        push[8] = UInt32(1) if self.collect_stats else UInt32(0)
        if self.rt_safe_lod and rt_active:
            push[9] = bitcast[DType.uint32](self.rt_bias_base)
            push[10] = bitcast[DType.uint32](self.rt_bias_slope)
        if phase == 0:
            self.last_slot = slot
            vk.dispatch_aaa(slot, self.pipe_cull, self.layout, self.sets_p1[slot], push, (n_inst + 7) // 8, (max_ml_per_inst + 7) // 8)
        else:
            vk.dispatch_aaa(slot, self.pipe_cull, self.layout, self.sets_p2[slot], push, (max_draws + 63) // 64, 1)

    def record_hiz_build(mut self, mut vk: VulkanGpu, slot: Int):
        """Depth must be in SHADER_READ_ONLY layout and its writes visible to compute."""
        var push = List[UInt32](length=SOTA_PUSH_WORDS, fill=0)
        push[0] = UInt32(self.hiz_w[0])
        push[1] = UInt32(self.hiz_h[0])
        push[2] = UInt32(self.hiz_off[0])
        vk.dispatch_aaa(slot, self.pipe_hiz_copy, self.layout, self.sets_p1[slot], push, (self.hiz_w[0] + 7) // 8, (self.hiz_h[0] + 7) // 8)
        for l in range(1, self.hiz_levels):
            vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ))
            var p = List[UInt32](length=SOTA_PUSH_WORDS, fill=0)
            p[0] = UInt32(self.hiz_off[l - 1])
            p[1] = UInt32(self.hiz_w[l - 1])
            p[2] = UInt32(self.hiz_h[l - 1])
            p[3] = UInt32(self.hiz_off[l])
            p[4] = UInt32(self.hiz_w[l])
            p[5] = UInt32(self.hiz_h[l])
            vk.dispatch_aaa(slot, self.pipe_hiz_reduce, self.layout, self.sets_p1[slot], p, (self.hiz_w[l] + 7) // 8, (self.hiz_h[l] + 7) // 8)
        vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE | PIPELINE_DRAW_INDIRECT, Int(ACCESS_SHADER_READ | ACCESS_INDIRECT_READ))
        self.hiz_valid = True

    def invalidate_hiz(mut self):
        self.hiz_valid = False

    def readback_hiz_level(self, level: Int) -> List[Float32]:
        var out = List[Float32]()
        if level < 0 or level >= self.hiz_levels or self.buf_hiz.size == 0:
            return out^
        var n = self.hiz_w[level] * self.hiz_h[level]
        var a = self.buf_hiz.cpu_address + self.hiz_off[level] * 4
        for i in range(n):
            out.append(bitcast[DType.float32](get32(a + i * 4)))
        return out^

    def build_dag(mut self, mesh: NativeMesh) -> ClusterDag:
        """Cluster DAG for `mesh`: `.flpk` cache hit, else a JobPool-parallel build (bit-identical to serial)."""
        if not self.dag_pool_started and self.dag_build_threads != 0:
            var n = default_worker_count() if self.dag_build_threads < 0 else self.dag_build_threads
            if n > 0:
                _ = self.dag_pool.start(n)
            self.dag_pool_started = True
        return build_cluster_dag_cached(mesh, self.dag_cache_dir, self.dag_pool, self.dag_cache_stats)

    def shutdown(mut self, mut vk: VulkanGpu):
        if self.dag_pool_started and self.dag_pool.ready():
            self.dag_pool.stop()
        self.dag_pool_started = False
        if self.pipe_cull != 0:
            vk.destroy_pipeline(self.pipe_cull)
        if self.pipe_hiz_copy != 0:
            vk.destroy_pipeline(self.pipe_hiz_copy)
        if self.pipe_hiz_reduce != 0:
            vk.destroy_pipeline(self.pipe_hiz_reduce)
        self.pipe_cull = 0
        self.pipe_hiz_copy = 0
        self.pipe_hiz_reduce = 0
        vk.destroy_buffer(self.buf_dag)
        vk.destroy_buffer(self.buf_hiz)
        for b in self.buf_stats:
            vk.destroy_buffer(b)
        self.buf_dag = GpuBuffer()
        self.buf_hiz = GpuBuffer()
        self.buf_stats = List[GpuBuffer]()
        vk.destroy_aaa_descriptors(self.pool, self.dsl, self.layout)
        self.pool = 0
        self.dsl = 0
        self.layout = 0
        self.active = False
