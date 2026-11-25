"""Flint v5 geometry kit verification (pure Mojo, procedural inputs only, no model files).

  (a) QEM LOD chain (`flint.assets.mesh_lod`): every level watertight, Hausdorff within budget,
      skin weights / UVs carried through decimation; meshlets valid with <= 64 verts / 124 tris.
  (b) UV atlas (`flint.assets.uv_unwrap`): 0 overlapping UV triangle pairs, 0 texel overlaps between
      charts, all UVs in [0,1], area-weighted L2 stretch and worst chart stretch bounded.
  (c) Isotropic remesh (`flint.procedural.remesh`): watertight, genus kept, edge-length CV and
      valence-6 share improved, Hausdorff bound; quad pairing.
  (d) Isosurfaces (`flint.procedural.isosurface`): MC / DC watertight with the right Euler number and
      volume, DC keeps box corners better than MC, sparse bricks == dense grid.
  (e) Retargeting (`flint.anim.retarget`): humanoid roles from SOMA-style names and from an unnamed
      `bone_N` rig with other proportions; foot-contact slide with foot IK << without; bone directions
      follow the source.
"""
from std.math import abs, cos, sin, sqrt, max, min
from std.testing import assert_true
from flint.core.math import Vec3, Quat
from flint.assets.model import NativeMesh
from flint.assets.mesh_kit import mesh_topology, hausdorff_distance, vpos
from flint.assets.meshlet import simplify_mesh_qem
from flint.assets.mesh_lod import LodConfig, build_lod_chain, build_meshlets, validate_meshlets
from flint.assets.uv_unwrap import UnwrapConfig, unwrap_mesh, uv_report
from flint.procedural.remesh import RemeshConfig, isotropic_remesh, pair_quads, triangle_quality
from flint.procedural.isosurface import (grid_from_sphere, grid_from_box, grid_from_torus, marching_cubes,
    dual_contouring, sparse_from_dense, marching_cubes_sparse, dual_contouring_sparse, 
    marching_cubes_values)
from flint.anim.retarget import (RetargetSkeleton, RetargetClip, RetargetConfig, retarget, detect_humanoid_roles,
    foot_contact_error, ROLE_COUNT, R_L_LOWER_LEG, R_R_LOWER_LEG)


def _volume(m: NativeMesh) -> Float64:
    var s: Float64 = 0
    for t in range(m.triangle_count()):
        var a = vpos(m, m.indices[t * 3])
        var b = vpos(m, m.indices[t * 3 + 1])
        var c = vpos(m, m.indices[t * 3 + 2])
        s += Float64(a.dot(b.cross(c))) / 6.0
    return s


def _torus() -> NativeMesh:
    """Watertight genus-1 test surface with a 2-bone skin and planar UVs."""
    var m = marching_cubes(grid_from_torus(56, 0.6, 0.25, 1.0))
    var n = m.vertex_count()
    for v in range(n):
        var p = vpos(m, v)
        m.texcoords.append(p.x * 0.5 + 0.5)
        m.texcoords.append(p.z * 0.5 + 0.5)
        var w = min(Float32(1), max(Float32(0), p.x * 0.8 + 0.5))
        m.joints.append(0)
        m.joints.append(1)
        m.joints.append(0)
        m.joints.append(0)
        m.weights.append(1 - w)
        m.weights.append(w)
        m.weights.append(0)
        m.weights.append(0)
    return m^


def test_lod_and_meshlets() raises:
    var m = _torus()
    assert_true(mesh_topology(m).watertight(), "torus input must be watertight")
    var cfg = LodConfig()
    cfg.ratios = [Float32(1), 0.5, 0.25, 0.125]
    var chain = build_lod_chain(m, cfg)
    assert_true(len(chain) == 4, "LOD chain must keep all four levels within the error budget")
    var prev_tris = m.triangle_count() + 1
    for i in range(len(chain)):
        var lm = chain[i].mesh.copy()
        var t = mesh_topology(lm)
        assert_true(t.watertight() and t.euler == 0, "LOD" + String(i) + " must stay a watertight torus")
        assert_true(lm.triangle_count() < prev_tris, "LOD triangle counts must decrease")
        prev_tris = lm.triangle_count()
        var rel = chain[i].hausdorff.relative()
        var budget = Float32(0.0005) if i == 1 else (Float32(0.0015) if i == 2 else Float32(0.004))
        if i > 0:
            assert_true(rel < budget, "LOD" + String(i) + " Hausdorff " + String(rel) + " over " + String(budget))
        # attributes: UV and 4 skin weights per vertex, weights normalised
        assert_true(len(lm.texcoords) == lm.vertex_count() * 2 and len(lm.weights) == lm.vertex_count() * 4, "LOD lost attributes")
        for v in range(lm.vertex_count()):
            var s = lm.weights[v * 4] + lm.weights[v * 4 + 1] + lm.weights[v * 4 + 2] + lm.weights[v * 4 + 3]
            assert_true(abs(s - 1) < 1e-3, "LOD skin weights must stay normalised")
        var ms = build_meshlets(lm)
        assert_true(validate_meshlets(lm, ms), "meshlets must cover every triangle once within limits")
        assert_true(ms.vertex_refs_per_triangle() < 1.0, "meshlet vertex reuse too low")
    print("  lod: ", m.triangle_count(), "->", chain[3].mesh.triangle_count(), "tris, H(1/8) =", chain[3].hausdorff.relative())


def test_uv_unwrap() raises:
    var m = _torus()
    var cfg = UnwrapConfig()
    cfg.resolution = 512
    cfg.padding = 2
    var u = unwrap_mesh(m, cfg)
    var rep = uv_report(u.mesh, u.chart_of_triangle, u.resolution)
    assert_true(rep.charts >= 2, "a torus needs at least two charts to become discs")
    assert_true(rep.overlapping_pairs == 0, "UV atlas must have 0 overlapping triangle pairs")
    assert_true(rep.texel_overlaps == 0, "UV atlas must have 0 texels shared between charts")
    assert_true(rep.out_of_range == 0, "all UVs must lie in [0,1]")
    assert_true(rep.stretch.flipped == 0, "no flipped UV triangles")
    assert_true(rep.stretch.l2 < 1.05, "area-weighted L2 stretch " + String(rep.stretch.l2))
    assert_true(rep.max_chart_l2 < 1.25, "worst chart L2 stretch " + String(rep.max_chart_l2))
    assert_true(mesh_topology(u.mesh).triangles == m.triangle_count(), "unwrap must keep every triangle")
    print("  uv:  ", rep)


def test_remesh() raises:
    var m = _torus()
    var q0 = triangle_quality(m)
    var cfg = RemeshConfig()
    cfg.target_vertices = m.vertex_count() // 2
    var r = isotropic_remesh(m, cfg)
    var t = mesh_topology(r)
    var q1 = triangle_quality(r)
    assert_true(t.watertight() and t.euler == 0, "remesh must keep a watertight torus")
    assert_true(q1.edge_cv < 0.2 and q1.edge_cv < q0.edge_cv, "remesh must make edges uniform")
    assert_true(q1.valence6_ratio > 0.7, "remesh valence-6 share " + String(q1.valence6_ratio))
    assert_true(q1.mean_min_angle_deg > 45, "remesh mean min angle " + String(q1.mean_min_angle_deg))
    var h = hausdorff_distance(m, r, 8000)
    assert_true(h.relative() < 0.01, "remesh Hausdorff " + String(h.relative()))
    var nv = r.vertex_count()
    assert_true(nv > cfg.target_vertices * 6 // 10 and nv < cfg.target_vertices * 14 // 10, "remesh vertex budget " + String(nv))
    var quads = pair_quads(r)
    assert_true(quads.quad_ratio() > 0.5, "quad pairing ratio " + String(quads.quad_ratio()))
    print("  remesh:", q0, "->", q1, "H =", h.relative(), "quads =", quads.quad_ratio())


def test_isosurface() raises:
    var pi: Float64 = 3.14159265358979
    var gs = grid_from_sphere(48, 0.7, 1.0)
    var exact_s = 4.0 / 3.0 * pi * 0.343
    var mc = marching_cubes(gs)
    var dc = dual_contouring(gs)
    for i in range(2):
        var mm = mc.copy() if i == 0 else dc.copy()
        var t = mesh_topology(mm)
        assert_true(t.watertight() and t.euler == 2 and t.components == 1, "sphere isosurface must be one closed sphere")
        assert_true(abs(_volume(mm) - exact_s) / exact_s < 0.01, "sphere volume (outward winding) " + String(_volume(mm)))
    var gt = grid_from_torus(48, 0.6, 0.22, 1.0)
    var tt = mesh_topology(dual_contouring(gt))
    assert_true(tt.watertight() and tt.euler == 0, "DC torus must have genus 1")
    # sharp features: DC's QEF vertices sit on box corners/edges, MC cuts them
    var h = Vec3(0.5, 0.35, 0.6)
    var gb = grid_from_box(41, h, 1.0)
    var exact_b = 8.0 * 0.5 * 0.35 * 0.6
    var eb_mc = abs(_volume(marching_cubes(gb)) - exact_b)
    var eb_dc = abs(_volume(dual_contouring(gb)) - exact_b)
    assert_true(eb_dc < eb_mc * 0.5, "DC must preserve box corners better than MC")
    # sparse bricks with interior tiles give the same surface as the dense grid
    var sp = sparse_from_dense(gs, 3.0 * gs.spacing)
    var smc = marching_cubes_sparse(sp)
    var sdc = dual_contouring_sparse(sp)
    assert_true(smc.triangle_count() == mc.triangle_count() and abs(_volume(smc) - _volume(mc)) < 1e-5, "sparse MC == dense MC")
    assert_true(sdc.triangle_count() == dc.triangle_count() and abs(_volume(sdc) - _volume(dc)) < 1e-5, "sparse DC == dense DC")
    # TripoSG-style occupancy drop-in ([i][j][k] order, inside where value > level)
    var n = 40
    var vals = List[Float32](length=n * n * n, fill=0)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                var p = Vec3(Float32(i), Float32(j), Float32(k)) * (2.0 / Float32(n - 1)) - Vec3(1, 1, 1)
                vals[(i * n + j) * n + k] = 0.6 - p.length()
    var occ = marching_cubes_values(vals, n, -1.0, 2.0 / Float32(n - 1))
    var to = mesh_topology(occ)
    assert_true(to.watertight() and to.euler == 2 and _volume(occ) > 0, "occupancy MC must be a closed outward sphere")
    print("  iso: MC sphere vol err", abs(_volume(mc) - exact_s) / exact_s, " box err MC", eb_mc, "DC", eb_dc, " bricks", sp.brick_count())


def _soma_like() -> RetargetSkeleton:
    var s = RetargetSkeleton()
    var hips = s.add("Hips", -1, Vec3(0, 0.98, 0))
    var sp = s.add("Spine1", hips, Vec3(0, 0.12, 0))
    var ch = s.add("Chest", sp, Vec3(0, 0.15, 0))
    var nk = s.add("Neck1", ch, Vec3(0, 0.2, 0))
    _ = s.add("Head", nk, Vec3(0, 0.1, 0))
    for side in range(2):
        var nm = String("Left") if side == 0 else String("Right")
        var sx: Float32 = 1 if side == 0 else -1
        var cl = s.add(nm + "Shoulder", ch, Vec3(0.05 * sx, 0.15, 0))
        var ua = s.add(nm + "Arm", cl, Vec3(0.12 * sx, 0, 0))
        var la = s.add(nm + "ForeArm", ua, Vec3(0.27 * sx, 0, 0))
        _ = s.add(nm + "Hand", la, Vec3(0.25 * sx, 0, 0))
    for side in range(2):
        var nm = String("Left") if side == 0 else String("Right")
        var sx: Float32 = 1 if side == 0 else -1
        var th = s.add(nm + "UpLeg", hips, Vec3(0.1 * sx, 0, 0))
        var sh = s.add(nm + "Leg", th, Vec3(0, -0.45, 0))
        var ft = s.add(nm + "Foot", sh, Vec3(0, -0.45, 0))
        _ = s.add(nm + "ToeBase", ft, Vec3(0, -0.08, 0.14))
    return s^


def _bone_rig() -> RetargetSkeleton:
    """Unnamed `bone_N` rig (UniRig style): shorter thigh, longer shin, extra spine joint, A-pose arms."""
    var s = RetargetSkeleton()
    var hips = s.add("bone_0", -1, Vec3(0, 1.12, 0))
    var b1 = s.add("bone_1", hips, Vec3(0, 0.1, 0))
    var b2 = s.add("bone_2", b1, Vec3(0, 0.12, 0))
    var b3 = s.add("bone_3", b2, Vec3(0, 0.12, 0))
    var nk = s.add("bone_4", b3, Vec3(0, 0.15, 0))
    var hd = s.add("bone_5", nk, Vec3(0, 0.1, 0.01))
    _ = s.add("bone_6", hd, Vec3(0, 0.18, 0))
    var id = 7
    for side in range(2):
        var sx: Float32 = 1 if side == 0 else -1
        var cl = s.add("bone_" + String(id), b3, Vec3(0.04 * sx, 0.1, 0))
        var ua = s.add("bone_" + String(id + 1), cl, Vec3(0.14 * sx, 0, 0))
        var la = s.add("bone_" + String(id + 2), ua, Vec3(0.2 * sx, -0.16, 0))
        _ = s.add("bone_" + String(id + 3), la, Vec3(0.19 * sx, -0.15, 0))
        id += 4
    for side in range(2):
        var sx: Float32 = 1 if side == 0 else -1
        var th = s.add("bone_" + String(id), hips, Vec3(0.11 * sx, -0.04, 0))
        var sh = s.add("bone_" + String(id + 1), th, Vec3(0, -0.42, 0.01))
        var ft = s.add("bone_" + String(id + 2), sh, Vec3(0, -0.56, -0.02))
        _ = s.add("bone_" + String(id + 3), ft, Vec3(0, -0.08, 0.16))
        id += 4
    return s^


def _squat_clip() -> RetargetClip:
    """Feet planted, hips dropping: thigh -a, knee +2a, ankle -a keeps each source ankle exactly fixed."""
    var skel = _soma_like()
    var nj = skel.count()
    var clip = RetargetClip(skel^, 30)
    var x = Vec3(1, 0, 0)
    for f in range(60):
        var a = Float32(0.6) * (1 - cos(Float32(f) * 2 * 3.14159265 / 60)) * 0.5 + 0.05
        var rs = List[Quat](length=nj, fill=Quat())
        for side in range(2):
            var base = 13 + side * 4
            rs[base] = Quat.from_axis_angle(x, -a)
            rs[base + 1] = Quat.from_axis_angle(x, 2 * a)
            rs[base + 2] = Quat.from_axis_angle(x, -a)
        clip.rotations.append(rs^)
        clip.root.append(Vec3(0, 0.08 + 0.9 * cos(a), 0))
    return clip^


def test_retarget() raises:
    var src = _squat_clip()
    var sm = detect_humanoid_roles(src.skeleton)
    assert_true(sm.from_names >= 15, "SOMA-style names must resolve most roles by name")
    var tgt = _bone_rig()
    var tm = detect_humanoid_roles(tgt)
    var found = 0
    for r in range(ROLE_COUNT):
        if tm.joint[r] >= 0:
            found += 1
    assert_true(found == ROLE_COUNT, "bone_N rig must resolve all roles from topology, got " + String(found))
    var cfg = RetargetConfig()
    cfg.foot_ik = False
    var raw = retarget(src, tgt, cfg)
    var e0 = foot_contact_error(src, raw)
    cfg.foot_ik = True
    var res = retarget(src, tgt, cfg)
    var e1 = foot_contact_error(src, res)
    assert_true(e1.contact_frames > 100, "planted feet must be detected as contacts")
    assert_true(e0.max_slide > 0.01, "proportion mismatch must make raw feet slide")
    assert_true(e1.max_slide < 0.002 and e1.max_slide < e0.max_slide * 0.1, "foot IK slide " + String(e1.max_slide))
    assert_true(e1.max_height_error < 0.005, "foot IK height error " + String(e1.max_height_error))
    # target shins tilt the same way as the source's (knees forward) at the deepest frame
    var sp = List[Vec3]()
    var sr = List[Quat]()
    src.world(30, sp, sr)
    var tp = List[Vec3]()
    var tr = List[Quat]()
    res.clip.world(30, tp, tr)
    for side in range(2):
        var shin = R_L_LOWER_LEG if side == 0 else R_R_LOWER_LEG
        var ds = (sp[sm.joint[shin]] - sp[sm.joint[shin - 1]]).normalized()
        var dt = (tp[tm.joint[shin]] - tp[tm.joint[shin - 1]]).normalized()
        assert_true(ds.dot(dt) > 0.9, "retargeted thigh direction must follow the source")
    print("  retarget: slide no-IK", e0.max_slide, "-> IK", e1.max_slide, " contacts", e1.contact_frames)


def main() raises:
    print("[geometry kit] LOD + meshlets")
    test_lod_and_meshlets()
    print("[geometry kit] UV atlas")
    test_uv_unwrap()
    print("[geometry kit] isotropic remesh")
    test_remesh()
    print("[geometry kit] marching cubes / dual contouring")
    test_isosurface()
    print("[geometry kit] skeleton retarget")
    test_retarget()
    print("geometry kit: all checks passed")
