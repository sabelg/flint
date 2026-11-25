"""Flint v5 MagicArticulate / Puppeteer suite (`test_flint_v5_magicarticulate.mojo`).

Verifies, without model files:
  1. Token embeddings: slot indices (special ids keep their own row, coordinates cycle through
     3 + (k - 1) mod tokens-per-bone) and the decoder's length budget per variant.
  2. Detokenizing: bone tokens (6 per bone, `[1:-1]` trimming, specials dropped) and Puppeteer
     joint tokens (parent index + 1; an out-of-range parent fails like the demo's None).
  3. Post-processing: np.unique joint order, bone de-duplication, merging of components closer
     than 0.02 and bridging of farther ones; root choice for hierarchical, spatial and joint order;
     parent-first ordering that cuts parent loops.
  4. Watertight shell: the unsigned distance grid of a cube is exact near the surface, the 2/128
     level set is a closed double shell whose normals point to larger |d|, and the sampled cloud
     is normalised to max |x| = 0.9995 with unit normals.
  5. With `MA_DIR` (graphs from `tools/magicarticulate/export.py`) and `MA_REF` (its `ref/` dumps):
     parity with the PyTorch reference on every dumped point cloud: decoder prefix max |diff|,
     greedy tokens identical, post-processed joints and parents identical. `MA_VARIANT` selects
     "magicarticulate" (default) or "puppeteer".
  6. With `MA_DIR`, `MA_MESH` (a GLB) and `UNIRIG_DIR`: end-to-end rigged model with unit weights.
"""
from std.math import abs, sqrt
from std.testing import assert_true, assert_equal
from std.collections import Optional
from flint.ml.magicarticulate import (
    MA_BOS, MA_EOS, MA_SPECIAL, MaConfig, MaSkeleton, MagicArticulate, slot_index, ids_to_bones,
    joints_and_bones, merge_and_fix, postprocess, tree_order, generate_ids, rig_native_model_ma,
)
from flint.ml.magicarticulate_mesh import (
    unsigned_distance_grid, extract_level_set, watertight_soup, sample_soup, normalize_point_cloud,
    MaPointCloud, round_f16,
)
from flint.ml.unirig import UrRng
from flint.ml.tensor import Tensor, fptr_at
from flint.assets.glb_native import load_native_glb
from flint.platform.os import env_now, read_bytes


def _tok(c: Int) -> Int:
    return c + MA_SPECIAL


def test_slots() raises:
    assert_equal(slot_index(MA_BOS, 0, 6), MA_BOS)
    assert_equal(slot_index(_tok(5), 1, 6), 3)
    assert_equal(slot_index(_tok(5), 6, 6), 8)
    assert_equal(slot_index(_tok(5), 7, 6), 3)
    assert_equal(slot_index(MA_EOS, 9, 6), MA_EOS)
    assert_equal(slot_index(_tok(1), 4, 4), 6)
    assert_equal(slot_index(_tok(1), 0, 4), 6)     # python's (-1) % 4 + 3
    var ma = MaConfig.magicarticulate()
    var pp = MaConfig.puppeteer()
    assert_equal(ma.max_length(), 859)
    assert_equal(pp.max_length(), 663)
    assert_equal(pp.generate_length(), 406)
    print("  slot embeddings / lengths: ok")


def test_detokenize() raises:
    var cfg = MaConfig.magicarticulate()
    var ids = List[Int]()
    ids.append(MA_BOS)
    for t in [64, 64, 0, 64, 64, 64, 64, 64, 64, 64, 64, 127]:
        ids.append(_tok(t))
    ids.append(MA_EOS)
    var bones = List[Int]()
    assert_true(ids_to_bones(ids, cfg, bones))
    assert_equal(len(bones), 12)
    assert_equal(bones[2], 0)
    assert_equal(bones[11], 127)
    # an incomplete bone does not decode
    var bad = ids.copy()
    _ = bad.pop()
    bad.append(_tok(3))
    bad.append(MA_EOS)
    assert_true(not ids_to_bones(bad, cfg, bones))
    # Puppeteer: joint 0 root, joint 1 child of 0, joint 2 child of 1
    var pc = MaConfig.puppeteer()
    var jt = List[Int]()
    jt.append(MA_BOS)
    for t in [64, 64, 0, 0, 64, 64, 64, 1, 64, 64, 127, 2]:
        jt.append(_tok(t))
    jt.append(MA_EOS)
    assert_true(ids_to_bones(jt, pc, bones))
    assert_equal(len(bones), 12)
    assert_equal(bones[2], 0)      # bone 0 parent = joint 0
    assert_equal(bones[5], 64)     # child = joint 1
    assert_equal(bones[11], 127)
    var jbad = jt.copy()
    jbad[len(jbad) - 2] = _tok(9)  # parent 8 of 3 joints
    assert_true(not ids_to_bones(jbad, pc, bones))
    print("  detokenize (bones / joints): ok")


def test_postprocess() raises:
    # bones: A(10,0,0)->B(0,0,0), B->C(0,0,20); D(0,0,21) isolated pair D->E(0,0,40);
    # far pair F(100,0,0)->G(110,0,0)
    var bins: List[Int] = [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 21, 0, 0, 40, 100, 0, 0, 110, 0, 0]
    var jb = List[Int]()
    var bo = List[Int]()
    joints_and_bones(bins, jb, bo)
    assert_equal(len(jb) // 3, 7)
    # lexicographic: (0,0,0) (0,0,20) (0,0,21) (0,0,40) (10,0,0) (100,0,0) (110,0,0)
    assert_equal(jb[3 * 4], 10)
    assert_equal(bo[0], 4)
    assert_equal(bo[1], 0)
    var cfg = MaConfig.magicarticulate()
    var cen = List[Float64](length=3, fill=0.0)
    var sk = postprocess(bins, cfg, cen)
    # (0,0,20)/(0,0,21) are 1/128 apart: merged; F-G is bridged to the nearest joint
    assert_equal(sk.num_joints(), 6)
    assert_equal(sk.root, 3)                 # first bone's parent A = (10, 0, 0)
    assert_equal(sk.parents[sk.root], -1)
    var roots = 0
    for p in sk.parents:
        if p < 0:
            roots += 1
    assert_equal(roots, 1)
    # spatial order: root nearest the centroid, BFS parents
    var sp = MaConfig.magicarticulate(False)
    var c2: List[Float64] = [Float64(110) / 128 - 0.5, -0.5, -0.5]
    var sk2 = postprocess(bins, sp, c2)
    assert_equal(sk2.root, 5)
    assert_equal(sk2.parents[5], -1)
    # joint order: root = parent that is never a child
    var jbins: List[Int] = [0, 0, 0, 5, 5, 5, 5, 5, 5, 9, 9, 9]
    var sk3 = postprocess(jbins, MaConfig.puppeteer(), cen)
    assert_equal(sk3.root, 0)
    assert_equal(sk3.parents[1], 0)
    assert_equal(sk3.parents[2], 1)
    # parent-first order with a loop 2 <-> 3
    var par: List[Int] = [-1, 0, 3, 2]
    var o = tree_order(par, 0)
    assert_equal(len(o), 4)
    assert_equal(o[0], 0)
    assert_equal(o[1], 1)
    print("  post-processing / roots / tree order: ok")


def test_watertight() raises:
    # a closed cube [-0.5, 0.5]^3 (12 triangles)
    var v: List[Float32] = [-0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5,
                            -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5]
    var f: List[Int] = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3]
    var n = 32
    var d = unsigned_distance_grid(v, f, n)
    # grid point (i, j, k) at -1 + i / 16; (24, 16, 16) -> (0.5, 0, 0) on the +x face
    assert_true(abs(d[24 + n * (16 + n * 16)]) < 1e-6)
    assert_true(abs(d[26 + n * (16 + n * 16)] - 0.125) < 1e-5)
    assert_true(abs(d[16 + n * (16 + n * 16)] - 0.5) < 1e-5)   # centre (sweep-filled)
    var soup = extract_level_set(d, n, Float32(2.0 / Float64(n)))
    assert_true(len(soup) > 0 and len(soup) % 9 == 0)
    # orientation: towards larger |d| (outward on the outer shell, inward on the inner one)
    var good = 0
    var tot = 0
    for t in range(len(soup) // 9):
        var cx: Float32 = 0
        var cy: Float32 = 0
        var cz: Float32 = 0
        for q in range(3):
            cx += soup[t * 9 + q * 3] / 3
            cy += soup[t * 9 + q * 3 + 1] / 3
            cz += soup[t * 9 + q * 3 + 2] / 3
        cx -= 16.0      # grid index 16 is the origin
        cy -= 16.0
        cz -= 16.0
        var e1x = soup[t * 9 + 3] - soup[t * 9]
        var e1y = soup[t * 9 + 4] - soup[t * 9 + 1]
        var e1z = soup[t * 9 + 5] - soup[t * 9 + 2]
        var e2x = soup[t * 9 + 6] - soup[t * 9]
        var e2y = soup[t * 9 + 7] - soup[t * 9 + 1]
        var e2z = soup[t * 9 + 8] - soup[t * 9 + 2]
        var nx = e1y * e2z - e1z * e2y
        var ny = e1z * e2x - e1x * e2z
        var nz = e1x * e2y - e1y * e2x
        var r = max(abs(cx), max(abs(cy), abs(cz)))
        var outer = r > 8.0       # the face sits at 8 cells from the centre
        var radial = nx * cx + ny * cy + nz * cz
        if (outer and radial > 0) or (not outer and radial < 0):
            good += 1
        tot += 1
    assert_true(Float64(good) > 0.98 * Float64(tot), "shell orientation")
    var shell = watertight_soup(v, f, 5)
    var rng = UrRng(1)
    var xyz = List[Float32]()
    var nrm = List[Float32]()
    sample_soup(shell, 2048, rng, xyz, nrm)
    assert_equal(len(xyz), 2048 * 3)
    var pc = normalize_point_cloud(xyz^, nrm^)
    var m: Float32 = 0
    for x in pc.xyz:
        m = max(m, abs(x))
    assert_true(abs(m - 0.9995) < 1e-3)
    for i in range(pc.count()):
        var l = sqrt(pc.nrm[i * 3] ** 2 + pc.nrm[i * 3 + 1] ** 2 + pc.nrm[i * 3 + 2] ** 2)
        assert_true(abs(l - 1.0) < 2e-3)
    assert_true(round_f16(0.1) != 0.1 and abs(round_f16(0.1) - 0.1) < 1e-4)
    # the joint frame maps back onto the mesh: bounding-box centre of the cloud -> about the origin
    assert_true(abs(pc.to_mesh(0.0, 0)) < 0.05)
    print("  watertight shell / sampling / normalisation: ok")


def _read_f32(path: String, n: Int) raises -> List[Float32]:
    var b = read_bytes(path)
    if len(b) != n * 4:
        raise Error("unexpected size of " + path)
    var out = List[Float32](length=n, fill=0.0)
    var src = fptr_at(Int(b.unsafe_ptr()))
    for i in range(n):
        out[i] = src.unsafe_offset(i).unsafe_load[alignment=1]()
    _ = len(b)
    return out^


def _read_text(path: String) raises -> String:
    var b = read_bytes(path)
    if len(b) == 0:
        raise Error("cannot read " + path)
    return String(unsafe_from_utf8=b)


def _read_ints(path: String) raises -> List[Int]:
    var out = List[Int]()
    for w in _read_text(path).split():
        out.append(Int(String(w)))
    return out^


def test_parity() raises:
    var dir = env_now("MA_DIR")
    var ref_dir = env_now("MA_REF")
    if dir.byte_length() == 0 or ref_dir.byte_length() == 0:
        print("  reference parity: MA_DIR / MA_REF not set, skipped")
        return
    var variant = env_now("MA_VARIANT")
    if variant.byte_length() == 0:
        variant = "magicarticulate"
    var m = MagicArticulate.load(dir, variant, True)
    var names = env_now("MA_REF_NAMES")
    if names.byte_length() == 0:
        names = "bird giraffe tira tripo_carrot"
    for nm_s in names.split():
        var nm = String(nm_s)
        var base = ref_dir + "/" + nm
        var pcv = _read_f32(base + ".pc.bin", 8192 * 6)
        var pc = MaPointCloud()
        for i in range(8192):
            for c in range(3):
                pc.xyz.append(pcv[i * 6 + c])
                pc.nrm.append(pcv[i * 6 + 3 + c])
        var cond = m.encode(pc)
        var rc = _read_f32(base + ".cond.bin", 257 * 1024)
        var md: Float32 = 0
        var mx: Float32 = 0
        for i in range(257 * 1024):
            md = max(md, abs(cond.f[i] - rc[i]))
            mx = max(mx, abs(rc[i]))
        var ids = generate_ids(m.decoder, cond, m.emb, m.cfg)
        var rids = _read_ints(base + ".tokens.txt")
        var same = len(ids) == len(rids)
        var first = -1
        for i in range(min(len(ids), len(rids))):
            if ids[i] != rids[i]:
                same = False
                first = i
                break
        # the reference skeleton (from its own tokens) through our post-processing
        var bins = List[Int]()
        var cen = List[Float64](length=3, fill=0.0)
        var rig_ok: Bool
        var lines = _read_text(base + ".rig.txt").split("\n")
        if ids_to_bones(rids, m.cfg, bins):      # (hierarchical / joint order: no centroid needed)
            var sk = postprocess(bins, m.cfg, cen)
            var head = String(lines[0]).split()
            var J = Int(String(head[1]))
            rig_ok = J == sk.num_joints() and Int(String(head[3])) == sk.root
            if rig_ok:
                for j in range(J):
                    var w = String(lines[1 + j]).split()
                    for c in range(3):
                        if abs(Float64(String(w[c])) - sk.joints[j * 3 + c]) > 1e-6:
                            rig_ok = False
                    if Int(String(w[3])) != sk.parents[j]:
                        rig_ok = False
        else:
            rig_ok = String(lines[0]) == "failed"
        print("   ", nm, ": prefix max|diff|", md, "(max |x|", mx, ")  tokens", len(ids), "vs", len(rids),
              "identical" if same else "DIFFER at " + String(first), " rig", "identical" if rig_ok else "DIFFERS")
        assert_true(md < 1e-2 * max(mx, 1.0), "decoder prefix differs from the reference")
        assert_true(same, "greedy tokens differ from the reference")
        assert_true(rig_ok, "post-processed skeleton differs from the reference")
    print("  reference parity (" + variant + ", " + m.device() + "): ok")


def test_end_to_end() raises:
    var dir = env_now("MA_DIR")
    var mesh = env_now("MA_MESH")
    var skin = env_now("UNIRIG_DIR")
    if dir.byte_length() == 0 or mesh.byte_length() == 0 or skin.byte_length() == 0:
        print("  end-to-end rig: MA_DIR / MA_MESH / UNIRIG_DIR not set, skipped")
        return
    var variant = env_now("MA_VARIANT")
    if variant.byte_length() == 0:
        variant = "magicarticulate"
    var m = MagicArticulate.load(dir, variant, True, skin)
    var src = load_native_glb(mesh)
    assert_true(len(src.meshes) > 0, "could not load " + mesh)
    var out = rig_native_model_ma(m, src)
    var J = len(out.skins[0].joints)
    assert_true(J >= 3 and J < 300, "implausible joint count " + String(J))
    for k in range(len(out.meshes)):
        ref w = out.meshes[k].weights
        for v in range(len(w) // 4):
            var s = w[v * 4] + w[v * 4 + 1] + w[v * 4 + 2] + w[v * 4 + 3]
            assert_true(abs(s - 1.0) < 1e-4)
    print("  end-to-end rig (" + variant + ") on", m.device(), ":", J, "joints")


def main() raises:
    print("=== Flint v5 MagicArticulate Suite ===")
    test_slots()
    test_detokenize()
    test_postprocess()
    test_watertight()
    test_parity()
    test_end_to_end()
    print("=== ALL FLINT v5 MAGICARTICULATE TESTS PASSED ===")
