"""Flint v5 UniRig auto-rigging suite (`test_flint_v5_unirig.mojo`).

Verifies, without model files:
  1. Skeleton tokens: the grammar allows coordinates after the class token, and `detokenize`
     turns a chain plus a branch into joints / parents (branch parent = nearest earlier joint).
  2. `top_influences`: the four largest weights per vertex, renormalised; empty rows bind to 0.
  3. PTv3 sub-manifold neighbour tables: centre slot, +x neighbour, first point wins in a voxel,
     index N for empty slots; `weld` merges coincident vertices.
  4. Skinned GLB round trip: `write`/`parse_native_glb` keep JOINTS_0 / WEIGHTS_0, the skin's
     joint list, inverse bind matrices and the node's skin.
  5. With `UNIRIG_DIR` (converted graphs, `tools/unirig/export.py`) and `UNIRIG_MESH` (a GLB):
     greedy end-to-end rig on the GPU when present, with a plausible skeleton and unit weights.
"""
from std.math import abs
from std.testing import assert_true, assert_equal
from flint.ml.unirig import (
    UR_BOS, UR_EOS, UR_BRANCH, UR_CLS_ARTICULATION, UR_DISCRETE, grammar_state, allowed_tokens,
    detokenize, UrGenConfig,
)
from flint.ml.unirig_skin import neighbor_table, weld
from flint.ml.unirig_rig import UniRig, top_influences, rig_native_model
from flint.assets.model import NativeModel, NativeMesh, NativeNode, NativeSkin
from flint.assets.glb_writer import encode_native_glb
from flint.assets.glb_native import parse_native_glb, load_native_glb
from flint.core.math import Vec3
from flint.platform.os import env_now


def test_tokens() raises:
    var seq = List[Int]()
    seq.append(UR_BOS)
    seq.append(UR_CLS_ARTICULATION)
    var ok = allowed_tokens(grammar_state(seq))
    assert_true(ok[0] and ok[UR_DISCRETE - 1], "coordinates allowed after the class token")
    assert_true(not ok[UR_BOS], "no second BOS")
    # root (128,128,0) -> (128,128,128) -> (128,128,255); branch from the root to (255,128,0)
    var ids = List[Int]()
    ids.append(UR_BOS)
    ids.append(UR_CLS_ARTICULATION)
    for t in [128, 128, 0, 128, 128, 128, 128, 128, 255]:
        ids.append(t)
    ids.append(UR_BRANCH)
    for t in [128, 128, 0, 255, 128, 0]:
        ids.append(t)
    ids.append(UR_EOS)
    var sk = detokenize(ids)
    assert_equal(sk.num_bones(), 4)
    assert_equal(sk.parents[0], -1)
    assert_equal(sk.parents[1], 0)
    assert_equal(sk.parents[2], 1)
    assert_equal(sk.parents[3], 0)
    assert_equal(sk.cls, UR_CLS_ARTICULATION)
    assert_true(abs(sk.joints[3].x - (255.5 / 256.0 * 2.0 - 1.0)) < 1e-6)
    print("  tokens / detokenize: ok")


def test_top_influences() raises:
    var J = 6
    var w = List[Float32](length=3 * J, fill=0.0)
    # vertex 0: six weights, top four are joints 5, 1, 3, 0
    var v0: List[Float32] = [0.15, 0.3, 0.05, 0.2, 0.0, 0.3]
    for j in range(J):
        w[j] = v0[j]
    w[J + 2] = 1.0          # vertex 1: one joint
    # vertex 2: all zero
    var jt = List[UInt16]()
    var jw = List[Float32]()
    top_influences(w, J, jt, jw)
    assert_equal(len(jt), 12)
    var s: Float32 = 0
    for k in range(4):
        s += jw[k]
        assert_true(Int(jt[k]) != 2 and Int(jt[k]) != 4, "dropped the smallest weights")
    assert_true(abs(s - 1.0) < 1e-6)
    assert_true(abs(jw[0] - 0.3 / 0.95) < 1e-6)
    assert_equal(Int(jt[4]), 2)
    assert_true(abs(jw[4] - 1.0) < 1e-6 and jw[5] == 0)
    assert_equal(Int(jt[8]), 0)
    assert_true(jw[8] == 1.0)
    print("  top_influences: ok")


def test_neighbors_and_weld() raises:
    var grid: List[Int] = [0, 0, 0, 1, 0, 0, 0, 0, 0]
    var t = neighbor_table(grid, 3)
    assert_equal(len(t), 3 * 27)
    assert_equal(t[13], 0)                 # centre
    assert_equal(t[2 * 9 + 1 * 3 + 1], 1)  # +x
    assert_equal(t[2 * 27 + 13], 0)        # duplicate voxel: the first point answers
    assert_equal(t[27 + 0], 3)             # empty slot -> N
    var v: List[Float32] = [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]
    var f: List[Int] = [0, 1, 2, 3, 4, 2]
    var wv = List[Float32]()
    var wf = List[Int]()
    var remap = weld(v, f, wv, wf)
    assert_equal(len(wv) // 3, 4)
    assert_equal(remap[1], remap[3])
    assert_equal(wf[3], wf[1])
    print("  neighbour tables / weld: ok")


def test_skinned_glb_roundtrip() raises:
    var m = NativeModel()
    var mesh = NativeMesh()
    for p in [0, 0, 0, 1, 0, 0, 0, 1, 0]:
        mesh.positions.append(Float32(p))
    for _ in range(3):
        mesh.normals.append(0)
        mesh.normals.append(0)
        mesh.normals.append(1)
        mesh.texcoords.append(0)
        mesh.texcoords.append(0)
        for _ in range(4):
            mesh.colors.append(255)
    for i in [0, 1, 2]:
        mesh.indices.append(i)
    for v in range(3):
        mesh.joints.append(UInt16(v % 2))
        mesh.joints.append(UInt16(1 - v % 2))
        mesh.joints.append(0)
        mesh.joints.append(0)
        mesh.weights.append(0.75)
        mesh.weights.append(0.25)
        mesh.weights.append(0)
        mesh.weights.append(0)
    mesh.skin = 0
    m.meshes.append(mesh^)
    var mn = NativeNode()
    mn.name = "body"
    mn.mesh = 0
    m.nodes.append(mn^)
    var j0 = NativeNode()
    j0.name = "root"
    j0.children.append(2)
    m.nodes.append(j0^)
    var j1 = NativeNode()
    j1.name = "tip"
    j1.translation = Vec3(0, 1, 0)
    m.nodes.append(j1^)
    m.roots.append(0)
    m.roots.append(1)
    var sk = NativeSkin()
    sk.joints.append(1)
    sk.joints.append(2)
    for _ in range(2):
        for c in range(16):
            sk.inverse_bind.append(Float32(1.0) if c % 5 == 0 else Float32(0.0))
    sk.inverse_bind[16 + 13] = -1.0
    m.skins.append(sk^)
    var bytes = encode_native_glb(m)
    var back = parse_native_glb(bytes)
    assert_equal(len(back.skins), 1)
    assert_equal(len(back.skins[0].joints), 2)
    assert_equal(back.skins[0].joints[1], 2)
    assert_true(back.skins[0].inverse_bind[16 + 13] == -1.0)
    assert_equal(len(back.meshes), 1)
    assert_equal(len(back.meshes[0].joints), 12)
    assert_equal(Int(back.meshes[0].joints[4]), 1)
    assert_true(abs(back.meshes[0].weights[1] - 0.25) < 1e-7)
    print("  skinned GLB round trip: ok")


def test_end_to_end() raises:
    var dir = env_now("UNIRIG_DIR")
    var mesh = env_now("UNIRIG_MESH")
    if dir.byte_length() == 0 or mesh.byte_length() == 0:
        print("  end-to-end rig: UNIRIG_DIR / UNIRIG_MESH not set, skipped")
        return
    var rig = UniRig.load(dir, True)
    var src = load_native_glb(mesh)
    assert_true(len(src.meshes) > 0, "could not load " + mesh)
    var out = rig_native_model(rig, src, UrGenConfig.greedy())
    var J = len(out.skins[0].joints)
    assert_true(J >= 5 and J < 200, "implausible bone count " + String(J))
    for k in range(len(out.meshes)):
        ref w = out.meshes[k].weights
        for v in range(len(w) // 4):
            var s = w[v * 4] + w[v * 4 + 1] + w[v * 4 + 2] + w[v * 4 + 3]
            assert_true(abs(s - 1.0) < 1e-4)
    print("  end-to-end rig on", rig.device(), ":", J, "joints")


def main() raises:
    print("=== Flint v5 UniRig Suite ===")
    test_tokens()
    test_top_influences()
    test_neighbors_and_weld()
    test_skinned_glb_roundtrip()
    test_end_to_end()
    print("=== ALL FLINT v5 UNIRIG TESTS PASSED ===")
