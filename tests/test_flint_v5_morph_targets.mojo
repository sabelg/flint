"""Morph targets (glTF blend shapes) end to end (`tests/test_flint_v5_morph_targets.mojo`).

A hand-built GLB quad (v0 (0,0,0), v1 (1,0,0), v2 (1,0,1), v3 (0,0,1), normals +Y) with 8 targets:
target 0 lifts corner v2 by (0,1,0) and tilts its normal, targets 1..7 lift v1 by 0.1 * t. It
has `mesh.weights` (0.25 on target 0), a LINEAR weight clip (target 0: 0 at t=0 -> 1 at t=2) and
a STEP clip. Checked:
  (a) the loader: targets, synthesized names, default and node-override weights, weight tracks
      kept apart from the pose channels, and their sampling;
  (b) the CPU side: sparse rows, the reference evaluator, static uploads baked at their weights;
  (c) the GPU skinning pass of `FlintAaaVulkanRenderer` at weight 0, 0.5 and 1, static and
      animated, all 8 targets at once, previous-frame weights, morph-only and skinned meshes
      (morphs before bones), and a skinned mesh without targets unchanged before and after the
      morph region grows.
"""
from std.math import abs, sqrt
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from flint.core.math import Vec3
from flint.core.transform import Mat4, Quat
from flint.assets.glb_native import parse_native_glb
from flint.assets.glb_writer import append_f32_le, append_u32_le, pad4
from flint.assets.model import NativeMesh, NativeModel
from flint.render.aaa_skinning_rt import (
    AAA_SKIN_PREV_POS_BASE,
    DualQuat,
    build_morph_csr,
    morph_vertex_cpu,
    aaa_dq_skinning_shader,
)
from flint.render.aaa_visbuffer import BIND_AAA_VERTICES, BIND_AAA_SKINNING
from flint.render.aaa_renderer import FlintAaaVulkanRenderer, AaaSkinnedMeshEntry, morphed_mesh
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.spirv_validate import validate_spirv_module

comptime TARGETS: Int = 8
comptime TOL: Float32 = 1e-5


struct GlbBuild(Movable):
    var bin: List[UInt8]
    var views: List[String]
    var accessors: List[String]

    def __init__(out self):
        self.bin = List[UInt8]()
        self.views = List[String]()
        self.accessors = List[String]()

    def _view(mut self, offset: Int, length: Int) -> Int:
        self.views.append('{"buffer":0,"byteOffset":' + String(offset) + ',"byteLength":' + String(length) + "}")
        return len(self.views) - 1

    def floats(mut self, values: List[Float32], kind: String, components: Int) -> Int:
        var offset = len(self.bin)
        for v in values:
            append_f32_le(self.bin, v)
        var view = self._view(offset, len(values) * 4)
        self.accessors.append('{"bufferView":' + String(view) + ',"componentType":5126,"count":' + String(len(values) // components) + ',"type":"' + kind + '"}')
        return len(self.accessors) - 1

    def u16_indices(mut self, values: List[Int]) -> Int:
        var offset = len(self.bin)
        for v in values:
            self.bin.append(UInt8(v & 255))
            self.bin.append(UInt8((v >> 8) & 255))
        var view = self._view(offset, len(values) * 2)
        pad4(self.bin)
        self.accessors.append('{"bufferView":' + String(view) + ',"componentType":5123,"count":' + String(len(values)) + ',"type":"SCALAR"}')
        return len(self.accessors) - 1

    def u8_vec4(mut self, values: List[Int]) -> Int:
        var offset = len(self.bin)
        for v in values:
            self.bin.append(UInt8(v))
        var view = self._view(offset, len(values))
        pad4(self.bin)
        self.accessors.append('{"bufferView":' + String(view) + ',"componentType":5121,"count":' + String(len(values) // 4) + ',"type":"VEC4"}')
        return len(self.accessors) - 1


def _joined(items: List[String]) -> String:
    var s = String()
    for i in range(len(items)):
        if i > 0:
            s += ","
        s += items[i]
    return s


def _vec3s(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> List[Float32]:
    return [a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, d.x, d.y, d.z]


def build_quad_glb(skinned: Bool, node_weights: Bool) -> List[UInt8]:
    """The test quad as a .glb (see the module docstring)."""
    var g = GlbBuild()
    var zero = Vec3(0, 0, 0)
    var pos = g.floats(_vec3s(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 0, 1), Vec3(0, 0, 1)), "VEC3", 3)
    var up = Vec3(0, 1, 0)
    var nrm = g.floats(_vec3s(up, up, up, up), "VEC3", 3)
    var idx = g.u16_indices([0, 2, 1, 0, 3, 2])
    var targets = List[String]()
    for t in range(TARGETS):
        var dp: List[Float32]
        var dn: List[Float32]
        if t == 0:
            dp = _vec3s(zero, zero, Vec3(0, 1, 0), zero)
            dn = _vec3s(zero, zero, Vec3(0.6, -0.2, 0), zero)
        else:
            dp = _vec3s(zero, Vec3(0, Float32(0.1) * Float32(t), 0), zero, zero)
            dn = _vec3s(zero, zero, zero, zero)
        var a_dp = g.floats(dp, "VEC3", 3)
        var a_dn = g.floats(dn, "VEC3", 3)
        targets.append('{"POSITION":' + String(a_dp) + ',"NORMAL":' + String(a_dn) + "}")
    var attributes = '"POSITION":' + String(pos) + ',"NORMAL":' + String(nrm)
    if skinned:
        var joints = g.u8_vec4([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        var weights = g.floats([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], "VEC4", 4)
        attributes += ',"JOINTS_0":' + String(joints) + ',"WEIGHTS_0":' + String(weights)

    # Clip 0 "lift" (LINEAR): target 0 goes 0 -> 1 over two seconds. Clip 1 "snap" (STEP): 0, then 1 at t=1.
    var key0 = List[Float32](length=TARGETS, fill=0)
    var key1 = List[Float32](length=TARGETS, fill=0)
    key1[0] = 1
    var keys = key0.copy()
    keys.extend(key1.copy())
    var lift_in = g.floats([0, 2], "SCALAR", 1)
    var lift_out = g.floats(keys, "SCALAR", 1)
    var snap_in = g.floats([0, 1], "SCALAR", 1)
    var snap_out = g.floats(keys, "SCALAR", 1)

    var default_w = List[String]()
    var node_w = List[String]()
    for t in range(TARGETS):
        default_w.append("0.25" if t == 0 else "0")
        node_w.append("0.75" if t == 0 else "0")

    var json = String('{"asset":{"version":"2.0"},"scene":0,')
    json += '"scenes":[{"nodes":[0' + (",1" if skinned else "") + "]}],"
    var node0 = '{"name":"Quad","mesh":0'
    if skinned:
        node0 += ',"skin":0'
    if node_weights:
        node0 += ',"weights":[' + _joined(node_w) + "]"
    node0 += "}"
    json += '"nodes":[' + node0 + (',{"name":"Bone"}' if skinned else "") + "],"
    if skinned:
        json += '"skins":[{"joints":[1]}],'
    json += '"meshes":[{"name":"Quad","weights":[' + _joined(default_w) + '],"primitives":[{"attributes":{' + attributes + '},"indices":' + String(idx) + ',"targets":[' + _joined(targets) + "]}]}],"
    json += '"animations":['
    json += '{"name":"lift","samplers":[{"input":' + String(lift_in) + ',"output":' + String(lift_out) + ',"interpolation":"LINEAR"}],"channels":[{"sampler":0,"target":{"node":0,"path":"weights"}}]},'
    json += '{"name":"snap","samplers":[{"input":' + String(snap_in) + ',"output":' + String(snap_out) + ',"interpolation":"STEP"}],"channels":[{"sampler":0,"target":{"node":0,"path":"weights"}}]}'
    json += "],"
    json += '"accessors":[' + _joined(g.accessors) + "],"
    json += '"bufferViews":[' + _joined(g.views) + "],"
    json += '"buffers":[{"byteLength":' + String(len(g.bin)) + "}]}"

    var json_bytes = List[UInt8]()
    for b in json.as_bytes():
        json_bytes.append(b)
    pad4(json_bytes, 0x20)
    pad4(g.bin)
    var glb = List[UInt8]()
    append_u32_le(glb, 0x46546C67)
    append_u32_le(glb, 2)
    append_u32_le(glb, 12 + 8 + len(json_bytes) + 8 + len(g.bin))
    append_u32_le(glb, len(json_bytes))
    append_u32_le(glb, 0x4E4F534A)
    glb.extend(json_bytes^)
    append_u32_le(glb, len(g.bin))
    append_u32_le(glb, 0x004E4942)
    glb.extend(g.bin.copy())
    return glb^


def _close(a: Float32, b: Float32, what: String, tol: Float32 = TOL) raises:
    if abs(a - b) > tol:
        print("  mismatch " + what + ": got " + String(a) + ", want " + String(b))
    assert_true(abs(a - b) <= tol)


def _close3(a: Vec3, b: Vec3, what: String, tol: Float32 = TOL) raises:
    _close(a.x, b.x, what + ".x", tol)
    _close(a.y, b.y, what + ".y", tol)
    _close(a.z, b.z, what + ".z", tol)


def _lift_weights(w0: Float32) -> List[Float32]:
    var w = List[Float32](length=TARGETS, fill=0)
    w[0] = w0
    return w^


def _lifted_normal(w: Float32) -> Vec3:
    return (Vec3(0, 1, 0) + Vec3(0.6, -0.2, 0) * w).normalized()


# --------------------------------------------------------------------------------------------
# (a) loader + sampling, (b) CPU side
# --------------------------------------------------------------------------------------------

def test_loader_and_sampling() raises:
    var model = parse_native_glb(build_quad_glb(skinned=False, node_weights=False))
    assert_equal(len(model.meshes), 1)
    ref mesh = model.meshes[0]
    assert_equal(mesh.morph_count, TARGETS)
    assert_equal(mesh.morph_first, 0)
    assert_equal(len(mesh.morph_positions), TARGETS * 4 * 3)
    assert_equal(len(model.morph_names), TARGETS)
    assert_true(model.morph_names[0] == "Quad.morph0")
    assert_true(model.morph_names[7] == "Quad.morph7")
    _close(model.morph_default_weights[0], 0.25, "mesh.weights[0]")
    _close(model.morph_weights[0], 0.25, "morph_weights starts at the defaults")
    _close(model.mesh_morph_weights(0)[0], 0.25, "mesh_morph_weights")
    # Weight tracks are kept out of the pose channels.
    assert_equal(len(model.animations), 2)
    assert_equal(len(model.animations[0].channels), 0)
    assert_equal(len(model.animations[0].weight_channels), 1)
    assert_true(model.has_morph_animation(0))
    _close(model.animations[0].duration, 2.0, "lift duration")
    # LINEAR: 0, 0.5, 1 at t = 0, 1, 2; clamped outside; other targets stay 0.
    _close(model.sample_morph_weights(0, 0.0)[0], 0.0, "lift t=0")
    _close(model.sample_morph_weights(0, 1.0)[0], 0.5, "lift t=1")
    _close(model.sample_morph_weights(0, 2.0)[0], 1.0, "lift t=2")
    _close(model.sample_morph_weights(0, 9.0)[0], 1.0, "lift clamped")
    _close(model.sample_morph_weights(0, 1.0)[3], 0.0, "lift other target")
    # STEP holds the earlier key.
    _close(model.sample_morph_weights(1, 0.5)[0], 0.0, "snap t=0.5")
    _close(model.sample_morph_weights(1, 1.0)[0], 1.0, "snap t=1")
    # No clip: the defaults.
    _close(model.sample_morph_weights(-1, 0.0)[0], 0.25, "no clip")
    model.animate_morph_weights(0, 1.0)
    _close(model.morph_weights[0], 0.5, "animate_morph_weights")

    var overridden = parse_native_glb(build_quad_glb(skinned=False, node_weights=True))
    _close(overridden.morph_default_weights[0], 0.75, "node.weights overrides mesh.weights")

    var skinned = parse_native_glb(build_quad_glb(skinned=True, node_weights=False))
    assert_equal(skinned.meshes[0].morph_count, TARGETS)
    assert_equal(len(skinned.meshes[0].joints), 16)
    assert_equal(len(skinned.skins), 1)
    print("  [ok] loader: 8 targets, names, defaults, node override, LINEAR/STEP weight tracks")


def test_cpu_side() raises:
    var model = parse_native_glb(build_quad_glb(skinned=False, node_weights=False))
    ref mesh = model.meshes[0]
    var csr = build_morph_csr(mesh.vertex_count(), mesh.morph_count, mesh.morph_positions, mesh.morph_normals)
    assert_equal(csr.targets, TARGETS)
    assert_equal(csr.entries(), 8)  # v1 x targets 1..7, v2 x target 0
    assert_equal(len(csr.rows), 5)
    assert_equal(Int(csr.rows[0]), 0)
    assert_equal(Int(csr.rows[1]), 0)
    assert_equal(Int(csr.rows[2]), 7)
    assert_equal(Int(csr.rows[3]), 8)
    assert_equal(Int(csr.rows[4]), 8)
    for w in [Float32(0), Float32(0.5), Float32(1)]:
        var moved = morph_vertex_cpu(csr, 2, _lift_weights(w), Vec3(1, 0, 1), Vec3(0, 1, 0))
        _close3(moved[0], Vec3(1, w, 1), "cpu v2")
    var all_on = List[Float32](length=TARGETS, fill=1)
    _close3(morph_vertex_cpu(csr, 1, all_on, Vec3(1, 0, 0), Vec3(0, 1, 0))[0], Vec3(1, 2.8, 0), "cpu v1, 8 targets")
    # A mesh uploaded static is baked at its weights.
    var baked = morphed_mesh(mesh, _lift_weights(0.5))
    _close(baked.positions[2 * 3 + 1], 0.5, "baked v2.y")
    _close3(Vec3(baked.normals[6], baked.normals[7], baked.normals[8]), _lifted_normal(0.5), "baked v2 normal")
    _close(baked.positions[1 * 3 + 1], 0.0, "baked v1.y")
    # The CPU skinning reference reads the mesh's own weights.
    _close(model.mesh_morph_weights(0)[0], 0.25, "mesh weights")
    var rep = validate_spirv_module(aaa_dq_skinning_shader(BIND_AAA_VERTICES, BIND_AAA_SKINNING))
    for e in rep.errors:
        print("  SPIR-V: " + e)
    assert_true(rep.valid)
    print("  [ok] CPU: sparse rows, reference evaluator, static bake, skinning kernel validates")


# --------------------------------------------------------------------------------------------
# (c) GPU skinning pass
# --------------------------------------------------------------------------------------------

def _vertex(ren: FlintAaaVulkanRenderer, entry: AaaSkinnedMeshEntry, v: Int) -> Tuple[Vec3, Vec3]:
    var p = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=ren.buf_vertices.cpu_address + (entry.mesh_entry.first_vertex + v) * 12 * 4)
    return (
        Vec3(p[unsafe_offset=0], p[unsafe_offset=1], p[unsafe_offset=2]),
        Vec3(p[unsafe_offset=3], p[unsafe_offset=4], p[unsafe_offset=5]),
    )


def _prev_position(ren: FlintAaaVulkanRenderer, entry: AaaSkinnedMeshEntry, v: Int) -> Vec3:
    var p = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=ren.buf_skinning.cpu_address + (AAA_SKIN_PREV_POS_BASE + (entry.skin_bind_first_slot + v) * 3) * 4)
    return Vec3(p[unsafe_offset=0], p[unsafe_offset=1], p[unsafe_offset=2])


def _frame(mut ren: FlintAaaVulkanRenderer, entry: AaaSkinnedMeshEntry) raises:
    ren.begin_frame(Vec3(0.5, 2.0, -2.0), Vec3(0.5, 0.0, 0.5), fov_y_deg=55.0)
    ren.draw_instance(entry.mesh_entry.mesh_id, Mat4(), Mat4(), has_prev=True)
    assert_true(ren.render_frame(capture_readback=True))


def _plain_quad() -> NativeMesh:
    var m = NativeMesh()
    m.positions = [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    m.normals = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    m.texcoords = [0, 0, 1, 0, 1, 1, 0, 1]
    m.indices = [0, 2, 1, 0, 3, 2]
    m.joints = List[UInt16](length=16, fill=0)
    m.weights = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    return m^


def test_gpu_skinning() raises:
    var ren = FlintAaaVulkanRenderer(width=64, height=64)
    assert_true(ren.active)
    ren.enable_vsm = False
    ren.svgf_passes = 0
    var mat = ren.add_material(AaaMaterial(base_color=Vec3(0.8, 0.8, 0.8), roughness=0.6, metallic=0.0))
    var turn = DualQuat.from_quat_trans(Quat(0, 0, Float32(0.70710678), Float32(0.70710678)), Vec3(0, 0, 0))
    var turned = List[DualQuat](length=1, fill=turn)

    # 1. A skinned mesh without targets: no morph region, the result of the bone blend alone.
    var plain_mesh = _plain_quad()
    var plain = ren.add_skinned_mesh(plain_mesh, List[UInt32](), List[Vec3](), mat, bone_count=1)
    assert_equal(plain.morph_count, 0)
    assert_equal(ren.skin_morph_capacity, 0)
    var buffer_before = ren.buf_skinning.size
    _ = ren.update_skinned_mesh_bones(plain, turned, turned)
    var plain_v2 = _vertex(ren, plain, 2)
    _close3(plain_v2[0], turn.transform_point(Vec3(1, 0, 1)), "plain v2", 1e-4)
    _close3(plain_v2[1], turn.transform_dir(Vec3(0, 1, 0)), "plain v2 normal", 1e-4)

    # 2. Morph-only mesh from the GLB (no joints): one identity bone, the node places it.
    var model = parse_native_glb(build_quad_glb(skinned=False, node_weights=False))
    var up = ren.upload_native_model(model, build_lods=False, upload_skinned=True, upload_morph_only=True)
    assert_equal(len(up.skinned_entries), 1)
    var e = up.skinned_entries[0].copy()
    assert_true(e.morph_only)
    assert_equal(e.model_mesh, 0)
    assert_equal(e.morph_count, TARGETS)
    assert_true(ren.skin_morph_capacity > 0)
    assert_true(ren.buf_skinning.size > buffer_before)
    # Its default weight (0.25) applies on the first frame, with no call.
    _frame(ren, e)
    _close(_vertex(ren, e, 2)[0].y, 0.25, "default weight on the GPU")
    # Static weights 0, 0.5, 1 (immediate dispatch), position and normal; the other corners stay.
    for w in [Float32(0), Float32(0.5), Float32(1)]:
        _ = ren.update_skinned_mesh_morph_weights(e, _lift_weights(w), defer_to_frame=False)
        var v2 = _vertex(ren, e, 2)
        _close3(v2[0], Vec3(1, w, 1), "gpu v2 at " + String(w))
        _close3(v2[1], _lifted_normal(w), "gpu v2 normal at " + String(w))
        _close3(_vertex(ren, e, 0)[0], Vec3(0, 0, 0), "gpu v0")
        _close3(_vertex(ren, e, 3)[0], Vec3(0, 0, 1), "gpu v3")
    # Previous-frame weights feed the motion vectors.
    _ = ren.update_skinned_mesh_morph_weights(e, _lift_weights(1.0), _lift_weights(0.2), defer_to_frame=False)
    _close(_prev_position(ren, e, 2).y, 0.2, "previous-frame weight")
    # All 8 targets at once.
    _ = ren.update_skinned_mesh_morph_weights(e, List[Float32](length=TARGETS, fill=1), defer_to_frame=False)
    _close3(_vertex(ren, e, 1)[0], Vec3(1, 2.8, 0), "gpu v1, 8 targets", 1e-4)
    _close3(_vertex(ren, e, 2)[0], Vec3(1, 1, 1), "gpu v2, 8 targets")
    # Animated: the LINEAR clip sampled at t = 0, 1, 2, through the deferred per-frame dispatch.
    var times: List[Float32] = [0.0, 1.0, 2.0]
    var expect: List[Float32] = [0.0, 0.5, 1.0]
    for i in range(3):
        model.animate_morph_weights(0, times[i])
        _ = ren.update_skinned_mesh_morph_weights(e, model.mesh_morph_weights(e.model_mesh))
        _ = ren.update_skinned_mesh_morph_weights(e, model.mesh_morph_weights(e.model_mesh))  # one dispatch still
        assert_equal(len(ren.pending_skin_updates), 1)
        _frame(ren, e)
        _close(_vertex(ren, e, 2)[0].y, expect[i], "animated v2 at t=" + String(times[i]))

    # 3. A skinned mesh with targets: morphs move the bind pose, then the bones turn it.
    var rig = parse_native_glb(build_quad_glb(skinned=True, node_weights=False))
    var rig_up = ren.upload_native_model(rig, build_lods=False, upload_skinned=True)
    assert_equal(len(rig_up.skinned_entries), 1)
    var r = rig_up.skinned_entries[0].copy()
    assert_true(not r.morph_only)
    assert_equal(r.morph_count, TARGETS)
    _ = ren.update_skinned_mesh_bones(r, turned, turned, defer_to_frame=True)
    _ = ren.update_skinned_mesh_morph_weights(r, _lift_weights(0.5), defer_to_frame=False)
    _close3(_vertex(ren, r, 2)[0], turn.transform_point(Vec3(1, 0.5, 1)), "skinned morph v2", 1e-4)
    _close3(_vertex(ren, r, 2)[0], Vec3(-0.5, 1, 1), "skinned morph v2 (morph then turn)", 1e-4)
    _close3(_vertex(ren, r, 2)[1], turn.transform_dir(_lifted_normal(0.5)), "skinned morph v2 normal", 1e-4)

    # 4. Without upload_morph_only, the morph-only mesh is static, baked at its current weights.
    model.morph_weights = _lift_weights(1.0)
    var static_up = ren.upload_native_model(model, build_lods=False, upload_skinned=True)
    assert_equal(len(static_up.skinned_entries), 0)
    var sp = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=ren.buf_vertices.cpu_address + (static_up.mesh_entries[0].first_vertex + 2) * 12 * 4)
    _close(sp[unsafe_offset=1], 1.0, "static upload baked v2.y")

    # 5. The mesh without targets is unchanged after the morph region grew.
    _ = ren.update_skinned_mesh_bones(plain, turned, turned)
    var again = _vertex(ren, plain, 2)
    _close3(again[0], plain_v2[0], "plain v2 after growth", 0.0)
    _close3(again[1], plain_v2[1], "plain v2 normal after growth", 0.0)
    ren.shutdown()
    print("  [ok] GPU: weights 0/0.5/1 static and animated, 8 targets, prev weights, morph-only, morph-then-bones, no-target mesh unchanged")


def main() raises:
    print("[Flint v5 Morph Targets] glTF morph targets: loader, sampling, GPU skinning pass...")
    test_loader_and_sampling()
    test_cpu_side()
    test_gpu_skinning()
    print("[Flint v5 Morph Targets] ALL PASSED")
