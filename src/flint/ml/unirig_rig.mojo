"""UniRig end to end (`flint.ml.unirig_rig`): mesh in, skeleton + skin weights out.

`UniRig.load(dir)` opens the five converted graphs (see `tools/unirig/export.py`):
`unirig_skel_encoder.onnx`, `unirig_skel_opt.onnx`, `unirig_skin_global.onnx`,
`unirig_skin_body.onnx` and `unirig_skin_head.onnx`. `rig(v, f)` works on one triangle soup in
UniRig's Z-up frame; `rig_native_model` takes a loaded glTF scene (Y-up, any node hierarchy),
bakes it to world space, rigs it and returns a skinned copy (joint nodes, one skin, four
influences per vertex) ready for `write_native_glb`.
"""

from std.math import sqrt
from std.time import perf_counter_ns
from flint.core.math import Vec3
from flint.core.transform import Mat4
from flint.assets.model import NativeModel, NativeMesh, NativeNode, NativeSkin
from flint.ml.tensor import Tensor
from flint.ml.onnx import OnnxModel
from flint.ml.opt import OptDecoder
from flint.ml.unirig import UrSkeleton, UrGenConfig, UrRng, UrVec3, generate_skeleton
from flint.ml.unirig_skin import UniRigSkin, normals_of, sample_mesh, michelangelo_queries

comptime UR_SKEL_SAMPLES: Int = 65536
comptime UR_SKEL_VERTEX_SAMPLES: Int = 8192
comptime UR_SKEL_TOKENS: Int = 1024
comptime UR_MAX_INFLUENCES: Int = 4


struct UrRig(Movable):
    """A rig in the input's space: skeleton and dense weights [V, J] (rows sum to 1)."""
    var skeleton: UrSkeleton
    var weights: List[Float32]

    def __init__(out self, var skeleton: UrSkeleton, var weights: List[Float32]):
        self.skeleton = skeleton^
        self.weights = weights^


def _split(rows: List[Float32], width: Int, first: Int) -> List[Float32]:
    var n = len(rows) // width
    var out = List[Float32](length=n * 3, fill=0.0)
    for i in range(n):
        for c in range(3):
            out[i * 3 + c] = rows[i * width + first + c]
    return out^


struct UniRig(Movable):
    var encoder: OnnxModel
    var decoder: OptDecoder
    var skinner: UniRigSkin
    var verbose: Bool

    def __init__(out self, var encoder: OnnxModel, var decoder: OptDecoder, var skinner: UniRigSkin):
        self.encoder = encoder^
        self.decoder = decoder^
        self.skinner = skinner^
        self.verbose = False

    @staticmethod
    def load(dir: String, gpu: Bool = True) raises -> UniRig:
        var enc = OnnxModel.load(dir + "/unirig_skel_encoder.onnx")
        if gpu:
            _ = enc.use_gpu()
        var dec = OptDecoder.load(dir + "/unirig_skel_opt.onnx", gpu)
        var sk = UniRigSkin.load(dir, gpu)
        return UniRig(enc^, dec^, sk^)

    def device(self) raises -> String:
        return self.skinner.device()

    def _log(self, what: String, t0: Int):
        if self.verbose:
            print("[unirig]", what, Float64(perf_counter_ns() - t0) / 1e9, "s")

    def skeleton(mut self, v: List[Float32], f: List[Int], cfg: UrGenConfig) raises -> UrSkeleton:
        """Skeleton for a Z-up triangle mesh (flat xyz, flat triangle indices), in the mesh's space."""
        var nv = len(v) // 3
        if nv == 0 or len(f) < 3:
            raise Error("UniRig: empty mesh")
        var lo = List[Float64](length=3, fill=1e300)
        var hi = List[Float64](length=3, fill=-1e300)
        for i in range(nv):
            for c in range(3):
                lo[c] = min(lo[c], Float64(v[i * 3 + c]))
                hi[c] = max(hi[c], Float64(v[i * 3 + c]))
        var scale = max(max(hi[0] - lo[0], hi[1] - lo[1]), hi[2] - lo[2]) / 2
        if scale <= 0:
            raise Error("UniRig: degenerate mesh bounds")
        var vn = List[Float32](length=nv * 3, fill=0.0)
        for i in range(nv):
            for c in range(3):
                vn[i * 3 + c] = Float32((Float64(v[i * 3 + c]) - (lo[c] + hi[c]) / 2) / scale)
        var t0 = perf_counter_ns()
        var nrm = List[Float32]()
        var fnrm = List[Float32]()
        normals_of(vn, f, nrm, fnrm)
        var rng = UrRng(cfg.seed * 0x9E3779B97F4A7C15 + 3)
        var smp = sample_mesh(vn, f, nrm, fnrm, UR_SKEL_SAMPLES, UR_SKEL_VERTEX_SAMPLES, rng)
        var q = michelangelo_queries(smp.xyz, smp.nrm, rng, UR_SKEL_TOKENS)
        var n = smp.count()
        var m = len(q) // 6
        self.encoder.set_input("pc", Tensor.floats([1, n, 3], smp.xyz.copy()))
        self.encoder.set_input("feats", Tensor.floats([1, n, 3], smp.nrm.copy()))
        self.encoder.set_input("qpc", Tensor.floats([1, m, 3], _split(q, 6, 0)))
        self.encoder.set_input("qfeats", Tensor.floats([1, m, 3], _split(q, 6, 3)))
        self.encoder.run()
        var cond = self.encoder.output("cond")
        var c2 = cond.reshaped([cond.shape[1], cond.shape[2]])
        self._log("skeleton encoder", t0)
        t0 = perf_counter_ns()
        var sk = generate_skeleton(self.decoder, c2, cfg)
        self._log("skeleton decoder (" + String(sk.num_bones()) + " bones)", t0)
        var center = UrVec3(Float32((lo[0] + hi[0]) / 2), Float32((lo[1] + hi[1]) / 2), Float32((lo[2] + hi[2]) / 2))
        return sk.transformed(center, Float32(scale))

    def rig(mut self, v: List[Float32], f: List[Int], cfg: UrGenConfig) raises -> UrRig:
        """Skeleton + dense skin weights [V, J] for a Z-up triangle mesh."""
        var sk = self.skeleton(v, f, cfg)
        var J = sk.num_bones()
        var joints = List[Float32](length=J * 3, fill=0.0)
        for j in range(J):
            joints[j * 3] = sk.joints[j].x
            joints[j * 3 + 1] = sk.joints[j].y
            joints[j * 3 + 2] = sk.joints[j].z
        var t0 = perf_counter_ns()
        var w = self.skinner.skin(v, f, joints, sk.parents, cfg.seed)
        self._log("skin", t0)
        return UrRig(sk^, w^)


def top_influences(weights: List[Float32], J: Int, mut joints: List[UInt16], mut out_w: List[Float32]):
    """Dense [V, J] weights -> the 4 largest per vertex (renormalised; a vertex with no weight is
    bound fully to joint 0)."""
    var nv = len(weights) // J if J > 0 else 0
    joints = List[UInt16](length=nv * UR_MAX_INFLUENCES, fill=0)
    out_w = List[Float32](length=nv * UR_MAX_INFLUENCES, fill=0.0)
    for i in range(nv):
        var best = List[Int](length=UR_MAX_INFLUENCES, fill=-1)
        for j in range(J):
            var wj = weights[i * J + j]
            if wj <= 0:
                continue
            var slot = UR_MAX_INFLUENCES
            while slot > 0 and (best[slot - 1] < 0 or weights[i * J + best[slot - 1]] < wj):
                slot -= 1
            if slot < UR_MAX_INFLUENCES:
                var k = UR_MAX_INFLUENCES - 1
                while k > slot:
                    best[k] = best[k - 1]
                    k -= 1
                best[slot] = j
        var tot: Float32 = 0
        for s in range(UR_MAX_INFLUENCES):
            if best[s] >= 0:
                tot += weights[i * J + best[s]]
        if tot <= 0:
            out_w[i * UR_MAX_INFLUENCES] = 1.0
            continue
        for s in range(UR_MAX_INFLUENCES):
            if best[s] >= 0:
                joints[i * UR_MAX_INFLUENCES + s] = UInt16(best[s])
                out_w[i * UR_MAX_INFLUENCES + s] = weights[i * J + best[s]] / tot


@always_inline
def _yup_to_zup(p: Vec3) -> Vec3:
    """glTF Y-up -> UniRig Z-up (Blender's importer: (x, -z, y))."""
    return Vec3(p.x, -p.z, p.y)


@always_inline
def _zup_to_yup(p: UrVec3) -> Vec3:
    return Vec3(p.x, p.z, -p.y)


def _unit(v: Vec3) -> Vec3:
    var l = sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    if l <= 1e-20:
        return Vec3(0, 1, 0)
    return Vec3(v.x / l, v.y / l, v.z / l)


def rig_native_model(mut rig: UniRig, mut src: NativeModel, cfg: UrGenConfig) raises -> NativeModel:
    """Rigs every mesh instance of a glTF scene together: meshes are baked to world space (one
    node each, identity transform), a joint node tree is added and all meshes share one skin."""
    # mesh instances: (mesh, world matrix)
    var inst_mesh = List[Int]()
    var inst_node = List[Int]()
    var inst_xf = List[Mat4]()
    if len(src.nodes) == 0:
        for mi in range(len(src.meshes)):
            inst_mesh.append(mi)
            inst_node.append(-1)
            inst_xf.append(Mat4())
    else:
        var world = src.global_transforms()
        for n in range(len(src.nodes)):
            var ms = src.nodes[n].mesh_indices.copy()
            if len(ms) == 0 and src.nodes[n].mesh >= 0:
                ms.append(src.nodes[n].mesh)
            for mi in ms:
                if mi >= 0 and mi < len(src.meshes):
                    inst_mesh.append(mi)
                    inst_node.append(n)
                    inst_xf.append(world[n].copy())
    var out = NativeModel()
    out.textures = src.textures.copy()
    var v = List[Float32]()
    var f = List[Int]()
    var base = List[Int]()
    for k in range(len(inst_mesh)):
        ref m = src.meshes[inst_mesh[k]]
        var nv = m.vertex_count()
        base.append(len(v) // 3)
        var mesh = m.copy()
        for i in range(nv):
            var p = inst_xf[k].transform_point(Vec3(m.positions[i * 3], m.positions[i * 3 + 1], m.positions[i * 3 + 2]))
            mesh.positions[i * 3] = p.x
            mesh.positions[i * 3 + 1] = p.y
            mesh.positions[i * 3 + 2] = p.z
            var z = _yup_to_zup(p)
            v.append(z.x)
            v.append(z.y)
            v.append(z.z)
            if i * 3 + 2 < len(m.normals):
                var nn = _unit(inst_xf[k].transform_vector(Vec3(m.normals[i * 3], m.normals[i * 3 + 1], m.normals[i * 3 + 2])))
                mesh.normals[i * 3] = nn.x
                mesh.normals[i * 3 + 1] = nn.y
                mesh.normals[i * 3 + 2] = nn.z
        for d in range(len(mesh.morph_positions) // 3):
            var dp = inst_xf[k].transform_vector(Vec3(mesh.morph_positions[d * 3], mesh.morph_positions[d * 3 + 1], mesh.morph_positions[d * 3 + 2]))
            mesh.morph_positions[d * 3] = dp.x
            mesh.morph_positions[d * 3 + 1] = dp.y
            mesh.morph_positions[d * 3 + 2] = dp.z
        for d in range(len(mesh.morph_normals) // 3):
            var dn = inst_xf[k].transform_vector(Vec3(mesh.morph_normals[d * 3], mesh.morph_normals[d * 3 + 1], mesh.morph_normals[d * 3 + 2]))
            mesh.morph_normals[d * 3] = dn.x
            mesh.morph_normals[d * 3 + 1] = dn.y
            mesh.morph_normals[d * 3 + 2] = dn.z
        if len(m.indices) > 0:
            for idx in m.indices:
                f.append(base[k] + idx)
        else:
            for idx in range(nv - nv % 3):
                f.append(base[k] + idx)
        mesh.centers = List[Float32]()
        mesh.bounds_verts = 0
        out.meshes.append(mesh^)
    if len(v) == 0:
        raise Error("UniRig: the model has no mesh vertices")

    var r = rig.rig(v, f, cfg)
    ref sk = r.skeleton
    var J = sk.num_bones()
    var jt = List[UInt16]()
    var jw = List[Float32]()
    top_influences(r.weights, J, jt, jw)

    # mesh nodes, then joint nodes
    for k in range(len(inst_mesh)):
        var node = NativeNode()
        node.name = src.nodes[inst_node[k]].name if inst_node[k] >= 0 else "mesh_" + String(k)
        node.mesh = k
        out.nodes.append(node^)
        out.roots.append(k)
        var nv = out.meshes[k].vertex_count()
        var b = base[k]
        out.meshes[k].joints = List[UInt16](length=nv * UR_MAX_INFLUENCES, fill=0)
        out.meshes[k].weights = List[Float32](length=nv * UR_MAX_INFLUENCES, fill=0.0)
        for e in range(nv * UR_MAX_INFLUENCES):
            out.meshes[k].joints[e] = jt[b * UR_MAX_INFLUENCES + e]
            out.meshes[k].weights[e] = jw[b * UR_MAX_INFLUENCES + e]
        out.meshes[k].skin = 0
    var first_joint = len(out.nodes)
    var skin = NativeSkin()
    for j in range(J):
        var node = NativeNode()
        node.name = sk.names[j] if j < len(sk.names) and sk.names[j].byte_length() > 0 else "bone_" + String(j)
        var wp = _zup_to_yup(sk.joints[j])
        var p = sk.parents[j]
        if p >= 0:
            var pp = _zup_to_yup(sk.joints[p])
            node.translation = Vec3(wp.x - pp.x, wp.y - pp.y, wp.z - pp.z)
        else:
            node.translation = wp
        node.base_translation = node.translation
        out.nodes.append(node^)
        skin.joints.append(first_joint + j)
        var ib = Mat4()
        for c in range(16):
            skin.inverse_bind.append(ib.values[c])
        skin.inverse_bind[j * 16 + 12] = -wp.x
        skin.inverse_bind[j * 16 + 13] = -wp.y
        skin.inverse_bind[j * 16 + 14] = -wp.z
    for j in range(J):
        var p = sk.parents[j]
        if p >= 0:
            out.nodes[first_joint + p].children.append(first_joint + j)
        else:
            out.roots.append(first_joint + j)
    out.skins.append(skin^)
    out.max_influences = UR_MAX_INFLUENCES
    return out^
