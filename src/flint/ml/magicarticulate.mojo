"""MagicArticulate and Puppeteer skeleton generation (`flint.ml.magicarticulate`).

Pure-Mojo ports of the autoregressive skeleton generators of MagicArticulate (CVPR 2025) and
Puppeteer (NeurIPS 2025), both by ByteDance Seed / NTU, Apache-2.0 (see `licenses/`):

  1. The mesh is made watertight and point-sampled (`flint.ml.magicarticulate_mesh`), and the
     8192 points with normals are encoded into 257 conditioning tokens by the Michelangelo
     perceiver plus the models' projections (`<variant>_encoder.onnx`, run by `flint.ml.onnx`).
  2. A post-LayerNorm OPT-350M (`flint.ml.opt`, GPU-resident KV cache) continues from those
     tokens. Every generated token also gets a learned "cond" embedding, a slot embedding (which
     coordinate of a bone / joint it is) and, for Puppeteer, a target-aware position embedding
     per joint. Decoding is greedy, as in both demos.
  3. Tokens are 128-step coordinates in [-0.5, 0.5]^3 (+3 for BOS / EOS / PAD):
       MagicArticulate: 6 per bone (parent xyz, child xyz), hierarchical or spatial order;
       Puppeteer:       4 per joint (xyz, parent index + 1, 0 for none).
     The demos' post-processing (unique joints, bone de-duplication, nearest-component merging)
     and root choice are reproduced exactly; then the joints are ordered parent-first.

Skin weights are not part of MagicArticulate's release, and Puppeteer's skinning network depends
on NVIDIA PartField (non-commercial), so `MagicArticulate.rig` weights the skeleton with the
UniRig skin network (`flint.ml.unirig_skin`, MIT) when its graphs are given.
"""

from std.math import sqrt
from std.time import perf_counter_ns
from std.collections import Optional
from flint.ml.tensor import Tensor, fptr_at
from flint.ml.onnx import OnnxModel
from flint.ml.opt import OptDecoder
from flint.ml.unirig import UrSkeleton, UrVec3
from flint.ml.unirig_skin import UniRigSkin
from flint.ml.unirig_rig import UrRig, BakedScene, bake_native_model, skin_baked_model
from flint.ml.magicarticulate_mesh import MaPointCloud, mesh_point_cloud
from flint.assets.model import NativeModel

comptime MA_BOS: Int = 0
comptime MA_EOS: Int = 1
comptime MA_PAD: Int = 2
comptime MA_SPECIAL: Int = 3
comptime MA_DISCRETE: Int = 128
comptime MA_COND_LEN: Int = 257
comptime MA_POINTS: Int = 8192
comptime MA_MERGE_TOL: Float64 = 0.0025


struct MaConfig(Copyable, Movable):
    """Which model and how to run it (the demo scripts' arguments)."""
    var joint_token: Bool     # Puppeteer: 4 tokens per joint; else 6 per bone (MagicArticulate)
    var hier: Bool            # MagicArticulate hierarchical order (root = first bone's parent); else spatial
    var max_bones: Int        # n_max_bones
    var points: Int           # input_pc_num
    var watertight: Bool      # apply_marching_cubes
    var octree_depth: Int
    var seed: UInt64

    def __init__(out self):
        self.joint_token = False
        self.hier = True
        self.max_bones = 100
        self.points = MA_POINTS
        self.watertight = True
        self.octree_depth = 7
        self.seed = 0

    @staticmethod
    def magicarticulate(hier: Bool = True) -> MaConfig:
        var c = MaConfig()
        c.hier = hier
        return c^

    @staticmethod
    def puppeteer() -> MaConfig:
        var c = MaConfig()
        c.joint_token = True
        c.hier = False
        return c^

    def bone_per_token(self) -> Int:
        return 4 if self.joint_token else 6

    def max_length(self) -> Int:
        var nb = self.max_bones + 1 if self.joint_token else self.max_bones
        return nb * self.bone_per_token() + 2 + MA_COND_LEN

    def generate_length(self) -> Int:
        return self.max_length() - MA_COND_LEN


@always_inline
def _pymod(a: Int, m: Int) -> Int:
    var r = a % m
    return r + m if r < 0 else r


def slot_index(token: Int, k: Int, bone_per_token: Int) -> Int:
    """`OPTBonePositionalEmbedding` index for generated token `k` (0 based; token 0 is the predicted
    BOS): special tokens use their own id, coordinates 3 + (k - 1) mod tokens-per-bone."""
    if token < MA_SPECIAL:
        return token
    return _pymod(k - 1, bone_per_token) + MA_SPECIAL


struct MaEmbeds(Movable):
    """The decoder's extra input embeddings, read from the weights file."""
    var hidden: Int
    var cond1: List[Float32]          # cond_embed[1], added to every generated token
    var slot: List[Float32]           # token_embed_positions [bpt + 3, hidden]
    var slots: Int
    var tape: List[Float32]           # target_aware_pos_embed [n, hidden] (Puppeteer), else empty
    var tape_rows: Int

    def __init__(out self, w: OnnxModel, hidden: Int) raises:
        self.hidden = hidden
        var ce = w.weight("model.decoder.cond_embed.weight")
        var pe = w.weight("model.decoder.token_embed_positions.weight")
        var pce = fptr_at(ce.fptr())
        var ppe = fptr_at(pe.fptr())
        self.cond1 = List[Float32](length=hidden, fill=0.0)
        for i in range(hidden):
            self.cond1[i] = pce.unsafe_offset(hidden + i).unsafe_load[alignment=1]()
        self.slots = pe.shape[0]
        self.slot = List[Float32](length=self.slots * hidden, fill=0.0)
        for i in range(self.slots * hidden):
            self.slot[i] = ppe.unsafe_offset(i).unsafe_load[alignment=1]()
        self.tape = List[Float32]()
        self.tape_rows = 0
        if w.has("target_aware_pos_embed"):
            var t = w.weight("target_aware_pos_embed")
            var pt = fptr_at(t.fptr())
            self.tape_rows = t.shape[0]
            self.tape = List[Float32](length=self.tape_rows * hidden, fill=0.0)
            for i in range(self.tape_rows * hidden):
                self.tape[i] = pt.unsafe_offset(i).unsafe_load[alignment=1]()

    def extra(self, token: Int, k: Int, cfg: MaConfig) -> List[Float32]:
        """Embedding added to generated token `k` beside its token and position embeddings."""
        var h = self.hidden
        var e = self.cond1.copy()
        var bpt = cfg.bone_per_token()
        var s = slot_index(token, k, bpt)
        if s < self.slots:
            for i in range(h):
                e[i] += self.slot[s * h + i]
        if self.tape_rows > 0:
            # SkeletonOPTDecoder: target_pos_embed[k - 1] = tape[1 + (k - 1) // bpt] for the
            # (max joints - 1) * bpt generated positions after the BOS
            var nj = (cfg.max_length() - 2 - MA_COND_LEN) // bpt
            var pos = k - 1
            if pos >= 0 and pos < (nj - 1) * bpt:
                var row = 1 + pos // bpt
                if row < self.tape_rows:
                    for i in range(h):
                        e[i] += self.tape[row * h + i]
        return e^


def generate_ids(mut dec: OptDecoder, cond: Tensor, emb: MaEmbeds, cfg: MaConfig) raises -> List[Int]:
    """Greedy continuation of the 257-token prefix `cond` [257, hidden]; returns the generated ids
    (the first is the model's BOS), ending with EOS unless the length limit was hit."""
    var G = cfg.generate_length()
    var L = cond.shape[0]
    var steps_cap = min(G, dec.max_pos - L)
    var logits = dec.start(cond, steps_cap, 1)
    var V = dec.vocab
    var seq = List[Int]()
    var anc = List[Int32]()
    for step in range(steps_cap):
        var best = 0
        for t in range(1, V):
            if logits[t] > logits[best]:
                best = t
        seq.append(best)
        if best == MA_EOS or len(seq) >= steps_cap:
            break
        anc.append(0)
        var toks = List[Int]()
        toks.append(best)
        logits = dec.step_extra(toks, step, anc, emb.extra(best, step, cfg))
    return seq^


def ids_to_bones(ids: List[Int], cfg: MaConfig, mut bones: List[Int]) -> Bool:
    """`SkeletonGPT.generate` + `detokenize[_joint_token]`: generated ids -> bones as coordinate
    bins [B * 6] (parent xyz, child xyz). False when the sequence does not decode (the demo's None)."""
    bones = List[Int]()
    var G = cfg.generate_length()
    # outputs = EOS-filled [G], generated ids written at the front, then [1:-1] and specials dropped
    var toks = List[Int]()
    for i in range(1, G - 1):
        var t = ids[i] if i < len(ids) else MA_EOS
        if t >= MA_SPECIAL:
            toks.append(t - MA_SPECIAL)
    if not cfg.joint_token:
        if len(toks) % 6 != 0 or len(toks) == 0:
            return False
        bones = toks^
        return True
    if len(toks) % 4 != 0 or len(toks) == 0:
        return False
    var nj = len(toks) // 4
    for c in range(nj):
        var p = toks[c * 4 + 3]
        if p > 0:
            var pi = p - 1
            if pi >= nj:
                return False
            for a in range(3):
                bones.append(toks[pi * 4 + a])
            for a in range(3):
                bones.append(toks[c * 4 + a])
    return len(bones) > 0


@always_inline
def bin_coord(b: Int) -> Float64:
    """`undiscretize`: bin / 128 - 0.5 (exact in float16)."""
    return Float64(b) / Float64(MA_DISCRETE) - 0.5


struct MaSkeleton(Copyable, Movable):
    """A post-processed skeleton in the normalised [-0.5, 0.5] frame: joints [J * 3], `bones`
    [B * 2] (parent, child), the chosen root and parents (-1 for roots), demo order."""
    var joints: List[Float64]
    var bones: List[Int]
    var root: Int
    var parents: List[Int]
    var ids: List[Int]

    def __init__(out self):
        self.joints = List[Float64]()
        self.bones = List[Int]()
        self.root = -1
        self.parents = List[Int]()
        self.ids = List[Int]()

    def num_joints(self) -> Int:
        return len(self.joints) // 3


def _less3(a: List[Int], i: Int, j: Int) -> Bool:
    for c in range(3):
        if a[i * 3 + c] != a[j * 3 + c]:
            return a[i * 3 + c] < a[j * 3 + c]
    return False


def joints_and_bones(bins: List[Int], mut joints: List[Int], mut bones: List[Int]):
    """`pred_joints_and_bones`: np.unique over the parent then child coordinates (lexicographic),
    bones as index pairs without self loops."""
    var b = len(bins) // 6
    var rows = List[Int](capacity=b * 6)
    for i in range(b):
        for c in range(3):
            rows.append(bins[i * 6 + c])
    for i in range(b):
        for c in range(3):
            rows.append(bins[i * 6 + 3 + c])
    var n = 2 * b
    var order = List[Int](capacity=n)
    for i in range(n):
        order.append(i)
    # insertion sort (n <= 2 * 101 * 2)
    for i in range(1, n):
        var j = i
        while j > 0 and _less3(rows, order[j], order[j - 1]):
            var t = order[j]
            order[j] = order[j - 1]
            order[j - 1] = t
            j -= 1
    joints = List[Int]()
    var inv = List[Int](length=n, fill=0)
    var u = -1
    for r in range(n):
        var i = order[r]
        if r == 0 or _less3(rows, order[r - 1], i):
            u += 1
            for c in range(3):
                joints.append(rows[i * 3 + c])
        inv[i] = u
    bones = List[Int]()
    for i in range(b):
        if inv[i] != inv[b + i]:
            bones.append(inv[i])
            bones.append(inv[b + i])


@always_inline
def _d(j: List[Float64], a: Int, b: Int) -> Float64:
    var s: Float64 = 0
    for c in range(3):
        var d = j[a * 3 + c] - j[b * 3 + c]
        s += d * d
    return sqrt(s)


def _components(n: Int, bones: List[Int]) -> List[List[Int]]:
    """`find_connected_components`: BFS from each unvisited joint, neighbours in bone order."""
    var adj = List[List[Int]]()
    for _ in range(n):
        adj.append(List[Int]())
    for i in range(len(bones) // 2):
        adj[bones[i * 2]].append(bones[i * 2 + 1])
        adj[bones[i * 2 + 1]].append(bones[i * 2])
    var seen = List[Bool](length=n, fill=False)
    var comps = List[List[Int]]()
    for i in range(n):
        if seen[i]:
            continue
        var comp = List[Int]()
        comp.append(i)
        seen[i] = True
        var h = 0
        while h < len(comp):
            var node = comp[h]
            h += 1
            for nb in adj[node]:
                if not seen[nb]:
                    seen[nb] = True
                    comp.append(nb)
        comps.append(comp^)
    return comps^


def ensure_connectivity(mut joints: List[Float64], mut bones: List[Int], mut root: Int, threshold: Float64):
    """`ensure_skeleton_connectivity`: while there are several components, take the closest pair of
    joints across components (first pair on ties); merge them below `threshold`, else add a bone."""
    var limit = len(joints) // 3
    var it = 0
    while True:
        var n = len(joints) // 3
        var comps = _components(n, bones)
        if len(comps) <= 1:
            break
        var best: Float64 = 1e300
        var j1 = -1
        var j2 = -1
        for a in range(len(comps)):
            for b in range(a + 1, len(comps)):
                var md: Float64 = 1e300
                var ma = -1
                var mb = -1
                for x in comps[a]:
                    for y in comps[b]:
                        var dd = _d(joints, x, y)
                        if dd < md:
                            md = dd
                            ma = x
                            mb = y
                if md < best:
                    best = md
                    j1 = ma
                    j2 = mb
        if j1 < 0:
            break
        if best < threshold:
            var nb = List[Int]()
            for i in range(len(bones) // 2):
                var p = j1 if bones[i * 2] == j2 else bones[i * 2]
                var c = j1 if bones[i * 2 + 1] == j2 else bones[i * 2 + 1]
                if p != c:
                    nb.append(p)
                    nb.append(c)
            if root == j2:
                root = j1
            var nj = List[Float64]()
            for i in range(n):
                if i != j2:
                    for c in range(3):
                        nj.append(joints[i * 3 + c])
            joints = nj^
            for i in range(len(nb)):
                if nb[i] > j2:
                    nb[i] -= 1
            bones = nb^
            if root > j2:
                root -= 1
        else:
            bones.append(j1)
            bones.append(j2)
        it += 1
        if it > limit:
            break


def merge_and_fix(mut joints: List[Float64], mut bones: List[Int], mut root: Int, tol: Float64 = MA_MERGE_TOL):
    """`merge_duplicate_joints_and_fix_bones` (root = -1 when the demo passes none)."""
    var n = len(joints) // 3
    var rep = List[Int](length=n, fill=-1)
    var used = List[Bool](length=n, fill=False)
    for i in range(n):
        if used[i]:
            continue
        var group = List[Int]()
        group.append(i)
        for j in range(i + 1, n):
            if not used[j] and _d(joints, i, j) < tol:
                group.append(j)
                used[j] = True
        used[i] = True
        var r = group[0]
        if root >= 0:
            for g in group:
                if g == root:
                    r = root
        for g in group:
            rep[g] = r
    var mroot = rep[root] if root >= 0 else -1
    var ub = List[Int]()
    for i in range(len(bones) // 2):
        var p = rep[bones[i * 2]]
        var c = rep[bones[i * 2 + 1]]
        if p == c:
            continue
        var dup = False
        for k in range(len(ub) // 2):
            if ub[k * 2] == p and ub[k * 2 + 1] == c:
                dup = True
                break
        if not dup:
            ub.append(p)
            ub.append(c)
    var keep = List[Bool](length=n, fill=False)
    for x in ub:
        keep[x] = True
    if mroot >= 0:
        keep[mroot] = True
    var remap = List[Int](length=n, fill=-1)
    var nj = List[Float64]()
    var m = 0
    for i in range(n):
        if keep[i]:
            remap[i] = m
            m += 1
            for c in range(3):
                nj.append(joints[i * 3 + c])
    for i in range(len(ub)):
        ub[i] = remap[ub[i]]
    joints = nj^
    bones = ub^
    root = remap[mroot] if mroot >= 0 else -1
    ensure_connectivity(joints, bones, root, tol * 8)


def postprocess(bins: List[Int], cfg: MaConfig, centroid: List[Float64]) -> MaSkeleton:
    """The demo's post-processing and root / hierarchy choice (`save_skeleton_to_txt[_joint]`).
    `centroid` (mesh-vertex mean in the normalised frame) picks the root in spatial order."""
    var sk = MaSkeleton()
    var jb = List[Int]()
    var bones = List[Int]()
    joints_and_bones(bins, jb, bones)
    var joints = List[Float64](capacity=len(jb))
    for b in jb:
        joints.append(bin_coord(b))
    var hier = cfg.hier and not cfg.joint_token
    var root = bones[0] if hier and len(bones) > 0 else -1
    merge_and_fix(joints, bones, root)
    var n = len(joints) // 3
    var nb = len(bones) // 2
    var parents = List[Int](length=n, fill=-1)
    if cfg.joint_token:
        # root: a joint that is a parent but never a child (the smallest; one in practice)
        var is_child = List[Bool](length=n, fill=False)
        for i in range(nb):
            is_child[bones[i * 2 + 1]] = True
        root = -1
        for i in range(nb):
            var p = bones[i * 2]
            if not is_child[p] and (root < 0 or p < root):
                root = p
        if root < 0 and nb > 0:
            root = bones[0]
    elif not hier:
        var best: Float64 = 1e300
        for j in range(n):
            var s: Float64 = 0
            for c in range(3):
                var d = joints[j * 3 + c] - centroid[c]
                s += d * d
            if s < best:
                best = s
                root = j
    if cfg.joint_token or hier:
        var has = List[Bool](length=n, fill=False)
        for i in range(nb):
            var c = bones[i * 2 + 1]
            if not has[c]:
                has[c] = True
                parents[c] = bones[i * 2]
        if root >= 0:
            parents[root] = -1
    elif root >= 0:
        var adj = List[List[Int]]()
        for _ in range(n):
            adj.append(List[Int]())
        for i in range(nb):
            adj[bones[i * 2]].append(bones[i * 2 + 1])
            adj[bones[i * 2 + 1]].append(bones[i * 2])
        var seen = List[Bool](length=n, fill=False)
        var q = List[Int]()
        q.append(root)
        seen[root] = True
        var h = 0
        while h < len(q):
            var cur = q[h]
            h += 1
            for x in adj[cur]:
                if not seen[x]:
                    seen[x] = True
                    parents[x] = cur
                    q.append(x)
    sk.joints = joints^
    sk.bones = bones^
    sk.root = root
    sk.parents = parents^
    return sk^


def tree_order(parents: List[Int], root: Int) -> List[Int]:
    """Parent-first (depth-first preorder) joint order: the root's tree first, then any other
    roots by index. Parent loops (possible with conflicting generated bones) are cut at the joint
    first reached."""
    var n = len(parents)
    var children = List[List[Int]]()
    for _ in range(n):
        children.append(List[Int]())
    for j in range(n):
        if parents[j] >= 0:
            children[parents[j]].append(j)
    var order = List[Int]()
    var seen = List[Bool](length=n, fill=False)
    var starts = List[Int]()
    if root >= 0 and root < n:
        starts.append(root)
    for j in range(n):
        if parents[j] < 0 and j != root:
            starts.append(j)
    for j in range(n):          # whatever is left sits on a parent loop
        starts.append(j)
    for s in starts:
        if seen[s]:
            continue
        var stack = List[Int]()
        stack.append(s)
        while len(stack) > 0:
            var x = stack.pop()
            if seen[x]:
                continue
            seen[x] = True
            order.append(x)
            var c = len(children[x]) - 1
            while c >= 0:
                if not seen[children[x][c]]:
                    stack.append(children[x][c])
                c -= 1
    return order^


def to_ur_skeleton(ms: MaSkeleton, pc: MaPointCloud) -> UrSkeleton:
    """Joints in the mesh's space, ordered parent-first, as a `UrSkeleton` (tails: the only child,
    else half the parent bone extended)."""
    var n = ms.num_joints()
    var order = tree_order(ms.parents, ms.root)
    var pos = List[Int](length=n, fill=-1)
    for i in range(n):
        pos[order[i]] = i
    var sk = UrSkeleton()
    for i in range(n):
        var j = order[i]
        sk.joints.append(UrVec3(Float32(pc.to_mesh(ms.joints[j * 3], 0)), Float32(pc.to_mesh(ms.joints[j * 3 + 1], 1)),
                                Float32(pc.to_mesh(ms.joints[j * 3 + 2], 2))))
        var p = ms.parents[j]
        # a parent placed after its child is on a cut loop: the child becomes a root
        sk.parents.append(pos[p] if p >= 0 and pos[p] < i else -1)
        sk.names.append("joint" + String(j))
    var nchild = List[Int](length=n, fill=0)
    var only = List[Int](length=n, fill=-1)
    for i in range(n):
        var p = sk.parents[i]
        if p >= 0:
            nchild[p] += 1
            only[p] = i
    for i in range(n):
        if nchild[i] == 1:
            sk.tails.append(sk.joints[only[i]])
        else:
            var p = sk.parents[i]
            var d = sk.joints[i] - sk.joints[p] if p >= 0 else UrVec3(0, 0, Float32(0.05 * pc.pc_scale / pc.scale))
            sk.tails.append(sk.joints[i] + d * 0.5)
    sk.tokens = ms.ids.copy()
    return sk^


struct MagicArticulate(Movable):
    """Encoder + decoder of one checkpoint and, optionally, the UniRig skin network."""
    var encoder: OnnxModel
    var decoder: OptDecoder
    var emb: MaEmbeds
    var cfg: MaConfig
    var skinner: Optional[UniRigSkin]
    var verbose: Bool
    var last_ms: MaSkeleton

    def __init__(out self, var encoder: OnnxModel, var decoder: OptDecoder, var cfg: MaConfig) raises:
        self.emb = MaEmbeds(decoder.weights, decoder.hidden)
        self.encoder = encoder^
        self.decoder = decoder^
        self.cfg = cfg^
        self.skinner = None
        self.verbose = False
        self.last_ms = MaSkeleton()

    @staticmethod
    def load(dir: String, variant: String, gpu: Bool = True, skin_dir: String = "") raises -> MagicArticulate:
        """`variant` is "magicarticulate" (hierarchical checkpoint), "magicarticulate-spatial" or
        "puppeteer"; `dir` holds `<variant>_encoder.onnx` and `<variant>_opt.onnx` from
        `tools/magicarticulate/export.py`. `skin_dir` (UniRig graphs) enables `rig`."""
        var base = String("puppeteer") if variant == "puppeteer" else String("magicarticulate")
        var cfg = MaConfig.puppeteer() if variant == "puppeteer" else MaConfig.magicarticulate(variant != "magicarticulate-spatial")
        var enc = OnnxModel.load(dir + "/" + base + "_encoder.onnx")
        if gpu:
            _ = enc.use_gpu()
        var dec = OptDecoder.load(dir + "/" + base + "_opt.onnx", gpu)
        var m = MagicArticulate(enc^, dec^, cfg^)
        if skin_dir.byte_length() > 0:
            m.skinner = UniRigSkin.load(skin_dir, gpu)
        return m^

    def device(self) raises -> String:
        return self.decoder.device()

    def _log(self, what: String, t0: Int):
        if self.verbose:
            print("[magicarticulate]", what, Float64(perf_counter_ns() - t0) / 1e9, "s")

    def encode(mut self, pc: MaPointCloud) raises -> Tensor:
        """Point cloud -> decoder prefix [257, hidden]."""
        var n = pc.count()
        self.encoder.set_input("pc", Tensor.floats([1, n, 3], pc.xyz.copy()))
        self.encoder.set_input("normals", Tensor.floats([1, n, 3], pc.nrm.copy()))
        self.encoder.run()
        var cond = self.encoder.output("cond")
        return cond.reshaped([cond.shape[1], cond.shape[2]])

    def skeleton_from_cloud(mut self, pc: MaPointCloud, centroid: List[Float64]) raises -> MaSkeleton:
        """Normalised point cloud -> post-processed skeleton (normalised frame, demo order)."""
        var t0 = perf_counter_ns()
        var cond = self.encode(pc)
        self._log("encoder", t0)
        t0 = perf_counter_ns()
        var ids = generate_ids(self.decoder, cond, self.emb, self.cfg)
        self._log("decoder (" + String(len(ids)) + " tokens)", t0)
        var bins = List[Int]()
        if not ids_to_bones(ids, self.cfg, bins):
            raise Error("MagicArticulate: the generated tokens do not form a skeleton (" + String(len(ids)) + " ids)")
        var ms = postprocess(bins, self.cfg, centroid)
        ms.ids = ids^
        return ms^

    def skeleton(mut self, v: List[Float32], f: List[Int]) raises -> UrSkeleton:
        """Skeleton for a Z-up triangle mesh (flat xyz, flat indices), in the mesh's space."""
        var nv = len(v) // 3
        if nv == 0 or len(f) < 3:
            raise Error("MagicArticulate: empty mesh")
        var t0 = perf_counter_ns()
        var pc = mesh_point_cloud(v, f, self.cfg.points, self.cfg.seed, self.cfg.watertight, self.cfg.octree_depth)
        self._log("watertight + sampling", t0)
        # mesh-vertex centroid in the normalised frame (spatial-order root)
        var cen = List[Float64](length=3, fill=0.0)
        for i in range(nv):
            for c in range(3):
                cen[c] += Float64(v[i * 3 + c])
        for c in range(3):
            cen[c] = ((cen[c] / Float64(nv) - pc.center[c]) * pc.scale - pc.pc_center[c]) / pc.pc_scale
        var ms = self.skeleton_from_cloud(pc, cen)
        var sk = to_ur_skeleton(ms, pc)
        self.last_ms = ms^
        return sk^

    def rig(mut self, v: List[Float32], f: List[Int]) raises -> UrRig:
        """Skeleton + dense skin weights [V, J] (UniRig skin network) for a Z-up triangle mesh."""
        if not self.skinner:
            raise Error("MagicArticulate: no skin network loaded (pass the UniRig graph directory)")
        var sk = self.skeleton(v, f)
        var J = sk.num_bones()
        var joints = List[Float32](length=J * 3, fill=0.0)
        for j in range(J):
            joints[j * 3] = sk.joints[j].x
            joints[j * 3 + 1] = sk.joints[j].y
            joints[j * 3 + 2] = sk.joints[j].z
        var t0 = perf_counter_ns()
        var w = self.skinner.value().skin(v, f, joints, sk.parents, self.cfg.seed)
        self._log("skin (UniRig)", t0)
        return UrRig(sk^, w^)


def rig_native_model_ma(mut m: MagicArticulate, mut src: NativeModel) raises -> NativeModel:
    """`rig_native_model` with a MagicArticulate / Puppeteer skeleton: every mesh instance baked
    to world space, one joint tree and one skin (four influences per vertex)."""
    var sc = bake_native_model(src)
    var r = m.rig(sc.v, sc.f)
    return skin_baked_model(sc, r.skeleton, r.weights)
