"""NVIDIA Audio2Face-3D in pure Mojo (`flint.ml.audio2face`): speech -> ARKit-52 blend shapes.

Runs the v2.3 / v2.3.1 regression networks (Mark, Claire, James;
https://huggingface.co/nvidia/Audio2Face-3D-v2.3-Mark, NVIDIA Open Model License, commercial use
allowed) straight from the published `network.onnx` + `.npz` / `.json` files on `OnnxModel`
(MAX GPU when present), and ports the SDK's post-processing (https://github.com/NVIDIA/Audio2Face-3D-SDK,
MIT): audio windows, PCA reconstruction, the skin / tongue / teeth / eye animators and the
bounded least-squares blend-shape solve.

Pipeline per output frame i (default 30 fps):
  1. Audio front end: 16 kHz mono (`moonshine_audio` WAV decoder + windowed-sinc `Resampler`);
     window = samples [floor(i * 16000 / fps) - 4160, +8320) zero padded, times `input_strength`.
  2. Network: window + emotion [16 implicit (from `source_shot` / `source_frame` of
     `implicit_emo_db.npz`) | 10 explicit (neutral or caller-set)] -> 301 values: 272 skin PCA
     coefficients, 10 tongue PCA coefficients, 5 jaw points (15), 2 eyes x (x, y) degrees (4).
     Audio2Emotion is not used (its licence restricts it); emotions are caller inputs.
  3. Skin: delta = skin_strength * PCA(c) + eye / lip offset poses; two-pole smoothing with
     separate upper / lower face rates blended by a sigmoid face mask over neutral height;
     tongue: neutral + strength * PCA(c) + height / depth offsets.
  4. Blend-shape solve (skin 52 ARKit poses on the frontal mask, tongue 16): BVLS in [0, 1] on
     D^T D + L1 / L2 / temporal / symmetry terms, cancel pairs, multipliers and offsets.
  5. Jaw: rigid fit of the 5 jaw points to neutral -> column-major 4x4. Eyes: offsets + strength
     * network + saccade table (live time at 30 fps).

Everything before the solve is linear and the smoothing uses one rate for every element, so the
skin is never rebuilt: `prepare()` folds PCA basis, strengths, face mask and solver basis into
two [k, 272] projections (k = active poses) and per-frame work is the network plus a small
BVLS over the active poses (43 for Mark). Results match the full-geometry reference (`tools/audio2face/reference.py`).

  var a2f = Audio2Face.load(model_dir)            # GPU when available
  var clip = a2f.process_wav("line.wav")          # A2fClip: frames x 52 ARKit weights (+ tongue, jaw, eyes)
  apply_clip_to_model(model, clip, t)             # drive NativeModel morph targets by name

Streaming: `stream_reset()`, then `stream_push(samples16)` returns the frames whose windows are
complete; `stream_finish()` flushes the tail.
"""

from std.math import sqrt, exp, abs
from std.time import perf_counter_ns
from flint.platform.os import read_bytes, write_bytes
from flint.assets.json import parse_json, JsonDocument
from flint.assets.model import NativeModel
from flint.assets.glb_writer import append_u32_le, append_f32_le, pad4, fmt_f32
from flint.assets.json_write import json_quote
from flint.ml.onnx import OnnxModel
from flint.ml.npz import Npz
from flint.ml.tensor import Tensor, fptr_at
from flint.ml.ops import matmul_2d
from flint.ml.moonshine_audio import read_wav, to_model_rate

comptime A2F_RATE: Int = 16000


# ----------------------------------------------------------------------------- parameters

struct A2fParams(Copyable, Movable):
    """Animator parameters (`model_config.json` "config"); change them, then call `prepare()`."""
    var input_strength: Float32
    var upper_face_smoothing: Float32
    var lower_face_smoothing: Float32
    var upper_face_strength: Float32
    var lower_face_strength: Float32
    var face_mask_level: Float32
    var face_mask_softness: Float32
    var skin_strength: Float32
    var blink_strength: Float32
    var blink_offset: Float32
    var eyelid_open_offset: Float32
    var lip_open_offset: Float32
    var tongue_strength: Float32
    var tongue_height_offset: Float32
    var tongue_depth_offset: Float32
    var lower_teeth_strength: Float32
    var lower_teeth_height_offset: Float32
    var lower_teeth_depth_offset: Float32
    var eyeballs_strength: Float32
    var saccade_strength: Float32
    var right_eye_rot_x_offset: Float32
    var right_eye_rot_y_offset: Float32
    var left_eye_rot_x_offset: Float32
    var left_eye_rot_y_offset: Float32
    var eye_saccade_seed: Float32
    var source_shot: String
    var source_frame: Int

    def __init__(out self):
        self.input_strength = 1
        self.upper_face_smoothing = 0
        self.lower_face_smoothing = 0
        self.upper_face_strength = 1
        self.lower_face_strength = 1
        self.face_mask_level = 0.6
        self.face_mask_softness = 0.0085
        self.skin_strength = 1
        self.blink_strength = 1
        self.blink_offset = 0
        self.eyelid_open_offset = 0
        self.lip_open_offset = 0
        self.tongue_strength = 1
        self.tongue_height_offset = 0
        self.tongue_depth_offset = 0
        self.lower_teeth_strength = 1
        self.lower_teeth_height_offset = 0
        self.lower_teeth_depth_offset = 0
        self.eyeballs_strength = 1
        self.saccade_strength = 0
        self.right_eye_rot_x_offset = 0
        self.right_eye_rot_y_offset = 0
        self.left_eye_rot_x_offset = 0
        self.left_eye_rot_y_offset = 0
        self.eye_saccade_seed = 0
        self.source_shot = String("g2a_neutral")
        self.source_frame = 0


def _num(doc: JsonDocument, node: Int, key: String, fallback: Float32) -> Float32:
    var n = doc.object_get(node, key)
    if n < 0:
        return fallback
    return Float32(doc.number(n, Float64(fallback)))


def _load_json(path: String) raises -> JsonDocument:
    var doc = parse_json(read_bytes(path))
    if not doc.valid:
        raise Error("audio2face: cannot read " + path)
    return doc^


def read_params(path: String) raises -> A2fParams:
    var doc = _load_json(path)
    var c = doc.object_get(doc.root, "config")
    if c < 0:
        raise Error("audio2face: no \"config\" in " + path)
    var p = A2fParams()
    p.input_strength = _num(doc, c, "input_strength", 1)
    p.upper_face_smoothing = _num(doc, c, "upper_face_smoothing", 0)
    p.lower_face_smoothing = _num(doc, c, "lower_face_smoothing", 0)
    p.upper_face_strength = _num(doc, c, "upper_face_strength", 1)
    p.lower_face_strength = _num(doc, c, "lower_face_strength", 1)
    p.face_mask_level = _num(doc, c, "face_mask_level", 0.6)
    p.face_mask_softness = _num(doc, c, "face_mask_softness", 0.0085)
    p.skin_strength = _num(doc, c, "skin_strength", 1)
    p.blink_strength = _num(doc, c, "blink_strength", 1)
    p.eyelid_open_offset = _num(doc, c, "eyelid_open_offset", 0)
    p.lip_open_offset = _num(doc, c, "lip_open_offset", 0)
    p.tongue_strength = _num(doc, c, "tongue_strength", 1)
    p.tongue_height_offset = _num(doc, c, "tongue_height_offset", 0)
    p.tongue_depth_offset = _num(doc, c, "tongue_depth_offset", 0)
    p.lower_teeth_strength = _num(doc, c, "lower_teeth_strength", 1)
    p.lower_teeth_height_offset = _num(doc, c, "lower_teeth_height_offset", 0)
    p.lower_teeth_depth_offset = _num(doc, c, "lower_teeth_depth_offset", 0)
    p.eyeballs_strength = _num(doc, c, "eyeballs_strength", 1)
    p.saccade_strength = _num(doc, c, "saccade_strength", 0)
    p.right_eye_rot_x_offset = _num(doc, c, "right_eye_rot_x_offset", 0)
    p.right_eye_rot_y_offset = _num(doc, c, "right_eye_rot_y_offset", 0)
    p.left_eye_rot_x_offset = _num(doc, c, "left_eye_rot_x_offset", 0)
    p.left_eye_rot_y_offset = _num(doc, c, "left_eye_rot_y_offset", 0)
    p.eye_saccade_seed = _num(doc, c, "eye_saccade_seed", 0)
    p.source_shot = doc.text(doc.object_get(c, "source_shot"), "g2a_neutral")
    p.source_frame = Int(_num(doc, c, "source_frame", 0))
    return p^


# ----------------------------------------------------------------------------- audio front end

def a2f_frame_count(samples: Int, fps: Int) -> Int:
    """Frames of a closed track: every frame whose target sample lies inside the audio."""
    if samples <= 0 or fps <= 0:
        return 0
    # target(i) = floor(i * 16000 / fps) < samples  <=>  i < ceil(samples * fps / 16000)
    return (samples * fps + A2F_RATE - 1) // A2F_RATE


def a2f_target_sample(frame: Int, fps: Int) -> Int:
    return frame * A2F_RATE // fps


def a2f_window(audio: List[Float32], frame: Int, fps: Int, length: Int, offset: Int, strength: Float32,
               mut out: List[Float32], at: Int, base: Int = 0):
    """Window `frame` (start = target - offset) times `strength` into out[at, at + length),
    zero outside the audio (the SDK's AudioAccumulator::Read); audio[0] is sample `base`."""
    var start = a2f_target_sample(frame, fps) - offset - base
    var n = len(audio)
    for j in range(length):
        var s = start + j
        out[at + j] = audio[s] * strength if s >= 0 and s < n else Float32(0)


# ----------------------------------------------------------------------------- small dense solvers

def solve_spd(a: List[Float64], b: List[Float64], n: Int) -> List[Float64]:
    """x with a x = b (a n x n row-major): Gaussian elimination with partial pivoting."""
    var m = a.copy()
    var x = b.copy()
    for col in range(n):
        var piv = col
        var best = abs(m[col * n + col])
        for r in range(col + 1, n):
            if abs(m[r * n + col]) > best:
                best = abs(m[r * n + col])
                piv = r
        if best == 0:
            continue
        if piv != col:
            for c in range(n):
                var t = m[col * n + c]
                m[col * n + c] = m[piv * n + c]
                m[piv * n + c] = t
            var tb = x[col]
            x[col] = x[piv]
            x[piv] = tb
        var inv = 1.0 / m[col * n + col]
        for r in range(col + 1, n):
            var f = m[r * n + col] * inv
            if f != 0:
                for c in range(col, n):
                    m[r * n + c] -= f * m[col * n + c]
                x[r] -= f * x[col]
    var i = n - 1
    while i >= 0:
        var s = x[i]
        for c in range(i + 1, n):
            s -= m[i * n + c] * x[c]
        x[i] = s / m[i * n + i] if m[i * n + i] != 0 else 0.0
        i -= 1
    return x^


def _lstsq_cols(a: List[Float32], k: Int, cols: List[Int], b: List[Float32]) -> List[Float32]:
    """argmin |A[:, cols] y - b| for the k x k matrix A (row-major), in float64 normal equations."""
    var f = len(cols)
    var n = List[Float64](length=f * f, fill=0.0)
    var r = List[Float64](length=f, fill=0.0)
    for i in range(f):
        for j in range(i, f):
            var s = 0.0
            for row in range(k):
                s += Float64(a[row * k + cols[i]]) * Float64(a[row * k + cols[j]])
            n[i * f + j] = s
            n[j * f + i] = s
        var t = 0.0
        for row in range(k):
            t += Float64(a[row * k + cols[i]]) * Float64(b[row])
        r[i] = t
    var y = solve_spd(n, r, f)
    var out = List[Float32](length=f, fill=0.0)
    for i in range(f):
        out[i] = Float32(y[i])
    return out^


def _constrained_min(mut x: List[Float32], a: List[Float32], b: List[Float32], on: List[Int], k: Int):
    var free = List[Int]()
    for i in range(k):
        if on[i] == 0:
            free.append(i)
    if len(free) == 0:
        return
    var bf = b.copy()
    if len(free) < k:
        # b - A x0, x0 = x with its free components zeroed
        for row in range(k):
            var s: Float32 = 0
            for c in range(k):
                if on[c] != 0:
                    s += a[row * k + c] * x[c]
            bf[row] = b[row] - s
    var y = _lstsq_cols(a, k, free, bf)
    for i in range(len(free)):
        x[free[i]] = y[i]


def _cost_grad(x: List[Float32], a: List[Float32], b: List[Float32], k: Int, mut g: List[Float32]) -> Float32:
    var r = List[Float32](length=k, fill=0.0)
    var cost: Float32 = 0
    for row in range(k):
        var s: Float32 = 0
        for c in range(k):
            s += a[row * k + c] * x[c]
        r[row] = s - b[row]
        cost += r[row] * r[row]
    for c in range(k):
        var s: Float32 = 0
        for row in range(k):
            s += a[row * k + c] * r[row]
        g[c] = s
    return 0.5 * cost


def bvls(a: List[Float32], b: List[Float32], lo: List[Float32], hi: List[Float32], k: Int,
         tol: Float32 = 1e-10) -> List[Float32]:
    """Bounded-variable least squares min |A x - b|, lo <= x <= hi (A k x k row-major), following
    the SDK's `bvls.cpp` active-set iteration step for step (inner solves in float64)."""
    var x = List[Float32](length=k, fill=0.0)
    var on = List[Int](length=k, fill=0)
    while True:  # initial feasible point
        _constrained_min(x, a, b, on, k)
        var found = 0
        for i in range(k):
            if on[i] != 0:
                continue
            if x[i] < lo[i]:
                x[i] = lo[i]
                on[i] = -1
                found += 1
            elif x[i] > hi[i]:
                x[i] = hi[i]
                on[i] = 1
                found += 1
        if found == 0:
            break
    var g = List[Float32](length=k, fill=0.0)
    var cost = _cost_grad(x, a, b, k, g)
    for i in range(k):
        on[i] = -1 if x[i] == lo[i] else (1 if x[i] == hi[i] else 0)
    var opt = _kkt(g, on, k)
    var prev = opt
    for _ in range(k):
        if opt < tol:
            break
        _ = _cost_grad(x, a, b, k, g)
        var best = -1
        var bv: Float32 = 0
        for i in range(k):
            if x[i] == lo[i]:
                if g[i] < 0 and -g[i] > bv:
                    bv = -g[i]
                    best = i
            elif x[i] == hi[i]:
                if g[i] > 0 and g[i] > bv:
                    bv = g[i]
                    best = i
        if best == -1:
            break
        on[best] = 0
        while True:
            var s = x.copy()
            _constrained_min(s, a, b, on, k)
            var alpha: Float32 = 1
            var idx = -1
            var bval: Float32 = 0
            for i in range(k):
                if x[i] == s[i]:
                    continue
                if s[i] < lo[i]:
                    var at = (x[i] - lo[i]) / (x[i] - s[i])
                    if at < alpha:
                        idx = i
                        bval = lo[i]
                        alpha = at
                elif s[i] > hi[i]:
                    var at = (hi[i] - x[i]) / (s[i] - x[i])
                    if at < alpha:
                        idx = i
                        bval = hi[i]
                        alpha = at
            for i in range(k):
                x[i] += alpha * (s[i] - x[i])
            if idx != -1:
                x[idx] = bval
            if alpha == 1:
                break
            for i in range(k):
                on[i] = -1 if x[i] == lo[i] else (1 if x[i] == hi[i] else 0)
        var cost_new = _cost_grad(x, a, b, k, g)
        if cost_new > cost:
            break
        cost = cost_new
        opt = _kkt(g, on, k)
        if abs(prev - opt) < 1e-6:
            break
        prev = opt
    return x^


def _kkt(g: List[Float32], on: List[Int], k: Int) -> Float32:
    var m: Float32 = -3.4e38
    for i in range(k):
        var v = abs(g[i]) if on[i] == 0 else g[i] * Float32(on[i])
        if v > m:
            m = v
    return m


# ----------------------------------------------------------------------------- blend-shape solver

struct BsSolver(Movable):
    """One blend-shape solve (skin or tongue): `bs_*.npz` poses + `bs_*_config.json` settings."""
    var names: List[String]      # every pose but "neutral", in file order
    var active: List[Int]        # indices into names of the solved poses
    var mult: List[Float32]
    var offs: List[Float32]
    var k: Int
    var a: List[Float32]         # [k, k] system matrix
    var temporal: Float32        # strengthTemporalSmoothing * scale (the SDK's b term)
    var scale: Float32
    var cancel: List[Int]        # pairs (i, j) of active indices, flattened
    var prev: List[Float32]
    var l1: Float32
    var l2: Float32
    var t_reg: Float32
    var sym: Float32
    var bb_size: Float32
    var sym_pairs: List[Int]

    def __init__(out self):
        self.names = List[String]()
        self.active = List[Int]()
        self.mult = List[Float32]()
        self.offs = List[Float32]()
        self.k = 0
        self.a = List[Float32]()
        self.temporal = 0
        self.scale = 1
        self.cancel = List[Int]()
        self.prev = List[Float32]()
        self.l1 = 0
        self.l2 = 0
        self.t_reg = 0
        self.sym = 0
        self.bb_size = 54.7
        self.sym_pairs = List[Int]()

    def read_config(mut self, path: String, poses: Int) raises:
        var doc = _load_json(path)
        var c = doc.object_get(doc.root, "blendshape_params")
        if c < 0:
            raise Error("audio2face: no \"blendshape_params\" in " + path)
        self.l1 = _num(doc, c, "strengthL1regularization", 1)
        self.l2 = _num(doc, c, "strengthL2regularization", 3.5)
        self.t_reg = _num(doc, c, "strengthTemporalSmoothing", 0)
        self.sym = _num(doc, c, "strengthSymmetry", 100)
        self.bb_size = _num(doc, c, "templateBBSize", 54.7)
        var act = doc.object_get(c, "bsSolveActivePoses")
        var can = doc.object_get(c, "bsSolveCancelPoses")
        var sy = doc.object_get(c, "bsSolveSymmetryPoses")
        var mu = doc.object_get(c, "bsWeightMultipliers")
        var of = doc.object_get(c, "bsWeightOffsets")
        if doc.array_size(act) != poses or doc.array_size(mu) != poses or doc.array_size(of) != poses:
            raise Error("audio2face: pose count mismatch in " + path)
        self.active.clear()
        self.mult.clear()
        self.offs.clear()
        var can_ids = List[Int]()
        var sym_ids = List[Int]()
        for i in range(poses):
            if doc.integer(doc.at(act, i), 0) != 0:
                self.active.append(i)
                can_ids.append(doc.integer(doc.at(can, i), -1))
                sym_ids.append(doc.integer(doc.at(sy, i), -1))
            self.mult.append(Float32(doc.number(doc.at(mu, i), 1)))
            self.offs.append(Float32(doc.number(doc.at(of, i), 0)))
        self.k = len(self.active)
        self.cancel = _pairs(can_ids)
        self.sym_pairs = _pairs(sym_ids)

    def build(mut self, ata: List[Float64], neutral_bbox_diag: Float64):
        """System matrix from D^T D over the solved vertices (BlendshapeSolverBase::Prepare)."""
        var k = self.k
        var sf = (neutral_bbox_diag / Float64(self.bb_size)) ** 2
        self.scale = Float32(sf)
        var s32 = Float32(sf)
        self.a = List[Float32](length=k * k, fill=0.0)
        for i in range(k):
            for j in range(k):
                var v = Float32(ata[i * k + j])
                v += self.l1 * self.l1 * 0.25 * s32
                if i == j:
                    v += self.l2 * 10 * s32 + self.t_reg * 100 * s32
                self.a[i * k + j] = v
        # symmetry: + sym * 10 * scale * S^T S, S rows (e_a - e_b)
        var q = 0
        while q + 1 < len(self.sym_pairs):
            var pa = self.sym_pairs[q]
            var pb = self.sym_pairs[q + 1]
            var w = self.sym * 10 * s32
            self.a[pa * k + pa] += w
            self.a[pb * k + pb] += w
            self.a[pa * k + pb] -= w
            self.a[pb * k + pa] -= w
            q += 2
        self.temporal = self.t_reg * s32
        self.reset()

    def reset(mut self):
        self.prev = List[Float32](length=self.k, fill=0.0)

    def solve(mut self, atb: List[Float32]) -> List[Float32]:
        """Weights for every pose (inactive 0) after multipliers / offsets; `atb` = D^T (pose - neutral)."""
        var k = self.k
        var b = List[Float32](length=k, fill=0.0)
        for i in range(k):
            b[i] = atb[i] + self.temporal * self.prev[i]
        var lo = List[Float32](length=k, fill=0.0)
        var hi = List[Float32](length=k, fill=1.0)
        var x = bvls(self.a, b, lo, hi, k)
        if len(self.cancel) > 0:
            var q = 0
            while q + 1 < len(self.cancel):
                var i = self.cancel[q]
                var j = self.cancel[q + 1]
                hi[j if x[i] >= x[j] else i] = 1e-10
                q += 2
            x = bvls(self.a, b, lo, hi, k)
        self.prev = x.copy()
        var w = List[Float32](length=len(self.names), fill=0.0)
        for i in range(k):
            w[self.active[i]] = x[i]
        for i in range(len(w)):
            w[i] = w[i] * self.mult[i] + self.offs[i]
        return w^


def _pairs(ids: List[Int]) -> List[Int]:
    """Index pairs of equal ids (other than -1) that occur exactly twice, in first-seen order."""
    var out = List[Int]()
    var seen = List[Int]()
    for i in range(len(ids)):
        var v = ids[i]
        if v == -1:
            continue
        var dup = False
        for s in seen:
            if s == v:
                dup = True
        if dup:
            continue
        seen.append(v)
        var members = List[Int]()
        for j in range(len(ids)):
            if ids[j] == v:
                members.append(j)
        if len(members) == 2:
            out.append(members[0])
            out.append(members[1])
    return out^


# ----------------------------------------------------------------------------- jaw rigid fit

def _jacobi_max_eigvec(m_in: List[Float64]) -> List[Float64]:
    """Unit eigenvector of the largest eigenvalue of a symmetric 4x4 (row-major), cyclic Jacobi."""
    var a = m_in.copy()
    var v = List[Float64](length=16, fill=0.0)
    for i in range(4):
        v[i * 4 + i] = 1
    for _ in range(64):
        var off = 0.0
        for p in range(4):
            for q in range(p + 1, 4):
                off += a[p * 4 + q] * a[p * 4 + q]
        if off < 1e-30:
            break
        for p in range(4):
            for q in range(p + 1, 4):
                var apq = a[p * 4 + q]
                if abs(apq) < 1e-300:
                    continue
                var theta = (a[q * 4 + q] - a[p * 4 + p]) / (2 * apq)
                var t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + sqrt(theta * theta + 1))
                var c = 1 / sqrt(t * t + 1)
                var s = t * c
                for r in range(4):
                    var arp = a[r * 4 + p]
                    var arq = a[r * 4 + q]
                    a[r * 4 + p] = c * arp - s * arq
                    a[r * 4 + q] = s * arp + c * arq
                for r in range(4):
                    var apr = a[p * 4 + r]
                    var aqr = a[q * 4 + r]
                    a[p * 4 + r] = c * apr - s * aqr
                    a[q * 4 + r] = s * apr + c * aqr
                for r in range(4):
                    var vrp = v[r * 4 + p]
                    var vrq = v[r * 4 + q]
                    v[r * 4 + p] = c * vrp - s * vrq
                    v[r * 4 + q] = s * vrp + c * vrq
    var best = 0
    for i in range(1, 4):
        if a[i * 4 + i] > a[best * 4 + best]:
            best = i
    var out = List[Float64](length=4, fill=0.0)
    for r in range(4):
        out[r] = v[r * 4 + best]
    return out^


def rigid_fit(source: List[Float32], target: List[Float32], points: Int) -> List[Float32]:
    """Column-major 4x4 [R | t] minimising sum |R s + t - t_i|^2 (proper rotation; Horn's
    quaternion method, equal to the SDK's SVD Kabsch fit for non-degenerate point sets)."""
    var sm = List[Float64](length=3, fill=0.0)
    var tm = List[Float64](length=3, fill=0.0)
    for i in range(points):
        for c in range(3):
            sm[c] += Float64(source[i * 3 + c])
            tm[c] += Float64(target[i * 3 + c])
    for c in range(3):
        sm[c] /= Float64(points)
        tm[c] /= Float64(points)
    var h = List[Float64](length=9, fill=0.0)  # h[r, c] = sum s_r t_c
    for i in range(points):
        for r in range(3):
            for c in range(3):
                h[r * 3 + c] += (Float64(source[i * 3 + r]) - sm[r]) * (Float64(target[i * 3 + c]) - tm[c])
    var sxx = h[0]
    var sxy = h[1]
    var sxz = h[2]
    var syx = h[3]
    var syy = h[4]
    var syz = h[5]
    var szx = h[6]
    var szy = h[7]
    var szz = h[8]
    var n: List[Float64] = [
        sxx + syy + szz, syz - szy, szx - sxz, sxy - syx,
        syz - szy, sxx - syy - szz, sxy + syx, szx + sxz,
        szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy,
        sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz,
    ]
    var q = _jacobi_max_eigvec(n)
    var w = q[0]
    var x = q[1]
    var y = q[2]
    var z = q[3]
    var r = List[Float64](length=9, fill=0.0)  # row-major
    r[0] = w * w + x * x - y * y - z * z
    r[1] = 2 * (x * y - w * z)
    r[2] = 2 * (x * z + w * y)
    r[3] = 2 * (x * y + w * z)
    r[4] = w * w - x * x + y * y - z * z
    r[5] = 2 * (y * z - w * x)
    r[6] = 2 * (x * z - w * y)
    r[7] = 2 * (y * z + w * x)
    r[8] = w * w - x * x - y * y + z * z
    var out = List[Float32](length=16, fill=0.0)
    for row in range(3):
        for col in range(3):
            out[col * 4 + row] = Float32(r[row * 3 + col])
        out[12 + row] = Float32(tm[row] - (r[row * 3 + 0] * sm[0] + r[row * 3 + 1] * sm[1] + r[row * 3 + 2] * sm[2]))
    out[15] = 1
    return out^


# ----------------------------------------------------------------------------- results

struct A2fClip(Movable):
    """Per-frame Audio2Face output at `fps`."""
    var fps: Int
    var frames: Int
    var skin_names: List[String]     # ARKit-52 (bs_skin.npz poseNames order)
    var tongue_names: List[String]   # 16 tongue poses
    var skin: List[Float32]          # [frames, len(skin_names)]
    var tongue: List[Float32]        # [frames, len(tongue_names)]
    var jaw: List[Float32]           # [frames, 16] column-major lower-teeth transform
    var eyes: List[Float32]          # [frames, 6]: right (x, y, 0), left (x, y, 0), degrees
    var net: List[Float32]           # [frames, 301] raw network output (kept for checks)

    def __init__(out self):
        self.fps = 30
        self.frames = 0
        self.skin_names = List[String]()
        self.tongue_names = List[String]()
        self.skin = List[Float32]()
        self.tongue = List[Float32]()
        self.jaw = List[Float32]()
        self.eyes = List[Float32]()
        self.net = List[Float32]()

    def skin_count(self) -> Int:
        return len(self.skin_names)

    def seconds(self) -> Float64:
        return Float64(self.frames) / Float64(self.fps) if self.fps > 0 else 0.0

    def skin_weight(self, frame: Int, pose: Int) -> Float32:
        return self.skin[frame * len(self.skin_names) + pose]

    def pose_index(self, name: String) -> Int:
        for i in range(len(self.skin_names)):
            if self.skin_names[i] == name:
                return i
        return -1

    def sample_skin(self, t: Float64) -> List[Float32]:
        """ARKit weights at time t seconds (linear between frames, clamped)."""
        var n = len(self.skin_names)
        var out = List[Float32](length=n, fill=0.0)
        if self.frames == 0:
            return out^
        var f = t * Float64(self.fps)
        if f <= 0:
            f = 0
        var i0 = Int(f)
        if i0 >= self.frames - 1:
            for j in range(n):
                out[j] = self.skin[(self.frames - 1) * n + j]
            return out^
        var u = Float32(f - Float64(i0))
        for j in range(n):
            out[j] = self.skin[i0 * n + j] * (1 - u) + self.skin[(i0 + 1) * n + j] * u
        return out^

    def append(mut self, other: A2fClip):
        if len(self.skin_names) == 0:
            self.skin_names = other.skin_names.copy()
            self.tongue_names = other.tongue_names.copy()
            self.fps = other.fps
        self.frames += other.frames
        for v in other.skin:
            self.skin.append(v)
        for v in other.tongue:
            self.tongue.append(v)
        for v in other.jaw:
            self.jaw.append(v)
        for v in other.eyes:
            self.eyes.append(v)
        for v in other.net:
            self.net.append(v)

    def csv(self) -> String:
        """time, the 52 skin weights, the 16 tongue weights, eyes (4 angles); one row per frame."""
        var s = String("time")
        for nm in self.skin_names:
            s += "," + nm
        for nm in self.tongue_names:
            s += "," + nm
        s += ",eyeRightX,eyeRightY,eyeLeftX,eyeLeftY\n"
        var ns = len(self.skin_names)
        var nt = len(self.tongue_names)
        for f in range(self.frames):
            s += String(Float64(f) / Float64(self.fps))
            for j in range(ns):
                s += "," + String(self.skin[f * ns + j])
            for j in range(nt):
                s += "," + String(self.tongue[f * nt + j])
            s += "," + String(self.eyes[f * 6 + 0]) + "," + String(self.eyes[f * 6 + 1])
            s += "," + String(self.eyes[f * 6 + 3]) + "," + String(self.eyes[f * 6 + 4]) + "\n"
        return s^

    def write_csv(self, path: String) -> Bool:
        var b = List[UInt8](self.csv().as_bytes())
        return write_bytes(path, b)


# ----------------------------------------------------------------------------- morph targets

def _lower(s: String) -> String:
    var out = String()
    for i in range(s.byte_length()):
        var c = s[byte=i]
        var o = ord(c)
        if o >= 65 and o <= 90:
            out += chr(o + 32)
        else:
            out += c
    return out^


def _is_alnum(c: String) -> Bool:
    var o = ord(c)
    return (o >= 48 and o <= 57) or (o >= 65 and o <= 90) or (o >= 97 and o <= 122)


def morph_name_matches(morph: String, pose: String) -> Bool:
    """`morph` names ARKit pose `pose`: equal ignoring case, or ends with it after a separator
    (`blendShape1.jawOpen`, `ARKit_jawOpen`, `Face:jawOpen`)."""
    var m = _lower(morph)
    var p = _lower(pose)
    if m == p:
        return True
    var ml = m.byte_length()
    var pl = p.byte_length()
    if ml <= pl or not m.endswith(p):
        return False
    return not _is_alnum(String(m[byte=ml - pl - 1]))


def morph_map(model_morphs: List[String], poses: List[String]) -> List[Int]:
    """For each model morph target, the pose index it follows, or -1."""
    var out = List[Int](length=len(model_morphs), fill=-1)
    for i in range(len(model_morphs)):
        for j in range(len(poses)):
            if morph_name_matches(model_morphs[i], poses[j]):
                out[i] = j
                break
    return out^


def apply_clip_to_model(mut model: NativeModel, clip: A2fClip, t: Float64) -> Int:
    """Set every morph target of `model` named after an ARKit pose to the clip's weight at t
    seconds; returns how many targets were driven."""
    var w = clip.sample_skin(t)
    var map = morph_map(model.morph_names, clip.skin_names)
    var driven = 0
    for i in range(len(map)):
        if map[i] >= 0:
            model.set_morph_weight(i, w[map[i]])
            driven += 1
    return driven


# ----------------------------------------------------------------------------- the runtime

struct Audio2Face(Movable):
    var dir: String
    var net: OnnxModel
    var on_gpu: Bool
    var model_data: Npz        # kept mapped: the PCA bases are borrowed from it
    var bs_skin: Npz
    var bs_tongue: Npz
    var params: A2fParams
    var fps: Int
    var batch: Int             # frames per network run
    var buffer_len: Int
    var buffer_ofs: Int
    var n_skin: Int            # PCA coefficients
    var n_tongue: Int
    var n_jaw: Int
    var n_eyes: Int
    var n_out: Int
    var implicit: List[Float32]
    var emotion: List[Float32]     # explicit emotions (network_info "explicit_emotions"), caller-set
    var emotion_names: List[String]
    var skin: BsSolver
    var tongue: BsSolver
    # folded skin: atb = c0 + smooth_u(Qu c + qu0) + smooth_l(Ql c + ql0); Q* are [n_skin, k]
    var qu: List[Float32]
    var ql: List[Float32]
    var qu0: List[Float32]
    var ql0: List[Float32]
    var c0: List[Float32]
    var alpha_u: Float32
    var alpha_l: Float32
    # folded tongue: atb = ct0 + Qt c_t, Qt [n_tongue, kt]
    var qt: List[Float32]
    var ct0: List[Float32]
    var jaw0: List[Float32]
    var saccade: List[Float32]
    # sequential state
    var su1: List[Float32]
    var su2: List[Float32]
    var sl1: List[Float32]
    var sl2: List[Float32]
    var started: Bool
    var live: Float32
    # streaming
    var sbuf: List[Float32]
    var sbase: Int             # absolute 16 kHz index of sbuf[0]
    var snext: Int
    # timings (ns) of the last process()
    var t_net: Int
    var t_post: Int

    def __init__(out self):
        self.dir = String()
        self.net = OnnxModel()
        self.on_gpu = False
        self.model_data = Npz()
        self.bs_skin = Npz()
        self.bs_tongue = Npz()
        self.params = A2fParams()
        self.fps = 30
        self.batch = 32
        self.buffer_len = 8320
        self.buffer_ofs = 4160
        self.n_skin = 0
        self.n_tongue = 0
        self.n_jaw = 0
        self.n_eyes = 0
        self.n_out = 0
        self.implicit = List[Float32]()
        self.emotion = List[Float32]()
        self.emotion_names = List[String]()
        self.skin = BsSolver()
        self.tongue = BsSolver()
        self.qu = List[Float32]()
        self.ql = List[Float32]()
        self.qu0 = List[Float32]()
        self.ql0 = List[Float32]()
        self.c0 = List[Float32]()
        self.alpha_u = -1
        self.alpha_l = -1
        self.qt = List[Float32]()
        self.ct0 = List[Float32]()
        self.jaw0 = List[Float32]()
        self.saccade = List[Float32]()
        self.su1 = List[Float32]()
        self.su2 = List[Float32]()
        self.sl1 = List[Float32]()
        self.sl2 = List[Float32]()
        self.started = False
        self.live = 0
        self.sbuf = List[Float32]()
        self.sbase = 0
        self.snext = 0
        self.t_net = 0
        self.t_post = 0

    @staticmethod
    def load(dir: String, gpu: Bool = True, fps: Int = 30, batch: Int = 32) raises -> Audio2Face:
        """A v2.3 / v2.3.1 regression model folder (network.onnx, network_info.json,
        model_config.json, model_data.npz, implicit_emo_db.npz, bs_skin/bs_tongue .npz + config)."""
        var a = Audio2Face()
        a.dir = dir
        a.fps = fps
        a.batch = max(1, batch)
        var info = _load_json(dir + "/network_info.json")
        var id = info.object_get(info.root, "id")
        if info.text(info.object_get(id, "type")) != "regression":
            raise Error("audio2face: " + dir + " is not a regression model (v3.0 diffusion is not supported)")
        var p = info.object_get(info.root, "params")
        a.n_skin = info.integer(info.object_get(p, "num_shapes_skin"), 272)
        a.n_tongue = info.integer(info.object_get(p, "num_shapes_tongue"), 10)
        a.n_jaw = info.integer(info.object_get(p, "result_jaw_size"), 15)
        a.n_eyes = info.integer(info.object_get(p, "result_eyes_size"), 4)
        a.n_out = a.n_skin + a.n_tongue + a.n_jaw + a.n_eyes
        var emo = info.object_get(p, "explicit_emotions")
        for i in range(info.array_size(emo)):
            a.emotion_names.append(info.text(info.at(emo, i)))
        a.emotion = List[Float32](length=len(a.emotion_names), fill=0.0)
        var ap = info.object_get(info.root, "audio_params")
        a.buffer_len = info.integer(info.object_get(ap, "buffer_len"), 8320)
        a.buffer_ofs = info.integer(info.object_get(ap, "buffer_ofs"), 4160)
        if info.integer(info.object_get(ap, "samplerate"), A2F_RATE) != A2F_RATE:
            raise Error("audio2face: only 16 kHz models are supported")
        a.params = read_params(dir + "/model_config.json")
        # implicit emotion vector
        var edb = Npz.open(dir + "/implicit_emo_db.npz")
        var names = edb.strings("emo_spec_names")
        var starts = edb.ints("emo_spec_start")
        var sizes = edb.ints("emo_spec_size")
        var db = edb.floats("emo_db")
        var dim = db.shape[1]
        var shot = -1
        for i in range(len(names)):
            if names[i] == a.params.source_shot:
                shot = i
        if shot < 0 or a.params.source_frame < 0 or a.params.source_frame >= sizes[shot]:
            raise Error("audio2face: emotion shot " + a.params.source_shot + " not in implicit_emo_db.npz")
        var row = starts[shot] + a.params.source_frame
        for j in range(dim):
            a.implicit.append(db.fget(row * dim + j))
        _ = edb.has("")  # keep the mapping alive until the row is copied
        a.model_data = Npz.open(dir + "/model_data.npz")
        a.bs_skin = Npz.open(dir + "/bs_skin.npz")
        a.bs_tongue = Npz.open(dir + "/bs_tongue.npz")
        a.skin.names = _pose_names(a.bs_skin)
        a.tongue.names = _pose_names(a.bs_tongue)
        a.skin.read_config(dir + "/bs_skin_config.json", len(a.skin.names))
        a.tongue.read_config(dir + "/bs_tongue_config.json", len(a.tongue.names))
        var jaw = a.model_data.floats("neutral_jaw")
        for i in range(jaw.numel()):
            a.jaw0.append(jaw.fget(i))
        var sac = a.model_data.floats("saccade_rot_matrix")
        for i in range(sac.numel()):
            a.saccade.append(sac.fget(i))
        a.net = OnnxModel.load(dir + "/network.onnx")
        if gpu:
            a.on_gpu = a.net.use_gpu()
        a.prepare()
        return a^

    def gpu_name(self) raises -> String:
        return self.net.gpu_name() if self.on_gpu else String("cpu")

    def set_emotion(mut self, name: String, value: Float32) -> Bool:
        """Explicit emotion weight (0..1) by name, e.g. "joy"; all 0 is neutral."""
        for i in range(len(self.emotion_names)):
            if self.emotion_names[i] == name:
                self.emotion[i] = value
                return True
        return False

    def prepare(mut self) raises:
        """Fold the animator parameters, PCA bases and solver bases (call after editing `params`)."""
        var p = self.params.copy()
        var ns = self.n_skin
        # ---- skin
        var mean = self.model_data.floats("shapes_mean_skin")       # [V, 3]
        var basis = self.model_data.floats("shapes_matrix_skin")    # [ns, V, 3]
        var eye = self.model_data.floats("eye_close_pose_delta")
        var lip = self.model_data.floats("lip_open_pose_delta")
        var nv = mean.shape[0]
        var neutral = self.bs_skin.floats("neutral")
        var mask_v = List[Int]()
        if self.bs_skin.has("frontalMask"):
            mask_v = self.bs_skin.ints("frontalMask")
        else:
            for v in range(nv):
                mask_v.append(v)
        var pm = len(mask_v) * 3
        var idx = List[Int](length=pm, fill=0)
        for i in range(len(mask_v)):
            for c in range(3):
                idx[i * 3 + c] = mask_v[i] * 3 + c
        # face mask over neutral height (all vertices)
        var ymin: Float32 = 3.4e38
        var ymax: Float32 = -3.4e38
        for v in range(nv):
            var y = mean.fget(v * 3 + 1)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
        var k = self.skin.k
        var d = self._deltas(self.bs_skin, self.skin, idx)              # [pm, k]
        var w = List[Float32](length=pm * 2 * k, fill=0.0)              # [pm, 2k]
        self.qu0 = List[Float32](length=k, fill=0.0)
        self.ql0 = List[Float32](length=k, fill=0.0)
        self.c0 = List[Float32](length=k, fill=0.0)
        var off_eye = -p.eyelid_open_offset + p.blink_offset * p.blink_strength
        for e in range(pm):
            var g = idx[e]
            var y = mean.fget((g // 3) * 3 + 1)
            var m = Float32(1.0 / (1.0 + exp(-Float64((p.face_mask_level - (y - ymin) / (ymax - ymin)) / p.face_mask_softness))))
            var wu = p.upper_face_strength * (1 - m)
            var wl = p.lower_face_strength * m
            var const_e = eye.fget(g) * off_eye + lip.fget(g) * p.lip_open_offset
            var dn = mean.fget(g) - neutral.fget(g)
            for j in range(k):
                var dv = d[e * k + j]
                w[e * 2 * k + j] = dv * wu * p.skin_strength
                w[e * 2 * k + k + j] = dv * wl * p.skin_strength
                self.qu0[j] += dv * wu * const_e
                self.ql0[j] += dv * wl * const_e
                self.c0[j] += dv * dn
        # basis restricted to the solved positions [ns, pm]
        var bsub = List[Float32](length=ns * pm, fill=0.0)
        var stride = nv * 3
        var pb = fptr_at(basis.fptr())
        for s in range(ns):
            for e in range(pm):
                bsub[s * pm + e] = pb.unsafe_offset(s * stride + idx[e]).unsafe_load()
        var q = List[Float32](length=ns * 2 * k, fill=0.0)
        matmul_2d(self.net.pool, Int(bsub.unsafe_ptr()), Int(w.unsafe_ptr()), Int(q.unsafe_ptr()), 0, ns, pm, 2 * k, False)
        self.qu = List[Float32](length=ns * k, fill=0.0)
        self.ql = List[Float32](length=ns * k, fill=0.0)
        for s in range(ns):
            for j in range(k):
                self.qu[s * k + j] = q[s * 2 * k + j]
                self.ql[s * k + j] = q[s * 2 * k + k + j]
        _ = bsub^
        _ = w^
        self.skin.build(_ata(d, pm, k), _bbox_diag(neutral))
        var dt = 1.0 / Float64(self.fps)
        self.alpha_u = Float32(1.0 - 0.5 ** (dt / Float64(p.upper_face_smoothing))) if p.upper_face_smoothing > 0 else Float32(-1)
        self.alpha_l = Float32(1.0 - 0.5 ** (dt / Float64(p.lower_face_smoothing))) if p.lower_face_smoothing > 0 else Float32(-1)
        # ---- tongue (no mask: every vertex)
        var tmean = self.model_data.floats("shapes_mean_tongue")
        var tbasis = self.model_data.floats("shapes_matrix_tongue")
        var tneutral = self.bs_tongue.floats("neutral")
        var pt = tmean.numel()
        var tidx = List[Int](length=pt, fill=0)
        for e in range(pt):
            tidx[e] = e
        var kt = self.tongue.k
        var dtg = self._deltas(self.bs_tongue, self.tongue, tidx)
        self.ct0 = List[Float32](length=kt, fill=0.0)
        self.qt = List[Float32](length=self.n_tongue * kt, fill=0.0)
        var ptb = fptr_at(tbasis.fptr())
        for e in range(pt):
            var offv: Float32 = 0
            if e % 3 == 1:
                offv = p.tongue_height_offset
            elif e % 3 == 2:
                offv = p.tongue_depth_offset
            var base = tmean.fget(e) + offv - tneutral.fget(e)
            for j in range(kt):
                self.ct0[j] += dtg[e * kt + j] * base
            for s in range(self.n_tongue):
                var bv = ptb.unsafe_offset(s * pt + e).unsafe_load() * p.tongue_strength
                for j in range(kt):
                    self.qt[s * kt + j] += bv * dtg[e * kt + j]
        self.tongue.build(_ata(dtg, pt, kt), _bbox_diag(tneutral))
        _ = mean^
        _ = basis^
        _ = eye^
        _ = lip^
        _ = neutral^
        _ = tmean^
        _ = tbasis^
        _ = tneutral^
        self.reset()

    def _deltas(self, npz: Npz, solver: BsSolver, idx: List[Int]) raises -> List[Float32]:
        """[len(idx), k]: the solved poses' deltas at the given flat positions."""
        var k = solver.k
        var n = len(idx)
        var d = List[Float32](length=n * k, fill=0.0)
        for j in range(k):
            var t = npz.floats(solver.names[solver.active[j]])
            var pt = fptr_at(t.fptr())
            for e in range(n):
                d[e * k + j] = pt.unsafe_offset(idx[e]).unsafe_load()
            _ = t^
        return d^

    def reset(mut self):
        """Start a new track: clears smoothing, solver history and eye live time."""
        self.started = False
        self.live = 0
        self.skin.reset()
        self.tongue.reset()
        self.su1 = List[Float32](length=self.skin.k, fill=0.0)
        self.su2 = List[Float32](length=self.skin.k, fill=0.0)
        self.sl1 = List[Float32](length=self.skin.k, fill=0.0)
        self.sl2 = List[Float32](length=self.skin.k, fill=0.0)
        self.sbuf = List[Float32]()
        self.sbase = 0
        self.snext = 0

    def _new_clip(self) -> A2fClip:
        var c = A2fClip()
        c.fps = self.fps
        c.skin_names = self.skin.names.copy()
        c.tongue_names = self.tongue.names.copy()
        return c^

    def infer(mut self, audio16: List[Float32], first: Int, count: Int, base: Int = 0) raises -> List[Float32]:
        """Raw network output [count, n_out] for frames first .. first + count - 1 (audio16[0] is
        sample `base` of the track)."""
        var out = List[Float32](length=count * self.n_out, fill=0.0)
        var done = 0
        var ne = len(self.implicit) + len(self.emotion)
        while done < count:
            var b = min(self.batch, count - done)
            var win = List[Float32](length=b * self.buffer_len, fill=0.0)
            var emo = List[Float32](length=b * ne, fill=0.0)
            for i in range(b):
                a2f_window(audio16, first + done + i, self.fps, self.buffer_len, self.buffer_ofs,
                           self.params.input_strength, win, i * self.buffer_len, base)
                for j in range(len(self.implicit)):
                    emo[i * ne + j] = self.implicit[j]
                for j in range(len(self.emotion)):
                    emo[i * ne + len(self.implicit) + j] = self.emotion[j]
            self.net.set_input("input", Tensor.floats([b, 1, self.buffer_len], win^))
            self.net.set_input("emotion", Tensor.floats([b, 1, ne], emo^))
            self.net.run()
            var r = self.net.output("result")
            for i in range(b * self.n_out):
                out[done * self.n_out + i] = r.fget(i)
            done += b
        return out^

    def animate(mut self, net: List[Float32], count: Int, mut clip: A2fClip):
        """Post-process `count` consecutive frames of network output (state carries across calls)."""
        var p = self.params.copy()
        var ns = self.n_skin
        var k = self.skin.k
        var kt = self.tongue.k
        var nsac = len(self.saccade) // 2
        var offs: List[Float32] = [p.right_eye_rot_x_offset, p.right_eye_rot_y_offset,
                                   p.left_eye_rot_x_offset, p.left_eye_rot_y_offset]
        var npts = len(self.jaw0) // 3
        for f in range(count):
            var base = f * self.n_out
            for i in range(self.n_out):
                clip.net.append(net[base + i])
            # skin
            var pu = self.qu0.copy()
            var pl = self.ql0.copy()
            for s in range(ns):
                var c = net[base + s]
                if c == 0:
                    continue
                for j in range(k):
                    pu[j] += c * self.qu[s * k + j]
                    pl[j] += c * self.ql[s * k + j]
            if not self.started or self.alpha_u <= 0:
                self.su1 = pu.copy()
                self.su2 = pu.copy()
            else:
                for j in range(k):
                    self.su1[j] += (pu[j] - self.su1[j]) * self.alpha_u
                    self.su2[j] += (self.su1[j] - self.su2[j]) * self.alpha_u
            if not self.started or self.alpha_l <= 0:
                self.sl1 = pl.copy()
                self.sl2 = pl.copy()
            else:
                for j in range(k):
                    self.sl1[j] += (pl[j] - self.sl1[j]) * self.alpha_l
                    self.sl2[j] += (self.sl1[j] - self.sl2[j]) * self.alpha_l
            self.started = True
            var atb = List[Float32](length=k, fill=0.0)
            for j in range(k):
                atb[j] = self.c0[j] + self.su2[j] + self.sl2[j]
            for v in self.skin.solve(atb):
                clip.skin.append(v)
            # tongue
            var att = self.ct0.copy()
            for s in range(self.n_tongue):
                var c = net[base + ns + s]
                for j in range(kt):
                    att[j] += c * self.qt[s * kt + j]
            for v in self.tongue.solve(att):
                clip.tongue.append(v)
            # jaw
            var jb = base + ns + self.n_tongue
            var tgt = List[Float32](length=npts * 3, fill=0.0)
            for i in range(npts):
                for c in range(3):
                    var o: Float32 = 0
                    if c == 1:
                        o = p.lower_teeth_height_offset
                    elif c == 2:
                        o = p.lower_teeth_depth_offset
                    tgt[i * 3 + c] = self.jaw0[i * 3 + c] + net[jb + i * 3 + c] * p.lower_teeth_strength + o
            for v in rigid_fit(self.jaw0, tgt, npts):
                clip.jaw.append(v)
            # eyes
            var eb = jb + self.n_jaw
            var tt = _wrap(p.eye_saccade_seed + self.live, Float32(nsac))
            var fi = min(Int(tt), nsac - 1)
            for e in range(2):
                for ax in range(2):
                    clip.eyes.append(offs[e * 2 + ax] + net[eb + e * 2 + ax] * p.eyeballs_strength
                                     + p.saccade_strength * self.saccade[2 * fi + ax])
                clip.eyes.append(0)
            self.live = _wrap(self.live + Float32(1.0 / Float64(self.fps)) * 30, Float32(nsac))
            clip.frames += 1

    def process(mut self, audio16: List[Float32]) raises -> A2fClip:
        """A whole 16 kHz mono track (closed): every frame whose target sample is in the audio."""
        self.reset()
        var clip = self._new_clip()
        var frames = a2f_frame_count(len(audio16), self.fps)
        var t0 = perf_counter_ns()
        var net = self.infer(audio16, 0, frames)
        var t1 = perf_counter_ns()
        self.animate(net, frames, clip)
        self.t_net = Int(t1 - t0)
        self.t_post = Int(perf_counter_ns() - t1)
        return clip^

    def process_wav(mut self, path: String) raises -> A2fClip:
        """Any-rate PCM / float WAV, downmixed and resampled to 16 kHz."""
        var wav = read_wav(path)
        if not wav.valid:
            raise Error("audio2face: cannot read WAV " + path)
        return self.process(to_model_rate(wav))

    def stream_reset(mut self):
        self.reset()

    def stream_push(mut self, samples16: List[Float32]) raises -> A2fClip:
        """Append 16 kHz audio; returns the frames whose windows are now complete."""
        for v in samples16:
            self.sbuf.append(v)
        var clip = self._new_clip()
        var n = self.sbase + len(self.sbuf)
        var last = self.snext
        while a2f_target_sample(last, self.fps) - self.buffer_ofs + self.buffer_len <= n:
            last += 1
        if last > self.snext:
            self._stream_run(last, clip)
        return clip^

    def stream_finish(mut self) raises -> A2fClip:
        """The remaining frames of the track, as if it were closed now."""
        var clip = self._new_clip()
        var total = a2f_frame_count(self.sbase + len(self.sbuf), self.fps)
        if total > self.snext:
            self._stream_run(total, clip)
        return clip^

    def _stream_run(mut self, last: Int, mut clip: A2fClip) raises:
        var buf = self.sbuf.copy()
        var net = self.infer(buf, self.snext, last - self.snext, self.sbase)
        self.animate(net, last - self.snext, clip)
        self.snext = last
        # drop the samples no later window reads
        var keep = a2f_target_sample(last, self.fps) - self.buffer_ofs - self.sbase
        if keep > 0:
            keep = min(keep, len(self.sbuf))
            var rest = List[Float32](capacity=len(self.sbuf) - keep)
            for k in range(keep, len(self.sbuf)):
                rest.append(self.sbuf[k])
            self.sbuf = rest^
            self.sbase += keep


def _wrap(v: Float32, m: Float32) -> Float32:
    var r = v - m * Float32(Int(v / m)) if v >= 0 else v - m * Float32(Int(v / m) - 1)
    if r >= m:
        r -= m
    if r < 0:
        r += m
    return r


def _pose_names(npz: Npz) raises -> List[String]:
    var names = npz.strings("poseNames")
    if len(names) == 0 or names[0] != "neutral":
        raise Error("audio2face: poseNames[0] must be \"neutral\" in " + npz.path)
    var out = List[String]()
    for i in range(1, len(names)):
        out.append(names[i])
    return out^


def _ata(d: List[Float32], n: Int, k: Int) -> List[Float64]:
    var out = List[Float64](length=k * k, fill=0.0)
    for e in range(n):
        for i in range(k):
            var di = Float64(d[e * k + i])
            if di == 0:
                continue
            for j in range(i, k):
                out[i * k + j] += di * Float64(d[e * k + j])
    for i in range(k):
        for j in range(i):
            out[i * k + j] = out[j * k + i]
    return out^


def _bbox_diag(neutral: Tensor) -> Float64:
    var lo = List[Float64](length=3, fill=1e300)
    var hi = List[Float64](length=3, fill=-1e300)
    var n = neutral.numel() // 3
    for v in range(n):
        for c in range(3):
            var x = Float64(neutral.fget(v * 3 + c))
            lo[c] = min(lo[c], x)
            hi[c] = max(hi[c], x)
    var s = 0.0
    for c in range(3):
        s += (hi[c] - lo[c]) * (hi[c] - lo[c])
    return sqrt(s)



# ----------------------------------------------------------------------------- animated GLB

struct _GlbBuild(Movable):
    var bin: List[UInt8]
    var views: List[String]
    var accessors: List[String]

    def __init__(out self):
        self.bin = List[UInt8]()
        self.views = List[String]()
        self.accessors = List[String]()

    def floats(mut self, data: List[Float32], comps: Int, kind: String, minmax: Bool) -> Int:
        """One float accessor in its own buffer view; returns the accessor index."""
        var off = len(self.bin)
        for v in data:
            append_f32_le(self.bin, v)
        var count = len(data) // comps
        self.views.append(String('{"buffer": 0, "byteOffset": ') + String(off) + ', "byteLength": ' + String(len(self.bin) - off) + "}")
        var s = String('{"bufferView": ') + String(len(self.views) - 1) + ', "componentType": 5126, "count": ' + String(count) + ', "type": "' + kind + '"'
        if minmax:
            var lo = List[Float32](length=comps, fill=3.4e38)
            var hi = List[Float32](length=comps, fill=-3.4e38)
            for i in range(count):
                for c in range(comps):
                    lo[c] = min(lo[c], data[i * comps + c])
                    hi[c] = max(hi[c], data[i * comps + c])
            s += ', "min": [' + _join(lo) + '], "max": [' + _join(hi) + "]"
        s += "}"
        self.accessors.append(s)
        return len(self.accessors) - 1


def _join(v: List[Float32]) -> String:
    var s = String()
    for i in range(len(v)):
        if i > 0:
            s += ", "
        s += fmt_f32(v[i])
    return s^


def encode_face_glb(a2f: Audio2Face, clip: A2fClip) raises -> List[UInt8]:
    """A point-cloud GLB of the model's solved face vertices (bs_skin neutral on the frontal
    mask) with the 52 ARKit poses as named morph targets and the clip as a LINEAR "weights"
    animation, so any glTF viewer plays the result."""
    var neutral = a2f.bs_skin.floats("neutral")
    var mask = List[Int]()
    if a2f.bs_skin.has("frontalMask"):
        mask = a2f.bs_skin.ints("frontalMask")
    else:
        for v in range(neutral.numel() // 3):
            mask.append(v)
    var nv = len(mask)
    var g = _GlbBuild()
    var pos = List[Float32](length=nv * 3, fill=0.0)
    for i in range(nv):
        for c in range(3):
            pos[i * 3 + c] = neutral.fget(mask[i] * 3 + c)
    var pos_acc = g.floats(pos, 3, "VEC3", True)
    var targets = String()
    var names = String()
    var np = len(a2f.skin.names)
    for j in range(np):
        var t = a2f.bs_skin.floats(a2f.skin.names[j])
        var d = List[Float32](length=nv * 3, fill=0.0)
        for i in range(nv):
            for c in range(3):
                d[i * 3 + c] = t.fget(mask[i] * 3 + c)
        var acc = g.floats(d, 3, "VEC3", True)
        if j > 0:
            targets += ", "
            names += ", "
        targets += '{"POSITION": ' + String(acc) + "}"
        names += json_quote(a2f.skin.names[j])
    var json = String('{"asset": {"version": "2.0", "generator": "Flint flint.ml.audio2face"}, "scene": 0, ')
    json += '"scenes": [{"nodes": [0]}], "nodes": [{"name": "a2f_face", "mesh": 0}], '
    json += '"meshes": [{"name": "a2f_face", "primitives": [{"attributes": {"POSITION": ' + String(pos_acc) + '}, "mode": 0, "targets": [' + targets + ']}], '
    var zeros = List[Float32](length=np, fill=0.0)
    json += '"weights": [' + _join(zeros) + '], "extras": {"targetNames": [' + names + "]}}]"
    if clip.frames > 0 and clip.skin_count() == np:
        var times = List[Float32](length=clip.frames, fill=0.0)
        for f in range(clip.frames):
            times[f] = Float32(Float64(f) / Float64(clip.fps))
        var t_acc = g.floats(times, 1, "SCALAR", True)
        var w_acc = g.floats(clip.skin, 1, "SCALAR", False)
        json += ', "animations": [{"name": "audio2face", "samplers": [{"input": ' + String(t_acc) + ', "output": ' + String(w_acc)
        json += ', "interpolation": "LINEAR"}], "channels": [{"sampler": 0, "target": {"node": 0, "path": "weights"}}]}]'
    json += ', "buffers": [{"byteLength": ' + String(len(g.bin)) + '}], "bufferViews": ['
    for i in range(len(g.views)):
        json += (", " if i > 0 else "") + g.views[i]
    json += '], "accessors": ['
    for i in range(len(g.accessors)):
        json += (", " if i > 0 else "") + g.accessors[i]
    json += "]}"
    var jb = List[UInt8]()
    for b in json.as_bytes():
        jb.append(b)
    pad4(jb, 0x20)
    pad4(g.bin, 0)
    var out = List[UInt8](capacity=28 + len(jb) + len(g.bin))
    append_u32_le(out, 0x46546C67)
    append_u32_le(out, 2)
    append_u32_le(out, 28 + len(jb) + len(g.bin))
    append_u32_le(out, len(jb))
    append_u32_le(out, 0x4E4F534A)
    for b in jb:
        out.append(b)
    append_u32_le(out, len(g.bin))
    append_u32_le(out, 0x004E4942)
    for b in g.bin:
        out.append(b)
    _ = neutral^
    return out^


def write_face_glb(path: String, a2f: Audio2Face, clip: A2fClip) raises -> Bool:
    var b = encode_face_glb(a2f, clip)
    return write_bytes(path, b)
