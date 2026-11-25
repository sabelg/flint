"""Learned Motion Matching networks (`flint.anim.lmm`).

Pure-Mojo inference for the three small networks of "Learned Motion Matching" (Holden, Kanoun,
Perepichka, Popa; SIGGRAPH 2020), following the MIT reference implementation
(https://github.com/orangeduck/Motion-Matching, `nnet.h` and `lmm.h`):

* **decompressor** (features + latent -> pose): local bone positions, two-axis rotations,
  velocities and angular velocities of every non-root bone, the root's local velocity and
  angular velocity, and the two foot contacts. Replaces the pose database.
* **stepper** (features + latent -> their time derivative): advances the playing frame by dt.
  Replaces stepping the frame index.
* **projector** (query -> nearest features + latent): replaces the nearest-neighbour search.

`MlpNet` is the reference `nnet`: input normalisation, ReLU on every layer but the last,
output denormalisation; weights are `[in][out]` row-major and the file layout is the
reference `.bin` (`u32` sizes, little-endian `f32` data), as `tools/learned_mm/train.py`
writes it. The matrix-vector product skips zero (ReLU'd) activations and vectorises over the
outputs; every output accumulates its inputs in a fixed order, so results are deterministic.
The networks are trained offline from a Flint `MmDatabase`; weights are not shipped.
"""
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.platform.os import read_bytes
from flint.anim.mm_database import MmBytes
from flint.anim.mm_springs import MmPose, mm_quat_from_xform_xy, mm_quat_from_scaled_angle_axis, mm_quat_mul_vec, mm_quat_mul

comptime _W: Int = 8


struct MlpNet(Copyable, Movable):
    var input_mean: List[Float32]
    var input_std: List[Float32]
    var output_mean: List[Float32]
    var output_std: List[Float32]
    var rows: List[Int]
    var cols: List[Int]
    var weights: List[List[Float32]]
    var biases: List[List[Float32]]

    def __init__(out self):
        self.input_mean = List[Float32]()
        self.input_std = List[Float32]()
        self.output_mean = List[Float32]()
        self.output_std = List[Float32]()
        self.rows = List[Int]()
        self.cols = List[Int]()
        self.weights = List[List[Float32]]()
        self.biases = List[List[Float32]]()

    def layer_count(self) -> Int:
        return len(self.weights)

    def input_size(self) -> Int:
        return len(self.input_mean)

    def output_size(self) -> Int:
        return len(self.output_mean)

    def parameter_count(self) -> Int:
        var n = 0
        for i in range(len(self.weights)):
            n += len(self.weights[i]) + len(self.biases[i])
        return n

    @staticmethod
    def _array1d(mut r: MmBytes) raises -> List[Float32]:
        var n = r.read_u32()
        var out = List[Float32](capacity=n)
        for _ in range(n):
            out.append(r.read_f32())
        return out^

    @staticmethod
    def load(path: String) raises -> MlpNet:
        var bytes = read_bytes(path)
        if len(bytes) == 0:
            raise Error("lmm: cannot read " + path)
        var r = MmBytes(bytes^)
        var nn = MlpNet()
        nn.input_mean = MlpNet._array1d(r)
        nn.input_std = MlpNet._array1d(r)
        nn.output_mean = MlpNet._array1d(r)
        nn.output_std = MlpNet._array1d(r)
        var count = r.read_u32()
        for _ in range(count):
            var rows = r.read_u32()
            var cols = r.read_u32()
            var w = List[Float32](capacity=rows * cols)
            for _ in range(rows * cols):
                w.append(r.read_f32())
            nn.rows.append(rows)
            nn.cols.append(cols)
            nn.weights.append(w^)
            nn.biases.append(MlpNet._array1d(r))
        if len(nn.weights) == 0 or nn.rows[0] != len(nn.input_mean) or nn.cols[len(nn.cols) - 1] != len(nn.output_mean):
            raise Error("lmm: inconsistent network " + path)
        for i in range(1, len(nn.rows)):
            if nn.rows[i] != nn.cols[i - 1]:
                raise Error("lmm: inconsistent layer sizes in " + path)
        return nn^


struct MlpEval(Copyable, Movable):
    """Preallocated activations for one network (`layers[0]` is the input)."""
    var layers: List[List[Float32]]

    def __init__(out self, nn: MlpNet):
        self.layers = List[List[Float32]]()
        self.layers.append(List[Float32](length=nn.input_size(), fill=0))
        for i in range(nn.layer_count()):
            self.layers.append(List[Float32](length=nn.cols[i], fill=0))

    def output(self) -> ref [self.layers] List[Float32]:
        return self.layers[len(self.layers) - 1]


def _linear(mut out: List[Float32], x: List[Float32], w: List[Float32], b: List[Float32], cols: Int):
    var po = out.unsafe_ptr()
    var pw = w.unsafe_ptr()
    for j in range(cols):
        po[unsafe_offset=j] = b[j]
    var vec_end = cols - cols % _W
    for i in range(len(x)):
        var xi = x[i]
        if xi == 0:
            continue
        var row = i * cols
        var xv = SIMD[DType.float32, _W](xi)
        var j = 0
        while j < vec_end:
            var acc = po.unsafe_offset(j).unsafe_load[width=_W, alignment=1]()
            var wv = pw.unsafe_offset(row + j).unsafe_load[width=_W, alignment=1]()
            po.unsafe_offset(j).unsafe_store[alignment=1](acc + xv * wv)
            j += _W
        while j < cols:
            po[unsafe_offset=j] = po[unsafe_offset=j] + xi * pw[unsafe_offset=row + j]
            j += 1


def mlp_evaluate(mut ev: MlpEval, nn: MlpNet):
    """Evaluates `nn` on `ev.layers[0]` (raw input); the denormalised result is the last layer."""
    ref inp = ev.layers[0]
    for i in range(len(inp)):
        inp[i] = (inp[i] - nn.input_mean[i]) / nn.input_std[i]
    var nl = nn.layer_count()
    for k in range(nl):
        var x = ev.layers[k].copy()
        _linear(ev.layers[k + 1], x, nn.weights[k], nn.biases[k], nn.cols[k])
        if k != nl - 1:
            ref y = ev.layers[k + 1]
            for j in range(len(y)):
                y[j] = max(y[j], Float32(0))
    ref o = ev.layers[nl]
    for j in range(len(o)):
        o[j] = o[j] * nn.output_std[j] + nn.output_mean[j]


struct LmmNetworks(Copyable, Movable):
    var decompressor: MlpNet
    var stepper: MlpNet
    var projector: MlpNet
    var nfeatures: Int
    var nlatent: Int
    var dec_ev: MlpEval
    var step_ev: MlpEval
    var proj_ev: MlpEval

    def __init__(out self, var decompressor: MlpNet, var stepper: MlpNet, var projector: MlpNet) raises:
        self.nfeatures = projector.input_size()
        self.nlatent = projector.output_size() - self.nfeatures
        if self.nlatent <= 0 or decompressor.input_size() != self.nfeatures + self.nlatent or stepper.input_size() != self.nfeatures + self.nlatent:
            raise Error("lmm: decompressor / stepper / projector sizes do not agree")
        self.dec_ev = MlpEval(decompressor)
        self.step_ev = MlpEval(stepper)
        self.proj_ev = MlpEval(projector)
        self.decompressor = decompressor^
        self.stepper = stepper^
        self.projector = projector^

    @staticmethod
    def load(dir: String) raises -> LmmNetworks:
        return LmmNetworks(
            MlpNet.load(dir + "/decompressor.bin"), MlpNet.load(dir + "/stepper.bin"), MlpNet.load(dir + "/projector.bin")
        )

    def decompress(
        mut self, mut out: MmPose, mut contacts: List[Bool], features: List[Float32], latent: List[Float32],
        root_position: Vec3, root_rotation: Quat, dt: Float32 = 1.0 / 60.0,
    ):
        """Pose for (features, latent): bones 1.. local, root integrated from the given root
        transform with the predicted root velocities."""
        ref inp = self.dec_ev.layers[0]
        for i in range(self.nfeatures):
            inp[i] = features[i]
        for i in range(self.nlatent):
            inp[self.nfeatures + i] = latent[i]
        mlp_evaluate(self.dec_ev, self.decompressor)
        ref o = self.dec_ev.layers[len(self.dec_ev.layers) - 1]
        var nb = out.count()
        var off = 0
        for i in range(nb - 1):
            out.pos[i + 1] = Vec3(o[off + i * 3], o[off + i * 3 + 1], o[off + i * 3 + 2])
        off += (nb - 1) * 3
        for i in range(nb - 1):
            var b = off + i * 6
            out.rot[i + 1] = mm_quat_from_xform_xy(Vec3(o[b], o[b + 2], o[b + 4]), Vec3(o[b + 1], o[b + 3], o[b + 5]))
        off += (nb - 1) * 6
        for i in range(nb - 1):
            out.vel[i + 1] = Vec3(o[off + i * 3], o[off + i * 3 + 1], o[off + i * 3 + 2])
        off += (nb - 1) * 3
        for i in range(nb - 1):
            out.ang[i + 1] = Vec3(o[off + i * 3], o[off + i * 3 + 1], o[off + i * 3 + 2])
        off += (nb - 1) * 3
        var rv = mm_quat_mul_vec(root_rotation, Vec3(o[off], o[off + 1], o[off + 2]))
        var ra = mm_quat_mul_vec(root_rotation, Vec3(o[off + 3], o[off + 4], o[off + 5]))
        off += 6
        out.pos[0] = rv * dt + root_position
        out.rot[0] = mm_quat_mul(mm_quat_from_scaled_angle_axis(ra * dt), root_rotation).normalized()
        out.vel[0] = rv
        out.ang[0] = ra
        if len(contacts) < 2:
            contacts = List[Bool](length=2, fill=False)
        contacts[0] = o[off] > Float32(0.5)
        contacts[1] = o[off + 1] > Float32(0.5)

    def step(mut self, mut features: List[Float32], mut latent: List[Float32], dt: Float32 = 1.0 / 60.0):
        ref inp = self.step_ev.layers[0]
        for i in range(self.nfeatures):
            inp[i] = features[i]
        for i in range(self.nlatent):
            inp[self.nfeatures + i] = latent[i]
        mlp_evaluate(self.step_ev, self.stepper)
        ref o = self.step_ev.layers[len(self.step_ev.layers) - 1]
        for i in range(self.nfeatures):
            features[i] += dt * o[i]
        for i in range(self.nlatent):
            latent[i] += dt * o[self.nfeatures + i]

    def project(
        mut self, mut proj_features: List[Float32], mut proj_latent: List[Float32], query_normalized: List[Float32],
        curr_features: List[Float32], transition_cost: Float32 = 0,
    ) -> Tuple[Bool, Float32]:
        """Projects a normalised query; returns (transition, cost). Without a transition the
        projected features are the current ones."""
        ref inp = self.proj_ev.layers[0]
        for i in range(self.nfeatures):
            inp[i] = query_normalized[i]
        mlp_evaluate(self.proj_ev, self.projector)
        ref o = self.proj_ev.layers[len(self.proj_ev.layers) - 1]
        for i in range(self.nfeatures):
            proj_features[i] = o[i]
        for i in range(self.nlatent):
            proj_latent[i] = o[self.nfeatures + i]
        var trns: Float32 = 0
        var cost: Float32 = 0
        for i in range(self.nfeatures):
            var d = curr_features[i] - proj_features[i]
            trns += d * d
            var e = query_normalized[i] - proj_features[i]
            cost += e * e
        if trns > transition_cost * transition_cost:
            return (True, cost + transition_cost)
        cost = 0
        for i in range(self.nfeatures):
            proj_features[i] = curr_features[i]
            var e = query_normalized[i] - curr_features[i]
            cost += e * e
        return (False, cost)
