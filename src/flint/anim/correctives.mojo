"""Corrective morphs & Gaussian RBF Pose-Space Deformation (`flint.anim.correctives`).

Upgrades in Flint v4:
- Deterministic `flint.core.detmath` (`det_asin`, `det_atan2`, `det_sin`, `det_cos`, `det_exp`).
- `RbfPoseDriver`: Gaussian Radial Basis Function pose-space solver precomputing
  `Phi^-1` (`inv_phi`) and `W = Phi^-1 * Y` at bind/finalize time via Gauss-Jordan
  partial pivoting so sculpted cardinal poses hit exact Kronecker delta weights
  (`w_m = 1, others = 0`), rest pose hits `0`, and diagonal poses interpolate
  cleanly near `sum(w_m) ~ 1.0` without `2x` additive linear bulge.
- Non-mutating `apply_to_pose` helpers alongside `NativeModel.set_morph_weight`.
"""
from std.math import sqrt
from flint.core.detmath import det_asin, det_atan2, det_sin, det_cos, det_exp
from flint.core.transform import Quat
from flint.assets.model import NativeModel
from flint.assets.manifest import ManifestCorrective
from flint.anim.pose import Pose


struct Swing(Copyable, ImplicitlyCopyable, Movable):
    var x: Float32
    var z: Float32
    var twist: Float32

    def __init__(out self, x: Float32, z: Float32, twist: Float32):
        self.x = x
        self.z = z
        self.twist = twist


def swing_twist_y(rotation: Quat, rest: Quat) -> Swing:
    """Splits a bone's turn from rest into axial Y twist and swing `(x, z)`."""
    var turn = rest.conjugate().multiply(rotation).normalized().shortest()
    var half = det_atan2(turn.y, turn.w)
    var swing = turn.multiply(Quat(0, -det_sin(half), 0, det_cos(half)))
    var x = max(Float32(-1), min(Float32(1), swing.x))
    var z = max(Float32(-1), min(Float32(1), swing.z))
    return Swing(Float32(2.0) * det_asin(x), Float32(2.0) * det_asin(z), Float32(2.0) * half)


def corrective_weight(x: Float32, z: Float32, rotation: Quat, rest: Quat) -> Float32:
    var swing = swing_twist_y(rotation, rest)
    return max(Float32(0), min(Float32(1), x * swing.x + z * swing.z))


struct Corrective(Copyable, ImplicitlyCopyable, Movable):
    var node: Int
    var morph: Int
    var x: Float32
    var z: Float32

    def __init__(out self):
        self.node = -1
        self.morph = -1
        self.x = 0
        self.z = 0


struct Correctives(Copyable, Movable):
    var handle: Int
    var rows: List[Corrective]
    var weights: List[Float32]

    def __init__(out self):
        self.handle = -1
        self.rows = List[Corrective]()
        self.weights = List[Float32]()

    def count(self) -> Int:
        return len(self.rows)

    def bind(mut self, model: NativeModel, handle: Int, table: List[ManifestCorrective]):
        if handle == self.handle:
            return
        self.handle = handle
        self.rows.clear()
        self.weights.clear()
        for entry in table:
            var row = Corrective()
            row.node = model.find_node(entry.bone)
            for index in range(len(model.morph_names)):
                if model.morph_names[index] == entry.morph:
                    row.morph = index
            if row.node < 0 or row.morph < 0:
                continue
            row.x = entry.x
            row.z = entry.z
            self.rows.append(row)
            self.weights.append(0)

    def apply(mut self, mut model: NativeModel):
        for index in range(len(self.rows)):
            var row = self.rows[index]
            if row.node >= len(model.nodes):
                continue
            var weight = corrective_weight(row.x, row.z, model.nodes[row.node].rotation, model.nodes[row.node].base_rotation)
            self.weights[index] = weight
            model.set_morph_weight(row.morph, weight)

    def apply_to_pose(mut self, pose: Pose, model: NativeModel, mut out_morph_weights: List[Float32]):
        if len(out_morph_weights) < len(model.morph_names):
            out_morph_weights.resize(len(model.morph_names), Float32(0.0))
        for index in range(len(self.rows)):
            var row = self.rows[index]
            if row.node < 0 or row.node >= pose.count() or row.node >= len(model.nodes):
                continue
            var weight = corrective_weight(row.x, row.z, pose.r[row.node], model.nodes[row.node].base_rotation)
            self.weights[index] = weight
            if row.morph >= 0 and row.morph < len(out_morph_weights):
                out_morph_weights[row.morph] = weight


# ---------------------------------------------------------------------------
# Gaussian Radial Basis Function (RBF) Pose-Space Solver (`RbfPoseDriver`)
# ---------------------------------------------------------------------------

def _invert_square_matrix_f64(mat: List[Float32], n: Int) -> List[Float32]:
    """Gauss-Jordan elimination with partial pivoting in `Float64`, returning `mat^-1`."""
    var aug = List[Float64](length=n * n * 2, fill=0.0)
    for r in range(n):
        for c in range(n):
            aug[r * (2 * n) + c] = Float64(mat[r * n + c])
        aug[r * (2 * n) + n + r] = Float64(1.0)

    for col in range(n):
        var piv = col
        var best_abs = abs(aug[col * (2 * n) + col])
        for row in range(col + 1, n):
            var v = abs(aug[row * (2 * n) + col])
            if v > best_abs:
                best_abs = v
                piv = row
        if best_abs < Float64(1.0e-12):
            var ident = List[Float32](length=n * n, fill=0.0)
            for d in range(n):
                ident[d * n + d] = Float32(1.0)
            return ident^
        if piv != col:
            for k in range(2 * n):
                var tmp = aug[col * (2 * n) + k]
                aug[col * (2 * n) + k] = aug[piv * (2 * n) + k]
                aug[piv * (2 * n) + k] = tmp

        var diag = aug[col * (2 * n) + col]
        var inv_diag = Float64(1.0) / diag
        for k in range(2 * n):
            aug[col * (2 * n) + k] *= inv_diag

        for row in range(n):
            if row == col:
                continue
            var factor = aug[row * (2 * n) + col]
            if abs(factor) > Float64(0.0):
                for k in range(2 * n):
                    aug[row * (2 * n) + k] -= factor * aug[col * (2 * n) + k]

    var inv_out = List[Float32](length=n * n, fill=0.0)
    for r in range(n):
        for c in range(n):
            inv_out[r * n + c] = Float32(aug[r * (2 * n) + n + c])
    return inv_out^


struct RbfPoseCenter(Copyable, Movable):
    var swing_x: Float32
    var swing_z: Float32
    var twist: Float32
    var target_weights: List[Float32]

    def __init__(out self, sx: Float32, sz: Float32, twist: Float32, var weights: List[Float32]):
        self.swing_x = sx
        self.swing_z = sz
        self.twist = twist
        self.target_weights = weights^


struct RbfPoseDriver(Copyable, Movable):
    """Gaussian RBF pose-space deformer driving `morph_targets` from a bone's
    `(swing_x, swing_z, twist)` channels.

    Precomputes `inv_phi = Phi^-1` (`Phi_ij = exp(-||c_i - c_j||^2 / (2*sigma^2))`)
    and `coeff_matrix = Phi^-1 * Y` at `finalize()` / `bind()` time so diagonal
    combined swings interpolate cleanly (`sum(w_m) ~ 1.0`) without `2x` linear bulge.
    """
    var node: Int
    var morph_targets: List[Int]
    var use_twist: Bool
    var sigma: Float32
    var centers: List[RbfPoseCenter]
    var inv_phi: List[Float32]
    var coeff_matrix: List[Float32]
    var weights: List[Float32]
    var is_bound: Bool

    def __init__(out self, node: Int = -1, use_twist: Bool = False):
        self.node = node
        self.morph_targets = List[Int]()
        self.use_twist = use_twist
        self.sigma = 0.0
        self.centers = List[RbfPoseCenter]()
        self.inv_phi = List[Float32]()
        self.coeff_matrix = List[Float32]()
        self.weights = List[Float32]()
        self.is_bound = False

    def add_sculpted_pose(
        mut self,
        swing_x: Float32,
        swing_z: Float32,
        morph_target: Int,
        twist: Float32 = 0.0,
    ):
        """Registers a sculpted pose activating `morph_target` at weight `1.0`."""
        var slot = -1
        for i in range(len(self.morph_targets)):
            if self.morph_targets[i] == morph_target:
                slot = i
                break
        if slot < 0:
            slot = len(self.morph_targets)
            self.morph_targets.append(morph_target)
            for c in range(len(self.centers)):
                self.centers[c].target_weights.append(Float32(0.0))

        var w = List[Float32](length=len(self.morph_targets), fill=0.0)
        w[slot] = Float32(1.0)
        self.centers.append(RbfPoseCenter(swing_x, swing_z, twist, w^))
        self.is_bound = False

    def finalize(mut self, sigma_override: Float32 = 0.0):
        """Inserts neutral rest pose `(0, 0, 0) -> 0` if absent, estimates Gaussian
        width `sigma`, and precomputes `inv_phi = Phi^-1` and `coeff_matrix = Phi^-1 * Y`."""
        var k_morphs = len(self.morph_targets)
        if len(self.centers) == 0 or k_morphs == 0:
            self.is_bound = False
            return

        # Ensure rest pose `(0, 0, 0)` is anchored at zero morph weights
        var has_rest = False
        for c in range(len(self.centers)):
            var d2 = self.centers[c].swing_x * self.centers[c].swing_x + self.centers[c].swing_z * self.centers[c].swing_z
            if self.use_twist:
                d2 += self.centers[c].twist * self.centers[c].twist
            if d2 < Float32(1.0e-6):
                has_rest = True
                break
        if not has_rest:
            var zero_w = List[Float32](length=k_morphs, fill=0.0)
            self.centers.insert(0, RbfPoseCenter(0.0, 0.0, 0.0, zero_w^))

        var n = len(self.centers)
        if sigma_override > Float32(1.0e-4):
            self.sigma = sigma_override
        else:
            # Mean nearest-neighbor distance * 1/sqrt(2) gives ~1.0 partition sum on diagonals
            var sum_nn = Float32(0.0)
            var count_nn = 0
            for i in range(n):
                var best_d2 = Float32(1.0e30)
                for j in range(n):
                    if i == j:
                        continue
                    var dx = self.centers[i].swing_x - self.centers[j].swing_x
                    var dz = self.centers[i].swing_z - self.centers[j].swing_z
                    var dt = (self.centers[i].twist - self.centers[j].twist) if self.use_twist else Float32(0.0)
                    var d2 = dx * dx + dz * dz + dt * dt
                    if d2 > Float32(1.0e-8) and d2 < best_d2:
                        best_d2 = d2
                if best_d2 < Float32(1.0e20):
                    sum_nn += sqrt(best_d2)
                    count_nn += 1
            var mean_nn = (sum_nn / Float32(count_nn)) if count_nn > 0 else Float32(1.0)
            self.sigma = max(Float32(0.15), mean_nn * Float32(0.70710678))

        var inv_2sig2 = Float32(1.0) / (Float32(2.0) * self.sigma * self.sigma)
        var phi = List[Float32](length=n * n, fill=0.0)
        for i in range(n):
            for j in range(n):
                var dx = self.centers[i].swing_x - self.centers[j].swing_x
                var dz = self.centers[i].swing_z - self.centers[j].swing_z
                var dt = (self.centers[i].twist - self.centers[j].twist) if self.use_twist else Float32(0.0)
                var d2 = dx * dx + dz * dz + dt * dt
                var val = det_exp(-d2 * inv_2sig2)
                if i == j:
                    val += Float32(1.0e-5)
                phi[i * n + j] = val

        self.inv_phi = _invert_square_matrix_f64(phi, n)
        self.coeff_matrix = List[Float32](length=n * k_morphs, fill=0.0)
        for i in range(n):
            for m in range(k_morphs):
                var acc = Float32(0.0)
                for j in range(n):
                    var y_jm = self.centers[j].target_weights[m] if m < len(self.centers[j].target_weights) else Float32(0.0)
                    acc += self.inv_phi[i * n + j] * y_jm
                self.coeff_matrix[i * k_morphs + m] = acc

        self.weights = List[Float32](length=k_morphs, fill=0.0)
        self.is_bound = True

    def evaluate(mut self, swing_x: Float32, swing_z: Float32, twist: Float32 = 0.0) -> List[Float32]:
        """Evaluates clamped `[0, 1]` morph weights at `(swing_x, swing_z, twist)`."""
        if not self.is_bound:
            self.finalize()
        var k_morphs = len(self.morph_targets)
        var n = len(self.centers)
        if not self.is_bound or k_morphs == 0 or n == 0:
            return List[Float32]()

        var inv_2sig2 = Float32(1.0) / (Float32(2.0) * self.sigma * self.sigma)
        var k_vec = List[Float32](length=n, fill=0.0)
        for i in range(n):
            var dx = swing_x - self.centers[i].swing_x
            var dz = swing_z - self.centers[i].swing_z
            var dt = (twist - self.centers[i].twist) if self.use_twist else Float32(0.0)
            k_vec[i] = det_exp(-(dx * dx + dz * dz + dt * dt) * inv_2sig2)

        for m in range(k_morphs):
            var w = Float32(0.0)
            for i in range(n):
                w += k_vec[i] * self.coeff_matrix[i * k_morphs + m]
            self.weights[m] = max(Float32(0.0), min(Float32(1.0), w))
        return self.weights.copy()

    def apply(mut self, mut model: NativeModel):
        if self.node < 0 or self.node >= len(model.nodes):
            return
        var sw = swing_twist_y(model.nodes[self.node].rotation, model.nodes[self.node].base_rotation)
        _ = self.evaluate(sw.x, sw.z, sw.twist)
        for m in range(len(self.morph_targets)):
            model.set_morph_weight(self.morph_targets[m], self.weights[m])

    def apply_to_pose(mut self, pose: Pose, model: NativeModel, mut out_morph_weights: List[Float32]):
        if self.node < 0 or self.node >= pose.count() or self.node >= len(model.nodes):
            return
        if len(out_morph_weights) < len(model.morph_names):
            out_morph_weights.resize(len(model.morph_names), Float32(0.0))
        var sw = swing_twist_y(pose.r[self.node], model.nodes[self.node].base_rotation)
        _ = self.evaluate(sw.x, sw.z, sw.twist)
        for m in range(len(self.morph_targets)):
            var idx = self.morph_targets[m]
            if idx >= 0 and idx < len(out_morph_weights):
                out_morph_weights[idx] = self.weights[m]
