# SOMA representation and DDIM adapted from NVIDIA Kimodo, Apache-2.0.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Kimodo SOMA RP v1.1 text-conditioned generation in native Mojo, at 30 Hz."""
from std.math import cos, sin, sqrt, log, isfinite
from flint.ml.motionbricks import (
    MbClip,
    MbXform,
    MbVec3,
    MbQuat,
    soma_joint_names,
    quat_from_basis,
)
from flint.ml.onnx import OnnxModel
from flint.ml.tensor import Tensor, DT_BOOL, DT_I64
from flint.platform.os import read_bytes

comptime KIMODO_FEATURES = 369
comptime _PI = 3.141592653589793


def _normal_noise(seed: UInt32, count: Int) -> List[Float32]:
    var state = seed
    var noise = List[Float32](capacity=count + 1)
    while len(noise) < count:
        state = state * 1664525 + 1013904223
        var radius = sqrt(-2 * log((Float64(state) + 0.5) / 4294967296.0))
        state = state * 1664525 + 1013904223
        var phase = 2 * _PI * (Float64(state) + 0.5) / 4294967296.0
        noise.append(Float32(radius * cos(phase)))
        if len(noise) < count:
            noise.append(Float32(radius * sin(phase)))
    return noise^


def _time(step: Int, steps: Int) -> Int:
    var numerator = 999 * step
    var stride = steps - 1
    var q = numerator // stride
    var r = 2 * (numerator % stride)
    return q + Int(r > stride or (r == stride and q % 2 == 1))


def _alphas() -> List[Float32]:
    var a = List[Float32](capacity=1000)
    var product = Float64(1)
    for i in range(1000):
        var p = cos((Float64(i) / 1000 + 0.008) / 1.008 * _PI / 2)
        var q = cos((Float64(i + 1) / 1000 + 0.008) / 1.008 * _PI / 2)
        var beta = Float32(min(Float64(0.999), 1 - (q * q) / (p * p)))
        product *= Float64(Float32(1) - beta)
        a.append(Float32(product))
    return a^


def _numbers(path: String, count: Int, npy: Bool) raises -> List[Float32]:
    var data = read_bytes(path)
    var off = 0
    if npy:
        if (
            len(data) < 10
            or data[0] != 147
            or String(unsafe_from_utf8=data[1:6]) != "NUMPY"
            or data[6] != 1
        ):
            raise Error("Invalid NumPy statistics: " + path)
        off = 10 + Int(data[8]) + 256 * Int(data[9])
        if off > len(data):
            raise Error("Truncated statistics: " + path)
        var header = String(unsafe_from_utf8=data[10:off])
        if "'<f8'" not in header or "(" + String(count) + ",)" not in header:
            raise Error("Expected little-endian float64 vector: " + path)
    if len(data) - off != count * 8:
        raise Error("Incorrect SOMA asset size: " + path)
    var out = List[Float32](capacity=count)
    for i in range(count):
        var p = (
            data.unsafe_ptr()
            .unsafe_offset(off + i * 8)
            .unsafe_bitcast[Float64]()
        )
        var v = Float32(p.unsafe_load[alignment=1]())
        if not isfinite(v):
            raise Error("Invalid SOMA asset value: " + path)
        out.append(v)
    return out^


def _vec(data: List[Float32], i: Int) -> MbVec3:
    return MbVec3(data[i], data[i + 1], data[i + 2])


def _cross(a: MbVec3, b: MbVec3) -> MbVec3:
    return MbVec3(
        a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x
    )


struct Kimodo(Movable):
    var denoiser: OnnxModel
    var mean: List[Float32]
    var scale: List[Float32]
    var rest: List[Float32]

    def __init__(out self, dir: String, gpu: Bool = True) raises:
        self.mean = List[Float32]()
        self.scale = List[Float32]()
        for group in ["global_root", "body"]:
            var count = 5 if group == "global_root" else 364
            self.mean.extend(
                _numbers(dir + "/" + group + "-mean.npy", count, True)
            )
            var std = _numbers(dir + "/" + group + "-std.npy", count, True)
            for s in std:
                self.scale.append(sqrt(s * s + Float32(1e-5)))
        self.rest = _numbers(dir + "/joints.f64", 90, False)
        self.denoiser = OnnxModel.load(dir + "/denoiser.onnx")
        if gpu:
            _ = self.denoiser.use_gpu()

    def sample(
        mut self,
        embedding: List[Float32],
        frames: Int,
        steps: Int = 20,
        seed: UInt32 = 1,
        guidance: Float32 = 2,
    ) raises -> MbClip:
        if (
            frames < 2
            or frames > 300
            or steps < 2
            or steps > 1000
            or len(embedding) != 4096
            or not isfinite(guidance)
        ):
            raise Error(
                "Kimodo needs 2-300 frames, 2-1000 steps and one 4096-value"
                " text embedding"
            )
        var n = frames * KIMODO_FEATURES
        var noise = _normal_noise(seed, n)
        var alphas = _alphas()
        var text = Tensor.zeros([3, 50, 4096])
        for i in range(4096):
            if not isfinite(embedding[i]):
                raise Error("Nonfinite text embedding")
            text.f[i] = embedding[i]
        var shape: List[Int] = [3, frames, KIMODO_FEATURES]
        for idx in range(steps - 1, -1, -1):
            var t = _time(idx, steps)
            var batch = Tensor.zeros(shape.copy())
            for branch in range(3):
                for i in range(n):
                    batch.f[branch * n + i] = noise[i]
            self.denoiser.set_input("motion", batch^)
            self.denoiser.set_input(
                "motion_valid",
                Tensor.ints(
                    DT_BOOL, [3, frames], List[Int](length=3 * frames, fill=1)
                ),
            )
            self.denoiser.set_input("text_embedding", text.copy())
            self.denoiser.set_input(
                "timestep", Tensor.ints(DT_I64, [3], [t, t, t])
            )
            self.denoiser.set_input("first_heading_angle", Tensor.zeros([3]))
            self.denoiser.set_input(
                "constraint_mask",
                Tensor.ints(
                    DT_BOOL, shape.copy(), List[Int](length=3 * n, fill=0)
                ),
            )
            self.denoiser.set_input(
                "observed_motion", Tensor.zeros(shape.copy())
            )
            self.denoiser.run()
            var pred = self.denoiser.output("predicted_clean_motion")
            if pred.shape != shape or pred.dtype != 1:
                raise Error("Incompatible Kimodo denoiser output")
            var prev = alphas[_time(idx - 1, steps)] if idx > 0 else Float32(1)
            var a = alphas[t]
            var gain = sqrt((1 - prev) / (1 - a))
            for i in range(n):
                var uncond = pred.fget(2 * n + i)
                var clean = uncond + guidance * (pred.fget(i) - uncond)
                noise[i] = sqrt(prev) * clean + gain * (
                    noise[i] - sqrt(a) * clean
                )
        for i in range(n):
            var k = i % KIMODO_FEATURES
            noise[i] = noise[i] * self.scale[k] + self.mean[k]
            if not isfinite(noise[i]):
                raise Error("Nonfinite Kimodo motion")
        return self._decode(noise, frames)

    def _decode(self, data: List[Float32], frames: Int) raises -> MbClip:
        if frames < 1 or frames > 300 or len(data) != frames * KIMODO_FEATURES:
            raise Error("Invalid denormalized SOMA feature buffer")
        var clip = MbClip()
        clip.skeleton.names = soma_joint_names()
        clip.skeleton.parents = [
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            6,
            6,
            3,
            10,
            11,
            12,
            13,
            13,
            3,
            16,
            17,
            18,
            19,
            19,
            0,
            22,
            23,
            24,
            0,
            26,
            27,
            28,
        ]
        for j in range(30):
            var p = _vec(self.rest, 3 * j)
            if j > 0:
                p = p - _vec(self.rest, 3 * clip.skeleton.parents[j])
            clip.skeleton.reference.append(MbXform(p, MbQuat.identity()))
        for f in range(frames):
            var base = f * KIMODO_FEATURES
            var pose = clip.skeleton.reference.copy()
            var globals = List[MbQuat](capacity=30)
            pose[0].p = MbVec3(
                data[base + 5] + data[base],
                data[base + 6],
                data[base + 7] + data[base + 2],
            )
            for j in range(30):
                var off = base + 95 + j * 6
                var x = _vec(data, off).normalized()
                var z = _cross(x, _vec(data, off + 3)).normalized()
                if x.length() < 0.99 or z.length() < 0.99:
                    raise Error("Degenerate predicted SOMA rotation")
                var q = quat_from_basis(x, _cross(z, x), z).normalized()
                globals.append(q)
                pose[j].q = (
                    q if j
                    == 0 else globals[clip.skeleton.parents[j]].inverse() * q
                )
            clip.frames.append(pose^)
        return clip^
