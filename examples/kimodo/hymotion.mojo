"""Run HY-Motion Lite flow matching for an offline-encoded text cue."""
from hymotion_text import encode_hy
from std.math import isfinite
from flint.ml.onnx import OnnxModel
from flint.ml.tensor import Tensor, DT_BOOL
from flint.ml.kimodo import _normal_noise
from flint.ml.motionbricks import (
    MbClip,
    MbXform,
    MbVec3,
    quat_from_basis,
    world_pose,
)
from flint.core.math import Vec3
from flint.platform.os import read_bytes, env_now
from flint.assets.glb_native import little_u32
from flint.anim.take_constraints import mb


def values(data: List[UInt8], offset: Int, count: Int) raises -> List[Float32]:
    if offset < 0 or count < 0 or offset + 4 * count > len(data):
        raise Error("Truncated HY-Motion data")
    var out = List[Float32](capacity=count)
    for i in range(count):
        var x = (
            data.unsafe_ptr()
            .unsafe_offset(offset + 4 * i)
            .unsafe_bitcast[Float32]()
            .unsafe_load[alignment=1]()
        )
        if not isfinite(x):
            raise Error("Non-finite HY-Motion value")
        out.append(x)
    return out^


def weights(path: String, count: Int) raises -> List[Float32]:
    var bytes = read_bytes(path)
    if len(bytes) != count * 4:
        raise Error("Unexpected size for " + path)
    return values(bytes, 0, count)


def module(dir: String, name: String) raises -> OnnxModel:
    var model = OnnxModel.load(dir + "/" + name + ".onnx", 4)
    if env_now("FLINT_HYMOTION_CPU") != "1":
        _ = model.use_gpu()
    return model^


def generate_hy(
    reference: MbClip,
    cue: String,
    dir: String,
    frames: Int,
    steps: Int,
    seed: UInt32,
    guidance: Float32,
    prompt: String,
) raises -> MbClip:
    if (
        frames < 2
        or frames > 60
        or steps < 1
        or steps > 100
        or guidance < 0
        or guidance > 20
    ):
        raise Error(
            "HY-Motion Lite needs 2–60 frames, 1–100 Euler steps and guidance"
            " in 0–20"
        )
    var pooled_text: List[Float32]
    var text: List[Float32]
    if cue:
        var raw = read_bytes(cue)
        if len(raw) < 8 or little_u32(raw, 0) != 0x31435948:
            raise Error("Expected HYC1 cue at " + cue)
        var tokens = little_u32(raw, 4)
        if (
            tokens < 1
            or tokens > 32
            or len(raw) != 8 + 4 * (768 + tokens * 4096)
        ):
            raise Error("Invalid HY-Motion text cue")
        pooled_text = values(raw, 8, 768)
        text = values(raw, 8 + 768 * 4, tokens * 4096)
    else:
        var encoding = encode_hy(prompt, dir)
        pooled_text = encoding[0].copy()
        text = encoding[1].copy()
    var count = len(text) // 4096
    var v = weights(dir + "/null_vtxt_feat.bin", 768)
    v.extend(pooled_text^)
    var pooled = Tensor.floats([2, 1, 768], v^)
    var ctx = Tensor.zeros([2, 32, 4096])
    var silent = weights(dir + "/null_ctxt_input.bin", 4096)
    var valid = List[Int](length=64, fill=0)
    var bias = Tensor.zeros([2, 1, 1, 92])
    for i in range(32):
        for j in range(4096):
            ctx.f[i * 4096 + j] = silent[j]
            if i < count:
                ctx.f[(32 + i) * 4096 + j] = text[i * 4096 + j]
        valid[i] = Int(i < count)
        valid[32 + i] = valid[i]
    for b in range(2):
        for k in range(92):
            if (k >= frames and k < 60) or k >= 60 + count:
                bias.f[b * 92 + k] = -1e9
    var head = module(dir, "head")
    var blocks = List[OnnxModel]()
    for k in range(18):
        blocks.append(module(dir, "block-" + String(k)))
    var tail = module(dir, "tail")
    var x = _normal_noise(seed, 60 * 201)
    var valid_mask = Tensor.ints(DT_BOOL, [2, 32], valid^)
    for step in range(steps):
        var xx = x.copy()
        xx.extend(x.copy())
        var t = Float32(step) / Float32(steps)
        head.set_input("time", Tensor.floats([2], [t, t]))
        head.set_input("x", Tensor.floats([2, 60, 201], xx^))
        head.set_input("context", ctx.copy())
        head.set_input("pooled", pooled.copy())
        head.set_input("valid", valid_mask.copy())
        head.run()
        var motion = head.output("motion")
        var z = head.output("text")
        var adapter = head.output("adapter")
        for k in range(len(blocks)):
            ref block = blocks[k]
            block.set_input("x", motion^)
            block.set_input("z", z^)
            block.set_input("a", adapter.copy())
            block.set_input("penalty", bias.copy())
            block.run()
            motion = block.output("motion")
            z = block.output("text")
        tail.set_input("x", motion^)
        tail.set_input("a", adapter^)
        tail.run()
        var flow = tail.output("flow")
        if flow.shape != [2, 60, 201]:
            raise Error("Unexpected HY-Motion flow shape")
        for k in range(len(x)):
            var baseline = flow.fget(k)
            x[k] += (
                baseline + guidance * (flow.fget(k + len(x)) - baseline)
            ) / Float32(steps)
            if not isfinite(x[k]):
                raise Error("Non-finite HY-Motion flow")
    var mean = weights(dir + "/mean.bin", 201)
    var std = weights(dir + "/std.bin", 201)
    for k in range(len(x)):
        x[k] = x[k] * std[k % 201] + mean[k % 201]
    var clip = MbClip()
    clip.sample_rate = 30
    clip.skeleton = reference.skeleton.copy()
    var names: List[String] = [
        "Hips",
        "LeftLeg",
        "RightLeg",
        "Spine1",
        "LeftShin",
        "RightShin",
        "Spine2",
        "LeftFoot",
        "RightFoot",
        "Chest",
        "LeftToeBase",
        "RightToeBase",
        "Neck1",
        "LeftShoulder",
        "RightShoulder",
        "Head",
        "LeftArm",
        "RightArm",
        "LeftForeArm",
        "RightForeArm",
        "LeftHand",
        "RightHand",
    ]
    var joints = List[Int]()
    for name in names:
        joints.append(clip.skeleton.joint_index(name))
    var floor: Float32 = 1e10
    for f in range(frames):
        var pose = clip.skeleton.reference.copy()
        var base = f * 201
        pose[0].p = MbVec3(x[base], x[base + 1], x[base + 2])
        for j in range(len(joints)):
            var k = base + 3 + 6 * j
            var a = Vec3(x[k], x[k + 2], x[k + 4]).normalized()
            var b = Vec3(x[k + 1], x[k + 3], x[k + 5])
            b = (b - a * a.dot(b)).normalized()
            pose[joints[j]].q = quat_from_basis(mb(a), mb(b), mb(a.cross(b)))
        var w = world_pose(clip.skeleton, pose, MbXform.identity())
        floor = min(floor, min(w[joints[10]].p.y, w[joints[11]].p.y))
        clip.frames.append(pose^)
    for f in range(frames):
        clip.frames[f][0].p.y -= floor
    return clip^
