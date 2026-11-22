"""Prompt providers and content-keyed beat/transition caching for Studio."""
from std.math import sqrt, exp
from flint.assets.json import JsonDocument
from flint.ml.motionbricks import MbClip, blend_xform, MbVec3
from flint.ml.kimodo import Kimodo
from flint.ml.kimodo_text import KimodoTextEncoder, KimodoTokenizer
from flint.platform.os import (
    env_now,
    file_size,
    file_stamp,
    make_dirs,
    read_bytes,
)
from recipe_io import field, string, integer, relative
from take_files import load_take, save_new
from hymotion import generate_hy
from recipe_io import num
from transition import join_takes, model_path


def hash_bytes(
    data: List[UInt8], var h: UInt64 = 14695981039346656037
) -> UInt64:
    for b in data:
        h = (h ^ UInt64(b)) * 1099511628211
    return h


def hash_text(s: String, h: UInt64 = 14695981039346656037) -> UInt64:
    return hash_bytes(List[UInt8](s.as_bytes()), h)


def model_dir() -> String:
    var dir = env_now("FLINT_KIMODO_DIR")
    return dir if dir else env_now("HOME") + "/.cache/flint/kimodo-soma-rp-v1.1"


def prompt_take(d: JsonDocument, n: Int, file: String) raises -> MbClip:
    var frames = integer(d, n, "frames", 60)
    var steps = integer(d, n, "steps", 20)
    var seed = integer(d, n, "seed", 1)
    if frames < 2 or frames > 300 or steps < 2 or steps > 1000 or seed < 0:
        raise Error("Invalid beat duration, denoising steps or seed")
    var provider = string(d, n, "provider", "kimodo")
    if provider == "hymotion-lite":
        var hy_dir = relative(
            file,
            string(
                d,
                n,
                "modelDirectory",
                env_now("HOME") + "/.cache/flint/hymotion-lite",
            ),
        )
        var cue = string(d, n, "cue")
        var prompt = string(d, n, "prompt")
        if Bool(cue) == Bool(prompt):
            raise Error("HY-Motion Lite needs a prompt or prepared cue")
        return generate_hy(
            load_take(relative(file, string(d, n, "skeleton"))),
            relative(file, cue) if cue else String(),
            hy_dir,
            frames,
            steps,
            UInt32(seed),
            num(d, n, "guidance", 5),
            prompt,
        )
    var dir = model_dir()
    var tokenizer = KimodoTokenizer(dir + "/tokenizer.json")
    var encoder = KimodoTextEncoder(
        dir + "/model_int4_b128_fp32act_s64.onnx_data"
    )
    var prompt = string(d, n, "prompt")
    if not prompt:
        raise Error("A generated beat needs a prompt")
    var embedding = encoder.encode(tokenizer.encode(prompt))
    if provider == "kimodo":
        var generator = Kimodo(dir)
        return generator.sample(embedding, frames, steps, UInt32(seed))
    if provider != "library":
        raise Error("Unknown prompt provider " + provider)
    var lib = field(d, n, "library")
    var count = d.array_size(lib)
    if count < 2:
        raise Error("Library synthesis needs at least two captioned BVHs")
    var scores = List[Float32]()
    for i in range(count):
        var caption = string(d, d.at(lib, i), "caption")
        if not caption:
            raise Error("Missing motion library caption")
        var vec = encoder.encode(tokenizer.encode(caption))
        var a: Float64 = 0
        var b: Float64 = 0
        var c: Float64 = 0
        for k in range(len(vec)):
            a += Float64(embedding[k]) * Float64(vec[k])
            b += Float64(embedding[k]) * Float64(embedding[k])
            c += Float64(vec[k]) * Float64(vec[k])
        scores.append(Float32(a / max(Float64(1e-20), sqrt(b * c))))
    var best = 0
    for i in range(1, count):
        if scores[i] > scores[best]:
            best = i
    var second = 1 if best == 0 else 0
    for i in range(count):
        if i != best and scores[i] > scores[second]:
            second = i
    var left = load_take(relative(file, string(d, d.at(lib, best), "source")))
    var right = load_take(
        relative(file, string(d, d.at(lib, second), "source"))
    )
    var weight = 1 / (
        1 + exp(min(Float32(60), (scores[best] - scores[second]) * 35))
    )
    var result = MbClip()
    result.skeleton = left.skeleton.copy()
    var order = List[Int]()
    for name in left.skeleton.names:
        order.append(right.skeleton.joint_index(name))
    for f in range(frames):
        var phase = Float32(f) / Float32(frames - 1)
        var a = left.sample(phase * left.duration())
        var b = right.sample(phase * right.duration())
        var offset = left.frames[0][0].p - right.frames[0][0].p
        b[0].p = b[0].p + MbVec3(offset.x, 0, offset.z)
        for j in range(len(a)):
            var q = blend_xform(a[j], b[order[j]], weight)
            if j:
                q.p = left.skeleton.reference[j].p
            a[j] = q
        result.frames.append(a^)
    print(
        "Library blend:",
        string(d, d.at(lib, best), "caption"),
        "+",
        string(d, d.at(lib, second), "caption"),
        "weight",
        weight,
    )
    return result^


def cached_beat(
    d: JsonDocument, n: Int, file: String, dir: String
) raises -> String:
    var source = string(d, n, "source")
    if source:
        return relative(file, source)
    var h = hash_text(
        "beat-v1:"
        + string(d, n, "provider", "kimodo")
        + ":"
        + string(d, n, "prompt")
        + ":"
        + String(integer(d, n, "frames", 60))
        + ":"
        + String(integer(d, n, "steps", 20))
        + ":"
        + String(integer(d, n, "seed", 1))
        + ":"
        + model_dir()
    )
    for filename in [
        String("tokenizer.json"),
        String("denoiser.onnx"),
        String("model_int4_b128_fp32act_s64.onnx_data"),
    ]:
        var stamp = file_stamp(model_dir() + "/" + filename)
        h = hash_text(String(stamp[0]) + ":" + String(stamp[1]), h)
    if string(d, n, "provider") == "hymotion-lite":
        var cue = string(d, n, "cue")
        if cue:
            h = hash_bytes(read_bytes(relative(file, cue)), h)
        h = hash_bytes(read_bytes(relative(file, string(d, n, "skeleton"))), h)
        var model = relative(
            file,
            string(
                d,
                n,
                "modelDirectory",
                env_now("HOME") + "/.cache/flint/hymotion-lite",
            ),
        )
        h = hash_text(model + ":" + String(num(d, n, "guidance", 5)), h)
        var assets: List[String] = [
            "head.onnx",
            "tail.onnx",
            "mean.bin",
            "std.bin",
            "null_ctxt_input.bin",
            "null_vtxt_feat.bin",
        ]
        if not cue:
            assets.extend(
                [
                    "clip-text.onnx",
                    "qwen.bf16",
                    "qwen-tokens.json",
                    "clip-tokens.json",
                ]
            )
        for k in range(18):
            assets.append("block-" + String(k) + ".onnx")
        for asset in assets:
            var stamp = file_stamp(model + "/" + asset)
            h = hash_text(String(stamp[0]) + ":" + String(stamp[1]), h)
    var lib = field(d, n, "library")
    for i in range(d.array_size(lib)):
        var item = d.at(lib, i)
        h = hash_text(string(d, item, "caption"), h)
        h = hash_bytes(read_bytes(relative(file, string(d, item, "source"))), h)
    var path = dir + "/beat-" + String(h) + ".bvh"
    if file_size(path) < 1:
        var beat = prompt_take(d, n, file)
        cache_write(beat, path)
    return path^


def cache_write(take: MbClip, path: String) raises:
    try:
        save_new(take, path)
    except err:
        if file_size(path) < 1:
            raise Error(String(err))


def timeline(d: JsonDocument, arr: Int, file: String) raises -> MbClip:
    var dir = relative(file, "build/motion-cache")
    if not make_dirs(dir) or d.array_size(arr) < 1:
        raise Error("Timeline needs beats and a writable project cache")
    var prev_path = cached_beat(d, d.at(arr, 0), file, dir)
    var prev = load_take(prev_path)
    var result = prev.copy()
    for i in range(1, d.array_size(arr)):
        var path = cached_beat(d, d.at(arr, i), file, dir)
        var target = load_take(path)
        var stamp = file_stamp(model_path())
        var h = hash_text(
            "join-v1:"
            + model_path()
            + ":"
            + String(stamp[0])
            + ":"
            + String(stamp[1])
        )
        h = hash_bytes(read_bytes(prev_path), h)
        h = hash_bytes(read_bytes(path), h)
        var cache = dir + "/join-" + String(h) + ".bvh"
        if file_size(cache) < 1:
            cache_write(join_takes(prev, target), cache)
        var pair = load_take(cache)
        var shift = (
            result.frames[result.num_frames() - 1][0].p
            - prev.frames[prev.num_frames() - 1][0].p
        )
        var order = List[Int]()
        for name in result.skeleton.names:
            order.append(pair.skeleton.joint_index(name))
        for f in range(prev.num_frames(), pair.num_frames()):
            var pose = result.skeleton.reference.copy()
            for j in range(len(order)):
                pose[j] = pair.frames[f][order[j]]
            pose[0].p = pose[0].p + shift
            result.frames.append(pose^)
        prev = target^
        prev_path = path.copy()
    return result^
