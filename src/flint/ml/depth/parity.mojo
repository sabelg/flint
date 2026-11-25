"""Parity checks of `flint.ml.depth` against `tools/depth/reference.py` dumps.

`check_parity(model, dir)` feeds the reference's own network input (`input.f32`) to the Mojo
model and compares every output the dump has; `check_preprocess(model, dir, image)` compares the
Mojo resize / normalisation of the original image with that input.
"""

from std.math import sqrt
from std.ffi import external_call
from flint.platform.os import read_bytes, write_bytes
from flint.assets.json import parse_json, JsonDocument
from flint.ml.depth.model import DepthModel, DepthResult, FAMILY_DA2, FAMILY_DA3, FAMILY_MOGE
from flint.ml.depth.image import load_rgb, preprocess_da2, preprocess_da3, preprocess_moge
from flint.ml.depth.moge import base_grid


def read_f32(path: String) raises -> List[Float32]:
    var b = read_bytes(path)
    if len(b) == 0 or len(b) % 4 != 0:
        raise Error("depth: cannot read float32 file " + path)
    var n = len(b) // 4
    var out = List[Float32](length=n, fill=0.0)
    _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), Int(b.unsafe_ptr()), n * 4)
    _ = len(b)  # keep the bytes alive past the raw copy
    return out^


def has_file(path: String) -> Bool:
    return len(read_bytes(path)) > 0


struct Diff(Copyable, Movable):
    var name: String
    var n: Int
    var max_abs: Float64
    var mean_abs: Float64
    var max_ref: Float64
    var rel: Float64          # max_abs / max |ref|
    var mean_rel: Float64     # mean |a - b| / |b| over |b| > 1e-6
    var corr: Float64

    def __init__(out self, name: String):
        self.name = name
        self.n = 0
        self.max_abs = 0
        self.mean_abs = 0
        self.max_ref = 0
        self.rel = 0
        self.mean_rel = 0
        self.corr = 0

    def line(self) -> String:
        return (self.name + ": n=" + String(self.n) + " max|d|=" + String(Float32(self.max_abs)) + " mean|d|="
            + String(Float32(self.mean_abs)) + " max|d|/max|ref|=" + String(Float32(self.rel)) + " mean rel="
            + String(Float32(self.mean_rel)) + " corr=" + String(self.corr))


def compare(name: String, a: List[Float32], b: List[Float32]) raises -> Diff:
    """Statistics of a vs reference b over the finite entries of both."""
    if len(a) != len(b):
        raise Error("depth parity: " + name + " has " + String(len(a)) + " values, reference " + String(len(b)))
    var d = Diff(name)
    var sa = 0.0
    var sb = 0.0
    var saa = 0.0
    var sbb = 0.0
    var sab = 0.0
    var nrel = 0
    for i in range(len(a)):
        var x = Float64(a[i])
        var y = Float64(b[i])
        if not (abs(x) < 1e30 and abs(y) < 1e30):
            continue
        var e = abs(x - y)
        d.n += 1
        d.max_abs = max(d.max_abs, e)
        d.mean_abs += e
        d.max_ref = max(d.max_ref, abs(y))
        if abs(y) > 1e-6:
            d.mean_rel += e / abs(y)
            nrel += 1
        sa += x
        sb += y
        saa += x * x
        sbb += y * y
        sab += x * y
    if d.n > 0:
        var n = Float64(d.n)
        d.mean_abs /= n
        var va = saa / n - (sa / n) * (sa / n)
        var vb = sbb / n - (sb / n) * (sb / n)
        var cv = sab / n - (sa / n) * (sb / n)
        d.corr = cv / sqrt(max(va * vb, 1e-300))
    if nrel > 0:
        d.mean_rel /= Float64(nrel)
    d.rel = d.max_abs / max(d.max_ref, 1e-30)
    return d^


def _f64s(xs: List[Float64]) -> List[Float32]:
    var o = List[Float32](capacity=len(xs))
    for x in xs:
        o.append(Float32(x))
    return o^


def check_parity(mut m: DepthModel, dir: String) raises -> List[Diff]:
    var doc = parse_json(read_bytes(dir + "/meta.json"))
    if not doc.valid:
        raise Error("depth parity: cannot read " + dir + "/meta.json")
    var S = doc.integer(doc.object_get(doc.root, "S"), 1)
    var H = doc.integer(doc.object_get(doc.root, "H"), 0)
    var W = doc.integer(doc.object_get(doc.root, "W"), 0)
    var oh = doc.integer(doc.object_get(doc.root, "orig_h"), 0)
    var ow = doc.integer(doc.object_get(doc.root, "orig_w"), 0)
    var x = read_f32(dir + "/input.f32")
    var out = List[Diff]()
    if m.family == FAMILY_MOGE:
        var r = m.run_moge(x, H // 14, W // 14, oh, ow)
        out.append(compare("points (forward)", r.raw, read_f32(dir + "/points_raw.f32")))
        if has_file(dir + "/normal.f32"):
            out.append(compare("normal (forward)", r.normal_raw, read_f32(dir + "/normal.f32")))
        if has_file(dir + "/mask.f32"):
            out.append(compare("mask (forward)", r.prob, read_f32(dir + "/mask.f32")))
        out.append(compare("depth (infer)", r.depth, read_f32(dir + "/depth.f32")))
        out.append(compare("points (infer)", r.points, read_f32(dir + "/points.f32")))
        var k = read_f32(dir + "/intrinsics.f32")
        var mk: List[Float64] = [r.intrinsics[0] / Float64(ow), r.intrinsics[4] / Float64(oh)]
        var rk: List[Float32] = [k[0], k[4]]
        out.append(compare("fx, fy (normalised)", _f64s(mk), rk))
        var ms: List[Float32] = [Float32(r.scale)]
        var rs: List[Float32] = [Float32(doc.number(doc.object_get(doc.root, "metric_scale"), 1.0))]
        out.append(compare("metric scale", ms, rs))
        return out^
    if m.family == FAMILY_DA2:
        var r = m.run_da(x, S, H, W, oh, ow, 0.0)
        var raw = List[Float32](length=H * W, fill=0.0)
        for i in range(H * W):
            raw[i] = max(r.raw[i], Float32(0))
        out.append(compare("disparity (network size)", raw, read_f32(dir + "/raw.f32")))
        out.append(compare("disparity (image size)", r.depth, read_f32(dir + "/depth.f32")))
        return out^
    var r = m.run_da(x, S, H, W, H, W, 0.0)
    if S > 1:
        var rv = Diff("reference view " + String(r.ref_view))
        out.append(rv^)
    if has_file(dir + "/depth_head.f32"):
        out.append(compare("depth (head)", r.depth_head, read_f32(dir + "/depth_head.f32")))
    out.append(compare("depth", r.depth, read_f32(dir + "/depth.f32")))
    if has_file(dir + "/conf.f32") and len(r.conf) > 0:
        out.append(compare("confidence", r.conf, read_f32(dir + "/conf.f32")))
    if has_file(dir + "/sky.f32") and len(r.sky) > 0:
        out.append(compare("sky", r.sky, read_f32(dir + "/sky.f32")))
    if has_file(dir + "/extrinsics.f32") and r.has_camera:
        out.append(compare("extrinsics", _f64s(r.extrinsics), read_f32(dir + "/extrinsics.f32")))
        out.append(compare("intrinsics", _f64s(r.intrinsics), read_f32(dir + "/intrinsics.f32")))
    return out^


def check_preprocess(mut m: DepthModel, dir: String, image: String) raises -> Diff:
    """The Mojo preprocessing of `image` vs the reference network input."""
    var ref_x = read_f32(dir + "/input.f32")
    var img = load_rgb(image)
    if m.family == FAMILY_MOGE:
        var g = base_grid(m.moge.num_tokens(), img.h, img.w)
        var p = preprocess_moge(img, g[0], g[1], m.moge.mean, m.moge.std)
        return compare("input (MoGe antialiased bilinear)", p.data, ref_x)
    if m.family == FAMILY_DA2:
        var p = preprocess_da2(img, m.input_size)
        return compare("input (DA2 cv2 cubic)", p.data, ref_x)
    var p = preprocess_da3(img, m.input_size)
    var first = List[Float32](capacity=len(p.data))
    for i in range(min(len(p.data), len(ref_x))):
        first.append(ref_x[i])
    return compare("input (DA3 cv2 area / cubic, view 0)", p.data, first)
