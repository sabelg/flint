"""GEM-X in pure Mojo (`flint.ml.gemx`): monocular video to whole-body motion on the 77-joint
SOMA skeleton that `flint.ml.motionbricks` animates.

NVIDIA GEM-X (code Apache-2.0, weights NVIDIA Open Model License) runs three networks, which
`tools/gemx/export.py` converts to plain float32 ONNX graphs for `OnnxModel` (GPU when present):

  - `gemx_vitpose.onnx`: native77 ViTPose (DINOv3 ViT-H/16+ and a deconvolution head) turns a
    256x192 person crop into 77 heatmaps; Flint averages them with the mirrored crop's, as GEM-X
    does, and decodes full-image keypoints.
  - `gemx_pose.onnx`: the Body Pose encoder (DINOv3 ViT-H/16+ and a 6-layer cross-attention
    decoder) turns a 256x256 crop plus its CLIFF box vector into one 1024-D pose token.
  - `gemx_motion.onnx`: the Motion transformer reads the whole clip (keypoints, boxes, camera
    rotation, pose tokens) and predicts the SOMA parameters of every frame.

The crops reproduce OpenCV 5's `getAffineTransform` + `warpAffine(INTER_LINEAR)` bit for bit, so
Flint's network inputs equal the PyTorch reference's. The person is given by one box per frame
(from any detector) or tracked from a first-frame box with the 77 keypoints (`Gemx.track`):
GEM-X's own YOLOX person detector is not ported, because its weights were trained on the
non-commercial Human-Art dataset. The camera is static or its per-frame rotations are given;
GEM-X's SIFT background camera estimation is not ported.

`GemxMotion` holds the decoded predictions; `gemx_clip` poses a SOMA `MbSkeleton` with them
(the embedded neutral SOMA-X skeleton, or e.g. a MotionBricks BVH's), `write_bvh` saves that
clip, and `clip_to_native_animation` turns it into a `NativeAnimation` for a loaded model. Video decoding stays outside Flint, e.g.
`ffmpeg -i clip.mp4 -vf fps=30 frames/%05d.png`.
"""

from std.math import sqrt, exp, atan2, asin, acos, floor, fma, cos, sin
from std.time import perf_counter_ns
from flint.platform.os import list_directory, read_bytes, write_bytes
from flint.assets.image import decode_png
from flint.assets.model import NativeAnimation, NativeChannel
from flint.ml.tensor import Tensor
from flint.ml.onnx import OnnxModel
from flint.ml.motionbricks import MbSkeleton, MbClip, MbXform, MbVec3, MbQuat, quat_from_basis

comptime GX_JOINTS: Int = 77          # native77 keypoints = SOMA joints 1..77 (Hips first)
comptime GX_BODY: Int = 76            # posed joints after Hips
comptime GX_MOTION_DIM: Int = 655
comptime GX_TOKEN: Int = 1024
comptime GX_MAX_LEN: Int = 120        # the Motion transformer's attention window
comptime GX_HEAT_W: Int = 48
comptime GX_HEAT_H: Int = 64


# ---------------------------------------------------------------- frames


struct GxFrame(Copyable, Movable):
    """One RGB8 video frame (row major, 3 bytes per pixel)."""
    var width: Int
    var height: Int
    var rgb: List[UInt8]

    def __init__(out self, width: Int, height: Int):
        self.width = width
        self.height = height
        self.rgb = List[UInt8](length=width * height * 3, fill=0)


def frame_from_png(path: String) raises -> GxFrame:
    var img = decode_png(read_bytes(path), 1 << 15)
    if not img.valid:
        raise Error("gemx: cannot decode PNG " + path)
    var f = GxFrame(img.width, img.height)
    for k in range(img.width * img.height):
        var p = img.pixels[k]
        f.rgb[3 * k] = UInt8((p >> 16) & 255)
        f.rgb[3 * k + 1] = UInt8((p >> 8) & 255)
        f.rgb[3 * k + 2] = UInt8(p & 255)
    return f^


def png_frame_paths(dir: String) -> List[String]:
    """The directory's `.png` files in name order (00000.png, 00001.png, ...)."""
    var names = List[String]()
    for n in list_directory(dir):
        if n.endswith(".png"):
            names.append(n)
    for i in range(1, len(names)):          # insertion sort: frame lists are short
        var k = i
        while k > 0 and names[k] < names[k - 1]:
            var t = names[k]
            names[k] = names[k - 1]
            names[k - 1] = t
            k -= 1
    var out = List[String]()
    for n in names:
        out.append(dir + "/" + n)
    return out^


def load_frames(dir: String, limit: Int = -1) raises -> List[GxFrame]:
    var paths = png_frame_paths(dir)
    var n = len(paths) if limit < 0 else min(limit, len(paths))
    if n == 0:
        raise Error("gemx: no .png frames in " + dir)
    var out = List[GxFrame]()
    for i in range(n):
        out.append(frame_from_png(paths[i]))
        if out[i].width != out[0].width or out[i].height != out[0].height:
            raise Error("gemx: frame " + paths[i] + " has another size")
    return out^


# ---------------------------------------------------------------- OpenCV-exact crops


def affine_from_points(src: List[Float32], dst: List[Float32]) -> List[Float64]:
    """OpenCV `getAffineTransform` of three point pairs (x0 y0 x1 y1 x2 y2): the 6x6 system it
    builds, solved with its partial-pivoting `LUImpl` in double. Returns the 2x3 matrix."""
    var a = List[Float64](length=36, fill=0.0)
    var b = List[Float64](length=6, fill=0.0)
    for i in range(3):
        var j = i * 12
        var k = i * 12 + 6
        a[j] = Float64(src[2 * i])
        a[k + 3] = Float64(src[2 * i])
        a[j + 1] = Float64(src[2 * i + 1])
        a[k + 4] = Float64(src[2 * i + 1])
        a[j + 2] = 1.0
        a[k + 5] = 1.0
        b[2 * i] = Float64(dst[2 * i])
        b[2 * i + 1] = Float64(dst[2 * i + 1])
    comptime M = 6
    for i in range(M):
        var k = i
        for j in range(i + 1, M):
            if abs(a[j * M + i]) > abs(a[k * M + i]):
                k = j
        if k != i:
            for j in range(i, M):
                var t = a[i * M + j]
                a[i * M + j] = a[k * M + j]
                a[k * M + j] = t
            var tb = b[i]
            b[i] = b[k]
            b[k] = tb
        var d = -1.0 / a[i * M + i]
        for j in range(i + 1, M):
            var alpha = a[j * M + i] * d
            for c in range(i + 1, M):
                a[j * M + c] += alpha * a[i * M + c]
            b[j] += alpha * b[i]
    var i = M - 1
    while i >= 0:
        var s = b[i]
        for c in range(i + 1, M):
            s -= a[i * M + c] * b[c]
        b[i] = s / a[i * M + i]
        i -= 1
    return b^


@always_inline
def _round_u8(v: Float32) -> UInt8:
    """`saturate_cast<uchar>(float)`: round half to even, clamp to 0..255."""
    var x = Float64(v)
    var r = floor(x + 0.5)
    if r - x == 0.5 and Int(r) % 2 != 0:
        r -= 1.0
    if r < 0.0:
        return 0
    if r > 255.0:
        return 255
    return UInt8(Int(r))


def warp_affine(img: GxFrame, m: List[Float64], width: Int, height: Int, x0: Int = 0, x1: Int = -1) -> List[UInt8]:
    """OpenCV 5 `warpAffine(INTER_LINEAR, BORDER_CONSTANT 0)` of an RGB8 image into a
    `width` x `height` crop, keeping output columns `x0..x1-1` (all by default). Reproduces the
    float32 path of its NEON kernel: the inverted matrix in float32, a per-row offset, fused
    multiply-adds for the per-pixel source position and the bilinear blend."""
    var xe = width if x1 < 0 else x1
    var ow = xe - x0
    var dd = m[0] * m[4] - m[1] * m[3]
    dd = 1.0 / dd if dd != 0.0 else 0.0
    var a11 = m[4] * dd
    var a22 = m[0] * dd
    var a12 = -m[1] * dd
    var a21 = -m[3] * dd
    var b1 = -a11 * m[2] - a12 * m[5]
    var b2 = -a21 * m[2] - a22 * m[5]
    var m0 = Float32(a11)
    var m1 = Float32(a12)
    var m2 = Float32(b1)
    var m3 = Float32(a21)
    var m4 = Float32(a22)
    var m5 = Float32(b2)
    var out = List[UInt8](length=ow * height * 3, fill=0)
    var w = img.width
    var h = img.height
    for y in range(height):
        var fy = Float32(y)
        var rx = fy * m1
        rx = rx + m2
        var ry = fy * m4
        ry = ry + m5
        for x in range(x0, xe):
            var fx = Float32(x)
            var sx = fma(m0, fx, rx)
            var sy = fma(m3, fx, ry)
            var fix = floor(sx)
            var fiy = floor(sy)
            var ax = sx - fix
            var ay = sy - fiy
            var ix = Int(fix)
            var iy = Int(fiy)
            if ix < -1 or iy < -1 or ix >= w or iy >= h:
                continue
            var o = ((y * ow) + (x - x0)) * 3
            for c in range(3):
                var p00: Float32 = 0
                var p01: Float32 = 0
                var p10: Float32 = 0
                var p11: Float32 = 0
                if iy >= 0:
                    if ix >= 0:
                        p00 = Float32(img.rgb[(iy * w + ix) * 3 + c])
                    if ix + 1 < w:
                        p01 = Float32(img.rgb[(iy * w + ix + 1) * 3 + c])
                if iy + 1 < h:
                    if ix >= 0:
                        p10 = Float32(img.rgb[((iy + 1) * w + ix) * 3 + c])
                    if ix + 1 < w:
                        p11 = Float32(img.rgb[((iy + 1) * w + ix + 1) * 3 + c])
                var v0 = fma(ax, p01 - p00, p00)
                var v1 = fma(ax, p11 - p10, p10)
                out[o + c] = _round_u8(fma(ay, v1 - v0, v0))
    return out^


def _normalize_chw(crop: List[UInt8], w: Int, h: Int, mut dst: List[Float32], offset: Int, mirror: Bool):
    """ImageNet-normalized CHW float32 (`(u8 / 255 - mean) / std`), optionally mirrored."""
    var mean = SIMD[DType.float32, 4](0.485, 0.456, 0.406, 0)
    var sd = SIMD[DType.float32, 4](0.229, 0.224, 0.225, 1)
    for c in range(3):
        for y in range(h):
            for x in range(w):
                var sx = w - 1 - x if mirror else x
                var v = Float32(crop[(y * w + sx) * 3 + c]) / 255.0
                dst[offset + (c * h + y) * w + x] = (v - mean[c]) / sd[c]


def vitpose_crop(img: GxFrame, cx: Float32, cy: Float32, size: Float32) -> List[UInt8]:
    """GEM-X `crop_native77_inputs`: the 255-edge affine of the square box onto 256x256, keeping
    columns 32..223 (a 192x256 RGB8 crop)."""
    var half = Float64(size) / 2.0
    var src: List[Float32] = [Float32(Float64(cx) - half), Float32(Float64(cy) - half),
                              Float32(Float64(cx) + half), Float32(Float64(cy) - half), cx, cy]
    var dst: List[Float32] = [0.0, 0.0, 255.0, 0.0, 127.5, 127.5]
    return warp_affine(img, affine_from_points(src, dst), 256, 256, 32, 224)


def pose_crop(img: GxFrame, cx: Float32, cy: Float32, size: Float32) -> List[UInt8]:
    """GEM-X `crop_rgb(crop_convention="motion_training")`: a 256x256 RGB8 crop."""
    var half = Float64(size) / 2.0
    var src: List[Float32] = [Float32(Float64(cx) - half), Float32(Float64(cy) - half),
                              Float32(Float64(cx) + half), Float32(Float64(cy) - half),
                              cx, Float32(Float64(cy) + half)]
    var dst: List[Float32] = [0.0, 0.0, 256.0, 0.0, 128.0, 256.0]
    return warp_affine(img, affine_from_points(src, dst), 256, 256)


# ---------------------------------------------------------------- boxes


def square_crop(x0: Float32, y0: Float32, x1: Float32, y1: Float32) -> SIMD[DType.float32, 4]:
    """GEM-X `square_crops(enlarge=1.2)`: centre and side `1.2 * max(h, w / 0.75)`."""
    var w = x1 - x0
    var h = y1 - y0
    var s = max(h, w / Float32(0.75)) * Float32(1.2)
    return SIMD[DType.float32, 4]((x0 + x1) / 2, (y0 + y1) / 2, s, 0)


def boxes_to_crops(xyxy: List[Float32], width: Int, height: Int) -> List[Float32]:
    """GEM-X track post-processing of one box per frame (x0 y0 x1 y1): clamp to the image, a
    5-frame replicate-padded moving average (skipped for 5 frames or fewer), then the 1.2x square
    crop. Returns (cx, cy, size) per frame."""
    var t = len(xyxy) // 4
    var b = xyxy.copy()
    for i in range(t):
        b[4 * i] = min(max(b[4 * i], 0.0), Float32(width - 1))
        b[4 * i + 2] = min(max(b[4 * i + 2], 0.0), Float32(width - 1))
        b[4 * i + 1] = min(max(b[4 * i + 1], 0.0), Float32(height - 1))
        b[4 * i + 3] = min(max(b[4 * i + 3], 0.0), Float32(height - 1))
    var s = b.copy()
    if t > 5:
        var k = Float32(1) / 5
        for i in range(t):
            for c in range(4):
                var acc: Float32 = 0
                for d in range(-2, 3):
                    var j = min(max(i + d, 0), t - 1)
                    acc += b[4 * j + c] * k
                s[4 * i + c] = acc
    var out = List[Float32](length=3 * t, fill=0.0)
    for i in range(t):
        var q = square_crop(s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3])
        out[3 * i] = q[0]
        out[3 * i + 1] = q[1]
        out[3 * i + 2] = q[2]
    return out^


def keypoint_box(kp: List[Float32], offset: Int, width: Int, height: Int) -> List[Float32]:
    """The tracker's next person box from 77 keypoints (x, y, conf) at `offset`: the extent of
    joints with confidence above 0.3, padded by 15% on every side and clamped to the image.
    Empty when fewer than 4 joints are confident."""
    var n = 0
    var x0: Float32 = 0
    var y0: Float32 = 0
    var x1: Float32 = 0
    var y1: Float32 = 0
    for j in range(GX_JOINTS):
        var o = offset + 3 * j
        if kp[o + 2] > 0.3:
            if n == 0:
                x0 = kp[o]
                x1 = kp[o]
                y0 = kp[o + 1]
                y1 = kp[o + 1]
            else:
                x0 = min(x0, kp[o])
                x1 = max(x1, kp[o])
                y0 = min(y0, kp[o + 1])
                y1 = max(y1, kp[o + 1])
            n += 1
    if n < 4:
        return List[Float32]()
    var pw = (x1 - x0) * Float32(0.15)
    var ph = (y1 - y0) * Float32(0.15)
    return [max(Float32(0), x0 - pw), max(Float32(0), y0 - ph),
            min(Float32(width) - 1, x1 + pw), min(Float32(height) - 1, y1 + ph)]


# ---------------------------------------------------------------- heatmaps


def soma77_flip() -> List[Int]:
    """Left/right joint swap of the native77 keypoints (eyes, arms + fingers, legs)."""
    var f = List[Int](capacity=GX_JOINTS)
    for i in range(GX_JOINTS):
        f.append(i)
    f[9] = 10
    f[10] = 9
    for i in range(11, 39):
        f[i] = i + 28
        f[i + 28] = i
    for i in range(67, 72):
        f[i] = i + 5
        f[i + 5] = i
    return f^


def decode_heatmaps(heat: List[Float32], original: Int, flipped: Int, cx: Float32, cy: Float32,
                    size: Float32, mut kp: List[Float32], kp_offset: Int):
    """GEM-X flip-averaged native77 decoding: `(original + mirrored(flipped)) / 2`, the argmax with
    quarter-pixel refinement, the UDP edge mapping back into the image and the peak as
    confidence. `original` / `flipped` index [77, 64, 48] blocks of `heat`."""
    comptime W = GX_HEAT_W
    comptime H = GX_HEAT_H
    var flip = soma77_flip()
    var m = List[Float32](length=W * H, fill=0.0)
    var sx = size * 24 / 32 / 200
    var sy = size / 200
    for k in range(GX_JOINTS):
        var a = original + k * W * H
        var b = flipped + flip[k] * W * H
        var best = 0
        for y in range(H):
            for x in range(W):
                var v = (heat[a + y * W + x] + heat[b + y * W + (W - 1 - x)]) * Float32(0.5)
                m[y * W + x] = v
                if v > m[best]:
                    best = y * W + x
        var ix = best % W
        var iy = best // W
        var px = Float32(ix)
        var py = Float32(iy)
        if 1 < ix and ix < W - 1:
            var d = m[iy * W + ix + 1] - m[iy * W + ix - 1]
            px += Float32(0.25) if d > 0 else (Float32(-0.25) if d < 0 else Float32(0))
        if 1 < iy and iy < H - 1:
            var d = m[(iy + 1) * W + ix] - m[(iy - 1) * W + ix]
            py += Float32(0.25) if d > 0 else (Float32(-0.25) if d < 0 else Float32(0))
        var o = kp_offset + 3 * k
        kp[o] = px / Float32(W - 1) * (sx * 200) + cx - sx * 100
        kp[o + 1] = py / Float32(H - 1) * (sy * 200) + cy - sy * 100
        kp[o + 2] = m[best]


# ---------------------------------------------------------------- rotations


@always_inline
def _norm3(x: Float32, y: Float32, z: Float32) -> Float32:
    return max(sqrt(x * x + y * y + z * z), Float32(1e-12))


def rotation_6d_to_matrix(d: List[Float32], o: Int, mut r: List[Float32], ro: Int):
    """PyTorch3D `rotation_6d_to_matrix`: Gram-Schmidt of the two 3-vectors at `d[o:o+6]`; the
    rows of the 3x3 result (row major at `r[ro:ro+9]`) are b1, b2, b1 x b2."""
    var n1 = _norm3(d[o], d[o + 1], d[o + 2])
    var b1x = d[o] / n1
    var b1y = d[o + 1] / n1
    var b1z = d[o + 2] / n1
    var dot = b1x * d[o + 3] + b1y * d[o + 4] + b1z * d[o + 5]
    var ux = d[o + 3] - dot * b1x
    var uy = d[o + 4] - dot * b1y
    var uz = d[o + 5] - dot * b1z
    var n2 = _norm3(ux, uy, uz)
    var b2x = ux / n2
    var b2y = uy / n2
    var b2z = uz / n2
    r[ro] = b1x
    r[ro + 1] = b1y
    r[ro + 2] = b1z
    r[ro + 3] = b2x
    r[ro + 4] = b2y
    r[ro + 5] = b2z
    r[ro + 6] = b1y * b2z - b1z * b2y
    r[ro + 7] = b1z * b2x - b1x * b2z
    r[ro + 8] = b1x * b2y - b1y * b2x


def _mat_mul(a: List[Float32], ao: Int, b: List[Float32], bo: Int, mut c: List[Float32], co: Int, bt: Bool = False):
    """c = a @ b (or a @ b^T) for row-major 3x3 blocks."""
    var t = List[Float32](length=9, fill=0.0)
    for i in range(3):
        for j in range(3):
            var s: Float32 = 0
            for k in range(3):
                var bv = b[bo + j * 3 + k] if bt else b[bo + k * 3 + j]
                s += a[ao + i * 3 + k] * bv
            t[i * 3 + j] = s
    for k in range(9):
        c[co + k] = t[k]


def _axis_angle_matrix(x: Float32, y: Float32, z: Float32, mut r: List[Float32], ro: Int):
    var angle = sqrt(x * x + y * y + z * z)
    var c = Float32(1)
    var s = Float32(0)
    var nx = Float32(0)
    var ny = Float32(0)
    var nz = Float32(0)
    if angle > 1e-12:
        nx = x / angle
        ny = y / angle
        nz = z / angle
        c = cos(angle)
        s = sin(angle)
    var v = 1 - c
    r[ro] = c + nx * nx * v
    r[ro + 1] = nx * ny * v - nz * s
    r[ro + 2] = nx * nz * v + ny * s
    r[ro + 3] = ny * nx * v + nz * s
    r[ro + 4] = c + ny * ny * v
    r[ro + 5] = ny * nz * v - nx * s
    r[ro + 6] = nz * nx * v - ny * s
    r[ro + 7] = nz * ny * v + nx * s
    r[ro + 8] = c + nz * nz * v


def matrix_to_quat(r: List[Float32], o: Int) -> MbQuat:
    return quat_from_basis(MbVec3(r[o], r[o + 3], r[o + 6]), MbVec3(r[o + 1], r[o + 4], r[o + 7]),
                           MbVec3(r[o + 2], r[o + 5], r[o + 8])).normalized()


def _quat_matrix(q: MbQuat) -> List[Float32]:
    var cx = q.rotate(MbVec3(1, 0, 0))
    var cy = q.rotate(MbVec3(0, 1, 0))
    var cz = q.rotate(MbVec3(0, 0, 1))
    return [cx.x, cy.x, cz.x, cx.y, cy.y, cz.y, cx.z, cy.z, cz.z]


# ---------------------------------------------------------------- motion


struct GemxMotion(Copyable, Movable):
    """GEM-X predictions for one clip of `frames` frames (per-frame arrays are row major).

    Rotations are row-major 3x3 matrices. `body_rot` holds the 76 SOMA joints after Hips
    (`soma78_joint_names()[2:]`) relative to the SOMA T-pose; `incam_*` are the Hips rotation and
    body translation in OpenCV camera coordinates (x right, y down, z forward, metres),
    `world_*` the same in GEM-X's gravity-aligned world rollout (y up, metres).
    """
    var frames: Int
    var width: Int
    var height: Int
    var bbx_xys: List[Float32]       # T x 3: square crop centre and size, pixels
    var kp2d: List[Float32]          # T x 77 x 3: native77 keypoints x, y, confidence
    var motion: List[Float32]        # T x 655: normalized network motion (identity / scale averaged)
    var pred_cam: List[Float32]      # T x 3: weak-perspective camera (s, tx, ty)
    var static_conf: List[Float32]   # T x C: foot / contact static-confidence logits
    var body_rot: List[Float32]      # T x 76 x 9
    var incam_rot: List[Float32]     # T x 9
    var incam_transl: List[Float32]  # T x 3
    var world_rot: List[Float32]     # T x 9
    var world_transl: List[Float32]  # T x 3
    var identity: List[Float32]      # 128 SOMA identity coefficients
    var scale: List[Float32]         # 57 scale parameters (global 1, then 56 local)

    def __init__(out self):
        self.frames = 0
        self.width = 0
        self.height = 0
        self.bbx_xys = List[Float32]()
        self.kp2d = List[Float32]()
        self.motion = List[Float32]()
        self.pred_cam = List[Float32]()
        self.static_conf = List[Float32]()
        self.body_rot = List[Float32]()
        self.incam_rot = List[Float32]()
        self.incam_transl = List[Float32]()
        self.world_rot = List[Float32]()
        self.world_transl = List[Float32]()
        self.identity = List[Float32]()
        self.scale = List[Float32]()


def static_cam_angvel(frames: Int) -> List[Float32]:
    """GEM-X `--camera-mode static`: an identity 6-D camera rotation between all frames."""
    var out = List[Float32](capacity=6 * frames)
    for _ in range(frames):
        for v in [Float32(1), 0, 0, 0, 1, 0]:
            out.append(v)
    return out^


def motion_focal(width: Int, height: Int) -> Float32:
    """GEM-X Motion intrinsics: focal length = image diagonal, principal point = centre."""
    return Float32(sqrt(Float64(width) * Float64(width) + Float64(height) * Float64(height)))


def motion_observations(kp2d: List[Float32], bbx_xys: List[Float32], frames: Int) -> List[Float32]:
    """GEM-X `normalize_keypoints`: keypoints in the crop's [-1, 1] frame; confidence zeroed for
    points outside the (inclusive) crop."""
    var out = List[Float32](length=frames * GX_JOINTS * 3, fill=0.0)
    for t in range(frames):
        var cx = bbx_xys[3 * t]
        var cy = bbx_xys[3 * t + 1]
        var s = bbx_xys[3 * t + 2]
        var hx = cx + s / 2
        var hy = cy + s / 2
        var lx = cx - s / 2
        var ly = cy - s / 2
        for j in range(GX_JOINTS):
            var o = (t * GX_JOINTS + j) * 3
            var x = kp2d[o]
            var y = kp2d[o + 1]
            var outside = x < lx or x > hx or y < ly or y > hy
            out[o] = 2 * (x - cx) / s
            out[o + 1] = 2 * (y - cy) / s
            out[o + 2] = Float32(0) if outside else kp2d[o + 2]
    return out^


def motion_cliff(bbx_xys: List[Float32], frames: Int, width: Int, height: Int) -> List[Float32]:
    """GEM-X `compute_bbox_info_bedlam` (and the Body Pose CLIFF vector): the crop centre offset
    from the principal point and the crop size, over the focal length."""
    var f = motion_focal(width, height)
    var px = Float32(width) / 2
    var py = Float32(height) / 2
    var out = List[Float32](length=3 * frames, fill=0.0)
    for t in range(frames):
        out[3 * t] = (bbx_xys[3 * t] - px) / f
        out[3 * t + 1] = (bbx_xys[3 * t + 1] - py) / f
        out[3 * t + 2] = bbx_xys[3 * t + 2] / f
    return out^


def motion_attention_bias(frames: Int) -> List[Float32]:
    """0 where frame i may attend frame j, -1e30 elsewhere: clips longer than the trained
    120-frame window attend a sliding 120-frame window, as the GEM-X network masks them."""
    var out = List[Float32](length=frames * frames, fill=0.0)
    if frames <= GX_MAX_LEN:
        return out^
    for i in range(frames):
        var start = min(frames - GX_MAX_LEN, max(0, i - GX_MAX_LEN // 2))
        var stop = max(GX_MAX_LEN, min(frames, i + GX_MAX_LEN // 2))
        for j in range(frames):
            if j < start or j >= stop:
                out[i * frames + j] = -1e30
    return out^


def gaussian_kernel(sigma: Float64, radius: Int) -> List[Float32]:
    """scipy `_gaussian_kernel1d(sigma, 0, radius)` in double, then float32."""
    var w = List[Float64]()
    var total = 0.0
    for x in range(-radius, radius + 1):
        var v = exp(-0.5 / (sigma * sigma) * Float64(x * x))
        w.append(v)
        total += v
    var out = List[Float32]()
    for v in w:
        out.append(Float32(v / total))
    return out^


def _smooth_rows(x: List[Float32], frames: Int, width: Int) -> List[Float32]:
    """GEM-X `gaussian_smooth(sigma=3)` along time with replicate padding."""
    var k = gaussian_kernel(3.0, 12)
    var out = List[Float32](length=len(x), fill=0.0)
    for t in range(frames):
        for c in range(width):
            var acc: Float32 = 0
            for d in range(25):
                var s = min(max(t + d - 12, 0), frames - 1)
                acc += x[s * width + c] * k[d]
            out[t * width + c] = acc
    return out^


def decode_motion(mut m: GemxMotion, raw: List[Float32], pred_cam: List[Float32], mean: List[Float32],
                  sd: List[Float32], cam_angvel: List[Float32]):
    """GEM-X `MotionPipeline` after the network: the clip average of identity / log-scale
    (`avgbeta`), de-normalization, the 6-D rotations, the full-perspective camera translation
    and the gravity-view world rollout."""
    var t_n = m.frames
    m.motion = raw.copy()
    for c in range(456, 640):
        var s: Float32 = 0
        for t in range(t_n):
            s += raw[t * GX_MOTION_DIM + c]
        var a = s / Float32(t_n)
        for t in range(t_n):
            m.motion[t * GX_MOTION_DIM + c] = a
    var x = List[Float32](length=t_n * GX_MOTION_DIM, fill=0.0)
    for t in range(t_n):
        for c in range(GX_MOTION_DIM):
            x[t * GX_MOTION_DIM + c] = m.motion[t * GX_MOTION_DIM + c] * sd[c] + mean[c]
    m.body_rot = List[Float32](length=t_n * GX_BODY * 9, fill=0.0)
    m.incam_rot = List[Float32](length=t_n * 9, fill=0.0)
    var gv = List[Float32](length=t_n * 9, fill=0.0)
    var vel = List[Float32](length=t_n * 3, fill=0.0)
    for t in range(t_n):
        var o = t * GX_MOTION_DIM
        for j in range(GX_BODY):
            rotation_6d_to_matrix(x, o + 6 * j, m.body_rot, (t * GX_BODY + j) * 9)
        rotation_6d_to_matrix(x, o + 640, m.incam_rot, t * 9)
        rotation_6d_to_matrix(x, o + 646, gv, t * 9)
        for c in range(3):
            vel[3 * t + c] = x[o + 652 + c]
    m.identity = List[Float32]()
    for c in range(456, 584):
        m.identity.append(x[c])
    m.scale = [Float32(1)]
    for c in range(584, 640):
        m.scale.append(exp(x[c]))
    m.pred_cam = pred_cam.copy()
    # compute_transl_full_cam
    var f = motion_focal(m.width, m.height)
    var icx = Float32(m.width) / 2
    var icy = Float32(m.height) / 2
    m.incam_transl = List[Float32](length=3 * t_n, fill=0.0)
    for t in range(t_n):
        var sb = pred_cam[3 * t] * m.bbx_xys[3 * t + 2]
        var cx = 2 * (m.bbx_xys[3 * t] - icx) / (sb + Float32(1e-9))
        var cy = 2 * (m.bbx_xys[3 * t + 1] - icy) / (sb + Float32(1e-9))
        var tz = 2 * f / (sb + Float32(1e-9))
        m.incam_transl[3 * t] = pred_cam[3 * t + 1] + cx
        m.incam_transl[3 * t + 1] = pred_cam[3 * t + 2] + cy
        m.incam_transl[3 * t + 2] = tz
    # rollout_world: the camera's yaw change between frames, smoothed, accumulated onto the
    # gravity-view orientation; the root displacement integrated in that frame.
    var cam = List[Float32](length=t_n * 9, fill=0.0)
    for t in range(t_n):
        rotation_6d_to_matrix(cam_angvel, 6 * t, cam, 9 * t)
    var aa = List[Float32](length=3 * t_n, fill=0.0)
    var c2gv = List[Float32](length=9, fill=0.0)
    var nxt = List[Float32](length=9, fill=0.0)
    for t in range(t_n):
        _mat_mul(gv, 9 * t, m.incam_rot, 9 * t, c2gv, 0, bt=True)
        _mat_mul(c2gv, 0, cam, 9 * t, nxt, 0, bt=True)
        var n1 = _norm3(c2gv[2], 0, c2gv[8])
        var v1x = c2gv[2] / n1
        var v1z = c2gv[8] / n1
        var n2 = _norm3(nxt[2], 0, nxt[8])
        var v2x = nxt[2] / n2
        var v2z = nxt[8] / n2
        # cross(v2, v1) with zero y components: only the y component survives
        var ay = v2z * v1x - v2x * v1z
        var ang = acos(min(max(v1x * v2x + v1z * v2z, Float32(-1)), Float32(1)))
        var na = max(abs(ay), Float32(1e-12))
        aa[3 * t + 1] = ay / na * ang
    var sm = _smooth_rows(aa, t_n, 3)
    var acc = List[Float32](length=9, fill=0.0)
    acc[0] = 1
    acc[4] = 1
    acc[8] = 1
    var step = List[Float32](length=9, fill=0.0)
    m.world_rot = List[Float32](length=9 * t_n, fill=0.0)
    for t in range(t_n):
        if t > 0:
            _axis_angle_matrix(sm[3 * t], sm[3 * t + 1], sm[3 * t + 2], step, 0)
            var prev = acc.copy()
            _mat_mul(prev, 0, step, 0, acc, 0, bt=True)      # acc @ R(aa)^T
        _mat_mul(acc, 0, gv, 9 * t, m.world_rot, 9 * t)
    m.world_transl = List[Float32](length=3 * t_n, fill=0.0)
    var px: Float32 = 0
    var py: Float32 = 0
    var pz: Float32 = 0
    for t in range(t_n):
        m.world_transl[3 * t] = px
        m.world_transl[3 * t + 1] = py
        m.world_transl[3 * t + 2] = pz
        var r = 9 * t
        px += m.world_rot[r] * vel[3 * t] + m.world_rot[r + 1] * vel[3 * t + 1] + m.world_rot[r + 2] * vel[3 * t + 2]
        py += m.world_rot[r + 3] * vel[3 * t] + m.world_rot[r + 4] * vel[3 * t + 1] + m.world_rot[r + 5] * vel[3 * t + 2]
        pz += m.world_rot[r + 6] * vel[3 * t] + m.world_rot[r + 7] * vel[3 * t + 1] + m.world_rot[r + 8] * vel[3 * t + 2]
    # gravity view (y down) to y up: 180 degrees about z
    for t in range(t_n):
        for k in range(6):
            m.world_rot[9 * t + k] = -m.world_rot[9 * t + k]
        m.world_transl[3 * t] = -m.world_transl[3 * t]
        m.world_transl[3 * t + 1] = -m.world_transl[3 * t + 1]


# ---------------------------------------------------------------- runner


def _host_floats(t: Tensor) -> List[Float32]:
    return t.owned_floats()


struct Gemx(Movable):
    """The three GEM-X networks, loaded from the directory `tools/gemx/export.py` wrote."""
    var vitpose: OnnxModel
    var pose: OnnxModel
    var net: OnnxModel
    var batch: Int                 # frames per ViTPose / Body Pose run
    var gpu: Bool
    var codec_mean: List[Float32]
    var codec_std: List[Float32]
    var ns_crop: Int
    var ns_vitpose: Int
    var ns_pose: Int
    var ns_motion: Int

    def __init__(out self):
        self.vitpose = OnnxModel()
        self.pose = OnnxModel()
        self.net = OnnxModel()
        self.batch = 4
        self.gpu = False
        self.codec_mean = List[Float32]()
        self.codec_std = List[Float32]()
        self.ns_crop = 0
        self.ns_vitpose = 0
        self.ns_pose = 0
        self.ns_motion = 0

    @staticmethod
    def load(dir: String, gpu: Bool = True, keypoints: Bool = True, pose: Bool = True, motion: Bool = True) raises -> Gemx:
        """Load `gemx_vitpose.onnx`, `gemx_pose.onnx` and `gemx_motion.onnx` (each optional),
        on the GPU when `gpu` and one is present."""
        var g = Gemx()
        var on_gpu = gpu
        if keypoints:
            g.vitpose = OnnxModel.load(dir + "/gemx_vitpose.onnx")
            if gpu:
                on_gpu = g.vitpose.use_gpu() and on_gpu
        if pose:
            g.pose = OnnxModel.load(dir + "/gemx_pose.onnx")
            if gpu:
                on_gpu = g.pose.use_gpu() and on_gpu
        if motion:
            g.net = OnnxModel.load(dir + "/gemx_motion.onnx")
            if gpu:
                on_gpu = g.net.use_gpu() and on_gpu
            g.codec_mean = g.net.weight("codec_mean").owned_floats()
            g.codec_std = g.net.weight("codec_std").owned_floats()
        g.gpu = on_gpu
        return g^

    def keypoints(mut self, frames: List[GxFrame], bbx_xys: List[Float32], first: Int = 0, count: Int = -1) raises -> List[Float32]:
        """Native77 keypoints (x, y, confidence; 77 x 3 per frame) of frames `first..` in their
        square crops (cx, cy, size per frame of `bbx_xys`, indexed like `frames`)."""
        var n = len(frames) - first if count < 0 else count
        var out = List[Float32](length=n * GX_JOINTS * 3, fill=0.0)
        comptime IMG = 3 * 256 * 192
        var b = self.batch
        var start = 0
        while start < n:
            var t0 = perf_counter_ns()
            var data = List[Float32](length=2 * b * IMG, fill=0.0)
            for i in range(b):
                var f = first + min(start + i, n - 1)    # pad the last batch with its last frame
                var crop = vitpose_crop(frames[f], bbx_xys[3 * f], bbx_xys[3 * f + 1], bbx_xys[3 * f + 2])
                _normalize_chw(crop, 192, 256, data, i * IMG, False)
                _normalize_chw(crop, 192, 256, data, (b + i) * IMG, True)
            var t1 = perf_counter_ns()
            self.vitpose.set_input("img", Tensor.floats([2 * b, 3, 256, 192], data^))
            self.vitpose.run()
            var heat = _host_floats(self.vitpose.output("heatmaps"))
            self.ns_crop += Int(t1 - t0)
            self.ns_vitpose += Int(perf_counter_ns() - t1)
            comptime HM = GX_JOINTS * GX_HEAT_W * GX_HEAT_H
            for i in range(min(b, n - start)):
                var f = first + start + i
                decode_heatmaps(heat, i * HM, (b + i) * HM, bbx_xys[3 * f], bbx_xys[3 * f + 1],
                                bbx_xys[3 * f + 2], out, (start + i) * GX_JOINTS * 3)
            start += b
        return out^

    def track(mut self, frames: List[GxFrame], x0: Float32, y0: Float32, x1: Float32, y1: Float32) raises -> List[Float32]:
        """Follow one person from a first-frame box (pixels) with the native77 keypoints: each
        frame's box is its confident keypoints' padded extent, found from the previous frame's box
        (three refinements on the first frame). Returns x0 y0 x1 y1 per frame."""
        var w = frames[0].width
        var h = frames[0].height
        var box: List[Float32] = [x0, y0, x1, y1]
        var out = List[Float32](capacity=4 * len(frames))
        var saved = self.batch
        self.batch = 1
        for t in range(len(frames)):
            for _ in range(3 if t == 0 else 1):
                var q = square_crop(box[0], box[1], box[2], box[3])
                var xys = List[Float32](length=3 * len(frames), fill=0.0)
                xys[3 * t] = q[0]
                xys[3 * t + 1] = q[1]
                xys[3 * t + 2] = q[2]
                var kp = self.keypoints(frames, xys, t, 1)
                var nb = keypoint_box(kp, 0, w, h)
                if len(nb) == 4:
                    box = nb^
            for k in range(4):
                out.append(box[k])
        self.batch = saved
        return out^

    def pose_tokens(mut self, frames: List[GxFrame], bbx_xys: List[Float32]) raises -> List[Float32]:
        """Body Pose tokens (1024 per frame) of every frame's square crop."""
        var n = len(frames)
        var cliff = motion_cliff(bbx_xys, n, frames[0].width, frames[0].height)
        var out = List[Float32](length=n * GX_TOKEN, fill=0.0)
        comptime IMG = 3 * 256 * 256
        var b = self.batch
        var start = 0
        while start < n:
            var t0 = perf_counter_ns()
            var data = List[Float32](length=b * IMG, fill=0.0)
            var cl = List[Float32](length=b * 3, fill=0.0)
            for i in range(b):
                var f = min(start + i, n - 1)
                var crop = pose_crop(frames[f], bbx_xys[3 * f], bbx_xys[3 * f + 1], bbx_xys[3 * f + 2])
                _normalize_chw(crop, 256, 256, data, i * IMG, False)
                for c in range(3):
                    cl[3 * i + c] = cliff[3 * f + c]
            var t1 = perf_counter_ns()
            self.pose.set_input("img", Tensor.floats([b, 3, 256, 256], data^))
            self.pose.set_input("cliff", Tensor.floats([b, 3], cl^))
            self.pose.run()
            var tok = _host_floats(self.pose.output("token"))
            self.ns_crop += Int(t1 - t0)
            self.ns_pose += Int(perf_counter_ns() - t1)
            for i in range(min(b, n - start)):
                for c in range(GX_TOKEN):
                    out[(start + i) * GX_TOKEN + c] = tok[i * GX_TOKEN + c]
            start += b
        return out^

    def motion(mut self, kp2d: List[Float32], bbx_xys: List[Float32], tokens: List[Float32], frames: Int,
               width: Int, height: Int, cam_angvel: List[Float32] = List[Float32]()) raises -> GemxMotion:
        """Run the Motion transformer over the whole clip and decode it. `cam_angvel` is the
        6-D camera rotation from each frame to the next (static camera when empty)."""
        var cam = static_cam_angvel(frames) if len(cam_angvel) == 0 else cam_angvel.copy()
        var pos = List[Float32](capacity=frames)
        for t in range(frames):
            pos.append(Float32(t))
        var t0 = perf_counter_ns()
        self.net.set_input("obs", Tensor.floats([1, frames, GX_JOINTS, 3], motion_observations(kp2d, bbx_xys, frames)))
        self.net.set_input("cliff", Tensor.floats([1, frames, 3], motion_cliff(bbx_xys, frames, width, height)))
        self.net.set_input("cam_angvel", Tensor.floats([1, frames, 6], cam.copy()))
        self.net.set_input("imgseq", Tensor.floats([1, frames, GX_TOKEN], tokens.copy()))
        self.net.set_input("positions", Tensor.floats([1, frames], pos^))
        self.net.set_input("attn_bias", Tensor.floats([frames, frames], motion_attention_bias(frames)))
        self.net.run()
        var raw = _host_floats(self.net.output("motion"))
        var pcam = _host_floats(self.net.output("pred_cam"))
        var m = GemxMotion()
        m.frames = frames
        m.width = width
        m.height = height
        m.bbx_xys = bbx_xys.copy()
        m.kp2d = kp2d.copy()
        m.static_conf = _host_floats(self.net.output("static_conf"))
        decode_motion(m, raw, pcam, self.codec_mean, self.codec_std, cam)
        self.ns_motion += Int(perf_counter_ns() - t0)
        return m^

    def run(mut self, frames: List[GxFrame], xyxy: List[Float32]) raises -> GemxMotion:
        """The whole pipeline on RGB frames (30 fps) with one person box per frame."""
        var w = frames[0].width
        var h = frames[0].height
        var xys = boxes_to_crops(xyxy, w, h)
        var kp = self.keypoints(frames, xys)
        var tok = self.pose_tokens(frames, xys)
        return self.motion(kp, xys, tok, len(frames), w, h)


# ---------------------------------------------------------------- clips


def soma77_skeleton() -> MbSkeleton:
    """The neutral SOMA-X skeleton with Hips as joint 0 (77 joints, metres, y up), in the
    T-pose-oriented joint frames MotionBricks' SOMA BVH files use: all-identity local rotations
    are not a pose, `gemx_clip` of a rest prediction is the T-pose."""
    var names = soma78_joint_names()
    var parents = soma78_parents()
    var off = soma77_t_pose_offsets()
    var sk = MbSkeleton()
    for s in range(1, 78):
        sk.names.append(names[s])
        sk.parents.append(parents[s] - 1)
        var x = MbXform.identity()
        x.p = MbVec3(off[3 * (s - 1)], off[3 * (s - 1) + 1], off[3 * (s - 1) + 2])
        sk.reference.append(x)
    return sk^


def gemx_clip(m: GemxMotion, world: Bool = True, sample_rate: Float32 = 30) -> MbClip:
    """`gemx_clip` on the neutral SOMA-X skeleton (`soma77_skeleton()`)."""
    return gemx_clip(m, soma77_skeleton(), world, sample_rate)


def gemx_clip(m: GemxMotion, skeleton: MbSkeleton, world: Bool = True, sample_rate: Float32 = 30) -> MbClip:
    """Pose `skeleton` (a SOMA skeleton such as `soma77_skeleton()` or a MotionBricks BVH's,
    joints matched by name) with the predictions: each joint's local rotation is the SOMA
    T-pose-relative rotation in the rig's T-pose-oriented joint frames,
    `orient[parent]^-1 * R * orient[joint]`, as SOMA-X poses its body. The Hips follow the
    predicted translation: the world rollout starts at the skeleton's rest Hips position (so the
    body stands on y = 0); without `world` they sit at the camera-space Hips position.
    Bone lengths stay the skeleton's; joints SOMA lacks keep their reference transform."""
    var names = soma78_joint_names()
    var parents = soma78_parents()
    var oq = soma78_t_pose_orient()
    var orient = List[MbQuat]()
    for j in range(78):
        orient.append(MbQuat(oq[4 * j], oq[4 * j + 1], oq[4 * j + 2], oq[4 * j + 3]).normalized())
    var nj = skeleton.num_joints()
    var soma_of = List[Int](length=nj, fill=-1)
    for j in range(nj):
        for s in range(1, 78):
            if names[s] == skeleton.names[j]:
                soma_of[j] = s
    var clip = MbClip()
    clip.skeleton = skeleton.copy()
    clip.sample_rate = sample_rate
    var rot = m.world_rot.copy() if world else m.incam_rot.copy()
    var tr = m.world_transl.copy() if world else m.incam_transl.copy()
    for t in range(m.frames):
        var pose = List[MbXform](capacity=nj)
        for j in range(nj):
            var x = skeleton.reference[j]
            var s = soma_of[j]
            if s == 1:
                var q = matrix_to_quat(rot, 9 * t)
                x.q = (q * orient[1]).normalized()
                x.p = MbVec3(tr[3 * t], tr[3 * t + 1], tr[3 * t + 2])
                if world:
                    x.p = x.p + skeleton.reference[j].p
            elif s >= 2:
                var q = matrix_to_quat(m.body_rot, (t * GX_BODY + s - 2) * 9)
                x.q = (orient[parents[s]].inverse() * q * orient[s]).normalized()
            pose.append(x)
        clip.frames.append(pose^)
    return clip^


def _fmt(v: Float32) -> String:
    var r = Float64(Int(Float64(v) * 1e5 + (0.5 if v >= 0 else -0.5))) / 1e5
    return String(r)


def _euler_zyx(q: MbQuat) -> SIMD[DType.float32, 4]:
    """Degrees (z, y, x) with q = Rz * Ry * Rx, the BVH channel order `write_bvh` uses."""
    var r = _quat_matrix(q)
    var sy = min(max(-r[6], Float32(-1)), Float32(1))
    var y = asin(sy)
    var z: Float32
    var x: Float32
    if abs(sy) < 0.99999:
        z = atan2(r[3], r[0])
        x = atan2(r[7], r[8])
    else:
        z = atan2(-r[1], r[4])
        x = 0
    comptime DEG: Float32 = 57.29577951308232
    return SIMD[DType.float32, 4](z * DEG, y * DEG, x * DEG, 0)


def _bvh_joint(clip: MbClip, j: Int, depth: Int, mut s: String):
    var pad = String()
    for _ in range(depth):
        pad += "  "
    var p = clip.skeleton.reference[j].p
    if depth == 0:
        s += "ROOT " + clip.skeleton.names[j] + "\n{\n"
    else:
        s += pad + "JOINT " + clip.skeleton.names[j] + "\n" + pad + "{\n"
    s += pad + "  OFFSET " + _fmt(p.x * 100) + " " + _fmt(p.y * 100) + " " + _fmt(p.z * 100) + "\n"
    if depth == 0:
        s += pad + "  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n"
    else:
        s += pad + "  CHANNELS 3 Zrotation Yrotation Xrotation\n"
    var leaf = True
    for c in range(clip.skeleton.num_joints()):
        if clip.skeleton.parents[c] == j:
            leaf = False
            _bvh_joint(clip, c, depth + 1, s)
    if leaf:
        s += pad + "  End Site\n" + pad + "  {\n" + pad + "    OFFSET 0 0 0\n" + pad + "  }\n"
    s += pad + "}\n"


def _bvh_order(clip: MbClip, j: Int, mut order: List[Int]):
    order.append(j)
    for c in range(clip.skeleton.num_joints()):
        if clip.skeleton.parents[c] == j:
            _bvh_order(clip, c, order)


def write_bvh(clip: MbClip, path: String) raises:
    """Save a clip with one root joint (joint 0) as BVH: centimetres, ZYX Euler degrees, which
    `flint.ml.motionbricks.load_bvh` and DCC tools read back."""
    var s = String("HIERARCHY\n")
    _bvh_joint(clip, 0, 0, s)
    var order = List[Int]()
    _bvh_order(clip, 0, order)
    s += "MOTION\nFrames: " + String(clip.num_frames()) + "\nFrame Time: " + String(1.0 / Float64(clip.sample_rate)) + "\n"
    for f in range(clip.num_frames()):
        var line = String()
        for j in order:
            var x = clip.frames[f][j]
            if j == 0:
                line += _fmt(x.p.x * 100) + " " + _fmt(x.p.y * 100) + " " + _fmt(x.p.z * 100) + " "
            var e = _euler_zyx(x.q)
            line += _fmt(e[0]) + " " + _fmt(e[1]) + " " + _fmt(e[2]) + " "
        s += line + "\n"
    var bytes = List[UInt8](s.as_bytes())
    if not write_bytes(path, bytes):
        raise Error("gemx: cannot write " + path)


def clip_to_native_animation(clip: MbClip, node_of_joint: List[Int], name: String = "gemx") -> NativeAnimation:
    """A `NativeAnimation` of the clip for a loaded model: one rotation channel per joint whose
    `node_of_joint` entry is a node index (>= 0), and the root joint's translation. The model's
    rest rotations must use the same T-pose-oriented joint frames as the clip's skeleton."""
    var anim = NativeAnimation()
    anim.name = name
    var n = clip.num_frames()
    var times = List[Float32](capacity=n)
    for f in range(n):
        times.append(Float32(f) / clip.sample_rate)
    anim.duration = Float32(max(n - 1, 0)) / clip.sample_rate
    for j in range(min(len(node_of_joint), clip.skeleton.num_joints())):
        if node_of_joint[j] < 0:
            continue
        var ch = NativeChannel()
        ch.node = node_of_joint[j]
        ch.path = 1
        ch.times = times.copy()
        for f in range(n):
            var q = clip.frames[f][j].q
            ch.values.append(q.x)
            ch.values.append(q.y)
            ch.values.append(q.z)
            ch.values.append(q.w)
        anim.channels.append(ch^)
        if clip.skeleton.parents[j] < 0:
            var tc = NativeChannel()
            tc.node = node_of_joint[j]
            tc.path = 0
            tc.times = times.copy()
            for f in range(n):
                var p = clip.frames[f][j].p
                tc.values.append(p.x)
                tc.values.append(p.y)
                tc.values.append(p.z)
            anim.channels.append(tc^)
    return anim^


# ---------------------------------------------------------------- SOMA tables


def soma78_parents() -> List[Int]:
    """Parent of each of `soma78_joint_names()` (-1 for Root)."""
    return [-1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 4, 12, 13, 14, 15, 16, 17, 18, 15, 20, 21, 22, 23, 15, 25, 26, 27, 28, 15, 30, 31, 32, 33, 15, 35, 36, 37, 38, 4, 40, 41, 42, 43, 44, 45, 46, 43, 48, 49, 50, 51, 43, 53, 54, 55, 56, 43, 58, 59, 60, 61, 43, 63, 64, 65, 66, 1, 68, 69, 70, 71, 1, 73, 74, 75, 76]


def soma78_joint_names() -> List[String]:
    """The 78 SOMA joints: Root, Hips, then the 76 posed joints in GEM-X `body_pose` order."""
    return [
        "Root", "Hips", "Spine1", "Spine2", "Chest", "Neck1",
        "Neck2", "Head", "HeadEnd", "Jaw", "LeftEye", "RightEye",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumb1", "LeftHandThumb2",
        "LeftHandThumb3", "LeftHandThumbEnd", "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4",
        "LeftHandIndexEnd", "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
        "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd", "LeftHandPinky1",
        "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd", "RightShoulder", "RightArm",
        "RightForeArm", "RightHand", "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
        "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd", "RightHandMiddle1",
        "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd", "RightHandRing1", "RightHandRing2",
        "RightHandRing3", "RightHandRing4", "RightHandRingEnd", "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
        "RightHandPinky4", "RightHandPinkyEnd", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
        "LeftToeEnd", "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
    ]


def soma78_t_pose_orient() -> List[Float32]:
    """The SOMA-X v0.2.0 T-pose world orientation of each joint as a quaternion (x, y, z, w):
    `t_pose_world__v0.2.0` of SOMA-X `SOMA_neutral.npz` (Apache-2.0, see licenses/soma-x-LICENSE.txt)."""
    var q: List[Float32] = [
        0.0, 0.0, 0.0, 1.0,  # Root
        0.496573097, 0.503403575, 0.496573096, 0.503403576,  # Hips
        0.494499245, 0.505440893, 0.494499245, 0.505440893,  # Spine1
        0.47935829, 0.519822691, 0.47935829, 0.519822691,  # Spine2
        0.49911489, 0.500883545, 0.49911489, 0.500883546,  # Chest
        0.568820436, 0.420051558, 0.568820436, 0.420051559,  # Neck1
        0.568820442, 0.42005155, 0.568820442, 0.42005155,  # Neck2
        0.467083105, 0.530879811, 0.467083104, 0.530879811,  # Head
        0.467083105, 0.530879811, 0.467083104, 0.530879811,  # HeadEnd
        0.469715691, 0.528551943, 0.469715705, 0.528551955,  # Jaw
        0.533350203, 0.498629082, 0.499143971, 0.466649649,  # LeftEye
        0.499143947, 0.466649651, 0.53335023, 0.498629076,  # RightEye
        0.696101645, 0.12426786, -0.124267859, 0.696101643,  # LeftShoulder
        0.707106782, 1.62442989e-09, -1.62061972e-09, 0.70710678,  # LeftArm
        0.707106782, 1.62442989e-09, -1.62061972e-09, 0.70710678,  # LeftForeArm
        0.764435718, -0.000130036235, 0.000291513155, 0.644699877,  # LeftHand
        0.945194553, -0.232387034, 0.140145664, 0.181556371,  # LeftHandThumb1
        1.0, 1.15173009e-08, -1.07839444e-08, -3.77634755e-17,  # LeftHandThumb2
        1.0, 1.15173009e-08, -1.07839444e-08, -3.77634755e-17,  # LeftHandThumb3
        1.0, 1.15173009e-08, -1.07839444e-08, -3.77634755e-17,  # LeftHandThumbEnd
        0.756369064, -0.00856905501, 0.011466998, 0.65398847,  # LeftHandIndex1
        0.707106781, 1.45869058e-09, -1.13114537e-09, 0.707106781,  # LeftHandIndex2
        0.707106781, 1.45869058e-09, -1.13114537e-09, 0.707106781,  # LeftHandIndex3
        0.707106781, 1.45869058e-09, -1.13114537e-09, 0.707106781,  # LeftHandIndex4
        0.707106781, 1.45869058e-09, -1.13114537e-09, 0.707106781,  # LeftHandIndexEnd
        0.742223731, 0.0379297938, -0.0732773763, 0.665053148,  # LeftHandMiddle1
        0.707106781, 4.20371932e-09, -2.88663028e-09, 0.707106781,  # LeftHandMiddle2
        0.707106781, 4.20371932e-09, -2.88663028e-09, 0.707106781,  # LeftHandMiddle3
        0.707106781, 4.20371932e-09, -2.88663028e-09, 0.707106781,  # LeftHandMiddle4
        0.707106781, 4.20371932e-09, -2.88663028e-09, 0.707106781,  # LeftHandMiddleEnd
        0.732729216, 0.0473016673, -0.11182827, 0.669600543,  # LeftHandRing1
        0.707106784, 2.38795973e-09, -2.40942544e-09, 0.707106779,  # LeftHandRing2
        0.707106784, 2.38795973e-09, -2.40942544e-09, 0.707106779,  # LeftHandRing3
        0.707106784, 2.38795973e-09, -2.40942544e-09, 0.707106779,  # LeftHandRing4
        0.707106784, 2.38795973e-09, -2.40942544e-09, 0.707106779,  # LeftHandRingEnd
        0.700757075, 0.0260387369, -0.202007456, 0.683706438,  # LeftHandPinky1
        0.707106782, -5.22238783e-09, 5.05772154e-09, 0.70710678,  # LeftHandPinky2
        0.707106782, -5.22238783e-09, 5.05772154e-09, 0.70710678,  # LeftHandPinky3
        0.707106782, -5.22238783e-09, 5.05772154e-09, 0.70710678,  # LeftHandPinky4
        0.707106782, -5.22238783e-09, 5.05772154e-09, 0.70710678,  # LeftHandPinkyEnd
        -0.696101643, -0.124267859, -0.12426786, 0.696101645,  # RightShoulder
        -0.70710678, -1.63732405e-09, -1.60772556e-09, 0.707106782,  # RightArm
        -0.70710678, -1.63732405e-09, -1.60772556e-09, 0.707106782,  # RightForeArm
        -0.644699877, 0.000291513139, 0.000130036254, 0.764435718,  # RightHand
        -0.181544221, 0.140132746, 0.232381774, 0.945200096,  # RightHandThumb1
        -7.83733941e-10, 1.15677382e-08, 2.0386881e-09, 1.0,  # RightHandThumb2
        -7.83733941e-10, 1.15677382e-08, 2.0386881e-09, 1.0,  # RightHandThumb3
        -7.83733941e-10, 1.15677382e-08, 2.0386881e-09, 1.0,  # RightHandThumbEnd
        -0.653971564, 0.0115086693, 0.00855739896, 0.75638318,  # RightHandIndex1
        -0.707106779, -1.73970012e-09, -1.57679206e-09, 0.707106784,  # RightHandIndex2
        -0.707106779, -1.73970012e-09, -1.57679206e-09, 0.707106784,  # RightHandIndex3
        -0.707106779, -1.73970012e-09, -1.57679206e-09, 0.707106784,  # RightHandIndex4
        -0.707106779, -1.73970012e-09, -1.57679206e-09, 0.707106784,  # RightHandIndexEnd
        -0.665054038, -0.0732731994, -0.0379292893, 0.742223372,  # RightHandMiddle1
        -0.70710678, -2.21045369e-10, -1.61673038e-09, 0.707106782,  # RightHandMiddle2
        -0.70710678, -2.21045369e-10, -1.61673038e-09, 0.707106782,  # RightHandMiddle3
        -0.70710678, -2.21045369e-10, -1.61673038e-09, 0.707106782,  # RightHandMiddle4
        -0.70710678, -2.21045369e-10, -1.61673038e-09, 0.707106782,  # RightHandMiddleEnd
        -0.669600923, -0.111824161, -0.0473028122, 0.732729422,  # RightHandRing1
        -0.707106781, 8.23180634e-10, 3.4573588e-09, 0.707106781,  # RightHandRing2
        -0.707106781, 8.23180634e-10, 3.4573588e-09, 0.707106781,  # RightHandRing3
        -0.707106781, 8.23180634e-10, 3.4573588e-09, 0.707106781,  # RightHandRing4
        -0.707106781, 8.23180634e-10, 3.4573588e-09, 0.707106781,  # RightHandRingEnd
        -0.683707784, -0.202002444, -0.0260354008, 0.700757331,  # RightHandPinky1
        -0.707106778, 3.19066174e-09, 3.30281311e-09, 0.707106784,  # RightHandPinky2
        -0.707106778, 3.19066174e-09, 3.30281311e-09, 0.707106784,  # RightHandPinky3
        -0.707106778, 3.19066174e-09, 3.30281311e-09, 0.707106784,  # RightHandPinky4
        -0.707106778, 3.19066174e-09, 3.30281311e-09, 0.707106784,  # RightHandPinkyEnd
        0.5, -0.5, 0.5, -0.5,  # LeftLeg
        0.5, -0.5, 0.5, -0.5,  # LeftShin
        0.695349988, -0.128407142, 0.695349988, -0.128407142,  # LeftFoot
        0.699539475, -0.0995340785, 0.700572044, -0.0996809977,  # LeftToeBase
        0.700381425, -0.0934257245, 0.70141524, -0.0935636454,  # LeftToeEnd
        0.5, 0.5, 0.5, 0.5,  # RightLeg
        0.5, 0.5, 0.5, 0.5,  # RightShin
        0.128407142, 0.695349988, 0.128407142, 0.695349988,  # RightFoot
        0.0996821725, 0.700572103, 0.0995351877, 0.699539091,  # RightToeBase
        0.093564815, 0.701415306, 0.0934268504, 0.700381051,  # RightToeEnd
    ]
    return q^


def soma77_t_pose_offsets() -> List[Float32]:
    """Bone offsets of the neutral SOMA-X body (identity 0, metres) for joints 1..77 (Hips first):
    each joint's rest position relative to its parent in the parent's T-pose frame. Hips is placed
    so the toes touch y = 0. Computed with the SOMA-X model (Apache-2.0)."""
    var o: List[Float32] = [
        0.0, 0.947508641, 0.0,  # Hips
        0.052145424, 0.000796801434, -9.04733351e-06,  # Spine1
        0.0731782303, -2.01749776e-09, -4.52980329e-11,  # Spine2
        0.0719098993, -0.000659802749, 1.5253475e-10,  # Chest
        0.255282275, -0.00487956666, -0.000116642511,  # Neck1
        0.0723667345, -2.35499389e-08, -5.89612607e-10,  # Neck2
        0.0628340572, -2.77572187e-08, -5.57599195e-10,  # Head
        0.163160843, -0.00161318807, 7.23385065e-06,  # HeadEnd
        0.000698110293, 0.0310077907, 6.20008224e-06,  # Jaw
        0.0401685353, 0.0828401392, 0.0308850048,  # LeftEye
        0.0401709138, 0.0828173986, -0.0309425226,  # RightEye
        0.223737593, 0.0513690347, 0.0177874251,  # LeftShoulder
        0.159192965, -4.38787336e-08, -2.38138704e-07,  # LeftArm
        0.272817864, -3.93295747e-08, -5.96046446e-08,  # LeftForeArm
        0.249137715, -7.0975965e-08, 3.25060528e-16,  # LeftHand
        0.0174550949, 0.0325238859, 0.00830376859,  # LeftHandThumb1
        0.039765305, 2.42807069e-08, 5.00762262e-08,  # LeftHandThumb2
        0.0271279796, 6.01540664e-08, -6.05169587e-08,  # LeftHandThumb3
        0.0302412492, 6.02171194e-08, -6.06216581e-08,  # LeftHandThumbEnd
        0.0279480044, 0.0219366934, 0.000830872189,  # LeftHandIndex1
        0.059833246, -1.09766873e-09, 8.67636232e-09,  # LeftHandIndex2
        0.0358532624, -3.26743915e-08, 3.97779901e-11,  # LeftHandIndex3
        0.0214259003, -1.9093191e-09, 1.19233061e-07,  # LeftHandIndex4
        0.0262959568, -2.85125011e-09, 5.96338191e-08,  # LeftHandIndexEnd
        0.0276231647, 0.00804135569, -0.00411462555,  # LeftHandMiddle1
        0.0569580012, 2.42307586e-08, -1.04115231e-07,  # LeftHandMiddle2
        0.0442451212, -9.07871765e-10, 1.55921361e-10,  # LeftHandMiddle3
        0.0282735808, 2.9481121e-08, 5.97042786e-08,  # LeftHandMiddle4
        0.0235348926, 6.0112509e-08, 5.96875796e-08,  # LeftHandMiddleEnd
        0.0247269153, -0.00454748122, 0.00137770081,  # LeftHandRing1
        0.0536861626, -4.79146355e-08, 1.19303284e-08,  # LeftHandRing2
        0.043351233, -3.48599714e-09, 1.19212801e-07,  # LeftHandRing3
        0.026283741, -1.71756347e-09, -5.9602516e-08,  # LeftHandRing4
        0.0195578337, 2.80476335e-08, 1.58384037e-12,  # LeftHandRingEnd
        0.0230322692, -0.0177510321, 0.00505174589,  # LeftHandPinky1
        0.0484214421, -1.04044384e-08, 1.06178754e-08,  # LeftHandPinky2
        0.0303350055, -3.27967057e-10, 5.97658211e-08,  # LeftHandPinky3
        0.0168532113, -2.99845294e-08, -5.95151e-08,  # LeftHandPinky4
        0.0202501988, 2.95833854e-08, 1.07593491e-10,  # LeftHandPinkyEnd
        0.223712202, 0.0509898122, -0.0191670766,  # RightShoulder
        -0.159190419, 3.76280214e-08, -2.79870672e-10,  # RightArm
        -0.273908167, 9.52810757e-09, -1.76562586e-16,  # RightForeArm
        -0.248276636, 1.13706446e-08, -5.96046449e-08,  # RightHand
        -0.0174561088, -0.0325244776, -0.00830279066,  # RightHandThumb1
        -0.0397605647, 5.87026466e-08, -5.69041441e-08,  # RightHandThumb2
        -0.0271303041, -5.94520068e-08, -4.64178456e-10,  # RightHandThumb3
        -0.0302383286, 1.701241e-10, -5.17355074e-10,  # RightHandThumbEnd
        -0.0279512202, -0.0219344293, -0.000836491118,  # RightHandIndex1
        -0.0598387682, -2.85151963e-08, -1.23382165e-07,  # RightHandIndex2
        -0.0358499885, 3.25797476e-08, 1.19211313e-07,  # RightHandIndex3
        -0.0214251876, -2.79495658e-08, -5.96034355e-08,  # RightHandIndex4
        -0.0263006091, -5.77541362e-08, 5.96061292e-08,  # RightHandIndexEnd
        -0.0276245357, -0.00804165069, 0.00411428274,  # RightHandMiddle1
        -0.0569632524, -1.44421741e-08, -6.07047645e-09,  # RightHandMiddle2
        -0.0442454219, 1.32068092e-09, 5.93489343e-08,  # RightHandMiddle3
        -0.0282719135, 3.03873436e-08, -5.97680326e-08,  # RightHandMiddle4
        -0.0235390663, -2.88332084e-10, -5.97406802e-08,  # RightHandMiddleEnd
        -0.0247280487, 0.00454709605, -0.00137518597,  # RightHandRing1
        -0.0536897996, 2.07544932e-08, 1.11248015e-07,  # RightHandRing2
        -0.0433519483, -2.5253715e-08, 6.09266737e-10,  # RightHandRing3
        -0.0262816548, -2.744055e-08, 3.69361391e-10,  # RightHandRing4
        -0.0195600986, 2.23412014e-09, 2.74896986e-10,  # RightHandRingEnd
        -0.0230353691, 0.0177500946, -0.00505209999,  # RightHandPinky1
        -0.0484177505, -1.73146502e-08, 1.08320274e-08,  # RightHandPinky2
        -0.0303400141, 9.85731405e-11, -5.97726105e-08,  # RightHandPinky3
        -0.0168551197, -2.97475641e-08, -9.3311767e-11,  # RightHandPinky4
        -0.0202481735, -2.97365401e-08, -1.12096091e-10,  # RightHandPinkyEnd
        -0.0811483295, 0.0211730585, 0.112629876,  # LeftLeg
        0.40759141, 7.35581188e-09, -1.47704438e-08,  # LeftShin
        0.394938022, 1.85346248e-08, 1.26659351e-10,  # LeftFoot
        0.135499649, -1.34529077e-08, 1.60747805e-11,  # LeftToeBase
        0.0598681318, -0.0010450046, 2.61692226e-09,  # LeftToeEnd
        -0.081180578, 0.0206008777, -0.112656236,  # RightLeg
        -0.407592684, 9.47690175e-11, 1.30717788e-10,  # RightShin
        -0.394935012, -2.97104964e-08, 1.26658385e-10,  # RightFoot
        -0.135499642, 1.07919207e-08, 1.60747805e-11,  # RightToeBase
        -0.0598687517, 0.00104500888, -3.24364203e-08,  # RightToeEnd
    ]
    return o^
