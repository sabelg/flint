"""Fixed-camera single-performer capture using native RTMPose and MotionBERT inference."""
from std.math import floor, sqrt
from flint.assets.glb_native import little_u32
from flint.assets.image import NativeImage
from flint.assets.jpeg import decode_image_auto
from flint.assets.json import JsonDocument
from flint.anim.take_constraints import aim_joint, mb, vec
from flint.core.math import Vec3
from flint.ml.motionbricks import (
    MbClip,
    MbXform,
    MbVec3,
    quat_from_basis,
    world_pose,
)
from flint.ml.onnx import OnnxModel
from flint.ml.tensor import Tensor
from flint.platform.os import env_now, list_directory, read_bytes
from recipe_io import vector, string, num, integer, field, relative


struct Frames(Movable):
    var files: List[String]
    var avi: List[UInt8]
    var spans: List[Tuple[Int, Int]]
    var fps: Float32

    def __init__(out self, path: String) raises:
        self.files = []
        self.avi = []
        self.spans = []
        self.fps = 0
        if path.lower().endswith(".avi"):
            self.avi = read_bytes(path)
            if (
                len(self.avi) < 12
                or little_u32(self.avi, 0) != 0x46464952
                or little_u32(self.avi, 8) != 0x20495641
            ):
                raise Error("Expected an MJPEG AVI")
            self.scan(12, len(self.avi), False, 0)
        else:
            for name in list_directory(path):
                var ext = name.lower()
                if (
                    ext.endswith(".png")
                    or ext.endswith(".jpg")
                    or ext.endswith(".jpeg")
                ):
                    self.files.append(path + "/" + name)
        if self.count() < 2:
            raise Error(
                "Capture needs at least two PNG/JPEG frames or an MJPEG AVI"
            )

    def scan(mut self, start: Int, end: Int, movi: Bool, depth: Int) raises:
        if depth > 8:
            raise Error("Invalid AVI nesting")
        var i = start
        while i + 8 <= end:
            var tag = little_u32(self.avi, i)
            var size = little_u32(self.avi, i + 4)
            var begin = i + 8
            var stop = begin + size
            if stop > end:
                raise Error("Truncated AVI chunk")
            if tag == 0x5453494C and size >= 4:
                var kind = little_u32(self.avi, begin)
                self.scan(
                    begin + 4, stop, movi or kind == 0x69766F6D, depth + 1
                )
            elif tag == 0x68697661 and size >= 4:
                var us = little_u32(self.avi, begin)
                if us > 0:
                    self.fps = 1000000 / Float32(us)
            elif (
                movi
                and (tag >> 16 == 0x6364 or tag >> 16 == 0x6264)
                and size > 0
            ):
                if (
                    size < 3
                    or self.avi[begin] != 255
                    or self.avi[begin + 1] != 216
                ):
                    raise Error(
                        "Use MJPEG AVI or a numbered PNG/JPEG image sequence"
                    )
                self.spans.append((begin, stop))
            i = stop + (size % 2)

    def count(self) -> Int:
        return max(len(self.files), len(self.spans))

    def image(self, f: Int) raises -> NativeImage:
        var bytes = read_bytes(self.files[f]) if self.files else List[UInt8](
            self.avi[self.spans[f][0] : self.spans[f][1]]
        )
        var im = decode_image_auto(bytes)
        if not im.valid:
            raise Error("Invalid capture image at frame " + String(f))
        return im^


def detect(
    mut model: OnnxModel, im: NativeImage, roi: List[Float32]
) raises -> List[Vec3]:
    # ROI in source pixels. Preserve aspect ratio in the 192 x 256 model input.
    var x: Float32 = 0
    var y: Float32 = 0
    var w = Float32(im.width)
    var h = Float32(im.height)
    if roi:
        x = roi[0]
        y = roi[1]
        w = roi[2]
        h = roi[3]
    if (
        min(w, h) <= 0
        or x < 0
        or y < 0
        or x + w > Float32(im.width) + 0.01
        or y + h > Float32(im.height) + 0.01
    ):
        raise Error("Capture ROI falls outside the image")
    var stride = max(w / 192, h / 256)
    var x0 = x + w / 2 - 96 * stride
    var y0 = y + h / 2 - 128 * stride
    var t = Tensor.zeros([1, 3, 256, 192])
    var means: List[Float32] = [123.675, 116.28, 103.53]
    var stds: List[Float32] = [58.395, 57.12, 57.375]
    for j in range(256):
        for i in range(192):
            var u = x0 + Float32(i) * stride
            var v = y0 + Float32(j) * stride
            var ix = Int(floor(u))
            var iy = Int(floor(v))
            for k in range(3):
                var rgb: Float32 = 0
                for dy in range(2):
                    for dx in range(2):
                        var xx = ix + dx
                        var yy = iy + dy
                        if (
                            xx >= 0
                            and yy >= 0
                            and xx < im.width
                            and yy < im.height
                        ):
                            var c = (
                                im.pixels[yy * im.width + xx]
                                >> UInt32(16 - 8 * k)
                            ) & 255
                            rgb += (
                                Float32(c)
                                * (1 - abs(u - Float32(xx)))
                                * (1 - abs(v - Float32(yy)))
                            )
                t.f[k * 49152 + j * 192 + i] = (rgb - means[k]) / stds[k]
    model.set_input("input", t^)
    model.run()
    var xs = model.output("simcc_x")
    var ys = model.output("simcc_y")
    if xs.shape != [1, 17, 384] or ys.shape != [1, 17, 512]:
        raise Error("Expected RTMPose COCO17 SimCC outputs")
    var joints = List[Vec3]()
    for j in range(17):
        var a = 0
        var b = 0
        for i in range(1, 384):
            if xs.fget(j * 384 + i) > xs.fget(j * 384 + a):
                a = i
        for i in range(1, 512):
            if ys.fget(j * 512 + i) > ys.fget(j * 512 + b):
                b = i
        joints.append(
            Vec3(
                x0 + Float32(a) * 0.5 * stride,
                y0 + Float32(b) * 0.5 * stride,
                min(xs.fget(j * 384 + a), ys.fget(j * 512 + b)),
            )
        )
    return joints^


def h36(j: List[Vec3]) -> List[Vec3]:
    var hip = (j[11] + j[12]) * 0.5
    var neck = (j[5] + j[6]) * 0.5
    return [
        hip,
        j[12],
        j[14],
        j[16],
        j[11],
        j[13],
        j[15],
        (hip + neck) * 0.5,
        neck,
        j[0],
        (j[1] + j[2]) * 0.5,
        j[5],
        j[7],
        j[9],
        j[6],
        j[8],
        j[10],
    ]


struct View(Movable):
    var poses: List[List[Vec3]]
    var width: Int
    var height: Int
    var fps: Float32
    var eye: Vec3
    var forward: Vec3
    var right: Vec3
    var up: Vec3
    var fx: Float32
    var fy: Float32
    var cx: Float32
    var cy: Float32

    def __init__(
        out self,
        mut net: OnnxModel,
        d: JsonDocument,
        n: Int,
        file: String,
        stereo: Bool,
    ) raises:
        var frames = Frames(relative(file, string(d, n, "frames")))
        self.poses = []
        self.width = 0
        self.height = 0
        self.fps = frames.fps
        self.eye = Vec3()
        self.forward = Vec3(0, 0, -1)
        self.up = Vec3(0, 1, 0)
        self.right = Vec3(1, 0, 0)
        var roi = List[Float32]()
        var ar = field(d, n, "roi")
        if ar >= 0:
            if d.array_size(ar) != 4:
                raise Error("ROI needs [x,y,width,height]")
            for i in range(4):
                roi.append(Float32(d.number(d.at(ar, i), -1)))
        for f in range(frames.count()):
            var im = frames.image(f)
            if f and (im.width != self.width or im.height != self.height):
                raise Error("Capture size changed within a clip")
            self.width = im.width
            self.height = im.height
            self.poses.append(h36(detect(net, im, roi)))
        self.cx = num(d, n, "cx", Float64(self.width) * 0.5)
        self.cy = num(d, n, "cy", Float64(self.height) * 0.5)
        self.fx = num(d, n, "fx", 0)
        self.fy = num(d, n, "fy", Float64(self.fx))
        if stereo:
            self.eye = vector(d, n, "eye")
            self.forward = (vector(d, n, "lookAt") - self.eye).normalized()
            self.right = self.forward.cross(vector(d, n, "up")).normalized()
            self.up = self.right.cross(self.forward)
            if min(self.fx, self.fy) <= 0 or self.up.length_squared() < 0.9:
                raise Error("Invalid camera intrinsics or extrinsics")

    def ray(self, p: Vec3) -> Vec3:
        return (
            self.forward
            + self.right * ((p.x - self.cx) / self.fx)
            - self.up * ((p.y - self.cy) / self.fy)
        ).normalized()


def triangulate(
    a: View, b: View, f: Int, threshold: Float32, limit: Float32
) raises -> List[Vec3]:
    var result = List[Vec3]()
    for j in range(17):
        var p = a.poses[f][j]
        var q = b.poses[f][j]
        if min(p.z, q.z) < threshold:
            raise Error(
                "Low stereo confidence at frame "
                + String(f)
                + ", joint "
                + String(j)
            )
        var u = a.ray(p)
        var v = b.ray(q)
        var w = b.eye - a.eye
        var c = u.dot(v)
        var denom = 1 - c * c
        if denom < 1e-6:
            raise Error("Insufficient stereo parallax")
        var t = (w.dot(u) - c * w.dot(v)) / denom
        var s = c * t - w.dot(v)
        var left = a.eye + u * t
        var right = b.eye + v * s
        if min(t, s) <= 0 or (left - right).length() > limit:
            raise Error("Inconsistent stereo rays at frame " + String(f))
        result.append(
            (left * p.z + right * q.z) * (1 / max(Float32(1e-8), p.z + q.z))
        )
    return result^


def lift(
    view: View, path: String, threshold: Float32
) raises -> List[List[Vec3]]:
    var model = OnnxModel.load(path, 4)
    var out = List[List[Vec3]]()
    for start in range(0, len(view.poses), 5):
        var input = Tensor.zeros([1, 9, 17, 3])
        for i in range(9):
            var f = min(len(view.poses) - 1, max(0, start + i - 2))
            var seen = 0
            for j in range(17):
                var p = view.poses[f][j]
                seen += Int(p.z >= threshold)
                var k = (17 * i + j) * 3
                var scale = Float32(min(view.width, view.height)) * 0.5
                input.f[k] = (p.x - Float32(view.width) * 0.5) / scale
                input.f[k + 1] = (p.y - Float32(view.height) * 0.5) / scale
                input.f[k + 2] = min(Float32(1), max(Float32(0), p.z))
            if seen < 10:
                raise Error("Insufficient visible joints at frame " + String(f))
        model.set_input("keypoints", input^)
        model.run()
        var pred = model.output("pose3d")
        if pred.shape != [1, 9, 17, 3]:
            raise Error("Expected MotionBERT nine-frame H36M output")
        for i in range(min(5, len(view.poses) - start)):
            var pts = List[Vec3]()
            for j in range(17):
                var k = ((i + 2) * 17 + j) * 3
                pts.append(
                    Vec3(pred.fget(k), -pred.fget(k + 1), -pred.fget(k + 2))
                )
            out.append(pts^)
    return out^


def fit_soma(
    reference: MbClip,
    points: List[List[Vec3]],
    view: View,
    stereo: Bool,
    fps: Float32,
) raises -> MbClip:
    var clip = MbClip()
    clip.skeleton = reference.skeleton.copy()
    clip.sample_rate = fps
    var names: List[String] = [
        "Hips",
        "Spine1",
        "Chest",
        "Neck1",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "LeftLeg",
        "LeftShin",
        "RightLeg",
        "RightShin",
    ]
    var children: List[String] = [
        "Spine1",
        "Chest",
        "Neck1",
        "Head",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightArm",
        "RightForeArm",
        "RightHand",
        "LeftShin",
        "LeftFoot",
        "RightShin",
        "RightFoot",
    ]
    var a: List[Int] = [0, 7, 7, 8, 8, 11, 12, 8, 14, 15, 4, 5, 1, 2]
    var b: List[Int] = [7, 8, 8, 10, 11, 12, 13, 14, 15, 16, 5, 6, 2, 3]
    var js = List[Int]()
    var cs = List[Int]()
    for i in range(len(names)):
        js.append(clip.skeleton.joint_index(names[i]))
        cs.append(clip.skeleton.joint_index(children[i]))
    var rest = world_pose(
        clip.skeleton, clip.skeleton.reference, MbXform.identity()
    )
    var left = clip.skeleton.joint_index("LeftFoot")
    var right = clip.skeleton.joint_index("RightFoot")
    var rest_span = (rest[js[10]].p - rest[js[12]].p).length()
    var hips = view.poses[0][4] - view.poses[0][1]
    var px_span = max(Float32(1), sqrt(hips.x * hips.x + hips.y * hips.y))
    var ground: Float32 = 1e10
    for f in range(len(points)):
        clip.frames.append(clip.skeleton.reference.copy())
        ref p = points[f]
        var across = (p[4] - p[1]).normalized()
        var z = across.cross(p[8] - p[0]).normalized()
        if z.length_squared() < 0.8:
            raise Error("Degenerate captured pelvis")
        clip.frames[f][0].q = quat_from_basis(
            mb(across), mb(z.cross(across)), mb(z)
        )
        var screen = view.poses[f][0] - view.poses[0][0]
        clip.frames[f][0].p = mb(p[0]) if stereo else MbVec3(
            screen.x * rest_span / px_span, -screen.y * rest_span / px_span, 0
        )
        for k in range(1, len(js)):
            aim_joint(clip, f, js[k], cs[k], p[b[k]] - p[a[k]])
        if not stereo:
            var posed = world_pose(
                clip.skeleton, clip.frames[f], MbXform.identity()
            )
            ground = min(ground, min(posed[left].p.y, posed[right].p.y))
    if not stereo:
        for f in range(clip.num_frames()):
            clip.frames[f][0].p.y -= ground
    return clip^


def capture_take(
    reference: MbClip, d: JsonDocument, n: Int, file: String
) raises -> MbClip:
    var arr = field(d, n, "views")
    var nv = d.array_size(arr)
    if nv != 1 and nv != 2:
        raise Error("Capture needs one view or a calibrated camera pair")
    var cache = env_now("HOME") + "/.cache/flint/rtmpose/"
    var net = OnnxModel.load(
        relative(file, string(d, n, "detector", cache + "body17.onnx")), 4
    )
    var first = View(net, d, d.at(arr, 0), file, nv == 2)
    var fps = num(d, n, "fps", Float64(first.fps))
    if fps <= 0 or fps > 240:
        raise Error("Provide capture fps in 0..240 for image sequences")
    var threshold = num(d, n, "confidence", 0.10)
    if threshold <= 0 or threshold > 1:
        raise Error("Use a confidence threshold above zero and at most one")
    if nv == 1:
        var pts = lift(
            first,
            relative(file, string(d, n, "lifter", cache + "lift9.onnx")),
            threshold,
        )
        return fit_soma(reference, pts, first, False, fps)
    var second = View(net, d, d.at(arr, 1), file, True)
    if len(first.poses) != len(second.poses) or (
        first.fps > 0 and second.fps > 0 and abs(first.fps - second.fps) > 0.01
    ):
        raise Error(
            "Stereo cameras need synchronized frames at the same cadence"
        )
    var pts = List[List[Vec3]]()
    for f in range(len(first.poses)):
        pts.append(
            triangulate(
                first, second, f, threshold, num(d, n, "rayTolerance", 0.12)
            )
        )
    var holdout = field(d, n, "validationView")
    if holdout >= 0:
        var v = View(net, d, holdout, file, True)
        if len(v.poses) != len(pts):
            raise Error("Validation camera frame count differs")
        var total: Float64 = 0
        var worst: Float32 = 0
        var seen = 0
        for f in range(len(pts)):
            for j in range(17):
                var ray = pts[f][j] - v.eye
                var depth = ray.dot(v.forward)
                if depth <= 0 or v.poses[f][j].z < threshold:
                    continue
                var dx = (
                    v.cx + v.fx * ray.dot(v.right) / depth - v.poses[f][j].x
                )
                var dy = v.cy - v.fy * ray.dot(v.up) / depth - v.poses[f][j].y
                var e = sqrt(dx * dx + dy * dy)
                total += Float64(e)
                worst = max(worst, e)
                seen += 1
        if not seen:
            raise Error("Validation camera has no visible joints")
        print(
            "Held-out reprojection: mean",
            total / Float64(seen),
            "px, max",
            worst,
            "px,",
            seen,
            "observations",
        )
    return fit_soma(reference, pts, first, True, fps)
