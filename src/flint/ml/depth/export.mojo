"""Outputs of `flint.ml.depth`: depth / normal PNGs, float EXRs, point clouds and mesh blockouts.

  write_depth_png     Turbo-coloured inverse depth (2nd .. 98th percentile), or 16-bit-like
                      grey when `grey` is set
  write_normal_png    camera-space normals as (n + 1) / 2 RGB (OpenCV axes: x right, y down,
                      z forward)
  write_exr           uncompressed scanline OpenEXR, 32-bit float channels (depth "Z",
                      normals / points "R G B")
  camera_points       per-pixel camera-space points from a `DepthResult` (MoGe points as is,
                      DA3 through the decoded intrinsics, others with an assumed 60 degree FOV)
  depth_mesh          grid mesh blockout of a point map: one quad per `step` pixels, cut where
                      neighbouring depths jump by more than `max_jump`, vertex colours from the
                      image, built with `flint.ml.triposg_mesh` normals into a `NativeModel`
  point_splats        the same points as tiny camera-facing triangles (a GLB point cloud)

GLB coordinates are glTF's (y up, camera looking down -z): OpenCV points map to (x, -y, -z).
"""

from std.math import sqrt, tan, floor
from flint.platform.os import write_bytes
from flint.assets.image import write_png
from flint.assets.model import NativeModel, NativeMesh, NativeNode
from flint.assets.glb_writer import write_native_glb
from flint.ml.triposg_mesh import TriMesh, vertex_normals
from flint.ml.depth.model import DepthResult, FAMILY_DA2, FAMILY_DA3, FAMILY_MOGE
from flint.ml.depth.image import Rgb8


def _finite(x: Float32) -> Bool:
    return abs(x) < 1e30


def _percentile(var v: List[Float32], q: Float64) -> Float32:
    if len(v) == 0:
        return 0
    sort(v)
    var i = Int(q * Float64(len(v) - 1))
    return v[min(max(i, 0), len(v) - 1)]


def turbo(t: Float32) -> UInt32:
    """Google Turbo colour map (polynomial fit), t in [0, 1] -> 0xFFRRGGBB."""
    var x = min(max(t, Float32(0)), Float32(1))
    var r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    var g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    var b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    var ri = UInt32(min(max(Int(r * 255 + 0.5), 0), 255))
    var gi = UInt32(min(max(Int(g * 255 + 0.5), 0), 255))
    var bi = UInt32(min(max(Int(b * 255 + 0.5), 0), 255))
    return 0xFF000000 | (ri << 16) | (gi << 8) | bi


def depth_pixels(d: List[Float32], w: Int, h: Int, offset: Int, disparity: Bool, grey: Bool) -> List[UInt32]:
    """A depth (or disparity) map as ARGB pixels: near is warm / bright."""
    var n = w * h
    var inv = List[Float32](length=n, fill=0.0)
    var vals = List[Float32]()
    for i in range(n):
        var v = d[offset + i]
        var ok = _finite(v) and v > 0
        var s: Float32 = v if disparity else (Float32(1) / v if ok else 0)
        inv[i] = s if ok else -1
        if ok:
            vals.append(s)
    var lo = _percentile(vals.copy(), 0.02)
    var hi = _percentile(vals^, 0.98)
    var span = max(hi - lo, Float32(1e-12))
    var px = List[UInt32](length=n, fill=0xFF000000)
    for i in range(n):
        if inv[i] < 0:
            continue
        var t = (inv[i] - lo) / span
        if grey:
            var g = UInt32(min(max(Int(t * 255 + 0.5), 0), 255))
            px[i] = 0xFF000000 | (g << 16) | (g << 8) | g
        else:
            px[i] = turbo(t)
    return px^


def write_depth_png(path: String, r: DepthResult, view: Int = 0, grey: Bool = False) raises:
    var px = depth_pixels(r.depth, r.w, r.h, view * r.w * r.h, r.is_disparity, grey)
    if not write_png(path, px, r.w, r.h):
        raise Error("depth: cannot write " + path)


def write_normal_png(path: String, normal: List[Float32], w: Int, h: Int, offset: Int = 0) raises:
    var px = List[UInt32](length=w * h, fill=0xFF000000)
    for i in range(w * h):
        var c = List[UInt32](length=3, fill=0)
        var any = False
        for k in range(3):
            var v = normal[offset + 3 * i + k]
            if v != 0:
                any = True
            c[k] = UInt32(min(max(Int((v * 0.5 + 0.5) * 255 + 0.5), 0), 255))
        if any:
            px[i] = 0xFF000000 | (c[0] << 16) | (c[1] << 8) | c[2]
    if not write_png(path, px, w, h):
        raise Error("depth: cannot write " + path)


# ----------------------------------------------------------------------------- OpenEXR

def _u8(mut b: List[UInt8], v: Int):
    b.append(UInt8(v & 255))


def _i32(mut b: List[UInt8], v: Int):
    for k in range(4):
        b.append(UInt8((v >> (8 * k)) & 255))


def _u64(mut b: List[UInt8], v: Int):
    for k in range(8):
        b.append(UInt8((v >> (8 * k)) & 255))


def _f32(mut b: List[UInt8], v: Float32):
    var bits = Int(bitcast_f32(v))
    _i32(b, bits)


def bitcast_f32(v: Float32) -> UInt32:
    from std.memory import bitcast
    return bitcast[DType.uint32, 1](SIMD[DType.float32, 1](v))[0]


def _str(mut b: List[UInt8], s: String):
    for i in range(s.byte_length()):
        b.append(s.as_bytes()[i])
    b.append(0)


def _attr(mut b: List[UInt8], name: String, kind: String, var payload: List[UInt8]):
    _str(b, name)
    _str(b, kind)
    _i32(b, len(payload))
    b.extend(payload^)


def encode_exr(names: List[String], planes: List[List[Float32]], w: Int, h: Int, offset: Int = 0) raises -> List[UInt8]:
    """Uncompressed scanline OpenEXR with one FLOAT channel per name (sorted by name as the format
    requires); planes[i] holds w * h values from `offset`."""
    var C = len(names)
    var order = List[Int]()
    for i in range(C):
        order.append(i)
    for i in range(C):
        for j in range(i + 1, C):
            if names[order[j]] < names[order[i]]:
                var t = order[i]
                order[i] = order[j]
                order[j] = t
    var b = List[UInt8]()
    _i32(b, 20000630)
    _i32(b, 2)
    var ch = List[UInt8]()
    for i in range(C):
        _str(ch, names[order[i]])
        _i32(ch, 2)          # FLOAT
        _u8(ch, 0)
        _u8(ch, 0)
        _u8(ch, 0)
        _u8(ch, 0)
        _i32(ch, 1)
        _i32(ch, 1)
    ch.append(0)
    _attr(b, "channels", "chlist", ch^)
    var comp = List[UInt8]()
    _u8(comp, 0)
    _attr(b, "compression", "compression", comp^)
    var box = List[UInt8]()
    _i32(box, 0)
    _i32(box, 0)
    _i32(box, w - 1)
    _i32(box, h - 1)
    _attr(b, "dataWindow", "box2i", box.copy())
    _attr(b, "displayWindow", "box2i", box^)
    var lo = List[UInt8]()
    _u8(lo, 0)
    _attr(b, "lineOrder", "lineOrder", lo^)
    var par = List[UInt8]()
    _f32(par, 1.0)
    _attr(b, "pixelAspectRatio", "float", par^)
    var swc = List[UInt8]()
    _f32(swc, 0.0)
    _f32(swc, 0.0)
    _attr(b, "screenWindowCenter", "v2f", swc^)
    var sww = List[UInt8]()
    _f32(sww, 1.0)
    _attr(b, "screenWindowWidth", "float", sww^)
    b.append(0)
    var line_bytes = 8 + C * w * 4
    var table_start = len(b)
    var first = table_start + 8 * h
    for y in range(h):
        _u64(b, first + y * line_bytes)
    for y in range(h):
        _i32(b, y)
        _i32(b, C * w * 4)
        for i in range(C):
            var p = order[i]
            for x in range(w):
                _f32(b, planes[p][offset + y * w + x])
    return b^


def write_exr(path: String, names: List[String], planes: List[List[Float32]], w: Int, h: Int, offset: Int = 0) raises:
    var b = encode_exr(names, planes, w, h, offset)
    if not write_bytes(path, b):
        raise Error("depth: cannot write " + path)


def write_depth_exr(path: String, r: DepthResult, view: Int = 0) raises:
    var names: List[String] = ["Z"]
    var planes = List[List[Float32]]()
    planes.append(r.depth.copy())
    write_exr(path, names, planes, r.w, r.h, view * r.w * r.h)


def write_vec3_exr(path: String, v: List[Float32], w: Int, h: Int, view: Int = 0) raises:
    """An interleaved [h, w, 3] map (normals, points) as R, G, B float channels."""
    var n = w * h
    var planes = List[List[Float32]]()
    for k in range(3):
        var p = List[Float32](length=n, fill=0.0)
        for i in range(n):
            p[i] = v[(view * n + i) * 3 + k]
        planes.append(p^)
    var names: List[String] = ["R", "G", "B"]
    write_exr(path, names, planes, w, h)


# ----------------------------------------------------------------------------- points and meshes

def camera_points(r: DepthResult, view: Int = 0, fov_deg: Float64 = 60.0) -> List[Float32]:
    """[h, w, 3] OpenCV camera-space points of one view (inf where invalid)."""
    var n = r.w * r.h
    var out = List[Float32](length=n * 3, fill=Float32.MAX * 2)
    if r.family == FAMILY_MOGE and len(r.points) == n * 3 * r.views:
        for i in range(n * 3):
            out[i] = r.points[view * n * 3 + i]
        return out^
    var fx: Float64
    var fy: Float64
    var cx = Float64(r.w) / 2
    var cy = Float64(r.h) / 2
    if r.has_camera:
        fx = r.intrinsics[view * 9]
        fy = r.intrinsics[view * 9 + 4]
        cx = r.intrinsics[view * 9 + 2]
        cy = r.intrinsics[view * 9 + 5]
    else:
        fx = (Float64(r.w) / 2) / tan(fov_deg * 3.141592653589793 / 360.0)
        fy = fx
    # disparity -> depth up to scale (normalised to a median of 1)
    var vals = List[Float32]()
    for i in range(n):
        var v = r.depth[view * n + i]
        if _finite(v) and v > 0:
            vals.append(v)
    var med = _percentile(vals^, 0.5)
    for y in range(r.h):
        for x in range(r.w):
            var i = y * r.w + x
            var v = r.depth[view * n + i]
            if not (_finite(v) and v > 0) or not r.mask[view * n + i]:
                continue
            var d = Float64(v)
            if r.is_disparity:
                d = Float64(med) / d
            out[3 * i] = Float32((Float64(x) + 0.5 - cx) / fx * d)
            out[3 * i + 1] = Float32((Float64(y) + 0.5 - cy) / fy * d)
            out[3 * i + 2] = Float32(d)
    return out^


def to_world(var pts: List[Float32], r: DepthResult, view: Int) -> List[Float32]:
    """Camera points of `view` into the world frame of its DA3 extrinsics (identity otherwise)."""
    if not r.has_camera or r.family != FAMILY_DA3:
        return pts^
    var e = view * 12
    for i in range(len(pts) // 3):
        var x = Float64(pts[3 * i]) - r.extrinsics[e + 3]
        var y = Float64(pts[3 * i + 1]) - r.extrinsics[e + 7]
        var z = Float64(pts[3 * i + 2]) - r.extrinsics[e + 11]
        if not _finite(pts[3 * i + 2]):
            continue
        # R^T (p - t)
        pts[3 * i] = Float32(r.extrinsics[e + 0] * x + r.extrinsics[e + 4] * y + r.extrinsics[e + 8] * z)
        pts[3 * i + 1] = Float32(r.extrinsics[e + 1] * x + r.extrinsics[e + 5] * y + r.extrinsics[e + 9] * z)
        pts[3 * i + 2] = Float32(r.extrinsics[e + 2] * x + r.extrinsics[e + 6] * y + r.extrinsics[e + 10] * z)
    return pts^


def _rgba(img: Rgb8, x: Int, y: Int, w: Int, h: Int) -> Tuple[UInt8, UInt8, UInt8]:
    var sx = min(img.w - 1, x * img.w // w)
    var sy = min(img.h - 1, y * img.h // h)
    var k = (sy * img.w + sx) * 3
    return (img.rgb[k], img.rgb[k + 1], img.rgb[k + 2])


struct ColoredMesh(Movable):
    var mesh: TriMesh
    var colors: List[UInt8]

    def __init__(out self):
        self.mesh = TriMesh()
        self.colors = List[UInt8]()


def append_depth_mesh(mut cm: ColoredMesh, pts: List[Float32], img: Rgb8, w: Int, h: Int, step: Int = 2,
                      max_jump: Float32 = 0.06):
    """Grid triangles over a [h, w, 3] point map (camera depth along z before any world
    transform is the caller's choice: jumps are measured on point distance to the origin)."""
    var gw = (w - 1) // step + 1
    var gh = (h - 1) // step + 1
    var vid = List[Int](length=gw * gh, fill=-1)
    var dist = List[Float32](length=gw * gh, fill=0.0)
    var base = cm.mesh.vertex_count()
    var nv = 0
    for gy in range(gh):
        for gx in range(gw):
            var x = gx * step
            var y = gy * step
            var i = y * w + x
            var px = pts[3 * i]
            var py = pts[3 * i + 1]
            var pz = pts[3 * i + 2]
            if not (_finite(px) and _finite(py) and _finite(pz)):
                continue
            vid[gy * gw + gx] = base + nv
            dist[gy * gw + gx] = sqrt(px * px + py * py + pz * pz)
            nv += 1
            cm.mesh.positions.append(px)
            cm.mesh.positions.append(-py)
            cm.mesh.positions.append(-pz)
            var c = _rgba(img, x, y, w, h)
            cm.colors.append(c[0])
            cm.colors.append(c[1])
            cm.colors.append(c[2])
            cm.colors.append(255)
    for gy in range(gh - 1):
        for gx in range(gw - 1):
            var a = gy * gw + gx
            var b = a + 1
            var c = a + gw
            var d = c + 1
            if vid[a] < 0 or vid[b] < 0 or vid[c] < 0 or vid[d] < 0:
                continue
            var lo = min(min(dist[a], dist[b]), min(dist[c], dist[d]))
            var hi = max(max(dist[a], dist[b]), max(dist[c], dist[d]))
            if hi - lo > max_jump * lo * Float32(step):
                continue
            # counter-clockwise seen from the camera (glTF front faces)
            cm.mesh.indices.append(vid[a])
            cm.mesh.indices.append(vid[c])
            cm.mesh.indices.append(vid[b])
            cm.mesh.indices.append(vid[b])
            cm.mesh.indices.append(vid[c])
            cm.mesh.indices.append(vid[d])


def append_point_splats(mut cm: ColoredMesh, pts: List[Float32], img: Rgb8, w: Int, h: Int, step: Int = 2,
                        size: Float32 = 0.004):
    """One small triangle per `step`-th pixel, sized relative to its distance."""
    for y in range(0, h, step):
        for x in range(0, w, step):
            var i = y * w + x
            var px = pts[3 * i]
            var py = -pts[3 * i + 1]
            var pz = -pts[3 * i + 2]
            if not (_finite(px) and _finite(py) and _finite(pz)):
                continue
            var s = size * sqrt(px * px + py * py + pz * pz) * Float32(step)
            var v0 = cm.mesh.vertex_count()
            cm.mesh.positions.append(px - s)
            cm.mesh.positions.append(py - s * 0.577)
            cm.mesh.positions.append(pz)
            cm.mesh.positions.append(px + s)
            cm.mesh.positions.append(py - s * 0.577)
            cm.mesh.positions.append(pz)
            cm.mesh.positions.append(px)
            cm.mesh.positions.append(py + s * 1.155)
            cm.mesh.positions.append(pz)
            var c = _rgba(img, x, y, w, h)
            for _ in range(3):
                cm.colors.append(c[0])
                cm.colors.append(c[1])
                cm.colors.append(c[2])
                cm.colors.append(255)
            cm.mesh.indices.append(v0)
            cm.mesh.indices.append(v0 + 1)
            cm.mesh.indices.append(v0 + 2)


def colored_model(cm: ColoredMesh, name: String) -> NativeModel:
    var model = NativeModel()
    var mesh = NativeMesh()
    mesh.positions = cm.mesh.positions.copy()
    mesh.normals = vertex_normals(cm.mesh)
    mesh.indices = cm.mesh.indices.copy()
    mesh.colors = cm.colors.copy()
    mesh.base_color = 0xFFFFFFFF
    mesh.roughness = 0.9
    model.meshes.append(mesh^)
    var node = NativeNode()
    node.name = name
    node.mesh = 0
    node.mesh_indices.append(0)
    model.nodes.append(node^)
    model.roots.append(0)
    return model^


def write_glb(path: String, cm: ColoredMesh, name: String) raises:
    var model = colored_model(cm, name)
    if not write_native_glb(path, model):
        raise Error("depth: cannot write " + path)
