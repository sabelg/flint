"""SigLIP image input (`flint.ml.siglip.image`): PIL-exact resampling and GLB thumbnails.

`siglip_preprocess(img, size)` is `SiglipImageProcessor` with the PIL backend: RGBA is composited
over white (transformers' `convert_to_rgb`), the 8-bit image is resized to size x size with PIL's
fixed-point two-pass `Image.resize` (`resample=2`, BILINEAR: a triangle filter widened by the
scale when downsampling, 22-bit coefficients, horizontal pass then vertical, each rounded and
clamped to 8 bits), then x / 255 normalised with mean = std = 0.5 to channels-last [S, S, 3].
`resample=3` selects PIL's BICUBIC (a = -0.5) the same way.

`render_glb_thumbnail(model, size)` draws a GLB into an image with `flint.render.simd_canvas`
(no GPU or window): world transforms from the node hierarchy, the model framed from a
three-quarter view over a light backdrop, per-triangle Lambert shading of the base colour times
the vertex colour and the base-colour texture sampled at the triangle's centroid UV.
"""

from std.math import ceil, floor, sqrt
from flint.assets.image import NativeImage
from flint.assets.model import NativeModel
from flint.core.math import Vec3, Mat4
from flint.render.simd_canvas import Camera3D, SimdCanvas, shade_rgb

comptime RESAMPLE_BILINEAR: Int = 2
comptime RESAMPLE_BICUBIC: Int = 3
comptime _PRECISION_BITS: Int = 22


def _filter(x0: Float64, kind: Int) -> Float64:
    var x = abs(x0)
    if kind == RESAMPLE_BICUBIC:
        var a: Float64 = -0.5
        if x < 1.0:
            return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        if x < 2.0:
            return (((x - 5.0) * x + 8.0) * x - 4.0) * a
        return 0.0
    if x < 1.0:
        return 1.0 - x
    return 0.0


struct _Coeffs(Movable):
    var lo: List[Int]
    var n: List[Int]
    var k: List[Int]       # fixed-point weights, ksize per output
    var ksize: Int

    def __init__(out self):
        self.lo = List[Int]()
        self.n = List[Int]()
        self.k = List[Int]()
        self.ksize = 0


def _coeffs(in_size: Int, out_size: Int, kind: Int) -> _Coeffs:
    """PIL `precompute_coeffs` + `normalize_coeffs_8bpc`."""
    var support0: Float64 = 2.0 if kind == RESAMPLE_BICUBIC else 1.0
    var scale = Float64(in_size) / Float64(out_size)
    var fscale = max(scale, 1.0)
    var support = support0 * fscale
    var c = _Coeffs()
    c.ksize = Int(ceil(support)) * 2 + 1
    var ss = 1.0 / fscale
    var w = List[Float64](length=c.ksize, fill=0.0)
    for xx in range(out_size):
        var center = (Float64(xx) + 0.5) * scale
        var xmin = max(Int(center - support + 0.5), 0)
        var xmax = min(Int(center + support + 0.5), in_size) - xmin
        var total: Float64 = 0.0
        for x in range(c.ksize):
            w[x] = 0.0
        for x in range(xmax):
            var v = _filter((Float64(x + xmin) - center + 0.5) * ss, kind)
            w[x] = v
            total += v
        for x in range(c.ksize):
            var v = w[x] / total if (x < xmax and total != 0.0) else 0.0
            var f = v * Float64(1 << _PRECISION_BITS)
            c.k.append(Int(f - 0.5) if v < 0.0 else Int(f + 0.5))
        c.lo.append(xmin)
        c.n.append(xmax)
    return c^


@always_inline
def _clip8(v: Int) -> UInt8:
    var s = v >> _PRECISION_BITS
    if s < 0:
        return 0
    if s > 255:
        return 255
    return UInt8(s)


def pil_resize_rgb8(src: List[UInt8], w: Int, h: Int, ow: Int, oh: Int, kind: Int = RESAMPLE_BILINEAR) -> List[UInt8]:
    """PIL `Image.resize((ow, oh), kind)` of an 8-bit RGB image ([h, w, 3] bytes)."""
    var cur = src.copy()
    var cw = w
    if ow != w:
        var cx = _coeffs(w, ow, kind)
        var tmp = List[UInt8](length=h * ow * 3, fill=0)
        for y in range(h):
            for xx in range(ow):
                var lo = cx.lo[xx]
                for ch in range(3):
                    var acc = 1 << (_PRECISION_BITS - 1)
                    for x in range(cx.n[xx]):
                        acc += Int(cur[(y * cw + lo + x) * 3 + ch]) * cx.k[xx * cx.ksize + x]
                    tmp[(y * ow + xx) * 3 + ch] = _clip8(acc)
        cur = tmp^
        cw = ow
    if oh != h:
        var cy = _coeffs(h, oh, kind)
        var out = List[UInt8](length=oh * cw * 3, fill=0)
        for yy in range(oh):
            var lo = cy.lo[yy]
            for x in range(cw):
                for ch in range(3):
                    var acc = 1 << (_PRECISION_BITS - 1)
                    for y in range(cy.n[yy]):
                        acc += Int(cur[((lo + y) * cw + x) * 3 + ch]) * cy.k[yy * cy.ksize + y]
                    out[(yy * cw + x) * 3 + ch] = _clip8(acc)
        cur = out^
    return cur^


def image_rgb8(img: NativeImage, background: UInt32 = 0xFFFFFF) -> List[UInt8]:
    """[h, w, 3] bytes from 0xAARRGGBB pixels; alpha is composited over `background` the way
    PIL's `alpha_composite` onto an opaque image does (rounded 8-bit blend)."""
    var n = img.width * img.height
    var out = List[UInt8](length=n * 3, fill=0)
    var bg_r = Int((background >> 16) & 255)
    var bg_g = Int((background >> 8) & 255)
    var bg_b = Int(background & 255)
    for i in range(n):
        var p = img.pixels[i]
        var a = Int((p >> 24) & 255)
        var r = Int((p >> 16) & 255)
        var g = Int((p >> 8) & 255)
        var b = Int(p & 255)
        if a < 255:
            r = _blend(r, bg_r, a)
            g = _blend(g, bg_g, a)
            b = _blend(b, bg_b, a)
        out[i * 3] = UInt8(r)
        out[i * 3 + 1] = UInt8(g)
        out[i * 3 + 2] = UInt8(b)
    return out^


@always_inline
def _blend(src: Int, dst: Int, a: Int) -> Int:
    """PIL `ImagingAlphaComposite` over an opaque destination (7-bit coefficients)."""
    var x = 128 * (src * a + dst * (255 - a)) + (0x80 << 7)
    return (((x >> 8) + x) >> 8) >> 7


def siglip_preprocess_rgb8(rgb: List[UInt8], w: Int, h: Int, size: Int, kind: Int = RESAMPLE_BILINEAR) -> List[Float32]:
    var r = pil_resize_rgb8(rgb, w, h, size, size, kind)
    var out = List[Float32](length=size * size * 3, fill=0.0)
    for i in range(size * size * 3):
        # (x * (1/255) - 0.5) / 0.5, in float32 like the torch / numpy pipeline
        var v = Float32(r[i]) * Float32(0.00392156862745098)
        out[i] = (v - 0.5) / 0.5
    return out^


def siglip_preprocess(img: NativeImage, size: Int, kind: Int = RESAMPLE_BILINEAR) -> List[Float32]:
    """A decoded image -> normalised channels-last [size, size, 3] SigLIP input."""
    return siglip_preprocess_rgb8(image_rgb8(img), img.width, img.height, size, kind)


# ----------------------------------------------------------------------------- GLB thumbnails

def _world(model: NativeModel, node: Int, parent: Mat4, mut out: List[Mat4], depth: Int):
    if node < 0 or node >= len(model.nodes) or depth > 64:
        return
    var m = parent.multiply(model.nodes[node].local_transform())
    out[node] = m.copy()
    for c in model.nodes[node].children:
        _world(model, c, m, out, depth + 1)


def _sample(model: NativeModel, tex: Int, u: Float32, v: Float32) -> UInt32:
    if tex < 0 or tex >= len(model.textures):
        return 0xFFFFFFFF
    var t = model.textures[tex].copy()
    if t.width <= 0 or t.height <= 0 or len(t.pixels) < t.width * t.height:
        return 0xFFFFFFFF
    var fu = u - floor(u)
    var fv = v - floor(v)
    var x = min(t.width - 1, max(0, Int(fu * Float32(t.width))))
    var y = min(t.height - 1, max(0, Int(fv * Float32(t.height))))
    return t.pixels[y * t.width + x]


@always_inline
def _mul_rgb(a: UInt32, b: UInt32) -> UInt32:
    var r = ((a >> 16) & 255) * ((b >> 16) & 255) // 255
    var g = ((a >> 8) & 255) * ((b >> 8) & 255) // 255
    var bl = (a & 255) * (b & 255) // 255
    return (r << 16) | (g << 8) | bl


def render_glb_thumbnail(model: NativeModel, size: Int = 448, background: UInt32 = 0xF2F2F2) -> NativeImage:
    """A `size` x `size` software render of every mesh in `model` (see module docs)."""
    var img = NativeImage()
    img.width = size
    img.height = size
    var canvas = SimdCanvas(size, size)
    canvas.clear(background, background)
    # world matrices: meshes referenced by nodes, else drawn in model space
    var world = List[Mat4]()
    for _ in range(len(model.nodes)):
        world.append(Mat4())
    for r in model.roots:
        _world(model, r, Mat4(), world, 0)
    var mesh_mats = List[Mat4]()
    var mesh_ids = List[Int]()
    for n in range(len(model.nodes)):
        for mi in model.nodes[n].mesh_indices:
            if mi >= 0 and mi < len(model.meshes):
                mesh_ids.append(mi)
                mesh_mats.append(world[n].copy())
    if len(mesh_ids) == 0:
        for mi in range(len(model.meshes)):
            mesh_ids.append(mi)
            mesh_mats.append(Mat4())
    # bounds
    var lo = Vec3(1e30, 1e30, 1e30)
    var hi = Vec3(-1e30, -1e30, -1e30)
    var any = False
    for k in range(len(mesh_ids)):
        var mesh = model.meshes[mesh_ids[k]].copy()
        for vi in range(len(mesh.positions) // 3):
            var p = mesh_mats[k].transform_point(Vec3(mesh.positions[vi * 3], mesh.positions[vi * 3 + 1],
                                                      mesh.positions[vi * 3 + 2]))
            lo = Vec3(min(lo.x, p.x), min(lo.y, p.y), min(lo.z, p.z))
            hi = Vec3(max(hi.x, p.x), max(hi.y, p.y), max(hi.z, p.z))
            any = True
    if not any:
        img.pixels = canvas.pixels.copy()
        return img^
    var center = (lo + hi) * 0.5
    var ext = hi - lo
    var radius = max(Float32(1e-4), Float32(0.5) * sqrt(ext.x * ext.x + ext.y * ext.y + ext.z * ext.z))
    var fov: Float32 = 35.0
    var dist = radius / Float32(0.30) * Float32(1.08)    # tan(35 deg / 2) ~ 0.315
    var dir = Vec3(0.62, 0.42, 0.66).normalized()
    var cam = Camera3D(size, size, center + dir * dist, center, fov)
    cam.near_z = max(Float32(1e-4), dist - radius * 2.0)
    var sun = Vec3(0.35, 0.85, 0.45)
    for k in range(len(mesh_ids)):
        var mesh = model.meshes[mesh_ids[k]].copy()
        var nv = len(mesh.positions) // 3
        var tri = len(mesh.indices) // 3
        var has_uv = len(mesh.texcoords) >= nv * 2 and mesh.texture >= 0
        var has_col = len(mesh.colors) >= nv * 4
        for t in range(tri):
            var i0 = mesh.indices[t * 3]
            var i1 = mesh.indices[t * 3 + 1]
            var i2 = mesh.indices[t * 3 + 2]
            if i0 < 0 or i1 < 0 or i2 < 0 or i0 >= nv or i1 >= nv or i2 >= nv:
                continue
            var p0 = mesh_mats[k].transform_point(Vec3(mesh.positions[i0 * 3], mesh.positions[i0 * 3 + 1], mesh.positions[i0 * 3 + 2]))
            var p1 = mesh_mats[k].transform_point(Vec3(mesh.positions[i1 * 3], mesh.positions[i1 * 3 + 1], mesh.positions[i1 * 3 + 2]))
            var p2 = mesh_mats[k].transform_point(Vec3(mesh.positions[i2 * 3], mesh.positions[i2 * 3 + 1], mesh.positions[i2 * 3 + 2]))
            var col = mesh.base_color & 0xFFFFFF
            if has_col:
                var cr = (UInt32(mesh.colors[i0 * 4]) + UInt32(mesh.colors[i1 * 4]) + UInt32(mesh.colors[i2 * 4])) // 3
                var cg = (UInt32(mesh.colors[i0 * 4 + 1]) + UInt32(mesh.colors[i1 * 4 + 1]) + UInt32(mesh.colors[i2 * 4 + 1])) // 3
                var cb = (UInt32(mesh.colors[i0 * 4 + 2]) + UInt32(mesh.colors[i1 * 4 + 2]) + UInt32(mesh.colors[i2 * 4 + 2])) // 3
                col = _mul_rgb(col, (cr << 16) | (cg << 8) | cb)
            if has_uv:
                var u = (mesh.texcoords[i0 * 2] + mesh.texcoords[i1 * 2] + mesh.texcoords[i2 * 2]) / 3.0
                var v = (mesh.texcoords[i0 * 2 + 1] + mesh.texcoords[i1 * 2 + 1] + mesh.texcoords[i2 * 2 + 1]) / 3.0
                col = _mul_rgb(col, _sample(model, mesh.texture, u, v))
            canvas.draw_tri_3d_lit(cam, p0, p1, p2, col, sun)
    img.pixels = canvas.pixels.copy()
    for i in range(len(img.pixels)):
        img.pixels[i] = img.pixels[i] | 0xFF000000
    return img^
