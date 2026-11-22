"""Author a radial fern from one upright, isolated frond (stem at the bottom)."""
from std.math import cos, sin
from flint.assets.image import NativeImage
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh, NativeTexture
from flint.core.math import Vec3
from flint.procedural.foliage import _rand01_fol

comptime _BANDS = 16


struct FernReference(Movable):
    # Half-width as a fraction of frond length, sampled from root to tip.
    var widths: List[Float32]
    var atlas: NativeTexture

    def __init__(out self, image: NativeImage) raises:
        var w = image.width
        var h = image.height
        if not image.valid or w < 2 or h < 2 or len(image.pixels) != w * h:
            raise Error("Expected a decoded frond image at least 2 by 2 pixels")
        var transparent = False
        for pixel in image.pixels:
            if pixel >> 24 < 128:
                transparent = True
                break
        var backdrop = image.pixels[0]
        var left = List[Int](length=h, fill=w)
        var right = List[Int](length=h, fill=-1)
        var reds = List[Int](length=h, fill=0)
        var greens = reds.copy()
        var blues = reds.copy()
        var counts = reds.copy()
        var top = h
        var bottom = -1
        for y in range(h):
            for x in range(w):
                var p = image.pixels[y * w + x]
                var r = Int((p >> 16) & 255)
                var g = Int((p >> 8) & 255)
                var b = Int(p & 255)
                if transparent:
                    if p >> 24 < 128:
                        continue
                elif (
                    max(
                        abs(r - Int((backdrop >> 16) & 255)),
                        max(
                            abs(g - Int((backdrop >> 8) & 255)),
                            abs(b - Int(backdrop & 255)),
                        ),
                    )
                    < 36
                ):
                    continue
                left[y] = min(left[y], x)
                right[y] = max(right[y], x)
                reds[y] += r
                greens[y] += g
                blues[y] += b
                counts[y] += 1
                top = min(top, y)
                bottom = max(bottom, y)
        var height = bottom - top + 1
        if height < _BANDS:
            raise Error(
                "Isolate one upright frond at least 16 foreground pixels tall"
            )
        self.widths = List[Float32]()
        self.atlas = NativeTexture()
        self.atlas.width = 4
        self.atlas.height = _BANDS
        for band in range(_BANDS):
            var widest = 0
            var n = 0
            var rgb = Vec3()
            for offset in range(
                band * height // _BANDS, (band + 1) * height // _BANDS
            ):
                var y = bottom - offset
                widest = max(widest, right[y] - left[y] + 1)
                n += counts[y]
                rgb = rgb + Vec3(
                    Float32(reds[y]), Float32(greens[y]), Float32(blues[y])
                )
            if n == 0:
                raise Error("Foreground has a gap across an entire frond band")
            self.widths.append(Float32(widest) * 0.5 / Float32(height))
            rgb = rgb * (1 / Float32(n))
            for column in range(4):
                # Dark margin and lighter midrib, shared by all leaflets and LODs.
                var tint = Float32(
                    1.14
                ) if column == 1 or column == 2 else Float32(0.83)
                self.atlas.pixels.append(
                    UInt32(0xFF000000)
                    | (UInt32(min(Float32(255), rgb.x * tint)) << 16)
                    | (UInt32(min(Float32(255), rgb.y * tint)) << 8)
                    | UInt32(min(Float32(255), rgb.z * tint))
                )


def _spine(radial: Vec3, t: Float32, length: Float32) -> Vec3:
    return (
        radial * (0.035 + 0.92 * t * t) + Vec3(0, 1.48 * t - 0.85 * t * t, 0)
    ) * length


def _vertex(
    mut mb: MeshBuilder, p: Vec3, normal: Vec3, u: Float32, t: Float32
) -> Int:
    var v = mb.add_vertex(p, normal, u, (0.5 + t * (_BANDS - 1)) / _BANDS)
    # Wind influence is exactly zero at the root.
    mb.mesh.texcoords1.append(t * t)
    mb.mesh.texcoords1.append(0)
    return v


def build_reference_fern(
    reference: FernReference, lod: Int = 0, seed: Int = 1
) raises -> NativeMesh:
    """Build one opaque, double-sided mesh. Bind reference.atlas at texture slot zero.
    """
    if lod < 0 or lod > 2:
        raise Error("Fern LOD must be 0, 1 or 2")
    var mb = MeshBuilder(
        texture=0, metallic=0, roughness=0.74, double_sided=True
    )
    var pairs = 8 >> lod
    var up = Vec3(0, 1, 0)
    for i in range(7):
        var angle = Float32(i) * 2.399963 + 0.45 * _rand01_fol(seed, i)
        var length = 0.72 + 0.46 * _rand01_fol(seed, i + 7)
        var radial = Vec3(cos(angle), 0, sin(angle))
        var side = Vec3(-radial.z, 0, radial.x)
        var prev_l = 0
        var prev_r = 0
        for j in range(pairs + 1):
            var t = Float32(j) / Float32(pairs)
            var p = _spine(radial, t, length)
            var half_stem = length * (0.009 - 0.007 * t)
            var l = _vertex(mb, p - side * half_stem, up, 0.39, t)
            var r = _vertex(mb, p + side * half_stem, up, 0.61, t)
            if j > 0:
                mb.add_quad(prev_l, prev_r, r, l)
            prev_l = l
            prev_r = r
        for j in range(pairs):
            var t = 0.10 + 0.84 * (Float32(j) + 0.5) / Float32(pairs)
            var half_span = reference.widths[Int(t * _BANDS)] * length
            # Attach to the rendered stem segment, including on the coarse mesh.
            var segment = Int(t * Float32(pairs))
            var start = _spine(
                radial, Float32(segment) / Float32(pairs), length
            )
            var end = _spine(
                radial, Float32(segment + 1) / Float32(pairs), length
            )
            var p = start + (end - start) * (
                t * Float32(pairs) - Float32(segment)
            )
            var tangent = (
                radial * (1.84 * t) + up * (1.48 - 1.7 * t)
            ).normalized()
            var normal = (
                up * (1.84 * t) - radial * (1.48 - 1.7 * t)
            ).normalized()
            for sign in [-1, 1]:
                var outward = side * Float32(sign)
                var axis = outward * half_span + tangent * (half_span * 0.55)
                var width = min(
                    half_span * 0.28, length * 0.53 / Float32(pairs)
                )
                var mid = p + axis * 0.42
                var a = _vertex(mb, p, normal, 0.5, t)
                var b = _vertex(
                    mb, mid - tangent * width, normal - tangent * 0.22, 0.125, t
                )
                var c = _vertex(
                    mb,
                    p + axis + normal * (half_span * 0.10),
                    normal,
                    0.5,
                    min(Float32(1), t + 0.08),
                )
                var d = _vertex(
                    mb, mid + tangent * width, normal + tangent * 0.22, 0.875, t
                )
                if lod == 0:
                    var ridge = _vertex(
                        mb, mid + normal * width * 0.35, normal, 0.5, t
                    )
                    mb.add_tri(a, b, ridge)
                    mb.add_tri(b, c, ridge)
                    mb.add_tri(c, d, ridge)
                    mb.add_tri(d, a, ridge)
                else:
                    mb.add_quad(a, b, c, d)
    var mesh = mb^.finish()
    mesh.ensure_bounds()
    return mesh^
