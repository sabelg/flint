"""Render Hi3D geometry in the six TexVerse cameras and project material views back to a GLB."""
from std.math import isfinite
from flint.assets.image import NativeImage
from flint.assets.model import NativeMesh, NativeTexture
from flint.core.math import Vec3


def _blender(a: Vec3) -> Vec3:
    return Vec3(a.x, -a.z, a.y)


def _vertex(values: List[Float32], i: Int) -> Vec3:
    return Vec3(values[3 * i], values[3 * i + 1], values[3 * i + 2])


def _camera(v: Vec3, tile: Int) -> Vec3:
    # OpenGL camera axes for canonical views [000,001,003; 002,004,005].
    if tile == 0:
        return Vec3(v.x, v.z, -v.y)
    if tile == 1:
        return Vec3(-v.y, v.z, -v.x)
    if tile == 2:
        return Vec3(v.y, v.z, v.x)
    if tile == 3:
        return Vec3(-v.x, v.z, v.y)
    if tile == 4:
        return Vec3(v.x, v.y, v.z)
    return Vec3(-v.x, v.y, -v.z)


struct UltraTexProjector(Movable):
    var side: Int
    var center: Vec3
    var distance: Float32
    var depth: List[Float32]
    var normals: NativeImage
    var positions: List[Vec3]
    var shading: List[Vec3]
    var faces: List[Int]

    def __init__(out self, mesh: NativeMesh, side: Int) raises:
        if (
            side < 16
            or side > 2048
            or side % 16
            or len(mesh.positions) < 9
            or len(mesh.positions) % 3
            or len(mesh.indices) < 3
            or len(mesh.indices) % 3
            or len(mesh.normals) != len(mesh.positions)
        ):
            raise Error("Expected indexed Hi3D triangles with vertex normals and a tile size divisible by 16")
        self.side = side
        self.faces = mesh.indices.copy()
        self.positions = List[Vec3]()
        self.shading = List[Vec3]()
        var lo = _blender(_vertex(mesh.positions, 0))
        var hi = lo
        for i in range(len(mesh.positions) // 3):
            var p = _blender(_vertex(mesh.positions, i))
            var n = _blender(_vertex(mesh.normals, i))
            if not (
                isfinite(p.x)
                and isfinite(p.y)
                and isfinite(p.z)
                and isfinite(n.x)
                and isfinite(n.y)
                and isfinite(n.z)
            ):
                raise Error("Nonfinite mesh geometry")
            self.positions.append(p)
            self.shading.append(n)
            lo = Vec3(min(lo.x, p.x), min(lo.y, p.y), min(lo.z, p.z))
            hi = Vec3(max(hi.x, p.x), max(hi.y, p.y), max(hi.z, p.z))
        for v in self.faces:
            if v < 0 or v >= len(self.positions):
                raise Error("Mesh index out of bounds")
        self.center = (hi + lo) * 0.5
        self.distance = 0
        for v in self.positions:
            for tile in range(6):
                var cam = _camera(v - self.center, tile)
                self.distance = max(
                    self.distance, cam.z + Float32(1.5625 / 0.46) * max(abs(cam.x), abs(cam.y))
                )
        if self.distance <= 0 or not isfinite(self.distance):
            raise Error("Degenerate mesh bounds")
        self.normals = NativeImage()
        self.normals.valid = True
        self.normals.width = side * 3
        self.normals.height = side * 2
        self.normals.pixels = List[UInt32](length=6 * side * side, fill=0)
        self.depth = List[Float32](length=6 * side * side, fill=0)
        for tile in range(6):
            var projected = List[Vec3](capacity=len(self.positions))
            for p in self.positions:
                projected.append(self.project(p, tile))
            for f in range(0, len(self.faces), 3):
                self._raster(projected, f, tile)

    def project(self, p: Vec3, tile: Int) -> Vec3:
        var c = _camera(p - self.center, tile)
        var inv = 1 / max(Float32(1e-12), self.distance - c.z)
        var scale = Float32(self.side) * 1.5625 * inv
        return Vec3(Float32(self.side) * 0.5 + c.x * scale, Float32(self.side) * 0.5 - c.y * scale, inv)

    def _pixel(self, x: Int, y: Int, tile: Int) -> Int:
        return ((tile // 3) * self.side + y) * 3 * self.side + (tile % 3) * self.side + x

    def _raster(mut self, projected: List[Vec3], f: Int, tile: Int):
        var ia = self.faces[f]
        var ib = self.faces[f + 1]
        var ic = self.faces[f + 2]
        var a = projected[ia]
        var b = projected[ib]
        var c = projected[ic]
        var det = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y)
        if abs(det) < 1e-10:
            return
        for y in range(
            max(0, Int(min(a.y, min(b.y, c.y)))), min(self.side, 1 + Int(max(a.y, max(b.y, c.y))))
        ):
            for x in range(
                max(0, Int(min(a.x, min(b.x, c.x)))), min(self.side, 1 + Int(max(a.x, max(b.x, c.x))))
            ):
                var dx = Float32(x) + 0.5 - c.x
                var dy = Float32(y) + 0.5 - c.y
                var wa = ((b.y - c.y) * dx + (c.x - b.x) * dy) / det
                var wb = ((c.y - a.y) * dx + (a.x - c.x) * dy) / det
                var wc = 1 - wa - wb
                if min(wa, min(wb, wc)) < 0:
                    continue
                wa *= a.z
                wb *= b.z
                wc *= c.z
                var z = wa + wb + wc
                var index = self._pixel(x, y, tile)
                if z <= self.depth[index]:
                    continue
                self.depth[index] = z
                var normal = (
                    self.shading[ia] * wa + self.shading[ib] * wb + self.shading[ic] * wc
                ).normalized()
                var red = UInt32(127.5 * (min(Float32(1), max(Float32(-1), normal.x)) + 1) + 0.5)
                var green = UInt32(127.5 * (min(Float32(1), max(Float32(-1), normal.y)) + 1) + 0.5)
                var blue = UInt32(127.5 * (min(Float32(1), max(Float32(-1), normal.z)) + 1) + 0.5)
                self.normals.pixels[index] = 0xFF000000 | (red << 16) | (green << 8) | blue

    def mesh_with_uvs(self) -> NativeMesh:
        """Choose the most front-facing visible projection per face; split vertices at view seams."""
        var mesh = NativeMesh()
        var mapped = Dict[Int, Int]()
        for f in range(0, len(self.faces), 3):
            var a = self.positions[self.faces[f]]
            var b = self.positions[self.faces[f + 1]]
            var c = self.positions[self.faces[f + 2]]
            var center = (a + b + c) * (1 / Float32(3))
            var normal = (b - a).cross(c - a)
            normal = normal * (1 / max(Float32(1e-30), normal.length()))
            var best = 0
            var score = -Float32.MAX
            for tile in range(6):
                var p = self.project(center, tile)
                var depth = self.depth[
                    self._pixel(
                        min(self.side - 1, max(0, Int(p.x))), min(self.side - 1, max(0, Int(p.y))), tile
                    )
                ]
                var value = _camera(normal, tile).z
                if depth > p.z * (1 + Float32(2) / Float32(self.side)):
                    value -= 3
                if value > score:
                    best = tile
                    score = value
            for j in range(3):
                var v = self.faces[f + j]
                var key = v * 6 + best
                var id = mapped.get(key, -1)
                if id < 0:
                    id = len(mesh.positions) // 3
                    mapped[key] = id
                    var p = self.project(self.positions[v], best)
                    var pos = self.positions[v]
                    var norm = self.shading[v]
                    mesh.positions.extend([pos.x, pos.z, -pos.y])
                    mesh.normals.extend([norm.x, norm.z, -norm.y])
                    mesh.texcoords.append((p.x + Float32((best % 3) * self.side)) / Float32(3 * self.side))
                    mesh.texcoords.append((p.y + Float32((best // 3) * self.side)) / Float32(2 * self.side))
                mesh.indices.append(id)
        return mesh^

    def blend(self, image: NativeImage, linear_data: Bool = False) raises -> NativeTexture:
        """Reproject visible surface samples into a shared six-view texture and pad chart edges."""
        var w = self.normals.width
        var h = self.normals.height
        if not image.valid or image.width != w or image.height != h or len(image.pixels) != w * h:
            raise Error("Material atlas dimensions must match normal views")
        var tex = NativeTexture()
        tex.width = w
        tex.height = h
        tex.pixels = image.pixels.copy()
        var coverage = List[Bool](length=w * h, fill=False)
        var decode = List[Float32]()
        for j in range(256):
            var x = Float32(j) / 255
            decode.append(
                x if linear_data else (x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** Float32(2.4))
            )
        for tile in range(6):
            for y in range(self.side):
                for x in range(self.side):
                    var idx = self._pixel(x, y, tile)
                    if self.depth[idx] <= 0:
                        continue
                    coverage[idx] = True
                    var dist = 1 / self.depth[idx]
                    var scale = dist / (1.5625 * Float32(self.side))
                    var u = (Float32(x) + 0.5 - Float32(self.side) * 0.5) * scale
                    var v = (Float32(self.side) * 0.5 - Float32(y) - 0.5) * scale
                    var z = self.distance - dist
                    var pos = Vec3(u, -z, v)
                    if tile == 1:
                        pos = Vec3(-z, -u, v)
                    elif tile == 2:
                        pos = Vec3(z, u, v)
                    elif tile == 3:
                        pos = Vec3(-u, z, v)
                    elif tile == 4:
                        pos = Vec3(u, v, z)
                    elif tile == 5:
                        pos = Vec3(-u, v, -z)
                    pos = pos + self.center
                    var normal = _unpack_normal(self.normals.pixels[idx])
                    var sum = SIMD[DType.float32, 4](0)
                    var total = Float32(0)
                    for view in range(6):
                        var angle = _camera(normal, view).z
                        if angle <= 0:
                            continue
                        var projected = self.project(pos, view)
                        var xx = Int(projected.x)
                        var yy = Int(projected.y)
                        if min(xx, yy) < 0 or max(xx, yy) >= self.side:
                            continue
                        var sample = self._pixel(xx, yy, view)
                        var depth = self.depth[sample]
                        if (
                            depth <= 0
                            or abs(depth - projected.z) > projected.z * Float32(2) / Float32(self.side)
                            or normal.dot(_unpack_normal(self.normals.pixels[sample])) < 0.8
                        ):
                            continue
                        var weight = angle * angle * angle * angle
                        total += weight
                        comptime for c in range(3):
                            sum[c] += weight * decode[Int((image.pixels[sample] >> UInt32(8 * c)) & 255)]
                    if total > 0:
                        var pixel = UInt32(0xFF000000)
                        comptime for c in range(3):
                            var a = min(Float32(1), max(Float32(0), sum[c] / total))
                            var val = a if linear_data else (
                                12.92 * a if a <= 0.0031308 else 1.055 * (a ** Float32(1 / 2.4)) - 0.055
                            )
                            pixel |= UInt32(min(Float32(255), max(Float32(0), 255 * val + 0.5))) << UInt32(
                                8 * c
                            )
                        tex.pixels[idx] = pixel
        # Expand within each view only, so bilinear sampling at silhouettes sees surface color.
        for _ in range(3):
            var next = coverage.copy()
            for i in range(w * h):
                if coverage[i]:
                    continue
                var x = (i % w) % self.side
                var y = (i // w) % self.side
                for neighbor in range(4):
                    var src = i
                    if neighbor == 0 and x > 0:
                        src -= 1
                    elif neighbor == 1 and x + 1 < self.side:
                        src += 1
                    elif neighbor == 2 and y > 0:
                        src -= w
                    elif neighbor == 3 and y + 1 < self.side:
                        src += w
                    if coverage[src]:
                        tex.pixels[i] = tex.pixels[src]
                        next[i] = True
                        break
            coverage = next^
        return tex^


def _unpack_normal(pixel: UInt32) -> Vec3:
    return Vec3(
        Float32((pixel >> 16) & 255) / 127.5 - 1,
        Float32((pixel >> 8) & 255) / 127.5 - 1,
        Float32(pixel & 255) / 127.5 - 1,
    )
