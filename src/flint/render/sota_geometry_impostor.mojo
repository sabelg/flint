"""Sota:geometry — octahedral impostors for distant foliage and crowds, baked in pure Mojo.

Bake (CPU, offline or at load): the mesh is rasterised orthographically from `N x N` view directions laid out on
a hemi-octahedron (grid cell (i, j) -> `(u, v)` in [-1, 1]^2 -> `p = ((u + v) / 2, (u - v) / 2)`,
`dir = normalize(p.x, 1 - |p.x| - |p.y|, p.y)`), each into a `res x res` cell of one atlas:
  albedo  sRGB colour (material base colour x vertex colour) + coverage alpha, edge-dilated so mips do not
          bleed the background into the silhouette;
  normal  the surface normal in the cell's tangent frame (T = screen right, B = screen down = d/dv, N = view
          direction), so the renderer's ordinary normal-mapping path shades the card like the mesh.

Runtime: one card mesh per cell, facing that cell's direction with the cell's UV rectangle (alpha-masked material
with the two textures). Drawing an impostor = picking the cell nearest to the object-space view direction and
issuing an ordinary `draw_instance` of that card, so impostors go through the visbuffer, alpha-cutout, RT shadow
and GI paths with no new pass and no per-frame mesh rebuild. `impostor_lod_select` switches mesh <-> impostor by
projected size.

Sources: Ryan Brucks "Octahedral Impostors" (2018 write-up); Engel et al. on hemi-octahedral mapping;
Cigolle et al. "A Survey of Efficient Representations for Independent Unit Vectors" (JCGT 2014). No code copied.
"""
from std.collections import List
from std.math import abs, max, min, sqrt
from flint.assets.model import NativeMesh
from flint.assets.mesh_builder import MeshBuilder
from flint.core.math import Vec3


struct OctImpostorBake(Copyable, Movable):
    """Baked atlas (`frames * res` square): sRGB ARGB albedo with coverage alpha + tangent-space normals."""
    var frames: Int
    var res: Int
    var dim: Int
    var center: Vec3
    var radius: Float32
    var albedo: List[UInt32]
    var normals: List[Vec3]
    var coverage: Float32          # fraction of covered texels (diagnostics)

    def __init__(out self, frames: Int, res: Int, center: Vec3, radius: Float32):
        self.frames = frames
        self.res = res
        self.dim = frames * res
        self.center = center
        self.radius = radius
        self.albedo = List[UInt32](length=self.dim * self.dim, fill=UInt32(0))
        self.normals = List[Vec3](length=self.dim * self.dim, fill=Vec3(0.0, 0.0, 1.0))
        self.coverage = 0.0


def hemi_oct_dir(frames: Int, fx: Int, fy: Int) -> Vec3:
    """View direction (object space, pointing from the object to the eye) of grid cell (fx, fy)."""
    var u = (Float32(fx) + Float32(0.5)) / Float32(frames) * Float32(2.0) - Float32(1.0)
    var v = (Float32(fy) + Float32(0.5)) / Float32(frames) * Float32(2.0) - Float32(1.0)
    var px = (u + v) * Float32(0.5)
    var pz = (u - v) * Float32(0.5)
    var py = Float32(1.0) - abs(px) - abs(pz)
    return Vec3(px, max(py, Float32(0.0)), pz).normalized()


def hemi_oct_cell(frames: Int, view_dir: Vec3) -> Tuple[Int, Int]:
    """Nearest grid cell of an object-space view direction (below-horizon directions clamp to the horizon)."""
    var d = Vec3(view_dir.x, max(view_dir.y, Float32(0.0)), view_dir.z)
    var s = abs(d.x) + abs(d.y) + abs(d.z)
    if s < Float32(1e-8):
        d = Vec3(0.0, 1.0, 0.0)
        s = Float32(1.0)
    var px = d.x / s
    var pz = d.z / s
    var u = px + pz
    var v = px - pz
    var fx = Int((u * Float32(0.5) + Float32(0.5)) * Float32(frames))
    var fy = Int((v * Float32(0.5) + Float32(0.5)) * Float32(frames))
    return (max(0, min(frames - 1, fx)), max(0, min(frames - 1, fy)))


def impostor_basis(view_dir: Vec3) -> Tuple[Vec3, Vec3]:
    """(right, up) of the card facing `view_dir`; world up is +y (falls back to +z when looking straight down)."""
    var up_ref = Vec3(0.0, 1.0, 0.0)
    if abs(view_dir.y) > Float32(0.999):
        up_ref = Vec3(0.0, 0.0, -1.0)
    var r = up_ref.cross(view_dir).normalized()
    var u = view_dir.cross(r).normalized()
    return (r, u)


def _srgb8_to_lin(v: Int) -> Float32:
    var c = Float32(v) / Float32(255.0)
    return c / Float32(12.92) if c <= Float32(0.04045) else ((c + Float32(0.055)) / Float32(1.055)) ** Float32(2.4)


def _lin_to_srgb8(v: Float32) -> Int:
    var c = max(Float32(0.0), min(Float32(1.0), v))
    var s = c * Float32(12.92) if c <= Float32(0.0031308) else Float32(1.055) * c ** (Float32(1.0) / Float32(2.4)) - Float32(0.055)
    return Int(s * Float32(255.0) + Float32(0.5))


def bake_octahedral_impostor(mesh: NativeMesh, base_color_lin: Vec3, frames: Int = 8, res: Int = 64) -> OctImpostorBake:
    """Rasterises `mesh` from `frames x frames` hemi-octahedral directions (1 sample per texel, z-buffered)."""
    var nv = len(mesh.positions) // 3
    var lo = Vec3(1e30, 1e30, 1e30)
    var hi = Vec3(-1e30, -1e30, -1e30)
    for i in range(nv):
        var p = Vec3(mesh.positions[i * 3], mesh.positions[i * 3 + 1], mesh.positions[i * 3 + 2])
        lo = Vec3(min(lo.x, p.x), min(lo.y, p.y), min(lo.z, p.z))
        hi = Vec3(max(hi.x, p.x), max(hi.y, p.y), max(hi.z, p.z))
    var c = (lo + hi) * Float32(0.5)
    var rad = Float32(0.0)
    for i in range(nv):
        var p = Vec3(mesh.positions[i * 3], mesh.positions[i * 3 + 1], mesh.positions[i * 3 + 2])
        rad = max(rad, (p - c).length())
    rad *= Float32(1.01)
    var out = OctImpostorBake(frames, res, c, rad)
    var has_col = len(mesh.colors) >= nv * 4
    # linear vertex albedo
    var vcol = List[Vec3](capacity=nv)
    for i in range(nv):
        if has_col:
            vcol.append(Vec3(
                base_color_lin.x * _srgb8_to_lin(Int(mesh.colors[i * 4])),
                base_color_lin.y * _srgb8_to_lin(Int(mesh.colors[i * 4 + 1])),
                base_color_lin.z * _srgb8_to_lin(Int(mesh.colors[i * 4 + 2])),
            ))
        else:
            vcol.append(base_color_lin)
    var zbuf = List[Float32](length=res * res, fill=Float32(-1e30))
    var colb = List[Vec3](length=res * res, fill=Vec3(0.0, 0.0, 0.0))
    var nrmb = List[Vec3](length=res * res, fill=Vec3(0.0, 0.0, 1.0))
    var covered = 0
    var ntri = len(mesh.indices) // 3
    var sx = List[Float32](length=nv, fill=0.0)
    var sy = List[Float32](length=nv, fill=0.0)
    var sz = List[Float32](length=nv, fill=0.0)
    for fy in range(frames):
        for fx in range(frames):
            var vd = hemi_oct_dir(frames, fx, fy)
            var basis = impostor_basis(vd)
            var r = basis[0]
            var u = basis[1]
            var bvec = Vec3(-u.x, -u.y, -u.z)          # +v of the texture runs down the card
            for i in range(res * res):
                zbuf[i] = Float32(-1e30)
            var scale = Float32(res) / (Float32(2.0) * rad)
            for i in range(nv):
                var d = Vec3(mesh.positions[i * 3], mesh.positions[i * 3 + 1], mesh.positions[i * 3 + 2]) - c
                sx[i] = (d.dot(r) + rad) * scale
                sy[i] = (rad - d.dot(u)) * scale
                sz[i] = d.dot(vd)
            for t in range(ntri):
                var a = mesh.indices[t * 3]
                var b = mesh.indices[t * 3 + 1]
                var cc = mesh.indices[t * 3 + 2]
                var area = (sx[b] - sx[a]) * (sy[cc] - sy[a]) - (sx[cc] - sx[a]) * (sy[b] - sy[a])
                if abs(area) < Float32(1e-12):
                    continue
                var x0 = max(0, Int(min(sx[a], min(sx[b], sx[cc]))))
                var x1 = min(res - 1, Int(max(sx[a], max(sx[b], sx[cc]))))
                var y0 = max(0, Int(min(sy[a], min(sy[b], sy[cc]))))
                var y1 = min(res - 1, Int(max(sy[a], max(sy[b], sy[cc]))))
                var inv = Float32(1.0) / area
                for py in range(y0, y1 + 1):
                    var cy = Float32(py) + Float32(0.5)
                    for px in range(x0, x1 + 1):
                        var cx = Float32(px) + Float32(0.5)
                        var w0 = ((sx[b] - cx) * (sy[cc] - cy) - (sx[cc] - cx) * (sy[b] - cy)) * inv
                        var w1 = ((sx[cc] - cx) * (sy[a] - cy) - (sx[a] - cx) * (sy[cc] - cy)) * inv
                        var w2 = Float32(1.0) - w0 - w1
                        if w0 < Float32(0.0) or w1 < Float32(0.0) or w2 < Float32(0.0):
                            continue
                        var z = w0 * sz[a] + w1 * sz[b] + w2 * sz[cc]
                        var k = py * res + px
                        if z <= zbuf[k]:
                            continue
                        zbuf[k] = z
                        colb[k] = vcol[a] * w0 + vcol[b] * w1 + vcol[cc] * w2
                        var n = Vec3(
                            mesh.normals[a * 3] * w0 + mesh.normals[b * 3] * w1 + mesh.normals[cc * 3] * w2,
                            mesh.normals[a * 3 + 1] * w0 + mesh.normals[b * 3 + 1] * w1 + mesh.normals[cc * 3 + 1] * w2,
                            mesh.normals[a * 3 + 2] * w0 + mesh.normals[b * 3 + 2] * w1 + mesh.normals[cc * 3 + 2] * w2,
                        ).normalized()
                        if n.dot(vd) < Float32(0.0):
                            n = n * Float32(-1.0)       # back faces of open / double-sided geometry
                        nrmb[k] = Vec3(n.dot(r), n.dot(bvec), max(Float32(0.05), n.dot(vd))).normalized()
            # dilate colour/normal into empty texels (3 rings) so filtering never pulls in black
            var filled = List[Bool](length=res * res, fill=False)
            for i in range(res * res):
                filled[i] = zbuf[i] > Float32(-1e29)
            var cov = List[Bool](length=res * res, fill=False)
            for i in range(res * res):
                cov[i] = filled[i]
            for _ in range(3):
                var nxt = filled.copy()
                for py in range(res):
                    for px in range(res):
                        var k = py * res + px
                        if filled[k]:
                            continue
                        var acc = Vec3(0.0, 0.0, 0.0)
                        var accn = Vec3(0.0, 0.0, 0.0)
                        var cnt = 0
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                var qx = px + dx
                                var qy = py + dy
                                if qx < 0 or qy < 0 or qx >= res or qy >= res:
                                    continue
                                var q = qy * res + qx
                                if filled[q]:
                                    acc = acc + colb[q]
                                    accn = accn + nrmb[q]
                                    cnt += 1
                        if cnt > 0:
                            colb[k] = acc * (Float32(1.0) / Float32(cnt))
                            nrmb[k] = accn.normalized()
                            nxt[k] = True
                filled = nxt^
            for py in range(res):
                for px in range(res):
                    var k = py * res + px
                    var dst = (fy * res + py) * out.dim + fx * res + px
                    var a8 = UInt32(255) if cov[k] else UInt32(0)
                    if cov[k]:
                        covered += 1
                    var cl = colb[k]
                    out.albedo[dst] = (a8 << 24) | (UInt32(_lin_to_srgb8(cl.x)) << 16) | (UInt32(_lin_to_srgb8(cl.y)) << 8) | UInt32(_lin_to_srgb8(cl.z))
                    out.normals[dst] = nrmb[k] if filled[k] else Vec3(0.0, 0.0, 1.0)
    out.coverage = Float32(covered) / Float32(out.dim * out.dim)
    return out^


def build_impostor_cards(bake: OctImpostorBake) -> List[NativeMesh]:
    """One alpha-masked card per cell: centred on the bake centre, facing the cell direction, UVs = the cell."""
    var out = List[NativeMesh]()
    var inv = Float32(1.0) / Float32(bake.frames)
    for fy in range(bake.frames):
        for fx in range(bake.frames):
            var vd = hemi_oct_dir(bake.frames, fx, fy)
            var basis = impostor_basis(vd)
            var r = basis[0] * bake.radius
            var u = basis[1] * bake.radius
            var u0 = Float32(fx) * inv
            var u1 = Float32(fx + 1) * inv
            var v0 = Float32(fy) * inv
            var v1 = Float32(fy + 1) * inv
            var mb = MeshBuilder(alpha_mode=1)
            var c = bake.center
            var i0 = mb.add_vertex(c - r + u, vd, u0, v0)
            var i1 = mb.add_vertex(c + r + u, vd, u1, v0)
            var i2 = mb.add_vertex(c + r - u, vd, u1, v1)
            var i3 = mb.add_vertex(c - r - u, vd, u0, v1)
            mb.add_quad(i0, i1, i2, i3)
            out.append(mb^.finish())
    return out^


def impostor_view_cell(frames: Int, obj_pos: Vec3, yaw: Float32, eye: Vec3) -> Int:
    """Card index for an instance at `obj_pos` rotated by `yaw` (radians about +y) seen from `eye`."""
    var d = eye - obj_pos
    var cs = _cos(yaw)
    var sn = _sin(yaw)
    # object space = rotate by -yaw
    var lx = cs * d.x - sn * d.z
    var lz = sn * d.x + cs * d.z
    var cell = hemi_oct_cell(frames, Vec3(lx, d.y, lz).normalized())
    return cell[1] * frames + cell[0]


def impostor_lod_select(radius: Float32, obj_pos: Vec3, eye: Vec3, screen_h: Int, tan_half_fov_y: Float32, switch_px: Float32) -> Bool:
    """True when the object's projected diameter is below `switch_px` pixels (draw the impostor)."""
    var dist = max((obj_pos - eye).length(), Float32(1e-3))
    var px = Float32(2.0) * radius / (dist * tan_half_fov_y) * Float32(screen_h) * Float32(0.5)
    return px < switch_px


def _cos(x: Float32) -> Float32:
    from std.math import cos
    return cos(x)


def _sin(x: Float32) -> Float32:
    from std.math import sin
    return sin(x)


def hemi_oct_blend(frames: Int, view_dir: Vec3) -> Tuple[Int, Int, Int, Float32, Float32, Float32]:
    """The 3 grid cells around `view_dir` and their barycentric weights (sum 1). Cell centres sit on an
    integer lattice in grid space; the containing lattice square is split along its diagonal."""
    var d = Vec3(view_dir.x, max(view_dir.y, Float32(0.0)), view_dir.z)
    var s = abs(d.x) + abs(d.y) + abs(d.z)
    if s < Float32(1e-8):
        d = Vec3(0.0, 1.0, 0.0)
        s = Float32(1.0)
    var u = d.x / s + d.z / s
    var v = d.x / s - d.z / s
    var gu = (u * Float32(0.5) + Float32(0.5)) * Float32(frames) - Float32(0.5)
    var gv = (v * Float32(0.5) + Float32(0.5)) * Float32(frames) - Float32(0.5)
    gu = max(Float32(0.0), min(Float32(frames - 1), gu))
    gv = max(Float32(0.0), min(Float32(frames - 1), gv))
    var x0 = min(frames - 2, Int(gu)) if frames > 1 else 0
    var y0 = min(frames - 2, Int(gv)) if frames > 1 else 0
    var a = gu - Float32(x0)
    var b = gv - Float32(y0)
    var x1 = min(frames - 1, x0 + 1)
    var y1 = min(frames - 1, y0 + 1)
    if a + b <= Float32(1.0):
        return (y0 * frames + x0, y0 * frames + x1, y1 * frames + x0, Float32(1.0) - a - b, a, b)
    return (y1 * frames + x1, y1 * frames + x0, y0 * frames + x1, a + b - Float32(1.0), Float32(1.0) - a, Float32(1.0) - b)


def impostor_view_blend(frames: Int, obj_pos: Vec3, yaw: Float32, eye: Vec3) -> Tuple[Int, Int, Int, Float32, Float32, Float32]:
    """`hemi_oct_blend` for an instance at `obj_pos` rotated by `yaw` seen from `eye`."""
    var d = eye - obj_pos
    var cs = _cos(yaw)
    var sn = _sin(yaw)
    return hemi_oct_blend(frames, Vec3(cs * d.x - sn * d.z, d.y, sn * d.x + cs * d.z).normalized())


def impostor_view_cell_dithered(frames: Int, obj_pos: Vec3, yaw: Float32, eye: Vec3, frame_index: Int, instance_key: Int) -> Int:
    """Stochastic 3-cell blend: picks one of the 3 barycentric cells with probability = its weight, decorrelated
    per instance and per frame (golden-ratio sequence). Temporal accumulation (TSR) turns it into the
    weighted blend, so impostors no longer snap when the view crosses a cell boundary."""
    var bl = impostor_view_blend(frames, obj_pos, yaw, eye)
    var h = UInt32(instance_key) * UInt32(2654435761)
    h = (h ^ (h >> 15)) * UInt32(0x2C1B3C6D)
    var base = Float32(Int(h >> 8)) / Float32(16777216.0)
    var r = base + Float32(frame_index % 4096) * Float32(0.61803398875)
    r = r - Float32(Int(r))
    if r < bl[3]:
        return bl[0]
    if r < bl[3] + bl[4]:
        return bl[1]
    return bl[2]
