"""Botanical Branching Trees, `512x512` Alpha-Cutout Sprig Atlases, Bark & Understory (`foliage.mojo`).

Implements Flint Engine v3 botanical assets in 100% pure Mojo:
  1. High-craft `BC7` (`0xAARRGGBB` alpha-cutout) + `BC5` (`Vec3` tangent normal) procedural atlases:
     - `bake_oak_leaf_sprig_512()`: forked woody twig bearing 36 overlapping lobed oak leaves (`alpha = 255`
       inside leaf SDF lobes/twig, `alpha = 0` in sky gaps), midrib + lateral veins & waxy `BC5` normals.
     - `bake_conifer_needle_sprig_512()`: resinous brown shoot + 6 sub-twigs packed with ~220 individual
       fine pine/spruce needles (`alpha = 255` on needles, `alpha = 0` between needles) + cylindrical normals.
     - `bake_palm_frond_512()`: arched golden-green rachis with 48 individual serrated V-folded leaflets.
     - `bake_fissured_bark_256()`: deeply furrowed longitudinal oak/pine bark ridges + mossy root crevices.
  2. Multi-seed 3D botanical trees (`build_botanical_oak`, `build_botanical_pine`, `build_botanical_palm`,
     `build_urban_ginkgo_tree`) arranging 60–84 curved 3D cross-foil sprig cards (`alpha_mode = 1`,
     `double_sided = True`) over gnarled root-flared limbs (`double_sided = False`) with `0.70` lobe normal
     transfer, radial crown `ao` (`95..255`), warm crown-to-interior vertex `colors`, and distinct `seed` asymmetry.
  3. Park & urban understory builders:
     - `build_grass_wildflower_tuft(seed)`
     - `build_manicured_hedge_and_planter(length, width, height)`
     - `build_flowering_bush(seed, radius)`
"""
from std.math import cos, floor, isfinite, max, min, sin, sqrt
from flint.core.math import Vec3
from flint.assets.model import NativeMesh
from flint.assets.mesh_builder import MeshBuilder

comptime _PI_FOL: Float32 = 3.14159265
comptime _TAU_FOL: Float32 = 6.28318530
comptime _GOLDEN_ANGLE: Float32 = 2.39996323


@always_inline
def _rand01_fol(seed: Int, idx: Int) -> Float32:
    var h = UInt32(((seed * 747796405) + (idx * 1891336453) + 277803737) & 0x7FFFFFFF)
    h = ((h >> ((h >> 28) + UInt32(4))) ^ h) * UInt32(277803737)
    h = (h >> 22) ^ h
    return Float32(h & UInt32(0x00FFFFFF)) / Float32(16777216.0)


@always_inline
def _hash2_fol(ix: Int, iy: Int, salt: Int) -> Float32:
    var h = UInt32(((ix * 374761393) + (iy * 668265263) + (salt * 1442695041)) & 0x7FFFFFFF)
    h = (h ^ (h >> 13)) * UInt32(1274126177)
    h = h ^ (h >> 16)
    return Float32(h & UInt32(0x00FFFFFF)) / Float32(16777216.0)


@always_inline
def _vnoise_tile_fol(x: Float32, y: Float32, period: Int, salt: Int) -> Float32:
    var ix = Int(floor(x))
    var iy = Int(floor(y))
    var fx = x - Float32(ix)
    var fy = y - Float32(iy)
    var ux = fx * fx * (Float32(3.0) - Float32(2.0) * fx)
    var uy = fy * fy * (Float32(3.0) - Float32(2.0) * fy)
    var x0 = ((ix % period) + period) % period
    var y0 = ((iy % period) + period) % period
    var x1 = (x0 + 1) % period
    var y1 = (y0 + 1) % period
    var c00 = _hash2_fol(x0, y0, salt)
    var c10 = _hash2_fol(x1, y0, salt)
    var c01 = _hash2_fol(x0, y1, salt)
    var c11 = _hash2_fol(x1, y1, salt)
    var nx0 = c00 + (c10 - c00) * ux
    var nx1 = c01 + (c11 - c01) * ux
    return nx0 + (nx1 - nx0) * uy


@always_inline
def _pack_argb_fol(a: Int, r: Int, g: Int, b: Int) -> UInt32:
    return (
        (UInt32(max(0, min(255, a))) << 24)
        | (UInt32(max(0, min(255, r))) << 16)
        | (UInt32(max(0, min(255, g))) << 8)
        | UInt32(max(0, min(255, b)))
    )


@always_inline
def _clamp_ao_u8(val_01: Float32, lo: Int = 95, hi: Int = 255) -> UInt8:
    return UInt8(max(lo, min(hi, Int(val_01 * Float32(255.0) + Float32(0.5)))))


# ============================================================================
# 1. `512x512` Botanical Alpha-Cutout Sprig Atlases & `256x256` Fissured Bark
# ============================================================================

def _splat_woody_twig_segment(
    mut albedo: List[UInt32],
    mut normals: List[Vec3],
    dim: Int,
    ax: Float32,
    ay: Float32,
    bx: Float32,
    by: Float32,
    r0: Float32,
    r1: Float32,
    col_r: Int,
    col_g: Int,
    col_b: Int,
):
    """Rasterizes a tapered woody twig/rachis segment into its tight pixel bounding box with cylindrical `BC5` normals."""
    var max_r = max(r0, r1) + Float32(1.5)
    var x0 = max(1, Int(min(ax, bx) - max_r))
    var x1 = min(dim - 2, Int(max(ax, bx) + max_r) + 1)
    var y0 = max(0, Int(min(ay, by) - max_r))
    var y1 = min(dim - 2, Int(max(ay, by) + max_r) + 1)

    var abx = bx - ax
    var aby = by - ay
    var ab_len2 = max(Float32(1e-4), abx * abx + aby * aby)
    var ab_len = sqrt(ab_len2)
    # Perpendicular unit normal in 2D image space `(nx_dir, ny_dir)`
    var px_n = -aby / ab_len
    var py_n = abx / ab_len

    for py in range(y0, y1 + 1):
        var fy = Float32(py) + Float32(0.5)
        for px in range(x0, x1 + 1):
            var fx = Float32(px) + Float32(0.5)
            var t = max(Float32(0.0), min(Float32(1.0), ((fx - ax) * abx + (fy - ay) * aby) / ab_len2))
            var cx = ax + abx * t
            var cy = ay + aby * t
            var dx = fx - cx
            var dy = fy - cy
            var dist = sqrt(dx * dx + dy * dy)
            var rad = r0 + (r1 - r0) * t
            if dist <= rad:
                var across = max(Float32(-1.0), min(Float32(1.0), (dx * px_n + dy * py_n) / max(Float32(0.5), rad)))
                var nz = sqrt(max(Float32(0.12), Float32(1.0) - across * across * Float32(0.82)))
                var bark_grain = Float32(0.84) + Float32(0.24) * nz
                var idx = py * dim + px
                albedo[idx] = _pack_argb_fol(
                    255,
                    Int(Float32(col_r) * bark_grain),
                    Int(Float32(col_g) * bark_grain),
                    Int(Float32(col_b) * bark_grain),
                )
                normals[idx] = Vec3(px_n * across * Float32(0.75), -py_n * across * Float32(0.75), nz).normalized()


def _splat_lobed_oak_leaf(
    mut albedo: List[UInt32],
    mut normals: List[Vec3],
    dim: Int,
    petiole_x: Float32,
    petiole_y: Float32,
    dir_x: Float32,
    dir_y: Float32,
    leaf_len: Float32,
    leaf_wid: Float32,
    hue_var: Float32,
    sun_var: Float32,
):
    """Rasterizes a single scalloped/lobed `Quercus` oak leaf with midrib, 5 lateral vein pairs,
    contact drop-shadow on underlying leaves, crisp `alpha = 255` silhouette, and waxy `BC5` normals."""
    var side_x = -dir_y
    var side_y = dir_x
    var tip_x = petiole_x + dir_x * leaf_len
    var tip_y = petiole_y + dir_y * leaf_len

    var pad = leaf_wid * Float32(0.72) + Float32(3.0)
    var x0 = max(2, Int(min(petiole_x, tip_x) - pad))
    var x1 = min(dim - 3, Int(max(petiole_x, tip_x) + pad) + 1)
    var y0 = max(2, Int(min(petiole_y, tip_y) - pad))
    var y1 = min(dim - 3, Int(max(petiole_y, tip_y) + pad) + 1)

    var inv_len = Float32(1.0) / max(Float32(4.0), leaf_len)
    var inv_hw = Float32(1.0) / max(Float32(2.0), leaf_wid * Float32(0.5))

    for py in range(y0, y1 + 1):
        var fy = Float32(py) + Float32(0.5) - petiole_y
        for px in range(x0, x1 + 1):
            var fx = Float32(px) + Float32(0.5) - petiole_x
            var lv = (fx * dir_x + fy * dir_y) * inv_len
            if lv < Float32(-0.04) or lv > Float32(1.03):
                continue
            var lu = (fx * side_x + fy * side_y) * inv_hw

            var idx = py * dim + px

            # Tiny woody petiole stem (`lv in [-0.04 .. 0.06]`)
            if lv < Float32(0.05) and abs(lu) < Float32(0.07):
                albedo[idx] = _pack_argb_fol(255, 96, 78, 46)
                normals[idx] = Vec3(side_x * lu * Float32(4.0), -side_y * lu * Float32(4.0), Float32(0.92)).normalized()
                continue

            if lv < Float32(0.0) or lv > Float32(1.0):
                continue

            # Obovate oak envelope (wider at `lv ~ 0.62`) + 5 pairs of deep rounded oak sinus lobes!
            var base_env = sin(sqrt(lv) * _PI_FOL) * (Float32(0.56) + Float32(0.42) * lv)
            var lobe_wave = Float32(1.0) + Float32(0.28) * cos(lv * Float32(5.2) * _TAU_FOL - Float32(0.55)) * sin(lv * _PI_FOL)
            var max_u = base_env * lobe_wave
            var abs_u = abs(lu)

            # Soft self-occlusion shadow cast onto any already-drawn lower leaf just outside margin
            if abs_u > max_u and abs_u < max_u + Float32(0.14):
                var cur_px = albedo[idx]
                if ((cur_px >> 24) & UInt32(255)) > UInt32(128):
                    var sh_f = Float32(0.80) + Float32(0.20) * ((abs_u - max_u) / Float32(0.14))
                    var cr = Int(Float32((cur_px >> 16) & UInt32(255)) * sh_f)
                    var cg = Int(Float32((cur_px >> 8) & UInt32(255)) * sh_f)
                    var cb = Int(Float32(cur_px & UInt32(255)) * sh_f)
                    albedo[idx] = _pack_argb_fol(255, cr, cg, cb)
                continue

            if abs_u > max_u:
                continue

            # Central midrib + herring-bone lateral veins aligned with the 5 oak lobes
            var midrib_w = Float32(0.048) * (Float32(1.0) - Float32(0.65) * lv)
            var midrib = max(Float32(0.0), Float32(1.0) - abs_u / max(Float32(0.01), midrib_w))
            var vein_phase = (lv - abs_u * Float32(0.26)) * Float32(5.2)
            var vein_dist = abs(vein_phase - floor(vein_phase + Float32(0.5)))
            var lat_vein = max(Float32(0.0), Float32(1.0) - vein_dist * Float32(9.5)) * max(Float32(0.0), Float32(1.0) - abs_u * Float32(0.75))
            var vein_mask = max(midrib, lat_vein * Float32(0.65))

            # Rich chlorophyll gradient: deep olive core -> sunlit emerald blade -> warm golden-chartreuse veins/margin
            var edge_t = abs_u / max(Float32(0.05), max_u)
            var shade = Float32(0.76) + sun_var * Float32(0.34) + lv * Float32(0.14) - edge_t * edge_t * Float32(0.10)
            var r_f = (Float32(54.0) + hue_var * Float32(28.0) + vein_mask * Float32(42.0)) * shade
            var g_f = (Float32(118.0) + hue_var * Float32(26.0) + vein_mask * Float32(46.0)) * shade
            var b_f = (Float32(42.0) + hue_var * Float32(10.0) + vein_mask * Float32(18.0)) * shade

            # Waxy leaf pillow curl + embossed vein normal in tangent space
            var curl_across = (lu / max(Float32(0.08), max_u)) * Float32(0.34) - (Float32(1.0) if lu >= Float32(0.0) else Float32(-1.0)) * midrib * Float32(0.22)
            var curl_along = (lv - Float32(0.45)) * Float32(0.22)
            var tnx = side_x * curl_across + dir_x * curl_along
            var tny = -(side_y * curl_across + dir_y * curl_along)
            var tnz = sqrt(max(Float32(0.25), Float32(1.0) - tnx * tnx - tny * tny))

            albedo[idx] = _pack_argb_fol(255, Int(r_f), Int(g_f), Int(b_f))
            normals[idx] = Vec3(tnx, tny, tnz).normalized()


def bake_oak_leaf_sprig_512() -> Tuple[List[UInt32], List[Vec3]]:
    """Bakes a `512x512` alpha-masked scalloped oak branchlet (`alpha = 255` on twig + 36 lobed oak leaves,
    `alpha = 0` with solid olive edge-bleed in sky gaps) + coupled `BC5` waxy leaf-curl/vein normals."""
    comptime DIM: Int = 512
    # Edge-bled zero-alpha olive fill prevents dark bilinear fringes on mipmapped cutout borders!
    var albedo = List[UInt32](length=DIM * DIM, fill=UInt32(0x00346828))
    var normals = List[Vec3](length=DIM * DIM, fill=Vec3(0.0, 0.0, 1.0))

    # Central forked woody oak twig (`v = 0` at `y = 6` -> `y = 420`) + 4 lateral sub-twigs
    _splat_woody_twig_segment(albedo, normals, DIM, 256.0, 4.0, 252.0, 195.0, 6.0, 4.2, 92, 68, 46)
    _splat_woody_twig_segment(albedo, normals, DIM, 252.0, 195.0, 260.0, 395.0, 4.2, 2.0, 98, 74, 50)
    _splat_woody_twig_segment(albedo, normals, DIM, 254.0, 96.0, 124.0, 228.0, 3.6, 1.6, 94, 70, 48)
    _splat_woody_twig_segment(albedo, normals, DIM, 253.0, 128.0, 392.0, 256.0, 3.6, 1.6, 94, 70, 48)
    _splat_woody_twig_segment(albedo, normals, DIM, 255.0, 236.0, 138.0, 358.0, 3.0, 1.5, 98, 74, 50)
    _splat_woody_twig_segment(albedo, normals, DIM, 257.0, 262.0, 378.0, 376.0, 3.0, 1.5, 98, 74, 50)

    # Emit 36 lobed oak leaves in two passes (`0..17` shaded lower layer, `18..35` sunlit upper rosette clusters)
    for li in range(36):
        var upper = li >= 18
        var branch_id = li % 6
        var t_b = _rand01_fol(701, li * 3 + 1) * Float32(0.78) + Float32(0.16)
        var ax: Float32
        var ay: Float32
        var base_ang: Float32

        if branch_id == 0:
            ax = Float32(256.0) - t_b * Float32(4.0)
            ay = Float32(48.0) + t_b * Float32(165.0)
            base_ang = Float32(1.5707963) + (Float32(-0.95) if ((li & 1) == 0) else Float32(0.95))
        elif branch_id == 1:
            ax = Float32(252.0) + t_b * Float32(8.0)
            ay = Float32(195.0) + t_b * Float32(205.0)
            base_ang = Float32(1.5707963) + (_rand01_fol(701, li * 5) - Float32(0.5)) * Float32(1.35)
        elif branch_id == 2:
            ax = Float32(254.0) - t_b * Float32(130.0)
            ay = Float32(96.0) + t_b * Float32(132.0)
            base_ang = Float32(2.38) + (_rand01_fol(701, li * 7) - Float32(0.5)) * Float32(0.95)
        elif branch_id == 3:
            ax = Float32(253.0) + t_b * Float32(139.0)
            ay = Float32(128.0) + t_b * Float32(128.0)
            base_ang = Float32(0.76) + (_rand01_fol(701, li * 11) - Float32(0.5)) * Float32(0.95)
        elif branch_id == 4:
            ax = Float32(255.0) - t_b * Float32(117.0)
            ay = Float32(236.0) + t_b * Float32(122.0)
            base_ang = Float32(2.22) + (_rand01_fol(701, li * 13) - Float32(0.5)) * Float32(0.88)
        else:
            ax = Float32(257.0) + t_b * Float32(121.0)
            ay = Float32(262.0) + t_b * Float32(114.0)
            base_ang = Float32(0.92) + (_rand01_fol(701, li * 17) - Float32(0.5)) * Float32(0.88)

        var l_len = Float32(78.0) + _rand01_fol(701, li * 19) * Float32(34.0)
        var l_wid = l_len * (Float32(0.56) + _rand01_fol(701, li * 23) * Float32(0.10))
        var hue_v = _rand01_fol(701, li * 29)
        var sun_v = (Float32(0.55) + _rand01_fol(701, li * 31) * Float32(0.45)) if upper else (_rand01_fol(701, li * 31) * Float32(0.38))

        _splat_lobed_oak_leaf(
            albedo,
            normals,
            DIM,
            ax,
            ay,
            cos(base_ang),
            sin(base_ang),
            l_len,
            l_wid,
            hue_v,
            sun_v,
        )

    return (albedo^, normals^)


def _splat_conifer_needle(
    mut albedo: List[UInt32],
    mut normals: List[Vec3],
    dim: Int,
    ax: Float32,
    ay: Float32,
    dir_x: Float32,
    dir_y: Float32,
    n_len: Float32,
    half_w: Float32,
    r_col: Int,
    g_col: Int,
    b_col: Int,
):
    """Rasterizes a single crisp tapered conifer needle (`alpha = 255`) with cylindrical `BC5` cross-section normals."""
    var bx = ax + dir_x * n_len
    var by = ay + dir_y * n_len
    var pad = half_w + Float32(1.5)
    var x0 = max(2, Int(min(ax, bx) - pad))
    var x1 = min(dim - 3, Int(max(ax, bx) + pad) + 1)
    var y0 = max(2, Int(min(ay, by) - pad))
    var y1 = min(dim - 3, Int(max(ay, by) + pad) + 1)

    var side_x = -dir_y
    var side_y = dir_x
    var inv_l = Float32(1.0) / max(Float32(4.0), n_len)

    for py in range(y0, y1 + 1):
        var fy = Float32(py) + Float32(0.5) - ay
        for px in range(x0, x1 + 1):
            var fx = Float32(px) + Float32(0.5) - ax
            var tv = (fx * dir_x + fy * dir_y) * inv_l
            if tv < Float32(0.0) or tv > Float32(1.0):
                continue
            var tu = fx * side_x + fy * side_y
            var cur_hw = half_w * (Float32(1.0) - tv * tv * Float32(0.72))
            if abs(tu) <= cur_hw:
                var across = tu / max(Float32(0.35), cur_hw)
                var nz = sqrt(max(Float32(0.18), Float32(1.0) - across * across * Float32(0.78)))
                var tip_lift = Float32(0.82) + tv * Float32(0.28) + nz * Float32(0.14)
                var idx = py * dim + px
                albedo[idx] = _pack_argb_fol(
                    255,
                    Int(Float32(r_col) * tip_lift),
                    Int(Float32(g_col) * tip_lift),
                    Int(Float32(b_col) * tip_lift),
                )
                normals[idx] = Vec3(side_x * across * Float32(0.70), -side_y * across * Float32(0.70), nz).normalized()


def bake_conifer_needle_sprig_512() -> Tuple[List[UInt32], List[Vec3]]:
    """Bakes a `512x512` alpha-masked pine/spruce bough shoot (`~224` individual fine needles off a
    forked resinous brown twig, `alpha = 255` on needles/shoot, `alpha = 0` between needles)."""
    comptime DIM: Int = 512
    var albedo = List[UInt32](length=DIM * DIM, fill=UInt32(0x00224E2A))
    var normals = List[Vec3](length=DIM * DIM, fill=Vec3(0.0, 0.0, 1.0))

    # Central resinous shoot + 6 whorled lateral branchlets (`[x0, y0, x1, y1]`)
    var shoots = List[Array[Float32, 4]]()
    var s_main = Array[Float32, 4](fill=0.0)
    s_main[0] = 256.0; s_main[1] = 6.0; s_main[2] = 256.0; s_main[3] = 438.0
    shoots.append(s_main^)

    var shoot_defs: List[List[Float32]] = [
        [256.0, 74.0, 108.0, 224.0],
        [256.0, 82.0, 404.0, 232.0],
        [256.0, 178.0, 118.0, 336.0],
        [256.0, 186.0, 394.0, 344.0],
        [256.0, 286.0, 158.0, 424.0],
        [256.0, 292.0, 354.0, 428.0],
    ]
    for si in range(len(shoot_defs)):
        var a4 = Array[Float32, 4](fill=0.0)
        a4[0] = shoot_defs[si][0]; a4[1] = shoot_defs[si][1]
        a4[2] = shoot_defs[si][2]; a4[3] = shoot_defs[si][3]
        shoots.append(a4^)

    # Pass 1: shaded lower-rank needles; Pass 2: woody resinous twigs; Pass 3: sunlit upper-rank needles (`7 * 32 = 224` needles!)
    for layer in range(2):
        if layer == 1:
            _splat_woody_twig_segment(albedo, normals, DIM, 256.0, 4.0, 256.0, 438.0, 5.2, 2.0, 102, 68, 42)
            for si in range(1, len(shoots)):
                _splat_woody_twig_segment(
                    albedo, normals, DIM, shoots[si][0], shoots[si][1], shoots[si][2], shoots[si][3], 3.4, 1.5, 98, 66, 40
                )

        for si in range(len(shoots)):
            var sx0 = shoots[si][0]; var sy0 = shoots[si][1]
            var sx1 = shoots[si][2]; var sy1 = shoots[si][3]
            var dx = sx1 - sx0; var dy = sy1 - sy0
            var slen = max(Float32(1.0), sqrt(dx * dx + dy * dy))
            var tx = dx / slen; var ty = dy / slen

            var nx_perp = -ty
            var ny_perp = tx
            var n_per_layer = 16
            for ni in range(n_per_layer):
                var t_pos = (Float32(ni) + Float32(0.5) + (_rand01_fol(911, layer * 256 + si * 32 + ni) - Float32(0.5)) * Float32(0.45)) / Float32(n_per_layer)
                t_pos = max(Float32(0.06), min(Float32(0.98), t_pos))
                var px = sx0 + dx * t_pos
                var py = sy0 + dy * t_pos

                var side_sgn = Float32(-1.0) if ((ni & 1) == 0) else Float32(1.0)
                var splay = Float32(0.48) + _rand01_fol(917, layer * 256 + si * 32 + ni) * Float32(0.42)
                var cs = cos(splay)
                var ss = sin(splay) * side_sgn
                var ndx = tx * cs + nx_perp * ss
                var ndy = ty * cs + ny_perp * ss

                var n_len = (Float32(52.0) + _rand01_fol(923, layer * 256 + si * 32 + ni) * Float32(28.0)) * (Float32(1.05) - t_pos * Float32(0.28))
                var hw = Float32(1.55) if layer == 1 else Float32(1.75)
                var shade_t = Float32(0.62) + Float32(layer) * Float32(0.32) + _rand01_fol(929, si * 32 + ni) * Float32(0.16)
                var rc = Int(Float32(42.0) * shade_t)
                var gc = Int(Float32(106.0) * shade_t)
                var bc = Int(Float32(52.0) * shade_t)

                _splat_conifer_needle(albedo, normals, DIM, px, py, ndx, ndy, n_len, hw, rc, gc, bc)

    return (albedo^, normals^)


def bake_palm_frond_512() -> Tuple[List[UInt32], List[Vec3]]:
    """Bakes a `512x512` alpha-masked tropical palm frond (`48` individual lanceolate V-folded serrated
    leaflets off a golden-green central rachis, `alpha = 255` on leaflets, `alpha = 0` between leaflets)."""
    comptime DIM: Int = 512
    var albedo = List[UInt32](length=DIM * DIM, fill=UInt32(0x002E6E2E))
    var normals = List[Vec3](length=DIM * DIM, fill=Vec3(0.0, 0.0, 1.0))

    # 24 pairs (`48` total) of V-folded lanceolate leaflets sweeping outward from the central rachis
    for pair_i in range(24):
        var t_r = (Float32(pair_i) + Float32(0.6)) / Float32(24.5)
        var ry = Float32(16.0) + t_r * Float32(462.0)
        var rx = Float32(256.0) + sin(t_r * Float32(2.2)) * Float32(6.0)
        var env = sin(max(Float32(0.08), min(Float32(0.96), t_r)) * _PI_FOL)
        var l_len = Float32(68.0) + env * Float32(158.0)
        var max_hw = Float32(6.5) + env * Float32(4.2)

        for side in range(2):
            var sgn = Float32(-1.0) if side == 0 else Float32(1.0)
            var sweep_a = Float32(0.42) + t_r * Float32(0.58) + (_rand01_fol(503, pair_i * 2 + side) - Float32(0.5)) * Float32(0.08)
            var dir_x = sgn * cos(sweep_a)
            var dir_y = sin(sweep_a)
            var side_x = -dir_y
            var side_y = dir_x

            var bx = rx + dir_x * l_len
            var by = ry + dir_y * l_len
            var pad = max_hw + Float32(2.0)
            var x0 = max(2, Int(min(rx, bx) - pad))
            var x1 = min(DIM - 3, Int(max(rx, bx) + pad) + 1)
            var y0 = max(2, Int(min(ry, by) - pad))
            var y1 = min(DIM - 3, Int(max(ry, by) + pad) + 1)
            var inv_l = Float32(1.0) / l_len
            var hue = _rand01_fol(509, pair_i * 2 + side)

            for py in range(y0, y1 + 1):
                var fy = Float32(py) + Float32(0.5) - ry
                for px in range(x0, x1 + 1):
                    var fx = Float32(px) + Float32(0.5) - rx
                    var lv = (fx * dir_x + fy * dir_y) * inv_l
                    if lv < Float32(0.0) or lv > Float32(1.0):
                        continue
                    var lu = fx * side_x + fy * side_y
                    # Lanceolate taper + subtle serrated leaflet margin
                    var serrate = Float32(1.0) + Float32(0.07) * sin(lv * Float32(42.0))
                    var hw = max_hw * sin(sqrt(lv) * _PI_FOL) * serrate
                    if abs(lu) <= hw:
                        var across = lu / max(Float32(0.4), hw)
                        # V-folded leaflet crease along midrib
                        var fold = (Float32(0.52) if across >= Float32(0.0) else Float32(-0.52))
                        var midrib = max(Float32(0.0), Float32(1.0) - abs(across) * Float32(5.5))
                        var sun_tip = Float32(0.82) + lv * Float32(0.28) + hue * Float32(0.12)
                        var cr = Int((Float32(52.0) + lv * Float32(34.0) + midrib * Float32(36.0)) * sun_tip)
                        var cg = Int((Float32(138.0) + midrib * Float32(32.0)) * sun_tip)
                        var cb = Int((Float32(46.0) + midrib * Float32(14.0)) * sun_tip)
                        var tnx = side_x * fold
                        var tny = -side_y * fold
                        var tnz = sqrt(max(Float32(0.25), Float32(1.0) - tnx * tnx - tny * tny))
                        var idx = py * DIM + px
                        albedo[idx] = _pack_argb_fol(255, cr, cg, cb)
                        normals[idx] = Vec3(tnx, tny, tnz).normalized()

    # Central golden-green palm rachis spine drawn cleanly down the center
    _splat_woody_twig_segment(albedo, normals, DIM, 256.0, 2.0, 260.0, 250.0, 6.2, 3.8, 118, 128, 54)
    _splat_woody_twig_segment(albedo, normals, DIM, 260.0, 250.0, 256.0, 496.0, 3.8, 1.4, 112, 132, 52)
    return (albedo^, normals^)


def bake_fissured_bark_256() -> Tuple[List[UInt32], List[Vec3]]:
    """Bakes a tileable `256x256` deeply furrowed longitudinal oak/pine bark texture (`BC7`) +
    high-relief ridge `BC5` normal map with mossy olive-lichen root-base crevices."""
    comptime DIM: Int = 256
    var albedo = List[UInt32](length=DIM * DIM, fill=UInt32(0xFF5A422E))
    var normals = List[Vec3](length=DIM * DIM, fill=Vec3(0.0, 0.0, 1.0))
    var h_map = List[Float32](length=DIM * DIM, fill=0.5)

    for y in range(DIM):
        var fy = Float32(y) / Float32(DIM)
        for x in range(DIM):
            var fx = Float32(x) / Float32(DIM)
            var warp = _vnoise_tile_fol(fx * Float32(8.0), fy * Float32(6.0), 8, 11) * Float32(0.22)
            var u_furrow = (fx + warp) * Float32(14.0)
            # Deep V-shaped longitudinal bark fissures between flat-topped weathered plates
            var ridge_raw = abs((u_furrow - floor(u_furrow)) * Float32(2.0) - Float32(1.0))
            var plate = min(Float32(1.0), ridge_raw * Float32(1.55))
            var micro = _vnoise_tile_fol(fx * Float32(32.0), fy * Float32(24.0), 32, 29)
            var h_crack = _vnoise_tile_fol(fx * Float32(16.0), fy * Float32(48.0), 16, 47)
            var height_v = plate * Float32(0.72) + micro * Float32(0.20) + h_crack * Float32(0.08)
            h_map[y * DIM + x] = height_v

            # Root-collar moss/lichen concentrated in lower crevices (`y < 64` or deep fissures)
            var root_zone = max(Float32(0.0), Float32(1.0) - Float32(y) / Float32(68.0))
            var moss = max(Float32(0.0), min(Float32(1.0), (Float32(0.48) - plate) * Float32(2.2) + root_zone * Float32(0.55) * (micro - Float32(0.35))))

            var br = Float32(40.0) + height_v * Float32(86.0)
            var bg = Float32(28.0) + height_v * Float32(64.0)
            var bb = Float32(20.0) + height_v * Float32(46.0)

            var mr = br * (Float32(1.0) - moss) + Float32(72.0) * moss
            var mg = bg * (Float32(1.0) - moss) + Float32(106.0) * moss
            var mb_c = bb * (Float32(1.0) - moss) + Float32(52.0) * moss
            albedo[y * DIM + x] = _pack_argb_fol(255, Int(mr), Int(mg), Int(mb_c))

    for y in range(DIM):
        var ym = (y + DIM - 1) % DIM
        var yp = (y + 1) % DIM
        for x in range(DIM):
            var xm = (x + DIM - 1) % DIM
            var xp = (x + 1) % DIM
            var dx = (h_map[y * DIM + xm] - h_map[y * DIM + xp]) * Float32(2.6)
            var dy = (h_map[ym * DIM + x] - h_map[yp * DIM + x]) * Float32(1.8)
            normals[y * DIM + x] = Vec3(dx, dy, Float32(1.0)).normalized()

    return (albedo^, normals^)


# ============================================================================
# 2. Multi-Seed 3D Botanical Tree Builders (`56–84` Curved Cross-Foil Sprigs)
# ============================================================================

def _add_fluted_tapered_limb(
    mut mb: MeshBuilder,
    pts: List[Vec3],
    radii: List[Float32],
    sides: Int,
    root_flute_amp: Float32 = 0.0,
    flute_count: Int = 5,
    ao_start: Float32 = 0.46,
    ao_end: Float32 = 0.86,
    cap_tip: Bool = True,
    bark_col_base: UInt32 = 0xFF6B5038,
    bark_col_upper: UInt32 = 0xFF8A6E54,
):
    """Sweeps a multi-station tapered woody trunk/limb with root buttress flutes & tileable bark UVs."""
    var n_st = len(pts)
    if n_st < 2 or sides < 3:
        return

    var tangs = List[Vec3](length=n_st, fill=Vec3(0.0, 1.0, 0.0))
    for i in range(n_st):
        var d = pts[min(n_st - 1, i + 1)] - pts[max(0, i - 1)]
        if d.length_squared() < Float32(1e-8):
            d = Vec3(0.0, 1.0, 0.0)
        tangs[i] = d.normalized()

    var ref_up = Vec3(0.0, 0.0, 1.0) if abs(tangs[0].y) > Float32(0.9) else Vec3(0.0, 1.0, 0.0)
    var r_vec = ref_up.cross(tangs[0]).normalized()
    var u_vec = tangs[0].cross(r_vec).normalized()

    var r0_c = Float32((bark_col_base >> 16) & UInt32(255))
    var g0_c = Float32((bark_col_base >> 8) & UInt32(255))
    var b0_c = Float32(bark_col_base & UInt32(255))
    var r1_c = Float32((bark_col_upper >> 16) & UInt32(255))
    var g1_c = Float32((bark_col_upper >> 8) & UInt32(255))
    var b1_c = Float32(bark_col_upper & UInt32(255))

    var base_v = mb.mesh.vertex_count()
    var stride = sides + 1

    for i in range(n_st):
        if i > 0:
            var v1 = pts[i] - pts[i - 1]
            var c1 = v1.length_squared()
            if c1 > Float32(1e-10):
                var rl = r_vec - v1 * ((Float32(2.0) / c1) * v1.dot(r_vec))
                var tl = tangs[i - 1] - v1 * ((Float32(2.0) / c1) * v1.dot(tangs[i - 1]))
                var v2 = tangs[i] - tl
                var c2 = v2.length_squared()
                r_vec = (rl - v2 * ((Float32(2.0) / c2) * v2.dot(rl))) if c2 > Float32(1e-10) else rl
                u_vec = tangs[i].cross(r_vec).normalized()
                r_vec = u_vec.cross(tangs[i]).normalized()

        var tv = Float32(i) / Float32(n_st - 1)
        var flute_env = max(Float32(0.0), Float32(1.0) - tv * Float32(3.0))
        var flute_a = root_flute_amp * flute_env * flute_env
        var base_ao = ao_start + (ao_end - ao_start) * tv
        var v_col = _pack_argb_fol(
            255,
            Int(r0_c + (r1_c - r0_c) * tv),
            Int(g0_c + (g1_c - g0_c) * tv),
            Int(b0_c + (b1_c - b0_c) * tv),
        )

        for s in range(stride):
            var tu = Float32(s) / Float32(sides)
            var phi = tu * _TAU_FOL
            var cp = cos(phi)
            var sp = sin(phi)
            var flute_w = cos(phi * Float32(flute_count))
            var rad = radii[i] * (Float32(1.0) + flute_a * flute_w)
            var out_dir = (r_vec * cp + u_vec * sp).normalized()
            var ao_v = base_ao - max(Float32(0.0), -flute_w) * flute_a * Float32(0.35)
            _ = mb.add_vertex(pts[i] + out_dir * rad, out_dir, tu * Float32(2.0), tv * Float32(3.0), color=v_col, ao=_clamp_ao_u8(ao_v, 85, 240))

    for i in range(n_st - 1):
        for s in range(sides):
            var i0 = base_v + i * stride + s
            mb.add_quad(i0, i0 + 1, i0 + stride + 1, i0 + stride)

    if cap_tip and radii[n_st - 1] > Float32(0.004):
        var last_ring = base_v + (n_st - 1) * stride
        var tip_v = mb.add_vertex(
            pts[n_st - 1] + tangs[n_st - 1] * (radii[n_st - 1] * Float32(0.6)),
            tangs[n_st - 1],
            0.5,
            3.0,
            color=bark_col_upper,
            ao=_clamp_ao_u8(ao_end, 100, 245),
        )
        for s in range(sides):
            mb.add_tri(tip_v, last_ring + s, last_ring + s + 1)


def _add_curved_crossfoil_sprig_pair(
    mut mb: MeshBuilder,
    anchor: Vec3,
    shoot_dir: Vec3,
    sprig_len: Float32,
    sprig_wid: Float32,
    droop: Float32,
    lobe_center: Vec3,
    crown_center: Vec3,
    crown_radius: Float32,
    roll_angle: Float32,
    normal_transfer: Float32,
    inner_col: UInt32,
    outer_col: UInt32,
):
    """Emits `2` intersecting 3D arched/cupped sprig cards (`2` curved cards x `4` tris = `8` tris/pair;
    `32–42` pairs = `64–84` curved 3D sprig cards/tree!) mapped `v: 0 -> 1` from woody twig base to leaf tips,
    with `0.70` spherical-lobe normal transfer, radial crown `ao` (`95..255`), and warm crown-to-interior `colors`!"""
    var fwd = shoot_dir.normalized()
    if fwd.length_squared() < Float32(0.05):
        fwd = Vec3(0.0, 1.0, 0.0)
    var ref_v = Vec3(1.0, 0.0, 0.0) if abs(fwd.y) > Float32(0.9) else Vec3(0.0, 1.0, 0.0)
    var b_right = ref_v.cross(fwd).normalized()
    var b_up = fwd.cross(b_right).normalized()

    var ir = Float32((inner_col >> 16) & UInt32(255))
    var ig = Float32((inner_col >> 8) & UInt32(255))
    var ib = Float32(inner_col & UInt32(255))
    var or_c = Float32((outer_col >> 16) & UInt32(255))
    var og = Float32((outer_col >> 8) & UInt32(255))
    var ob = Float32(outer_col & UInt32(255))

    var inv_cr = Float32(1.0) / max(Float32(0.25), crown_radius)
    var hw = sprig_wid * Float32(0.5)

    # 2 crossed curved cards rotated `74°` (`1.29 rad`) around the sprig axis for true 3D volume from all view angles!
    for card_i in range(2):
        var ang = roll_angle + Float32(card_i) * Float32(1.2915436)
        var ca = cos(ang)
        var sa = sin(ang)
        var side = (b_right * ca + b_up * sa).normalized()
        var n_card = fwd.cross(side).normalized()
        if n_card.y < Float32(-0.2):
            n_card = Vec3(-n_card.x, -n_card.y, -n_card.z)

        # 2-segment arched ribbon (`v = 0.0, 0.52, 1.0`, `u = 0.0, 1.0` -> `6` verts, `4` tris per curved card)
        var c_base = mb.mesh.vertex_count()
        for st_i in range(3):
            var tv = Float32(0.0) if st_i == 0 else (Float32(0.52) if st_i == 1 else Float32(1.0))
            var arch_lift = sin(tv * _PI_FOL) * sprig_len * Float32(0.11)
            var grav_drop = tv * tv * droop
            var spine_p = anchor + fwd * (tv * sprig_len) + n_card * arch_lift - Vec3(0.0, grav_drop, 0.0)
            var w_scale = Float32(0.72) if st_i == 0 else (Float32(1.05) if st_i == 1 else Float32(0.92))

            for u_i in range(2):
                var tu = Float32(u_i)
                var sgn_u = Float32(-1.0) if u_i == 0 else Float32(1.0)
                var cup = hw * Float32(0.14)
                var vp = spine_p + side * (sgn_u * hw * w_scale) - n_card * cup
                var geom_n = (n_card + side * (sgn_u * Float32(0.28))).normalized()
                var lobe_n = (vp - lobe_center).normalized()
                if lobe_n.length_squared() < Float32(0.05):
                    lobe_n = Vec3(0.0, 1.0, 0.0)
                var trans_n = (geom_n * (Float32(1.0) - normal_transfer) + lobe_n * normal_transfer).normalized()

                var r_crown = min(Float32(1.0), (vp - crown_center).length() * inv_cr)
                var sun_w = max(Float32(0.0), min(Float32(1.0), r_crown * Float32(0.68) + max(Float32(0.0), lobe_n.y) * Float32(0.32)))
                var ao_01 = Float32(0.37) + sun_w * Float32(0.63)
                var vcol = _pack_argb_fol(
                    255,
                    Int(ir + (or_c - ir) * sun_w),
                    Int(ig + (og - ig) * sun_w),
                    Int(ib + (ob - ib) * sun_w),
                )
                _ = mb.add_vertex(vp, trans_n, tu, tv, color=vcol, ao=_clamp_ao_u8(ao_01, 95, 255))

        mb.add_quad(c_base + 0, c_base + 2, c_base + 3, c_base + 1)
        mb.add_quad(c_base + 2, c_base + 4, c_base + 5, c_base + 3)


def build_botanical_oak(seed: Int, height: Float32 = 6.0) -> Tuple[NativeMesh, NativeMesh]:
    """Builds `(trunk_mesh, leaf_canopy_mesh)` with distinct `seed` asymmetry:
    - `trunk_mesh` (`double_sided = False`): 6-buttress root-flared trunk + `4` asymmetric gnarled main limbs + `12` sub-limbs.
    - `leaf_canopy_mesh` (`alpha_mode = 1`, `double_sided = True`): `72` curved 3D cross-foil sprig cards (`36` pairs = `288`->`576` tris)
      with `0.70` spherical-lobe normal transfer, radial crown `ao` (`95..255`), and warm sunlit-to-olive vertex `colors`!
    """
    var h = max(Float32(2.4), min(Float32(200.0), height)) if isfinite(height) else Float32(6.0)
    var mb_trunk = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.82, metallic=0.0, double_sided=False)
    var mb_leaves = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.56, metallic=0.0, alpha_mode=1, double_sided=True)

    # Seed-driven silhouette asymmetry (lean direction, fork height, and crown bias)
    var lean_az = _rand01_fol(seed, 1) * _TAU_FOL
    var lean_amp = h * (Float32(0.05) + _rand01_fol(seed, 2) * Float32(0.09))
    var fork_y = h * (Float32(0.31) + _rand01_fol(seed, 3) * Float32(0.10))
    var fork_pt = Vec3(cos(lean_az) * lean_amp, fork_y, sin(lean_az) * lean_amp)
    var r0 = h * Float32(0.060)

    var tr_pts: List[Vec3] = [
        Vec3(0.0, 0.0, 0.0),
        Vec3(fork_pt.x * Float32(0.28), fork_y * Float32(0.34), fork_pt.z * Float32(0.28)),
        Vec3(fork_pt.x * Float32(0.68), fork_y * Float32(0.70), fork_pt.z * Float32(0.68)),
        fork_pt,
    ]
    var tr_rad: List[Float32] = [r0 * Float32(1.56), r0 * Float32(1.04), r0 * Float32(0.88), r0 * Float32(0.78)]
    _add_fluted_tapered_limb(mb_trunk, tr_pts, tr_rad, sides=8, root_flute_amp=0.28, flute_count=6, cap_tip=False)

    var crown_center = fork_pt + Vec3(0.0, h * Float32(0.26), 0.0)
    var crown_radius = h * Float32(0.46)
    var base_yaw = _rand01_fol(seed, 4) * _TAU_FOL

    # 4 asymmetric primary limbs x 3 secondary sub-limbs = 12 sub-limbs; 3 cross-foil pairs per sub-limb = `36` pairs (`72` curved cards!)
    for limb_i in range(4):
        var az = base_yaw + Float32(limb_i) * Float32(1.5707963) + (_rand01_fol(seed, 10 + limb_i) - Float32(0.5)) * Float32(0.65)
        var asym_reach = h * (Float32(0.20) + _rand01_fol(seed, 20 + limb_i) * Float32(0.13))
        var asym_lift = h * (Float32(0.22) + _rand01_fol(seed, 30 + limb_i) * Float32(0.14))
        var mid_p = fork_pt + Vec3(cos(az) * asym_reach * Float32(0.56), asym_lift * Float32(0.52), sin(az) * asym_reach * Float32(0.56))
        var end_p = fork_pt + Vec3(cos(az) * asym_reach, asym_lift, sin(az) * asym_reach)
        var lobe_c = end_p + Vec3(0.0, h * Float32(0.04), 0.0)

        var l_pts: List[Vec3] = [fork_pt, mid_p, end_p]
        var l_rad: List[Float32] = [r0 * Float32(0.54), r0 * Float32(0.36), r0 * Float32(0.20)]
        _add_fluted_tapered_limb(mb_trunk, l_pts, l_rad, sides=6, ao_start=0.52, ao_end=0.74)

        for sub_j in range(3):
            var sub_az = az + (Float32(sub_j) - Float32(1.0)) * Float32(0.78) + (_rand01_fol(seed, 40 + limb_i * 4 + sub_j) - Float32(0.5)) * Float32(0.32)
            var sub_pitch = Float32(0.24) + _rand01_fol(seed, 60 + limb_i * 4 + sub_j) * Float32(0.55)
            var sub_dir = Vec3(cos(sub_az) * cos(sub_pitch), sin(sub_pitch), sin(sub_az) * cos(sub_pitch)).normalized()
            var sub_anchor = mid_p * Float32(0.30) + end_p * Float32(0.70)
            var sub_tip = sub_anchor + sub_dir * (h * Float32(0.17))

            var s_pts: List[Vec3] = [sub_anchor, sub_tip]
            var s_rad: List[Float32] = [r0 * Float32(0.19), r0 * Float32(0.055)]
            _add_fluted_tapered_limb(mb_trunk, s_pts, s_rad, sides=4, ao_start=0.50, ao_end=0.70)

            for k in range(3):
                var sp_az = sub_az + Float32(k) * _GOLDEN_ANGLE
                var sp_up = Float32(0.12) + Float32(k) * Float32(0.26)
                var sp_dir = (sub_dir * Float32(0.62) + Vec3(cos(sp_az) * Float32(0.65), sp_up, sin(sp_az) * Float32(0.65))).normalized()
                var sp_base = sub_anchor * (Float32(0.35) - Float32(k) * Float32(0.12)) + sub_tip * (Float32(0.65) + Float32(k) * Float32(0.12))
                _add_curved_crossfoil_sprig_pair(
                    mb_leaves,
                    sp_base,
                    sp_dir,
                    h * Float32(0.22),
                    h * Float32(0.20),
                    h * Float32(0.035),
                    lobe_c,
                    crown_center,
                    crown_radius,
                    sp_az,
                    normal_transfer=0.70,
                    inner_col=0xFF8AA87E,
                    outer_col=0xFFFFFBE0,
                )

    var trunk_m = mb_trunk^.finish()
    var leaf_m = mb_leaves^.finish()
    trunk_m.ensure_bounds()
    leaf_m.ensure_bounds()
    return (trunk_m^, leaf_m^)


def build_botanical_pine(seed: Int, height: Float32 = 5.5) -> Tuple[NativeMesh, NativeMesh]:
    """Builds `(trunk_mesh, needle_canopy_mesh)`:
    - `trunk_mesh` (`double_sided = False`): root-flared wind-curved trunk + `7` golden-angle whorled woody boughs.
    - `needle_canopy_mesh` (`alpha_mode = 1`, `double_sided = True`): `76` curved 3D cross-foil conifer sprig cards
      (`38` pairs = `608` tris) with `0.70` dome normal transfer, `95..255` radial AO, and crown-to-core `colors`.
    """
    var h = max(Float32(2.0), min(Float32(200.0), height)) if isfinite(height) else Float32(5.5)
    var mb_trunk = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.80, metallic=0.0, double_sided=False)
    var mb_needles = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.60, metallic=0.0, alpha_mode=1, double_sided=True)

    var sway_az = _rand01_fol(seed, 5) * _TAU_FOL
    var sway_amp = h * (Float32(0.03) + _rand01_fol(seed, 6) * Float32(0.07))
    var sway_x = cos(sway_az) * sway_amp
    var sway_z = sin(sway_az) * sway_amp

    var t_pts: List[Vec3] = [
        Vec3(0.0, 0.0, 0.0),
        Vec3(sway_x * Float32(0.25), h * Float32(0.18), sway_z * Float32(0.25)),
        Vec3(sway_x * Float32(0.60), h * Float32(0.48), sway_z * Float32(0.60)),
        Vec3(sway_x * Float32(0.88), h * Float32(0.76), sway_z * Float32(0.88)),
        Vec3(sway_x, h * Float32(0.96), sway_z),
    ]
    var r_base = h * Float32(0.044)
    var t_rad: List[Float32] = [
        r_base * Float32(1.42),
        r_base * Float32(0.94),
        r_base * Float32(0.64),
        r_base * Float32(0.34),
        r_base * Float32(0.06),
    ]
    _add_fluted_tapered_limb(
        mb_trunk, t_pts, t_rad, sides=8, root_flute_amp=0.22, flute_count=5,
        bark_col_base=0xFF5C3E28, bark_col_upper=0xFF82593C,
    )

    var crown_center = Vec3(sway_x * Float32(0.5), h * Float32(0.52), sway_z * Float32(0.5))
    var max_bough_r = h * (Float32(0.27) + _rand01_fol(seed, 7) * Float32(0.05))

    # 7 golden-angle whorled tiers (`35` boughs + `3` apical tufts = `38` cross-foil pairs = `76` curved 3D cards!)
    for tier in range(7):
        var tier_t = Float32(tier) / Float32(6.0)
        var ty = h * (Float32(0.20) + tier_t * Float32(0.68))
        var tier_origin = Vec3(sway_x * (ty / h), ty, sway_z * (ty / h))
        var bough_len = max_bough_r * (Float32(1.0) - tier_t * Float32(0.70))
        var n_boughs = 6 if tier < 3 else (5 if tier < 5 else 4)
        var base_az = _rand01_fol(seed, 12) * _TAU_FOL + Float32(tier) * _GOLDEN_ANGLE

        for b_i in range(n_boughs):
            var az = base_az + (Float32(b_i) / Float32(n_boughs)) * _TAU_FOL + (_rand01_fol(seed, 80 + tier * 8 + b_i) - Float32(0.5)) * Float32(0.32)
            var pitch = Float32(-0.20) + tier_t * Float32(0.42)
            var dir = Vec3(cos(az) * cos(pitch), sin(pitch), sin(az) * cos(pitch)).normalized()
            var asym = Float32(0.84) + _rand01_fol(seed, 140 + tier * 8 + b_i) * Float32(0.32)
            var cur_len = bough_len * asym

            if tier < 6:
                var b_pts: List[Vec3] = [
                    tier_origin,
                    tier_origin + dir * (cur_len * Float32(0.58)) - Vec3(0.0, cur_len * Float32(0.06), 0.0),
                ]
                var b_rad: List[Float32] = [r_base * Float32(0.26) * (Float32(1.0) - tier_t * Float32(0.5)), r_base * Float32(0.06)]
                _add_fluted_tapered_limb(mb_trunk, b_pts, b_rad, sides=4, ao_start=0.42, ao_end=0.68, cap_tip=False)

            var dome_c = tier_origin - Vec3(0.0, h * Float32(0.08), 0.0)
            _add_curved_crossfoil_sprig_pair(
                mb_needles,
                tier_origin + dir * (cur_len * Float32(0.14)),
                dir,
                cur_len * Float32(0.95),
                cur_len * Float32(0.74),
                cur_len * Float32(0.16) * (Float32(1.0) - tier_t * Float32(0.5)),
                dome_c,
                crown_center,
                max_bough_r,
                az * Float32(0.35),
                normal_transfer=0.70,
                inner_col=0xFF7B9E82,
                outer_col=0xFFF6FBE2,
            )

    var apex = Vec3(sway_x, h * Float32(0.89), sway_z)
    for k in range(3):
        var az = Float32(k) * Float32(2.0943951) + _rand01_fol(seed, 200 + k)
        var adir = Vec3(cos(az) * Float32(0.42), Float32(0.90), sin(az) * Float32(0.42)).normalized()
        _add_curved_crossfoil_sprig_pair(
            mb_needles,
            apex,
            adir,
            h * Float32(0.14),
            h * Float32(0.11),
            Float32(0.01),
            apex - Vec3(0.0, h * Float32(0.05), 0.0),
            crown_center,
            max_bough_r,
            az,
            normal_transfer=0.70,
            inner_col=0xFF9AC29C,
            outer_col=0xFFFFFDE8,
        )

    var trunk_m = mb_trunk^.finish()
    var needle_m = mb_needles^.finish()
    trunk_m.ensure_bounds()
    needle_m.ensure_bounds()
    return (trunk_m^, needle_m^)


def build_botanical_palm(seed: Int, height: Float32 = 5.8) -> Tuple[NativeMesh, NativeMesh]:
    """Builds `(trunk_mesh, frond_canopy_mesh)` (`double_sided = False` trunk, `alpha_mode = 1` + `double_sided = True`
    arching V-folded palm fronds mapped cleanly to `bake_palm_frond_512()`)."""
    var h = max(Float32(2.8), min(Float32(200.0), height)) if isfinite(height) else Float32(5.8)
    var mb_trunk = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.78, metallic=0.0, double_sided=False)
    var mb_fronds = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.50, metallic=0.02, alpha_mode=1, double_sided=True)

    var lean_ang = _rand01_fol(seed, 3) * _TAU_FOL
    var lean_mag = h * (Float32(0.10) + _rand01_fol(seed, 4) * Float32(0.10))
    var lean_dir = Vec3(cos(lean_ang), 0.0, sin(lean_ang))

    var tr_pts = List[Vec3](capacity=8)
    var tr_rad = List[Float32](capacity=8)
    var base_r = h * Float32(0.044)
    for i in range(8):
        var t = Float32(i) / Float32(7.0)
        var s_curve = t * t * (Float32(1.4) - Float32(0.4) * t)
        tr_pts.append(lean_dir * (lean_mag * s_curve) + Vec3(0.0, h * Float32(0.88) * t, 0.0))
        var scar_ring = Float32(1.0) + Float32(0.06) * cos(t * Float32(28.0))
        tr_rad.append(base_r * (Float32(1.30) - t * Float32(0.48)) * scar_ring)
    _add_fluted_tapered_limb(
        mb_trunk, tr_pts, tr_rad, sides=8, root_flute_amp=0.14, flute_count=6,
        bark_col_base=0xFF6E533B, bark_col_upper=0xFF94775A,
    )

    var crown = tr_pts[7]
    for c_i in range(3):
        var ca = Float32(c_i) * Float32(2.0943951) + lean_ang
        mb_trunk.add_ellipsoid(
            crown + Vec3(cos(ca) * base_r * Float32(1.05), -base_r * Float32(0.55), sin(ca) * base_r * Float32(1.05)),
            Vec3(base_r * Float32(0.65), base_r * Float32(0.75), base_r * Float32(0.65)),
            rings=4,
            segments=6,
            color=0xFF563A22,
        )

    # 14 arching V-folded pinnate fronds (`14 * 5 = 70` curved segments = `560` canopy tris!)
    var frond_len = h * (Float32(0.44) + _rand01_fol(seed, 8) * Float32(0.06))
    var az_seed = _rand01_fol(seed, 9) * _TAU_FOL
    for f_i in range(14):
        var upper_tier = f_i >= 7
        var az = az_seed + Float32(f_i) * _GOLDEN_ANGLE + (_rand01_fol(seed, 30 + f_i) - Float32(0.5)) * Float32(0.24)
        var rad_h = Vec3(cos(az), 0.0, sin(az))
        var side = Vec3(-sin(az), 0.0, cos(az))
        var lift0 = Float32(0.56) if upper_tier else Float32(0.22)
        var drop1 = Float32(0.30) if upper_tier else Float32(0.58)

        var segs = 5
        var f_base = mb_fronds.mesh.vertex_count()
        for s in range(segs + 1):
            var ts = Float32(s) / Float32(segs)
            var arch_y = (ts * lift0 - ts * ts * drop1) * frond_len
            var spine_p = crown + rad_h * (ts * frond_len) + Vec3(0.0, arch_y, 0.0)
            var env_w = Float32(0.35) + Float32(0.65) * sin(max(Float32(0.08), min(Float32(0.95), ts)) * _PI_FOL)
            var pinna_w = frond_len * Float32(0.24) * env_w
            var pinna_drop = pinna_w * (Float32(0.28) + ts * Float32(0.22))

            var dome_n = ((spine_p - crown) + Vec3(0.0, frond_len * Float32(0.55), 0.0)).normalized()
            var sun_w = min(Float32(1.0), Float32(0.35) + ts * Float32(0.50) + (Float32(0.15) if upper_tier else Float32(0.0)))
            var ao_v = _clamp_ao_u8(Float32(0.38) + sun_w * Float32(0.62), 95, 255)
            var fcol = _pack_argb_fol(
                255,
                Int(Float32(148.0) + sun_w * Float32(107.0)),
                Int(Float32(178.0) + sun_w * Float32(77.0)),
                Int(Float32(138.0) + sun_w * Float32(86.0)),
            )

            var p_l = spine_p - side * pinna_w - Vec3(0.0, pinna_drop, 0.0)
            var p_c = spine_p + Vec3(0.0, frond_len * Float32(0.016) * env_w, 0.0)
            var p_r = spine_p + side * pinna_w - Vec3(0.0, pinna_drop, 0.0)

            var n_l = (Vec3(0.0, 0.30, 0.0) - side * Float32(0.20) + dome_n * Float32(0.70)).normalized()
            var n_c = (Vec3(0.0, 0.30, 0.0) + dome_n * Float32(0.70)).normalized()
            var n_r = (Vec3(0.0, 0.30, 0.0) + side * Float32(0.20) + dome_n * Float32(0.70)).normalized()

            _ = mb_fronds.add_vertex(p_l, n_l, 0.0, ts, color=fcol, ao=ao_v)
            _ = mb_fronds.add_vertex(p_c, n_c, 0.5, ts, color=fcol, ao=ao_v)
            _ = mb_fronds.add_vertex(p_r, n_r, 1.0, ts, color=fcol, ao=ao_v)

        for s in range(segs):
            var r0_i = f_base + s * 3
            var r1_i = r0_i + 3
            mb_fronds.add_quad(r0_i + 0, r1_i + 0, r1_i + 1, r0_i + 1)
            mb_fronds.add_quad(r0_i + 1, r1_i + 1, r1_i + 2, r0_i + 2)

    var trunk_m = mb_trunk^.finish()
    var frond_m = mb_fronds^.finish()
    trunk_m.ensure_bounds()
    frond_m.ensure_bounds()
    return (trunk_m^, frond_m^)


def build_urban_ginkgo_tree(seed: Int, height: Float32 = 5.2) -> Tuple[NativeMesh, NativeMesh]:
    """Builds `(trunk_mesh, ginkgo_canopy_mesh)` (`64` curved 3D cross-foil golden-chartreuse sprig cards,
    `alpha_mode = 1`, `double_sided = True`, distinct `seed` scaffold limb asymmetry & `0.70` lobe normals)."""
    var h = max(Float32(2.4), min(Float32(200.0), height)) if isfinite(height) else Float32(5.2)
    var mb_trunk = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.78, metallic=0.0, double_sided=False)
    var mb_leaves = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.52, metallic=0.0, alpha_mode=1, double_sided=True)

    var lean_az = _rand01_fol(seed, 15) * _TAU_FOL
    var lean_x = cos(lean_az) * h * Float32(0.04)
    var lean_z = sin(lean_az) * h * Float32(0.04)
    var r0 = h * Float32(0.042)

    var leader_pts: List[Vec3] = [
        Vec3(0.0, 0.0, 0.0),
        Vec3(lean_x * Float32(0.3), h * Float32(0.28), lean_z * Float32(0.3)),
        Vec3(lean_x * Float32(0.7), h * Float32(0.58), lean_z * Float32(0.7)),
        Vec3(lean_x, h * Float32(0.88), lean_z),
    ]
    var leader_rad: List[Float32] = [r0 * Float32(1.34), r0 * Float32(0.92), r0 * Float32(0.56), r0 * Float32(0.14)]
    _add_fluted_tapered_limb(
        mb_trunk, leader_pts, leader_rad, sides=7, root_flute_amp=0.18, flute_count=5,
        bark_col_base=0xFF5E4E42, bark_col_upper=0xFF827062,
    )

    var crown_center = Vec3(lean_x * Float32(0.5), h * Float32(0.62), lean_z * Float32(0.5))
    var crown_radius = h * Float32(0.32)
    var base_az = _rand01_fol(seed, 16) * _TAU_FOL

    # 8 ascending scaffold branches x 4 cross-foil pairs = `32` pairs (`64` curved 3D sprig cards = `512` tris!)
    for b_i in range(8):
        var bt = Float32(b_i) / Float32(7.0)
        var by = h * (Float32(0.28) + bt * Float32(0.48))
        var az = base_az + Float32(b_i) * _GOLDEN_ANGLE + (_rand01_fol(seed, 60 + b_i) - Float32(0.5)) * Float32(0.35)
        var asym = Float32(0.82) + _rand01_fol(seed, 70 + b_i) * Float32(0.36)
        var reach = crown_radius * (Float32(1.06) - bt * Float32(0.48)) * asym
        var b_start = Vec3(lean_x * (by / h), by, lean_z * (by / h))
        var b_end = b_start + Vec3(cos(az) * reach, reach * Float32(0.76), sin(az) * reach)
        var b_mid = (b_start + b_end) * Float32(0.5) + Vec3(0.0, reach * Float32(0.06), 0.0)

        var b_pts: List[Vec3] = [b_start, b_mid, b_end]
        var b_rad: List[Float32] = [r0 * Float32(0.36) * (Float32(1.0) - bt * Float32(0.4)), r0 * Float32(0.20), r0 * Float32(0.06)]
        _add_fluted_tapered_limb(mb_trunk, b_pts, b_rad, sides=5, ao_start=0.48, ao_end=0.74)

        for k in range(4):
            var ka = az + Float32(k) * Float32(1.5707963) + Float32(0.35)
            var sp_dir = Vec3(cos(ka) * Float32(0.75), Float32(0.45) + Float32(k & 1) * Float32(0.25), sin(ka) * Float32(0.75)).normalized()
            var anchor = b_mid * (Float32(0.40) - Float32(k) * Float32(0.08)) + b_end * (Float32(0.60) + Float32(k) * Float32(0.08))
            _add_curved_crossfoil_sprig_pair(
                mb_leaves,
                anchor,
                sp_dir,
                h * Float32(0.16),
                h * Float32(0.15),
                h * Float32(0.025),
                b_end,
                crown_center,
                crown_radius,
                ka,
                normal_transfer=0.70,
                inner_col=0xFFB89E38,
                outer_col=0xFFFFE054,
            )

    var trunk_m = mb_trunk^.finish()
    var leaf_m = mb_leaves^.finish()
    trunk_m.ensure_bounds()
    leaf_m.ensure_bounds()
    return (trunk_m^, leaf_m^)


# ============================================================================
# 3. Park & Urban Understory Builders (`Grass Tufts`, `Hedges`, `Flowering Bushes`)
# ============================================================================

def build_grass_wildflower_tuft(seed: Int) -> NativeMesh:
    """Builds a `~20`-triangle 3D rosette of curved individual grass blades + 2 wildflower blossoms
    with dark root-collar AO (`105`) and sunlit chlorophyll + petal vertex `colors`."""
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.58, metallic=0.0, double_sided=True)
    var n_blades = 6
    var base_az = _rand01_fol(seed, 301) * _TAU_FOL

    # 6 arched 2-segment tapered grass blades (`6 * 3 = 18` tris)
    for i in range(n_blades):
        var az = base_az + (Float32(i) / Float32(n_blades)) * _TAU_FOL + (_rand01_fol(seed, 310 + i) - Float32(0.5)) * Float32(0.45)
        var rad = Vec3(cos(az), 0.0, sin(az))
        var side = Vec3(-sin(az), 0.0, cos(az))
        var bl_h = Float32(0.22) + _rand01_fol(seed, 320 + i) * Float32(0.16)
        var arch = Float32(0.10) + _rand01_fol(seed, 330 + i) * Float32(0.12)
        var bw = Float32(0.024)

        var p0_l = rad * Float32(0.03) - side * bw
        var p0_r = rad * Float32(0.03) + side * bw
        var p1_l = rad * (arch * Float32(0.48)) - side * (bw * Float32(0.68)) + Vec3(0.0, bl_h * Float32(0.58), 0.0)
        var p1_r = rad * (arch * Float32(0.48)) + side * (bw * Float32(0.68)) + Vec3(0.0, bl_h * Float32(0.58), 0.0)
        var p2_tip = rad * arch + Vec3(0.0, bl_h, 0.0)

        var n_up = (Vec3(0.0, 0.85, 0.0) + rad * Float32(0.35)).normalized()
        var c_root = _pack_argb_fol(255, 54, 96, 38)
        var c_mid = _pack_argb_fol(255, 86, 148, 52)
        var c_tip = _pack_argb_fol(255, 132, 194, 72)

        var v0 = mb.add_vertex(p0_l, n_up, 0.0, 0.0, color=c_root, ao=UInt8(105))
        var v1 = mb.add_vertex(p0_r, n_up, 1.0, 0.0, color=c_root, ao=UInt8(105))
        var v2 = mb.add_vertex(p1_l, n_up, 0.15, 0.55, color=c_mid, ao=UInt8(190))
        var v3 = mb.add_vertex(p1_r, n_up, 0.85, 0.55, color=c_mid, ao=UInt8(190))
        var v4 = mb.add_vertex(p2_tip, Vec3(0.0, 1.0, 0.0), 0.5, 1.0, color=c_tip, ao=UInt8(250))

        mb.add_quad(v0, v2, v3, v1)
        mb.add_tri(v2, v4, v3)

    # 2 tiny cupped wildflower heads (`2 * 2 = 4` tris -> `22` tris total!)
    var pal = seed % 3
    var fl_col: UInt32 = UInt32(0xFFFFF3BF) if pal == 0 else (UInt32(0xFFFFD43B) if pal == 1 else UInt32(0xFFE599F7))
    for f_i in range(2):
        var fa = base_az + Float32(f_i) * Float32(2.6) + Float32(0.5)
        var fc = Vec3(cos(fa) * Float32(0.09), Float32(0.26) + Float32(f_i) * Float32(0.05), sin(fa) * Float32(0.09))
        var r_fl = Float32(0.036)
        var f0 = mb.add_vertex(fc + Vec3(-r_fl, 0.008, -r_fl), Vec3(0.0, 1.0, 0.0), 0.0, 0.0, color=fl_col, ao=UInt8(245))
        var f1 = mb.add_vertex(fc + Vec3( r_fl, 0.008, -r_fl), Vec3(0.0, 1.0, 0.0), 1.0, 0.0, color=fl_col, ao=UInt8(245))
        var f2 = mb.add_vertex(fc + Vec3( r_fl, 0.008,  r_fl), Vec3(0.0, 1.0, 0.0), 1.0, 1.0, color=fl_col, ao=UInt8(245))
        var f3 = mb.add_vertex(fc + Vec3(-r_fl, 0.008,  r_fl), Vec3(0.0, 1.0, 0.0), 0.0, 1.0, color=fl_col, ao=UInt8(245))
        mb.add_quad(f0, f3, f2, f1)

    var out = mb^.finish()
    out.ensure_bounds()
    return out^


def build_manicured_hedge_and_planter(
    length: Float32 = 3.2,
    width: Float32 = 0.9,
    height: Float32 = 1.05,
) -> Tuple[NativeMesh, NativeMesh]:
    """Builds `(stone_curb_mesh, foliage_hedge_mesh)` for plazas, avenues, and coaster queues."""
    var safe_len = min(Float32(200.0), max(Float32(0.4), length)) if isfinite(length) else Float32(3.2)
    var safe_wid = min(Float32(50.0), max(Float32(0.5), width)) if isfinite(width) else Float32(0.9)
    var safe_h = min(Float32(50.0), max(Float32(0.5), height)) if isfinite(height) else Float32(1.05)
    var hl = max(Float32(0.4), safe_len * Float32(0.5))
    var hw = max(Float32(0.25), safe_wid * Float32(0.5))
    var h_tot = max(Float32(0.5), safe_h)
    var curb_h = min(Float32(0.28), h_tot * Float32(0.32))

    # 1. Chamfered granite planter curb + dark mulch soil bed
    var mb_curb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.64, metallic=0.05)
    mb_curb.add_beveled_box(Vec3(0.0, curb_h * Float32(0.5), 0.0), Vec3(hl, curb_h * Float32(0.5), hw), bevel=0.03, color=0xFFADB5BD)
    mb_curb.add_beveled_box(Vec3(0.0, curb_h - Float32(0.02), 0.0), Vec3(hl - Float32(0.08), 0.03, hw - Float32(0.08)), bevel=0.01, color=0xFF4A3525)
    var curb_m = mb_curb^.finish()
    for i in range(curb_m.vertex_count()):
        if curb_m.positions[i * 3 + 1] < Float32(0.06):
            curb_m.ao[i] = UInt8(100)
    curb_m.ensure_bounds()

    # 2. Manicured boxwood hedge core + ruffled outer leaf sprig cards (`alpha_mode = 1`, `double_sided = True`)
    var mb_hedge = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.58, metallic=0.0, alpha_mode=1, double_sided=True)
    var h_fol = h_tot - curb_h * Float32(0.75)
    var cy = curb_h * Float32(0.85) + h_fol * Float32(0.5)
    var h_hl = hl - Float32(0.08)
    var h_hw = hw - Float32(0.08)
    mb_hedge.add_beveled_box(Vec3(0.0, cy, 0.0), Vec3(h_hl, h_fol * Float32(0.5), h_hw), bevel=0.08, color=0xFF2B662E)

    # Stud perimeter & crown with ruffled 3D boxwood sprig cards for natural leafy silhouette breakup
    var n_along = min(512, max(3, Int(safe_len * Float32(2.2))))
    for i in range(n_along):
        var tx = -h_hl * Float32(0.82) + (Float32(i) / Float32(max(1, n_along - 1))) * (h_hl * Float32(1.64))
        for side in range(-1, 2, 2):
            var sz = Float32(side)
            var anchor = Vec3(tx, cy, sz * h_hw * Float32(0.92))
            var out_d = Vec3((_rand01_fol(i, side + 2) - Float32(0.5)) * Float32(0.35), Float32(0.35), sz * Float32(0.90)).normalized()
            _add_curved_crossfoil_sprig_pair(
                mb_hedge,
                anchor,
                out_d,
                Float32(0.22),
                Float32(0.24),
                Float32(0.02),
                Vec3(tx, cy, 0.0),
                Vec3(0.0, cy, 0.0),
                max(h_hl, h_fol),
                Float32(i) * _GOLDEN_ANGLE,
                normal_transfer=0.70,
                inner_col=0xFF347436,
                outer_col=0xFF74C652,
            )

    var hedge_m = mb_hedge^.finish()
    hedge_m.ensure_bounds()
    return (curb_m^, hedge_m^)


def build_flowering_bush(seed: Int, radius: Float32 = 0.85) -> NativeMesh:
    """Builds an organic domed flowering park bush (`alpha_mode = 1`, `double_sided = True`) combining
    crossed leafy sprig rosettes with vibrant 3D multi-petal flower clusters."""
    var r = max(Float32(0.30), min(Float32(50.0), radius)) if isfinite(radius) else Float32(0.85)
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.54, metallic=0.0, alpha_mode=1, double_sided=True)
    var center = Vec3(0.0, r * Float32(0.52), 0.0)

    # 14 crossed leafy sprig pairs forming a lush organic dome (`28` curved 3D sprig cards)
    for i in range(14):
        var phi = Float32(i) * _GOLDEN_ANGLE + _rand01_fol(seed, 400 + i) * Float32(0.3)
        var u_y = Float32(0.05) + (Float32(i) / Float32(13.0)) * Float32(0.92)
        var r_xz = sqrt(max(Float32(0.05), Float32(1.0) - u_y * u_y * Float32(0.85)))
        var dir = Vec3(cos(phi) * r_xz, u_y, sin(phi) * r_xz).normalized()
        var anchor = Vec3(0.0, r * Float32(0.12), 0.0) + dir * (r * Float32(0.22))
        _add_curved_crossfoil_sprig_pair(
            mb,
            anchor,
            dir,
            r * Float32(0.72),
            r * Float32(0.65),
            r * Float32(0.08),
            center,
            center,
            r,
            phi,
            normal_transfer=0.70,
            inner_col=0xFF2F6E34,
            outer_col=0xFF72C24E,
        )

    # 9 vibrant 3D cupped blossoms nestled across the sunlit outer dome
    var pal = seed % 3
    var bloom_col: UInt32 = UInt32(0xFFFF4D6D) if pal == 0 else (UInt32(0xFFFFBE0B) if pal == 1 else UInt32(0xFFDA77F2))
    for b_i in range(9):
        var ba = Float32(b_i) * _GOLDEN_ANGLE + Float32(0.8)
        var by_t = Float32(0.28) + (Float32(b_i) / Float32(8.0)) * Float32(0.68)
        var br_xz = sqrt(max(Float32(0.08), Float32(1.0) - by_t * by_t * Float32(0.8)))
        var b_dir = Vec3(cos(ba) * br_xz, by_t, sin(ba) * br_xz).normalized()
        var bp = center + b_dir * (r * Float32(0.78))
        mb.add_ellipsoid(bp, Vec3(r * Float32(0.11), r * Float32(0.08), r * Float32(0.11)), rings=4, segments=6, color=bloom_col)

    var out = mb^.finish()
    out.ensure_bounds()
    return out^
