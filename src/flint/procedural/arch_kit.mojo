"""Modular 3-Zone Architectural Kits, Clean Back-Wall `4x4` Interior Atlas & `512x512` Cyber-Billboards (`arch_kit.mojo`).

Implements Flint Engine v3 urban & architectural assets in 100% pure Mojo:
  1. `bake_interior_windowbox_atlas_512()`:
     - `512x512` `4x4` clean back-wall interior room atlas (`16` edge-to-edge `128x128` interior depth
       elevations with ZERO black outer window frames and ZERO baked horizontal blind stripes: multi-monitor
       trading desks, cyber-lab server racks with blade LEDs, executive bookcases & framed art, lounge neon
       bars, and indoor potted plants)
     - `512x512` exterior curtain-wall seam bevel + per-pane `±1.2°` float-glass warble normal map (`BC5`)
     - `512x512` coupled `0xAARRGGBB` ORM + interior luminance mask (`BC7_UNORM`).
  2. `bake_cyber_billboard_512(variant: Int)` (`0..7`):
     - `8` iconic `512x512` `0xFFRRGGBB` Neo-Tokyo / Night-City ad plates combining multi-stroke SDF
       Kanji/Katakana calligraphy (`電脳 / 新東京 / 重工 / 零式`) with stylized SDF Cyber-Kitsune, Turbine-Sun,
       `Vanguard` Hypercar, Oni Mask, Cyber-Optics Iris, Synth-Cola, Biotech Helix & Dragon emblems,
       RGB LED subpixel micro-grid, and glowing neon border tubes!
  3. Modular 3D chamfered architectural & street-furniture kits (`NativeMesh` + `_parts` `(base, emissive)` pairs):
     - `build_colonnade_arcade_bay()`
     - `build_facade_mullion_spandrel_grid()`
     - `build_rooftop_hvac_chiller()`
     - `build_rooftop_vtol_helipad()` & `build_rooftop_vtol_helipad_parts()`
     - `build_street_traffic_cantilever()` & `build_street_traffic_cantilever_parts()`
     - `build_street_bollard_and_grate()` & `build_street_bollard_and_grate_parts()`
     - `build_transit_shelter()` & `build_transit_shelter_parts()`
"""
from std.math import cos, max, min, sin, sqrt
from flint.core.math import Vec2, Vec3
from flint.core.transform import Quat
from flint.assets.model import NativeMesh
from flint.assets.mesh_builder import MeshBuilder
from flint.procedural.loft_bevel import extrude_profile_beveled

comptime _TAU_ARCH: Float32 = 6.28318530
comptime _GLASS_WARBLE_RAD: Float32 = 0.0209424


@always_inline
def _hash_pane(px: Int, py: Int, salt: Int) -> Float32:
    var h = UInt32(((px * 374761393) + (py * 668265263) + (salt * 1442695041)) & 0x7FFFFFFF)
    h = (h ^ (h >> 13)) * UInt32(1274126177)
    h = h ^ (h >> 16)
    return Float32(h & UInt32(0x00FFFFFF)) / Float32(16777216.0)


@always_inline
def _pack_argb(a: Int, r: Int, g: Int, b: Int) -> UInt32:
    return (
        (UInt32(max(0, min(255, a))) << 24)
        | (UInt32(max(0, min(255, r))) << 16)
        | (UInt32(max(0, min(255, g))) << 8)
        | UInt32(max(0, min(255, b)))
    )


def _finalize_arch_ao(mut mesh: NativeMesh, height_span: Float32, soffit_y: Float32 = -1.0):
    """Bakes ground-contact darkening (`y -> 0`) and overhead soffit cavity AO into `mesh.ao`."""
    var vc = mesh.vertex_count()
    if len(mesh.ao) < vc:
        mesh.ao = List[UInt8](length=vc, fill=UInt8(245))
    var inv_h = Float32(1.0) / max(Float32(0.25), height_span)

    for i in range(vc):
        var py = mesh.positions[i * 3 + 1]
        var ny = mesh.normals[i * 3 + 1]
        var base_ao = Float32(Int(mesh.ao[i])) / Float32(255.0)
        var g_t = max(Float32(0.0), min(Float32(1.0), py / Float32(0.45)))
        var g_occ = Float32(0.46) + Float32(0.54) * (g_t * g_t * (Float32(3.0) - Float32(2.0) * g_t))
        var s_occ = Float32(1.0) - max(Float32(0.0), -ny) * Float32(0.28)
        if soffit_y > Float32(0.0) and abs(py - soffit_y) < Float32(0.35) and ny < Float32(-0.3):
            s_occ *= Float32(0.72)
        var sky_lift = Float32(0.88) + Float32(0.12) * max(Float32(0.0), min(Float32(1.0), py * inv_h))
        var final_ao = max(Float32(0.32), min(Float32(1.0), base_ao * g_occ * s_occ * sky_lift))
        mesh.ao[i] = UInt8(max(78, min(255, Int(final_ao * Float32(255.0) + Float32(0.5)))))
    mesh.ensure_bounds()


def _merge_arch_pair(base_m: NativeMesh, emis_m: NativeMesh) -> NativeMesh:
    """Combines `(base_mesh, emissive_mesh)` into a single vertex-colored `NativeMesh`."""
    var out = base_m.copy()
    var vc0 = out.vertex_count()
    var evc = emis_m.vertex_count()
    for i in range(evc):
        out.positions.append(emis_m.positions[i * 3 + 0])
        out.positions.append(emis_m.positions[i * 3 + 1])
        out.positions.append(emis_m.positions[i * 3 + 2])
        out.normals.append(emis_m.normals[i * 3 + 0])
        out.normals.append(emis_m.normals[i * 3 + 1])
        out.normals.append(emis_m.normals[i * 3 + 2])
        out.texcoords.append(emis_m.texcoords[i * 2 + 0])
        out.texcoords.append(emis_m.texcoords[i * 2 + 1])
        out.colors.append(emis_m.colors[i * 4 + 0])
        out.colors.append(emis_m.colors[i * 4 + 1])
        out.colors.append(emis_m.colors[i * 4 + 2])
        out.colors.append(emis_m.colors[i * 4 + 3])
        out.ao.append(emis_m.ao[i])
    for idx in emis_m.indices:
        out.indices.append(vc0 + idx)
    out.ensure_bounds()
    return out^


# ============================================================================
# 1. Clean Back-Wall `512x512` `4x4` Interior Room Atlas + `±1.2°` Glass Warble
# ============================================================================

def bake_interior_windowbox_atlas_512() -> Tuple[List[UInt32], List[Vec3], List[UInt32]]:
    """Bakes a `512x512` (`4 x 4` tiles of `128x128`) clean back-wall interior room atlas:
    - `albedo_argb`: `16` full edge-to-edge interior back-wall scenes (ZERO black outer window frames
      and ZERO horizontal Venetian blind stripes — cleanly separated for the v3 3D WindowBox shader!):
        - `0..3`: Multi-monitor trading & engineering suites + glass partitions + potted plants
        - `4..7`: Cyber-lab mainframe server racks with blade LEDs + holographic wall display
        - `8..11`: Warm executive bookcases (discrete vertical book spines) + framed art + potted palms
        - `12..15`: Sky-lounge neon ring bars + backlit bottle alcoves + barstools & planters.
    - `mullion_warble_normals`: exterior curtain-wall `3px` bevel + per-pane `±1.2°` float-glass warble (`BC5`).
    - `orm_argb`: `R=AO`, `G=Roughness` (`14` mirror glass, `118` exterior mullion), `B=Metallic`, `A=EmissionMask`.
    """
    comptime DIM: Int = 512
    comptime TILE: Int = 128
    var albedo = List[UInt32](length=DIM * DIM, fill=UInt32(0xFF282E3A))
    var normals = List[Vec3](length=DIM * DIM, fill=Vec3(0.0, 0.0, 1.0))
    var orm = List[UInt32](length=DIM * DIM, fill=UInt32(0xFFFF0E10))

    for y in range(DIM):
        var ty = y // TILE
        var ly = y % TILE
        var v_norm = Float32(ly) / Float32(TILE - 1)  # 0 = top of back wall, 1 = floor baseboard

        for x in range(DIM):
            var tx = x // TILE
            var lx = x % TILE
            var u_norm = Float32(lx) / Float32(TILE - 1)
            var room_id = ty * 4 + tx
            var idx = y * DIM + x

            # Exterior window-pane normal map ONLY (`mullion_warble_normals` & `orm` bevel seam; `albedo` stays 100% clean!)
            var is_seam_x = (lx < 3 or lx >= TILE - 3)
            var is_seam_y = (ly < 3 or ly >= TILE - 3)
            if is_seam_x or is_seam_y:
                var bnx = Float32(-0.44) if lx < 3 else (Float32(0.44) if lx >= TILE - 3 else Float32(0.0))
                var bny = Float32(0.44) if ly < 3 else (Float32(-0.44) if ly >= TILE - 3 else Float32(0.0))
                normals[idx] = Vec3(bnx, bny, Float32(0.88)).normalized()
            else:
                var ph_x = _hash_pane(tx, ty, 11) * _TAU_ARCH
                var ph_y = _hash_pane(tx, ty, 23) * _TAU_ARCH
                var freq_x = Float32(1.0) + _hash_pane(tx, ty, 37) * Float32(1.4)
                var freq_y = Float32(1.0) + _hash_pane(tx, ty, 53) * Float32(1.4)
                var pillow_x = (u_norm - Float32(0.5)) * Float32(2.0)
                var pillow_y = (v_norm - Float32(0.5)) * Float32(2.0)
                var wx = (sin(u_norm * _TAU_ARCH * freq_x + ph_x) * Float32(0.65) + pillow_x * Float32(0.35)) * _GLASS_WARBLE_RAD
                var wy = (cos(v_norm * _TAU_ARCH * freq_y + ph_y) * Float32(0.65) + pillow_y * Float32(0.35)) * _GLASS_WARBLE_RAD
                normals[idx] = Vec3(wx, wy, Float32(1.0)).normalized()

            # Clean edge-to-edge interior back-wall shading (`0..127` full tile!)
            var wall_wash = Float32(0.92) - v_norm * Float32(0.30)
            # Soft twin ceiling-downlight scallops washing the upper back wall
            var dl1 = max(Float32(0.0), Float32(1.0) - ((u_norm - Float32(0.30)) * (u_norm - Float32(0.30)) * Float32(9.0) + v_norm * Float32(1.1)))
            var dl2 = max(Float32(0.0), Float32(1.0) - ((u_norm - Float32(0.70)) * (u_norm - Float32(0.70)) * Float32(9.0) + v_norm * Float32(1.1)))
            wall_wash += (dl1 + dl2) * Float32(0.24)

            var r_f: Float32
            var g_f: Float32
            var b_f: Float32
            var emis_a: Int

            if room_id <= 3:
                # Archetype A (`0..3`): Warm-White Trading & Engineering Suites + Triple Monitors + Potted Plant
                var panel_seam = ((lx % 32) == 0)
                var base_w = (Float32(0.84) if panel_seam else Float32(1.0)) * wall_wash
                r_f = Float32(198.0) * base_w
                g_f = Float32(192.0) * base_w
                b_f = Float32(178.0) * base_w
                emis_a = 152

                # Framed glass whiteboard / analytics wall chart (`lx = 16..84, ly = 22..56`)
                if lx >= 16 and lx <= 84 and ly >= 22 and ly <= 56:
                    var brd_frame = (lx == 16 or lx == 84 or ly == 22 or ly == 56)
                    if brd_frame:
                        r_f = Float32(72.0); g_f = Float32(80.0); b_f = Float32(94.0)
                    else:
                        r_f = Float32(218.0); g_f = Float32(226.0); b_f = Float32(236.0)
                        emis_a = 185

                # Triple widescreen trading monitors (`ly = 66..88`, `3` distinct screens)
                var m1 = (lx >= 14 and lx <= 36)
                var m2 = (lx >= 40 and lx <= 64)
                var m3 = (lx >= 68 and lx <= 90)
                if ly >= 66 and ly <= 88 and (m1 or m2 or m3):
                    var bezel = (ly == 66 or ly == 88 or lx == 14 or lx == 36 or lx == 40 or lx == 64 or lx == 68 or lx == 90)
                    if bezel:
                        r_f = Float32(26.0); g_f = Float32(30.0); b_f = Float32(38.0); emis_a = 70
                    else:
                        var chart_y = 78 - Int(sin(Float32(lx + room_id * 9) * Float32(0.28)) * Float32(6.0))
                        var is_trace = abs(ly - chart_y) <= 1
                        r_f = Float32(95.0) if not is_trace else Float32(80.0)
                        g_f = Float32(205.0) if not is_trace else Float32(252.0)
                        b_f = Float32(248.0) if not is_trace else Float32(195.0)
                        emis_a = 245

                # Potted indoor monstera/ficus plant (`lx = 96..118, ly = 58..116`)
                var pdx = Float32(lx - 107)
                var pdy = Float32(ly - 76)
                if pdx * pdx + pdy * pdy * Float32(0.65) < Float32(128.0):
                    var leaf_n = _hash_pane(lx // 3, ly // 3, room_id + 5)
                    if leaf_n > Float32(0.22):
                        r_f = Float32(52.0) + leaf_n * Float32(40.0)
                        g_f = Float32(122.0) + leaf_n * Float32(54.0)
                        b_f = Float32(58.0) + leaf_n * Float32(24.0)
                elif lx >= 101 and lx <= 113 and ly >= 92 and ly <= 118:
                    r_f = Float32(178.0); g_f = Float32(168.0); b_f = Float32(154.0)

                # Desk & credenza silhouette along lower wall (`ly >= 92`)
                if ly >= 92 and lx < 98:
                    r_f = Float32(82.0) if ly < 96 else Float32(54.0)
                    g_f = Float32(64.0) if ly < 96 else Float32(48.0)
                    b_f = Float32(50.0) if ly < 96 else Float32(46.0)
                    emis_a = 105

            elif room_id <= 7:
                # Archetype B (`4..7`): Cool Cyber-Labs, Mainframe Server Towers & Holo-Display
                r_f = Float32(64.0) * wall_wash
                g_f = Float32(106.0) * wall_wash
                b_f = Float32(138.0) * wall_wash
                emis_a = 148

                # 2 outer mainframe server towers (`lx = 10..38` and `lx = 90..118`, `ly = 18..122`)
                var in_rack_l = (lx >= 10 and lx <= 38 and ly >= 18 and ly <= 122)
                var in_rack_r = (lx >= 90 and lx <= 118 and ly >= 18 and ly <= 122)
                if in_rack_l or in_rack_r:
                    r_f = Float32(28.0); g_f = Float32(36.0); b_f = Float32(48.0); emis_a = 85
                    var slot_u = (ly - 20) % 8
                    var led_col = (lx - 12) % 6
                    if slot_u >= 2 and slot_u <= 4 and led_col >= 1 and led_col <= 3:
                        var h_led = _hash_pane(lx // 6, ly // 8, room_id * 17)
                        if h_led > Float32(0.36):
                            if h_led > Float32(0.84):
                                r_f = Float32(255.0); g_f = Float32(178.0); b_f = Float32(42.0)
                            else:
                                r_f = Float32(38.0); g_f = Float32(238.0); b_f = Float32(255.0)
                            emis_a = 248

                # Central wall-mounted cyan/emerald holographic diagnostic screen (`lx = 44..84, ly = 32..82`)
                if lx >= 44 and lx <= 84 and ly >= 32 and ly <= 82:
                    var hdx = Float32(lx - 64)
                    var hdy = Float32(ly - 57)
                    var rad_h = sqrt(hdx * hdx + hdy * hdy)
                    var ring = abs(rad_h - Float32(14.0)) < Float32(2.2) or abs(rad_h - Float32(7.0)) < Float32(1.5)
                    r_f = Float32(52.0) if not ring else Float32(140.0)
                    g_f = Float32(188.0) if not ring else Float32(252.0)
                    b_f = Float32(226.0) if not ring else Float32(255.0)
                    emis_a = 242 if ring else 210

                if ly >= 94 and lx >= 40 and lx <= 88:
                    r_f = Float32(44.0); g_f = Float32(56.0); b_f = Float32(70.0); emis_a = 95

            elif room_id <= 11:
                # Archetype C (`8..11`): Warm `2800K` Executive Library Bookcases & Spotlighted Framed Art
                var wood_slat = Float32(0.92) + Float32(0.08) * sin(Float32(lx) * Float32(1.2))
                r_f = Float32(214.0) * wall_wash * wood_slat
                g_f = Float32(162.0) * wall_wash * wood_slat
                b_f = Float32(104.0) * wall_wash * wood_slat
                emis_a = 158

                # Built-in recessed bookcase (`lx = 10..54, ly = 20..92`) with discrete vertical book spines!
                if lx >= 10 and lx <= 54 and ly >= 20 and ly <= 92:
                    var shelf_idx = (ly - 20) // 24
                    var shelf_y = (ly - 20) % 24
                    if shelf_y >= 21:
                        # Wooden shelf plank (limited purely to `lx = 10..54`!)
                        r_f = Float32(118.0); g_f = Float32(76.0); b_f = Float32(44.0); emis_a = 110
                    else:
                        var book_id = (lx - 10) // 4
                        var book_h = 6 + Int(_hash_pane(book_id, shelf_idx, room_id + 31) * Float32(10.0))
                        if shelf_y < book_h:
                            # Warm LED cove strip illuminating back of bookcase niche
                            var cove = Float32(1.18) - Float32(shelf_y) * Float32(0.05)
                            r_f = Float32(238.0) * cove; g_f = Float32(192.0) * cove; b_f = Float32(128.0) * cove
                            emis_a = 225
                        else:
                            var bhue = _hash_pane(book_id, shelf_idx, room_id + 73)
                            r_f = Float32(75.0) + bhue * Float32(135.0)
                            g_f = Float32(55.0) + (Float32(1.0) - bhue) * Float32(95.0)
                            b_f = Float32(50.0) + _hash_pane(book_id, shelf_idx, 99) * Float32(90.0)
                            emis_a = 140

                # Spotlighted Modern Framed Canvas Artwork (`lx = 66..114, ly = 26..76`)
                if lx >= 66 and lx <= 114 and ly >= 26 and ly <= 76:
                    var is_frame = (lx < 69 or lx > 111 or ly < 29 or ly > 73)
                    if is_frame:
                        r_f = Float32(48.0); g_f = Float32(38.0); b_f = Float32(30.0); emis_a = 95
                    else:
                        var ax = Float32(lx - 90)
                        var ay = Float32(ly - 51)
                        var sun_disc = (ax * ax + ay * ay) < Float32(144.0)
                        r_f = Float32(236.0) if sun_disc else Float32(228.0)
                        g_f = Float32(118.0) if sun_disc else Float32(214.0)
                        b_f = Float32(68.0) if sun_disc else Float32(188.0)
                        emis_a = 215

                # Warm cognac leather sofa silhouette (`ly >= 92`)
                if ly >= 92:
                    r_f = Float32(112.0); g_f = Float32(68.0); b_f = Float32(42.0); emis_a = 105

            else:
                # Archetype D (`12..15`): Sky-Lounge Neon Ring Bar & Penthouse Cocktail Suite
                r_f = Float32(82.0) * wall_wash
                g_f = Float32(68.0) * wall_wash
                b_f = Float32(108.0) * wall_wash
                emis_a = 142

                # Circular neon wall halo (`center (64, 50), radius = 22`)
                var ndx = Float32(lx - 64)
                var ndy = Float32(ly - 50)
                var ndist = sqrt(ndx * ndx + ndy * ndy)
                var ring_glow = max(Float32(0.0), Float32(1.0) - abs(ndist - Float32(21.0)) * Float32(0.28))
                if ring_glow > Float32(0.01):
                    var mag_mode = (room_id & 1) == 0
                    var nr = Float32(255.0) if mag_mode else Float32(48.0)
                    var ng = Float32(68.0) if mag_mode else Float32(228.0)
                    var nb = Float32(188.0) if mag_mode else Float32(255.0)
                    r_f = r_f + nr * ring_glow
                    g_f = g_f + ng * ring_glow
                    b_f = b_f + nb * ring_glow
                    emis_a = max(emis_a, Int(Float32(150.0) + ring_glow * Float32(105.0)))

                # Twin warm amber pendant lamps (`lx = 24, 104`, `ly = 38`)
                for p_side in range(2):
                    var px_l = 24 if p_side == 0 else 104
                    var ldx = Float32(lx - px_l)
                    var ldy = Float32(ly - 38)
                    var ld2 = ldx * ldx + ldy * ldy
                    if ld2 < Float32(36.0):
                        r_f = Float32(255.0); g_f = Float32(206.0); b_f = Float32(118.0); emis_a = 250

                # Backlit amber bar counter (`ly = 88..94`) & dark lounge base (`ly > 94`)
                if ly >= 88 and ly <= 94 and lx >= 14 and lx <= 114:
                    r_f = Float32(238.0); g_f = Float32(166.0); b_f = Float32(78.0); emis_a = 228
                elif ly > 94:
                    r_f = Float32(46.0); g_f = Float32(38.0); b_f = Float32(56.0); emis_a = 95

            albedo[idx] = _pack_argb(emis_a, Int(r_f), Int(g_f), Int(b_f))
            if is_seam_x or is_seam_y:
                orm[idx] = _pack_argb(emis_a, 165, 118, 210)
            else:
                orm[idx] = _pack_argb(emis_a, 255, 14, 16)

    return (albedo^, normals^, orm^)


# ============================================================================
# 2. `512x512` Neo-Tokyo / Night-City Kanji & SDF Illustrated Billboards (`0..7`)
# ============================================================================

def _draw_neon_segment(
    mut buf_r: List[Float32],
    mut buf_g: List[Float32],
    mut buf_b: List[Float32],
    dim: Int,
    ax: Float32,
    ay: Float32,
    bx: Float32,
    by: Float32,
    core_r: Float32,
    halo_r: Float32,
    col_r: Float32,
    col_g: Float32,
    col_b: Float32,
):
    """Additive/max-blended SDF neon stroke rasterizer with white-hot core + chromatic bloom halo."""
    var pad = core_r + halo_r + Float32(2.0)
    var x0 = max(0, Int(min(ax, bx) - pad))
    var x1 = min(dim - 1, Int(max(ax, bx) + pad) + 1)
    var y0 = max(0, Int(min(ay, by) - pad))
    var y1 = min(dim - 1, Int(max(ay, by) + pad) + 1)

    var abx = bx - ax
    var aby = by - ay
    var ab2 = max(Float32(1e-4), abx * abx + aby * aby)
    var inv_halo = Float32(1.0) / max(Float32(1.0), halo_r)

    for py in range(y0, y1 + 1):
        var fy = Float32(py) + Float32(0.5)
        for px in range(x0, x1 + 1):
            var fx = Float32(px) + Float32(0.5)
            var t = max(Float32(0.0), min(Float32(1.0), ((fx - ax) * abx + (fy - ay) * aby) / ab2))
            var dx = fx - (ax + abx * t)
            var dy = fy - (ay + aby * t)
            var d = sqrt(dx * dx + dy * dy)
            if d > core_r + halo_r:
                continue
            var core = max(Float32(0.0), min(Float32(1.0), (core_r + Float32(1.0) - d)))
            var halo = max(Float32(0.0), Float32(1.0) - max(Float32(0.0), d - core_r) * inv_halo)
            var glow = halo * halo * Float32(0.58)
            var idx = py * dim + px
            buf_r[idx] = max(buf_r[idx], col_r * glow + (col_r * Float32(0.65) + Float32(95.0)) * core)
            buf_g[idx] = max(buf_g[idx], col_g * glow + (col_g * Float32(0.65) + Float32(95.0)) * core)
            buf_b[idx] = max(buf_b[idx], col_b * glow + (col_b * Float32(0.65) + Float32(95.0)) * core)


def _draw_neon_ring(
    mut buf_r: List[Float32],
    mut buf_g: List[Float32],
    mut buf_b: List[Float32],
    dim: Int,
    cx: Float32,
    cy: Float32,
    radius: Float32,
    core_w: Float32,
    halo_w: Float32,
    col_r: Float32,
    col_g: Float32,
    col_b: Float32,
    filled: Bool = False,
):
    var pad = radius + core_w + halo_w + Float32(2.0)
    var x0 = max(0, Int(cx - pad))
    var x1 = min(dim - 1, Int(cx + pad) + 1)
    var y0 = max(0, Int(cy - pad))
    var y1 = min(dim - 1, Int(cy + pad) + 1)
    var inv_halo = Float32(1.0) / max(Float32(1.0), halo_w)

    for py in range(y0, y1 + 1):
        var dy = Float32(py) + Float32(0.5) - cy
        for px in range(x0, x1 + 1):
            var dx = Float32(px) + Float32(0.5) - cx
            var r = sqrt(dx * dx + dy * dy)
            var d = max(Float32(0.0), r - radius) if filled else abs(r - radius)
            if d > core_w + halo_w:
                continue
            var core = max(Float32(0.0), min(Float32(1.0), core_w + Float32(1.0) - d))
            var halo = max(Float32(0.0), Float32(1.0) - max(Float32(0.0), d - core_w) * inv_halo)
            var glow = halo * halo * Float32(0.55)
            var core_boost = Float32(0.88) if filled else Float32(1.0)
            var idx = py * dim + px
            buf_r[idx] = max(buf_r[idx], col_r * glow + col_r * core * core_boost)
            buf_g[idx] = max(buf_g[idx], col_g * glow + col_g * core * core_boost)
            buf_b[idx] = max(buf_b[idx], col_b * glow + col_b * core * core_boost)


def _draw_kanji_glyph(
    mut buf_r: List[Float32],
    mut buf_g: List[Float32],
    mut buf_b: List[Float32],
    dim: Int,
    glyph_id: Int,
    ox: Float32,
    oy: Float32,
    size: Float32,
    stroke_r: Float32,
    col_r: Float32,
    col_g: Float32,
    col_b: Float32,
):
    """Renders crisp multi-stroke Japanese Kanji / Katakana calligraphy (`glyph_id = 0..7`) in `[0..1]^2` box."""
    # Each stroke is `[x0, y0, x1, y1]` in normalized `[0..1]` glyph cell coordinates
    var strokes: List[List[Float32]]
    var gid = glyph_id & 7
    if gid == 0:
        # `電` (DEN - Electricity / Cyber)
        strokes = [
            [0.15, 0.08, 0.85, 0.08], [0.50, 0.08, 0.50, 0.36],
            [0.10, 0.20, 0.90, 0.20], [0.10, 0.20, 0.10, 0.34], [0.90, 0.20, 0.90, 0.34],
            [0.22, 0.28, 0.38, 0.28], [0.62, 0.28, 0.78, 0.28],
            [0.18, 0.44, 0.82, 0.44], [0.82, 0.44, 0.82, 0.72], [0.18, 0.72, 0.82, 0.72], [0.18, 0.44, 0.18, 0.72],
            [0.18, 0.58, 0.82, 0.58], [0.50, 0.44, 0.50, 0.90], [0.50, 0.90, 0.90, 0.90], [0.90, 0.90, 0.90, 0.78],
        ]
    elif gid == 1:
        # `脳` (NOU - Brain / Neural)
        strokes = [
            [0.12, 0.12, 0.34, 0.12], [0.34, 0.12, 0.34, 0.90], [0.12, 0.12, 0.12, 0.90],
            [0.12, 0.38, 0.34, 0.38], [0.12, 0.62, 0.34, 0.62],
            [0.46, 0.14, 0.54, 0.32], [0.64, 0.12, 0.70, 0.30], [0.88, 0.14, 0.78, 0.32],
            [0.46, 0.44, 0.88, 0.44], [0.88, 0.44, 0.88, 0.88], [0.46, 0.88, 0.88, 0.88], [0.46, 0.44, 0.46, 0.88],
            [0.54, 0.54, 0.80, 0.78], [0.80, 0.54, 0.54, 0.78],
        ]
    elif gid == 2:
        # `新` (SHIN - Neo / New)
        strokes = [
            [0.30, 0.08, 0.30, 0.20], [0.10, 0.20, 0.50, 0.20],
            [0.18, 0.22, 0.24, 0.38], [0.42, 0.22, 0.36, 0.38],
            [0.08, 0.40, 0.52, 0.40], [0.12, 0.56, 0.48, 0.56],
            [0.30, 0.40, 0.30, 0.92], [0.28, 0.58, 0.10, 0.84], [0.32, 0.60, 0.48, 0.78],
            [0.86, 0.12, 0.58, 0.24], [0.58, 0.24, 0.58, 0.88],
            [0.58, 0.46, 0.92, 0.46], [0.78, 0.46, 0.78, 0.92],
        ]
    elif gid == 3:
        # `東` (TOU - East / Tokyo)
        strokes = [
            [0.14, 0.18, 0.86, 0.18],
            [0.20, 0.32, 0.80, 0.32], [0.80, 0.32, 0.80, 0.62], [0.20, 0.62, 0.80, 0.62], [0.20, 0.32, 0.20, 0.62],
            [0.20, 0.47, 0.80, 0.47],
            [0.50, 0.06, 0.50, 0.94],
            [0.46, 0.62, 0.12, 0.88], [0.54, 0.62, 0.88, 0.88],
        ]
    elif gid == 4:
        # `重` (JUU - Heavy Industries)
        strokes = [
            [0.78, 0.10, 0.22, 0.16],
            [0.10, 0.28, 0.90, 0.28],
            [0.22, 0.40, 0.78, 0.40], [0.78, 0.40, 0.78, 0.66], [0.22, 0.66, 0.78, 0.66], [0.22, 0.40, 0.22, 0.66],
            [0.22, 0.53, 0.78, 0.53],
            [0.50, 0.14, 0.50, 0.90],
            [0.20, 0.78, 0.80, 0.78], [0.10, 0.90, 0.90, 0.90],
        ]
    elif gid == 5:
        # `零` (REI - Zero-Type)
        strokes = [
            [0.18, 0.10, 0.82, 0.10], [0.50, 0.10, 0.50, 0.36],
            [0.12, 0.22, 0.88, 0.22], [0.12, 0.22, 0.12, 0.36], [0.88, 0.22, 0.88, 0.36],
            [0.50, 0.40, 0.12, 0.64], [0.50, 0.40, 0.88, 0.64],
            [0.34, 0.58, 0.66, 0.58],
            [0.24, 0.72, 0.76, 0.72], [0.76, 0.72, 0.54, 0.88], [0.38, 0.80, 0.64, 0.94],
        ]
    elif gid == 6:
        # `光` (HIKARI - Light / Optics)
        strokes = [
            [0.50, 0.08, 0.50, 0.44], [0.20, 0.18, 0.34, 0.38], [0.80, 0.18, 0.66, 0.38],
            [0.10, 0.46, 0.90, 0.46],
            [0.40, 0.46, 0.16, 0.90],
            [0.60, 0.46, 0.60, 0.88], [0.60, 0.88, 0.88, 0.88], [0.88, 0.88, 0.88, 0.74],
        ]
    else:
        # `龍` / `サイバー` Stylized Katakana `サ イ` Emblem
        strokes = [
            [0.12, 0.30, 0.88, 0.30],
            [0.32, 0.12, 0.32, 0.56],
            [0.68, 0.10, 0.68, 0.62], [0.68, 0.62, 0.30, 0.90],
        ]

    var halo_r = stroke_r * Float32(3.0)
    for si in range(len(strokes)):
        ref s = strokes[si]
        _draw_neon_segment(
            buf_r, buf_g, buf_b, dim,
            ox + s[0] * size, oy + s[1] * size,
            ox + s[2] * size, oy + s[3] * size,
            stroke_r, halo_r, col_r, col_g, col_b,
        )


def bake_cyber_billboard_512(variant: Int) -> List[UInt32]:
    """Bakes one of `8` (`0..7`) iconic `512x512` `0xFFRRGGBB` Neo-Tokyo / Night-City neon billboards:
      - `0`: `KITSUNE CYBERNETICS` (`電脳` + SDF Cyber-Kitsune Fox Mask)
      - `1`: `SHIN-TOKYO HEAVY IND.` (`新東重` + Rising Cyber-Sun & Turbine)
      - `2`: `VANGUARD ZERO-TYPE` (`零` + Widebody Wedge Hypercar Silhouette & Laser Horizon)
      - `3`: `ONI TACTICAL DEFENSE` (`電重` + Samurai Oni Mask & Tusks)
      - `4`: `KBM CYBER-OPTICS` (`光脳` + Concentric Bionic Iris & HUD Reticle)
      - `5`: `SYNTH-COLA 2099` (`サ` + Effervescent Neon Bottle & Fizz Bubbles)
      - `6`: `HELIX BIOTECH` (`新光` + Intertwining DNA Double-Helix)
      - `7`: `DRAGON EXPRESS` (`東電` + S-Curve Cyber-Dragon & Lantern Crest)
    """
    comptime DIM: Int = 512
    var v_id = variant & 7
    var buf_r = List[Float32](length=DIM * DIM, fill=0.0)
    var buf_g = List[Float32](length=DIM * DIM, fill=0.0)
    var buf_b = List[Float32](length=DIM * DIM, fill=0.0)

    # 1. Rich atmospheric gradient backdrop per variant
    for y in range(DIM):
        var fy = Float32(y) / Float32(DIM - 1)
        for x in range(DIM):
            var fx = Float32(x) / Float32(DIM - 1)
            var idx = y * DIM + x
            if v_id == 0:
                buf_r[idx] = Float32(18.0) + fy * Float32(34.0)
                buf_g[idx] = Float32(12.0) + (Float32(1.0) - fy) * Float32(18.0)
                buf_b[idx] = Float32(42.0) + (Float32(1.0) - fx) * Float32(36.0)
            elif v_id == 1:
                buf_r[idx] = Float32(38.0) + (Float32(1.0) - fy) * Float32(42.0)
                buf_g[idx] = Float32(20.0) + (Float32(1.0) - fy) * Float32(18.0)
                buf_b[idx] = Float32(14.0)
            elif v_id == 2:
                var grid_h = Float32(1.0) if (y > 320 and ((y % 24) == 0 or ((x - 256) % 48) == 0)) else Float32(0.0)
                buf_r[idx] = Float32(16.0) + fy * Float32(48.0) + grid_h * Float32(65.0)
                buf_g[idx] = Float32(14.0) + grid_h * Float32(18.0)
                buf_b[idx] = Float32(48.0) + (Float32(1.0) - fy) * Float32(44.0) + grid_h * Float32(85.0)
            elif v_id == 3:
                buf_r[idx] = Float32(46.0) * (Float32(1.0) - fy * Float32(0.5))
                buf_g[idx] = Float32(10.0)
                buf_b[idx] = Float32(18.0)
            elif v_id == 4:
                buf_r[idx] = Float32(10.0)
                buf_g[idx] = Float32(28.0) + (Float32(1.0) - fy) * Float32(28.0)
                buf_b[idx] = Float32(46.0) + fx * Float32(32.0)
            elif v_id == 5:
                var split = (fx * Float32(0.7) + fy * Float32(0.3)) > Float32(0.52)
                buf_r[idx] = Float32(78.0) if split else Float32(14.0)
                buf_g[idx] = Float32(16.0) if split else Float32(58.0)
                buf_b[idx] = Float32(64.0) if split else Float32(76.0)
            elif v_id == 6:
                buf_r[idx] = Float32(10.0)
                buf_g[idx] = Float32(38.0) + (Float32(1.0) - fy) * Float32(24.0)
                buf_b[idx] = Float32(28.0) + fy * Float32(24.0)
            else:
                buf_r[idx] = Float32(42.0) + fy * Float32(28.0)
                buf_g[idx] = Float32(14.0)
                buf_b[idx] = Float32(56.0) * (Float32(1.0) - fx * Float32(0.4))

    # 2. Glowing double neon tube perimeter frame (`x, y in [18..494]`)
    var fr_r = Float32(255.0) if (v_id == 0 or v_id == 3 or v_id == 5) else Float32(36.0)
    var fr_g = Float32(48.0) if (v_id == 3) else (Float32(188.0) if v_id == 1 or v_id == 7 else Float32(236.0))
    var fr_b = Float32(255.0) if (v_id != 1 and v_id != 3) else Float32(64.0)
    _draw_neon_segment(buf_r, buf_g, buf_b, DIM, 20.0, 20.0, 492.0, 20.0, 3.0, 12.0, fr_r, fr_g, fr_b)
    _draw_neon_segment(buf_r, buf_g, buf_b, DIM, 492.0, 20.0, 492.0, 492.0, 3.0, 12.0, fr_r, fr_g, fr_b)
    _draw_neon_segment(buf_r, buf_g, buf_b, DIM, 492.0, 492.0, 20.0, 492.0, 3.0, 12.0, fr_r, fr_g, fr_b)
    _draw_neon_segment(buf_r, buf_g, buf_b, DIM, 20.0, 492.0, 20.0, 20.0, 3.0, 12.0, fr_r, fr_g, fr_b)

    # 3. Variant-specific SDF vector artwork + Multi-Stroke Kanji Calligraphy!
    if v_id == 0:
        # Variant 0: SDF Cyber-Kitsune Fox Mask (`center (318, 256)`) + Vertical `電 脳` Kanji
        _draw_kanji_glyph(buf_r, buf_g, buf_b, DIM, 0, 44.0, 56.0, 132.0, 4.2, 32.0, 244.0, 255.0)
        _draw_kanji_glyph(buf_r, buf_g, buf_b, DIM, 1, 44.0, 214.0, 132.0, 4.2, 32.0, 244.0, 255.0)
        # Kitsune ears, cheekbones, snout & glowing cyan eyes
        var fox: List[List[Float32]] = [
            [218.0, 92.0, 268.0, 196.0], [218.0, 92.0, 206.0, 232.0],
            [418.0, 92.0, 368.0, 196.0], [418.0, 92.0, 430.0, 232.0],
            [268.0, 196.0, 368.0, 196.0],
            [206.0, 232.0, 318.0, 396.0], [430.0, 232.0, 318.0, 396.0],
            [236.0, 132.0, 254.0, 196.0], [400.0, 132.0, 382.0, 196.0],
        ]
        for i in range(len(fox)):
            ref s = fox[i]
            _draw_neon_segment(buf_r, buf_g, buf_b, DIM, s[0], s[1], s[2], s[3], 4.0, 16.0, 255.0, 38.0, 148.0)
        # Angled glowing cyan fox eyes + forehead diamond
        _draw_neon_segment(buf_r, buf_g, buf_b, DIM, 246.0, 252.0, 294.0, 272.0, 4.5, 14.0, 36.0, 248.0, 255.0)
        _draw_neon_segment(buf_r, buf_g, buf_b, DIM, 390.0, 252.0, 342.0, 272.0, 4.5, 14.0, 36.0, 248.0, 255.0)
        _draw_neon_ring(buf_r, buf_g, buf_b, DIM, 318.0, 222.0, 14.0, 3.0, 10.0, 255.0, 208.0, 48.0)

    elif v_id == 1:
        # Variant 1: Shin-Tokyo Heavy Industries (`新 東 重`) + Blazing Rising Turbine Sun
        _draw_neon_ring(buf_r, buf_g, buf_b, DIM, 336.0, 220.0, 96.0, 4.0, 22.0, 255.0, 132.0, 28.0, filled=True)
        _draw_neon_ring(buf_r, buf_g, buf_b, DIM, 336.0, 220.0, 128.0, 3.5, 14.0, 255.0, 206.0, 52.0)
        for r_i in range(12):
            var a = (Float32(r_i) / Float32(12.0)) * _TAU_ARCH
            _draw_neon_segment(
                buf_r, buf_g, buf_b, DIM,
                Float32(336.0) + cos(a) * Float32(106.0), Float32(220.0) + sin(a) * Float32(106.0),
                Float32(336.0) + cos(a) * Float32(154.0), Float32(220.0) + sin(a) * Float32(154.0),
                3.5, 10.0, 255.0, 196.0, 48.0,
            )
        _draw_kanji_glyph(buf_r, buf_g, buf_b, DIM, 2, 42.0, 48.0, 118.0, 4.0, 255.0, 236.0, 168.0)
        _draw_kanji_glyph(buf_r, buf_g, buf_b, DIM, 3, 42.0, 184.0, 118.0, 4.0, 255.0, 236.0, 168.0)
        _draw_kanji_glyph(buf_r, buf_g, buf_b, DIM, 4, 42.0, 320.0, 118.0, 4.0, 255.0, 108.0, 36.0)

    elif v_id == 2:
        # Variant 2: `Vanguard Zero-Type` (`零`) + Sleek Wedge Supercar Profile & Plasma Tailbar
        _draw_neon_ring(buf_r, buf_g, buf_b, DIM, 312.0, 188.0, 88.0, 3.0, 18.0, 255.0, 48.0, 128.0, filled=True)
        _draw_kanji_glyph(buf_r, buf_g, buf_b, DIM, 5, 48.0, 52.0, 138.0, 4.5, 42.0, 246.0, 255.0)
        # Sculpted wedge supercar silhouette (`y ~ 260..340`)
        var car: List[List[Float32]] = [
            [76.0, 332.0, 148.0, 294.0], [148.0, 294.0, 248.0, 258.0],
            [248.0, 258.0, 368.0, 262.0], [368.0, 262.0, 442.0, 304.0],
            [442.0, 304.0, 448.0, 244.0], [422.0, 244.0, 458.0, 244.0],
            [72.0, 336.0, 448.0, 336.0],
        ]
        for i in range(len(car)):
            ref s = car[i]
            _draw_neon_segment(buf_r, buf_g, buf_b, DIM, s[0], s[1], s[2], s[3], 4.0, 15.0, 48.0, 238.0, 255.0)
        _draw_neon_ring(buf_r, buf_g, buf_b, DIM, 156.0, 336.0, 34.0, 4.0, 12.0, 255.0, 42.0, 118.0)
        _draw_neon_ring(buf_r, buf_g, buf_b, DIM, 368.0, 336.0, 34.0, 4.0, 12.0, 255.0, 42.0, 118.0)

    elif v_id == 3:
        # Variant 3: Samurai Oni Mask (`電 重`) + Horns & Curved Ivory Tusks
        _draw_kanji_glyph(buf_r, buf_g, buf_b, DIM, 0, 42.0, 62.0, 122.0, 4.0, 255.0, 52.0, 72.0)
        _draw_kanji_glyph(buf_r, buf_g, buf_b, DIM, 4, 42.0, 218.0, 122.0, 4.0, 255.0, 212.0, 64.0)
        var oni: List[List[Float32]] = [
            [236.0, 172.0, 204.0, 78.0], [404.0, 172.0, 436.0, 78.0],
            [228.0, 178.0, 412.0, 178.0],
            [228.0, 178.0, 244.0, 346.0], [412.0, 178.0, 396.0, 346.0],
            [244.0, 346.0, 320.0, 412.0], [396.0, 346.0, 320.0, 412.0],
            [252.0, 342.0, 224.0, 276.0], [388.0, 342.0, 416.0, 276.0],
        ]
        for i in range(len(oni)):
            ref s = oni[i]
            _draw_neon_segment(buf_r, buf_g, buf_b, DIM, s[0], s[1], s[2], s[3], 4.2, 16.0, 255.0, 44.0, 62.0)
        _draw_neon_ring(buf_r, buf_g, buf_b, DIM, 278.0, 244.0, 16.0, 3.5, 12.0, 38.0, 248.0, 255.0)
        _draw_neon_ring(buf_r, buf_g, buf_b, DIM, 362.0, 244.0, 16.0, 3.5, 12.0, 38.0, 248.0, 255.0)

    elif v_id == 4:
        # Variant 4: Cyber-Optics Bionic Iris (`光 脳`) + Radial HUD Aperture
        _draw_kanji_glyph(buf_r, buf_g, buf_b, DIM, 6, 44.0, 64.0, 124.0, 4.0, 42.0, 252.0, 228.0)
        _draw_kanji_glyph(buf_r, buf_g, buf_b, DIM, 1, 44.0, 220.0, 124.0, 4.0, 42.0, 252.0, 228.0)
        _draw_neon_ring(buf_r, buf_g, buf_b, DIM, 332.0, 256.0, 118.0, 3.5, 16.0, 36.0, 238.0, 255.0)
        _draw_neon_ring(buf_r, buf_g, buf_b, DIM, 332.0, 256.0, 74.0, 3.5, 14.0, 255.0, 198.0, 48.0)
        _draw_neon_ring(buf_r, buf_g, buf_b, DIM, 332.0, 256.0, 28.0, 4.0, 14.0, 255.0, 56.0, 138.0, filled=True)
        for k in range(16):
            var a = (Float32(k) / Float32(16.0)) * _TAU_ARCH
            _draw_neon_segment(
                buf_r, buf_g, buf_b, DIM,
                Float32(332.0) + cos(a) * Float32(42.0), Float32(256.0) + sin(a) * Float32(42.0),
                Float32(332.0) + cos(a) * Float32(104.0), Float32(256.0) + sin(a) * Float32(104.0),
                2.2, 7.0, 48.0, 246.0, 232.0,
            )

    elif v_id == 5:
        # Variant 5: Synth-Cola 2099 (`サ`) + Neon Can Silhouette & Rising Fizz Bubbles
        _draw_kanji_glyph(buf_r, buf_g, buf_b, DIM, 7, 48.0, 68.0, 136.0, 5.0, 255.0, 244.0, 64.0)
        var can_s: List[List[Float32]] = [
            [268.0, 104.0, 396.0, 104.0], [252.0, 136.0, 412.0, 136.0],
            [252.0, 136.0, 252.0, 398.0], [412.0, 136.0, 412.0, 398.0],
            [252.0, 398.0, 412.0, 398.0], [268.0, 104.0, 252.0, 136.0], [396.0, 104.0, 412.0, 136.0],
            [256.0, 248.0, 408.0, 212.0], [256.0, 286.0, 408.0, 250.0],
        ]
        for i in range(len(can_s)):
            ref s = can_s[i]
            _draw_neon_segment(buf_r, buf_g, buf_b, DIM, s[0], s[1], s[2], s[3], 4.0, 14.0, 42.0, 252.0, 255.0)
        _draw_neon_ring(buf_r, buf_g, buf_b, DIM, 296.0, 72.0, 12.0, 2.5, 9.0, 255.0, 82.0, 212.0)
        _draw_neon_ring(buf_r, buf_g, buf_b, DIM, 348.0, 58.0, 18.0, 2.5, 9.0, 48.0, 252.0, 255.0)
        _draw_neon_ring(buf_r, buf_g, buf_b, DIM, 384.0, 82.0, 9.0, 2.5, 8.0, 255.0, 236.0, 68.0)

    elif v_id == 6:
        # Variant 6: Helix Biotech (`新 光`) + Intertwining Double-Helix DNA Strand
        _draw_kanji_glyph(buf_r, buf_g, buf_b, DIM, 2, 44.0, 66.0, 124.0, 4.0, 72.0, 255.0, 164.0)
        _draw_kanji_glyph(buf_r, buf_g, buf_b, DIM, 6, 44.0, 224.0, 124.0, 4.0, 48.0, 238.0, 255.0)
        for st_i in range(18):
            var t0 = Float32(st_i) / Float32(18.0)
            var t1 = Float32(st_i + 1) / Float32(18.0)
            var y0_h = Float32(58.0) + t0 * Float32(396.0)
            var y1_h = Float32(58.0) + t1 * Float32(396.0)
            var x0_a = Float32(332.0) + sin(t0 * _TAU_ARCH * Float32(1.5)) * Float32(78.0)
            var x1_a = Float32(332.0) + sin(t1 * _TAU_ARCH * Float32(1.5)) * Float32(78.0)
            var x0_b = Float32(332.0) - sin(t0 * _TAU_ARCH * Float32(1.5)) * Float32(78.0)
            var x1_b = Float32(332.0) - sin(t1 * _TAU_ARCH * Float32(1.5)) * Float32(78.0)
            _draw_neon_segment(buf_r, buf_g, buf_b, DIM, x0_a, y0_h, x1_a, y1_h, 3.8, 12.0, 56.0, 255.0, 156.0)
            _draw_neon_segment(buf_r, buf_g, buf_b, DIM, x0_b, y0_h, x1_b, y1_h, 3.8, 12.0, 42.0, 224.0, 255.0)
            if (st_i & 1) == 0:
                _draw_neon_segment(buf_r, buf_g, buf_b, DIM, x0_a, y0_h, x0_b, y0_h, 2.4, 8.0, 212.0, 255.0, 96.0)

    else:
        # Variant 7: Dragon Express (`東 電`) + S-Curve Cyber-Dragon Crest & Gold Seal
        _draw_kanji_glyph(buf_r, buf_g, buf_b, DIM, 3, 44.0, 64.0, 126.0, 4.2, 255.0, 206.0, 54.0)
        _draw_kanji_glyph(buf_r, buf_g, buf_b, DIM, 0, 44.0, 222.0, 126.0, 4.2, 255.0, 64.0, 196.0)
        _draw_neon_ring(buf_r, buf_g, buf_b, DIM, 336.0, 248.0, 112.0, 3.5, 16.0, 255.0, 196.0, 48.0)
        for seg_i in range(14):
            var t0 = Float32(seg_i) / Float32(14.0)
            var t1 = Float32(seg_i + 1) / Float32(14.0)
            var dy0 = Float32(112.0) + t0 * Float32(276.0)
            var dy1 = Float32(112.0) + t1 * Float32(276.0)
            var dx0 = Float32(336.0) + sin(t0 * _TAU_ARCH * Float32(1.25)) * Float32(68.0)
            var dx1 = Float32(336.0) + sin(t1 * _TAU_ARCH * Float32(1.25)) * Float32(68.0)
            _draw_neon_segment(buf_r, buf_g, buf_b, DIM, dx0, dy0, dx1, dy1, 5.2, 16.0, 255.0, 72.0, 188.0)

    # Bottom telemetry / ticker bar (`y = 442..468`) across all 8 billboards
    for b_k in range(8):
        var bx0 = Float32(44 + b_k * 54)
        _draw_neon_segment(buf_r, buf_g, buf_b, DIM, bx0, 456.0, bx0 + Float32(36.0), 456.0, 3.2, 8.0, fr_r, fr_g, fr_b)

    # 4. Pack `0xFFRRGGBB` with subtle physical `4x4` Jumbotron RGB LED pitch micro-matrix!
    var out = List[UInt32](length=DIM * DIM, fill=UInt32(0xFF000000))
    for y in range(DIM):
        var row_gap = (y & 3) == 3
        for x in range(DIM):
            var col_gap = (x & 3) == 3
            var diode_mul = Float32(0.84) if (row_gap or col_gap) else Float32(1.04)
            var idx = y * DIM + x
            out[idx] = _pack_argb(
                255,
                Int(buf_r[idx] * diode_mul),
                Int(buf_g[idx] * diode_mul),
                Int(buf_b[idx] * diode_mul),
            )
    return out^


# ============================================================================
# 3. Modular 3-Zone Skyscraper & Emissive-Split Street Kits (`NativeMesh`)
# ============================================================================

def build_colonnade_arcade_bay(
    bay_width: Float32 = 6.0,
    height: Float32 = 4.8,
    depth: Float32 = 2.4,
) -> NativeMesh:
    """Builds a ground-floor recessed skyscraper colonnade arcade bay."""
    var hw = max(Float32(1.5), bay_width * Float32(0.5))
    var h = max(Float32(2.8), height)
    var hd = max(Float32(0.6), depth * Float32(0.5))

    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.46, metallic=0.12)
    var pier_x = hw - Float32(0.38)
    for side in range(-1, 2, 2):
        var px = Float32(side) * pier_x
        mb.add_beveled_box(Vec3(px, 0.22, hd - Float32(0.36)), Vec3(0.42, 0.22, 0.42), bevel=0.04, color=0xFF868E96)
        mb.add_beveled_box(Vec3(px, h * Float32(0.48), hd - Float32(0.36)), Vec3(0.34, h * Float32(0.46), 0.34), bevel=0.05, color=0xFFCED4DA)

    mb.add_beveled_box(Vec3(0.0, h - Float32(0.36), 0.0), Vec3(hw, 0.36, hd), bevel=0.05, color=0xFFADB5BD)
    mb.add_beveled_box(Vec3(0.0, h - Float32(0.76), hd - Float32(0.28)), Vec3(hw, 0.14, 0.30), bevel=0.03, color=0xFF868E96)

    var sf_z = -hd + Float32(0.18)
    mb.add_beveled_box(Vec3(-hw * Float32(0.52), 0.42, sf_z), Vec3(hw * Float32(0.32), 0.42, 0.14), bevel=0.03, color=0xFF495057)
    mb.add_beveled_box(Vec3( hw * Float32(0.52), 0.42, sf_z), Vec3(hw * Float32(0.32), 0.42, 0.14), bevel=0.03, color=0xFF495057)
    mb.add_beveled_box(Vec3(0.0, 2.55, sf_z), Vec3(hw * Float32(0.90), 0.10, 0.16), bevel=0.02, color=0xFF343A40)

    var out = mb^.finish()
    _finalize_arch_ao(out, h, soffit_y=h - Float32(0.72))
    return out^


def build_facade_mullion_spandrel_grid(
    width: Float32 = 12.0,
    height: Float32 = 12.0,
    cols: Int = 4,
    rows: Int = 3,
) -> NativeMesh:
    """Builds a 3D projecting skyscraper curtain-wall mullion fin & floor-slab spandrel grid."""
    var hw = max(Float32(2.0), width * Float32(0.5))
    var n_cols = max(1, cols)
    var n_rows = max(1, rows)
    var h = max(Float32(3.0), height)

    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.32, metallic=0.78)
    mb.mesh.clearcoat = 0.45
    mb.mesh.clearcoat_roughness = 0.08

    for c in range(n_cols + 1):
        var x = -hw + (Float32(c) / Float32(n_cols)) * (hw * Float32(2.0))
        var fin_w = Float32(0.11) if (c == 0 or c == n_cols) else Float32(0.065)
        var fin_d = Float32(0.22) if (c == 0 or c == n_cols) else Float32(0.15)
        mb.add_beveled_box(Vec3(x, h * Float32(0.5), fin_d), Vec3(fin_w, h * Float32(0.5), fin_d), bevel=0.018, color=0xFF454F5E)

    for r in range(n_rows + 1):
        var y = (Float32(r) / Float32(n_rows)) * h
        var sp_h = Float32(0.22) if (r == 0 or r == n_rows) else Float32(0.16)
        mb.add_beveled_box(Vec3(0.0, y, 0.18), Vec3(hw + Float32(0.06), sp_h, 0.18), bevel=0.025, color=0xFF343C48)

    var out = mb^.finish()
    _finalize_arch_ao(out, h)
    return out^


def build_rooftop_hvac_chiller() -> NativeMesh:
    """Builds an industrial rooftop twin-fan HVAC chiller unit with skid rails, louver cabinet & rotors."""
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.42, metallic=0.58)
    mb.add_beveled_box(Vec3(0.0, 0.08, -0.78), Vec3(1.68, 0.08, 0.10), bevel=0.015, color=0xFF495057)
    mb.add_beveled_box(Vec3(0.0, 0.08,  0.78), Vec3(1.68, 0.08, 0.10), bevel=0.015, color=0xFF495057)
    mb.add_beveled_box(Vec3(0.0, 0.84, 0.0), Vec3(1.60, 0.68, 0.82), bevel=0.05, color=0xFFBCC4CC)
    mb.add_beveled_box(Vec3(1.70, 0.90, 0.0), Vec3(0.12, 0.44, 0.48), bevel=0.025, color=0xFF868E96)

    for side in range(-1, 2, 2):
        var fx = Float32(side) * Float32(0.76)
        mb.add_cylinder(Vec3(fx, 1.50, 0.0), Vec3(fx, 1.74, 0.0), 0.56, 0.52, segments=14, color=0xFF9AA3AE)
        mb.add_cylinder(Vec3(fx, 1.54, 0.0), Vec3(fx, 1.58, 0.0), 0.46, 0.46, segments=12, color=0xFF212529)
        for b_i in range(5):
            var ang = (Float32(b_i) / Float32(5.0)) * _TAU_ARCH
            var q_b = Quat(0.0, sin(ang * Float32(0.5)), 0.0, cos(ang * Float32(0.5)))
            mb.add_beveled_box(
                Vec3(fx + cos(ang) * Float32(0.22), 1.62, sin(ang) * Float32(0.22)),
                Vec3(0.18, 0.015, 0.055),
                bevel=0.005,
                rot=q_b,
                color=0xFF343A40,
            )

    mb.add_cylinder(Vec3(-1.62, 0.55, 0.48), Vec3(-1.92, 0.55, 0.48), 0.075, 0.075, segments=8, color=0xFF343A40)
    mb.add_cylinder(Vec3(-1.90, 0.12, 0.48), Vec3(-1.90, 0.58, 0.48), 0.075, 0.075, segments=8, color=0xFF343A40)

    var out = mb^.finish()
    _finalize_arch_ao(out, Float32(1.80))
    return out^


def build_rooftop_vtol_helipad_parts(radius: Float32 = 5.5) -> Tuple[NativeMesh, NativeMesh]:
    """Returns `(base_deck_mesh, hdr_emissive_mesh)` for the octagonal rooftop VTOL / Spinner landing pad:
    - `base_deck_mesh`: chamfered octagonal concrete/steel deck, yellow hazard threshold stripes, pedestals & ramp.
    - `hdr_emissive_mesh`: illuminated white/cyan `'H'` landing insignia + `8` glowing amber perimeter beacons!
    """
    var r = max(Float32(2.8), radius)
    var oct = List[Vec2](capacity=8)
    for i in range(8):
        var a = (Float32(i) + Float32(0.5)) / Float32(8.0) * _TAU_ARCH
        oct.append(Vec2(cos(a) * r, sin(a) * r))

    var deck = extrude_profile_beveled(
        oct,
        height=0.42,
        corner_fillet=0.22,
        cap_chamfer=0.06,
        fillet_steps=2,
        base_color=0xFFFFFFFF,
        roughness=0.56,
        metallic=0.16,
    )

    var mb_base = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.52, metallic=0.20)
    var mb_emis = MeshBuilder(
        base_color=0xFFFFFFFF, roughness=0.08, metallic=0.0,
        emissive_color=Vec3(1.0, 1.0, 1.0), emissive_strength=6.5,
    )

    var dvc = deck.vertex_count()
    for i in range(dvc):
        _ = mb_base.add_vertex(
            Vec3(deck.positions[i * 3], deck.positions[i * 3 + 1] + Float32(0.32), deck.positions[i * 3 + 2]),
            Vec3(deck.normals[i * 3], deck.normals[i * 3 + 1], deck.normals[i * 3 + 2]),
            deck.texcoords[i * 2],
            deck.texcoords[i * 2 + 1],
            color=0xFF424A56,
            ao=deck.ao[i],
        )
    for idx in deck.indices:
        mb_base.mesh.indices.append(idx)

    # 8 pedestals + yellow safety perimeter threshold bars + 8 HDR amber approach beacons
    for i in range(8):
        var a = (Float32(i) / Float32(8.0)) * _TAU_ARCH
        var px = cos(a) * (r * Float32(0.78))
        var pz = sin(a) * (r * Float32(0.78))
        mb_base.add_beveled_box(Vec3(px, 0.18, pz), Vec3(0.28, 0.18, 0.28), bevel=0.03, color=0xFF2E343E)

        var q_seg = Quat(0.0, sin(-a * Float32(0.5)), 0.0, cos(-a * Float32(0.5)))
        mb_base.add_beveled_box(
            Vec3(cos(a) * (r * Float32(0.84)), 0.748, sin(a) * (r * Float32(0.84))),
            Vec3(0.11, 0.012, r * Float32(0.26)),
            bevel=0.004,
            rot=q_seg,
            color=0xFFFFCA28,
        )

        var bx = cos(a) * (r * Float32(0.92))
        var bz = sin(a) * (r * Float32(0.92))
        mb_base.add_cylinder(Vec3(bx, 0.72, bz), Vec3(bx, 0.78, bz), 0.12, 0.10, segments=8, color=0xFF2B3038)
        mb_emis.add_cylinder(Vec3(bx, 0.78, bz), Vec3(bx, 0.88, bz), 0.095, 0.075, segments=8, color=0xFFFFB703)

    # HDR Illuminated `'H'` VTOL landing insignia
    var h_span = r * Float32(0.36)
    mb_emis.add_beveled_box(Vec3(-h_span * Float32(0.65), 0.755, 0.0), Vec3(0.18, 0.018, h_span), bevel=0.008, color=0xFFE8FAFF)
    mb_emis.add_beveled_box(Vec3( h_span * Float32(0.65), 0.755, 0.0), Vec3(0.18, 0.018, h_span), bevel=0.008, color=0xFFE8FAFF)
    mb_emis.add_beveled_box(Vec3(0.0, 0.755, 0.0), Vec3(h_span * Float32(0.65), 0.018, 0.18), bevel=0.008, color=0xFFE8FAFF)

    var q_ramp = Quat(sin(Float32(0.14)), 0.0, 0.0, cos(Float32(0.14)))
    mb_base.add_beveled_box(Vec3(0.0, 0.36, -r - Float32(0.85)), Vec3(0.95, 0.08, 1.15), bevel=0.025, rot=q_ramp, color=0xFF525C68)

    var m_b = mb_base^.finish()
    var m_e = mb_emis^.finish()
    _finalize_arch_ao(m_b, Float32(0.95))
    m_e.ensure_bounds()
    return (m_b^, m_e^)


def build_rooftop_vtol_helipad(radius: Float32 = 5.5) -> NativeMesh:
    """Unified vertex-colored rooftop VTOL helipad mesh."""
    var p = build_rooftop_vtol_helipad_parts(radius)
    return _merge_arch_pair(p[0], p[1])


def build_street_traffic_cantilever_parts(
    span: Float32 = 7.5,
    height: Float32 = 5.8,
) -> Tuple[NativeMesh, NativeMesh]:
    """Returns `(mast_and_housings_mesh, hdr_signal_lenses_mesh)` for the urban traffic cantilever."""
    var h = max(Float32(4.2), height)
    var sp = max(Float32(3.5), span)
    var mb_base = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.34, metallic=0.78)
    var mb_emis = MeshBuilder(
        base_color=0xFFFFFFFF, roughness=0.06, metallic=0.0,
        emissive_color=Vec3(1.0, 1.0, 1.0), emissive_strength=8.0,
    )

    mb_base.add_cylinder(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.24, 0.0), 0.34, 0.28, segments=8, color=0xFF343A42)
    mb_base.add_cylinder(Vec3(0.0, 0.22, 0.0), Vec3(0.0, h, 0.0), 0.18, 0.12, segments=10, color=0xFF4A5462)

    var boom_y = h - Float32(0.32)
    mb_base.add_beveled_box(Vec3(0.18, boom_y, 0.0), Vec3(0.24, 0.24, 0.20), bevel=0.03, color=0xFF343A42)
    mb_base.add_cylinder(Vec3(0.15, boom_y, 0.0), Vec3(sp, boom_y + Float32(0.12), 0.0), 0.115, 0.065, segments=8, color=0xFF4A5462)

    # Illuminated street-name blade sign (`x = 1.35m`)
    mb_base.add_beveled_box(Vec3(1.35, boom_y + Float32(0.24), -0.01), Vec3(0.80, 0.15, 0.025), bevel=0.008, color=0xFF2B323C)
    mb_emis.add_beveled_box(Vec3(1.35, boom_y + Float32(0.24),  0.02), Vec3(0.76, 0.12, 0.015), bevel=0.005, color=0xFF2496FF)

    for s_i in range(2):
        var sx = sp * (Float32(0.46) + Float32(s_i) * Float32(0.40))
        var sy = boom_y + Float32(0.06) * Float32(s_i)
        mb_base.add_beveled_box(Vec3(sx, sy - Float32(0.12), 0.06), Vec3(0.18, 0.44, 0.12), bevel=0.025, color=0xFF22272E)
        for l_i in range(3):
            var ly = sy + Float32(0.16) - Float32(l_i) * Float32(0.26)
            var l_col: UInt32 = UInt32(0xFFFF3028) if l_i == 0 else (UInt32(0xFFFFC410) if l_i == 1 else UInt32(0xFF20F560))
            mb_emis.add_cylinder(Vec3(sx, ly, 0.16), Vec3(sx, ly, 0.24), 0.095, 0.095, segments=8, color=l_col)

    mb_base.add_beveled_box(Vec3(0.24, 2.35, 0.12), Vec3(0.14, 0.22, 0.12), bevel=0.02, color=0xFF262B34)
    mb_emis.add_beveled_box(Vec3(0.24, 2.35, 0.24), Vec3(0.10, 0.14, 0.015), bevel=0.004, color=0xFF38F2FF)

    var m_b = mb_base^.finish()
    var m_e = mb_emis^.finish()
    _finalize_arch_ao(m_b, h)
    m_e.ensure_bounds()
    return (m_b^, m_e^)


def build_street_traffic_cantilever(
    span: Float32 = 7.5,
    height: Float32 = 5.8,
) -> NativeMesh:
    """Unified vertex-colored traffic cantilever mesh."""
    var p = build_street_traffic_cantilever_parts(span, height)
    return _merge_arch_pair(p[0], p[1])


def build_street_bollard_and_grate_parts() -> Tuple[NativeMesh, NativeMesh]:
    """Returns `(bollard_and_grate_mesh, hdr_cyan_collar_mesh)`."""
    var mb_base = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.30, metallic=0.82)
    var mb_emis = MeshBuilder(
        base_color=0xFFFFFFFF, roughness=0.06, metallic=0.0,
        emissive_color=Vec3(1.0, 1.0, 1.0), emissive_strength=7.5,
    )

    mb_base.add_cylinder(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.08, 0.0), 0.18, 0.15, segments=12, color=0xFF343A42)
    mb_base.add_cylinder(Vec3(0.0, 0.08, 0.0), Vec3(0.0, 0.78, 0.0), 0.125, 0.120, segments=12, color=0xFF56606E)
    mb_emis.add_cylinder(Vec3(0.0, 0.78, 0.0), Vec3(0.0, 0.84, 0.0), 0.108, 0.108, segments=12, color=0xFF28ECFF)
    mb_base.add_cylinder(Vec3(0.0, 0.84, 0.0), Vec3(0.0, 0.96, 0.0), 0.120, 0.112, segments=12, color=0xFF748090)
    mb_base.add_ellipsoid(Vec3(0.0, 0.96, 0.0), Vec3(0.112, 0.04, 0.112), rings=4, segments=12, color=0xFF94A0B0)

    var gx = Float32(0.58)
    mb_base.add_beveled_box(Vec3(gx, 0.016, 0.0), Vec3(0.32, 0.016, 0.44), bevel=0.005, color=0xFF1C2026)
    mb_base.add_beveled_box(Vec3(gx - Float32(0.29), 0.026, 0.0), Vec3(0.03, 0.012, 0.44), bevel=0.004, color=0xFF3D444E)
    mb_base.add_beveled_box(Vec3(gx + Float32(0.29), 0.026, 0.0), Vec3(0.03, 0.012, 0.44), bevel=0.004, color=0xFF3D444E)
    for bar_i in range(-3, 4):
        var bz = Float32(bar_i) * Float32(0.11)
        mb_base.add_beveled_box(Vec3(gx, 0.026, bz), Vec3(0.26, 0.010, 0.024), bevel=0.003, color=0xFF48505C)

    var m_b = mb_base^.finish()
    var m_e = mb_emis^.finish()
    _finalize_arch_ao(m_b, Float32(1.0))
    m_e.ensure_bounds()
    return (m_b^, m_e^)


def build_street_bollard_and_grate() -> NativeMesh:
    """Unified vertex-colored street bollard & storm-drain grate mesh."""
    var p = build_street_bollard_and_grate_parts()
    return _merge_arch_pair(p[0], p[1])


def build_transit_shelter_parts(
    length: Float32 = 4.2,
    height: Float32 = 2.6,
    depth: Float32 = 1.6,
) -> Tuple[NativeMesh, NativeMesh]:
    """Returns `(shelter_structure_mesh, hdr_lightbox_and_led_mesh)`."""
    var hl = max(Float32(1.5), length * Float32(0.5))
    var h = max(Float32(2.2), height)
    var hd = max(Float32(0.65), depth * Float32(0.5))

    var mb_base = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.32, metallic=0.72)
    mb_base.mesh.clearcoat = 0.50
    mb_base.mesh.clearcoat_roughness = 0.08
    var mb_emis = MeshBuilder(
        base_color=0xFFFFFFFF, roughness=0.08, metallic=0.0,
        emissive_color=Vec3(1.0, 1.0, 1.0), emissive_strength=5.5,
    )

    var rz = -hd + Float32(0.10)
    for i in range(-1, 2):
        var ux = Float32(i) * (hl - Float32(0.22))
        mb_base.add_beveled_box(Vec3(ux, h * Float32(0.48), rz), Vec3(0.065, h * Float32(0.48), 0.075), bevel=0.015, color=0xFF38424E)
        var q_arm = Quat(sin(Float32(-0.04)), 0.0, 0.0, cos(Float32(-0.04)))
        mb_base.add_beveled_box(Vec3(ux, h - Float32(0.10), 0.0), Vec3(0.055, 0.055, hd * Float32(0.95)), bevel=0.012, rot=q_arm, color=0xFF485464)

    var q_roof = Quat(sin(Float32(-0.035)), 0.0, 0.0, cos(Float32(-0.035)))
    mb_base.add_beveled_box(Vec3(0.0, h, 0.04), Vec3(hl + Float32(0.12), 0.055, hd + Float32(0.08)), bevel=0.02, rot=q_roof, color=0xFF525E70)
    mb_emis.add_beveled_box(Vec3(0.0, h - Float32(0.07), -0.10), Vec3(hl * Float32(0.82), 0.018, 0.06), bevel=0.005, color=0xFFFFF0C2)

    mb_base.add_beveled_box(Vec3(hl - Float32(0.12), h * Float32(0.46), 0.0), Vec3(0.07, h * Float32(0.44), hd * Float32(0.78)), bevel=0.02, color=0xFF2C3440)
    mb_emis.add_beveled_box(Vec3(hl - Float32(0.04), h * Float32(0.46), 0.0), Vec3(0.015, h * Float32(0.36), hd * Float32(0.64)), bevel=0.005, color=0xFF5CE1FF)

    mb_base.add_beveled_box(Vec3(-hl * Float32(0.22), 0.46, rz + Float32(0.32)), Vec3(hl * Float32(0.58), 0.038, 0.22), bevel=0.012, color=0xFFC89058)
    mb_base.add_beveled_box(Vec3(-hl * Float32(0.58), 0.24, rz + Float32(0.24)), Vec3(0.045, 0.22, 0.16), bevel=0.01, color=0xFF38424E)
    mb_base.add_beveled_box(Vec3( hl * Float32(0.14), 0.24, rz + Float32(0.24)), Vec3(0.045, 0.22, 0.16), bevel=0.01, color=0xFF38424E)

    var m_b = mb_base^.finish()
    var m_e = mb_emis^.finish()
    _finalize_arch_ao(m_b, h, soffit_y=h - Float32(0.08))
    m_e.ensure_bounds()
    return (m_b^, m_e^)


def build_transit_shelter(
    length: Float32 = 4.2,
    height: Float32 = 2.6,
    depth: Float32 = 1.6,
) -> NativeMesh:
    """Unified vertex-colored transit shelter mesh."""
    var p = build_transit_shelter_parts(length, height, depth)
    return _merge_arch_pair(p[0], p[1])
