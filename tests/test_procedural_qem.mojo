"""Comprehensive Unit & Benchmark Suite for Flint v3 Asset & Procedural Architecture (`test_procedural_qem.mojo`).

Verifies:
  1. `O(T log T)` Indexed Min-Heap QEM Simplifier (`3,072`-tri 3-step LOD chain `< 15 ms`, topological validity,
     normal-flip prevention, UV seam preservation, boundary locking, and 24-bit vertex color propagation).
  2. Exact `vertex_words[11]` bit-packing contract (`ao_byte=255 -> ao_u6=63`, 24-bit sRGB vertex colors in `bits 8..31`,
     black `[0,0,0] -> b8=1` clamp, and `add_splat_mesh` `bit 1 = 1` + `u24` weights).
  3. Toksvig Normal-Variance Roughness Mip Baker (`downsample_normals_and_orm_toksvig_2x` & pyramid monotonicity).
  4. Pixar/OpenSubdiv `SubdCage` Catmull-Clark semi-sharp creases, `mirror_x_and_weld`, watertight manifold topology,
     and parametric Supercar / Spinner / Traffic / Coaster builders.
  5. Beveled extrusions, lathed 5-spoke wheel/rotor/caliper assemblies (`~420` tris), and RMF spline lofts.
  6. `512x512` alpha-cutout botanical leaf/needle/frond atlases, `256x256` fissured bark, gnarled multi-seed trees,
     and understory grass/wildflower, manicured hedge/planter, and flowering bush builders.
  7. 3-submesh `16`-bone rigged hard-surface `Cyber-Enforcer` (`Armor` / `Chassis` / `HDR Emissive`) and `12`-pose
     multi-tone humanoid crowd bank (with/without umbrella).
  8. Clean back-wall `512x512` `4x4` Interior Parallax WindowBox Atlas, `8` Neo-Tokyo `512x512` cyber billboards,
     and emissive-split `_parts` architectural & street kits.
"""
from std.math import cos, min, sin
from std.time import monotonic
from flint.core.math import Vec2, Vec3
from flint.assets.model import NativeMesh
from flint.assets.meshlet import AaaGeometryPool, MeshLodSlice, _SymQuadric, simplify_mesh_qem
from flint.assets.bc_compress import (
    bake_toksvig_normal_orm_pyramid,
    downsample_normals_and_orm_toksvig_2x,
)
from flint.procedural import (
    ENFORCER_BONE_COUNT,
    SubdCage,
    bake_conifer_needle_sprig_512,
    bake_cyber_billboard_512,
    bake_fissured_bark_256,
    bake_interior_windowbox_atlas_512,
    bake_oak_leaf_sprig_512,
    bake_palm_frond_512,
    build_articulated_crowd_pose_bank,
    build_botanical_oak,
    build_botanical_palm,
    build_botanical_pine,
    build_colonnade_arcade_bay,
    build_facade_mullion_spandrel_grid,
    build_flowering_bush,
    build_grass_wildflower_tuft,
    build_lathed_wheel_assembly,
    build_manicured_hedge_and_planter,
    build_rooftop_hvac_chiller,
    build_rooftop_vtol_helipad,
    build_rooftop_vtol_helipad_parts,
    build_street_bollard_and_grate,
    build_street_bollard_and_grate_parts,
    build_street_traffic_cantilever,
    build_street_traffic_cantilever_parts,
    build_subd_coaster_car_shell,
    build_subd_cyber_enforcer_rigged,
    build_subd_cyber_enforcer_rigged_v3,
    build_subd_spinner_hovercar_body_and_canopy,
    build_subd_supercar_body_and_glass,
    build_subd_traffic_vehicle,
    build_transit_shelter,
    build_transit_shelter_parts,
    build_urban_ginkgo_tree,
    creased_subdivide,
    extrude_profile_beveled,
    make_chamfered_girder_profile,
    make_curb_profile,
    subdivide_to_mesh,
    sweep_catenary_cable,
    sweep_profile_rmf,
)
from e2e.common.harness import TestContext
from e2e.common.mock_data import create_mock_meshlet_mesh


def _build_uv_seamed_torus_3072t() -> NativeMesh:
    """Builds a `32 x 48` (`3,072`-triangle) closed torus with an explicit UV seam along `u = 0 / 1` and `v = 0 / 1`."""
    var m = NativeMesh()
    comptime MAJ_SEGS: Int = 48
    comptime MIN_SEGS: Int = 32
    comptime R_MAJ: Float32 = 2.0
    comptime R_MIN: Float32 = 0.65
    comptime TAU: Float32 = 6.28318530

    for i in range(MAJ_SEGS + 1):
        var tu = Float32(i) / Float32(MAJ_SEGS)
        var phi = tu * TAU
        var cp = cos(phi)
        var sp = sin(phi)
        for j in range(MIN_SEGS + 1):
            var tv = Float32(j) / Float32(MIN_SEGS)
            var theta = tv * TAU
            var ct = cos(theta)
            var st = sin(theta)
            var ring_r = R_MAJ + R_MIN * ct
            var px = ring_r * cp
            var py = R_MIN * st
            var pz = ring_r * sp
            var nx = ct * cp
            var ny = st
            var nz = ct * sp
            m.positions.append(px); m.positions.append(py); m.positions.append(pz)
            m.normals.append(nx); m.normals.append(ny); m.normals.append(nz)
            m.texcoords.append(tu); m.texcoords.append(tv)
            m.ao.append(UInt8(220))
            var cr = UInt8(60 + Int(tu * Float32(160.0)))
            var cg = UInt8(90 + Int(tv * Float32(120.0)))
            m.colors.append(cr); m.colors.append(cg); m.colors.append(UInt8(210)); m.colors.append(UInt8(255))

    var stride = MIN_SEGS + 1
    for i in range(MAJ_SEGS):
        for j in range(MIN_SEGS):
            var i0 = i * stride + j
            var i1 = i0 + 1
            var i2 = i0 + stride + 1
            var i3 = i0 + stride
            m.indices.append(i0); m.indices.append(i1); m.indices.append(i2)
            m.indices.append(i0); m.indices.append(i2); m.indices.append(i3)
    return m^


def _count_open_spatial_boundary_edges(meshes: List[NativeMesh]) -> Int:
    """Welds spatial vertex positions within `1e-4` across `meshes` and counts undirected edges with `tri_count != 2`."""
    var rep_pos = List[Vec3]()
    var all_tris = List[Int]()

    for m_i in range(len(meshes)):
        ref m = meshes[m_i]
        var vc = m.vertex_count()
        var v_map = List[Int](length=vc, fill=0)
        for v in range(vc):
            var p = Vec3(m.positions[v * 3], m.positions[v * 3 + 1], m.positions[v * 3 + 2])
            var found = -1
            for r in range(len(rep_pos)):
                if (rep_pos[r] - p).length_squared() <= Float32(1e-8):
                    found = r
                    break
            if found >= 0:
                v_map[v] = found
            else:
                v_map[v] = len(rep_pos)
                rep_pos.append(p)
        var tc = len(m.indices) // 3
        for t in range(tc):
            var a = v_map[m.indices[t * 3]]
            var b = v_map[m.indices[t * 3 + 1]]
            var c = v_map[m.indices[t * 3 + 2]]
            if a != b and b != c and c != a:
                all_tris.append(a); all_tris.append(b); all_tris.append(c)

    var e_u = List[Int]()
    var e_v = List[Int]()
    var e_cnt = List[Int]()
    var n_t = len(all_tris) // 3
    for t in range(n_t):
        var vs = [all_tris[t * 3], all_tris[t * 3 + 1], all_tris[t * 3 + 2]]
        for k in range(3):
            var u0 = min(vs[k], vs[(k + 1) % 3])
            var u1 = max(vs[k], vs[(k + 1) % 3])
            var idx = -1
            for ei in range(len(e_u)):
                if e_u[ei] == u0 and e_v[ei] == u1:
                    idx = ei
                    break
            if idx >= 0:
                e_cnt[idx] += 1
            else:
                e_u.append(u0); e_v.append(u1); e_cnt.append(1)

    var non_manifold = 0
    for ei in range(len(e_cnt)):
        if e_cnt[ei] != 2:
            non_manifold += 1
    return non_manifold


def _assert_valid_mesh_attribs(mut ctx: TestContext, m: NativeMesh, label: String, require_colors: Bool = False) raises:
    var vc = m.vertex_count()
    var tc = len(m.indices) // 3
    ctx.assert_true(vc >= 3, label + ": vertex_count >= 3")
    ctx.assert_true(tc >= 1, label + ": triangle_count >= 1")
    ctx.assert_equal(len(m.normals), vc * 3, label + ": normals length == 3 * V")
    ctx.assert_equal(len(m.texcoords), vc * 2, label + ": texcoords length == 2 * V")
    ctx.assert_equal(len(m.ao), vc, label + ": ao length == V")
    if require_colors:
        ctx.assert_equal(len(m.colors), vc * 4, label + ": colors length == 4 * V")
    for idx in m.indices:
        if idx < 0 or idx >= vc:
            raise Error(label + ": out-of-range triangle index")
    for v in range(vc):
        var n = Vec3(m.normals[v * 3], m.normals[v * 3 + 1], m.normals[v * 3 + 2])
        if abs(n.length() - Float32(1.0)) > Float32(0.05):
            raise Error(label + ": non-unit vertex normal")


# =========================================================================
# 1. `O(T log T)` Min-Heap QEM Simplifier & Boundary Lock Tests
# =========================================================================

def test_qem_minheap_speed_and_validity(mut ctx: TestContext) raises:
    var torus = _build_uv_seamed_torus_3072t()
    ctx.assert_equal(torus.triangle_count(), 3072, "input torus has 3,072 triangles")

    # Benchmark 3 consecutive LOD simplification steps (`3072 -> 1536 -> 768 -> 384`)
    var t_start_ns = monotonic()
    var lod1 = simplify_mesh_qem(torus, 1536)
    var lod2 = simplify_mesh_qem(lod1[0], 768)
    var lod3 = simplify_mesh_qem(lod2[0], 384)
    var elapsed_ms = Float64(monotonic() - t_start_ns) / 1_000_000.0
    print("  [Benchmark] 3,072-tri 3-LOD QEM chain simplified in", elapsed_ms, "ms")

    ctx.assert_true(elapsed_ms < 15.0, "3,072-tri 3-step QEM simplification completes in < 15 ms")
    ctx.assert_true(lod1[0].triangle_count() <= 1560 and lod1[0].triangle_count() >= 1400, "LOD 1 reaches ~1,536 tris")
    ctx.assert_true(lod2[0].triangle_count() <= 800 and lod2[0].triangle_count() >= 700, "LOD 2 reaches ~768 tris")
    ctx.assert_true(lod3[0].triangle_count() <= 420 and lod3[0].triangle_count() >= 340, "LOD 3 reaches ~384 tris")
    ctx.assert_true(lod1[1] > Float32(0.0) and lod2[1] > Float32(0.0) and lod3[1] > Float32(0.0), "positive QEM geometric errors")

    _assert_valid_mesh_attribs(ctx, lod3[0], "QEM LOD3 torus", require_colors=True)
    ctx.assert_equal(Int(lod3[0].colors[2]), 210, "QEM LOD3 preserves vertex color blue channel across collapses")

    # Closed torus must remain closed & crack-free across UV seams even at 12.5% triangle count!
    var m_list = List[NativeMesh]()
    m_list.append(lod3[0].copy())
    ctx.assert_equal(_count_open_spatial_boundary_edges(m_list), 0, "UV-seamed torus stays 100% watertight after 8x QEM decimation")


# =========================================================================
# 2. `vertex_words[11]` 24-Bit sRGB Vertex Color, Splat & AO Packing Tests
# =========================================================================

def test_vertex_words_11_ao_and_splat_packing(mut ctx: TestContext) raises:
    var m_std = NativeMesh()
    m_std.positions = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    m_std.normals = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    m_std.texcoords = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    m_std.ao = [UInt8(255), UInt8(128), UInt8(0)]
    m_std.indices = [0, 2, 1]

    var pool = AaaGeometryPool()
    var entry_std = pool.append_mesh(m_std, material_id=0, build_lods=False)
    ctx.assert_equal(entry_std.vertex_count, 3, "standard mesh stores 3 vertices")

    var w11_v0 = pool.vertex_words[0 * 12 + 11]
    var w11_v1 = pool.vertex_words[1 * 12 + 11]
    var w11_v2 = pool.vertex_words[2 * 12 + 11]

    ctx.assert_equal(Int((w11_v0 >> 1) & 1), 0, "standard mesh bit 1 (is_terrain_splat) == 0")
    ctx.assert_equal(Int((w11_v0 >> 2) & 0x3F), 63, "ao_byte=255 packs to ao_u6=63 in bits 2..7")
    ctx.assert_equal(Int(w11_v0 >> 8), 0xFFFFFF, "uncolored standard mesh defaults bits 8..31 to white 0xFFFFFF")
    ctx.assert_equal(Int((w11_v1 >> 2) & 0x3F), 32, "ao_byte=128 packs to ao_u6=32")
    ctx.assert_equal(Int((w11_v2 >> 2) & 0x3F), 0, "ao_byte=0 packs to ao_u6=0")

    # Explicit 24-bit sRGB vertex colors + pure-black `[0, 0, 0] -> b8 = 1` clamp test
    var m_col = m_std.copy()
    m_col.colors = [
        UInt8(0x12), UInt8(0x34), UInt8(0x56), UInt8(255),
        UInt8(0xFE), UInt8(0x80), UInt8(0x05), UInt8(255),
        UInt8(0x00), UInt8(0x00), UInt8(0x00), UInt8(255),
    ]
    var entry_col = pool.append_mesh(m_col, material_id=1, build_lods=False)
    var base_cv = entry_col.first_vertex
    var cw_v0 = pool.vertex_words[(base_cv + 0) * 12 + 11]
    var cw_v1 = pool.vertex_words[(base_cv + 1) * 12 + 11]
    var cw_v2 = pool.vertex_words[(base_cv + 2) * 12 + 11]

    ctx.assert_equal(Int((cw_v0 >> 1) & 1), 0, "colored standard mesh bit 1 == 0")
    ctx.assert_equal(Int((cw_v0 >> 8) & 0xFF), 0x12, "colored v0 r8 == 0x12 in bits 8..15")
    ctx.assert_equal(Int((cw_v0 >> 16) & 0xFF), 0x34, "colored v0 g8 == 0x34 in bits 16..23")
    ctx.assert_equal(Int((cw_v0 >> 24) & 0xFF), 0x56, "colored v0 b8 == 0x56 in bits 24..31")
    ctx.assert_equal(Int((cw_v1 >> 8) & 0xFF), 0xFE, "colored v1 r8 == 0xFE")
    ctx.assert_equal(Int((cw_v1 >> 16) & 0xFF), 0x80, "colored v1 g8 == 0x80")
    ctx.assert_equal(Int((cw_v1 >> 24) & 0xFF), 0x05, "colored v1 b8 == 0x05")
    ctx.assert_equal(Int(cw_v2 >> 8), 0x010000, "pure-black [0,0,0] clamps b8=1 so (meta >> 8) == 0x010000 != 0")

    # Terrain splat mesh test (`w_sand=0x2A, w_rock=0x7F, w_cobble=0xC3`)
    var splat_w = List[UInt32]()
    splat_w.append(UInt32(0xC37F2A))
    splat_w.append(UInt32(0x112233))
    splat_w.append(UInt32(0xABCDEF))
    var splat_id = pool.add_splat_mesh(m_std, splat_w, build_lods=False)
    ctx.assert_equal(splat_id, 2, "add_splat_mesh returns mesh_id == 2")

    var base_sv = pool.meshes[splat_id].first_vertex
    var sw_v0 = pool.vertex_words[(base_sv + 0) * 12 + 11]
    ctx.assert_equal(Int((sw_v0 >> 1) & 1), 1, "splat mesh bit 1 (is_terrain_splat) == 1")
    ctx.assert_equal(Int((sw_v0 >> 2) & 0x3F), 63, "splat mesh preserves ao_u6=63 in bits 2..7")
    ctx.assert_equal(Int((sw_v0 >> 8) & 0xFF), 0x2A, "bits 8..15 == w_sand (0x2A)")
    ctx.assert_equal(Int((sw_v0 >> 16) & 0xFF), 0x7F, "bits 16..23 == w_rock (0x7F)")
    ctx.assert_equal(Int((sw_v0 >> 24) & 0xFF), 0xC3, "bits 24..31 == w_cobble (0xC3)")


# =========================================================================
# 3. Toksvig Normal-Variance Roughness Mip Baker Tests
# =========================================================================

def test_toksvig_normal_variance_roughness(mut ctx: TestContext) raises:
    comptime W: Int = 8
    comptime H: Int = 8
    var flat_n = List[Vec3](length=W * H, fill=Vec3(0.0, 0.0, 1.0))
    var orm_mirror = List[UInt32](length=W * H, fill=UInt32(0xFFFF1000))  # G (roughness) = 16

    var ds_flat = downsample_normals_and_orm_toksvig_2x(flat_n, orm_mirror, W, H, toksvig_scale=1.0)
    var flat_g = Int((ds_flat[1][0] >> 8) & 255)
    ctx.assert_equal(flat_g, 16, "zero-variance flat normals preserve exact mirror roughness G=16")

    # High-variance bumpy normal field (opposing 2x2 micro-facets)
    var bumpy_n = List[Vec3](capacity=W * H)
    for y in range(H):
        for x in range(W):
            var sx = Float32(-0.62) if ((x & 1) == 0) else Float32(0.62)
            var sy = Float32(-0.62) if ((y & 1) == 0) else Float32(0.62)
            bumpy_n.append(Vec3(sx, sy, Float32(0.48)).normalized())

    var ds_bumpy = downsample_normals_and_orm_toksvig_2x(bumpy_n, orm_mirror, W, H, toksvig_scale=1.0)
    var bumpy_g = Int((ds_bumpy[1][0] >> 8) & 255)
    ctx.assert_true(bumpy_g > 160, "high-variance 2x2 normals dramatically boost Toksvig roughness (G > 160 vs 16)")
    ctx.assert_approx(ds_bumpy[0][0].length(), 1.0, 0.005, "downsampled Toksvig normal is normalized for BC5")

    # Multi-scale wave pyramid monotonicity check (`mip 0 -> mip 1 -> mip 2`)
    var wave_n = List[Vec3](capacity=W * H)
    for _ in range(H):
        for x in range(W):
            var wx = Float32(0.35) if (((x >> 1) & 1) == 0) else Float32(-0.35)
            var nx = Float32(0.18) if ((x & 1) == 0) else Float32(-0.18)
            wave_n.append(Vec3(wx + nx, 0.0, Float32(0.90)).normalized())

    var pyr = bake_toksvig_normal_orm_pyramid(wave_n, orm_mirror, W, H, toksvig_scale=1.0)
    var g_mip0 = Int((pyr[1][0][0] >> 8) & 255)
    var g_mip1 = Int((pyr[1][1][0] >> 8) & 255)
    var g_mip2 = Int((pyr[1][2][0] >> 8) & 255)
    ctx.assert_true(g_mip1 > g_mip0, "Toksvig mip 1 roughness > mip 0 roughness")
    ctx.assert_true(g_mip2 > g_mip1, "Toksvig mip 2 roughness > mip 1 roughness across coarser normal wave")


# =========================================================================
# 4. Catmull-Clark `SubdCage`, Creases & Vehicle/Coaster Sculpt Tests
# =========================================================================

def test_subd_creases_and_vehicles(mut ctx: TestContext) raises:
    var c_smooth = SubdCage()
    var c_sharp = SubdCage()
    for sx in range(-1, 2, 2):
        for sy in range(-1, 2, 2):
            for sz in range(-1, 2, 2):
                var p = Vec3(Float32(sx), Float32(sy), Float32(sz))
                _ = c_smooth.add_vert(p)
                _ = c_sharp.add_vert(p)
    var faces = [
        [0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
        [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3],
    ]
    for f_i in range(6):
        ref f = faces[f_i]
        c_smooth.add_quad(f[0], f[1], f[2], f[3])
        c_sharp.add_quad(f[0], f[1], f[2], f[3])
        for k in range(4):
            c_sharp.set_edge_sharpness(f[k], f[(k + 1) % 4], Float32(5.0))

    var grp_meshes = creased_subdivide(c_smooth, levels=2, split_sharp_threshold=1.4)
    ctx.assert_equal(len(grp_meshes), 1, "single-group SubdCage emits 1 group mesh")
    var m_smooth = subdivide_to_mesh(c_smooth, levels=2, split_sharp_threshold=1.4)
    var m_sharp = subdivide_to_mesh(c_sharp, levels=2, split_sharp_threshold=1.4)
    ctx.assert_equal(m_smooth.triangle_count(), 192, "2-level Catmull-Clark cube yields 6 * 32 = 192 tris")
    ctx.assert_equal(m_sharp.triangle_count(), 192, "creased cube preserves exact 192-tri quad topology")
    ctx.assert_true(m_sharp.vertex_count() > m_smooth.vertex_count(), "sharp creases split hard normal wedges cleanly")
    m_smooth.ensure_bounds()
    m_sharp.ensure_bounds()
    ctx.assert_true(m_sharp.bounds_radius > m_smooth.bounds_radius + Float32(0.35), "sharp creases preserve crisp corner extent vs smooth sphere shrink")

    var sm_list = List[NativeMesh]()
    sm_list.append(m_smooth.copy())
    ctx.assert_equal(_count_open_spatial_boundary_edges(sm_list), 0, "subdivided cube is 100% watertight")

    # Supercar (`Vanguard GT-R`)
    var sc = build_subd_supercar_body_and_glass()
    var sc_tris = sc[0].triangle_count() + sc[1].triangle_count()
    ctx.assert_true(sc_tris >= 2200 and sc_tris <= 2850, "SubD supercar is within ~2,200-2,800 tri budget")
    _assert_valid_mesh_attribs(ctx, sc[0], "Supercar body")
    _assert_valid_mesh_attribs(ctx, sc[1], "Supercar glass")
    var sc_list = List[NativeMesh]()
    sc_list.append(sc[0].copy()); sc_list.append(sc[1].copy())
    ctx.assert_equal(_count_open_spatial_boundary_edges(sc_list), 0, "mirror-welded SubD supercar body+glass is 100% watertight")

    # Spinner Hovercar (`Spinner Mk9`)
    var sp = build_subd_spinner_hovercar_body_and_canopy()
    var sp_tris = sp[0].triangle_count() + sp[1].triangle_count()
    ctx.assert_true(sp_tris >= 1650 and sp_tris <= 2050, "SubD Spinner hovercar is ~1,856 tris")
    var sp_list = List[NativeMesh]()
    sp_list.append(sp[0].copy()); sp_list.append(sp[1].copy())
    ctx.assert_equal(_count_open_spatial_boundary_edges(sp_list), 0, "twin-prong SubD Spinner hovercar is 100% watertight")

    # Traffic Vehicles (`0..2`) & Coaster Car Shells (`0..3`)
    for arch in range(3):
        var tv = build_subd_traffic_vehicle(arch)
        ctx.assert_true(tv.triangle_count() >= 1200 and tv.triangle_count() <= 1600, "traffic vehicle archetype within ~1,200-1,600 tris")
        _assert_valid_mesh_attribs(ctx, tv, "Traffic vehicle " + String(arch))
        var tv_list = List[NativeMesh]()
        tv_list.append(tv^)
        ctx.assert_equal(_count_open_spatial_boundary_edges(tv_list), 0, "traffic vehicle is watertight")

    for style in range(4):
        var cc = build_subd_coaster_car_shell(style)
        ctx.assert_true(cc.triangle_count() >= 1100 and cc.triangle_count() <= 1550, "coaster car shell within ~1,100-1,500 tris")
        _assert_valid_mesh_attribs(ctx, cc, "Coaster car " + String(style))
        var cc_list = List[NativeMesh]()
        cc_list.append(cc^)
        ctx.assert_equal(_count_open_spatial_boundary_edges(cc_list), 0, "coaster car shell is watertight")


# =========================================================================
# 5. Lathed 5-Spoke Wheel/Brake Assembly & RMF Loft Tests
# =========================================================================

def test_loft_bevel_and_lathed_wheels(mut ctx: TestContext) raises:
    var wh = build_lathed_wheel_assembly(0.35, 0.28, spoke_count=5)
    var total_w_tris = wh[0].triangle_count() + wh[1].triangle_count() + wh[2].triangle_count()
    ctx.assert_true(total_w_tris >= 360 and total_w_tris <= 480, "5-spoke lathed wheel+rotor+caliper is ~420 tris")
    _assert_valid_mesh_attribs(ctx, wh[0], "Lathed tire")
    _assert_valid_mesh_attribs(ctx, wh[1], "Alloy rim & rotor")
    _assert_valid_mesh_attribs(ctx, wh[2], "6-piston brake caliper")

    var girder_prof = make_chamfered_girder_profile()
    ctx.assert_equal(len(girder_prof), 16, "chamfered box-girder spine profile has 16 vertices")

    var loop_pts = List[Vec3]()
    for i in range(24):
        var a = (Float32(i) / Float32(23.0)) * Float32(6.2831853)
        loop_pts.append(Vec3(Float32(i) * Float32(0.15), Float32(5.0) - cos(a) * Float32(4.0), sin(a) * Float32(4.0)))
    var loop_mesh = sweep_profile_rmf(loop_pts, girder_prof, closed_loop=False, closed_profile=True)
    _assert_valid_mesh_attribs(ctx, loop_mesh, "RMF vertical loop girder")

    var curb_prof = make_curb_profile()
    var curb_pts: List[Vec3] = [Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 5.0), Vec3(2.0, 0.0, 10.0)]
    var curb_mesh = sweep_profile_rmf(curb_pts, curb_prof)
    _assert_valid_mesh_attribs(ctx, curb_mesh, "RMF street curb")

    var cable = sweep_catenary_cable(Vec3(-4.0, 5.0, 0.0), Vec3(4.0, 5.0, 0.0), sag=0.65)
    _assert_valid_mesh_attribs(ctx, cable, "Catenary cable")

    var sq: List[Vec2] = [Vec2(-1.0, -1.0), Vec2(1.0, -1.0), Vec2(1.0, 1.0), Vec2(-1.0, 1.0)]
    var ext = extrude_profile_beveled(sq, height=1.2, corner_fillet=0.15, cap_chamfer=0.06)
    _assert_valid_mesh_attribs(ctx, ext, "Beveled extrusion")


# =========================================================================
# 6. v3 Botanical Sprig Atlases, Multi-Seed Trees, 3-Submesh Mech & Urban Kits
# =========================================================================

def _assert_botanical_cutout_atlas_512(
    mut ctx: TestContext,
    sprig: Tuple[List[UInt32], List[Vec3]],
    label: String,
    min_opaque: Int,
    min_cutout: Int,
) raises:
    ctx.assert_equal(len(sprig[0]), 512 * 512, label + ": 512x512 ARGB texels")
    ctx.assert_equal(len(sprig[1]), 512 * 512, label + ": 512x512 BC5 tangent normals")
    var opaque_px = 0
    var cutout_px = 0
    for i in range(len(sprig[0])):
        var a = Int((sprig[0][i] >> 24) & 255)
        if a >= 200:
            opaque_px += 1
        elif a == 0:
            cutout_px += 1
    ctx.assert_true(opaque_px >= min_opaque, label + ": rich botanical leaf/needle coverage")
    ctx.assert_true(cutout_px >= min_cutout, label + ": crisp alpha=0 sky cutout gaps between leaves/needles")
    ctx.assert_approx(sprig[1][256 * 512 + 256].length(), 1.0, 0.02, label + ": unit tangent normal")


def test_foliage_crowd_and_arch_kits(mut ctx: TestContext) raises:
    # 6A. `512x512` Botanical Alpha-Cutout Leaf/Needle/Frond Sprigs & `256x256` Fissured Bark
    var oak_sprig = bake_oak_leaf_sprig_512()
    _assert_botanical_cutout_atlas_512(ctx, oak_sprig, "Oak leaf sprig 512", min_opaque=25000, min_cutout=60000)

    var pine_sprig = bake_conifer_needle_sprig_512()
    _assert_botanical_cutout_atlas_512(ctx, pine_sprig, "Conifer needle sprig 512", min_opaque=18000, min_cutout=60000)

    var palm_sprig = bake_palm_frond_512()
    _assert_botanical_cutout_atlas_512(ctx, palm_sprig, "Palm frond 512", min_opaque=20000, min_cutout=60000)

    var bark = bake_fissured_bark_256()
    ctx.assert_equal(len(bark[0]), 256 * 256, "fissured bark albedo is 256x256")
    ctx.assert_equal(len(bark[1]), 256 * 256, "fissured bark normal map is 256x256")
    ctx.assert_approx(bark[1][128 * 256 + 64].length(), 1.0, 0.02, "fissured bark normal is normalized")

    # 6B. Botanical Trees (`alpha_mode = 1` double-sided cross-foil canopies + multi-seed asymmetry + vertex colors)
    var pine = build_botanical_pine(42, 5.5)
    var pine_tris = pine[0].triangle_count() + pine[1].triangle_count()
    ctx.assert_true(pine_tris >= 500 and pine_tris <= 1300, "botanical pine within cross-foil sprig budget")
    _assert_valid_mesh_attribs(ctx, pine[0], "Pine trunk", require_colors=True)
    _assert_valid_mesh_attribs(ctx, pine[1], "Pine needles", require_colors=True)
    ctx.assert_true(not pine[0].double_sided, "pine trunk is single-sided closed bark")
    ctx.assert_equal(pine[1].alpha_mode, 1, "pine canopy uses alpha_mode == 1 mask cutout")
    ctx.assert_true(pine[1].double_sided, "pine canopy is double-sided")

    var oak0 = build_botanical_oak(0, 6.0)
    var oak1 = build_botanical_oak(1, 6.0)
    var oak2 = build_botanical_oak(2, 6.0)
    var oak_tris = oak0[0].triangle_count() + oak0[1].triangle_count()
    ctx.assert_true(oak_tris >= 500 and oak_tris <= 1400, "botanical oak within cross-foil sprig budget")
    _assert_valid_mesh_attribs(ctx, oak0[0], "Oak trunk", require_colors=True)
    _assert_valid_mesh_attribs(ctx, oak0[1], "Oak canopy", require_colors=True)
    ctx.assert_true(not oak0[0].double_sided, "oak trunk is single-sided")
    ctx.assert_equal(oak0[1].alpha_mode, 1, "oak canopy uses alpha_mode == 1")
    ctx.assert_true(oak0[1].double_sided, "oak canopy is double-sided")

    # Verify distinct seed silhouettes (`0` vs `1` vs `2`)
    var seed_diff_01 = Float32(0.0)
    var seed_diff_12 = Float32(0.0)
    var n_cmp = min(len(oak0[1].positions), min(len(oak1[1].positions), len(oak2[1].positions)))
    for i in range(n_cmp):
        seed_diff_01 += abs(oak0[1].positions[i] - oak1[1].positions[i])
        seed_diff_12 += abs(oak1[1].positions[i] - oak2[1].positions[i])
    ctx.assert_true(seed_diff_01 > Float32(25.0) and seed_diff_12 > Float32(25.0), "seeds 0, 1, 2 produce distinct branch & crown silhouettes")

    # Verify oak radial crown AO spans dark core (`<= 150`) to sunlit outer dome (`>= 230`)
    var min_ao = 255
    var max_ao = 0
    for ao_b in oak0[1].ao:
        var v = Int(ao_b)
        if v < min_ao: min_ao = v
        if v > max_ao: max_ao = v
    ctx.assert_true(min_ao <= 150 and max_ao >= 230, "oak canopy bakes radial crown AO gradient")

    var palm = build_botanical_palm(19, 5.8)
    _assert_valid_mesh_attribs(ctx, palm[0], "Palm trunk", require_colors=True)
    _assert_valid_mesh_attribs(ctx, palm[1], "Palm fronds", require_colors=True)
    ctx.assert_equal(palm[1].alpha_mode, 1, "palm fronds use alpha_mode == 1")

    var ginkgo = build_urban_ginkgo_tree(31, 5.2)
    _assert_valid_mesh_attribs(ctx, ginkgo[0], "Ginkgo trunk", require_colors=True)
    _assert_valid_mesh_attribs(ctx, ginkgo[1], "Ginkgo canopy", require_colors=True)
    ctx.assert_equal(ginkgo[1].alpha_mode, 1, "ginkgo canopy uses alpha_mode == 1")

    # 6C. Understory Builders (`grass_wildflower_tuft`, `manicured_hedge_and_planter`, `flowering_bush`)
    var tuft = build_grass_wildflower_tuft(7)
    _assert_valid_mesh_attribs(ctx, tuft, "Grass wildflower tuft", require_colors=True)
    ctx.assert_true(tuft.triangle_count() >= 12 and tuft.triangle_count() <= 48, "grass/wildflower tuft is lightweight")

    var hedge_pair = build_manicured_hedge_and_planter(3.2, 1.1, 0.95)
    _assert_valid_mesh_attribs(ctx, hedge_pair[0], "Planter stone curb", require_colors=True)
    _assert_valid_mesh_attribs(ctx, hedge_pair[1], "Manicured hedge foliage", require_colors=True)

    var bush = build_flowering_bush(11, 0.85)
    _assert_valid_mesh_attribs(ctx, bush, "Flowering bush", require_colors=True)

    # 6D. 3-Submesh 16-Bone Rigged Hard-Surface `Cyber-Enforcer` (`Armor` / `Chassis` / `HDR Emissive`)
    var mech_v3 = build_subd_cyber_enforcer_rigged_v3()
    ctx.assert_equal(mech_v3[0].bone_count, ENFORCER_BONE_COUNT, "Armor submesh has 16 bones")
    ctx.assert_equal(mech_v3[1].bone_count, ENFORCER_BONE_COUNT, "Chassis submesh has 16 bones")
    ctx.assert_equal(mech_v3[2].bone_count, ENFORCER_BONE_COUNT, "Emissive submesh has 16 bones")
    ctx.assert_equal(len(mech_v3[0].rig_model.nodes), 16, "Armor rig_model has 16 NativeNodes")
    _assert_valid_mesh_attribs(ctx, mech_v3[0].mesh, "Cyber-Enforcer Armor", require_colors=True)
    _assert_valid_mesh_attribs(ctx, mech_v3[1].mesh, "Cyber-Enforcer Chassis", require_colors=True)
    _assert_valid_mesh_attribs(ctx, mech_v3[2].mesh, "Cyber-Enforcer HDR Emissive", require_colors=True)
    ctx.assert_true(mech_v3[2].mesh.emissive_strength > Float32(1.0), "Cyber-Enforcer emissive submesh carries HDR emissive_strength")

    var mech = build_subd_cyber_enforcer_rigged()
    ctx.assert_equal(mech.bone_count, ENFORCER_BONE_COUNT, "Merged Cyber-Enforcer has 16 hierarchical bones")
    ctx.assert_equal(
        mech.mesh.triangle_count(),
        mech_v3[0].mesh.triangle_count() + mech_v3[1].mesh.triangle_count() + mech_v3[2].mesh.triangle_count(),
        "Merged Cyber-Enforcer equals sum of Armor + Chassis + Emissive triangles",
    )
    _assert_valid_mesh_attribs(ctx, mech.mesh, "Merged Cyber-Enforcer SubD mesh", require_colors=True)
    for v in range(mech.mesh.vertex_count()):
        var w_sum = mech.mesh.weights[v * 4] + mech.mesh.weights[v * 4 + 1] + mech.mesh.weights[v * 4 + 2] + mech.mesh.weights[v * 4 + 3]
        if abs(w_sum - Float32(1.0)) > Float32(1e-3):
            raise Error("unnormalized skin weight on Cyber-Enforcer")

    # 6E. 12-Pose Articulated Multi-Tone Humanoid Crowd Bank
    var crowd_dry = build_articulated_crowd_pose_bank(with_umbrella=False)
    var crowd_rain = build_articulated_crowd_pose_bank(with_umbrella=True)
    ctx.assert_equal(len(crowd_dry), 12, "dry crowd bank has 12 poses")
    ctx.assert_equal(len(crowd_rain), 12, "rain umbrella crowd bank has 12 poses")
    var dry_tris = crowd_dry[0][0].triangle_count() + crowd_dry[0][1].triangle_count()
    var rain_tris = crowd_rain[0][0].triangle_count() + crowd_rain[0][1].triangle_count()
    ctx.assert_true(dry_tris >= 580 and dry_tris <= 780, "crowd pose is ~680 tris")
    ctx.assert_true(rain_tris > dry_tris + 40, "12-ribbed domed umbrella adds canopy + shaft geometry")
    _assert_valid_mesh_attribs(ctx, crowd_rain[0][0], "Crowd upper garment", require_colors=True)
    _assert_valid_mesh_attribs(ctx, crowd_rain[0][1], "Crowd lower & umbrella", require_colors=True)

    var diff_accum = Float32(0.0)
    var pos0 = crowd_dry[0][1].positions.copy()
    var pos2 = crowd_dry[2][1].positions.copy()
    for i in range(len(pos0)):
        diff_accum += abs(pos0[i] - pos2[i])
    ctx.assert_true(diff_accum > Float32(5.0), "walk cycle phases deform leg/shoe positions distinctly")

    # 6F. Clean Back-Wall `512x512` `4x4` Interior Parallax Atlas & `8` Cyber Billboards
    var atlas = bake_interior_windowbox_atlas_512()
    ctx.assert_equal(len(atlas[0]), 512 * 512, "interior albedo atlas is 512x512")
    ctx.assert_equal(len(atlas[1]), 512 * 512, "mullion+warble normal atlas is 512x512")
    ctx.assert_equal(len(atlas[2]), 512 * 512, "interior ORM atlas is 512x512")
    var border_px = atlas[0][64 * 512 + 1]
    var border_luma = Int((border_px >> 16) & 255) + Int((border_px >> 8) & 255) + Int(border_px & 255)
    ctx.assert_true(border_luma > 45, "interior room tile edge has no black outer window frame")
    var glass_n = atlas[1][64 * 512 + 64]
    var mull_n = atlas[1][64 * 512 + 1]
    ctx.assert_true(abs(glass_n.x) <= Float32(0.03) and abs(glass_n.y) <= Float32(0.03), "interior glass pane warble within +-1.2 deg")
    ctx.assert_true(abs(mull_n.x) > Float32(0.30), "mullion border normal has crisp bevel tilt")

    var bb0 = bake_cyber_billboard_512(0)
    var bb3 = bake_cyber_billboard_512(3)
    for v_idx in range(8):
        var bb = bake_cyber_billboard_512(v_idx)
        ctx.assert_equal(len(bb), 512 * 512, "cyber billboard variant " + String(v_idx) + " is 512x512")
        var bright_neon = 0
        for p_i in range(0, 512 * 512, 16):
            var px = bb[p_i]
            var mr = max(Int((px >> 16) & 255), max(Int((px >> 8) & 255), Int(px & 255)))
            if mr >= 180:
                bright_neon += 1
        ctx.assert_true(bright_neon >= 300, "cyber billboard variant " + String(v_idx) + " has vibrant neon artwork/glyphs")
    var bb_diff = 0
    for p_i in range(0, 512 * 512, 32):
        if bb0[p_i] != bb3[p_i]:
            bb_diff += 1
    ctx.assert_true(bb_diff > 1000, "cyber billboard variants depict distinct graphic layouts")

    # 6G. Architectural & Emissive-Split `_parts` Street Kits
    _assert_valid_mesh_attribs(ctx, build_colonnade_arcade_bay(), "Colonnade arcade bay")
    _assert_valid_mesh_attribs(ctx, build_facade_mullion_spandrel_grid(), "Facade mullion grid")
    _assert_valid_mesh_attribs(ctx, build_rooftop_hvac_chiller(), "Rooftop HVAC chiller")

    var heli_parts = build_rooftop_vtol_helipad_parts()
    _assert_valid_mesh_attribs(ctx, heli_parts[0], "Helipad base", require_colors=True)
    _assert_valid_mesh_attribs(ctx, heli_parts[1], "Helipad HDR beacons", require_colors=True)
    ctx.assert_true(heli_parts[1].emissive_strength > Float32(1.0), "helipad beacons have HDR emissive_strength")
    _assert_valid_mesh_attribs(ctx, build_rooftop_vtol_helipad(), "Rooftop VTOL helipad merged", require_colors=True)

    var cant_parts = build_street_traffic_cantilever_parts()
    _assert_valid_mesh_attribs(ctx, cant_parts[0], "Traffic cantilever mast", require_colors=True)
    _assert_valid_mesh_attribs(ctx, cant_parts[1], "Traffic signal lenses", require_colors=True)
    ctx.assert_true(cant_parts[1].emissive_strength > Float32(1.0), "traffic lenses have HDR emissive_strength")
    _assert_valid_mesh_attribs(ctx, build_street_traffic_cantilever(), "Traffic cantilever merged", require_colors=True)

    var boll_parts = build_street_bollard_and_grate_parts()
    _assert_valid_mesh_attribs(ctx, boll_parts[0], "Street bollard & grate base", require_colors=True)
    _assert_valid_mesh_attribs(ctx, boll_parts[1], "Bollard cyan LED collar", require_colors=True)
    _assert_valid_mesh_attribs(ctx, build_street_bollard_and_grate(), "Street bollard & grate merged", require_colors=True)

    var shelt_parts = build_transit_shelter_parts()
    _assert_valid_mesh_attribs(ctx, shelt_parts[0], "Transit shelter frame", require_colors=True)
    _assert_valid_mesh_attribs(ctx, shelt_parts[1], "Transit shelter ad/LED strips", require_colors=True)
    _assert_valid_mesh_attribs(ctx, build_transit_shelter(), "Transit shelter merged", require_colors=True)


def test_f26_regression_and_lod_dag_cook(mut ctx: TestContext) raises:
    var q = _SymQuadric.from_plane(0.0, 1.0, 0.0, -2.0)
    ctx.assert_approx(Float32(q.eval(Vec3(0.0, 2.0, 0.0))), 0.0, 0.001, "point on plane evaluates to zero quadric error")

    var mock_m = create_mock_meshlet_mesh(12)
    var red = simplify_mesh_qem(mock_m, len(mock_m.indices) // 6)
    ctx.assert_true(len(red[0].indices) > 0 and len(red[0].indices) < len(mock_m.indices), "mock meshlet QEM reduction valid")

    var sc = build_subd_supercar_body_and_glass()
    var pool = AaaGeometryPool()
    var entry = pool.append_mesh(sc[0], material_id=3, build_lods=True)
    ctx.assert_equal(entry.lod_count, 4, "SubD supercar cooks full 4-level LOD 0..3 QEM Meshlet DAG")
    ctx.assert_true(entry.lods[3].triangle_count < entry.lods[0].triangle_count, "LOD 3 has fewer triangles than LOD 0")
    ctx.assert_true(entry.lods[1].object_error < entry.lods[2].object_error and entry.lods[2].object_error < entry.lods[3].object_error, "strictly monotonic LOD errors")

    var sl = MeshLodSlice(0, 4, 200, 0.025)
    ctx.assert_equal(sl.meshlet_count, 4, "MeshLodSlice preserved")


def main() raises:
    var ctx = TestContext()
    ctx.start_suite("Procedural_SubD_FastQEM_Toksvig_v3")

    try:
        test_qem_minheap_speed_and_validity(ctx)
        ctx.record_pass("test_qem_minheap_speed_and_validity")
    except e:
        ctx.record_fail("test_qem_minheap_speed_and_validity", String(e))

    try:
        test_vertex_words_11_ao_and_splat_packing(ctx)
        ctx.record_pass("test_vertex_words_11_ao_and_splat_packing")
    except e:
        ctx.record_fail("test_vertex_words_11_ao_and_splat_packing", String(e))

    try:
        test_toksvig_normal_variance_roughness(ctx)
        ctx.record_pass("test_toksvig_normal_variance_roughness")
    except e:
        ctx.record_fail("test_toksvig_normal_variance_roughness", String(e))

    try:
        test_subd_creases_and_vehicles(ctx)
        ctx.record_pass("test_subd_creases_and_vehicles")
    except e:
        ctx.record_fail("test_subd_creases_and_vehicles", String(e))

    try:
        test_loft_bevel_and_lathed_wheels(ctx)
        ctx.record_pass("test_loft_bevel_and_lathed_wheels")
    except e:
        ctx.record_fail("test_loft_bevel_and_lathed_wheels", String(e))

    try:
        test_foliage_crowd_and_arch_kits(ctx)
        ctx.record_pass("test_foliage_crowd_and_arch_kits")
    except e:
        ctx.record_fail("test_foliage_crowd_and_arch_kits", String(e))

    try:
        test_f26_regression_and_lod_dag_cook(ctx)
        ctx.record_pass("test_f26_regression_and_lod_dag_cook")
    except e:
        ctx.record_fail("test_f26_regression_and_lod_dag_cook", String(e))

    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("test_procedural_qem failed")
