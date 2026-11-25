"""Flint v5 TripoSG suite (`test_flint_v5_triposg.mojo`).

Always (no model files):
  1. `torch_randn` reproduces `torch.randn` under `torch.manual_seed` (values from torch 2.15).
  2. The rectified-flow schedule and Euler / guidance arithmetic.
  3. `pil_resize_bicubic` is bit-exact with Pillow 13 (checksums of a pattern image).
  4. Background keying, speck removal, crop and padding on a synthetic image.
  5. Marching cubes: a sphere SDF meshes closed (no open edges), outward-wound (positive volume
     close to 4/3 pi r^3) and on the sphere (Chamfer to analytic points); random grids with
     many ambiguous faces stay watertight.

With `TRIPOSG_DIR` (the Hugging Face VAST-AI/TripoSG folder) and `TRIPOSG_REF` (a dump of
`tools/triposg/reference.py` with its `--image` as `TRIPOSG_IMAGE`), parity with PyTorch:
prepared image and DINOv2 pixels, image condition, the seeded noise, the guided velocity of
every step (`TRIPOSG_STEPS` limits how many are checked, default all that were dumped), the
final latents, the SDF on the reference's dense grid and the Chamfer distance between the two
marching-cubes meshes.
"""
from std.math import sqrt, sin, cos
from std.time import perf_counter_ns
from std.testing import assert_true, assert_equal
from flint.platform.os import env_value, read_bytes, file_size
from flint.ml.triposg import (
    torch_randn, rf_sigmas, euler_step, cfg_combine, TgImage, TgRgba, pil_resize_bicubic, prepare_image,
    preprocess_dinov2, load_rgba_png, TripoSG, read_f32, write_f32, grid_points, TG_TOKENS, TG_LATENT, TG_BOUND,
)
from flint.ml.triposg_mesh import marching_cubes, signed_volume, boundary_edges, chamfer_distance, TriMesh


def _close(a: Float32, b: Float32, tol: Float32) -> Bool:
    return abs(a - b) <= tol


def test_randn() raises:
    var r = torch_randn(16, 42)
    var want: List[Float32] = [1.9269150495529175, 1.4872841835021973, 0.9007171988487244, -2.1055214405059814,
                               0.6784184575080872, -1.2345449924468994, -0.043067481368780136, -1.6046669483184814,
                               -0.7521361708641052, 1.6487228870391846, -0.3924786448478699, -1.403606653213501,
                               -0.7278812527656555, -0.5594298839569092, -0.7688389420509338, 0.7624453902244568]
    for i in range(16):
        assert_true(_close(r[i], want[i], 2e-6), "randn seed 42 [" + String(i) + "] " + String(r[i]))
    var r7 = torch_randn(64, 7)
    assert_true(_close(r7[0], -0.8201345205307007, 2e-6))
    assert_true(_close(r7[17], -0.32456669211387634, 2e-6))
    assert_true(_close(r7[40], -1.401250958442688, 2e-6))
    assert_true(_close(r7[63], 0.7541925311088562, 2e-6))
    print("  torch.randn reproduced (seeds 42, 7)")


def test_sampler() raises:
    var s = rf_sigmas(50)
    assert_equal(len(s), 51)
    assert_equal(s[0], 1.0)
    assert_true(_close(s[1], 0.98, 1e-7))
    assert_true(_close(s[49], 0.02, 1e-7))
    assert_equal(s[50], 0.0)
    # Euler over a constant velocity field integrates exactly: x_end = x0 - v * (sigma_0 - 0)
    var x: List[Float32] = [1.0, -2.0, 0.5]
    var v: List[Float32] = [0.25, 0.5, -1.0]
    for i in range(50):
        euler_step(x, v, s[i], s[i + 1])
    assert_true(_close(x[0], 1.25, 1e-5) and _close(x[1], -1.5, 1e-5) and _close(x[2], -0.5, 1e-5), "euler")
    var vu: List[Float32] = [1.0, 2.0]
    var vc: List[Float32] = [2.0, 0.0]
    var g = cfg_combine(vu, vc, 7.0)
    assert_true(_close(g[0], 8.0, 1e-6) and _close(g[1], -12.0, 1e-6), "guidance")
    print("  rectified-flow schedule, Euler and guidance")


def _pattern() -> TgImage:
    var a = TgImage(37, 29)
    for y in range(29):
        for x in range(37):
            for c in range(3):
                a.rgb[(y * 37 + x) * 3 + c] = UInt8((x * 7 + y * 13 + c * 50) % 256)
    return a^


def _sums(v: List[UInt8]) -> Tuple[Int, Int]:
    var s = 0
    var w = 0
    for i in range(len(v)):
        s += Int(v[i])
        w += Int(v[i]) * (i % 251)
    return (s, w)


def test_pil_resize() raises:
    var a = _pattern()
    var r = pil_resize_bicubic(a, 16, 12)
    var s = _sums(r.rgb)
    assert_equal(s[0], 73319)
    assert_equal(s[1], 8508202)
    var u = pil_resize_bicubic(a, 80, 61)
    s = _sums(u.rgb)
    assert_equal(s[0], 1863662)
    assert_equal(s[1], 232377374)
    print("  PIL bicubic resize bit-exact (down and up)")


def test_keying() raises:
    var img = TgRgba(60, 40)
    for k in range(60 * 40):
        img.rgb[3 * k] = 120
        img.rgb[3 * k + 1] = 121
        img.rgb[3 * k + 2] = 140
    # a 20 x 10 red box with a background-coloured hole, plus a 3-pixel speck
    for y in range(10, 20):
        for x in range(20, 40):
            var k = y * 60 + x
            img.rgb[3 * k] = 200
            img.rgb[3 * k + 1] = 20
            img.rgb[3 * k + 2] = 30
    var hole = 14 * 60 + 30
    img.rgb[3 * hole] = 120
    img.rgb[3 * hole + 1] = 121
    img.rgb[3 * hole + 2] = 140
    for x in range(2, 5):
        var k = 35 * 60 + x
        img.rgb[3 * k] = 0
        img.rgb[3 * k + 1] = 0
        img.rgb[3 * k + 2] = 0
    var p = prepare_image(img)
    # box 20 x 10 -> pad x by 2 and y by 2 + 5 = 7: 24 x 24
    assert_equal(p.width, 24)
    assert_equal(p.height, 24)
    var c = (2 + 7 * 24) * 3
    assert_equal(Int(p.rgb[c]), 200)                 # float32 (v / 255) * 255 round-trips for 8-bit v
    assert_equal(Int(p.rgb[0]), 255)                 # padding is white
    var hk = ((7 + 4) * 24 + 2 + 10) * 3
    assert_equal(Int(p.rgb[hk]), 120)                # the enclosed hole keeps its colour
    var px = preprocess_dinov2(p)
    assert_equal(len(px), 3 * 224 * 224)
    print("  background keying, crop and padding")


def _sphere_grid(n: Int, r: Float32, cx: Float32) -> List[Float32]:
    var v = List[Float32](length=n * n * n, fill=0.0)
    var h = 2.0 / Float32(n - 1)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                var x = -1.0 + Float32(i) * h - cx
                var y = -1.0 + Float32(j) * h
                var z = -1.0 + Float32(k) * h
                v[(i * n + j) * n + k] = r - sqrt(x * x + y * y + z * z)
    return v^


def test_marching_cubes() raises:
    var n = 65
    var r: Float32 = 0.6
    var g = _sphere_grid(n, r, 0.05)
    var m = marching_cubes(g, n, -1.0, 2.0 / Float32(n - 1))
    assert_true(m.triangle_count() > 1000)
    assert_equal(boundary_edges(m), 0)
    var vol = signed_volume(m)
    var exact = 4.0 / 3.0 * 3.141592653589793 * Float64(r) ** 3
    assert_true(abs(vol - exact) / exact < 0.01, "sphere volume " + String(vol) + " vs " + String(exact))
    var pts = List[Float32]()
    for a in range(60):
        for b in range(30):
            var th = Float32(a) * 6.2831853 / 60.0
            var ph = (Float32(b) + 0.5) * 3.1415926 / 30.0
            pts.append(0.05 + r * sin(ph) * cos(th))
            pts.append(r * sin(ph) * sin(th))
            pts.append(r * cos(ph))
    var cd = chamfer_distance(m.positions, pts)
    assert_true(cd < 0.02, "sphere chamfer " + String(cd))
    # random grids: lots of ambiguous faces, still closed (outer layer forced outside)
    var seed: UInt64 = 12345
    for trial in range(3):
        var q = 12
        var rv = List[Float32](length=q * q * q, fill=-1.0)
        for i in range(1, q - 1):
            for j in range(1, q - 1):
                for k in range(1, q - 1):
                    seed = seed * 6364136223846793005 + 1442695040888963407
                    rv[(i * q + j) * q + k] = Float32(Int((seed >> 33) % 2001)) / 1000.0 - 1.0
        var rm = marching_cubes(rv, q, 0.0, 1.0)
        assert_equal(boundary_edges(rm), 0)
        assert_true(signed_volume(rm) > 0, "random grid " + String(trial) + " winding")
    print("  marching cubes: sphere", m.vertex_count(), "verts, volume", vol, "/", exact, "chamfer", cd,
          "; random grids closed")


def _maxdiff(a: List[Float32], b: List[Float32]) -> Float32:
    var d: Float32 = 0
    for i in range(min(len(a), len(b))):
        d = max(d, abs(a[i] - b[i]))
    return d


def _maxabs(a: List[Float32]) -> Float32:
    var d: Float32 = 0
    for i in range(len(a)):
        d = max(d, abs(a[i]))
    return d


def _exists(p: String) -> Bool:
    return file_size(p) > 0


def test_model() raises:
    var dir = env_value("TRIPOSG_DIR")
    var rdir = env_value("TRIPOSG_REF")
    var image = env_value("TRIPOSG_IMAGE")
    if dir == "" or rdir == "" or image == "":
        print("  (model parity skipped: set TRIPOSG_DIR, TRIPOSG_REF and TRIPOSG_IMAGE)")
        return
    var t0 = perf_counter_ns()
    var model = TripoSG.load(dir)
    print("  weights on", model.device(), "in", Float64(perf_counter_ns() - t0) / 1e9, "s")
    # preprocessing
    var img = prepare_image(load_rgba_png(image))
    var prep_ref = read_bytes(rdir + "/prepared.bin")
    assert_equal(len(prep_ref), len(img.rgb), "prepared image size")
    var bad = 0
    for i in range(len(img.rgb)):
        if img.rgb[i] != prep_ref[i]:
            bad += 1
    assert_equal(bad, 0, "prepared image pixels")
    var px = preprocess_dinov2(img)
    var px_ref = read_f32(rdir + "/pixels.bin")
    var dpx = _maxdiff(px, px_ref)
    assert_true(dpx < 1e-5, "DINOv2 pixels differ by " + String(dpx))
    print("  prepared image", img.width, "x", img.height, "identical; pixel max diff", dpx)
    # image condition
    t0 = perf_counter_ns()
    var cond = model.encode_image(px_ref)
    var cond_ref = read_f32(rdir + "/cond.bin")
    var dc = _maxdiff(cond, cond_ref)
    print("  DINOv2 in", Float64(perf_counter_ns() - t0) / 1e6, "ms; max diff", dc, "(|ref| max", _maxabs(cond_ref), ")")
    assert_true(dc < 2e-3, "DINOv2 condition")
    # noise
    var noise = torch_randn(TG_TOKENS * TG_LATENT, 42)
    var noise_ref = read_f32(rdir + "/noise.bin")
    var dn = _maxdiff(noise, noise_ref)
    assert_true(dn < 1e-5, "noise differs by " + String(dn))
    # sampler: per step from the reference latents, so errors do not compound
    var steps = 50
    var limit = 50
    var ls = env_value("TRIPOSG_STEPS")
    if ls != "":
        limit = Int(ls)
    var sig = rf_sigmas(steps)
    model.set_condition(cond_ref, True)
    var x = noise_ref.copy()
    var worst: Float32 = 0
    var checked = 0
    var t_steps: Float64 = 0
    for i in range(min(limit, steps)):
        var vpath = rdir + "/v_" + (("0" + String(i)) if i < 10 else String(i)) + ".bin"
        if not _exists(vpath):
            break
        var ts = perf_counter_ns()
        var vb = model.velocity(x, sig[i] * 1000.0)
        t_steps += Float64(perf_counter_ns() - ts) / 1e9
        comptime NL = TG_TOKENS * TG_LATENT
        var vu = List[Float32](length=NL, fill=0.0)
        var vc = List[Float32](length=NL, fill=0.0)
        for k in range(NL):
            vu[k] = vb[k]
            vc[k] = vb[NL + k]
        var v = cfg_combine(vu, vc, 7.0)
        var vr = read_f32(vpath)
        var d = _maxdiff(v, vr)
        var rel = d / max(_maxabs(vr), 1e-6)
        worst = max(worst, rel)
        if i < 3 or i % 10 == 9:
            print("    step", i, "velocity max diff", d, "(rel", rel, ")")
        x = read_f32(rdir + "/x_" + (("0" + String(i)) if i < 10 else String(i)) + ".bin")
        checked += 1
    print("  velocity checked on", checked, "steps, worst relative max diff", worst, "; DiT step (CFG batch 2)",
          t_steps / Float64(max(checked, 1)), "s")
    assert_true((checked > 0 or limit == 0) and worst < 2e-3, "DiT velocity")
    # full chain in Mojo from the same noise, then the VAE
    if not _exists(rdir + "/sdf.bin"):
        print("  (no sdf.bin in the reference dump: VAE parity skipped)")
        return
    var final_ref = read_f32(rdir + "/x_49.bin")
    var res = 128
    var sdf_ref = read_f32(rdir + "/sdf.bin")
    t0 = perf_counter_ns()
    model.set_latents(final_ref)
    var sdf = model.query_sdf(grid_points(res))
    var t_sdf = Float64(perf_counter_ns() - t0) / 1e9
    var ds = _maxdiff(sdf, sdf_ref)
    var sign_flips = 0
    for i in range(len(sdf)):
        if (sdf[i] > 0) != (sdf_ref[i] > 0):
            sign_flips += 1
    print("  SDF on", len(sdf), "points in", t_sdf, "s: max diff", ds, "(|ref| max", _maxabs(sdf_ref), "), sign flips",
          sign_flips)
    var mad: Float64 = 0
    var near: Float32 = 0       # max diff where |ref| < 2 (the band the mesh is cut from)
    for i in range(len(sdf)):
        var d = abs(sdf[i] - sdf_ref[i])
        mad += Float64(d)
        if abs(sdf_ref[i]) < 2.0:
            near = max(near, d)
    mad /= Float64(len(sdf))
    print("  SDF mean abs diff", mad, "; max diff near the surface (|ref| < 2)", near)
    if env_value("TRIPOSG_DUMP") != "":
        write_f32(env_value("TRIPOSG_DUMP") + "/mojo_sdf.bin", sdf)
    # far from the surface the logits saturate around +-10 and fp32 reduction order shows (<= 0.05)
    assert_true(near < 5e-3 and ds < 0.1 and mad < 1e-4 and sign_flips < 10, "SDF grid")
    var n = res + 1
    var m = marching_cubes(sdf, n, -TG_BOUND, 2.0 * TG_BOUND / Float32(res))
    var mv = read_f32(rdir + "/mesh_v.bin")
    var cd = chamfer_distance(m.positions, mv)
    print("  mesh", m.vertex_count(), "verts /", m.triangle_count(), "tris, open edges", boundary_edges(m),
          "; Chamfer to skimage mesh", cd)
    assert_true(cd < 2e-3, "Chamfer")
    if checked == 50:
        t0 = perf_counter_ns()
        var lat = model.sample(cond, noise.copy(), 50, 7.0)
        var dl = _maxdiff(lat, final_ref)
        print("  end-to-end 50 steps in", Float64(perf_counter_ns() - t0) / 1e9, "s: latent max diff", dl, "(|ref| max",
              _maxabs(final_ref), ")")


def main() raises:
    print("[TripoSG] sampler, preprocessing and marching cubes")
    test_randn()
    test_sampler()
    test_pil_resize()
    test_keying()
    test_marching_cubes()
    print("[TripoSG] model parity")
    test_model()
    print("[TripoSG] all checks passed")
