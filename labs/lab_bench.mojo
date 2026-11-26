"""Flint v5 large-scene render benchmark lab (`labs/lab_bench.mojo`, `bash tools/flint.sh lab bench`).

Area F (render SOTA): Khronos Sponza, Amazon Lumberyard Bistro (interior) and a dense CC0 forest,
loaded through Flint's own glTF/GLB path from `~/flint-models/_scenes` (never committed; LICENSES.md
there). For each scene x resolution it renders the static, moving-camera, moving-light and
skinned-character variants and prints per-pass GPU medians, CPU time, VRAM and FLIP against Flint's
path tracer in photo mode. Tables (markdown) and PNGs go to `~/flint-models/_eval/render/F/`.

Arguments (all optional, `--` before them when run through flint.sh):
  --scenes sponza,bistro,forest   --res 1080,2160   --frames 48   --ref-frames 96
  --preset high|default   --variants static,moving_cam,moving_light,skinned   --preview (640x360 static PNGs)
  --tsr4k 0.5 (render scale at 2160p)   --out <dir>   --robust (light-cut lag + ghosting probes)

Each resolution runs in its own child process (the lab re-executes itself with `--child`): a second
`FlintAaaVulkanRenderer` created in the same process after `shutdown()` of the first renders nothing
(every `render_frame` of the 2nd instance fails) - a v5 renderer-lifecycle bug recorded in the Area F notes.
"""
from std.ffi import external_call
from std.collections import List
from std.sys import argv
from flint.assets.image import write_png
from flint.platform.os import env_now, get_time, make_dirs, read_bytes, write_bytes
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.flip import flip_error_map, flip_heat_image, flip_mean
from flint.render.sota_perf_bench import (
    SOTA_BENCH_MOVING_CAMERA,
    SOTA_BENCH_MOVING_LIGHT,
    SOTA_BENCH_PRESET_DEFAULT,
    SOTA_BENCH_PRESET_HIGH,
    SOTA_BENCH_SKINNED,
    SOTA_BENCH_STATIC,
    SotaBenchCharacters,
    SotaBenchResult,
    sota_bench_flip,
    sota_bench_geometry_mb,
    sota_bench_apply_preset,
    sota_bench_hiz_shadow_probe,
    sota_bench_light_lag,
    sota_bench_motion_ghosting,
    sota_bench_sun,
    sota_bench_load_characters,
    sota_bench_load_scene,
    sota_bench_photo_reference,
    sota_bench_run,
    sota_bench_scene_dir,
    sota_bench_upload_scene,
    sota_bench_variant_name,
    sota_fmt,
    sota_perf_vram_usage,
)


def _split(s: String) -> List[String]:
    var out = List[String]()
    var cur = String("")
    for i in range(s.byte_length()):
        var c = String(s[byte=i])
        if c == ",":
            if cur.byte_length() > 0:
                out.append(cur)
            cur = ""
        else:
            cur += c
    if cur.byte_length() > 0:
        out.append(cur)
    return out^


def _scene_path(name: String) -> String:
    var d = sota_bench_scene_dir()
    if name == "sponza":
        return d + "/sponza_khronos/Sponza.gltf"
    if name == "bistro":
        return d + "/bistro/BistroInterior_Wine.glb"
    if name == "bistro_ext":
        return d + "/bistro/BistroExterior.glb"
    if name == "forest":
        return d + "/forest/Forest.glb"
    return name


def _pad(s: String, n: Int) -> String:
    var r = s
    while r.byte_length() < n:
        r += " "
    return r


def _system(cmd: String) -> Int:
    var owned = cmd
    return Int(external_call["system", Int32](owned.as_c_string_span()))


def main() raises:
    var a = argv()
    var child = False
    var md_name = String("")
    var scenes = _split("sponza,bistro,forest")
    var resolutions = List[Int]()
    resolutions.append(1080)
    resolutions.append(2160)
    var frames = 48
    var ref_frames = 96
    var preset = SOTA_BENCH_PRESET_HIGH
    var variants = List[Int]()
    variants.append(SOTA_BENCH_STATIC)
    variants.append(SOTA_BENCH_MOVING_CAMERA)
    variants.append(SOTA_BENCH_MOVING_LIGHT)
    variants.append(SOTA_BENCH_SKINNED)
    var preview = False
    var robust = False
    var no_hiz = False
    var no_cone = False
    var lods = True
    var tsr4k = Float32(0.5)
    var out_dir = env_now("HOME") + "/flint-models/_eval/render/F"
    var i = 1
    while i < len(a):
        var k = String(a[i])
        var v = String(a[i + 1]) if i + 1 < len(a) else String("")
        if k == "--scenes":
            scenes = _split(v); i += 1
        elif k == "--res":
            resolutions.clear()
            for r in _split(v):
                resolutions.append(Int(r))
            i += 1
        elif k == "--frames":
            frames = Int(v); i += 1
        elif k == "--ref-frames":
            ref_frames = Int(v); i += 1
        elif k == "--preset":
            preset = SOTA_BENCH_PRESET_DEFAULT if v == "default" else SOTA_BENCH_PRESET_HIGH; i += 1
        elif k == "--variants":
            variants.clear()
            for n in _split(v):
                if n == "static": variants.append(SOTA_BENCH_STATIC)
                elif n == "moving_cam": variants.append(SOTA_BENCH_MOVING_CAMERA)
                elif n == "moving_light": variants.append(SOTA_BENCH_MOVING_LIGHT)
                elif n == "skinned": variants.append(SOTA_BENCH_SKINNED)
            i += 1
        elif k == "--tsr4k":
            tsr4k = Float32(Float64(v)); i += 1
        elif k == "--out":
            out_dir = v; i += 1
        elif k == "--preview":
            preview = True
        elif k == "--robust":
            robust = True
        elif k == "--no-hiz":
            no_hiz = True
        elif k == "--no-lods":
            lods = False
        elif k == "--no-cone":
            no_cone = True
        elif k == "--child":
            child = True
        elif k == "--md-name":
            md_name = v; i += 1
        i += 1
    var preset_name0 = String("high") if preset == SOTA_BENCH_PRESET_HIGH else String("default")
    if not child and not preview and len(resolutions) > 1:
        # one process per resolution (see the module docstring), then merge the tables
        _ = make_dirs(out_dir)
        var body = String("")
        var rob_all = String("")
        var t0 = get_time()
        for rh in resolutions:
            var cmd = String("'") + String(a[0]) + "'"
            var j = 1
            while j < len(a):
                var kk = String(a[j])
                if kk == "--res" or kk == "--md-name":
                    j += 2
                    continue
                cmd += " '" + kk + "'"
                j += 1
            var part = String("bench_") + preset_name0 + "_" + String(rh) + "p.md"
            cmd += " --res " + String(rh) + " --child --md-name " + part
            print("[bench] --- " + String(rh) + "p (child process) ---")
            var rc = _system(cmd)
            if rc != 0:
                print("[bench] child for " + String(rh) + "p exited with " + String(rc))
            var txt = String(StringSlice(unsafe_from_utf8=read_bytes(out_dir + "/" + part)))
            # split each child table into its main rows and its robustness rows
            var in_rob = False
            var line = String("")
            for bi in range(txt.byte_length()):
                var c = String(txt[byte=bi])
                if c != "\n":
                    line += c
                    continue
                if line.startswith("## Dynamic robustness"):
                    in_rob = True
                elif line.startswith("| ") and not line.startswith("| scene"):
                    if in_rob:
                        rob_all += line + "\n"
                    else:
                        body += line + "\n"
                line = ""
        var mdm = String("# Flint large-scene benchmark (Area F)\n\npreset `") + preset_name0 + "`, " + String(frames)
        mdm += " measured frames after 8 warm-up frames, photo-mode reference " + String(ref_frames) + " frames x 4 spp.\n\n"
        mdm += "| scene | res (render) | variant | GPU ms med | p95 | frame period ms | CPU ms | wait ms | VRAM MB | FLIP | top passes (ms) |\n"
        mdm += "|---|---|---|---|---|---|---|---|---|---|---|\n" + body
        if robust:
            mdm += "\n## Dynamic robustness\n\n| scene | res | light-cut lag (frames) | FLIP 1st frame after cut | FLIP noise floor | ghosting FLIP (moving vs converged) | floor |\n|---|---|---|---|---|---|---|\n" + rob_all
        var mpath = out_dir + "/bench_" + preset_name0 + ".md"
        var mbytes = List[UInt8](mdm.as_bytes())
        _ = write_bytes(mpath, mbytes)
        print("[bench] all resolutions done in " + sota_fmt(get_time() - t0, 1) + " s -> " + mpath)
        return
    _ = make_dirs(out_dir)
    var preset_name = String("high") if preset == SOTA_BENCH_PRESET_HIGH else String("default")
    var md = String("# Flint large-scene benchmark (Area F)\n\npreset `") + preset_name + "`, " + String(frames)
    md += " measured frames after 8 warm-up frames, photo-mode reference " + String(ref_frames) + " frames x 4 spp.\n\n"
    md += "| scene | res (render) | variant | GPU ms med | p95 | frame period ms | CPU ms | wait ms | VRAM MB | FLIP | top passes (ms) |\n"
    md += "|---|---|---|---|---|---|---|---|---|---|---|\n"
    var rob = String("\n## Dynamic robustness\n\n| scene | res | light-cut lag (frames) | FLIP 1st frame after cut | FLIP noise floor | ghosting FLIP (moving vs converged) | floor |\n|---|---|---|---|---|---|---|\n")
    var t_all = get_time()
    var char_path = sota_bench_scene_dir() + "/characters/CesiumMan.glb"
    for s_name in scenes:
        var t_load = get_time()
        var scene = sota_bench_load_scene(s_name, _scene_path(s_name))
        if not scene.loaded:
            print("[bench] SKIP " + s_name + ": not found or rejected at " + _scene_path(s_name))
            continue
        print("[bench] " + s_name + ": " + String(len(scene.model.meshes)) + " primitives, " + String(scene.draw_count)
              + " draws, " + String(scene.unique_tris) + " unique / " + String(scene.drawn_tris) + " drawn tris, "
              + String(len(scene.model.textures)) + " textures, parse+decode " + sota_fmt(get_time() - t_load, 2) + " s")
        print("        bounds (" + sota_fmt(Float64(scene.bmin.x), 2) + ", " + sota_fmt(Float64(scene.bmin.y), 2) + ", " + sota_fmt(Float64(scene.bmin.z), 2)
              + ") - (" + sota_fmt(Float64(scene.bmax.x), 2) + ", " + sota_fmt(Float64(scene.bmax.y), 2) + ", " + sota_fmt(Float64(scene.bmax.z), 2) + ")")
        var res_list = resolutions.copy()
        if preview:
            res_list.clear()
            res_list.append(360)
        for rh in res_list:
            var rw = rh * 16 // 9
            var scale = tsr4k if rh >= 2000 else Float32(1.0)
            var gmb = sota_bench_geometry_mb(scene)
            var ren = FlintAaaVulkanRenderer(width=rw, height=rh, tsr_scale=scale, geometry_mb=gmb)
            if not ren.active:
                print("[bench] SKIP: no Vulkan device")
                return
            if no_hiz:
                ren.enable_hiz_cull = False
            if no_cone:
                ren.enable_cone_cull = False
            var vr0 = sota_perf_vram_usage(ren.vk)
            _ = sota_bench_upload_scene(ren, scene, 1024, lods)
            var vr1 = sota_perf_vram_usage(ren.vk)
            print("    " + String(rw) + "x" + String(rh) + " (render " + String(ren.render_w) + "x" + String(ren.render_h) + "): upload "
                  + sota_fmt(scene.upload_s, 2) + " s, scene VRAM " + sota_fmt(Float64(vr1[0] - vr0[0]) / 1048576.0, 0) + " MB, geometry pools "
                  + String(gmb) + " MB, overflowed meshes " + String(ren.geometry_overflow_meshes))
            var chars = SotaBenchCharacters()
            var tag = s_name + "_" + String(rh) + "p_" + preset_name
            if preview:
                var r = sota_bench_run(ren, scene, chars, SOTA_BENCH_STATIC, preset, 4, 8)
                _ = write_png(out_dir + "/preview_" + tag + ".png", r.pixels, rw, rh, stamped=False)
                print("    preview " + sota_fmt(r.gpu_ms, 2) + " ms -> " + out_dir + "/preview_" + tag + ".png")
                ren.shutdown()
                continue
            var reference = List[UInt32]()
            if preset == SOTA_BENCH_PRESET_HIGH:
                var t_ref = get_time()
                reference = sota_bench_photo_reference(ren, scene, ref_frames)
                if len(reference) > 0:
                    _ = write_png(out_dir + "/ref_pt_" + s_name + "_" + String(rh) + "p.png", reference, rw, rh, stamped=False)
                    print("    photo-mode reference: " + sota_fmt(get_time() - t_ref, 1) + " s")
            for v in variants:
                if v == SOTA_BENCH_SKINNED and not chars.loaded:
                    chars = sota_bench_load_characters(ren, char_path, 4)
                    if not chars.loaded:
                        print("    skinned: no character at " + char_path)
                        continue
                var r = sota_bench_run(ren, scene, chars, v, preset, frames)
                if not r.ok:
                    print("    " + sota_bench_variant_name(v) + ": render failed")
                    continue
                if v == SOTA_BENCH_STATIC and len(reference) == len(r.pixels):
                    var fmap = flip_error_map(reference, r.pixels, rw, rh)
                    r.flip = flip_mean(fmap)
                    var heat = flip_heat_image(fmap)
                    _ = write_png(out_dir + "/flip_" + tag + ".png", heat, rw, rh, stamped=False)
                _ = write_png(out_dir + "/" + tag + "_" + sota_bench_variant_name(v) + ".png", r.pixels, rw, rh, stamped=False)
                var flip_s = sota_fmt(r.flip, 3) if r.flip >= 0.0 else String("-")
                print("    " + _pad(sota_bench_variant_name(v), 13) + " GPU " + sota_fmt(r.gpu_ms, 2) + " ms (p95 " + sota_fmt(r.gpu_p95_ms, 2)
                      + ")  period " + sota_fmt(r.period_ms, 2) + "  CPU " + sota_fmt(r.cpu_ms, 2) + "  wait " + sota_fmt(r.wait_ms, 2) + "  VRAM " + sota_fmt(r.vram_mb, 0)
                      + " MB  FLIP " + flip_s + "  | " + r.top_passes(5))
                md += "| " + s_name + " | " + String(rh) + "p (" + String(r.render_w) + "x" + String(r.render_h) + ") | " + sota_bench_variant_name(v)
                md += " | " + sota_fmt(r.gpu_ms, 2) + " | " + sota_fmt(r.gpu_p95_ms, 2) + " | " + sota_fmt(r.period_ms, 2) + " | " + sota_fmt(r.cpu_ms, 2) + " | " + sota_fmt(r.wait_ms, 2)
                md += " | " + sota_fmt(r.vram_mb, 0) + " | " + flip_s + " | " + r.top_passes(3) + " |\n"
            if SOTA_BENCH_STATIC in variants:
                var hz = sota_bench_hiz_shadow_probe(ren, scene)
                print("    HiZ shadow-caster probe: FLIP(HiZ on vs off) = " + sota_fmt(hz[0], 4) + " (noise floor " + sota_fmt(hz[1], 4) + ")")
                md += "| " + s_name + " | " + String(rh) + "p | HiZ on vs off | | | | | | | " + sota_fmt(hz[0], 4) + " | noise floor " + sota_fmt(hz[1], 4) + " |\n"
                var on = hz[2].copy()
                _ = write_png(out_dir + "/hiz_on_" + tag + ".png", on, rw, rh, stamped=False)
            if robust:
                sota_bench_apply_preset(ren, preset, scene)
                var sun_b = sota_bench_sun(scene, SOTA_BENCH_MOVING_LIGHT, 1, 2)
                var sun_a = scene.sun_dir
                var lag = sota_bench_light_lag(ren, scene, sun_a, sun_b)
                var curve = String("")
                for c in range(min(8, len(lag.flip_curve))):
                    curve += sota_fmt(lag.flip_curve[c], 3) + " "
                print("    light cut: lag " + String(lag.lag_frames) + " frames (" + String(lag.failed_frames) + " failed renders), floor " + sota_fmt(lag.noise_floor, 4) + ", FLIP curve " + curve)
                _ = write_png(out_dir + "/lag_first_" + tag + ".png", lag.first, rw, rh, stamped=False)
                _ = write_png(out_dir + "/lag_ref_" + tag + ".png", lag.reference, rw, rh, stamped=False)
                var gh = sota_bench_motion_ghosting(ren, scene)
                print("    motion ghosting: FLIP " + sota_fmt(gh[0], 4) + " vs floor " + sota_fmt(gh[1], 4))
                var mv = gh[2].copy()
                var cv = gh[3].copy()
                _ = write_png(out_dir + "/ghost_moving_" + tag + ".png", mv, rw, rh, stamped=False)
                _ = write_png(out_dir + "/ghost_converged_" + tag + ".png", cv, rw, rh, stamped=False)
                rob += "| " + s_name + " | " + String(rh) + "p | " + String(lag.lag_frames) + " | " + sota_fmt(lag.flip_curve[0], 4) + " | " + sota_fmt(lag.noise_floor, 4)
                rob += " | " + sota_fmt(gh[0], 4) + " | " + sota_fmt(gh[1], 4) + " |\n"
            ren.shutdown()
    if robust:
        md += rob
    if preview:
        return
    var path = out_dir + "/" + (md_name if md_name.byte_length() > 0 else String("bench_") + preset_name + ".md")
    var bytes = List[UInt8](md.as_bytes())
    _ = write_bytes(path, bytes)
    print("[bench] done in " + sota_fmt(get_time() - t_all, 1) + " s -> " + path)
