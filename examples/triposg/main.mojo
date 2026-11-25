"""TripoSG image-to-3D CLI: one PNG in, a watertight mesh GLB out (optionally auto-rigged).

usage: mojo run -I src examples/triposg/main.mojo MODEL_DIR in.png out.glb
           [--steps 50] [--guidance 7.0] [--seed 42] [--res 512] [--noise FILE]
           [--rig UNIRIG_DIR [--rig-seed N]] [--preview out.png] [--latents FILE] [--save-latents FILE]

MODEL_DIR is the Hugging Face VAST-AI/TripoSG folder (image_encoder_dinov2/, transformer/, vae/).
Images with a real alpha channel are used as is; otherwise a plain background is keyed out.
`--rig` pipes the mesh into UniRig (graphs from `tools/unirig/export.py`) and writes a skinned GLB.
"""
from std.sys import argv
from std.time import perf_counter_ns
from flint.assets.glb_writer import write_native_glb
from flint.ml.triposg import (
    TripoSG, load_rgba_png, prepare_image, preprocess_dinov2, torch_randn, read_f32, write_f32, extract_sdf,
    TG_TOKENS, TG_LATENT, TG_BOUND,
)
from flint.ml.triposg_mesh import marching_cubes, mesh_to_model, boundary_edges, render_preview
from flint.assets.image import write_png
from flint.ml.unirig import UrGenConfig
from flint.ml.unirig_rig import UniRig, rig_native_model


def _secs(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e9


def main() raises:
    var a = argv()
    if len(a) < 4:
        print("usage: triposg MODEL_DIR in.png out.glb [--steps N] [--guidance G] [--seed S] [--res R]"
              " [--noise FILE] [--rig UNIRIG_DIR] [--rig-seed N]")
        return
    var steps = 50
    var guidance: Float32 = 7.0
    var seed = 42
    var res = 512
    var noise_file = String("")
    var rig_dir = String("")
    var rig_seed = 0
    var preview = String("")
    var latents_file = String("")
    var save_latents = String("")
    var i = 4
    while i < len(a):
        var s = String(a[i])
        if i + 1 < len(a):
            var v = String(a[i + 1])
            if s == "--steps":
                steps = Int(v)
                i += 1
            elif s == "--guidance":
                guidance = Float32(Float64(v))
                i += 1
            elif s == "--seed":
                seed = Int(v)
                i += 1
            elif s == "--res":
                res = Int(v)
                i += 1
            elif s == "--noise":
                noise_file = v
                i += 1
            elif s == "--rig":
                rig_dir = v
                i += 1
            elif s == "--save-latents":
                save_latents = v
                i += 1
            elif s == "--latents":
                latents_file = v
                i += 1
            elif s == "--preview":
                preview = v
                i += 1
            elif s == "--rig-seed":
                rig_seed = Int(v)
                i += 1
        i += 1
    var t_all = perf_counter_ns()
    var t0 = perf_counter_ns()
    var model = TripoSG.load(String(a[1]))
    model.verbose = True
    print("[triposg] weights uploaded to", model.device(), "in", _secs(t0), "s")

    var latents: List[Float32]
    if latents_file != "":
        latents = read_f32(latents_file)
    else:
        t0 = perf_counter_ns()
        var img = prepare_image(load_rgba_png(String(a[2])))
        var pixels = preprocess_dinov2(img)
        var cond = model.encode_image(pixels)
        print("[triposg] image", img.width, "x", img.height, "encoded in", _secs(t0), "s")

        var noise: List[Float32]
        if noise_file != "":
            noise = read_f32(noise_file)
        else:
            noise = torch_randn(TG_TOKENS * TG_LATENT, UInt64(seed))
        t0 = perf_counter_ns()
        latents = model.sample(cond, noise^, steps, guidance)
        print("[triposg]", steps, "flow steps in", _secs(t0), "s")
        if save_latents != "":
            write_f32(save_latents, latents)

    t0 = perf_counter_ns()
    model.set_latents(latents)
    var grid = extract_sdf(model, res)
    print("[triposg] SDF grid", grid.res, "^3:", grid.queried, "decoder queries in", _secs(t0), "s")
    t0 = perf_counter_ns()
    var mesh = marching_cubes(grid.values, grid.res + 1, -TG_BOUND, 2.0 * TG_BOUND / Float32(grid.res))
    print("[triposg] marching cubes:", mesh.vertex_count(), "vertices,", mesh.triangle_count(), "triangles in",
          _secs(t0), "s; open edges:", boundary_edges(mesh))
    if preview != "":
        # front and three-quarter views side by side
        var w = 512
        var front = render_preview(mesh, w, w, 0.0)
        var side = render_preview(mesh, w, w, 40.0)
        var both = List[UInt32](length=2 * w * w, fill=0)
        for y in range(w):
            for x in range(w):
                both[y * 2 * w + x] = front[y * w + x]
                both[y * 2 * w + w + x] = side[y * w + x]
        _ = write_png(preview, both, 2 * w, w, False)
        print("[triposg] preview", preview)
    var out = mesh_to_model(mesh)
    if rig_dir != "":
        t0 = perf_counter_ns()
        var rig = UniRig.load(rig_dir, True)
        var cfg = UrGenConfig()
        cfg.seed = UInt64(rig_seed)
        out = rig_native_model(rig, out, cfg)
        print("[triposg] rigged by UniRig:", len(out.skins[0].joints), "joints in", _secs(t0), "s")
    if not write_native_glb(String(a[3]), out):
        raise Error("could not write " + String(a[3]))
    print("[triposg] wrote", String(a[3]), "(total", _secs(t_all), "s)")
