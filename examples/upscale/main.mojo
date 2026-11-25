"""Image upscaling CLI: pixel-art magnifiers and Real-ESRGAN.

usage: flint-upscale IN.png OUT.png [--filter F] [--times N] [--palette-lock] [--vulkan]
                     [--model NET.safetensors [--blend WDN.safetensors --denoise S]
                      [--tile T] [--tile-pad P] [--cpu]] [--bench N]

  --filter   nearest | scale2x | epx | scale3x | scale4x | mmpx | superxbr   (default mmpx)
             Every one but superxbr keeps the exact input palette; --palette-lock makes
             Super-xBR snap to it as well. --times applies the filter N times (2x -> 4x -> 8x).
  --vulkan   run the filter as SPIR-V compute on the first Vulkan device (same pixels).
  --model    a Real-ESRGAN checkpoint converted by tools/upscale/convert.py (fetch.sh) instead of
             a filter: RealESRGAN_x4plus, RealESRGAN_x4plus_anime_6B, realesr-general-x4v3,
             realesr-animevideov3. --blend + --denoise S interpolate realesr-general-x4v3 with
             realesr-general-wdn-x4v3 (S = 1: no extra denoising, 0: full). --tile splits the
             input into T x T tiles with P (default 10) context pixels. --cpu skips the MAX GPU.
  --bench N  time N further runs and print the mean.

Reads 8 / 16-bit grey, RGB, palette and alpha PNGs; writes RGBA PNG.
"""
from std.sys import argv
from std.time import perf_counter_ns
from flint.platform.os import write_bytes
from flint.assets.image import NativeImage, encode_png
from flint.assets.png_any import load_png_any
from flint.render.pixel_scale import filter_from_name, pixel_scale, filter_factor
from flint.render.pixel_scale_vk import PixelScaleVulkan
from flint.ml.realesrgan import RealEsrgan


def _ms(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e6


def _usage():
    print("usage: flint-upscale IN.png OUT.png [--filter nearest|scale2x|epx|scale3x|scale4x|mmpx|superxbr]"
          " [--times N] [--palette-lock] [--vulkan] [--model NET.safetensors [--blend WDN.safetensors"
          " --denoise S] [--tile T] [--tile-pad P] [--cpu]] [--bench N]")


def main() raises:
    var a = argv()
    if len(a) < 3:
        _usage()
        return
    var inp = String(a[1])
    var outp = String(a[2])
    var filter_name = String("mmpx")
    var times = 1
    var lock = False
    var vulkan = False
    var model = String("")
    var blend = String("")
    var denoise: Float32 = 1.0
    var tile = 0
    var tile_pad = 10
    var cpu = False
    var bench = 0
    var i = 3
    while i < len(a):
        var s = String(a[i])
        var has_v = i + 1 < len(a)
        if s == "--palette-lock":
            lock = True
        elif s == "--vulkan":
            vulkan = True
        elif s == "--cpu":
            cpu = True
        elif has_v and s == "--filter":
            filter_name = String(a[i + 1])
            i += 1
        elif has_v and s == "--times":
            times = Int(String(a[i + 1]))
            i += 1
        elif has_v and s == "--model":
            model = String(a[i + 1])
            i += 1
        elif has_v and s == "--blend":
            blend = String(a[i + 1])
            i += 1
        elif has_v and s == "--denoise":
            denoise = Float32(Float64(String(a[i + 1])))
            i += 1
        elif has_v and s == "--tile":
            tile = Int(String(a[i + 1]))
            i += 1
        elif has_v and s == "--tile-pad":
            tile_pad = Int(String(a[i + 1]))
            i += 1
        elif has_v and s == "--bench":
            bench = Int(String(a[i + 1]))
            i += 1
        else:
            print("unknown argument " + s)
            _usage()
            return
        i += 1
    var img = load_png_any(inp)
    print("[upscale] " + inp + ": " + String(img.width) + " x " + String(img.height))
    var out = NativeImage()
    if model.byte_length() > 0:
        var t0 = perf_counter_ns()
        var net = RealEsrgan.load(model, not cpu, blend, denoise)
        print("[upscale] " + net.arch_name() + " x" + String(net.scale) + " on " + net.device_name() + ", loaded in "
              + String(_ms(t0)) + " ms")
        t0 = perf_counter_ns()
        out = net.upscale_image(img, tile, tile_pad)
        print("[upscale] first run " + String(_ms(t0)) + " ms (includes kernel compilation on a GPU)")
        if bench > 0:
            t0 = perf_counter_ns()
            for _ in range(bench):
                out = net.upscale_image(img, tile, tile_pad)
            var per = _ms(t0) / Float64(bench)
            print("[upscale] mean " + String(per) + " ms per image, "
                  + String(Float64(img.width * img.height) / (per * 1e3)) + " input Mpx/s")
    else:
        var f = filter_from_name(filter_name)
        if f < 0:
            print("unknown filter " + filter_name)
            return
        var vk = PixelScaleVulkan()
        if vulkan and not vk.open():
            raise Error("vulkan: " + vk.error)
        var runs = 1 + bench
        var t_first: Float64 = 0
        var t_rest: Float64 = 0
        for r in range(runs):
            var t0 = perf_counter_ns()
            var cur = img.copy()
            for _ in range(times):
                if vulkan:
                    cur = vk.run(cur, f, lock)
                else:
                    cur = pixel_scale(cur, f, lock)
            if r == 0:
                t_first = _ms(t0)
            else:
                t_rest += _ms(t0)
            out = cur^
        var dev = vk.device_name() if vulkan else String("CPU")
        print("[upscale] " + filter_name + " x" + String(filter_factor(f)) + " (" + String(times) + " times) on " + dev
              + ": " + String(t_first) + " ms")
        if bench > 0:
            print("[upscale] mean " + String(t_rest / Float64(bench)) + " ms per run (incl. upload / readback)")
    var px = out.pixels.copy()
    var data = encode_png(px, out.width, out.height, "", True)
    if not write_bytes(outp, data):
        raise Error("cannot write " + outp)
    print("[upscale] wrote " + outp + ": " + String(out.width) + " x " + String(out.height))
