"""Fast inpainting with MI-GAN (pure Mojo ONNX interpreter): sprite clean-up and seamless tiles.

usage: migan MIGAN.onnx IN.png OUT.png --mask MASK.png [--window N] [--one-pass] [--gpu]
       migan MIGAN.onnx IN.png OUT.png --tile [--band N] [--keep-rolled] [--gpu]

MIGAN.onnx comes from `bash tools/migan/fetch.sh DIR` (MIT, Picsart AI Research).
--mask: white (> 50 % grey) mask pixels are repainted; the mask must match the image size. Images
  larger than 512 px are repainted in 512 x 512 windows around the hole at full resolution
  (--window sets the size; --one-pass sends the whole image through the 512 x 512 graph at once).
--tile: makes the texture tile seamlessly: the wrap seams are moved to the middle, a --band px
  (default 32) cross over them is repainted and the image is rolled back (--keep-rolled leaves
  the repaint in the middle). Prints the wrap-seam step ratio before / after (~1 = seamless).
Alpha is kept as it is; only RGB is repainted.
"""
from std.sys import argv
from std.time import perf_counter_ns
from flint.platform.os import read_bytes
from flint.assets.jpeg import decode_image_auto
from flint.assets.image import write_png
from flint.ml.migan import MiGan, make_tileable, wrap_seam_error


def main() raises:
    var a = argv()
    if len(a) < 5:
        print("usage: migan MIGAN.onnx IN.png OUT.png --mask MASK.png [--window N] [--one-pass] [--gpu]")
        print("       migan MIGAN.onnx IN.png OUT.png --tile [--band N] [--keep-rolled] [--gpu]")
        return
    var model = String(a[1])
    var src = String(a[2])
    var dst = String(a[3])
    var mask_path = String()
    var tile = False
    var band = 32
    var window = 512
    var one_pass = False
    var keep_rolled = False
    var gpu = False
    var i = 4
    while i < len(a):
        var s = String(a[i])
        var has = i + 1 < len(a)
        if s == "--mask" and has:
            mask_path = String(a[i + 1])
            i += 1
        elif s == "--band" and has:
            band = Int(String(a[i + 1]))
            i += 1
        elif s == "--window" and has:
            window = Int(String(a[i + 1]))
            i += 1
        elif s == "--tile":
            tile = True
        elif s == "--one-pass":
            one_pass = True
        elif s == "--keep-rolled":
            keep_rolled = True
        elif s == "--gpu":
            gpu = True
        i += 1
    var img = decode_image_auto(read_bytes(src), 8192)
    if not img.valid:
        raise Error("migan: cannot decode " + src)
    var w = img.width
    var h = img.height
    var t0 = perf_counter_ns()
    var mg = MiGan.load(model, gpu)
    print("[migan] loaded in", Float64(perf_counter_ns() - t0) / 1e9, "s")
    var out: List[UInt32]
    t0 = perf_counter_ns()
    if tile:
        print("[migan] wrap-seam step ratio before:", wrap_seam_error(img.pixels, w, h))
        out = make_tileable(mg, img.pixels, w, h, band, not keep_rolled)
        print("[migan] wrap-seam step ratio after: ", wrap_seam_error(out, w, h))
    else:
        if mask_path.byte_length() == 0:
            raise Error("migan: pass --mask MASK.png or --tile")
        var mk = decode_image_auto(read_bytes(mask_path), 8192)
        if not mk.valid or mk.width != w or mk.height != h:
            raise Error("migan: the mask must be an image of the same size as the input")
        var hole = List[UInt8](length=w * h, fill=0)
        var n = 0
        for p in range(w * h):
            var q = mk.pixels[p]
            var lum = (Int((q >> 16) & 255) * 299 + Int((q >> 8) & 255) * 587 + Int(q & 255) * 114) // 1000
            if lum > 127:
                hole[p] = 1
                n += 1
        print("[migan] repainting", n, "pixels")
        if one_pass:
            out = mg.inpaint(img.pixels, hole, w, h)
        else:
            out = mg.inpaint_windows(img.pixels, hole, w, h, window)
    print("[migan]", w, "x", h, "done in", Float64(perf_counter_ns() - t0) / 1e9, "s")
    if not write_png(dst, out, w, h):
        raise Error("migan: cannot write " + dst)
    print("[migan] wrote", dst)
