"""MI-GAN fast inpainting (`flint.ml.migan`): tile seams, sprite clean-up, small object removal.

Runs Picsart's MI-GAN-512-Places2 graph (`migan.onnx` from Hugging Face
`andraniksargsyan/migan`, MIT code and weights, see `licenses/migan-LICENSE.txt`) on the Flint
ONNX interpreter. The graph takes a uint8 RGB image and a uint8 mask (255 = keep, 0 = fill),
resizes both to 512 x 512 (bilinear, half-pixel), runs the 7.4 M-parameter generator (StyleGAN-
style encoder / synthesis with depthwise 3 x 3 convolutions and 4 x 4 FIR resampling filters),
resizes back and blends the result into the hole. `MiGan.load` lowers every convolution to
`ops_conv2d` (direct depthwise, GEMM point-wise) and the full-resolution LeakyRelu / Clip /
Resize / Pad nodes to `ops_image`; one 512 x 512 pass is ~0.4 s on 20 CPU cores.

Flint's own convention: `hole` masks are one byte per pixel, non-zero = pixel to repaint;
images are 0xAARRGGBB pixels (alpha is kept, only RGB is inpainted).

  * `MiGan.inpaint(pixels, hole, w, h)` — whole image in one graph pass (fine up to ~512 px);
  * `MiGan.inpaint_windows(...)` — large images: 512 x 512 windows centred on the remaining hole
    pixels, each pass writing only its hole pixels and marking them known for the next window,
    so the repaint stays at full resolution;
  * `make_tileable(model, pixels, w, h, band)` — seamless textures: roll the image by half its size
    so the wrap seams form a cross in the middle, repaint a `band`-pixel cross, roll back. The
    result repeats without visible seams (the repainted band ends up on the borders).
"""

from flint.ml.onnx import OnnxModel
from flint.ml.tensor import Tensor, DT_U8

comptime MIGAN_RES: Int = 512


struct MiGan(Movable):
    var model: OnnxModel
    var res: Int

    def __init__(out self, var model: OnnxModel):
        self.model = model^
        self.res = MIGAN_RES

    @staticmethod
    def load(path: String, use_gpu: Bool = False, workers: Int = -1) raises -> MiGan:
        """`path` = `migan.onnx`. The CPU path is the fast one (the GPU only takes the dense
        point-wise products, which costs more in transfers than it saves at 512 x 512)."""
        var m = OnnxModel.load(path, workers)
        var names = m.input_names()
        if len(names) != 2 or names[0] != "image" or names[1] != "mask":
            raise Error("MI-GAN: expected a graph with inputs (image, mask): " + path)
        m.lower_conv2d(True)
        m.lower_image_ops()
        if use_gpu:
            _ = m.use_gpu()
        return MiGan(m^)

    def run_u8(mut self, rgb_chw: List[Int], keep: List[Int], w: Int, h: Int) raises -> List[Int]:
        """One graph pass on raw uint8 planes: image [3, h, w], mask [h, w] (255 keep / 0 fill).
        Returns the blended uint8 image [3, h, w]."""
        if len(rgb_chw) != 3 * w * h or len(keep) != w * h:
            raise Error("MI-GAN: image / mask size mismatch")
        self.model.set_input("image", Tensor.ints(DT_U8, [1, 3, h, w], rgb_chw.copy()))
        self.model.set_input("mask", Tensor.ints(DT_U8, [1, 1, h, w], keep.copy()))
        self.model.run()
        var o = self.model.output("result")
        return o.i.copy()

    def _pass(mut self, mut pixels: List[UInt32], mut hole: List[UInt8], w: Int, h: Int, x0: Int, y0: Int,
              ww: Int, wh: Int) raises -> Int:
        """Inpaint the window [x0, x0 + ww) x [y0, y0 + wh) in place; returns the pixels filled."""
        var n = ww * wh
        var img = List[Int](length=3 * n, fill=0)
        var keep = List[Int](length=n, fill=255)
        var filled = 0
        for y in range(wh):
            for x in range(ww):
                var p = pixels[(y0 + y) * w + x0 + x]
                var q = y * ww + x
                img[q] = Int((p >> 16) & 255)
                img[n + q] = Int((p >> 8) & 255)
                img[2 * n + q] = Int(p & 255)
                if hole[(y0 + y) * w + x0 + x] != 0:
                    keep[q] = 0
                    filled += 1
        if filled == 0:
            return 0
        var out = self.run_u8(img, keep, ww, wh)
        for y in range(wh):
            for x in range(ww):
                var q = y * ww + x
                var gi = (y0 + y) * w + x0 + x
                if keep[q] == 0:
                    var a = pixels[gi] & UInt32(0xFF000000)
                    pixels[gi] = a | (UInt32(out[q]) << 16) | (UInt32(out[n + q]) << 8) | UInt32(out[2 * n + q])
                    hole[gi] = 0
        return filled

    def inpaint(mut self, pixels: List[UInt32], hole: List[UInt8], w: Int, h: Int) raises -> List[UInt32]:
        """Repaint the `hole` pixels of a w x h image in one pass (the graph works at 512 x 512)."""
        var px = pixels.copy()
        var hl = hole.copy()
        _ = self._pass(px, hl, w, h, 0, 0, w, h)
        return px^

    def inpaint_windows(mut self, pixels: List[UInt32], hole: List[UInt8], w: Int, h: Int,
                        window: Int = MIGAN_RES) raises -> List[UInt32]:
        """Full-resolution repaint of a large image: windows of `window` px (clamped to the image)
        centred on the hole pixels that remain, in scan order."""
        var px = pixels.copy()
        var hl = hole.copy()
        var ww = min(window, w)
        var wh = min(window, h)
        var start = 0
        while True:
            var first = -1
            for i in range(start, w * h):
                if hl[i] != 0:
                    first = i
                    break
            if first < 0:
                break
            start = first
            var fx = first % w
            var fy = first // w
            # candidate window around the first remaining hole pixel, then re-centred on the
            # bounding box of the hole pixels it contains
            var cx0 = min(max(fx - ww // 2, 0), w - ww)
            var cy0 = min(max(fy, 0), h - wh)
            var bx0 = w
            var by0 = h
            var bx1 = -1
            var by1 = -1
            for y in range(cy0, cy0 + wh):
                for x in range(cx0, cx0 + ww):
                    if hl[y * w + x] != 0:
                        bx0 = min(bx0, x)
                        bx1 = max(bx1, x)
                        by0 = min(by0, y)
                        by1 = max(by1, y)
            var x0 = min(max((bx0 + bx1) // 2 - ww // 2, 0), w - ww)
            var y0 = min(max((by0 + by1) // 2 - wh // 2, 0), h - wh)
            # the first pixel must stay inside so every pass makes progress
            if fx < x0 or fx >= x0 + ww:
                x0 = cx0
            if fy < y0 or fy >= y0 + wh:
                y0 = cy0
            _ = self._pass(px, hl, w, h, x0, y0, ww, wh)
        return px^


def roll_pixels(pixels: List[UInt32], w: Int, h: Int, dx: Int, dy: Int) -> List[UInt32]:
    """Circular shift: out[y][x] = in[(y + dy) mod h][(x + dx) mod w]."""
    var out = List[UInt32](length=w * h, fill=0)
    for y in range(h):
        var sy = ((y + dy) % h + h) % h
        for x in range(w):
            var sx = ((x + dx) % w + w) % w
            out[y * w + x] = pixels[sy * w + sx]
    return out^


def seam_cross_mask(w: Int, h: Int, band: Int) -> List[UInt8]:
    """Hole mask of the wrap seams after `roll_pixels(.., w // 2, h // 2)`: a `band`-wide cross
    centred on column w - w // 2 and row h - h // 2."""
    var m = List[UInt8](length=w * h, fill=0)
    var cx = w - w // 2
    var cy = h - h // 2
    var half = max(band, 1) // 2
    var lo_x = cx - half
    var hi_x = cx + (max(band, 1) - half)
    var lo_y = cy - half
    var hi_y = cy + (max(band, 1) - half)
    for y in range(h):
        for x in range(w):
            if (x >= lo_x and x < hi_x) or (y >= lo_y and y < hi_y):
                m[y * w + x] = 1
    return m^


def wrap_seam_error(pixels: List[UInt32], w: Int, h: Int) -> Float64:
    """Mean absolute RGB step across the wrap seams (right -> left column, bottom -> top row)
    divided by the mean step between interior neighbours: ~1 for a seamless tile, larger with a
    visible seam."""
    var seam: Float64 = 0.0
    var inner: Float64 = 0.0
    var ns = 0
    var ni = 0
    for y in range(h):
        for x in range(w):
            var p = pixels[y * w + x]
            var r = pixels[y * w + (x + 1) % w]
            var d = pixels[((y + 1) % h) * w + x]
            var dr: Float64 = 0.0
            var dd: Float64 = 0.0
            for s in range(3):
                var sh = UInt32(16 - 8 * s)
                var a = Int((p >> sh) & 255)
                dr += Float64(abs(a - Int((r >> sh) & 255)))
                dd += Float64(abs(a - Int((d >> sh) & 255)))
            if x == w - 1:
                seam += dr
                ns += 1
            else:
                inner += dr
                ni += 1
            if y == h - 1:
                seam += dd
                ns += 1
            else:
                inner += dd
                ni += 1
    var mi = inner / Float64(max(ni, 1))
    return (seam / Float64(max(ns, 1))) / max(mi, 1e-6)


def make_tileable(mut model: MiGan, pixels: List[UInt32], w: Int, h: Int, band: Int = 32,
                  realign: Bool = True) raises -> List[UInt32]:
    """Seamless tiling: roll by half, MI-GAN-repaint the `band`-px seam cross, roll back
    (`realign = False` keeps the rolled layout with the repaint in the middle)."""
    var dx = w // 2
    var dy = h // 2
    var rolled = roll_pixels(pixels, w, h, dx, dy)
    var hole = seam_cross_mask(w, h, band)
    var fixed = model.inpaint_windows(rolled, hole, w, h)
    if not realign:
        return fixed^
    return roll_pixels(fixed, w, h, -dx, -dy)
