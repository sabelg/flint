"""FLIP: a perceptual difference between a rendered frame and its reference.

Package 0.12 adds this to the screenshot gate beside the per-channel tolerance.
It is LDR-FLIP (Andersson, Nilsson, Akenine-Moller, Oskarsson, Astrom and
Fairchild, "FLIP: A Difference Evaluator for Alternating Images", HPG 2020),
written from the paper:

1. Colour. Both images go from sRGB to the opponent space YCxCz, are blurred
   with the contrast sensitivity of the eye for the achromatic, red-green and
   blue-yellow channels (Gaussians whose widths are in degrees of view, so
   the viewing distance matters: `ppd`, pixels per degree), go back to linear
   RGB clamped to [0, 1] and on to CIELab with the Hunt adjustment (a and b
   scaled by L/100). The colour difference is the HyAB distance raised to
   0.7, remapped so that differences below 40% of the largest possible one
   (green against blue) are compressed into [0, 0.95].
2. Features. On the luminance, first and second derivatives of a Gaussian
   find edges and points; the feature difference is the larger change in
   edge or point strength, scaled by 1/sqrt(2) and raised to 0.5.
3. The error of a pixel is colour ^ (1 - feature): a small colour change on
   an edge counts for more.

The result is one number per pixel in [0, 1]; the gate reports the mean
(`flip_mean`) and writes the per-pixel map as a heat map with the magma
colour map (`flip_heat_image`). The default ppd is the paper's: a 0.7 m wide
3840-pixel monitor seen from 0.7 m, 67 pixels per degree.

This is presentation code (a test tool): it may use the standard library's
`pow`, `exp` and `sqrt`; nothing here feeds the simulation.
"""
from std.math import exp, sqrt, cbrt, ceil

comptime FLIP_PPD: Float32 = 67.02064
comptime PI32: Float32 = 3.14159265358979
comptime QC: Float32 = 0.7
comptime QF: Float32 = 0.5
comptime PC: Float32 = 0.4
comptime PT: Float32 = 0.95
comptime FEATURE_WIDTH_DEGREES: Float32 = 0.082
# D65 reference white in XYZ.
comptime WHITE_X: Float32 = 0.950428545
comptime WHITE_Y: Float32 = 1.0
comptime WHITE_Z: Float32 = 1.088900371


@always_inline
def srgb_to_linear(c: Float32) -> Float32:
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


@always_inline
def channel(pixel: UInt32, shift: Int) -> Float32:
    return Float32(Int((pixel >> UInt32(shift)) & 255)) / 255.0


@always_inline
def lab_f(t: Float32) -> Float32:
    var delta = Float32(6.0 / 29.0)
    if t > delta * delta * delta:
        return cbrt(t)
    return t / (3.0 * delta * delta) + Float32(4.0 / 29.0)


struct Lab(Copyable, ImplicitlyCopyable, Movable):
    var l: Float32
    var a: Float32
    var b: Float32

    def __init__(out self, l: Float32, a: Float32, b: Float32):
        self.l = l
        self.a = a
        self.b = b


def linear_rgb_to_hunt_lab(r: Float32, g: Float32, b: Float32) -> Lab:
    var x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / WHITE_X
    var y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / WHITE_Y
    var z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / WHITE_Z
    var fx = lab_f(x)
    var fy = lab_f(y)
    var fz = lab_f(z)
    var l = 116.0 * fy - 16.0
    var a = 500.0 * (fx - fy)
    var bb = 200.0 * (fy - fz)
    # Hunt: colourfulness grows with lightness.
    return Lab(l, 0.01 * l * a, 0.01 * l * bb)


@always_inline
def hyab(p: Lab, q: Lab) -> Float32:
    var da = p.a - q.a
    var db = p.b - q.b
    return abs(p.l - q.l) + sqrt(da * da + db * db)


def csf_kernel(a1: Float32, b1: Float32, a2: Float32, b2: Float32, ppd: Float32) -> List[Float32]:
    """One axis of a contrast-sensitivity filter, normalised to sum 1. The 2D
    filter is a sum of at most two Gaussians, each separable; the widths are
    in degrees, so a pixel is 1/ppd of a degree."""
    var widest = max(b1, b2)
    var radius = Int(ceil(3.0 * sqrt(widest / (2.0 * PI32 * PI32)) * ppd))
    var kernel = List[Float32]()
    var total = Float32(0)
    for offset in range(-radius, radius + 1):
        var x = Float32(offset) / ppd
        var value = a1 * sqrt(PI32 / b1) * exp(-PI32 * PI32 * x * x / b1)
        if a2 > 0:
            value += a2 * sqrt(PI32 / b2) * exp(-PI32 * PI32 * x * x / b2)
        kernel.append(value)
        total += value
    for index in range(len(kernel)):
        kernel[index] /= total
    return kernel^


def gaussian_kernel(a: Float32, b: Float32, ppd: Float32, radius: Int) -> List[Float32]:
    """a sqrt(pi/b) exp(-pi^2 x^2 / b) on one axis, NOT normalised: a sum of
    two of these keeps their relative weight."""
    var kernel = List[Float32]()
    for offset in range(-radius, radius + 1):
        var x = Float32(offset) / ppd
        kernel.append(a * sqrt(PI32 / b) * exp(-PI32 * PI32 * x * x / b))
    return kernel^


def blur_rows(source: List[Float32], width: Int, height: Int, kernel: List[Float32]) -> List[Float32]:
    """Convolution along x with clamped edges."""
    var radius = len(kernel) // 2
    var out = List[Float32](length=width * height, fill=0)
    for y in range(height):
        var row = y * width
        for x in range(width):
            var total = Float32(0)
            for k in range(len(kernel)):
                var sx = min(width - 1, max(0, x + k - radius))
                total += kernel[k] * source[row + sx]
            out[row + x] = total
    return out^


def blur_columns(source: List[Float32], width: Int, height: Int, kernel: List[Float32]) -> List[Float32]:
    var radius = len(kernel) // 2
    var out = List[Float32](length=width * height, fill=0)
    for y in range(height):
        for x in range(width):
            var total = Float32(0)
            for k in range(len(kernel)):
                var sy = min(height - 1, max(0, y + k - radius))
                total += kernel[k] * source[sy * width + x]
            out[y * width + x] = total
    return out^


def separable(source: List[Float32], width: Int, height: Int, along_x: List[Float32], along_y: List[Float32]) -> List[Float32]:
    return blur_columns(blur_rows(source, width, height, along_x), width, height, along_y)


struct Opponent(Movable):
    """An image in YCxCz, one plane per channel, and its luminance in [0, 1]."""
    var y: List[Float32]
    var cx: List[Float32]
    var cz: List[Float32]
    var luminance: List[Float32]

    def __init__(out self, pixels: List[UInt32], count: Int):
        self.y = List[Float32](length=count, fill=0)
        self.cx = List[Float32](length=count, fill=0)
        self.cz = List[Float32](length=count, fill=0)
        self.luminance = List[Float32](length=count, fill=0)
        for index in range(count):
            var p = pixels[index]
            var r = srgb_to_linear(channel(p, 16))
            var g = srgb_to_linear(channel(p, 8))
            var b = srgb_to_linear(channel(p, 0))
            var x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / WHITE_X
            var yy = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / WHITE_Y
            var z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / WHITE_Z
            self.y[index] = 116.0 * yy - 16.0
            self.cx[index] = 500.0 * (x - yy)
            self.cz[index] = 200.0 * (yy - z)
            self.luminance[index] = yy


def filtered_lab(image: Opponent, width: Int, height: Int, ppd: Float32) -> List[Lab]:
    """The colour pipeline up to Hunt-adjusted Lab, after the CSF blur."""
    var achromatic = csf_kernel(1.0, 0.0047, 0.0, 0.00001, ppd)
    var red_green = csf_kernel(1.0, 0.0053, 0.0, 0.00001, ppd)
    # Blue-yellow is two Gaussians: blur with each and add, normalised together.
    var by_radius = Int(ceil(3.0 * sqrt(Float32(0.04) / (2.0 * PI32 * PI32)) * ppd))
    var by_first = gaussian_kernel(34.1, 0.04, ppd, by_radius)
    var by_second = gaussian_kernel(13.5, 0.025, ppd, by_radius)
    var total = Float32(0)
    for i in range(len(by_first)):
        for j in range(len(by_first)):
            total += by_first[i] * by_first[j] + by_second[i] * by_second[j]
    var scale = sqrt(total)
    for i in range(len(by_first)):
        by_first[i] /= scale
        by_second[i] /= scale
    var y = separable(image.y, width, height, achromatic, achromatic)
    var cx = separable(image.cx, width, height, red_green, red_green)
    var cz_a = separable(image.cz, width, height, by_first, by_first)
    var cz_b = separable(image.cz, width, height, by_second, by_second)
    var count = width * height
    var out = List[Lab](capacity=count)
    for index in range(count):
        var fy = (y[index] + 16.0) / 116.0
        var fx = fy + cx[index] / 500.0
        var fz = fy - (cz_a[index] + cz_b[index]) / 200.0
        var xx = fx * WHITE_X
        var yy = fy * WHITE_Y
        var zz = fz * WHITE_Z
        var r = min(Float32(1), max(Float32(0), 3.2404542 * xx - 1.5371385 * yy - 0.4985314 * zz))
        var g = min(Float32(1), max(Float32(0), -0.9692660 * xx + 1.8760108 * yy + 0.0415560 * zz))
        var b = min(Float32(1), max(Float32(0), 0.0556434 * xx - 0.2040259 * yy + 1.0572252 * zz))
        out.append(linear_rgb_to_hunt_lab(r, g, b))
    return out^


def normalise_signed(mut kernel: List[Float32]):
    """Positive weights to sum 1, negative ones to sum -1."""
    var positive = Float32(0)
    var negative = Float32(0)
    for value in kernel:
        if value > 0:
            positive += value
        else:
            negative -= value
    for index in range(len(kernel)):
        if kernel[index] > 0:
            kernel[index] /= positive
        elif negative > 0:
            kernel[index] /= negative


def feature_kernels(ppd: Float32) -> Tuple[List[Float32], List[Float32], List[Float32]]:
    """One axis of the edge (first derivative) and point (second derivative)
    detectors and the plain Gaussian for the other axis. Positive weights sum
    to 1 and negative ones to -1; the Gaussian sums to 1."""
    var sigma = 0.5 * FEATURE_WIDTH_DEGREES * ppd
    var radius = Int(ceil(3.0 * sigma))
    var gauss = List[Float32]()
    var edge = List[Float32]()
    var point = List[Float32]()
    var g_total = Float32(0)
    for offset in range(-radius, radius + 1):
        var x = Float32(offset)
        var g = exp(-(x * x) / (2.0 * sigma * sigma))
        gauss.append(g)
        g_total += g
        edge.append(-x * g)
        point.append((x * x / (sigma * sigma) - 1.0) * g)
    for index in range(len(gauss)):
        gauss[index] /= g_total
    normalise_signed(edge)
    normalise_signed(point)
    return (gauss^, edge^, point^)


def feature_strengths(luminance: List[Float32], width: Int, height: Int, ppd: Float32) -> Tuple[List[Float32], List[Float32]]:
    var kernels = feature_kernels(ppd)
    ref gauss = kernels[0]
    ref edge = kernels[1]
    ref point = kernels[2]
    var edge_x = separable(luminance, width, height, edge, gauss)
    var edge_y = separable(luminance, width, height, gauss, edge)
    var point_x = separable(luminance, width, height, point, gauss)
    var point_y = separable(luminance, width, height, gauss, point)
    var count = width * height
    var edges = List[Float32](length=count, fill=0)
    var points = List[Float32](length=count, fill=0)
    for index in range(count):
        edges[index] = sqrt(edge_x[index] * edge_x[index] + edge_y[index] * edge_y[index])
        points[index] = sqrt(point_x[index] * point_x[index] + point_y[index] * point_y[index])
    return (edges^, points^)


def flip_error_map(reference: List[UInt32], test: List[UInt32], width: Int, height: Int, ppd: Float32 = FLIP_PPD) -> List[Float32]:
    """Per-pixel FLIP of `test` against `reference` (0xAARRGGBB), in [0, 1].
    Empty when the sizes do not match."""
    var count = width * height
    if count <= 0 or len(reference) < count or len(test) < count:
        return List[Float32]()
    var ref_image = Opponent(reference, count)
    var test_image = Opponent(test, count)
    var ref_lab = filtered_lab(ref_image, width, height, ppd)
    var test_lab = filtered_lab(test_image, width, height, ppd)
    var ref_features = feature_strengths(ref_image.luminance, width, height, ppd)
    var test_features = feature_strengths(test_image.luminance, width, height, ppd)

    var cmax = hyab(linear_rgb_to_hunt_lab(0, 1, 0), linear_rgb_to_hunt_lab(0, 0, 1)) ** QC
    var knee = PC * cmax
    var inverse_sqrt2 = Float32(0.70710678)
    var out = List[Float32](length=count, fill=0)
    for index in range(count):
        var colour = hyab(ref_lab[index], test_lab[index]) ** QC
        if colour < knee:
            colour = PT / knee * colour
        else:
            colour = PT + (colour - knee) / (cmax - knee) * (1.0 - PT)
        var edge = abs(ref_features[0][index] - test_features[0][index])
        var point = abs(ref_features[1][index] - test_features[1][index])
        var feature = (inverse_sqrt2 * max(edge, point)) ** QF
        out[index] = min(Float32(1), max(Float32(0), colour ** (1.0 - feature)))
    return out^


def flip_mean(error_map: List[Float32]) -> Float64:
    if len(error_map) == 0:
        return 1.0
    var total = Float64(0)
    for value in error_map:
        total += Float64(value)
    return total / Float64(len(error_map))


def magma(t_in: Float32) -> UInt32:
    """The magma colour map (a polynomial fit), 0 black-purple to 1 pale yellow."""
    var t = min(Float32(1), max(Float32(0), t_in))
    var r = -0.002136485053939582 + t * (0.2516605407371642 + t * (8.353717279216625 + t * (-27.66873308576866 + t * (52.17613981234068 + t * (-50.76852536473588 + t * 18.65570506591883)))))
    var g = -0.000749655052795221 + t * (0.6775232436837668 + t * (-3.577719514958484 + t * (14.26473078096533 + t * (-27.94360607168351 + t * (29.04658282127291 + t * -11.48977351997711)))))
    var b = -0.005386127855323933 + t * (2.494026599312351 + t * (0.3144679030132573 + t * (-13.64921318813922 + t * (12.94416944238394 + t * (4.23415299384598 + t * -5.601961508734096)))))
    var ri = UInt32(Int(min(Float32(1), max(Float32(0), Float32(r))) * 255.0 + 0.5))
    var gi = UInt32(Int(min(Float32(1), max(Float32(0), Float32(g))) * 255.0 + 0.5))
    var bi = UInt32(Int(min(Float32(1), max(Float32(0), Float32(b))) * 255.0 + 0.5))
    return 0xFF000000 | (ri << 16) | (gi << 8) | bi


def flip_heat_image(error_map: List[Float32]) -> List[UInt32]:
    var out = List[UInt32](capacity=len(error_map))
    for value in error_map:
        out.append(magma(value))
    return out^
