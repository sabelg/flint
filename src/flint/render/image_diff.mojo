"""Compare a rendered frame with a reference image, for the regression gate.

A pixel counts as different when any of its three channels differs by more than
`tolerance` levels (0-255). A frame passes when the two images have the same
size and at most `spread_per_million` of its pixels are different; how far those
pixels are off (`worst`) is reported but not limited. The difference image shows
each channel's difference multiplied by 16 on black, so an exact match is a
black image.

The second number is FLIP (render/flip.mojo, package 0.12): the mean of a
perceptual per-pixel difference in [0, 1]. A frame also fails when its mean
FLIP is above the limit given to `passes` (the gate uses SHOT_FLIP_LIMIT), so
a change spread thinly over many pixels, each within the channel tolerance,
still fails. `flip_map` holds the per-pixel values for the heat map
(`flip_heat_image`); it is empty when the images are identical. Mean absolute
channel difference is kept in `mean_abs`.
"""
from flint.assets.image import decode_png
from flint.render.flip import flip_error_map, flip_mean
from flint.platform.os import read_bytes


struct ImageDiff(Copyable, Movable):
    var width: Int
    var height: Int
    var same_size: Bool
    var differing: Int        # pixels with any channel difference at all
    var over_tolerance: Int   # pixels with a channel difference above the tolerance
    var worst: Int            # largest channel difference anywhere
    var mean_abs: Float64     # mean absolute channel difference, 0-255
    var flip_mean: Float64    # mean FLIP, 0-1 (0 when identical)
    var flip_map: List[Float32]  # per-pixel FLIP; empty when identical
    var image: List[UInt32]   # the difference picture, 0xFFRRGGBB

    def __init__(out self):
        self.width = 0
        self.height = 0
        self.same_size = False
        self.differing = 0
        self.over_tolerance = 0
        self.worst = 0
        self.mean_abs = 0
        self.flip_mean = 0
        self.flip_map = List[Float32]()
        self.image = List[UInt32]()

    def passes(self, spread_per_million: Int, flip_limit: Float64 = 1.0) -> Bool:
        if not self.same_size:
            return False
        if self.flip_mean > flip_limit:
            return False
        return self.over_tolerance * 1000000 <= self.width * self.height * spread_per_million

    def summary(self) -> String:
        if not self.same_size:
            return "size_mismatch"
        var share = Float64(self.over_tolerance) * 100.0 / Float64(max(1, self.width * self.height))
        return "differing=" + String(self.differing) + " over_tolerance=" + String(self.over_tolerance) + " (" + String(share) + "%) worst=" + String(self.worst) + " flip_mean=" + String(self.flip_mean) + " mean_abs=" + String(self.mean_abs)


def compare_pixels(actual: List[UInt32], expected: List[UInt32], width: Int, height: Int, tolerance: Int) -> ImageDiff:
    var diff = ImageDiff()
    diff.width = width
    diff.height = height
    if width <= 0 or height <= 0 or width > 65536 or height > 65536:
        return diff^
    if width > len(actual) // height or width > len(expected) // height:
        return diff^
    var count = width * height
    diff.same_size = True
    diff.image = List[UInt32](length=count, fill=0xFF000000)
    var total = 0
    for index in range(count):
        var a = actual[index]
        var b = expected[index]
        if (a & 0xFFFFFF) == (b & 0xFFFFFF):
            continue
        diff.differing += 1
        var largest = 0
        var scaled = UInt32(0xFF000000)
        for shift in [16, 8, 0]:
            var difference = abs(Int((a >> UInt32(shift)) & 255) - Int((b >> UInt32(shift)) & 255))
            total += difference
            largest = max(largest, difference)
            scaled |= UInt32(min(255, difference * 16)) << UInt32(shift)
        diff.image[index] = scaled
        diff.worst = max(diff.worst, largest)
        if largest > tolerance:
            diff.over_tolerance += 1
    diff.mean_abs = Float64(total) / Float64(count * 3)
    if diff.differing > 0:
        diff.flip_map = flip_error_map(expected, actual, width, height)
        diff.flip_mean = flip_mean(diff.flip_map)
    return diff^


def compare_with_png(actual: List[UInt32], width: Int, height: Int, reference_path: String, tolerance: Int) -> ImageDiff:
    """Compares against a PNG written by write_png. A missing or differently
    sized reference gives a diff that does not pass."""
    var data = read_bytes(reference_path)
    var reference = decode_png(data, 16384)
    if not reference.valid or reference.width != width or reference.height != height:
        var diff = ImageDiff()
        diff.width = width
        diff.height = height
        return diff^
    return compare_pixels(actual, reference.pixels, width, height, tolerance)
