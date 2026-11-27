"""Pixal3D single-view image conditioning. See licenses/pixal3d."""
from std.math import isfinite, tan
from flint.ml.tensor import Tensor, DT_I32
from flint.ml.trellis2_dino import PREFIX


struct PixalCamera(Copyable, Movable):
    var fov: Float32
    var distance: Float32
    var mesh_scale: Float32

    def __init__(out self, fov: Float32, distance: Float32, mesh_scale: Float32 = 1) raises:
        if not (
            fov > 0
            and fov < 3.14159265
            and isfinite(distance)
            and distance > 0
            and isfinite(mesh_scale)
            and mesh_scale > 0
        ):
            raise Error("Invalid Pixal3D camera FOV, distance or mesh scale")
        self.fov = fov
        self.distance = distance
        self.mesh_scale = mesh_scale

    @staticmethod
    def from_fov(fov: Float32, mesh_scale: Float32 = 1) raises -> Self:
        """Upstream manual-FOV camera fit at zero pixel extension."""
        if not (fov > 0 and fov < 3.14159265 and isfinite(mesh_scale) and mesh_scale > 0):
            raise Error("Invalid Pixal3D manual FOV or mesh scale")
        return Self(fov, 0.5 / (tan(fov * 0.5) * mesh_scale), mesh_scale)


def projection_pixels(
    image_size: Int, grid: Int, camera: PixalCamera, coords: Tensor = Tensor()
) raises -> Tensor:
    """Project dense or batch-zero sparse XYZ sites to pixel centers on the reference image."""
    if image_size < 16 or grid < 2 or grid > 256:
        raise Error("Invalid Pixal3D projection grid")
    var sparse = coords.rank() != 0
    var count = grid * grid * grid
    if sparse:
        if (
            coords.rank() != 2
            or coords.shape[1] != 4
            or coords.dtype != DT_I32
            or not coords.has_valid_storage()
        ):
            raise Error("Expected integer [N,4] Pixal3D coordinates")
        count = coords.shape[0]
    var pixels = Tensor.zeros([count, 2])
    var focal = Float32(image_size) * 0.5 / tan(camera.fov * 0.5)
    for i in range(count):
        var x = i // (grid * grid)
        var y = i // grid % grid
        var z = i % grid
        if sparse:
            x = coords.i[4 * i + 1]
            y = coords.i[4 * i + 2]
            z = coords.i[4 * i + 3]
            if coords.i[4 * i] != 0 or min(x, min(y, z)) < 0 or max(x, max(y, z)) >= grid:
                raise Error("Pixal3D sparse coordinate outside grid")
        # The grid and front camera share the Blender axis rotation. In
        # camera coordinates the third grid axis points toward the lens.
        var unit = 0.5 / camera.mesh_scale
        var wx = (2 * Float32(x) / Float32(grid - 1) - 1) * unit
        var wy = (2 * Float32(y) / Float32(grid - 1) - 1) * unit
        var wz = (2 * Float32(z) / Float32(grid - 1) - 1) * unit
        var depth = camera.distance - wz + Float32(1e-8)
        var u = Float32(image_size) * 0.5 + focal * wx / depth
        var v = Float32(image_size) * 0.5 - focal * wy / depth
        # grid_sample with align_corners=False, border padding, pixel centres.
        pixels.f[2 * i] = u
        pixels.f[2 * i + 1] = v
    return pixels^


def project_patch_features(
    tokens: Tensor,
    image_size: Int,
    grid: Int,
    camera: PixalCamera,
    coords: Tensor = Tensor(),
) raises -> Tensor:
    """Bilinearly sample DINO patch tokens using border padding and align_corners=False."""
    if (
        image_size < 16
        or image_size % 16 != 0
        or tokens.rank() != 3
        or tokens.shape[0] != 1
        or tokens.shape[2] < 1
        or not tokens.is_float()
        or not tokens.has_valid_storage()
    ):
        raise Error("Invalid Pixal3D image tokens")
    var patches = image_size // 16
    if tokens.shape[1] != PREFIX + patches * patches:
        raise Error("DINOv3 token count does not match image resolution")
    var pixels = projection_pixels(image_size, grid, camera, coords)
    var channels = tokens.shape[2]
    var count = pixels.shape[0]
    var out = Tensor.zeros([1, count, channels])
    for i in range(count):
        var u = pixels.f[2 * i]
        var v = pixels.f[2 * i + 1]
        var px = min(Float32(patches - 1), max(Float32(0), (u + 0.5) / 16 - 0.5))
        var py = min(Float32(patches - 1), max(Float32(0), (v + 0.5) / 16 - 0.5))
        var left = Int(px)
        var top = Int(py)
        var right = min(left + 1, patches - 1)
        var bottom = min(top + 1, patches - 1)
        var dx = px - Float32(left)
        var dy = py - Float32(top)
        var a = (PREFIX + top * patches + left) * channels
        var b = (PREFIX + top * patches + right) * channels
        var c = (PREFIX + bottom * patches + left) * channels
        var d = (PREFIX + bottom * patches + right) * channels
        for j in range(channels):
            out.f[i * channels + j] = (
                tokens.fget(a + j) * ((1 - dx) * (1 - dy))
                + tokens.fget(b + j) * (dx * (1 - dy))
                + tokens.fget(c + j) * ((1 - dx) * dy)
                + tokens.fget(d + j) * (dx * dy)
            )
    return out^


def structure_condition(tokens: Tensor, camera: PixalCamera) raises -> Tuple[Tensor, Tensor]:
    var projected = project_patch_features(tokens, 512, 16, camera)
    var globals = Tensor.zeros([1, PREFIX, tokens.shape[2]])
    for i in range(globals.numel()):
        globals.f[i] = tokens.fget(i)
    return (globals^, projected^)
