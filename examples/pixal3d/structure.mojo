"""Sample and decode the image-conditioned Pixal3D sparse structure."""
from std.sys import argv
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.hi3d_io import hi3d_read_rows, hi3d_write_rows
from flint.ml.hi3d_noise import hi3d_noise
from flint.ml.pixal3d_flow import PixalFlow
from flint.ml.trellis2 import TrellisStructureDecoder, structure_coordinates
from flint.ml.trellis2_flow import sample_structure_flow


def main() raises:
    var args = argv()
    if len(args) != 8:
        raise Error("Usage: structure FLOW DECODER GLOBAL.f32 PROJECTED.f32 SEED STEPS COORDS.i32")
    var seed = Int(String(args[5]))
    var steps = Int(String(args[6]))
    if seed < 0 or seed > 0xFFFFFFFF or steps < 1:
        raise Error("Expected uint32 seed and positive steps")
    var noise = hi3d_noise(1024, UInt32(seed))
    noise.shape = [8, 16, 16, 16]
    var model = PixalFlow(
        String(args[1]), hi3d_read_rows(String(args[3]), 1024), hi3d_read_rows(String(args[4]), 1024)
    )
    var latent = sample_structure_flow(model, noise^, steps)
    var decoder = TrellisStructureDecoder(String(args[2]))
    var pool = JobPool()
    _ = pool.start(default_worker_count())
    var coords = structure_coordinates(decoder.decode(pool, latent), 32)
    hi3d_write_rows(String(args[7]), coords)
    print("Sampled", coords.shape[0], "occupied sites on a 32-cubed grid")
