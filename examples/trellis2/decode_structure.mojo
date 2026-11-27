"""Decode a TRELLIS.2 structure latent. See this directory's README for the binary layout."""

from std.memory import bitcast
from std.sys import argv
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.trellis2 import TrellisStructureDecoder, read_structure_latent
from flint.platform.os import write_bytes


def main() raises:
    var args = argv()
    if len(args) != 5:
        raise Error("Usage: decode_structure CHECKPOINT.safetensors LATENT.f32 EDGE LOGITS.f32")
    var edge = Int(String(args[3]))
    var latent = read_structure_latent(String(args[2]), edge)
    var decoder = TrellisStructureDecoder(String(args[1]))
    var pool = JobPool()
    _ = pool.start(default_worker_count())
    var result = decoder.decode(pool, latent)
    var data = List[UInt8](capacity=result.numel() * 4)
    var occupied = 0
    for v in result.f:
        var bits = bitcast[DType.uint32](v)
        for b in range(4):
            data.append(UInt8(bits >> UInt32(8 * b)))
        occupied += Int(v > 0)
    if not write_bytes(String(args[4]), data):
        raise Error("Could not write occupancy logits")
    print("Structure grid", edge * 4, "cubed;", occupied, "occupied of", result.numel())
