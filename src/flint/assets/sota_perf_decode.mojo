"""Area F round 2 (render SOTA, perf): parallel image decode for the glTF / GLB loader.

`_native_model_from_document` (glb_native.mojo, `# --- sota:perf hook ---`) gathers every texture's
encoded PNG / JPEG bytes first and decodes them here on a short-lived `JobPool`, one job per image,
largest first. Each job writes only its own output slot, so the model is identical to the serial loop.
`FLINT_SOTA_PARALLEL_DECODE=0` keeps the decode on the calling thread.

Lives in `flint.assets` (not `flint.render`) because the loader must not import the renderer.
"""
from std.collections import List
from std.ffi import external_call
from std.math import max, min
from flint.core.detsort import stable_order
from flint.core.ecs import run_parallel_ctx
from flint.core.jobs import JobPool
from flint.platform.os import env_now
from flint.assets.image import NativeImage
from flint.assets.jpeg import decode_image_auto


struct _DecodeCtx(Movable):
    var blobs: Int  # List[List[UInt8]]
    var order: Int  # List[Int]
    var out: Int  # List[NativeImage]
    var max_dim: Int

    def __init__(out self):
        self.blobs = 0
        self.order = 0
        self.out = 0
        self.max_dim = 4096


def _decode_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_DecodeCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var blobs = Pointer[List[List[UInt8]], ImmutAnyOrigin](unsafe_from_address=c[].blobs)
    var order = Pointer[List[Int], ImmutAnyOrigin](unsafe_from_address=c[].order)
    var out = Pointer[List[NativeImage], MutAnyOrigin](unsafe_from_address=c[].out)
    var i = order[][job]
    if len(blobs[][i]) > 0:
        out[][i] = decode_image_auto(blobs[][i], c[].max_dim)


def sota_decode_threads() -> Int:
    var e = env_now("FLINT_SOTA_PARALLEL_DECODE")
    if e == "0":
        return 0
    var logical = Int(external_call["sysconf", Int64](Int32(84)))  # _SC_NPROCESSORS_ONLN
    return max(0, min(31, logical - 1))


def sota_decode_images(blobs: List[List[UInt8]], max_dim: Int) -> List[NativeImage]:
    """`decode_image_auto(blobs[i], max_dim)` for every i (empty blob -> invalid image)."""
    var n = len(blobs)
    var out = List[NativeImage](length=n, fill=NativeImage())
    var busy = 0
    for b in blobs:
        if len(b) > 0:
            busy += 1
    var threads = min(sota_decode_threads(), busy - 1)
    if threads <= 0:
        for i in range(n):
            if len(blobs[i]) > 0:
                out[i] = decode_image_auto(blobs[i], max_dim)
        return out^
    var neg = List[Int](capacity=n)
    for b in blobs:
        neg.append(-len(b))
    var order = stable_order(neg)
    var pool = JobPool()
    _ = pool.start(threads)
    var ctx = _DecodeCtx()
    ctx.blobs = Int(Pointer(to=blobs))
    ctx.order = Int(Pointer(to=order))
    ctx.out = Int(Pointer(to=out))
    ctx.max_dim = max_dim
    run_parallel_ctx(pool, n, ctx, _decode_worker)
    _ = len(order)
    pool.stop()
    return out^
