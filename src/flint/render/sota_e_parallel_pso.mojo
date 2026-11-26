"""Area E (`sota:E`): compile the renderer's compute pipelines in parallel on a `JobPool`.

`VulkanGpu.create_aaa_compute_pipeline` in batch mode (`sota_e_batch_on`) only records the SPIR-V and
layout and returns a token (`SOTA_E_PSO_TOKEN | index`). `sota_e_pso_batch_flush` then creates every
recorded pipeline concurrently: `vkCreateShaderModule` / `vkCreateComputePipelines` are free-threaded
for distinct objects and the `VkPipelineCache` is internally synchronised (Vulkan spec, "Threading
Behavior"). Each job builds its create-info structs in its own memory, never touching the context's
shared scratch arena or its lazily-filled `device_call` cache (the entry points are fetched up front).
Tokens resolve to real handles at bind / destroy time (`VulkanGpu.sota_e_resolve`), so renderer fields
keep their value. `FLINT_PARALLEL_PSO=0` turns batching off (serial creation, as before).
"""
from std.collections import List
from std.time import monotonic
from flint.core.ecs import run_parallel_ctx
from flint.core.jobs import JobPool, default_worker_count
from flint.platform.os import env_now
from flint.platform.vulkan import (
    VulkanGpu,
    call3,
    call4,
    call6,
    get64,
    put32,
    put64,
    STAGE_COMPUTE_SHADER,
    TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    TYPE_SHADER_MODULE_CREATE_INFO,
)


def sota_e_parallel_pso_enabled() -> Bool:
    # --- sota:perf hook --- getenv(3), not /proc/self/environ: a switch set with setenv in this process counts
    return env_now("FLINT_PARALLEL_PSO") != "0"


def sota_e_pso_batch_begin(mut vk: VulkanGpu):
    if not sota_e_parallel_pso_enabled() or vk.device == 0:
        return
    vk.sota_e_batch_on = True
    vk.sota_e_batch_layouts = List[Int]()
    vk.sota_e_batch_code = List[List[UInt32]]()
    vk.sota_e_batch_handles = List[Int]()


struct _PsoJobs(Movable):
    var device: Int
    var cache: Int
    var fn_module: Int
    var fn_pipes: Int
    var fn_destroy_module: Int
    var entry: Int          # "main\0"
    var code_ptr: Int       # List[Int]: per job SPIR-V address
    var code_bytes: Int     # List[Int]: per job byte size
    var layouts: Int        # List[Int]
    var out: Int            # List[Int]: pipeline handles (0 on failure)

    def __init__(out self, device: Int, cache: Int, fn_module: Int, fn_pipes: Int, fn_destroy_module: Int, entry: Int, code_ptr: Int, code_bytes: Int, layouts: Int, out_handles: Int):
        self.device = device
        self.cache = cache
        self.fn_module = fn_module
        self.fn_pipes = fn_pipes
        self.fn_destroy_module = fn_destroy_module
        self.entry = entry
        self.code_ptr = code_ptr
        self.code_bytes = code_bytes
        self.layouts = layouts
        self.out = out_handles


@always_inline
def _ld(base: Int, i: Int) -> Int:
    return Pointer[Int, MutAnyOrigin](unsafe_from_address=base + i * 8)[]


def _pso_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_PsoJobs, MutAnyOrigin](unsafe_from_address=ctx_addr)
    var mem = List[UInt64](length=40, fill=0)  # 320 bytes, 8-aligned, private to this job
    var base = Int(mem.unsafe_ptr())
    var module_info = base
    var slot = base + 48
    var info = base + 64
    put32(module_info, TYPE_SHADER_MODULE_CREATE_INFO)
    put64(module_info + 24, _ld(c[].code_bytes, job))
    put64(module_info + 32, _ld(c[].code_ptr, job))
    var out_slot = c[].out + job * 8
    if call4(c[].fn_module, c[].device, module_info, 0, slot) != 0:
        put64(out_slot, 0)
        return
    var module = get64(slot)
    put32(info, TYPE_COMPUTE_PIPELINE_CREATE_INFO)
    put32(info + 24, TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
    put32(info + 44, STAGE_COMPUTE_SHADER)
    put64(info + 48, module)
    put64(info + 56, c[].entry)
    put64(info + 72, _ld(c[].layouts, job))
    put32(info + 88, 0xFFFFFFFF)
    var res = call6(c[].fn_pipes, c[].device, c[].cache, 1, info, 0, slot)
    var pipe = get64(slot)
    _ = call3(c[].fn_destroy_module, c[].device, module, 0)
    put64(out_slot, pipe if res == 0 else 0)
    _ = mem[0]


def sota_e_pso_batch_flush(mut vk: VulkanGpu, workers: Int = -1) -> Int:
    """Compiles every batched pipeline; returns how many failed. Leaves batch mode."""
    if not vk.sota_e_batch_on:
        return 0
    vk.sota_e_batch_on = False
    var n = len(vk.sota_e_batch_code)
    var handles = List[Int](length=n, fill=0)
    if n == 0:
        vk.sota_e_batch_handles = handles^
        return 0
    var code_ptr = List[Int]()
    var code_bytes = List[Int]()
    for i in range(n):
        code_ptr.append(Int(vk.sota_e_batch_code[i].unsafe_ptr()))
        code_bytes.append(len(vk.sota_e_batch_code[i]) * 4)
    var layouts = vk.sota_e_batch_layouts.copy()
    var entry = List[UInt8]()
    entry.append(109); entry.append(97); entry.append(105); entry.append(110); entry.append(0)  # "main"
    var ctx = _PsoJobs(
        vk.device, vk.pipeline_cache,
        vk.device_call("vkCreateShaderModule"), vk.device_call("vkCreateComputePipelines"),
        vk.device_call("vkDestroyShaderModule"),
        Int(entry.unsafe_ptr()), Int(code_ptr.unsafe_ptr()), Int(code_bytes.unsafe_ptr()),
        Int(layouts.unsafe_ptr()), Int(handles.unsafe_ptr()),
    )
    var sota_perf_t0 = Float64(monotonic()) * 1.0e-9  # --- sota:perf hook --- batch compile wall time
    var pool = JobPool()
    var nw = default_worker_count() if workers < 0 else workers
    if nw > 1 and pool.start(min(nw, n)):
        run_parallel_ctx(pool, n, ctx, _pso_worker)
        pool.stop()
    else:
        for j in range(n):
            _pso_worker(Int(Pointer(to=ctx)), j)
    vk.sota_perf_compute_compile_s += Float64(monotonic()) * 1.0e-9 - sota_perf_t0  # --- sota:perf hook ---
    var failed = 0
    for i in range(n):
        if handles[i] == 0:
            failed += 1
    # Keep every buffer the workers read alive past the barrier.
    _ = len(code_ptr); _ = len(code_bytes); _ = len(layouts); _ = len(entry)
    _ = vk.sota_e_batch_code[0][0]
    vk.sota_e_batch_handles = handles^
    vk.sota_e_batch_code = List[List[UInt32]]()  # SPIR-V no longer needed
    return failed
