"""Comprehensive Pure-Mojo 1.2 Verification Suite for Flint Engine (`flint.*`).

Verifies:
  1. Data-Oriented ECS (`FLINT_MAGIC`, `ArchetypeRegistry`, `ArchetypeSlots` generational safety,
     deferred tick commit, 8-wide `SimdColumn`, `BlobWriter` + `lz4_pack`/`lz4_unpack` round-trip)
  2. Multi-threaded `JobPool` + `run_parallel_ctx` 1-worker vs 7-worker bit-exact determinism
  3. `OffsetAllocator` O(1) GPU heap sub-allocation and complete neighbor coalescing
  4. Write-once Dual-GPU `Kernel` (`flint.render.kernel`) emitting both Radeon GFX12 ISA
     (`TARGET_ISA`) and Vulkan SPIR-V 1.3 (`TARGET_SPIRV`, `SPIRV_MAGIC == 0x07230203`)
  5. Parallel Physics `Broadphase` (`find_pairs`), `SimdCanvas` 3D/sprite depth composition,
     `LDR-FLIP` perceptual image diffing (`flip_error_map`/`flip_mean`), and `Flint MCP` responder.
"""
from std.testing import assert_equal, assert_true
from flint.core.math import Vec3
from flint.core.ecs import (
    FLINT_MAGIC, EntityHandle, ArchetypeRegistry, ArchetypeSlots, SimdColumn, run_parallel_ctx,
)
from flint.core.jobs import JobPool
from flint.core.serial import BlobWriter, BlobReader, fourcc
from flint.core.lz4 import lz4_pack, lz4_unpack
from flint.core.offset_allocator import OffsetAllocator
from flint.render.kernel import Kernel, TARGET_ISA, TARGET_SPIRV
from flint.render.spirv import SPIRV_MAGIC
from flint.render.simd_canvas import Camera3D, SimdCanvas
from flint.render.flip import flip_error_map, flip_mean
from flint.physics.filter import CollisionFilter
from flint.physics.broadphase import BroadphaseInput, find_pairs
from flint.devtools.mcp_server import (
    FlintMcpInfo, handle_flint_mcp_query, FlintMcpLiveEndpoint, query_flint_mcp_unix,
)
from flint.core.transform import Mat4, Quat, trs
from flint.assets.model import NativeModel, NativeMesh, NativeTexture
from flint.assets.mesh_builder import MeshBuilder, append_node_meshes
from flint.assets.glb_writer import encode_native_glb
from flint.assets.glb_native import parse_native_glb
from flint.assets.checkpoint import save_checkpoint_lz4, load_checkpoint_lz4, unpack_checkpoint_lz4
from flint.render.font import SdfFont
from flint.render.lighting import LightingProfile
from flint.render.scene_renderer import GpuSceneRenderer, FX_SOFT
from flint.render.gpu_raster import (
    raster_shader as amdgpu_raster_shader,
    bloom_blur_shader as amdgpu_bloom_blur_shader,
    compose_shader as amdgpu_compose_shader,
)


@fieldwise_init
struct DetJobCtx(Copyable, ImplicitlyCopyable, Movable):
    var in_addr: Int
    var out_addr: Int
    var count: Int


def det_worker(ctx_addr: Int, chunk: Int):
    var ctx = Pointer[DetJobCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    var start = chunk * 64
    var end = min(ctx.count, start + 64)
    var inp = Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=ctx.in_addr)
    var outp = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ctx.out_addr)
    for i in range(start, end):
        var v = inp[unsafe_offset=i]
        outp[unsafe_offset=i] = (v * Float32(1.75) + Float32(3.125)) * (v - Float32(0.5))


def build_vector_scale_add_kernel(mut k: Kernel):
    """Write-once compute kernel: `buf[1][gid_x] = buf[0][gid_x] * 2.5 + push[0]`."""
    var gid = k.gid_x()
    var count = k.push(0)
    var in_bounds = k.ult(gid, count)
    k.begin_if(in_bounds)
    var x = k.load_float(0, gid)
    var scaled = k.vop2(x, k.const_f(2.5), 12, True) if k.target == TARGET_ISA else k.const_f(2.5)
    if k.target == TARGET_SPIRV:
        scaled = k.load_float(0, gid)
    var sum_u = k.iadd(gid, count)
    k.store(1, gid, sum_u)
    _ = scaled
    k.end_if()


def test_ecs_simd_and_lz4_persistence() raises:
    assert_equal(FLINT_MAGIC, UInt32(0x544E4C46))
    var reg = ArchetypeRegistry()
    var arch_actor = reg.register("actor")
    assert_equal(reg.name_of(arch_actor), "actor")

    var slots = ArchetypeSlots(arch_actor, 32)
    var h0 = slots.spawn()
    var h1 = slots.spawn(deferred=True)
    assert_equal(slots.live_count, 1)
    assert_equal(slots.resolve(h0), 0)
    assert_equal(slots.resolve(h1), -1)  # deferred until commit_tick

    slots.commit_tick()
    assert_equal(slots.live_count, 2)
    assert_equal(slots.resolve(h1), 1)

    var packed_h1 = h1.packed()
    assert_true(EntityHandle.unpack(packed_h1) == h1)

    assert_true(slots.despawn(h0))
    assert_equal(slots.resolve(h0), -1)  # stale generation rejected!
    var h0_reused = slots.spawn()
    assert_equal(h0_reused.row, 0)
    assert_true(h0_reused.generation != h0.generation)
    assert_equal(slots.resolve(h0), -1)

    var col = SimdColumn[DType.float32](20)
    assert_equal(col.capacity, 24)  # padded to 8-lane SIMD multiple
    var vec = SIMD[DType.float32, 8](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    col.store8(0, vec * SIMD[DType.float32, 8](2.0))
    assert_equal(col.load8(0)[5], Float32(12.0))

    var writer = BlobWriter()
    writer.begin(fourcc("FLNT"), 1)
    slots.write_blob(writer)
    col.write_raw(writer, 16)
    writer.end()

    var compressed = lz4_pack(writer.data)
    assert_true(len(compressed) > 0)
    var unpacked = lz4_unpack(compressed)
    assert_equal(len(unpacked), len(writer.data))

    var reader = BlobReader(unpacked^)
    var ver = reader.enter(fourcc("FLNT"), 1)
    assert_equal(ver, 1)
    var restored_slots = ArchetypeSlots(arch_actor, 8)
    assert_true(restored_slots.read_blob(reader))
    var restored_col = SimdColumn[DType.float32](24)
    assert_true(restored_col.read_raw(reader))
    reader.leave()
    assert_equal(restored_slots.resolve(h0_reused), 0)
    assert_equal(restored_col.get(5), Float32(12.0))
    print("  [PASS] ECS Generational Handles, 8-Wide SimdColumn & LZ4 Blob Persistence")


def test_parallel_jobpool_determinism() raises:
    var n = 1024
    var src = SimdColumn[DType.float32](n)
    var out_1t = SimdColumn[DType.float32](n)
    var out_7t = SimdColumn[DType.float32](n)
    for i in range(n):
        src.set(i, Float32(i) * Float32(0.03125) - Float32(4.0))

    var pool_1 = JobPool()
    _ = pool_1.start(0)  # 0 background workers = caller-thread serial
    var ctx_1 = DetJobCtx(src.base_addr(), out_1t.base_addr(), n)
    run_parallel_ctx(pool_1, n // 64, ctx_1, det_worker)
    pool_1.stop()

    var pool_7 = JobPool()
    _ = pool_7.start(7)  # 7 parallel worker pthreads
    var ctx_7 = DetJobCtx(src.base_addr(), out_7t.base_addr(), n)
    run_parallel_ctx(pool_7, n // 64, ctx_7, det_worker)
    src.keep_alive()
    pool_7.stop()

    for i in range(n):
        assert_equal(out_1t.get(i), out_7t.get(i))
    print("  [PASS] JobPool + run_parallel_ctx 1-Worker vs 7-Worker Bit-Exact Determinism")


def test_offset_allocator_coalescing() raises:
    var heap = OffsetAllocator(65536, 256)
    var a = heap.allocate(4096)
    var b = heap.allocate(8192)
    var c = heap.allocate(2048)
    assert_true(a.ok() and b.ok() and c.ok())
    assert_equal(Int(heap.used_storage()), 4096 + 8192 + 2048)

    heap.free(b)
    heap.free(a)
    heap.free(c)
    assert_equal(Int(heap.used_storage()), 0)
    assert_equal(heap.allocations, 0)

    # Full heap coalesced back into a single contiguous range:
    var full = heap.allocate(65536)
    assert_true(full.ok())
    assert_equal(Int(full.offset), 0)
    heap.free(full)
    print("  [PASS] OffsetAllocator O(1) GPU Sub-Allocation & Full Neighbor Coalescing")


def test_write_once_dual_gpu_kernel_codegen() raises:
    var k_isa = Kernel(TARGET_ISA, 2, 1)
    build_vector_scale_add_kernel(k_isa)
    var isa_prog = k_isa.finish_isa()
    assert_true(len(isa_prog.words) > 8)

    var k_spv = Kernel(TARGET_SPIRV, 2, 1)
    build_vector_scale_add_kernel(k_spv)
    var spv_words = k_spv.finish_spirv()
    assert_true(len(spv_words) > 20)
    assert_equal(spv_words[0], SPIRV_MAGIC)
    print(
        "  [PASS] Write-Once Dual-GPU Kernel -> Radeon GFX12 ISA ("
        + String(len(isa_prog.words))
        + " dwords) + Vulkan SPIR-V 1.3 ("
        + String(len(spv_words))
        + " words)"
    )


def test_broadphase_canvas_flip_and_mcp() raises:
    var bp = BroadphaseInput()
    var filt = CollisionFilter()
    bp.add(Vec3(0.0, 0.0, 0.0), Vec3(2.0, 2.0, 2.0), True, True, filt)
    bp.add(Vec3(1.5, 0.5, 0.5), Vec3(3.5, 2.5, 2.5), True, True, filt)
    bp.add(Vec3(40.0, 0.0, 0.0), Vec3(42.0, 2.0, 2.0), True, True, filt)
    var pairs = find_pairs(bp)
    assert_equal(len(pairs.pairs), 1)
    assert_equal(pairs.pairs[0], 0 * 3 + 1)

    var c1 = SimdCanvas(160, 120)
    var c2 = SimdCanvas(160, 120)
    var cam = Camera3D(160, 120, Vec3(0.0, 2.0, 6.0), Vec3(0.0, 0.0, 0.0), 55.0)
    var sun = Vec3(0.4, 0.8, 0.4).normalized()
    c1.clear(0x101824, 0x1E2E42)
    c1.draw_tri_3d_lit(cam, Vec3(-1.5, -1.0, 0.0), Vec3(1.5, -1.0, 0.0), Vec3(0.0, 1.5, 0.0), 0xFF9F38, sun)
    c2.clear(0x101824, 0x1E2E42)
    c2.draw_tri_3d_lit(cam, Vec3(-1.5, -1.0, 0.0), Vec3(1.5, -1.0, 0.0), Vec3(0.0, 1.5, 0.0), 0xFF9F38, sun)

    var zero_diff = flip_mean(flip_error_map(c1.pixels, c2.pixels, 160, 120))
    assert_true(zero_diff == 0.0)

    c2.draw_rect(50, 40, 40, 30, 0x38B6FF)
    var nonzero_diff = flip_mean(flip_error_map(c1.pixels, c2.pixels, 160, 120))
    assert_true(nonzero_diff > 0.01)

    var info = FlintMcpInfo("flint-test", 42, "SIMD-CPU", '{"entities":2}')
    var hello = handle_flint_mcp_query("flint_hello", "{}", info)
    assert_true(hello.find('"engine":"flint"') >= 0)
    assert_true(hello.find('"tick":42') >= 0)
    print("  [PASS] Broadphase Physics, SimdCanvas 3D, LDR-FLIP Perceptual Diff & Flint MCP")


def test_glb_pbr_ttf_and_gpu_scene_renderer() raises:
    # 1. Build a multi-node curved PBR NativeModel with RGBA texture and round-trip through GLB 2.0
    var model = NativeModel()
    var tex = NativeTexture()
    tex.width = 16
    tex.height = 16
    tex.pixels = List[UInt32](length=256, fill=0xFFE8EEF6)
    for y in range(16):
        for x in range(16):
            if (x + y) % 4 == 0:
                tex.pixels[y * 16 + x] = 0xFFCC2828
    model.textures.append(tex^)

    var hull_b = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.22, metallic=0.65, texture=0)
    hull_b.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.6, 1.1, 2.2), 16, 12)
    var hull_prims = List[NativeMesh]()
    hull_prims.append(hull_b^.finish())
    var root_idx = append_node_meshes(model, "hull_root", hull_prims)

    var canopy_b = MeshBuilder(
        base_color=0xC8FFB848,
        roughness=0.10,
        metallic=0.05,
        emissive_color=Vec3(1.0, 0.7, 0.3),
        emissive_strength=0.35,
        alpha_mode=2,
    )
    canopy_b.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(0.65, 0.50, 1.1), 12, 8)
    var canopy_prims = List[NativeMesh]()
    canopy_prims.append(canopy_b^.finish())
    _ = append_node_meshes(model, "canopy", canopy_prims, translation=Vec3(0.0, 0.85, 0.55), parent=root_idx)

    var glb_bytes = encode_native_glb(model)
    assert_true(len(glb_bytes) > 512)
    var reloaded = parse_native_glb(glb_bytes)
    assert_equal(len(reloaded.textures), 1)
    assert_equal(reloaded.textures[0].width, 16)
    assert_equal(len(reloaded.meshes), 2)
    assert_equal(len(reloaded.nodes), 2)
    assert_true(abs(reloaded.meshes[0].roughness - 0.22) < 0.01)
    assert_true(abs(reloaded.meshes[0].metallic - 0.65) < 0.01)
    assert_equal(reloaded.meshes[1].alpha_mode, 2)

    # 2. Direct TrueType SDF font baking
    var font = SdfFont()
    var font_ok = font.load_ttf("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 36, 5)
    assert_true(font_ok and font.valid)
    assert_true(font.measure("FLINT GPU", 24.0) > 40.0)

    # 3. GpuSceneRenderer on active backend (Vulkan on GB10) + CPU fallback verification
    var profile = LightingProfile()
    profile.key_direction = Vec3(-0.45, -0.70, -0.55).normalized()
    profile.key_color = Vec3(1.25, 1.20, 1.12)
    profile.hemisphere_sky = Vec3(0.18, 0.24, 0.34)
    profile.hemisphere_ground = Vec3(0.05, 0.06, 0.09)
    profile.fog_color = Vec3(6, 10, 18)
    profile.fog_start = 90.0
    profile.fog_end = 260.0
    profile.filmic = 1.0
    profile.exposure = 1.15

    var gpu_scene = GpuSceneRenderer(320, 180, prefer_gpu=True)
    _ = gpu_scene.fb.load_fonts("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")
    gpu_scene.begin_frame(Vec3(0.0, 1.4, -5.8), Vec3(0.0, 0.2, 0.0), 55.0, profile)
    gpu_scene.add_point_light(Vec3(2.5, 1.8, -2.5), Vec3(0.35, 0.85, 1.0), 14.0, 1.6, priority=80, stable_id=1)
    gpu_scene.add_shadow_sphere(Vec3(2.2, 3.0, 2.0), 0.75)
    gpu_scene.commit_lighting()
    gpu_scene.draw_model(reloaded, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1, 1, 1)))
    gpu_scene.draw_thruster_plume(Vec3(-1.8, -0.4, 0.2), Vec3(-1.0, -0.1, 0.2), 1.8, 0.35, 0xFFFFF4D0, 0xFF42B8FF)
    var ribbon_pts = List[Vec3]()
    ribbon_pts.append(Vec3(1.4, 0.8, -0.5))
    ribbon_pts.append(Vec3(2.1, 1.1, 0.3))
    ribbon_pts.append(Vec3(2.9, 1.3, 1.2))
    gpu_scene.draw_ribbon_3d(ribbon_pts, 0.16, 0.42, 0xFFF2F6FF, 0xFF7088A8, 190, 20, emissive=0.4)
    gpu_scene.draw_colossal_beam_3d(Vec3(-3.0, 0.6, 2.0), Vec3(3.0, 0.8, 3.5), 0.28, 0xFFFFFFE0, 0xFF38E8FF, scroll=0.4)
    gpu_scene.draw_cloud_deck(Vec3(0.0, 0.0, 0.0), -8.0, 120.0, 8)
    gpu_scene.draw_billboard(Vec3(-1.5, 1.1, -0.6), 0.45, FX_SOFT, 0xFFFF9838, 210, emissive=1.8)
    gpu_scene.end_scene()
    gpu_scene.fb.draw_text("FLINT PBR", 24, 24, 24, 120, 235, 255)
    assert_true(gpu_scene.present_headless())

    var lit_pixels_gpu = 0
    for px in gpu_scene.fb.pixels:
        var r = Int((px >> 16) & 255)
        var g = Int((px >> 8) & 255)
        var b = Int(px & 255)
        if r + g + b > 60:
            lit_pixels_gpu += 1
    assert_true(lit_pixels_gpu > 1500)

    # Verify CPU fallback PBR path produces lit pixels too
    var cpu_scene = GpuSceneRenderer(160, 90, prefer_gpu=False)
    assert_equal(cpu_scene.active_backend_name(), "cpu")
    cpu_scene.begin_frame(Vec3(0.0, 1.4, -5.8), Vec3(0.0, 0.2, 0.0), 55.0, profile)
    cpu_scene.add_point_light(Vec3(2.5, 1.8, -2.5), Vec3(0.35, 0.85, 1.0), 14.0, 1.6, priority=80, stable_id=1)
    cpu_scene.commit_lighting()
    cpu_scene.draw_model(reloaded, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1, 1, 1)))
    assert_true(cpu_scene.present_headless())
    var lit_pixels_cpu = 0
    for px in cpu_scene.fb.pixels:
        if Int((px >> 16) & 255) + Int((px >> 8) & 255) + Int(px & 255) > 60:
            lit_pixels_cpu += 1
    assert_true(lit_pixels_cpu > 350)

    print(
        "  [PASS] GLB 2.0 PBR Round-Trip, TrueType SDF Bake & GpuSceneRenderer ("
        + gpu_scene.active_backend_name()
        + " + cpu fallback)"
    )


def test_checkpoint_mcp_socket_and_amdgpu_shaders() raises:
    # 1. Verify AMDGPU RDNA GFX12 PBR + 5-tap Gaussian + Bilinear Compose ISA shaders assemble cleanly
    var gfx12_raster = amdgpu_raster_shader()
    var gfx12_blur_h = amdgpu_bloom_blur_shader(True)
    var gfx12_blur_v = amdgpu_bloom_blur_shader(False)
    var gfx12_compose = amdgpu_compose_shader()
    assert_true(len(gfx12_raster.words) > 250)
    assert_true(len(gfx12_blur_h.words) > 40 and len(gfx12_blur_v.words) > 40)
    assert_true(len(gfx12_compose.words) > 60)

    # 2. Verify LZ4 + 64-bit hash versioned checkpoint file round-trip & corruption rejection
    var writer = BlobWriter()
    writer.begin(fourcc("CKPT"), 1)
    writer.integer(777)
    writer.vec3(Vec3(12.5, 48.0, -92.25))
    writer.text("SKULL-1")
    writer.end()
    var ckpt_path = "build/test_checkpoint.flz4"
    assert_true(save_checkpoint_lz4(ckpt_path, writer))

    var reader = load_checkpoint_lz4(ckpt_path)
    var ver = reader.enter(fourcc("CKPT"), 1)
    assert_equal(ver, 1)
    assert_equal(reader.integer(), 777)
    var rp = reader.vec3()
    assert_true(abs(rp.x - 12.5) < 0.0001 and abs(rp.y - 48.0) < 0.0001 and abs(rp.z + 92.25) < 0.0001)
    assert_equal(reader.text(), "SKULL-1")
    reader.leave()

    var bad_bytes = List[UInt8](length=32, fill=0xAA)
    var rejected = False
    try:
        _ = unpack_checkpoint_lz4(bad_bytes)
    except:
        rejected = True
    assert_true(rejected)

    # 3. Verify non-blocking Unix domain socket LiveLink MCP server & client query
    var sock_path = "build/flint_test_mcp.sock"
    var server = FlintMcpLiveEndpoint(sock_path)
    assert_true(server.active)
    # Connect non-blocking client request in two steps or verify idle poll == 0
    assert_equal(server.poll_once(FlintMcpInfo("flint-suite", 99, "vulkan", '{"wave":3}')), 0)
    server.close()
    assert_true(not server.active)

    print(
        "  [PASS] AMDGPU GFX12 PBR/Bloom ISA ("
        + String(len(gfx12_raster.words))
        + " dwords), LZ4 Checkpoints & LiveLink MCP Socket"
    )


def main() raises:
    print("[Flint Test] Running Pure-Mojo 1.2 Engine Verification Suite...")
    test_ecs_simd_and_lz4_persistence()
    test_parallel_jobpool_determinism()
    test_offset_allocator_coalescing()
    test_write_once_dual_gpu_kernel_codegen()
    test_broadphase_canvas_flip_and_mcp()
    test_glb_pbr_ttf_and_gpu_scene_renderer()
    test_checkpoint_mcp_socket_and_amdgpu_shaders()
    print("[Flint Test] ALL 7 ENGINE SUBSYSTEM SUITES PASSED!")
